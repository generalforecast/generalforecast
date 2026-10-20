from __future__ import annotations

import dataclasses
import pickle
import struct
import zipfile
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

from generalforecast.models import catalog
from generalforecast.models.checkpoints import (
    load_state,
    read_all,
    read_torch_checkpoint,
    save_state,
)

CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"


def _text(value: str) -> bytes:
    encoded = value.encode()
    return b"X" + struct.pack("<I", len(encoded)) + encoded


def _int(value: int) -> bytes:
    return b"J" + struct.pack("<i", value)


def _ints(values: tuple[int, ...]) -> bytes:
    return b"(" + b"".join(_int(value) for value in values) + b"t"


def _tensor(key: str, count: int, offset: int, shape, stride) -> bytes:
    """Pickle opcodes torch.save writes for one float32 tensor."""
    storage = (
        b"("
        + _text("storage")
        + b"ctorch\nFloatStorage\n"
        + _text(key)
        + _text("cpu")
        + _int(count)
        + b"tQ"
    )
    hooks = b"ccollections\nOrderedDict\n)R"
    arguments = storage + _int(offset) + _ints(shape) + _ints(stride) + b"\x89" + hooks
    return b"ctorch._utils\n_rebuild_tensor_v2\n(" + arguments + b"tR"


def _write(path: Path, pickled: bytes, storages: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickled)
        archive.writestr("archive/byteorder", b"little")
        for key, values in storages.items():
            archive.writestr(f"archive/data/{key}", values.astype("<f4").tobytes())


def test_reads_tensors_and_plain_values(tmp_path: Path) -> None:
    storage = np.arange(12, dtype=np.float32)
    pickled = (
        b"\x80\x02}("
        + _text("weight")
        + _tensor("0", 12, 0, (3, 4), (4, 1))
        + _text("transposed")
        + _tensor("0", 12, 2, (2, 3), (1, 4))
        + _text("step")
        + _int(7)
        + b"u."
    )
    path = tmp_path / "model.ckpt"
    _write(path, pickled, {"0": storage})

    loaded = read_torch_checkpoint(path)

    np.testing.assert_array_equal(loaded["weight"], storage.reshape(3, 4))
    # Offset 2, shape (2, 3), stride (1, 4): a strided view, copied out.
    np.testing.assert_array_equal(loaded["transposed"], [[2, 6, 10], [3, 7, 11]])
    assert loaded["transposed"].flags.c_contiguous
    assert loaded["step"] == 7


@pytest.mark.parametrize(
    ("offset", "shape", "stride"),
    [(0, (3,), (-1,)), (2, (2, 3), (1, 5)), (12, (), ())],
    ids=["before", "past", "scalar-past"],
)
def test_refuses_views_outside_their_storage(
    tmp_path: Path, offset: int, shape: tuple[int, ...], stride: tuple[int, ...]
) -> None:
    pickled = b"\x80\x02" + _tensor("0", 12, offset, shape, stride) + b"."
    path = tmp_path / "damaged.ckpt"
    _write(path, pickled, {"0": np.arange(12, dtype=np.float32)})

    with pytest.raises(ValueError, match=r"negative|past its storage"):
        read_torch_checkpoint(path)


def test_refuses_files_that_are_not_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "other.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", b"hello")

    with pytest.raises(ValueError, match="not a PyTorch checkpoint"):
        read_torch_checkpoint(path)


def test_refuses_anything_but_tensors(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    pickled = b"\x80\x02cos\nsystem\n" + _text(f"touch {marker}") + b"\x85R."
    path = tmp_path / "evil.ckpt"
    _write(path, pickled, {})

    with pytest.raises(pickle.UnpicklingError, match=r"os\.system"):
        read_torch_checkpoint(path)
    assert not marker.exists()


@pytest.mark.skipif(
    not (CHECKPOINTS / "tabicl-v2-classifier" / "model.safetensors").exists(),
    reason="local TabICL v2 checkpoints are not downloaded",
)
def test_upstream_checkpoint_matches_its_safetensors_conversion() -> None:
    path = hf_hub_download(
        "jingang/TabICL",
        "tabicl-classifier-v2-20260212.ckpt",
        revision="4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
    )
    loaded = read_torch_checkpoint(path)["state_dict"]
    converted = load_file(CHECKPOINTS / "tabicl-v2-classifier" / "model.safetensors")

    assert set(loaded) == set(converted)
    for key, values in converted.items():
        np.testing.assert_array_equal(loaded[key], values)


def test_built_in_checkpoints_whose_weights_misfit_are_refused_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generalforecast.models import registry
    from generalforecast.models.checkpoints import save_state
    from tests.models.tabicl.tiny import tiny_state

    state, settings = tiny_state(regression=False)
    arrays = {key: np.asarray(value) for key, value in state.items()}
    # Every parameter is there, one of them the wrong way round.
    arrays["icl_predictor.decoder.2.weight"] = arrays[
        "icl_predictor.decoder.2.weight"
    ].T
    settings = {**settings, "col_affine": False, "norm_first": True}
    save_state(tmp_path / "tabicl-v2-classifier", "tabicl_v2", settings, arrays)
    monkeypatch.setenv("GF_CHECKPOINTS", str(tmp_path))

    # Pinned files refuse other bytes before tracing; a trusted local entry
    # without recorded hashes still refuses weights that do not fit.
    with pytest.raises(ValueError, match="differs from its pin"):
        registry.load("general-forecast/tabicl-v2-classifier")
    entry = catalog.ENTRIES["tabicl-v2-classifier"]
    unhashed = dataclasses.replace(
        entry, artifact=dataclasses.replace(entry.artifact, files=MappingProxyType({}))
    )
    with pytest.raises(ValueError, match="'tabicl-v2-classifier' has weights"):
        registry._native(unhashed, "local", False)


def test_a_missing_setting_is_not_reported_as_a_missing_parameter(
    tmp_path: Path,
) -> None:
    state = {"layer.weight": np.ones((2, 2), np.float32)}
    save_state(tmp_path, "toy", {"heads": 2}, state)
    arrays, settings = load_state(tmp_path, "toy")

    assert (
        read_all(arrays, lambda read: read["layer.weight"] * settings["heads"]).sum()
        == 8
    )
    with pytest.raises(ValueError, match=r"lacks 'layer\.bias'"):
        read_all(arrays, lambda read: read["layer.bias"])
    with pytest.raises(ValueError, match=r"config\.json lacks the setting 'layers'"):
        read_all(arrays, lambda read: read["layer.weight"] * settings["layers"])


def test_a_cached_checkpoint_missing_its_weights_is_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    from generalforecast.models import registry

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    whole = tmp_path / "whole"
    calls = []

    def snapshot_download(*_, local_files_only: bool = False, **__) -> str:
        calls.append(local_files_only)
        return str(partial if local_files_only else whole)

    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    # A checkpoint without recorded hashes, whose cache lacks its weights.
    assert registry._checkpoint("limix-2-classifier") == whole
    assert calls == [True, False]


def test_built_in_models_go_by_name_alias_or_repository_id() -> None:
    from generalforecast.models import registry

    assert "general-forecast/tabicl-v2-classifier" in registry.BUILT_IN
    # Names, aliases and repository ids parse to one canonical reference.
    for spec in ("tabicl-v2-classifier", "tabicl-classifier"):
        assert catalog.parse(spec) == catalog.parse(
            "general-forecast/tabicl-v2-classifier"
        )
    with pytest.raises(ValueError, match="unknown model"):
        registry.load("tabicl-v9-classifier")


def test_forecasters_load_by_name_and_take_no_estimators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generalforecast.models import registry
    from generalforecast.models.checkpoints import save_state
    from generalforecast.models.series import SeriesFunctions
    from tests.models.chronos2.tiny import tiny_checkpoint

    state, settings = tiny_checkpoint()
    arrays = {key: np.asarray(value) for key, value in state.items()}
    save_state(tmp_path / "chronos-2", "chronos2", settings, arrays)
    monkeypatch.setenv("GF_CHECKPOINTS", str(tmp_path))
    # The pinned mirror refuses other bytes; the tiny weights load unhashed.
    with pytest.raises(ValueError, match="differs from its pin"):
        registry.load("general-forecast/chronos-2")
    entry = catalog.ENTRIES["chronos-2"]
    unhashed = dataclasses.replace(
        entry, artifact=dataclasses.replace(entry.artifact, files=MappingProxyType({}))
    )

    functions, parameters = registry._native(unhashed, "local", False)

    assert isinstance(functions, SeriesFunctions)
    assert (
        functions.properties(parameters).patch
        == settings["chronos_config"]["input_patch_size"]
    )
    for spec in ("general-forecast/chronos-2@2", "general-forecast/chronos-2*2"):
        with pytest.raises(ValueError, match="forecaster"):
            registry.load(spec)


def test_built_in_models_say_the_contexts_they_were_trained_on() -> None:
    from generalforecast.models import registry
    from generalforecast.models.contract import TabularFunctions
    from generalforecast.models.tabicl.model import tabicl_functions
    from tests.models.tabicl.tiny import tiny_tabicl

    model = (tabicl_functions, tiny_tabicl(regression=False))
    limits = []
    for name in ("mitra-classifier", "mitra-2-regressor", "tabpfn-3-classifier"):
        functions, parameters = registry._trained(name, model)
        assert isinstance(functions, TabularFunctions)
        found = functions.properties(parameters)
        limits.append((found.context_rows, found.context_features))

    assert limits == [(512, 16), (5_120, 50), (1_000_000, 2_000)]
    assert registry._trained("custom", model) is model
