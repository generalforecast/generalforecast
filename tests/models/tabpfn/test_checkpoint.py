# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file

from generalforecast.models import checkpoints
from generalforecast.models.tabpfn import checkpoint
from generalforecast.models.tabpfn.checkpoint import read_published, tabpfn_checkpoint
from tests.models.test_checkpoints import _tensor, _text, _write

CONFIG = {"config": {"embed_dim": 4}, "inference_config": {"N_ESTIMATORS": 8}}


def _ckpt(path: Path) -> Path:
    """A TabPFN-3-shaped `torch.save` file with one 2 x 3 weight."""
    pickled = (
        b"\x80\x02}("
        + _text("architecture_name")
        + _text("tabpfn_v3")
        + _text("config")
        + b"}("
        + _text("embed_dim")
        + b"K\x04u"
        + _text("inference_config")
        + b"}("
        + _text("N_ESTIMATORS")
        + b"K\x08u"
        + _text("state_dict")
        + b"}("
        + _text("x_embed.weight")
        + _tensor("0", 6, 0, (2, 3), (3, 1))
        + b"u"
        + b"u."
    )
    _write(path, pickled, {"0": np.arange(6, dtype=np.float32)})
    return path


def _safetensors(path: Path) -> Path:
    metadata = {key: json.dumps(value) for key, value in CONFIG.items()}
    metadata["architecture_name"] = json.dumps("tabpfn_v3_5")
    save_file({"x_embed.weight": np.ones((2, 3), dtype=np.float32)}, path, metadata)
    return path


def test_reads_published_ckpt(tmp_path: Path) -> None:
    architecture, settings, state = read_published(_ckpt(tmp_path / "model.ckpt"))

    assert architecture == "tabpfn_v3"
    assert settings == {"embed_dim": 4, "inference_config": {"N_ESTIMATORS": 8}}
    np.testing.assert_array_equal(state["x_embed.weight"], np.arange(6).reshape(2, 3))


def test_reads_published_safetensors(tmp_path: Path) -> None:
    path = _safetensors(tmp_path / "model.safetensors")

    architecture, settings, state = read_published(path)

    assert architecture == "tabpfn_v3_5"
    assert settings == {"embed_dim": 4, "inference_config": {"N_ESTIMATORS": 8}}
    np.testing.assert_array_equal(state["x_embed.weight"], np.ones((2, 3)))


def test_converts_once_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _ckpt(tmp_path / "published.ckpt")
    downloads = []

    def download(*args: object, **kwargs: object) -> str:
        downloads.append(args)
        # Slow enough that the other caller asks while this one converts.
        time.sleep(0.2)
        return str(published)

    monkeypatch.setattr(checkpoints, "hf_hub_download", download)
    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with ThreadPoolExecutor(2) as pool:
        first, again = pool.map(
            tabpfn_checkpoint, ["general-forecast/tabpfn-3-classifier"] * 2
        )

    revision = checkpoint.PUBLISHED["general-forecast/tabpfn-3-classifier"].revision
    assert (
        first
        == again
        == tmp_path / "cache/generalforecast/checkpoints/tabpfn-3-classifier" / revision
    )
    assert len(downloads) == 1
    metadata = json.loads((first / "config.json").read_text())
    assert metadata["architecture_name"] == "tabpfn_v3"
    assert metadata["config"]["inference_config"] == {"N_ESTIMATORS": 8}
    np.testing.assert_array_equal(
        load_file(first / "model.safetensors")["x_embed.weight"],
        np.arange(6).reshape(2, 3),
    )


def test_offline_directory_must_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GF_CHECKPOINTS", str(tmp_path))
    (tmp_path / "tabpfn-3.5-classifier").mkdir()

    assert (
        tabpfn_checkpoint("general-forecast/tabpfn-3.5-classifier")
        == tmp_path / "tabpfn-3.5-classifier"
    )
    with pytest.raises(FileNotFoundError, match="tabpfn-3-regressor"):
        tabpfn_checkpoint("general-forecast/tabpfn-3-regressor")


@pytest.mark.parametrize("name", sorted(checkpoint.PUBLISHED))
def test_published_files_convert_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    published = checkpoint.PUBLISHED[name]
    path = hf_hub_download(
        published.repository, published.filename, revision=published.revision
    )
    if path.endswith(".ckpt"):
        expected = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    else:
        from safetensors.torch import load_file as load_torch

        expected = load_torch(path)

    expected = checkpoint.one_task(expected, regression=name.endswith("-regressor"))
    converted = load_file(tabpfn_checkpoint(name) / "model.safetensors")

    assert set(converted) == set(expected)
    for key, values in expected.items():
        np.testing.assert_array_equal(converted[key], values.numpy())


def test_catalog_converts_the_published_ckpt(tmp_path: Path) -> None:
    from generalforecast.models import artifacts, catalog

    entry = catalog.entry("general-forecast/tabpfn-3-classifier")
    (weights,) = (name for name in entry.artifact.files if name.endswith(".ckpt"))
    raw = tmp_path / "raw"
    raw.mkdir()
    _ckpt(raw / weights)
    (raw / "LICENSE").write_text("license")

    artifacts.convert(entry, raw, tmp_path / "out")

    metadata = json.loads((tmp_path / "out/config.json").read_text())
    assert metadata["architecture_name"] == "tabpfn_v3"
    assert (tmp_path / "out/LICENSE").read_text() == "license"
    np.testing.assert_array_equal(
        load_file(tmp_path / "out/model.safetensors")["x_embed.weight"],
        np.arange(6).reshape(2, 3),
    )
