# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from generalforecast.models.limix import checkpoint
from generalforecast.models.limix.checkpoint import limix_checkpoint, read_published
from tests.models.test_checkpoints import _tensor, _text, _write


def _ckpt(path: Path) -> Path:
    """A LimiX-2-shaped `torch.save` file with one weight per task and a shared one."""
    pickled = (
        b"\x80\x02}("
        + _text("arch_version")
        + _text("2.0")
        + _text("config")
        + b"}("
        + _text("embed_dim")
        + b"K\x04u"
        + _text("state_dict")
        + b"}("
        + _text("encoder_x.0.mask_embedding.weight")
        + _tensor("0", 6, 0, (2, 3), (3, 1))
        + _text("cls_y_decoder.0.weight")
        + _tensor("1", 2, 0, (2,), (1,))
        + _text("reg_y_decoder.0.weight")
        + _tensor("2", 2, 0, (2,), (1,))
        + _text("feature_decoder.0.weight")
        + _tensor("3", 2, 0, (2,), (1,))
        + b"u"
        + b"u."
    )
    arrays = {
        key: np.arange(size, dtype=np.float32)
        for key, size in zip("0123", (6, 2, 2, 2), strict=True)
    }
    _write(path, pickled, arrays)
    return path


def test_reads_published_ckpt(tmp_path: Path) -> None:
    settings, state = read_published(_ckpt(tmp_path / "model.ckpt"))

    assert settings == {"embed_dim": 4, "arch_version": "2.0"}
    np.testing.assert_array_equal(
        state["encoder_x.0.mask_embedding.weight"], np.arange(6).reshape(2, 3)
    )


def test_converts_each_task_once_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _ckpt(tmp_path / "published.ckpt")
    downloads = []
    monkeypatch.setattr(
        checkpoint,
        "hf_hub_download",
        lambda *args, **kwargs: downloads.append(args) or str(published),
    )
    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    first = limix_checkpoint("general-forecast/limix-2-classifier")
    again = limix_checkpoint("general-forecast/limix-2-classifier")
    regressor = limix_checkpoint("general-forecast/limix-2-regressor")

    revision = checkpoint.LIMIX_2.revision
    assert (
        first
        == again
        == tmp_path / "cache/generalforecast/checkpoints/limix-2-classifier" / revision
    )
    assert len(downloads) == 2
    metadata = json.loads((first / "config.json").read_text())
    assert metadata["architecture_name"] == "limix"
    assert metadata["config"]["arch_version"] == "2.0"
    assert set(load_file(first / "model.safetensors")) == {
        "encoder_x.0.mask_embedding.weight",
        "cls_y_decoder.0.weight",
    }
    assert set(load_file(regressor / "model.safetensors")) == {
        "encoder_x.0.mask_embedding.weight",
        "reg_y_decoder.0.weight",
    }


def test_offline_directory_must_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GF_CHECKPOINTS", str(tmp_path))
    (tmp_path / "limix-2-classifier").mkdir()

    assert (
        limix_checkpoint("general-forecast/limix-2-classifier")
        == tmp_path / "limix-2-classifier"
    )
    with pytest.raises(FileNotFoundError, match="limix-2-regressor"):
        limix_checkpoint("general-forecast/limix-2-regressor")


@pytest.mark.parametrize("name", sorted(checkpoint.PUBLISHED))
def test_published_files_convert_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    published = checkpoint.PUBLISHED[name]
    path = checkpoint.hf_hub_download(
        published.repository, published.filename, revision=published.revision
    )
    expected = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]

    expected = checkpoint.one_task(expected, regression=name.endswith("-regressor"))
    directory = limix_checkpoint(name)
    converted = load_file(directory / "model.safetensors")

    assert set(converted) == set(expected)
    for key, values in expected.items():
        np.testing.assert_array_equal(converted[key], values.numpy())
    checkpoint.load_limix(directory)
