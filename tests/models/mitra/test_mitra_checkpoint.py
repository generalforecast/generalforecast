# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from generalforecast.models.mitra import checkpoint
from generalforecast.models.mitra.checkpoint import mitra_checkpoint


def _published(directory: Path) -> dict[str, Path]:
    """AutoGluon's two files, as its repositories hold them."""
    directory.mkdir()
    config = directory / "config.json"
    config.write_text(json.dumps({"dim": 4, "task": "CLASSIFICATION"}))
    weights = directory / "model.safetensors"
    save_file({"final_layer.bias": np.arange(3, dtype=np.float32)}, weights)
    return {"config.json": config, "model.safetensors": weights}


def test_converts_once_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _published(tmp_path / "published")
    downloads = []

    def download(repository: str, filename: str, revision: str) -> str:
        downloads.append((repository, filename, revision))
        return str(files[filename])

    monkeypatch.setattr(checkpoint, "hf_hub_download", download)
    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    first = mitra_checkpoint("general-forecast/mitra-classifier")
    again = mitra_checkpoint("general-forecast/mitra-classifier")

    revision = checkpoint.PUBLISHED["general-forecast/mitra-classifier"].revision
    assert (
        first
        == again
        == tmp_path / "cache/generalforecast/checkpoints/mitra-classifier" / revision
    )
    assert len(downloads) == 2
    metadata = json.loads((first / "config.json").read_text())
    assert metadata == {
        "architecture_name": "mitra",
        "config": {"dim": 4, "task": "CLASSIFICATION"},
    }
    np.testing.assert_array_equal(
        load_file(first / "model.safetensors")["final_layer.bias"], np.arange(3)
    )


def test_offline_directory_must_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GF_CHECKPOINTS", str(tmp_path))
    (tmp_path / "mitra-classifier").mkdir()

    assert (
        mitra_checkpoint("general-forecast/mitra-classifier")
        == tmp_path / "mitra-classifier"
    )
    with pytest.raises(FileNotFoundError, match="mitra-regressor"):
        mitra_checkpoint("general-forecast/mitra-regressor")


def test_published_classifier_converts_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    published = checkpoint.PUBLISHED["general-forecast/mitra-classifier"]
    path = checkpoint.hf_hub_download(
        published.repository, "model.safetensors", revision=published.revision
    )

    directory = mitra_checkpoint("general-forecast/mitra-classifier")
    expected = load_file(path)
    converted = load_file(directory / "model.safetensors")

    assert set(converted) == set(expected)
    for key, values in expected.items():
        np.testing.assert_array_equal(converted[key], values)
    checkpoint.load_mitra(directory)
