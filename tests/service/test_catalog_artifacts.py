# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Curated checkpoint content verification and the offline-image preload."""

import hashlib

import pytest

from generalforecast.models import artifacts, catalog
from generalforecast.models.artifacts import ArtifactError
from generalforecast.service.catalog_artifacts import preload, verify
from generalforecast.service.contracts import ServiceError

ENTRY = catalog.ENTRIES["general-forecast/tabicl-v2-classifier"]


def test_content_verification_rejects_substitution_and_missing_license(tmp_path):
    files = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in ["config.json", "model.safetensors", "LICENSE"]
    }
    for name in files:
        (tmp_path / name).write_text(name)
    verify(tmp_path, files)
    (tmp_path / "model.safetensors").write_text("substituted weights")
    with pytest.raises(ServiceError) as caught:
        verify(tmp_path, files)
    assert caught.value.code == "ArtifactMismatch"
    (tmp_path / "model.safetensors").write_text("model.safetensors")
    (tmp_path / "LICENSE").unlink()
    with pytest.raises(ServiceError) as caught:
        verify(tmp_path, files)
    assert caught.value.code == "ModelUnavailable"


def test_preload_verifies_every_remote_qualified_model(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        artifacts,
        "checkpoint",
        lambda entry, **kwargs: loaded.append((entry.name, kwargs)),
    )
    preload()
    assert loaded == [
        (name, {"surface": "remote", "allow_downloads": True})
        for name in (
            "tabicl-v2-classifier",
            "tabicl-v2-regressor",
            *(
                f"kumo-tabular-{size}-{task}"
                for size in ("small", "medium", "large")
                for task in ("classifier", "regressor")
            ),
            "mitra-2-classifier",
            "mitra-2-regressor",
            "chronos-2",
            "timestral-209m",
            "timestral-209m-unmirrored",
            "timestral-28m",
            "timestral-28m-unmirrored",
        )
    ]


def test_preload_rejects_corrupted_download_before_fetching_another_pin(
    monkeypatch, tmp_path
):
    import huggingface_hub

    calls = []
    for filename in ENTRY.artifact.files:
        (tmp_path / filename).write_bytes(b"corrupted")

    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(ArtifactError) as caught:
        preload()
    assert caught.value.code == "ArtifactMismatch"
    assert len(calls) == 1
