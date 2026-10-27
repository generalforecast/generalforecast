# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from generalforecast.models import artifacts, catalog
from generalforecast.models.artifacts import ArtifactError


def test_remembered_digests_hash_a_file_again_only_once_it_changes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights")
    hashed: list[object] = []
    real = artifacts.sha256
    monkeypatch.setattr(
        artifacts, "sha256", lambda file: hashed.append(file) or real(file)
    )

    first = artifacts.remembered_sha256(path)
    assert artifacts.remembered_sha256(path) == first
    assert first == hashlib.sha256(b"weights").hexdigest()
    assert len(hashed) == 1
    path.write_bytes(b"other weights")
    assert (
        artifacts.remembered_sha256(path)
        == hashlib.sha256(b"other weights").hexdigest()
    )
    assert len(hashed) == 2


@pytest.mark.parametrize("online", [False, True])
def test_checkpoint_access_is_explicitly_configured(online, monkeypatch) -> None:
    import huggingface_hub

    entry = catalog.ENTRIES["general-forecast/tabicl-v2-classifier"]
    calls = []

    def download(*args, local_files_only, **kwargs):
        calls.append({**kwargs, "local_files_only": local_files_only})
        if local_files_only:
            raise LocalEntryNotFoundError("not cached")
        raise OSError("network down")

    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(ArtifactError) as caught:
        artifacts.checkpoint(entry, surface="remote", allow_downloads=online)
    assert caught.value.code == ("ModelUnavailable" if online else "NotFound")
    assert [call["local_files_only"] for call in calls] == (
        [True, True, False] if online else [True]
    )
    assert len(calls[0]["revision"]) == 40
    assert calls[0]["allow_patterns"] == list(entry.artifact.files)
