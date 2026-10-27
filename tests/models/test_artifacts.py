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


def _hub_snapshot(root, files: dict[str, bytes]):
    """A Hugging Face cache snapshot: each file links to a blob named by its hub ETag."""
    blobs, snapshot = root / "blobs", root / "snapshots" / "rev"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for name, data in files.items():
        if name.endswith(".safetensors"):
            etag = hashlib.sha256(data).hexdigest()
        else:
            etag = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        (blobs / etag).write_bytes(data)
        (snapshot / name).symlink_to(blobs / etag)
    return snapshot


REVISION_ONLY = "general-forecast/limix-2-classifier"


def test_revision_pins_are_held_to_upstream_digests(tmp_path, monkeypatch) -> None:
    import huggingface_hub

    entry = catalog.ENTRIES[REVISION_ONLY]
    assert not entry.artifact.files
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    snapshot = _hub_snapshot(
        tmp_path / "hub",
        {"config.json": b'{"a": 1}', "model.safetensors": b"weights"},
    )
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *_, **__: str(snapshot)
    )
    assert artifacts.checkpoint(entry, allow_downloads=False) == snapshot
    # Bytes that changed after the download no longer match the blob's name.
    (snapshot / "model.safetensors").resolve().write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="digest upstream published") as caught:
        artifacts.checkpoint(entry, allow_downloads=False)
    assert caught.value.code == "ArtifactMismatch"
    (snapshot / "model.safetensors").resolve().write_bytes(b"weights")
    (snapshot / "config.json").resolve().write_bytes(b'{"a": 2}')
    with pytest.raises(ArtifactError, match=r"config\.json"):
        artifacts.checkpoint(entry, allow_downloads=False)


def test_revision_pins_trust_their_first_use(tmp_path, monkeypatch) -> None:
    entry = catalog.ENTRIES[REVISION_ONLY]
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("GF_CHECKPOINTS", str(tmp_path / "checkpoints"))
    directory = tmp_path / "checkpoints" / entry.directory
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"weights")
    assert artifacts.checkpoint(entry) == directory
    assert artifacts.checkpoint(entry) == directory
    recorded = list((tmp_path / "xdg" / "generalforecast" / "revisions").iterdir())
    assert [path.name for path in recorded] == [
        f"{entry.name}@{entry.artifact.revision}.json"
    ]
    (directory / "model.safetensors").write_bytes(b"other weights")
    with pytest.raises(ArtifactError, match="first used at revision") as caught:
        artifacts.checkpoint(entry)
    assert caught.value.code == "ArtifactMismatch"
