# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib

from generalforecast.models import artifacts


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
