# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Private state files: owner-only, no-follow, atomic, locked."""

import os
import signal
import threading

import pytest

from generalforecast import _files
from generalforecast._files import (
    create_private,
    open_private_exclusive,
    private_lock,
    read_private,
    write_private,
)
from generalforecast.service.contracts import ServiceError

posix = pytest.mark.skipif(
    not _files.secure_persistence_supported(), reason="POSIX-only guarantees"
)


@posix
def test_atomic_write_is_private_and_replaces_only_regular_files(tmp_path):
    path = tmp_path / "state" / "profiles.json"
    write_private(path, b"one")
    write_private(path, b"two")
    assert read_private(path, limit=3) == b"two"
    assert path.stat().st_mode & 0o777 == 0o600
    assert (path.parent.stat().st_mode & 0o777) == 0o700
    assert sorted(p.name for p in path.parent.iterdir()) == ["profiles.json"]
    with pytest.raises(ServiceError, match="too large"):
        read_private(path, limit=2)
    assert read_private(tmp_path / "missing", limit=1, missing_ok=True) is None
    with pytest.raises(ServiceError):
        read_private(tmp_path / "missing", limit=1)


@posix
@pytest.mark.parametrize("kind", ["symlink", "public", "fifo", "directory"])
def test_reads_and_writes_reject_indirect_or_shared_files(tmp_path, kind):
    path, other = tmp_path / "config", tmp_path / "other"
    other.write_bytes(b"x")
    other.chmod(0o600)
    if kind == "symlink":
        path.symlink_to(other)
    elif kind == "public":
        path.write_bytes(b"x")
        path.chmod(0o644)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    else:
        path.mkdir(mode=0o700)
    with pytest.raises(ServiceError) as error:
        read_private(path, limit=10)
    assert error.value.code == "ValidationError"
    with pytest.raises(ServiceError):
        write_private(path, b"y")
    assert other.read_bytes() == b"x"


@posix
def test_exclusive_create_never_overwrites_and_cleans_up_failures(tmp_path):
    path = tmp_path / "key.json"
    create_private(path, b"first")
    with pytest.raises(ServiceError) as error:
        create_private(path, b"second")
    assert error.value.code == "Conflict" and path.read_bytes() == b"first"
    failed = tmp_path / "failed.json"
    with pytest.raises(RuntimeError), open_private_exclusive(failed) as output:
        output.write(b"partial")
        raise RuntimeError
    assert not failed.exists()


@posix
def test_lock_serializes_read_modify_write(tmp_path):
    path = tmp_path / "counter"
    write_private(path, b"0")

    def increment():
        for _ in range(25):
            with private_lock(path):
                value = int(read_private(path, limit=16) or b"0")
                write_private(path, str(value + 1).encode())

    threads = [threading.Thread(target=increment) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert read_private(path, limit=16) == b"100"
    assert (tmp_path / "counter.lock").stat().st_mode & 0o777 == 0o600


def test_unsupported_platform_fails_typed_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(_files, "secure_persistence_supported", lambda: False)
    for call in (
        lambda: read_private(tmp_path / "x", limit=1),
        lambda: write_private(tmp_path / "x", b"x"),
        lambda: create_private(tmp_path / "x", b"x"),
    ):
        with pytest.raises(ServiceError) as error:
            call()
        assert error.value.code == "UnsupportedOperation"
    with pytest.raises(ServiceError), private_lock(tmp_path / "x"):
        pass
    assert not list(tmp_path.iterdir())


@posix
def test_named_pipe_is_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "profiles.json"
    os.mkfifo(fifo, 0o600)

    def hung(*_):
        raise AssertionError("read_private blocked on a FIFO")

    previous = signal.signal(signal.SIGALRM, hung)
    signal.alarm(5)
    try:
        for missing_ok in (False, True):
            with pytest.raises(ServiceError) as error:
                read_private(fifo, limit=10, missing_ok=missing_ok)
            assert error.value.code == "ValidationError"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
