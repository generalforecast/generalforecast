# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Owner-private local files: no-follow reads, atomic 0600 writes and locks.

Importing this module is portable. Calls fail with a typed UnsupportedOperation
where the platform cannot provide no-follow opens, POSIX modes or advisory
locks, instead of silently persisting state with weaker guarantees.
"""

import contextlib
import errno
import os
import secrets
import stat
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from generalforecast.service.contracts import ServiceError

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
# How often a lock file's creation is retried after a racing creator's ENOENT.
_CREATE_ATTEMPTS = 50


def secure_persistence_supported() -> bool:
    """Reports whether owner-private persistence is enforceable on this host."""
    return (
        os.name == "posix"
        and bool(_NOFOLLOW and _DIRECTORY)
        and fcntl is not None
        and hasattr(os, "geteuid")
    )


def _supported() -> None:
    if not secure_persistence_supported():
        raise ServiceError(
            "UnsupportedOperation",
            "secure private files are unavailable on this platform",
            501,
        )


def _private(info: os.stat_result) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_mode & 0o077
        or info.st_uid != os.geteuid()
    ):
        raise ServiceError(
            "ValidationError",
            "requires a private regular file owned by the current user",
            422,
        )


def _unopenable(error: OSError) -> ServiceError:
    """The error for a private file that cannot be opened, naming its errno.

    A link refused by `O_NOFOLLOW` is not a private regular file; any other
    failure says what the system reported.
    """
    code = errno.errorcode.get(error.errno or 0, "unknown")
    if error.errno in (errno.ELOOP, errno.EMLINK):
        return ServiceError(
            "ValidationError",
            "requires a private regular file owned by the current user, "
            f"not a link ({code})",
            422,
        )
    return ServiceError(
        "ValidationError",
        f"private file cannot be opened: {os.strerror(error.errno or 0).lower()} ({code})",
        422,
    )


def _directory(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return os.open(path.parent, os.O_RDONLY | _DIRECTORY | _CLOEXEC)


def read_private(
    path: str | os.PathLike[str], *, limit: int, missing_ok: bool = False
) -> bytes | None:
    """Reads at most limit bytes from an owner-only regular file, never a link."""
    _supported()
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW | _CLOEXEC | _NONBLOCK)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ServiceError(
            "ValidationError", "private file is unavailable", 422
        ) from None
    except OSError as error:
        raise _unopenable(error) from error
    try:
        _private(os.fstat(fd))
        chunks, size = [], 0
        while size <= limit and (chunk := os.read(fd, limit + 1 - size)):
            chunks.append(chunk)
            size += len(chunk)
    finally:
        os.close(fd)
    if size > limit:
        raise ServiceError("ValidationError", "private file is too large", 422)
    return b"".join(chunks)


def write_private(path: str | os.PathLike[str], data: bytes) -> None:
    """Atomically replaces path with a fsynced 0600 file in the same directory."""
    _supported()
    target = Path(path)
    directory = _directory(target)
    temporary = f".{target.name}.{secrets.token_hex(8)}.tmp"
    try:
        try:
            _private(os.stat(target.name, dir_fd=directory, follow_symlinks=False))
        except FileNotFoundError:
            pass
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, target.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory)
        raise
    finally:
        os.close(directory)


@contextlib.contextmanager
def open_private_exclusive(path: str | os.PathLike[str]) -> Iterator[BinaryIO]:
    """Creates a new 0600 file; removes it if the caller's block fails."""
    _supported()
    target = Path(path)
    directory = _directory(target)
    try:
        try:
            fd = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC,
                0o600,
                dir_fd=directory,
            )
        except FileExistsError:
            raise ServiceError("Conflict", "private file already exists", 409) from None
        try:
            with os.fdopen(fd, "wb") as output:
                yield output
                output.flush()
                os.fsync(output.fileno())
            os.fsync(directory)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(target.name, dir_fd=directory)
            raise
    finally:
        os.close(directory)


def create_private(path: str | os.PathLike[str], data: bytes) -> None:
    """Creates path exclusively with mode 0600 and fsyncs the data."""
    with open_private_exclusive(path) as output:
        output.write(data)


@contextlib.contextmanager
def private_lock(path: str | os.PathLike[str]) -> Iterator[None]:
    """Holds an exclusive advisory lock on the owner-only path.lock sibling."""
    _supported()
    assert fcntl is not None
    target = Path(path)
    directory = _directory(target)
    try:
        fd = _open_lock(target.name + ".lock", directory)
    except OSError as error:
        raise _unopenable(error) from error
    finally:
        os.close(directory)
    try:
        _private(os.fstat(fd))
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _open_lock(name: str, directory: int) -> int:
    """Opens or creates a lock file, retrying a create that lost a race.

    On macOS, when several processes or threads create the same new file at
    once with `O_CREAT`, the losers can fail with ENOENT although the file
    exists a moment later; opening it again succeeds. Any other errno, or
    ENOENT that persists, is raised as it is.
    """
    for attempt in range(_CREATE_ATTEMPTS):
        try:
            return os.open(
                name,
                os.O_RDWR | os.O_CREAT | _NOFOLLOW | _CLOEXEC,
                0o600,
                dir_fd=directory,
            )
        except FileNotFoundError:
            if attempt == _CREATE_ATTEMPTS - 1:
                raise
            time.sleep(0.001 * (attempt + 1))
    raise AssertionError("unreachable")
