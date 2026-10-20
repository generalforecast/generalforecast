"""Directory-relative, no-follow inputs and durable atomic lock replacement."""

import os
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.resolver import BaseResolver

from generalforecast.packages.validation import PackageError, ensure, text

MAX_MANIFEST_BYTES = 1_048_576
MAX_DATA_BYTES = 128 * 1024 * 1024


def relative(value: Any) -> str:
    """Rejects traversal, hidden files, credentials and platform-specific escapes."""
    value = text(value)
    while value.startswith("./"):
        value = value[2:]
    path = PurePosixPath(value)
    ensure(
        not path.is_absolute() and "\\" not in value and ":" not in value,
        "path must be package-relative",
    )
    parts = value.split("/")
    ensure(
        all(
            part and part not in (".", "..") and not part.startswith(".")
            for part in parts
        ),
        "hidden paths and traversal are forbidden",
    )
    ensure(
        not any(
            part.lower() in {"secrets", "credentials", "id_rsa", "id_ed25519"}
            for part in parts
        ),
        "credential paths are forbidden",
    )
    return value


@contextmanager
def directory(root: Path):
    """Pins a directory handle without following any symlink component."""
    root = Path(os.path.abspath(root))
    fd = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
    except OSError:
        os.close(fd)
        raise PackageError(
            "UnsafePath", "package directory is unavailable or unsafe"
        ) from None
    try:
        yield fd
    finally:
        os.close(fd)


def read(root: Path, path: str, *, limit: int = MAX_MANIFEST_BYTES) -> bytes:
    """Reads one explicit regular file, detecting concurrent modification."""
    parts = relative(path).split("/")
    try:
        with directory(root) as base:
            fd = os.dup(base)
            try:
                for part in parts[:-1]:
                    child = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                    )
                    os.close(fd)
                    fd = child
                source = os.open(
                    parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
            finally:
                os.close(fd)
            with os.fdopen(source, "rb") as stream:
                before = os.fstat(stream.fileno())
                ensure(
                    stat.S_ISREG(before.st_mode) and before.st_size <= limit,
                    "file is not regular or exceeds the byte limit",
                    "UnsafePath",
                )
                result = stream.read(limit + 1)
                after = os.fstat(stream.fileno())
                ensure(
                    len(result) <= limit
                    and (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    "input changed while reading",
                    "SourceChanged",
                )
                return result
    except FileNotFoundError:
        raise PackageError("MissingFile", "required package file is missing") from None
    except OSError:
        raise PackageError(
            "UnsafePath", "package file is unavailable or unsafe"
        ) from None


class _Loader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        ensure(
            type(key) is str and key not in result, "YAML keys must be unique strings"
        )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_Loader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def parse_yaml(data: bytes) -> dict:
    """Refuses aliases, duplicate keys, tags and unbounded structures."""
    try:
        events = list(yaml.parse(data))
        ensure(len(events) <= 20000, "YAML is too large")
        ensure(
            not any(
                isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None)
                for event in events
            ),
            "YAML aliases are forbidden",
        )
        value = yaml.load(data, Loader=_Loader)
        ensure(type(value) is dict, "manifest must be an object")
        return value
    except (yaml.YAMLError, RecursionError, UnicodeError):
        raise PackageError("InvalidPackage", "invalid YAML") from None


def atomic_write(root: Path, path: str, data: bytes) -> None:
    """Fsyncs a private sibling temporary file, renames it, then fsyncs the folder."""
    path = relative(path)
    ensure("/" not in path, "generated files must be at the package root")
    temporary = f".gf-lock-{secrets.token_hex(12)}"
    with directory(root) as base:
        try:
            try:
                info = os.stat(path, dir_fd=base, follow_symlinks=False)
                ensure(
                    stat.S_ISREG(info.st_mode),
                    "refusing unsafe lock destination",
                    "UnsafePath",
                )
            except FileNotFoundError:
                pass
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=base,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path, src_dir_fd=base, dst_dir_fd=base)
            os.fsync(base)
        finally:
            try:
                os.unlink(temporary, dir_fd=base)
            except FileNotFoundError:
                pass
