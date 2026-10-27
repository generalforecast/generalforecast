# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verified checkpoint directories for built-in models, the one way they are found.

A built-in model's files come from `GF_CHECKPOINTS/<name>` when that is set,
as on machines without internet, or else from the Hugging Face cache, which
downloads them at their pinned revision unless the caller or `GF_OFFLINE=1`
forbids it; the first download reports one `Pulling` status line. Either way
every pinned file is hashed before anything reads it, and a missing pinned
config or weights file fails rather than passing unchecked. Predictors and remote
serving also require every pinned license file.

Families converted at load are hashed as upstream bytes before converting, and
the converted directory records a derived manifest: the upstream content digest
it came from and the SHA-256 of each file it wrote. Using a converted directory
verifies that manifest, so a stale or tampered cache never loads silently. Only
trusted local use reads a directory without catalog hashes, for models whose
entry pins a revision alone, or converted directories written before manifests
existed. Even those are held to digests (`verify_revision`): each file's
upstream digest where the Hugging Face cache records one, and on every use the
digests the files had the first time this machine used that revision.
"""

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from generalforecast.models.catalog import DERIVED_MANIFEST, ModelEntry, Surface

# The files of a checkpoint without recorded hashes, read as the original
# registry read them.
_DEFAULT_FILES = ("config.json", "model.safetensors", "LICENSE", "NOTICE")
_RUNTIME_FILES = ("config.json", "model.safetensors")

# Receives each status line as `(verb, message)`.
Progress = Callable[[str, str], None]


class ArtifactError(ValueError):
    """A checkpoint that is unavailable or differs from its pin, with a code."""

    def __init__(self, code: str, message: str, *, hint: str | None = None):
        """Keeps a machine-readable code and the hint apart from the message."""
        self.code = code
        self.hint = hint
        super().__init__(message)


def status(verb: str, message: str) -> None:
    """Writes one status line to stderr: the verb right-aligned in 12 columns, then the message."""
    print(f"{verb:>12} {message}", file=sys.stderr, flush=True)


def offline() -> bool:
    """Whether `GF_OFFLINE` forbids downloads on this machine."""
    return os.environ.get("GF_OFFLINE", "").strip().lower() not in ("", "0", "false")


def local_checkpoint(name: str) -> Path | None:
    """A checkpoint's directory under `GF_CHECKPOINTS`, or None if that is unset.

    Raises:
        FileNotFoundError: If `GF_CHECKPOINTS` is set but holds no such directory,
            as on a machine without internet that was given the wrong one.
    """
    local = os.environ.get("GF_CHECKPOINTS")
    if not local:
        return None
    path = Path(local) / name
    if not path.is_dir():
        raise FileNotFoundError(f"GF_CHECKPOINTS has no {name!r}: {path}")
    return path


def sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remembered_sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes, hashed once while its size, times and inode stay the same.

    Each process loading a checkpoint verifies it, so the digest is kept in
    `$XDG_CACHE_HOME/generalforecast/digests` under the file's identity.
    """
    status = path.stat()
    identity = "|".join(
        str(part)
        for part in (
            path.resolve(),
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
            status.st_ino,
        )
    )
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    memo = (
        root
        / "generalforecast"
        / "digests"
        / hashlib.sha256(identity.encode()).hexdigest()
    )
    try:
        digest = memo.read_text()
        if len(digest) == 64:
            return digest
    except OSError:
        pass
    digest = sha256(path)
    try:
        memo.parent.mkdir(parents=True, exist_ok=True)
        part = memo.with_suffix(f".{os.getpid()}.part")
        part.write_text(digest)
        part.replace(memo)
    except OSError:
        pass
    return digest


def verify(
    directory: Path,
    files: Mapping[str, str],
    optional: tuple[str, ...] = (),
    remembered: bool = False,
) -> None:
    """Refuses a directory whose pinned files are missing or differ.

    Args:
        directory: The checkpoint directory.
        files: SHA-256 of each pinned file.
        optional: Pinned files that may be absent; present, they must match.
        remembered: Whether a file unchanged since it was last hashed takes
            its remembered digest; see `remembered_sha256`.

    Raises:
        ArtifactError: `ModelUnavailable` for a missing required file,
            `ArtifactMismatch` for one whose bytes differ.
    """
    for name, expected in files.items():
        path = directory / name
        if not path.is_file():
            if name in optional:
                continue
            raise ArtifactError(
                "ModelUnavailable", f"checkpoint lacks its pinned file {name!r}"
            )
        digest = remembered_sha256(path) if remembered else sha256(path)
        if digest != expected:
            raise ArtifactError(
                "ArtifactMismatch", f"checkpoint file {name!r} differs from its pin"
            )


def checkpoint(
    entry: ModelEntry,
    *,
    surface: Surface = "local",
    allow_downloads: bool = True,
    progress: Progress | None = None,
) -> Path:
    """A built-in model's verified checkpoint directory.

    Args:
        entry: The model's catalog entry.
        surface: Where it runs. Predictors and remote serving need the entry's
            gate, every pinned file including license evidence and, for a
            converted family, a verified derived manifest.
        allow_downloads: Whether missing files may download at their pins;
            `GF_OFFLINE=1` forbids it whatever the caller says.
        progress: Receives the `Pulling` status line of a download as
            `(verb, message)`; by default it goes to stderr.

    Returns:
        A directory with the model's `config.json` and `model.safetensors`.

    Raises:
        ArtifactError: If the surface is not qualified, or files differ from
            their pins; with code `NotFound` and a `gf model pull` hint if
            they are missing and downloads are off.
    """
    strict = surface != "local"
    allow_downloads = allow_downloads and not offline()
    progress = progress or status
    if not entry.allows(surface):
        raise ArtifactError(
            "UnsupportedModel", f"{entry.id} is not qualified for {surface} use"
        )
    if strict and not entry.artifact.files:
        raise ArtifactError("UnsupportedModel", f"{entry.id} has no pinned files")
    if entry.artifact.convert:
        return _converted(entry, strict, allow_downloads, progress)
    directory = local_checkpoint(entry.directory) or _download(
        entry, strict, allow_downloads, progress
    )
    verify(
        directory,
        entry.artifact.files,
        () if strict else entry.artifact.license_files,
        remembered=not strict,
    )
    if not entry.artifact.files:
        verify_revision(entry, directory)
    return directory


def _git_blob(path: Path) -> str:
    """A file's git blob id: SHA-1 over `blob <size>`, a NUL byte and its bytes."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _upstream_digest(path: Path) -> tuple[str, str] | None:
    """The digest the hub published for a cached file: `(algorithm, hex)`, or None.

    The Hugging Face cache links each snapshot file to a blob named by the
    hub's ETag for it at the pinned revision: an LFS file's SHA-256, or a
    plain file's git blob id (SHA-1). A file outside that layout has none.
    """
    if not path.is_symlink():
        return None
    name = Path(os.readlink(path)).name
    if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
        return "sha256", name
    if len(name) == 40 and all(c in "0123456789abcdef" for c in name):
        return "git-blob", name
    return None


def _revisions() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "generalforecast" / "revisions"


def verify_revision(entry: ModelEntry, directory: Path) -> None:
    """Holds a checkpoint pinned by revision alone to the digests its files have.

    The catalog records no file hashes for such a model (its repository is
    not publicly readable when the catalog is built), so two checks stand
    in for them:

    - Upstream digests: a file the Hugging Face cache links to a blob named
      by the hub's digest must hash to it, so bytes that changed after the
      download are refused.
    - Trust on first use: the first time this machine uses the revision, the
      SHA-256 of each file is recorded under
      `$XDG_CACHE_HOME/generalforecast/revisions`; every later use must
      find the same bytes, wherever they come from (`GF_CHECKPOINTS` or the
      cache). The record trusts that first download; it does not prove what
      upstream published.

    Raises:
        ArtifactError: `ArtifactMismatch` naming the first file whose bytes
            differ from either digest.
    """
    names = sorted(name for name in _DEFAULT_FILES if (directory / name).is_file())
    digests: dict[str, str] = {}
    for name in names:
        path = directory / name
        digests[name] = remembered_sha256(path)
        upstream = _upstream_digest(path)
        if upstream is None:
            continue
        algorithm, expected = upstream
        actual = digests[name] if algorithm == "sha256" else _git_blob(path)
        if actual != expected:
            raise ArtifactError(
                "ArtifactMismatch",
                f"checkpoint file {name!r} differs from the digest upstream published",
            )
    record = _revisions() / f"{entry.name}@{entry.artifact.revision}.json"
    try:
        seen = json.loads(record.read_text())
    except (OSError, ValueError):
        seen = None
    if isinstance(seen, dict):
        for name, digest in digests.items():
            if seen.get(name, digest) != digest:
                raise ArtifactError(
                    "ArtifactMismatch",
                    f"checkpoint file {name!r} differs from the one first used at "
                    f"revision {entry.artifact.revision[:12]}",
                )
        if set(digests) <= set(seen):
            return
        digests = {**seen, **digests}
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        part = record.with_suffix(f".{os.getpid()}.part")
        part.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
        part.replace(record)
    except OSError:
        pass


def cached(entry: ModelEntry) -> bool:
    """Whether a model's files are on this machine, so loading it downloads nothing."""
    try:
        checkpoint(entry, allow_downloads=False)
    except ArtifactError as error:
        if error.code == "NotFound":
            return False
        raise
    return True


def _download(
    entry: ModelEntry, strict: bool, allow_downloads: bool, progress: Progress
) -> Path:
    """The pinned upstream files from the Hugging Face cache, downloading if allowed."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    artifact = entry.artifact
    patterns = list(artifact.files) or list(_DEFAULT_FILES)
    required = (
        list(artifact.files)
        if strict
        else [name for name in patterns if name in _RUNTIME_FILES]
    )

    def download(local_files_only: bool) -> Path:
        return Path(
            snapshot_download(
                artifact.repository,
                revision=artifact.revision,
                allow_patterns=patterns,
                local_files_only=local_files_only,
            )
        )

    unavailable = ArtifactError(
        "NotFound",
        f"{entry.id} is not cached",
        hint=f"run `gf model pull {entry.id}`"
        + (", or unset GF_OFFLINE" if offline() else ""),
    )

    def complete(directory: Path) -> bool:
        # The cache holds a revision's folder once any of its files arrived, so
        # a download cut short leaves one without its weights.
        return all((directory / name).is_file() for name in required)

    def from_cache() -> Path | None:
        # From the cache without a word; only a real download shows progress.
        try:
            cached = download(True)
        except LocalEntryNotFoundError:
            return None
        except Exception:  # noqa: BLE001
            raise unavailable from None
        return cached if complete(cached) else None

    if (cached := from_cache()) is not None:
        return cached
    if not allow_downloads:
        raise unavailable
    # One thread, and one process, pulls a revision; the others wait for it
    # and then find it in the cache.
    from filelock import FileLock

    with _PULLS_LOCK:
        lock = _PULLS.setdefault(artifact.revision, threading.Lock())
    cache = _hub_cache()
    cache.mkdir(parents=True, exist_ok=True)
    with lock, FileLock(cache / f"gf-pull-{artifact.revision}.lock"):
        if (cached := from_cache()) is not None:
            return cached
        progress("Pulling", _pulling(entry, patterns))
        try:
            with _quiet_hub():
                return download(False)
        except Exception:  # noqa: BLE001
            # Hub errors may carry signed URLs or tokens; none of it is surfaced.
            raise ArtifactError(
                "ModelUnavailable", f"{entry.id} could not be downloaded"
            ) from None


_PULLS: dict[str, threading.Lock] = {}
_PULLS_LOCK = threading.Lock()


def _hub_cache() -> Path:
    """The Hugging Face hub cache, `~/.cache/huggingface/hub` unless `HF_HUB_CACHE` says otherwise."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


@contextlib.contextmanager
def _quiet_hub() -> Iterator[None]:
    """Keeps the hub's progress bars and token advice off the terminal; the status line says enough."""
    import logging

    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    logger = logging.getLogger("huggingface_hub")
    level = logger.level
    shown = not are_progress_bars_disabled()
    logger.setLevel(logging.ERROR)
    disable_progress_bars()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logger.setLevel(level)
        if shown:
            enable_progress_bars()


def _pulling(entry: ModelEntry, patterns: list[str]) -> str:
    """The message of the `Pulling` line: the id, the download's size and where it goes."""
    cache = _hub_cache()
    if cache.name == "hub":
        cache = cache.parent
    try:
        destination = f"~/{cache.relative_to(Path.home())}"
    except ValueError:
        destination = str(cache)
    size = _size(entry, patterns)
    return f"{entry.id}  " + (f"({size}) " if size else "") + f"to {destination}"


def _size(entry: ModelEntry, patterns: list[str]) -> str | None:
    """The size of the files to download, such as `1.2 GB`, or None if the hub does not say."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(
            entry.artifact.repository,
            revision=entry.artifact.revision,
            files_metadata=True,
        )
    except Exception:  # noqa: BLE001
        return None
    total = sum(
        sibling.size or 0
        for sibling in info.siblings or ()
        if sibling.rfilename in patterns
    )
    if total <= 0:
        return None
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if total >= scale:
            return f"{total / scale:.1f} {unit}"
    return f"{total} B"


def _cache(entry: ModelEntry) -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (
        root
        / "generalforecast"
        / "checkpoints"
        / entry.directory
        / entry.artifact.revision
    )


def derived_manifest(entry: ModelEntry, directory: Path) -> dict[str, Any]:
    """What a converted directory records: its source and each file's SHA-256."""
    return {
        "source": {
            "repository": entry.artifact.repository,
            "revision": entry.artifact.revision,
            "digest": entry.artifact.digest,
            "convert": entry.artifact.convert,
        },
        "files": {
            path.name: sha256(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != DERIVED_MANIFEST
        },
    }


def _derived(entry: ModelEntry, directory: Path, strict: bool) -> bool:
    """Whether a converted directory verifies against its derived manifest.

    Returns:
        False if it has no manifest, which only trusted local use accepts.

    Raises:
        ArtifactError: If the manifest names other upstream content, or a file
            differs from it.
    """
    path = directory / DERIVED_MANIFEST
    if not path.is_file():
        if strict:
            raise ArtifactError(
                "ArtifactMismatch",
                f"converted {entry.id} checkpoint has no derived manifest",
            )
        return False
    try:
        manifest = json.loads(path.read_text())
        source, files = manifest["source"], manifest["files"]
    except (ValueError, KeyError, TypeError):
        raise ArtifactError(
            "ArtifactMismatch", f"converted {entry.id} manifest is unreadable"
        ) from None
    expected = derived_manifest(entry, directory)["source"]
    if source != expected or type(files) is not dict:
        raise ArtifactError(
            "ArtifactMismatch",
            f"converted {entry.id} checkpoint came from other upstream content",
        )
    needed = set(_RUNTIME_FILES)
    if strict:
        needed |= set(entry.artifact.license_files)
    if not needed <= files.keys():
        raise ArtifactError(
            "ArtifactMismatch", f"converted {entry.id} checkpoint lacks its files"
        )
    verify(directory, files)
    return True


def _usable(entry: ModelEntry, directory: Path, strict: bool) -> bool:
    """Whether a cached converted directory may load, rather than convert again."""
    try:
        return _derived(entry, directory, strict) or not strict
    except ArtifactError:
        return False


def _converted(
    entry: ModelEntry, strict: bool, allow_downloads: bool, progress: Progress
) -> Path:
    if (local := local_checkpoint(entry.directory)) is not None:
        _derived(entry, local, strict)
        return local
    directory = _cache(entry)
    if directory.is_dir() and _usable(entry, directory, strict):
        return directory
    from filelock import FileLock

    directory.parent.mkdir(parents=True, exist_ok=True)
    # One process converts; the others wait, then find its directory.
    with FileLock(directory.parent / f"{entry.artifact.revision}.lock"):
        if directory.is_dir():
            if _usable(entry, directory, strict):
                return directory
            # A cache this module wrote that no longer verifies converts again.
            shutil.rmtree(directory)
        raw = _download(entry, True, allow_downloads, progress)
        verify(raw, entry.artifact.files)
        # Written aside and renamed, so an interrupted conversion leaves nothing.
        with tempfile.TemporaryDirectory(dir=directory.parent) as scratch:
            out = Path(scratch) / entry.artifact.revision
            convert(entry, raw, out)
            manifest = derived_manifest(entry, out)
            (out / DERIVED_MANIFEST).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            os.replace(out, directory)
    return directory


def convert(entry: ModelEntry, raw: Path, directory: Path) -> None:
    """Writes a converted checkpoint from verified upstream files.

    The weights keep their PyTorch parameter names under the family's
    architecture, beside the upstream settings; license evidence is copied.
    """
    from safetensors.numpy import load_file

    from generalforecast.models.checkpoints import save_state

    assert entry.artifact.convert is not None
    if entry.family == "tabpfn":
        from generalforecast.models.tabpfn.checkpoint import one_task, read_published

        (weights,) = (
            name
            for name in entry.artifact.files
            if name.endswith((".safetensors", ".ckpt"))
        )
        architecture, settings, state = read_published(raw / weights)
        if architecture != entry.artifact.convert:
            raise ArtifactError(
                "ModelUnavailable", f"{entry.id} has an unexpected architecture"
            )
        regression = entry.capability.task == "regression"
        save_state(directory, architecture, settings, one_task(state, regression))
    else:
        settings = json.loads((raw / "config.json").read_text())
        save_state(
            directory,
            entry.artifact.convert,
            settings,
            load_file(raw / "model.safetensors"),
        )
    for name in entry.artifact.license_files:
        shutil.copyfile(raw / name, directory / name)
