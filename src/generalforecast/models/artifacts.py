"""Verified checkpoint directories for built-in models, the one way they are found.

A built-in model's files come from `GF_CHECKPOINTS/<name>` when that is set,
as on machines without internet, or else from the Hugging Face cache, which
downloads them at their pinned revision when the caller allows it. Either way
every pinned file is hashed before anything reads it, and a missing pinned
config or weights file fails rather than passing unchecked. Packages and remote
serving also require every pinned license file.

Families converted at load are hashed as upstream bytes before converting, and
the converted directory records a derived manifest: the upstream content digest
it came from and the SHA-256 of each file it wrote. Using a converted directory
verifies that manifest, so a stale or tampered cache never loads silently. Only
trusted local use reads a directory without hashes, for models whose catalog
entry records none, or converted directories written before manifests existed.
"""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from generalforecast.models.catalog import DERIVED_MANIFEST, ModelEntry, Surface

# The files of a checkpoint without recorded hashes, read as the original
# registry read them.
_LEGACY_FILES = ("config.json", "model.safetensors", "LICENSE", "NOTICE")
_RUNTIME_FILES = ("config.json", "model.safetensors")


class ArtifactError(ValueError):
    """A checkpoint that is unavailable or differs from its pin, with a code."""

    def __init__(self, code: str, message: str):
        """Keeps a machine-readable code apart from the message."""
        self.code = code
        super().__init__(message)


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


def verify(
    directory: Path, files: Mapping[str, str], optional: tuple[str, ...] = ()
) -> None:
    """Refuses a directory whose pinned files are missing or differ.

    Args:
        directory: The checkpoint directory.
        files: SHA-256 of each pinned file.
        optional: Pinned files that may be absent; present, they must match.

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
        if sha256(path) != expected:
            raise ArtifactError(
                "ArtifactMismatch", f"checkpoint file {name!r} differs from its pin"
            )


def checkpoint(
    entry: ModelEntry, *, surface: Surface = "local", allow_downloads: bool = True
) -> Path:
    """A built-in model's verified checkpoint directory.

    Args:
        entry: The model's catalog entry.
        surface: Where it runs. Packages and remote serving need the entry's
            gate, every pinned file including license evidence and, for a
            converted family, a verified derived manifest.
        allow_downloads: Whether missing files may download at their pins.

    Returns:
        A directory with the model's `config.json` and `model.safetensors`.

    Raises:
        ArtifactError: If the surface is not qualified, or files are missing
            or differ from their pins.
    """
    strict = surface != "local"
    if not entry.gates.allows(surface):
        raise ArtifactError(
            "UnsupportedModel", f"{entry.name} is not qualified for {surface} use"
        )
    if strict and not entry.artifact.files:
        raise ArtifactError("UnsupportedModel", f"{entry.name} has no pinned files")
    if entry.artifact.convert:
        return _converted(entry, strict, allow_downloads)
    directory = local_checkpoint(entry.directory) or _download(
        entry, strict, allow_downloads
    )
    verify(
        directory, entry.artifact.files, () if strict else entry.artifact.license_files
    )
    return directory


def _download(entry: ModelEntry, strict: bool, allow_downloads: bool) -> Path:
    """The pinned upstream files from the Hugging Face cache, downloading if allowed."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    artifact = entry.artifact
    patterns = list(artifact.files) or list(_LEGACY_FILES)
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
        "ModelUnavailable",
        f"pinned {entry.name} checkpoint is not cached; preload it or allow downloads",
    )
    try:
        # From the cache without a word; only a real download shows progress.
        cached = download(True)
        # The cache holds a revision's folder once any of its files arrived, so
        # a download cut short leaves one without its weights.
        if all((cached / name).is_file() for name in required):
            return cached
    except LocalEntryNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        raise unavailable from None
    if not allow_downloads:
        raise unavailable
    try:
        return download(False)
    except Exception:  # noqa: BLE001
        # Hub errors may carry signed URLs or tokens; none of it is surfaced.
        raise ArtifactError(
            "ModelUnavailable",
            f"pinned {entry.name} checkpoint could not be downloaded",
        ) from None


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
                f"converted {entry.name} checkpoint has no derived manifest",
            )
        return False
    try:
        manifest = json.loads(path.read_text())
        source, files = manifest["source"], manifest["files"]
    except (ValueError, KeyError, TypeError):
        raise ArtifactError(
            "ArtifactMismatch", f"converted {entry.name} manifest is unreadable"
        ) from None
    expected = derived_manifest(entry, directory)["source"]
    if source != expected or type(files) is not dict:
        raise ArtifactError(
            "ArtifactMismatch",
            f"converted {entry.name} checkpoint came from other upstream content",
        )
    needed = set(_RUNTIME_FILES)
    if strict:
        needed |= set(entry.artifact.license_files)
    if not needed <= files.keys():
        raise ArtifactError(
            "ArtifactMismatch", f"converted {entry.name} checkpoint lacks its files"
        )
    verify(directory, files)
    return True


def _usable(entry: ModelEntry, directory: Path, strict: bool) -> bool:
    """Whether a cached converted directory may load, rather than convert again."""
    try:
        return _derived(entry, directory, strict) or not strict
    except ArtifactError:
        return False


def _converted(entry: ModelEntry, strict: bool, allow_downloads: bool) -> Path:
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
        raw = _download(entry, True, allow_downloads)
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
                "ModelUnavailable", f"{entry.name} has an unexpected architecture"
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
