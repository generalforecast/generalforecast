# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, explicit-file Python revisions; compilation never imports code."""

import ast
import io
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from generalforecast.packages.files import atomic_write, read, relative
from generalforecast.packages.validation import (
    PackageError,
    bytes_digest,
    canonical,
    digest,
    ensure,
    fields,
    hash_value,
)
from generalforecast.python_packages.api import TableSchema

MAX_FILE = 8 * 1024 * 1024
MAX_ARCHIVE = 32 * 1024 * 1024
VERSION = "gf.python/v1"
_SECRET = re.compile(
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    rb"(?:gf_live_|hf_|ghp_|github_pat_)[A-Za-z0-9_-]{20,}|"
    rb"(?:AKIA|ASIA)[A-Z0-9]{16}"
)


@dataclass(frozen=True)
class PythonSpec:
    """Explicit opt-in source, offline pure-Python wheels and artifact allowlists.

    No directories, globs, URLs, environment expansion or automatic project scan.
    Wheel resolution is external: the lock pins every supplied wheel by bytes.
    """

    inputs: TableSchema
    outputs: TableSchema
    code: tuple[str, ...] = ("predictor.py",)
    dependencies: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    task: str = "regression"

    def to_dict(self) -> dict:
        """Encode the authored, executable-free contract."""
        return {
            "version": VERSION,
            "inputs": self.inputs.to_dict(),
            "outputs": self.outputs.to_dict(),
            "code": list(self.code),
            "dependencies": list(self.dependencies),
            "artifacts": list(self.artifacts),
            **({"task": self.task} if self.task != "regression" else {}),
        }

    @classmethod
    def from_dict(cls, value: object) -> "PythonSpec":
        """Validate the complete file allowlist and table contract."""
        value = fields(
            value,
            {"version", "inputs", "outputs", "code", "dependencies", "artifacts"},
            {"task"},
        )
        ensure(value["version"] == VERSION, "unsupported Python package version")
        task = value.get("task", "regression")
        ensure(
            task in ("regression", "classification", "forecast"),
            "unsupported Python task",
        )
        groups = []
        for key in ("code", "dependencies", "artifacts"):
            paths = value[key]
            ensure(type(paths) is list and len(paths) <= 128, "invalid file allowlist")
            group = tuple(sorted(relative(path) for path in paths))
            ensure(len(set(group)) == len(group), "duplicate package file")
            groups.append(group)
        code, dependencies, artifacts = groups
        ensure("predictor.py" in code, "predictor.py must be explicitly included")
        ensure(all(p.endswith(".py") for p in code), "code must be Python source")
        ensure(
            all(p.endswith("-py3-none-any.whl") for p in dependencies),
            "only offline pure Python py3-none-any wheels are supported",
        )
        ensure(
            len(set(code + dependencies + artifacts))
            == len(code + dependencies + artifacts),
            "file roles overlap",
        )
        return cls(
            TableSchema.from_dict(value["inputs"]),
            TableSchema.from_dict(value["outputs"]),
            code,
            dependencies,
            artifacts,
            task,
        )


def wheel_files(raw: bytes) -> dict[str, bytes]:
    """Validate offline wheels without installing or executing dependency code.

    Script entrypoints, .pth hooks, native wheels and .data relocation are refused.
    Dependency closure is deliberately explicit, not resolved using ambient pip.
    """
    result: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
            ensure(len(wheel.infolist()) <= 4096, "wheel contains too many files")
            total = 0
            for item in wheel.infolist():
                if item.is_dir():
                    continue
                path = relative(item.filename)
                ensure(
                    path == item.filename and path not in result,
                    "duplicate or noncanonical wheel path",
                )
                ensure(
                    (item.external_attr >> 16) & 0o170000 in (0, 0o100000),
                    "wheel must contain regular files",
                )
                ensure(
                    not path.endswith((".pth", ".so", ".pyc"))
                    and not any(p.endswith(".data") for p in path.split("/")),
                    "unsupported wheel executable hook or native extension",
                )
                ensure(
                    path.split("/")[0]
                    not in {
                        "generalforecast",
                        "sitecustomize.py",
                        "usercustomize.py",
                    },
                    "wheel shadows reserved runtime modules",
                )
                total += item.file_size
                ensure(
                    item.file_size <= MAX_FILE and total <= MAX_ARCHIVE,
                    "wheel exceeds expanded byte bound",
                )
                data = wheel.read(item)
                ensure(
                    not _SECRET.search(data),
                    "embedded credential material is forbidden",
                )
                result[path] = data
            metadata = [
                data
                for path, data in result.items()
                if path.endswith(".dist-info/WHEEL")
            ]
            ensure(
                len(metadata) == 1
                and b"Root-Is-Purelib: true" in metadata[0]
                and b"Tag: py3-none-any" in metadata[0],
                "wheel must declare pure Python platform-independent content",
            )
        return result
    except (zipfile.BadZipFile, RuntimeError, OSError):
        raise PackageError("InvalidPackage", "invalid offline wheel") from None


def _archive(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path, raw in sorted(files.items()):
            info = tarfile.TarInfo(path)
            info.mode, info.size = 0o444, len(raw)
            tar.addfile(info, io.BytesIO(raw))
    ensure(stream.tell() <= MAX_ARCHIVE, "source archive exceeds byte bound")
    return stream.getvalue()


def unpack(archive: bytes) -> dict[str, bytes]:
    """Read canonical regular-file-only source archives, never extract to a host path."""
    ensure(
        type(archive) is bytes and len(archive) <= MAX_ARCHIVE,
        "archive exceeds byte bound",
    )
    files = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            total = 0
            for member in tar:
                path = relative(member.name)
                total += member.size
                ensure(
                    member.isfile()
                    and member.sparse is None
                    and not member.pax_headers
                    and member.size <= MAX_FILE
                    and total <= MAX_ARCHIVE
                    and len(files) < 384,
                    "archive contains unsafe members",
                )
                ensure(
                    path == member.name and path not in files,
                    "archive contains duplicate or noncanonical paths",
                )
                stream = tar.extractfile(member)
                ensure(stream is not None, "archive member missing")
                assert stream is not None
                files[path] = stream.read(MAX_FILE + 1)
    except (tarfile.TarError, OSError):
        ensure(False, "invalid source archive")
    ensure(_archive(files) == archive, "archive is not canonical")
    return files


@dataclass(frozen=True)
class PythonRevision:
    """Immutable verified lock bytes and source archive; revision hashes all content."""

    lock: bytes
    archive: bytes

    def inspect(self) -> dict:
        """Verify hashes and declarations without importing any predictor code."""
        ensure(len(self.lock) <= 262144, "Python lock exceeds byte bound")
        try:
            manifest = json.loads(self.lock)
        except (ValueError, UnicodeError):
            raise PackageError("InvalidPackage", "invalid Python lock") from None
        fields(
            manifest,
            {"version", "revision", "archive", "dependency_digest", "spec", "files"},
        )
        ensure(canonical(manifest) == self.lock, "Python lock is not canonical")
        ensure(manifest["version"] == VERSION, "unsupported Python lock version")
        spec = PythonSpec.from_dict(manifest["spec"])
        ensure(spec.to_dict() == manifest["spec"], "Python spec is not normalized")
        hash_value(manifest["revision"])
        ensure(
            digest({k: v for k, v in manifest.items() if k != "revision"})
            == manifest["revision"],
            "Python revision digest mismatch",
        )
        ensure(
            bytes_digest(self.archive) == manifest["archive"],
            "Python source archive digest mismatch",
        )
        source = unpack(self.archive)
        expected = _entries(spec, source)
        ensure(expected == manifest["files"], "Python file manifest mismatch")
        ensure(
            digest([f for f in expected if f["role"] == "dependency"])
            == manifest["dependency_digest"],
            "Python dependency digest mismatch",
        )
        return manifest

    @property
    def digest(self) -> str:
        """Return a verified immutable revision identity."""
        return self.inspect()["revision"]

    @property
    def spec(self) -> PythonSpec:
        """Return the verified typed serving contract."""
        return PythonSpec.from_dict(self.inspect()["spec"])

    def write(self, root: Path) -> None:
        """Write generated files only; publish the lock last as the commit marker."""
        self.inspect()
        atomic_write(root, "gf-python-source.tar", self.archive)
        atomic_write(root, "gf-python.lock", self.lock)

    @classmethod
    def load(cls, root: Path) -> "PythonRevision":
        """Load and verify generated lock/archive with no-follow reads."""
        result = cls(
            read(root, "gf-python.lock", limit=262144),
            read(root, "gf-python-source.tar", limit=MAX_ARCHIVE),
        )
        result.inspect()
        return result


def _entries(spec: PythonSpec, source: dict[str, bytes]) -> list[dict]:
    ensure(
        set(source) == set(spec.code + spec.dependencies + spec.artifacts),
        "source files differ from explicit allowlist",
    )
    _check_file_paths(set(source))
    entries = []
    installed: set[str] = set()
    expanded = 0
    for role, paths in (
        ("code", spec.code),
        ("dependency", spec.dependencies),
        ("artifact", spec.artifacts),
    ):
        for path in paths:
            raw = source[path]
            ensure(not _SECRET.search(raw), "embedded credential material is forbidden")
            if role == "code":
                try:
                    ast.parse(raw, filename="<package>")
                except (SyntaxError, ValueError, UnicodeError, RecursionError):
                    ensure(False, "invalid predictor Python syntax")
            if role == "dependency":
                files = wheel_files(raw)
                ensure(not installed.intersection(files), "dependency wheels overlap")
                installed.update(files)
                _check_file_paths(installed)
                expanded += sum(map(len, files.values()))
                ensure(len(installed) <= 4096, "dependencies contain too many files")
            else:
                expanded += len(raw)
            ensure(
                expanded <= MAX_ARCHIVE, "expanded Python runtime exceeds byte bound"
            )
            entries.append(
                {
                    "path": path,
                    "role": role,
                    "size": len(raw),
                    "digest": bytes_digest(raw),
                }
            )
    return entries


def _check_file_paths(paths: set[str]) -> None:
    for path in paths:
        while "/" in path:
            path = path.rsplit("/", 1)[0]
            ensure(path not in paths, "package file conflicts with a directory")


def compile_python(root: Path, spec: PythonSpec) -> PythonRevision:
    """Build a deterministic source archive and lock from explicit no-follow files."""
    spec = PythonSpec.from_dict(spec.to_dict())
    source = {}
    total = 0
    for path in spec.code + spec.dependencies + spec.artifacts:
        raw = read(root, path, limit=MAX_FILE)
        total += len(raw)
        ensure(total <= MAX_ARCHIVE, "Python source exceeds byte bound")
        source[path] = raw
    entries = _entries(spec, source)
    archive = _archive(source)
    manifest = {
        "version": VERSION,
        "archive": bytes_digest(archive),
        "dependency_digest": digest([f for f in entries if f["role"] == "dependency"]),
        "spec": spec.to_dict(),
        "files": entries,
    }
    manifest["revision"] = digest(manifest)
    result = PythonRevision(canonical(manifest), archive)
    result.inspect()
    return result


def patch_kind(previous: PythonRevision, candidate: PythonRevision) -> str:
    """Dependencies/schema need rebuild; code/artifact revisions may warm-swap in dev."""
    old, new = previous.inspect(), candidate.inspect()
    if old["dependency_digest"] != new["dependency_digest"]:
        return "rebuild"
    if (
        previous.spec.inputs != candidate.spec.inputs
        or previous.spec.outputs != candidate.spec.outputs
        or previous.spec.task != candidate.spec.task
    ):
        return "rebuild"
    return "hot-swap"
