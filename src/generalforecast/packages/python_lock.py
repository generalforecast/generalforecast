# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Source-bound Python v2 locks and data-only probe admission."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from generalforecast.packages.files import atomic_write
from generalforecast.packages.python_spec import PythonPackageSpec, PythonSchema
from generalforecast.packages.validation import (
    PackageError,
    canonical,
    digest,
    ensure,
    fields,
    hash_value,
    integer,
)
from generalforecast.python_packages.api import MAX_WIRE, Table
from generalforecast.python_packages.binding import PythonBinding
from generalforecast.python_packages.compiler import (
    MAX_ARCHIVE,
    MAX_FILE,
    VERSION,
    PythonRevision,
    PythonSpec,
    unpack,
)


def _manifest(binding: PythonBinding, spec: PythonSpec) -> dict:
    """Check lock-only integrity; archive admission still requires verify_source."""
    value = json.loads(binding.lock)
    fields(
        value, {"version", "revision", "archive", "dependency_digest", "spec", "files"}
    )
    ensure(
        value["version"] == VERSION
        and canonical(value["spec"]) == canonical(spec.to_dict()),
        "Python binding spec differs from package spec",
        "LockMismatch",
    )
    hash_value(value["revision"])
    hash_value(value["dependency_digest"])
    ensure(
        value["archive"] == binding.source
        and digest({k: v for k, v in value.items() if k != "revision"})
        == value["revision"],
        "Python binding digest mismatch",
        "LockMismatch",
    )
    expected = [
        (role, path)
        for role, paths in (
            ("code", spec.code),
            ("dependency", spec.dependencies),
            ("artifact", spec.artifacts),
        )
        for path in paths
    ]
    entries = value["files"]
    ensure(
        type(entries) is list and len(entries) == len(expected), "invalid Python files"
    )
    total = 0
    for entry, (role, path) in zip(entries, expected, strict=True):
        fields(entry, {"path", "role", "size", "digest"})
        ensure(
            entry["path"] == path and entry["role"] == role, "Python file roles differ"
        )
        total += integer(entry["size"], 0, MAX_FILE)
        hash_value(entry["digest"])
    ensure(total <= MAX_ARCHIVE, "Python source exceeds byte bound")
    ensure(
        digest([entry for entry in entries if entry["role"] == "dependency"])
        == value["dependency_digest"],
        "Python dependency digest mismatch",
        "LockMismatch",
    )
    return value


def _pairs(items: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in items:
        ensure(key not in result, "duplicate probe JSON key")
        result[key] = value
    return result


def validate_probe(revision: PythonRevision) -> Table:
    """Validate the pinned probe as data only; never run it as inference or context."""
    manifest = revision.inspect()
    spec = PythonSpec.from_dict(manifest["spec"])
    ensure("probe.json" in spec.artifacts, "probe.json is a required artifact")
    raw = unpack(revision.archive)["probe.json"]
    ensure(len(raw) <= MAX_WIRE, "probe exceeds table byte bound")
    try:
        probe = Table.from_dict(json.loads(raw, object_pairs_hook=_pairs))
    except PackageError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError):
        raise PackageError("InvalidPackage", "invalid Python probe table") from None
    ensure(
        probe.schema == spec.inputs,
        "probe schema differs from Python inputs",
        "SchemaMismatch",
    )
    ensure(1 <= len(probe.rows) <= 100, "probe must contain 1..100 rows")
    return probe


@dataclass(frozen=True)
class PythonLock:
    """Content-addressed Python variant with no catalog ModelPin or model digest."""

    spec: PythonPackageSpec
    python: PythonBinding
    package_digest: str

    @property
    def kind(self) -> Literal["python"]:
        """Discriminate the lock without inspecting catalog-only attributes."""
        return "python"

    @property
    def identity(self) -> str:
        """Return internal immutable Python content provenance, not a catalog ref."""
        return "python@" + _manifest(self.python, self.spec.python)["revision"]

    @property
    def source_digest(self) -> str:
        """Pin normalized intent and the complete explicit source archive."""
        return digest({"spec": self.spec.to_dict(), "archive": self.python.source})

    @property
    def data_digest(self) -> None:
        """Probe data is an artifact, never a snapshot context."""
        return None

    @property
    def task(self) -> str:
        """Return the declared predictor task."""
        return self.spec.task

    @property
    def schema(self) -> PythonSchema:
        """Return exact ordered input/output fields."""
        return PythonSchema(self.spec.python.inputs, self.spec.python.outputs)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Return output names in declared order."""
        return tuple(field.name for field in self.spec.python.outputs.fields)

    @property
    def context_rows(self) -> int:
        """No training context is captured by a Python revision."""
        return 0

    @property
    def cpu(self) -> int:
        """Return fixed worker policy."""
        return self.spec.cpu

    @property
    def memory(self) -> str:
        """Return fixed worker policy."""
        return self.spec.memory

    def to_dict(self) -> dict:
        """Return the exact v2 wire contract with no source bytes or runtime receipt."""
        return {
            "lock_version": 2,
            "compiler_version": "python-1",
            "kind": "python",
            "spec": self.spec.to_dict(),
            "python": self.python.to_dict(),
            "source_digest": self.source_digest,
            "data_digest": None,
            "task": self.task,
            "schema": self.schema.to_dict(),
            "outputs": list(self.outputs),
            "context_rows": 0,
            "compute": {"cpu": self.cpu, "memory": self.memory},
            "package": self.package_digest,
        }

    @classmethod
    def build(cls, spec: PythonPackageSpec, revision: PythonRevision) -> "PythonLock":
        """Pin verified source and probe without executing predictor code."""
        spec = PythonPackageSpec.from_dict(spec.to_dict())
        draft = cls(spec, PythonBinding.from_revision(revision), "")
        draft.verify_source(revision.archive)
        body = draft.to_dict()
        body.pop("package")
        return replace(draft, package_digest=digest(body))

    @classmethod
    def from_dict(cls, value: Any) -> "PythonLock":
        """Verify every duplicate declaration and digest; source bytes are separate."""
        fields(
            value,
            {
                "lock_version",
                "compiler_version",
                "kind",
                "spec",
                "python",
                "source_digest",
                "data_digest",
                "task",
                "schema",
                "outputs",
                "context_rows",
                "compute",
                "package",
            },
        )
        ensure(
            type(value["lock_version"]) is int
            and value["lock_version"] == 2
            and value["compiler_version"] == "python-1"
            and value["kind"] == "python",
            "unsupported Python lock or compiler version",
            "UnsupportedLockVersion",
        )
        hash_value(value["package"])
        hash_value(value["source_digest"])
        ensure(
            digest({k: v for k, v in value.items() if k != "package"})
            == value["package"],
            "package digest does not match lock content",
            "LockMismatch",
        )
        spec = PythonPackageSpec.from_dict(value["spec"])
        binding = PythonBinding.from_dict(value["python"])
        _manifest(binding, spec.python)
        result = cls(spec, binding, value["package"])
        ensure(
            canonical(result.to_dict()) == canonical(value),
            "Python lock declarations differ from normalized package intent",
            "LockMismatch",
        )
        return result

    def verify_source(self, archive: bytes) -> PythonRevision:
        """Verify uploaded bytes and required probe before any isolated execution."""
        _manifest(self.python, self.spec.python)
        revision = self.python.verify(archive)
        validate_probe(revision)
        return revision

    def write(self, root: Path) -> None:
        """Publish only gf.lock atomically; archives are exported explicitly."""
        self.from_dict(self.to_dict())
        atomic_write(
            root,
            "gf.lock",
            yaml.safe_dump(
                self.to_dict(), sort_keys=False, allow_unicode=True
            ).encode(),
        )
