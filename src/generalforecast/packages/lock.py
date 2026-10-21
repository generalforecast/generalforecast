# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Versioned immutable compiler output with independently verifiable digests."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import yaml

from generalforecast.packages.files import atomic_write, parse_yaml, read
from generalforecast.packages.models import CatalogResolver, ModelPin, ModelResolver
from generalforecast.packages.schema import TableSchema
from generalforecast.packages.spec import PackageSpec
from generalforecast.packages.validation import (
    digest,
    ensure,
    fields,
    hash_value,
    names,
)

LOCK_VERSION = 1
COMPILER_VERSION = "1"

if TYPE_CHECKING:
    from generalforecast.packages.python_lock import PythonLock


@dataclass(frozen=True)
class PackageLock:
    """Portable revision content, without credentials, paths outside the package or runtime."""

    spec: PackageSpec
    model: ModelPin
    task: str
    schema: TableSchema
    source_digest: str
    data_digest: str | None
    model_digest: str
    package_digest: str
    outputs: tuple[str, ...]
    context_rows: int | None
    cpu: int
    memory: str

    @property
    def kind(self) -> Literal["catalog"]:
        """Discriminate the catalog variant without changing its v1 wire bytes."""
        return "catalog"

    @property
    def identity(self) -> str:
        """Return admitted model content provenance shared with the Python variant."""
        return self.model.resolved

    def verify_model(
        self, resolver: ModelResolver | None = None, *, offline: bool = True
    ) -> ModelPin:
        """Authenticates untrusted capability metadata against an admitted catalog.

        Content hashes prove integrity, not catalog admission. The model's
        identity must equal the catalog's, and the serving policy the lock was
        compiled under must stay within the policy the catalog admits now.
        Remote admission must call this before accepting a client-supplied lock.
        """
        trusted = (resolver or CatalogResolver()).resolve(
            self.model.resolved, offline=offline
        )
        locked = self.model
        ensure(
            trusted.identity() == locked.identity()
            and locked.memory == trusted.memory
            and all(
                (getattr(locked, key) is None) == (getattr(trusted, key) is None)
                and (getattr(locked, key) or 0) <= (getattr(trusted, key) or 0)
                for key in ("max_context_rows", "max_features", "max_horizon", "cpu")
            ),
            "locked model capabilities differ from admitted catalog",
            "ModelMismatch",
        )
        return trusted

    def admit(
        self, resolver: ModelResolver | None = None, *, offline: bool = True
    ) -> ModelPin:
        """Verifies the model, then the shared capability and policy admission."""
        from generalforecast.packages.capabilities import admit

        trusted = self.verify_model(resolver, offline=offline)
        admit(self.spec, self.model, self.task, self.outputs)
        return trusted

    def to_dict(self) -> dict:
        """Returns fresh wire values; package hashes every field except itself."""
        return {
            "lock_version": LOCK_VERSION,
            "compiler_version": COMPILER_VERSION,
            "package": self.package_digest,
            "source_digest": self.source_digest,
            "data_digest": self.data_digest,
            "model_digest": self.model_digest,
            "spec": self.spec.to_dict(),
            "model": self.model.to_dict(),
            "task": self.task,
            "schema": self.schema.to_dict(),
            "outputs": list(self.outputs),
            "context_rows": self.context_rows,
            "compute": {"cpu": self.cpu, "memory": self.memory},
        }

    @classmethod
    def build(
        cls,
        *,
        spec: PackageSpec,
        model: ModelPin,
        task: str,
        schema: TableSchema,
        source_digest: str,
        data_digest: str | None,
        outputs: tuple[str, ...],
        context_rows: int | None,
    ) -> "PackageLock":
        """Computes distinct model, source, data and complete package identities."""
        model_digest = digest(model.identity())
        draft = cls(
            spec,
            model,
            task,
            schema,
            source_digest,
            data_digest,
            model_digest,
            "",
            outputs,
            context_rows,
            spec.cpu or model.cpu,
            spec.memory or model.memory,
        )
        value = draft.to_dict()
        value.pop("package")
        return replace(draft, package_digest=digest(value))

    @classmethod
    def from_dict(cls, value: Any) -> "PackageLock | PythonLock":
        """Rejects unknown versions, tampering and inconsistent component hashes."""
        if type(value) is dict and value.get("lock_version") == 2:
            from generalforecast.packages.python_lock import PythonLock

            return PythonLock.from_dict(value)
        fields(
            value,
            {
                "lock_version",
                "compiler_version",
                "package",
                "source_digest",
                "data_digest",
                "model_digest",
                "spec",
                "model",
                "task",
                "schema",
                "outputs",
                "context_rows",
                "compute",
            },
        )
        ensure(
            type(value["lock_version"]) is int
            and value["lock_version"] == LOCK_VERSION
            and value["compiler_version"] == COMPILER_VERSION,
            "unsupported lock or compiler version",
            "UnsupportedLockVersion",
        )
        hash_value(value["source_digest"])
        hash_value(value["model_digest"])
        hash_value(value["package"])
        if value["data_digest"] is not None:
            hash_value(value["data_digest"])
        unsigned = {key: child for key, child in value.items() if key != "package"}
        ensure(
            digest(unsigned) == value["package"],
            "package digest does not match lock content",
            "LockMismatch",
        )
        spec = PackageSpec.from_dict(value["spec"])
        model = ModelPin.from_dict(value["model"])
        ensure(
            digest(model.identity()) == value["model_digest"],
            "model digest differs from locked capabilities",
            "LockMismatch",
        )
        ensure(
            value["task"] in model.tasks
            and (spec.task is None or spec.task == value["task"]),
            "locked task is inconsistent",
        )
        schema = TableSchema.from_dict(value["schema"])
        outputs = names(value["outputs"])
        ensure(
            bool(outputs) and set(outputs) <= set(model.outputs),
            "locked outputs exceed model capabilities",
        )
        fields(value["compute"], {"cpu", "memory"})
        ensure(
            value["compute"]
            == {"cpu": spec.cpu or model.cpu, "memory": spec.memory or model.memory},
            "locked compute differs from intent",
        )
        rows = value["context_rows"]
        ensure(
            rows is None or (type(rows) is int and 0 <= rows <= model.max_context_rows),
            "locked context exceeds model capacity",
        )
        result = cls(
            spec,
            model,
            value["task"],
            schema,
            value["source_digest"],
            value["data_digest"],
            value["model_digest"],
            value["package"],
            outputs,
            rows,
            value["compute"]["cpu"],
            value["compute"]["memory"],
        )
        ensure(
            result.to_dict() == value,
            "lock must use normalized compiler serialization",
            "LockMismatch",
        )
        return result

    @classmethod
    def load(cls, root) -> "PackageLock | PythonLock":
        """Reads exactly gf.lock without following a symlink."""
        return cls.from_dict(parse_yaml(read(root, "gf.lock")))

    def write(self, root) -> None:
        """Atomically writes gf.lock only, leaving authored input and data unchanged."""
        self.from_dict(self.to_dict())
        atomic_write(
            root,
            "gf.lock",
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True).encode(
                "utf-8"
            ),
        )
