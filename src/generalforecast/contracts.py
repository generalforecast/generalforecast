"""Shared immutable surface values and ports, not a second control plane."""

import datetime
from dataclasses import MISSING, asdict, dataclass, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from typing import Protocol, Self
from urllib.parse import urlsplit

import pyarrow as pa

from generalforecast.deployments.boundary import Scope
from generalforecast.packages.lock import PackageLock
from generalforecast.packages.python_lock import PythonLock
from generalforecast.packages.validation import (
    PackageError,
    ensure,
    fields,
    hash_value,
    integer,
    name,
    text,
)

API_PREFIX = "/v1alpha1"
STATES = frozenset({"pending", "building", "ready", "failed", "stopped"})


class WireValue:
    """Dataclass values serialize to fresh JSON objects with no hidden I/O."""

    def to_dict(self) -> dict:
        """Copies nested values into JSON-compatible dictionaries and lists."""

        def convert(value):
            if isinstance(value, dict):
                return {key: convert(child) for key, child in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(child) for child in value]
            return value

        ensure(
            is_dataclass(self) and not isinstance(self, type),
            "wire values must be dataclass instances",
        )
        assert is_dataclass(self) and not isinstance(self, type)
        return convert(asdict(self))

    @classmethod
    def from_dict(cls, value: dict) -> Self:
        """Decodes the exact wire fields without accepting unknown credentials."""
        assert is_dataclass(cls)
        required = {
            field.name
            for field in dataclass_fields(cls)
            if field.default is MISSING and field.default_factory is MISSING
        }
        allowed = {field.name for field in dataclass_fields(cls)}
        body = dict(fields(value, required, allowed - required))
        for key in ("residency", "secret_refs", "mapping"):
            if key in body:
                ensure(type(body[key]) in (list, tuple), "expected a wire array")
                body[key] = (
                    tuple(tuple(pair) for pair in body[key])
                    if key == "mapping"
                    else tuple(body[key])
                )
        return cls(**body)


@dataclass(frozen=True)
class Revision(WireValue):
    """A named deployment's immutable package identity and lifecycle state."""

    deployment: str
    number: int
    package_digest: str
    model: str
    created_at: float
    status: str = "pending"

    def __post_init__(self):
        """Checks portable identity without claiming the revision is serving."""
        name(self.deployment)
        integer(self.number)
        hash_value(self.package_digest)
        text(self.model)
        ensure(
            type(self.created_at) in (int, float) and 0 <= self.created_at < 1e12,
            "invalid revision time",
        )
        ensure(self.status in STATES, "invalid revision status")


@dataclass(frozen=True)
class Environment(WireValue):
    """Mutable environment pointer; generation supports compare-and-swap updates."""

    deployment: str
    name: str
    revision: int | None
    generation: int
    runtime: str
    status: str = "pending"

    def __post_init__(self):
        """Keeps environment identity distinct from the immutable package."""
        name(self.deployment)
        name(self.name)
        name(self.runtime)
        integer(self.generation, 0)
        if self.revision is not None:
            integer(self.revision)
        ensure(self.status in STATES, "invalid environment status")

    def promote(self, request: "PromoteRequest") -> "Environment":
        """Pure CAS transition; the scoped store must persist it atomically."""
        ensure(
            request.expected_generation == self.generation,
            "environment generation is stale",
            "Conflict",
        )
        return replace(
            self,
            revision=request.revision,
            generation=self.generation + 1,
            status="pending",
        )


@dataclass(frozen=True)
class PromoteRequest(WireValue):
    """Promotion and rollback use the same explicit revision and CAS generation."""

    revision: int
    expected_generation: int

    def __post_init__(self):
        """Rejects last-writer-wins environment changes."""
        integer(self.revision)
        integer(self.expected_generation, 0)


@dataclass(frozen=True)
class ContextVersion(WireValue):
    """The actual context observed by inference, not an accepted-write promise."""

    source: str
    version: str | int | None = None
    rows: int | None = None
    as_of: float | None = None

    def __post_init__(self):
        """Checks version metadata without fetching a live dataset."""
        text(self.source)
        if self.version is not None:
            ensure(type(self.version) in (int, str), "invalid context version")
        if self.rows is not None:
            integer(self.rows, 0)
        if self.as_of is not None:
            ensure(
                type(self.as_of) in (int, float) and 0 <= self.as_of < 1e12,
                "invalid context time",
            )


@dataclass(frozen=True)
class DatasetImportRef(WireValue):
    """An import binds a trusted source to a logical dataset; credentials are refs only."""

    dataset: str
    source: str
    format: str
    credential_ref: str | None = None
    digest: str | None = None
    mapping: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        """Rejects signed URLs, userinfo and inline credentials before transmission."""
        name(self.dataset)
        ensure(
            type(self.source) is str
            and 0 < len(self.source) <= 2048
            and not any(ord(c) < 32 for c in self.source),
            "invalid import source",
        )
        parsed = urlsplit(self.source)
        ensure(
            parsed.scheme
            in (
                "s3",
                "gs",
                "az",
                "azure",
                "snowflake",
                "bigquery",
                "dataset",
                "upload",
            ),
            "import sources must be cloud or uploaded logical references",
        )
        ensure(
            parsed.netloc
            and not parsed.query
            and not parsed.fragment
            and parsed.username is None
            and parsed.password is None,
            "source must not contain credentials or signed query parameters",
        )
        ensure(
            ".." not in parsed.path.split("/") and "\\" not in self.source,
            "source traversal is forbidden",
        )
        ensure(
            "%" not in self.source,
            "encoded source credentials and traversal are forbidden",
        )
        text(parsed.netloc)
        if parsed.path:
            text(parsed.path)
        ensure(
            self.format in ("csv", "tsv", "parquet", "arrow", "json", "jsonl", "table"),
            "unsupported import format",
        )
        if self.credential_ref is not None:
            ensure(
                self.credential_ref.startswith("secret/"),
                "use a trusted secret reference",
            )
            name(self.credential_ref.removeprefix("secret/"))
        if self.digest is not None:
            hash_value(self.digest)
        ensure(type(self.mapping) is tuple, "import mapping must be immutable")
        for target, source in self.mapping:
            text(target)
            text(source)
        ensure(
            len(dict(self.mapping)) == len(self.mapping),
            "duplicate import mapping columns",
        )


@dataclass(frozen=True)
class RuntimePlan(WireValue):
    """Placement and resources for an exact package, separate from provider provisioning."""

    runtime: str
    environment: str
    package_digest: str
    model_digest: str | None
    data_digest: str | None
    region: str | None
    residency: tuple[str, ...]
    cpu: int
    memory: str
    min_replicas: int
    max_replicas: int
    secret_refs: tuple[str, ...] = ()

    def __post_init__(self):
        """Validates caller-selected runtime and trusted secret references."""
        name(self.runtime)
        name(self.environment)
        hash_value(self.package_digest)
        if self.model_digest is not None:
            hash_value(self.model_digest)
        if self.data_digest is not None:
            hash_value(self.data_digest)
        integer(self.cpu, 1, 256)
        text(self.memory)
        integer(self.min_replicas, 0, 100)
        integer(self.max_replicas, 1, 100)
        ensure(self.min_replicas <= self.max_replicas, "invalid scaling bounds")
        ensure(
            type(self.residency) is tuple and type(self.secret_refs) is tuple,
            "runtime plan collections must be immutable",
        )
        for region in self.residency:
            text(region)
        if self.region is not None:
            text(self.region)
            ensure(
                not self.residency or self.region in self.residency,
                "runtime region violates residency",
            )
        for ref in self.secret_refs:
            ensure(ref.startswith("secret/"), "runtime secrets must be references")
            name(ref.removeprefix("secret/"))
        if self.model_digest is None:
            ensure(
                self.data_digest is None
                and self.region is None
                and not self.residency
                and self.cpu == 1
                and self.memory == "256Mi"
                and self.min_replicas == self.max_replicas == 1
                and not self.secret_refs,
                "Python runtime placement requires one isolated worker without catalog context or secret injection",
                "UnsupportedCapability",
            )

    @classmethod
    def for_package(
        cls,
        lock: PackageLock | PythonLock,
        *,
        runtime: str,
        environment: str,
        secret_refs: tuple[str, ...] = (),
    ) -> "RuntimePlan":
        """Compiles portable placement without creating infrastructure or identities."""
        lock = PackageLock.from_dict(lock.to_dict())
        if isinstance(lock, PythonLock):
            return cls(
                runtime,
                environment,
                lock.package_digest,
                None,
                None,
                None,
                (),
                lock.cpu,
                lock.memory,
                lock.spec.min_replicas,
                lock.spec.max_replicas,
                secret_refs,
            )
        return cls(
            runtime,
            environment,
            lock.package_digest,
            lock.model_digest,
            lock.data_digest,
            lock.spec.region,
            lock.spec.residency,
            lock.cpu,
            lock.memory,
            lock.spec.min_replicas,
            lock.spec.max_replicas,
            secret_refs,
        )


@dataclass(frozen=True)
class DevSession(WireValue):
    """An isolated development pointer; never an environment or production revision."""

    id: str
    deployment: str
    base_revision: int
    generation: int
    package_digest: str
    expires_at: float
    status: str = "pending"

    def __post_init__(self):
        """Checks patch CAS identity without inventing an execution boundary."""
        name(self.id)
        name(self.deployment)
        integer(self.base_revision)
        integer(self.generation, 0)
        hash_value(self.package_digest)
        ensure(
            type(self.expires_at) in (int, float) and 0 <= self.expires_at < 1e12,
            "invalid dev-session expiry",
        )
        ensure(self.status in STATES, "invalid dev-session status")

    def apply_patch(self, request: "PatchRequest", *, now: float) -> "DevSession":
        """Pure isolated CAS transition; never mutates an environment pointer."""
        ensure(
            type(now) in (int, float)
            and 0 <= now < self.expires_at
            and self.status != "stopped",
            "development session is expired or stopped",
            "Conflict",
        )
        ensure(
            request.base_revision == self.base_revision
            and request.expected_generation == self.generation,
            "development base revision or generation is stale",
            "Conflict",
        )
        ensure(
            request.lock.spec.name == self.deployment,
            "patch deployment differs from development session",
        )
        return replace(
            self,
            generation=self.generation + 1,
            package_digest=request.lock.package_digest,
            status="pending",
        )


@dataclass(frozen=True)
class PatchRequest:
    """A complete replacement compiled package at an expected dev generation."""

    expected_generation: int
    base_revision: int
    lock: PackageLock | PythonLock
    snapshot: str | None = None

    def __post_init__(self):
        """Validates patch identity; server must atomically compare and update."""
        integer(self.expected_generation, 0)
        integer(self.base_revision)
        ensure(isinstance(self.lock, (PackageLock, PythonLock)), "invalid package lock")
        PackageLock.from_dict(self.lock.to_dict())
        if isinstance(self.lock, PythonLock):
            ensure(self.snapshot is None, "Python patches cannot include a snapshot")
            return
        if self.snapshot is not None:
            hash_value(self.snapshot)
        ensure(
            (self.lock.spec.context.mode == "snapshot") == (self.snapshot is not None),
            "snapshot patches require uploaded content identity",
        )
        if self.snapshot is not None:
            ensure(
                self.snapshot == self.lock.data_digest,
                "snapshot identity differs from locked data",
            )

    def to_dict(self) -> dict:
        """Copies the locked package with no mutable source-tree or credential capture."""
        return {
            "expected_generation": self.expected_generation,
            "base_revision": self.base_revision,
            "lock": self.lock.to_dict(),
            "snapshot": self.snapshot,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "PatchRequest":
        """Decodes a complete compiled patch, never executable source or inline secrets."""
        body = dict(
            fields(
                value, {"expected_generation", "base_revision", "lock"}, {"snapshot"}
            )
        )
        body["lock"] = PackageLock.from_dict(body["lock"])
        return cls(**body)


@dataclass(frozen=True)
class PredictionResult:
    """Typed SDK result retaining model and actual context provenance."""

    table: pa.Table
    model: str
    task: str
    context: ContextVersion | None = None
    deployment: str | None = None
    revision: int | None = None
    environment: str | None = None

    def __post_init__(self):
        """Copies normalized Arrow data without importing optional dataframe libraries."""
        ensure(
            isinstance(self.table, pa.Table),
            "prediction results require an Arrow table",
        )
        ensure(
            len(set(self.table.column_names)) == len(self.table.column_names),
            "duplicate prediction output columns",
        )
        object.__setattr__(
            self, "table", self.table.combine_chunks().replace_schema_metadata(None)
        )
        text(self.model)
        ensure(
            self.task in ("classification", "regression", "forecast"),
            "invalid prediction task",
        )
        if self.deployment is not None:
            name(self.deployment)
        if self.revision is not None:
            integer(self.revision)
        if self.environment is not None:
            name(self.environment)

    def to_arrow(self) -> pa.Table:
        """Returns the immutable Arrow result, preserving output column names."""
        return self.table

    def to_pandas(self):
        """Converts results when pandas is installed; never installs it implicitly."""
        return self.table.to_pandas()

    def to_polars(self):
        """Converts results when Polars is installed; never collects remote data."""
        import polars

        return polars.from_arrow(self.table)

    def to_numpy(self):
        """Returns a two-dimensional array in output column order."""
        import numpy as np

        return (
            np.column_stack([column.to_numpy() for column in self.table.columns])
            if self.table.num_columns
            else np.empty((self.table.num_rows, 0))
        )

    def to_dict(self) -> dict:
        """Serializes flat result rows and version provenance for the shared HTTP API."""

        def cell(value):
            if isinstance(value, dict):
                return {key: cell(child) for key, child in value.items()}
            if isinstance(value, list):
                return [cell(child) for child in value]
            if isinstance(value, (datetime.date, datetime.datetime)):
                return value.isoformat()
            ensure(
                not isinstance(value, bytes),
                "binary outputs require Arrow serialization",
            )
            return value

        return {
            "model": self.model,
            "task": self.task,
            "context": self.context.to_dict() if self.context else None,
            "deployment": self.deployment,
            "revision": self.revision,
            "environment": self.environment,
            "predictions": [
                {key: cell(value) for key, value in row.items()}
                for row in self.table.to_pylist()
            ],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "PredictionResult":
        """Decodes HTTP output rows and their actual context provenance."""
        body = dict(
            fields(
                value,
                {"model", "task", "predictions"},
                {"context", "deployment", "revision", "environment"},
            )
        )
        rows = body.pop("predictions")
        ensure(
            type(rows) is list and all(type(row) is dict for row in rows),
            "predictions must be records",
        )
        keys = list(dict.fromkeys(key for row in rows for key in row))
        try:
            body["table"] = pa.table(
                {key: [row.get(key) for row in rows] for key in keys}
            )
        except (ValueError, TypeError, pa.ArrowException):
            raise PackageError("InvalidResult", "prediction rows are invalid") from None
        if body.get("context") is not None:
            body["context"] = ContextVersion.from_dict(body["context"])
        return cls(**body)


class RevisionStore(Protocol):
    """Durable scoped revision and environment port implemented by the API owner."""

    def create(
        self,
        scope: Scope,
        deployment: str,
        lock: PackageLock | PythonLock,
        *,
        snapshot: str | None = None,
    ) -> Revision:
        """Persists immutable package identity and authenticated snapshot binding."""
        ...

    def promote(
        self, scope: Scope, deployment: str, environment: str, request: PromoteRequest
    ) -> Environment:
        """Atomically updates one pointer after authorization and runtime admission."""
        ...


class DevPatchStore(Protocol):
    """Durable development CAS port; a production environment is not a dev session."""

    def patch(self, scope: Scope, session: str, request: PatchRequest) -> DevSession:
        """Rejects stale generation and leaves production immutable on every failure."""
        ...
