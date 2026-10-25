# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable surface values and ports, not a second control plane."""

import datetime
import re
from dataclasses import MISSING, asdict, dataclass, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from typing import Any, Protocol, Self
from urllib.parse import urlsplit

import pyarrow as pa

from generalforecast.predictors.validation import (
    PredictorError,
    digest,
    ensure,
    fields,
    hash_value,
    integer,
    name,
    text,
)
from generalforecast.service.boundary import Scope

API_PREFIX = "/v1"
STATES = frozenset({"pending", "building", "ready", "failed", "stopped"})
TASKS = ("classification", "regression", "forecast")
CONTEXT_KINDS = ("snapshot", "request", "live")
MODEL_ID = re.compile(r"general-forecast/[a-z0-9][a-z0-9.-]*(@[^@\s]+)?")
LOCK_VERSION = 1


@dataclass(frozen=True)
class PredictorLock:
    """A pushed `gf.lock`: the predictor's resolved intent, content-addressed.

    The hosted plane keeps the document byte-for-byte as `gf check` wrote it;
    `digest` names the revision and `identity` the pinned model.
    """

    document: dict

    def __post_init__(self):
        """Checks the lock's shape, the canonical model id and its context kind."""
        value = self.document
        fields(value, {"lock_version", "predictor", "model"})
        ensure(
            value["lock_version"] == LOCK_VERSION,
            "unsupported lock version; run `gf check` to write version 2",
            "UnsupportedLockVersion",
        )
        predictor = fields(
            value["predictor"],
            {
                "name",
                "model",
                "model_digest",
                "task",
                "target",
                "target_type",
                "inputs",
                "context",
                "selection",
            },
            {
                "classes",
                "id",
                "id_type",
                "time",
                "series",
                "frequency",
                "horizon",
                "quantiles",
            },
        )
        name(predictor["name"])
        ensure(
            type(predictor["model"]) is str
            and MODEL_ID.fullmatch(predictor["model"]) is not None,
            f"model `{predictor['model']}` is not a catalog id; use general-forecast/<name>",
            "ValidationError",
        )
        hash_value(predictor["model_digest"])
        ensure(predictor["task"] in TASKS, "invalid predictor task")
        text(predictor["target"])
        inputs = predictor["inputs"]
        ensure(
            type(inputs) is list
            and all(
                type(column) is dict
                and {"name", "type"} <= set(column)
                and type(column["name"]) is str
                and (column["type"] is None or type(column["type"]) is str)
                for column in inputs
            ),
            "lock inputs must list {name, type} columns",
        )
        context = fields(
            predictor["context"],
            {"kind"},
            {"digest", "rows", "query_rows", "path", "name", "max_rows"},
        )
        ensure(context["kind"] in CONTEXT_KINDS, "invalid context kind")
        if context["kind"] == "snapshot":
            hash_value(context.get("digest"))
            integer(context.get("rows"), 0)
        if context["kind"] == "live":
            name(context.get("name"))
        fields(value["model"], {"files"})
        object.__setattr__(self, "document", json_copy(value))

    @property
    def predictor(self) -> dict:
        """The resolved predictor section."""
        return self.document["predictor"]

    @property
    def name(self) -> str:
        """The predictor name the lock was checked for."""
        return self.predictor["name"]

    @property
    def model(self) -> str:
        """The canonical catalog id."""
        return self.predictor["model"]

    @property
    def identity(self) -> str:
        """The pinned model, `general-forecast/<name>@sha256:...`."""
        return f"{self.model}@{self.predictor['model_digest']}"

    @property
    def task(self) -> str:
        """classification, regression or forecast."""
        return self.predictor["task"]

    @property
    def context(self) -> dict:
        """The locked context: kind, and digest/rows or the live context name."""
        return self.predictor["context"]

    @property
    def context_kind(self) -> str:
        """snapshot, request or live."""
        return self.context["kind"]

    @property
    def context_name(self) -> str | None:
        """The live context this predictor binds to, when its kind is live."""
        return self.context.get("name") if self.context_kind == "live" else None

    @property
    def data_digest(self) -> str | None:
        """The snapshot's content digest, when the context is a snapshot."""
        return self.context.get("digest") if self.context_kind == "snapshot" else None

    @property
    def digest(self) -> str:
        """The revision's content address: the digest of the canonical lock."""
        return digest(self.document)

    def to_dict(self) -> dict:
        """A copy of the lock document."""
        return json_copy(self.document)

    @classmethod
    def from_dict(cls, value: Any) -> "PredictorLock":
        """Decodes a `gf.lock` document."""
        ensure(type(value) is dict, "lock must be a document")
        return cls(value)


def parse_lock(value: Any) -> "PredictorLock":
    """A pushed lock: the `gf.lock` of a catalog predictor."""
    ensure(
        type(value) is dict and "predictor" in value,
        "unsupported lock; run `gf check` to write a version 1 lock",
        "UnsupportedLockVersion",
    )
    return PredictorLock.from_dict(value)


def lock_name(lock: "PredictorLock") -> str:
    """The predictor a lock was checked for."""
    return lock.name


def lock_digest(lock: "PredictorLock") -> str:
    """The lock's content address."""
    return lock.digest


def json_copy(value: Any) -> Any:
    """A deep copy of plain JSON data."""
    if isinstance(value, dict):
        return {key: json_copy(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_copy(child) for child in value]
    return value


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
    """A predictor's immutable pushed identity and lifecycle state."""

    predictor: str
    number: int
    predictor_digest: str
    model: str
    created_at: float
    status: str = "pending"

    def __post_init__(self):
        """Checks portable identity without claiming the revision is serving."""
        name(self.predictor)
        integer(self.number)
        hash_value(self.predictor_digest)
        text(self.model)
        ensure(
            type(self.created_at) in (int, float) and 0 <= self.created_at < 1e12,
            "invalid revision time",
        )
        ensure(self.status in STATES, "invalid revision status")


@dataclass(frozen=True)
class Environment(WireValue):
    """Mutable environment pointer; generation supports compare-and-swap updates."""

    predictor: str
    name: str
    revision: int | None
    generation: int
    runtime: str
    status: str = "pending"

    def __post_init__(self):
        """Keeps environment identity distinct from the immutable revision."""
        name(self.predictor)
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
        """Checks version metadata without fetching a live context."""
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


def context_version(value: dict) -> ContextVersion:
    """Decodes a context as the SDK or the predictor API describes it.

    The predictor API sends `{kind, rows, digest?, as_of?}`; the kind is the
    source and the digest is the version.
    """
    if type(value) is dict and "kind" in value and "source" not in value:
        body = dict(fields(value, {"kind"}, {"rows", "digest", "as_of"}))
        as_of = body.get("as_of")
        if isinstance(as_of, str):
            as_of = datetime.datetime.fromisoformat(as_of).timestamp()
        return ContextVersion(
            source=body["kind"],
            version=body.get("digest"),
            rows=body.get("rows"),
            as_of=as_of,
        )
    return ContextVersion.from_dict(value)


@dataclass(frozen=True)
class ContextImportRef(WireValue):
    """An import binds a trusted source to a live context; credentials are refs only."""

    context: str
    source: str
    format: str
    credential_ref: str | None = None
    digest: str | None = None
    mapping: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        """Rejects signed URLs, userinfo and inline credentials before transmission."""
        name(self.context)
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
                "context",
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
    """Placement and resources for an exact revision, separate from provider provisioning."""

    runtime: str
    environment: str
    predictor_digest: str
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
        hash_value(self.predictor_digest)
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
    def for_lock(
        cls,
        lock: "PredictorLock",
        *,
        runtime: str,
        environment: str,
        secret_refs: tuple[str, ...] = (),
    ) -> "RuntimePlan":
        """Compiles portable placement without creating infrastructure or identities."""
        lock = parse_lock(lock.to_dict())
        return cls(
            runtime,
            environment,
            lock.digest,
            lock.predictor["model_digest"],
            lock.data_digest,
            None,
            (),
            1,
            "2Gi",
            1,
            1,
            secret_refs,
        )


@dataclass(frozen=True)
class PredictionResult:
    """Typed SDK result retaining model and actual context provenance."""

    table: pa.Table
    model: str
    task: str
    context: ContextVersion | None = None
    predictor: str | None = None
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
        if self.predictor is not None:
            name(self.predictor)
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
            "predictor": self.predictor,
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
                {"context", "predictor", "revision", "environment"},
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
            raise PredictorError(
                "InvalidResult", "prediction rows are invalid"
            ) from None
        if body.get("context") is not None:
            body["context"] = context_version(body["context"])
        return cls(**body)


class RevisionStore(Protocol):
    """Durable scoped revision and environment port implemented by the API owner."""

    def create(
        self,
        scope: Scope,
        predictor: str,
        lock: "PredictorLock",
        *,
        snapshot: str | None = None,
    ) -> Revision:
        """Persists immutable revision identity and authenticated snapshot binding."""
        ...

    def promote(
        self, scope: Scope, predictor: str, environment: str, request: PromoteRequest
    ) -> Environment:
        """Atomically updates one pointer after authorization and runtime admission."""
        ...
