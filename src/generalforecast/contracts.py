# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable surface values and ports, not a second control plane."""

import datetime
import re
from dataclasses import MISSING, asdict, dataclass, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from typing import Any, Self
from urllib.parse import urlsplit

import pyarrow as pa

# `gf.lock`'s version, defined by `generalforecast.predictors.lock`.
from generalforecast.predictors.lock import LOCK_VERSION  # noqa: F401
from generalforecast.predictors.validation import (
    PredictorError,
    ensure,
    fields,
    hash_value,
    identifier,
    integer,
    name,
    text,
)

API_PREFIX = "/v1"
STATES = frozenset({"pending", "building", "ready", "failed", "stopped"})
TASKS = ("classification", "regression", "forecast")
CONTEXT_KINDS = ("snapshot", "request", "live")


@dataclass(frozen=True)
class PredictorLock:
    """A pushed `gf.lock` (version 2): the predictor's resolved intent, content-addressed.

    The hosted plane keeps the document as `gf check` wrote it. `digest` is
    the lock's one identity, the document's own `digest` key, which revision
    dedupe, gf-server's table fingerprint and `gf serve` all use;
    `identity` is the pinned model. `generalforecast.predictors.lock`
    defines the format and validates it.
    """

    document: dict

    def __post_init__(self):
        """Checks the document field by field, its digest included."""
        from generalforecast.predictors.lock import validate

        object.__setattr__(self, "document", json_copy(validate(self.document)))

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
        """The context rows' digest, when the context is a snapshot."""
        return self.context.get("digest") if self.context_kind == "snapshot" else None

    @property
    def resources(self) -> dict:
        """The hosted serving resources the manifest declares; empty when none."""
        return dict(self.document.get("resources") or {})

    @property
    def digest(self) -> str:
        """The revision's content address: the lock's own `digest`."""
        return self.document["digest"]

    def to_dict(self) -> dict:
        """A copy of the lock document."""
        return json_copy(self.document)

    @classmethod
    def from_dict(cls, value: Any) -> "PredictorLock":
        """Decodes a `gf.lock` document."""
        ensure(type(value) is dict, "lock must be a document")
        return cls(value)


def parse_lock(value: Any) -> "PredictorLock":
    """A pushed lock: the version 2 `gf.lock` of a catalog predictor.

    A version 1 lock is refused with the hint to run `gf check`; the hosted
    plane rewrites the version 1 locks it stored with
    `generalforecast.predictors.lock.upgrade`.
    """
    ensure(
        type(value) is dict and "predictor" in value,
        "unsupported lock; run `gf check` to write a version 2 lock",
        "UnsupportedLockVersion",
    )
    return PredictorLock.from_dict(value)


def lock_name(lock: "PredictorLock") -> str:
    """The predictor a lock was checked for."""
    return lock.name


def lock_digest(lock: "PredictorLock") -> str:
    """The lock's content address, `PredictorLock.digest`."""
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
    status: str = "pending"

    def __post_init__(self):
        """Keeps environment identity distinct from the immutable revision."""
        name(self.predictor)
        name(self.name)
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


# An RFC 3339 date-time; the offset is optional, and a time without one is UTC.
_RFC3339 = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(\.\d{1,9})?([Zz]|[+-]\d{2}:\d{2})?"
)
# The latest instant an `as_of` may name, in UTC seconds (year 33658).
_LATEST = 1e12


def parse_as_of(value: Any) -> float | None:
    """A request's `as_of` in UTC seconds: the one reading every surface applies.

    gf-server, `gf serve`, the SDK and a hosted plane accept the same
    values: an RFC 3339 date-time, `2026-10-27T08:00:00Z`, whose offset is
    optional (a time without one is UTC), or a number of seconds since the
    epoch. None is no `as_of`.

    Raises:
        ServiceError: A 422 `ValidationError` with `param="as_of"` for
            anything else: a date alone, `yesterday`, a boolean, a negative
            or non-finite number, or a time past year 33658.
    """
    from generalforecast.service.contracts import ServiceError

    def refused() -> ServiceError:
        return ServiceError(
            "ValidationError",
            "as_of must be an RFC 3339 time or epoch seconds",
            422,
            param="as_of",
            hint="pass 2026-10-27T08:00:00Z, or seconds since 1970 in UTC",
        )

    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        moment = value
    elif type(value) in (int, float):
        if not 0 <= value < _LATEST:
            raise refused()
        return float(value)
    elif type(value) is str and _RFC3339.fullmatch(value.strip()):
        text = value.strip().replace("z", "Z").replace("t", "T")
        # Python reads at most microseconds; RFC 3339 allows more digits.
        text = re.sub(r"(\.\d{6})\d+", r"\1", text)
        try:
            moment = datetime.datetime.fromisoformat(text)
        except ValueError:
            raise refused() from None
    else:
        raise refused()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.UTC)
    seconds = moment.timestamp()
    if not 0 <= seconds < _LATEST:
        raise refused()
    return seconds


def format_as_of(seconds: float | None) -> str | None:
    """An `as_of` as every reply carries it: RFC 3339 in UTC, `2026-10-27T08:00:00Z`.

    Whole seconds print without a fraction; anything finer prints to the
    microsecond. None stays None.
    """
    if seconds is None:
        return None
    moment = datetime.datetime.fromtimestamp(seconds, datetime.UTC)
    spec = "seconds" if moment.microsecond == 0 else "microseconds"
    return moment.replace(tzinfo=None).isoformat(timespec=spec) + "Z"


@dataclass(frozen=True)
class ContextVersion(WireValue):
    """The actual context observed by inference, not an accepted-write promise.

    Attributes:
        source: The context's kind: `snapshot`, `live` or `request`.
        version: A live context's sequence; None for the other kinds.
        rows: The context rows inference read.
        as_of: When the context was read, in UTC seconds.
        name: A live context's name; None for the other kinds.
        digest: A snapshot's rows digest; None for the other kinds.

    The wire form, every prediction reply's `context` on `gf serve` and
    hosted serving alike, is `{source, name, version, digest, rows,
    as_of}`, `as_of` an RFC 3339 UTC time (`format_as_of`) or null.
    gf-server refreshes a published revision whose context `source` is
    `live`.
    """

    source: str
    version: int | None = None
    rows: int | None = None
    as_of: float | None = None
    name: str | None = None
    digest: str | None = None

    def __post_init__(self):
        """Checks version metadata without fetching a live context."""
        ensure(self.source in CONTEXT_KINDS, "invalid context source")
        ensure(
            (self.name is not None) == (self.source == "live"),
            "a live context, and only one, is named",
        )
        if self.name is not None:
            name(self.name)
        if self.version is not None:
            ensure(
                self.source == "live"
                and type(self.version) is int
                and self.version >= 0,
                "a live context's version is its sequence",
            )
        if self.digest is not None:
            ensure(self.source == "snapshot", "only a snapshot context has a digest")
            hash_value(self.digest)
        if self.rows is not None:
            integer(self.rows, 0)
        if self.as_of is not None:
            ensure(
                type(self.as_of) in (int, float) and 0 <= self.as_of < _LATEST,
                "invalid context time",
            )

    def to_dict(self) -> dict:
        """The wire form: every key, `as_of` in RFC 3339."""
        return {
            "source": self.source,
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "rows": self.rows,
            "as_of": format_as_of(self.as_of),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ContextVersion":
        """Reads the wire form; `as_of` may be RFC 3339 or epoch seconds."""
        body = dict(
            fields(value, {"source"}, {"name", "version", "digest", "rows", "as_of"})
        )
        body["as_of"] = parse_as_of(body.get("as_of"))
        return cls(**body)


def context_version(value: dict) -> ContextVersion:
    """Decodes a reply's or a stored record's context, in its current or an earlier form.

    The current form is `ContextVersion.to_dict`'s. Earlier ones still read,
    so stored records keep decoding: gf-server's `{kind, rows, digest,
    as_of}`, a `version` holding a snapshot's digest, and a source spelled
    `live/<name>`.
    """
    ensure(type(value) is dict, "a context is an object")
    if "kind" in value and "source" not in value:
        body = dict(fields(value, {"kind"}, {"rows", "digest", "as_of", "name"}))
        value = {"source": body.pop("kind"), **body}
    value = dict(value)
    source = value.get("source")
    if type(source) is str and source.startswith("live/"):
        value["source"], value["name"] = "live", source.removeprefix("live/")
    if type(value.get("version")) is str and "digest" not in value:
        value["digest"] = value.pop("version")
    return ContextVersion.from_dict(value)


@dataclass(frozen=True)
class PredictionRequest(WireValue):
    """A declared predictor's prediction request: `{queries, context?, as_of?}`.

    Attributes:
        queries: The rows to score.
        context: A request context's rows; None for a snapshot or live context.
        as_of: For a live context, the time to read it as of: an RFC 3339
            time or epoch seconds (`parse_as_of`). Every surface honours it
            or refuses it.
    """

    queries: tuple[dict, ...]
    context: tuple[dict, ...] | None = None
    as_of: str | float | None = None

    def __post_init__(self):
        """Checks the rows and the time."""
        ensure(
            type(self.queries) is tuple
            and 0 < len(self.queries) <= 10_000
            and all(type(row) is dict for row in self.queries),
            "queries must contain 1..10000 rows",
        )
        if self.context is not None:
            ensure(
                type(self.context) is tuple
                and all(type(row) is dict for row in self.context),
                "context must be a list of rows",
            )
        parse_as_of(self.as_of)

    @classmethod
    def from_dict(cls, value: dict) -> "PredictionRequest":
        """Decodes the request body's payload."""
        body = fields(value, {"queries"}, {"context", "as_of"})
        ensure(type(body["queries"]) is list, "queries must be a list of rows")
        context = body.get("context")
        ensure(
            context is None or type(context) is list, "context must be a list of rows"
        )
        return cls(
            tuple(body["queries"]),
            None if context is None else tuple(context),
            body.get("as_of"),
        )

    def to_dict(self) -> dict:
        """The request body's payload, keys present only when set."""
        payload: dict = {"queries": [dict(row) for row in self.queries]}
        if self.context is not None:
            payload["context"] = [dict(row) for row in self.context]
        if self.as_of is not None:
            payload["as_of"] = self.as_of
        return payload


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
            identifier(self.credential_ref.removeprefix("secret/"))
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
        identifier(self.runtime)
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
            identifier(ref.removeprefix("secret/"))
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
