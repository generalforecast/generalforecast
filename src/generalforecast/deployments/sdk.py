"""Compatibility aliases to the canonical versioned SDK, with no alpha transport."""

import datetime
import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

from generalforecast.contracts import (
    DevSession,
    Environment,
    PredictionResult,
    Revision,
)
from generalforecast.deployments.boundary import (
    JsonObject,
    Request,
    Resource,
    Scope,
    json_object,
)
from generalforecast.deployments.contracts import VERSION, ServiceError, name, require
from generalforecast.packages import Package, PackageLock, PythonLock
from generalforecast.packages.models import describe
from generalforecast.packages.validation import bytes_digest
from generalforecast.python_packages.compiler import MAX_ARCHIVE
from generalforecast.tabular import file_format, read_table, to_arrow

__all__ = ["AsyncClient", "Client", "SuiteClient", "endpoint"]  # noqa: F822

PYTHON_SOURCE_MEDIA_TYPE = "application/vnd.generalforecast.python-source+tar"


@dataclass(frozen=True, eq=False)
class Snapshot(Mapping[str, Any]):
    """Immutable metadata snapshot with fresh portable dictionaries on demand."""

    _encoded: str = field(repr=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        """Copies input so callers cannot mutate nested snapshot metadata."""
        return cls(
            json.dumps(json_object(dict(value)), ensure_ascii=False, allow_nan=False)
        )

    def to_dict(self) -> JsonObject:
        """Returns a detached wire value, including nested arrays and objects."""
        return json.loads(self._encoded)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class RevisionDetails:
    """An immutable revision and its exact content-addressed compiled lock."""

    revision: Revision
    lock: PackageLock | PythonLock


@dataclass(frozen=True, eq=False)
class WriteReceipt(Snapshot):
    """Durable dataset acceptance, distinct from serving visibility."""

    @property
    def sequence(self) -> int:
        """Returns the accepted logical transaction sequence."""
        return self["sequence"]

    @property
    def write_id(self) -> str:
        """Returns the replay identity to preserve on uncertain outcomes."""
        return self["write_id"]


@dataclass(frozen=True, eq=False)
class ImportReceipt(Snapshot):
    """Imported content identity and actual logical transaction metadata."""

    @property
    def digest(self) -> str:
        """Returns the independently verified normalized content digest."""
        return self["digest"]

    @property
    def sequence(self) -> int:
        """Returns the actual accepted logical transaction sequence."""
        return self["sequence"]

    @property
    def rows(self) -> int:
        """Returns the verified imported row count."""
        return self["rows"]


@dataclass(frozen=True, eq=False)
class BoundSnapshot(Snapshot):
    """A detached state value with methods explicitly returning fresh values."""

    _client: Any = field(repr=False, compare=False)
    name: str

    @classmethod
    def from_response(cls, client, resource: str, value: Mapping[str, Any]):
        """Copies wire metadata and binds only the authenticated parent client."""
        return cls(Snapshot.from_dict(value)._encoded, client, name(resource))


@dataclass(frozen=True, eq=False)
class Deployment(BoundSnapshot):
    """Immutable deployment metadata with explicit environment operations."""

    environment: Environment | None = None

    @classmethod
    def from_response(
        cls,
        client,
        resource: str,
        value: Mapping[str, Any],
        environment: Environment | None = None,
    ):
        """Captures state without mutating older deployment handles."""
        return cls(
            Snapshot.from_dict(value)._encoded, client, name(resource), environment
        )

    @property
    def revision(self) -> int | None:
        """Returns the selected environment revision, never desired-as-serving state."""
        return self.environment.revision if self.environment else self.get("revision")

    @property
    def status(self) -> str | None:
        """Returns reported lifecycle status, preserving pending/error states."""
        return self.environment.status if self.environment else self.get("status")

    def refresh(self) -> "Deployment":
        """Fetches a new snapshot without modifying this handle."""
        return self._client.deployments.get(
            self.name, env=self.environment.name if self.environment else None
        )

    def predict(self, queries: Any, **options: Any) -> PredictionResult:
        """Uses the captured environment selection, not a mutable global default."""
        captured_environment(self.environment, options)
        return self._client.deployments.predict(self.name, queries, **options)

    def wait(self, **options: Any) -> Environment:
        """Waits for the captured pointer by default."""
        captured_environment(self.environment, options, pointer=True)
        return self._client.deployments.wait(self.name, **options)

    def promote(self, *, revision: int, **options: Any) -> Environment:
        """Promotes under this snapshot's generation when an environment was selected."""
        self._guard(options)
        return self._client.deployments.promote(self.name, revision=revision, **options)

    def rollback(self, *, revision: int, **options: Any) -> Environment:
        """Rolls back under this snapshot's generation."""
        self._guard(options)
        return self._client.deployments.rollback(
            self.name, revision=revision, **options
        )

    def _guard(self, options):
        captured_environment(self.environment, options, guard=True)

    def metrics(self, **options: Any) -> Snapshot:
        """Fetches actual metrics for the selected environment."""
        captured_environment(self.environment, options)
        return self._client.deployments.metrics(self.name, **options)

    def logs(self, **options: Any) -> Snapshot:
        """Fetches bounded runtime logs for the selected environment."""
        captured_environment(self.environment, options)
        return self._client.deployments.logs(self.name, **options)

    def delete(self) -> Snapshot:
        """Requests deletion; the server owns authorization and cleanup."""
        return self._client.deployments.delete(self.name)


@dataclass(frozen=True, eq=False)
class Dataset(BoundSnapshot):
    """Logical dataset snapshot and explicit append/label/import/refresh actions."""

    @property
    def sequence(self) -> int | None:
        """Returns the actual accepted transaction sequence."""
        return self.get("sequence")

    def append(self, data: Any, *, id: str, idempotency_key: str) -> WriteReceipt:
        """Appends tabular observations without mutating this state snapshot."""
        return self._client.datasets.append(
            self.name, data, id=id, idempotency_key=idempotency_key
        )

    def label(
        self, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches labels using an explicit stable write identity."""
        return self._client.datasets.label(
            self.name, data, id=id, column=column, idempotency_key=idempotency_key
        )

    def import_source(self, reference) -> ImportReceipt:
        """Binds an import only when its logical dataset matches this handle."""
        require(reference.dataset == self.name, "import dataset differs from handle")
        return self._client.datasets.import_source(reference)

    def refresh(self) -> "Dataset":
        """Explicitly requests provider refresh rather than polling source silently."""
        return self._client.datasets.refresh(self.name)

    def status(self) -> "Dataset":
        """Fetches metadata without triggering provider ingestion."""
        return self._client.datasets.get(self.name)


@dataclass(frozen=True, eq=False)
class Job(BoundSnapshot):
    """Durable job snapshot; acceptance never implies completion."""

    @property
    def state(self) -> str:
        """Returns the actual persisted lifecycle state."""
        return self["state"]

    def poll(self) -> "Job":
        """Fetches a fresh job state without modifying this object."""
        return self._client.jobs.get(self.name)

    def cancel(self) -> "Job":
        """Cancels remotely; cancelling a local wait does not cancel the job."""
        return Job.from_response(
            self._client, self.name, self._client.jobs.control(self.name, "cancel")
        )

    def wait(self, *, timeout: float = 120, poll_interval: float = 0.2) -> "Job":
        """Polls with a bounded total deadline and raises for terminal failure."""
        return self._client.jobs.wait(
            self.name, timeout=timeout, poll_interval=poll_interval
        )

    def results(self, *, offset: int = 0, limit: int = 100) -> Snapshot:
        """Gets one bounded page, retaining every skipped/rejected occurrence."""
        return Snapshot.from_dict(
            self._client.jobs.results(self.name, offset=offset, limit=limit)
        )


@dataclass(frozen=True)
class Development:
    """Remote development state, isolated from immutable revisions and production."""

    state: DevSession
    _client: Any = field(repr=False, compare=False)

    @property
    def id(self) -> str:
        """Returns the server-created remote session identifier."""
        return self.state.id

    @property
    def generation(self) -> int:
        """Returns the generation required for the next optimistic patch."""
        return self.state.generation

    def refresh(self) -> "Development":
        """Fetches a new session snapshot."""
        return self._client.dev_sessions.get(self.state.deployment, self.id)

    def patch(self, package: Package) -> "Development":
        """Uploads a full replacement checked package using the captured generation."""
        return self._client.dev_sessions.patch(self, package)

    def predict(self, queries: Any, **options: Any) -> PredictionResult:
        """Scores only this development worker, never its production environment."""
        return self._client.dev_sessions.predict(
            self.state.deployment, self.id, queries, **options
        )

    def stop(self) -> "Development":
        """Stops this development worker without changing a production pointer."""
        return self._client.dev_sessions.stop(self.state.deployment, self.id)


@dataclass(frozen=True, eq=False)
class Runtime(BoundSnapshot):
    """Runtime metadata; configuration or a plan is not proof of provisioning."""

    @property
    def status(self) -> str | None:
        """Returns provider-reported state, not client-generated readiness."""
        return self.get("status")


@dataclass(frozen=True)
class DatasetSpec:
    """Typed existing logical dataset schema and explicit admission bounds."""

    name: str
    features: Sequence[str]
    targets: Sequence[Sequence[str]]
    max_rows: int = 1024
    max_writes: int = 10000
    max_pending_labels: int = 1024
    pending_ttl_seconds: float = 3600

    def __post_init__(self) -> None:
        """Copies sequence inputs so caller-owned lists cannot alter the schema."""
        name(self.name)
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "targets", tuple(tuple(pair) for pair in self.targets))

    def to_dict(self) -> JsonObject:
        """Uses the existing mutation service schema, never a parallel dataset API."""
        return {
            "api_version": VERSION,
            "kind": "dataset",
            "name": self.name,
            "features": list(self.features),
            "targets": {column: dtype for column, dtype in self.targets},
            "max_rows": self.max_rows,
            "max_writes": self.max_writes,
            "max_pending_labels": self.max_pending_labels,
            "pending_ttl_seconds": self.pending_ttl_seconds,
        }


@dataclass(frozen=True)
class JobSpec:
    """Typed durable task with a stable caller-owned submission identity."""

    name: str
    task: str
    submission_id: str
    parameters: Snapshot

    def to_dict(self) -> JsonObject:
        """Builds the canonical durable jobs payload."""
        return {
            "api_version": VERSION,
            "kind": "job",
            "name": self.name,
            "task": self.task,
            "submission_id": self.submission_id,
            "parameters": self.parameters.to_dict(),
        }


def envelope(
    project: str,
    request_id: str,
    kind,
    resource: str,
    payload: JsonObject | None = None,
) -> Request:
    """Constructs the shared bounded request, without authorizing scope locally."""
    return Request.parse(
        Request(
            Scope(project, request_id),
            Resource(kind, resource),
            {} if payload is None else payload,
        ).to_dict()
    )


def unwrap(value: JsonObject, request: Request) -> JsonObject:
    """Rejects cross-project, cross-resource and stale request responses."""
    metadata = request.to_dict()
    del metadata["payload"]
    if (
        set(value) != {*metadata, "result"}
        or any(value.get(k) != v for k, v in metadata.items())
        or type(value.get("result")) is not dict
    ):
        raise ServiceError(
            "ProtocolError", "response envelope does not match request", 502
        )
    return value["result"]


def table(data: Any, *, columns: Sequence[str] | None = None) -> pa.Table:
    """Loads bounded local files or normalized dataframes; arrays need feature names."""
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                require(
                    stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
                    "input must be a regular file",
                )
                raw = stream.read(134217729)
            require(len(raw) <= 134217728, "input file too large")
            return read_table(raw, file_format(str(path)))
        except OSError:
            raise ServiceError(
                "ValidationError", "local table unavailable", 422
            ) from None
    if type(data).__module__.split(".")[0] == "numpy":
        require(
            columns is not None, "NumPy inputs require explicit feature column order"
        )
    return to_arrow(data, columns=None if columns is None else list(columns))


def records(data: Any, *, columns: Sequence[str] | None = None) -> list[JsonObject]:
    """Converts normalized flat tables to bounded portable prediction records."""

    def cell(value):
        if isinstance(value, (datetime.datetime, datetime.date)):
            return value.isoformat()
        require(not isinstance(value, bytes), "binary inputs require an Arrow endpoint")
        return value

    return [
        {key: cell(value) for key, value in row.items()}
        for row in table(data, columns=columns).to_pylist()
    ]


def prediction_payload(
    queries: Any,
    *,
    context: Any = None,
    context_columns: Sequence[str] | None = None,
    target: str | None = None,
    features: Sequence[str] | None = None,
    options: JsonObject | None = None,
) -> JsonObject:
    """Keeps queries and context separate; never writes prediction inputs.

    An empty context list or tuple is no context rows, as a snapshot package's
    request context is, since a table of no columns has none to normalize.
    """
    if features is not None:
        require(
            not isinstance(features, str)
            and all(type(v) is str for v in features)
            and len(set(features)) == len(features),
            "invalid feature order",
        )
    payload: JsonObject = {"queries": records(queries, columns=features)}
    if context is not None:
        payload["context"] = (
            []
            if isinstance(context, (list, tuple)) and not context
            else records(context, columns=context_columns)
        )
    if target is not None:
        payload["target"] = target
    if features is not None:
        payload["features"] = list(features)
    if options is not None:
        payload["options"] = options
    return json_object(payload)


def one_shot_payload(
    queries: Any,
    *,
    model: str,
    target: str,
    context: Any,
    context_columns: Sequence[str] | None = None,
    task: str | None = None,
    features: Sequence[str] | None = None,
    options: JsonObject | None = None,
    outputs: Sequence[str] | None = None,
) -> JsonObject:
    """Uses catalog task metadata and declared roles, never target-value heuristics."""
    semantic = json_object({} if options is None else options)
    if outputs is not None:
        require(not isinstance(outputs, str), "outputs must be a sequence of names")
        require(
            "outputs" not in semantic or semantic["outputs"] == list(outputs),
            "conflicting output selections",
        )
        semantic["outputs"] = list(outputs)
    if task is None:
        tasks = describe(model).tasks
        require(len(tasks) == 1, "provide an explicit model task")
        task = tasks[0]
    query_table = table(queries, columns=features)
    if features is None:
        require(
            all(
                semantic.get(role) is None or type(semantic[role]) is str
                for role in ("id", "time")
            ),
            "invalid role column",
        )
        reserved = {target, semantic.get("id"), semantic.get("time")}
        for role in ("series", "past_inputs", "future_inputs"):
            columns = semantic.get(role, [])
            require(type(columns) is list, "role columns must be a list")
            require(
                all(type(column) is str for column in columns), "invalid role column"
            )
            reserved.update(columns)
        features = [
            column for column in query_table.column_names if column not in reserved
        ]
    payload = prediction_payload(
        query_table,
        context=context,
        context_columns=context_columns,
        target=target,
        features=features,
        options=semantic,
    )
    payload.update(model=model, task=task)
    return json_object(payload)


def captured_environment(
    environment: Environment | None,
    options: dict[str, Any],
    *,
    pointer: bool = False,
    guard: bool = False,
) -> None:
    """Never applies one environment's captured pointer or CAS to another."""
    if environment is None:
        return
    if options.get("env") is None:
        options["env"] = environment.name
    if options["env"] == environment.name:
        if pointer:
            if options.get("revision") is None:
                options["revision"] = environment.revision
            if options.get("generation") is None:
                options["generation"] = environment.generation
        if guard and options.get("expected_generation") is None:
            options["expected_generation"] = environment.generation


def environment_name(value: str | None, default: str) -> str:
    """Uses the configured environment only when no selection was supplied."""
    return name(default if value is None else value)


def checked_lock(package: Package) -> PackageLock | PythonLock:
    """Requires an explicit checked package; frozen verification never upgrades pins."""
    require(isinstance(package, Package), "push requires a checked Package")
    return package.compile(frozen=True)


def checked_python_source(package: Package, lock: PythonLock) -> bytes:
    """Verifies complete source and probe data without importing customer code."""
    archive = package.source_archive()
    lock.verify_source(archive)
    return archive


def python_source_upload(
    project: str, deployment: str, archive: bytes, source: str
) -> JsonObject:
    """Binds bounded source bytes to their exact project/deployment receipt."""
    name(project)
    name(deployment)
    require(
        type(archive) is bytes and 0 < len(archive) <= MAX_ARCHIVE,
        "invalid Python source archive",
    )
    require(
        type(source) is str and bytes_digest(archive) == source,
        "Python source digest mismatch",
    )
    return {
        "receipt_version": 1,
        "kind": "python-source",
        "project": project,
        "deployment": deployment,
        "source": source,
        "size": len(archive),
    }


def python_source_receipt(receipt: JsonObject, expected: JsonObject) -> Snapshot:
    """Rejects forged, foreign, partial or extended upload admission evidence."""
    if (
        type(receipt.get("receipt_version")) is not int
        or type(receipt.get("size")) is not int
        or receipt != expected
    ):
        raise ServiceError(
            "ProtocolError",
            "Python source receipt does not match uploaded content",
            502,
        )
    return Snapshot.from_dict(receipt)


def mutation(
    data: Any, *, id: str, operation: str, write_id: str, column: str | None = None
) -> JsonObject:
    """Adapts tabular append/labels to existing atomic component transactions."""
    rows = records(data)
    result = []
    for row in rows:
        require(
            id in row and type(row[id]) is str and bool(row[id]),
            "dataset writes require explicit nonempty string IDs",
        )
        if column is None:
            result.append(
                {
                    "key": row[id],
                    "features": {key: value for key, value in row.items() if key != id},
                }
            )
        else:
            require(column in row, "label column missing")
            result.append({"key": row[id], "labels": {column: row[column]}})
    return json_object(
        {"write_id": write_id, "operation": operation, "records": result}
    )


def inference_result(value: JsonObject) -> PredictionResult:
    """Decodes the foundation result, not legacy fixture inference outputs."""
    try:
        return PredictionResult.from_dict(value)
    except (KeyError, TypeError, ValueError):
        raise ServiceError("ProtocolError", "invalid prediction result", 502) from None


def __getattr__(name: str) -> Any:
    """Loads canonical clients lazily to avoid package initialization cycles."""
    if name in ("Client", "SuiteClient"):
        from generalforecast.client import Client

        return Client
    if name == "AsyncClient":
        from generalforecast.async_client import AsyncClient

        return AsyncClient
    if name == "endpoint":
        from generalforecast.transport import endpoint

        return endpoint
    raise AttributeError(name)
