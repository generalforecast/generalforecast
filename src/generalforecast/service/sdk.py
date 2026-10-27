# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Typed values and helpers the hosted SDK clients share."""

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
    Environment,
    PredictionResult,
    PredictorLock,
    Revision,
)
from generalforecast.predictors.models import describe
from generalforecast.predictors.validation import PredictorError
from generalforecast.service.boundary import (
    JsonObject,
    Request,
    Resource,
    Scope,
    json_object,
)
from generalforecast.service.contracts import ServiceError, name, require
from generalforecast.tabular import file_format, read_table, to_arrow

__all__ = ["AsyncClient", "Client", "SuiteClient", "endpoint"]  # noqa: F822


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
    """An immutable revision and the exact lock it was pushed from."""

    revision: Revision
    lock: PredictorLock


@dataclass(frozen=True, eq=False)
class WriteReceipt(Snapshot):
    """Durable live-context acceptance, distinct from serving visibility."""

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
class PredictorHandle(BoundSnapshot):
    """A hosted predictor's facts, and an environment if one was selected.

    `HostedPredictor` and `AsyncHostedPredictor` add the same operations,
    run synchronously or awaited.
    """

    environment: Environment | None = None

    @classmethod
    def from_response(
        cls,
        client,
        resource: str,
        value: Mapping[str, Any],
        environment: Environment | None = None,
    ):
        """Captures state without mutating older predictor handles."""
        return cls(
            Snapshot.from_dict(value)._encoded, client, name(resource), environment
        )

    @property
    def revision(self) -> int | None:
        """The selected environment's revision, else the latest registered one."""
        if self.environment is not None:
            return self.environment.revision
        return self.get("latest_revision")

    @property
    def status(self) -> str | None:
        """The selected environment's reported status; None without one."""
        return self.environment.status if self.environment is not None else None

    @property
    def environments(self) -> tuple[Environment, ...]:
        """Every environment the predictor has, as the service reported them."""
        return tuple(
            Environment.from_dict(value) for value in self.get("environments") or ()
        )

    def _selected(self, options: dict[str, Any], **captured: bool) -> dict[str, Any]:
        """Options with this handle's environment, pointer or CAS guard filled in."""
        captured_environment(self.environment, options, **captured)
        return options

    def _refresh_env(self) -> str | None:
        return self.environment.name if self.environment is not None else None


@dataclass(frozen=True, eq=False)
class HostedPredictor(PredictorHandle):
    """A hosted predictor's facts with explicit environment operations."""

    def refresh(self) -> "HostedPredictor":
        """Fetches a new snapshot without modifying this handle."""
        return self._client.predictors.get(self.name, env=self._refresh_env())

    def predict(self, queries: Any, **options: Any) -> PredictionResult:
        """Uses the captured environment selection, not a mutable global default."""
        return self._client.predictors.predict(
            self.name, queries, **self._selected(options)
        )

    def wait(self, **options: Any) -> Environment:
        """Waits for the captured pointer by default."""
        return self._client.predictors.wait(
            self.name, **self._selected(options, pointer=True)
        )

    def promote(self, *, revision: int, **options: Any) -> Environment:
        """Promotes under this snapshot's generation when an environment was selected."""
        return self._client.predictors.promote(
            self.name, revision=revision, **self._selected(options, guard=True)
        )

    def rollback(self, *, revision: int, **options: Any) -> Environment:
        """Rolls back under this snapshot's generation."""
        return self._client.predictors.rollback(
            self.name, revision=revision, **self._selected(options, guard=True)
        )

    def metrics(self, **options: Any) -> Snapshot:
        """Fetches actual metrics for the selected environment."""
        return self._client.predictors.metrics(self.name, **self._selected(options))

    def logs(self, **options: Any) -> Snapshot:
        """Fetches bounded runtime logs for the selected environment."""
        return self._client.predictors.logs(self.name, **self._selected(options))

    def delete(self) -> Snapshot:
        """Requests deletion; the server owns authorization and cleanup."""
        return self._client.predictors.delete(self.name)


@dataclass(frozen=True, eq=False)
class ContextHandle(BoundSnapshot):
    """A live context's facts; `LiveContext` and `AsyncLiveContext` add its writes."""

    @property
    def sequence(self) -> int | None:
        """Returns the actual accepted transaction sequence."""
        return self.get("sequence")

    @property
    def rows(self) -> int | None:
        """Rows the context holds."""
        return self.get("rows")

    def _owned(self, reference) -> Any:
        """The import reference, once it names this handle's context."""
        require(reference.context == self.name, "import context differs from handle")
        return reference


@dataclass(frozen=True, eq=False)
class LiveContext(ContextHandle):
    """A live context's facts and explicit append/label/import/refresh actions."""

    def append(self, data: Any, *, id: str, idempotency_key: str) -> WriteReceipt:
        """Appends rows without mutating this state snapshot."""
        return self._client.contexts.append(
            self.name, data, id=id, idempotency_key=idempotency_key
        )

    def label(
        self, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> WriteReceipt:
        """Attaches target values using an explicit stable write identity."""
        return self._client.contexts.label(
            self.name, data, id=id, column=column, idempotency_key=idempotency_key
        )

    def import_source(self, reference) -> ImportReceipt:
        """Binds an import only when its live context matches this handle."""
        return self._client.contexts.import_source(self._owned(reference))

    def refresh(self) -> ImportReceipt:
        """Explicitly requests provider refresh rather than polling source silently."""
        return self._client.contexts.refresh(self.name)

    def status(self) -> "LiveContext":
        """Fetches metadata without triggering provider ingestion."""
        return self._client.contexts.get(self.name)


@dataclass(frozen=True)
class ContextSpec:
    """A live context's schema: its inputs, its targets and their types, its bounds."""

    name: str
    inputs: Sequence[str]
    targets: Mapping[str, str] | Sequence[Sequence[str]]
    max_rows: int = 1024
    id: str | None = None
    max_writes: int = 10000
    max_pending_labels: int = 1024
    pending_ttl_seconds: float = 3600

    def __post_init__(self) -> None:
        """Copies sequence inputs so caller-owned lists cannot alter the schema."""
        name(self.name)
        object.__setattr__(self, "inputs", tuple(self.inputs))
        targets = (
            tuple(self.targets.items())
            if isinstance(self.targets, Mapping)
            else tuple(tuple(pair) for pair in self.targets)
        )
        object.__setattr__(self, "targets", targets)

    def to_dict(self) -> JsonObject:
        """The `POST /contexts` payload."""
        value: dict[str, Any] = {
            "name": self.name,
            "inputs": list(self.inputs),
            "targets": {column: dtype for column, dtype in self.targets},
            "max_rows": self.max_rows,
            "max_writes": self.max_writes,
            "max_pending_labels": self.max_pending_labels,
            "pending_ttl_seconds": self.pending_ttl_seconds,
        }
        if self.id is not None:
            value["id"] = self.id
        return value


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

    An empty context list or tuple is no context rows, as a snapshot predictor's
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


@dataclass(frozen=True)
class Pushed:
    """What `push` sends for a checked predictor: its lock and its context rows.

    Attributes:
        lock: The version 2 `gf.lock`, its resources included.
        snapshot: A snapshot predictor's context rows, exactly the rows the
            lock's context digest names: split, projected and normalized as
            `served_context` reads them. None for a request or live context.
    """

    lock: PredictorLock
    snapshot: pa.Table | None = None

    @property
    def name(self) -> str:
        """The predictor's name."""
        return self.lock.name


def pushed(predictor: "str | Path") -> Pushed:
    """Reads a checked predictor: a directory with `gf.yaml` and an up-to-date `gf.lock`.

    `gf push` and `Client.predictors.push` both send this: the lock `gf check`
    wrote and, for a snapshot context, the context rows its digest names,
    never the snapshot file's query rows.

    Raises:
        ServiceError: `LockRequired` for a missing, version 1 or stale lock;
            `SnapshotMismatch` for a snapshot that changed since `gf check`.
    """
    from generalforecast.errors import Error
    from generalforecast.predictors.lock import lock_document
    from generalforecast.predictors.resolution import verified
    from generalforecast.predictors.runtime.context import served_context
    from generalforecast.predictors.spec import load
    from generalforecast.predictors.tables import read_frame

    root = Path(predictor)
    try:
        # The lock must be exactly what gf check writes now, as gf serve and
        # gf.Predictor require; a forged or stale one never leaves.
        verified(load(root), root)
        document = lock_document(root)
    except Error as error:
        code = "LockRequired" if error.param == "lock" else error.code
        if error.reason == "snapshot_changed":
            code = "SnapshotMismatch"
        raise ServiceError(code, error.message, 422) from None
    lock = PredictorLock.from_dict(document)
    snapshot = None
    if lock.context_kind == "snapshot":
        path = lock.context["path"]
        try:
            snapshot = served_context(lock, read_frame(root, path))
        except Error as error:
            raise ServiceError("MissingFile", error.message, 422) from None
        except PredictorError as error:
            raise ServiceError("SnapshotMismatch", f"{error.message}", 422) from None
    return Pushed(lock, snapshot)


def append_payload(data: Any, *, id: str, write_id: str) -> JsonObject:
    """The `/contexts/{name}/append` payload: rows carrying their id column."""
    rows = []
    for row in records(data):
        require(
            id in row and row[id] not in (None, ""),
            "live context rows need a nonempty id",
        )
        rows.append(dict(row))
    return json_object({"write_id": write_id, "rows": rows})


def label_payload(data: Any, *, id: str, column: str, write_id: str) -> JsonObject:
    """The `/contexts/{name}/label` payload: one target's values by row id."""
    labels = []
    for row in records(data):
        require(
            id in row and row[id] not in (None, ""),
            "live context labels need a nonempty id",
        )
        require(column in row, "label column missing")
        labels.append({"id": row[id], "value": row[column]})
    return json_object({"write_id": write_id, "target": column, "labels": labels})


def inference_result(value: JsonObject) -> PredictionResult:
    """Decodes the foundation result, not fixture inference outputs."""
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
