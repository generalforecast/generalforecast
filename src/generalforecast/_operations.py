# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The hosted SDK's operations, written once for the synchronous and asynchronous clients.

Each operation is a generator. It yields the steps it needs and receives each
step's result:

- a `Call` to a versioned route, answered with the reply's unwrapped result;
- an `Upload` of an Arrow snapshot, answered with its receipt;
- a `Work` item, CPU-bound conversion run off the event loop, answered with
  its return value;
- a `Pause` between polls.

`Client` runs the steps in the calling thread and `AsyncClient` awaits them,
so for the same call both send the same requests, in the same order, and
decode the replies the same way. A step's failure is thrown back into the
operation, which may handle it, as `push` handles a missing predictor.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlencode

from generalforecast.contracts import (
    ContextImportRef,
    Environment,
    PredictionResult,
    PromoteRequest,
    Revision,
    parse_lock,
)
from generalforecast.predictors.validation import PredictorError, bytes_digest
from generalforecast.retry import ONCE, PREDICT, READ, WRITE, after_unknown_outcome
from generalforecast.service.boundary import (
    JsonObject,
    Request,
    Resource,
    ResourceKind,
    Scope,
    json_object,
)
from generalforecast.service.contracts import (
    MODEL_NAME_PATTERN,
    InputError,
    ServiceError,
    name,
    number,
    require,
)
from generalforecast.service.interfaces import (
    AcceptedWrite,
    ContextSnapshot,
    Mutation,
    SourcePin,
)
from generalforecast.service.sdk import (
    ContextSpec,
    ImportReceipt,
    RevisionDetails,
    Snapshot,
    WriteReceipt,
    append_payload,
    envelope,
    environment_name,
    inference_result,
    label_payload,
    pushed,
    table,
    unwrap,
)
from generalforecast.tabular import arrow_bytes

if TYPE_CHECKING:
    from generalforecast.transport import Profile

ARROW_STREAM = "application/vnd.apache.arrow.stream"
# The most rows an append, or labels a label, the service takes in one call;
# a larger write is sent in chunks of as many. See `LiveContextAppend`.
WRITE_CHUNK = 100
# The longest write id the service takes, in UTF-8 bytes.
WRITE_ID_LENGTH = 256
# The namespace of the write ids a chunked write sends; no caller's key may
# start with it, so a chunk can never replay a caller's own write.
CHUNK_PREFIX = "gf-chunk/"


@dataclass(frozen=True)
class Call:
    """One versioned request: a route, the resource it names and its payload.

    `path` is under `/v1/projects/{project}`, or under `/v1` for a global
    route. A GET sends no body; `params` become its query.

    `retry` is how the client may send it again after a transient failure
    (`generalforecast.retry`): `PREDICT`, `READ` or `WRITE`, for a write the
    service applies once however often it is sent, or `ONCE`. None retries
    a GET as a `READ` and sends any other request once.
    """

    method: str
    path: str
    kind: ResourceKind
    resource: str
    payload: JsonObject | None = None
    global_route: bool = False
    params: Mapping[str, Any] | None = None
    deadline: float | None = None
    retry: str | None = None

    @property
    def retry_class(self) -> str | None:
        """How the call is retried: its `retry`, a GET as a read, else never (None)."""
        if self.retry is not None:
            return None if self.retry == ONCE else self.retry
        return READ if self.method == "GET" else None


@dataclass(frozen=True)
class Upload:
    """An Arrow snapshot, `POST /snapshots`."""

    data: bytes


@dataclass(frozen=True)
class Chunk:
    """A chunk of a training upload the service takes, `POST /uploads/{id}/bytes`."""

    upload: str
    start: int
    data: bytes
    total: int


@dataclass(frozen=True)
class Work:
    """A CPU-bound function the asynchronous client runs in a worker thread."""

    function: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Pause:
    """A wait between polls."""

    seconds: float


Step = Call | Upload | Chunk | Work | Pause
type Operation[T] = Generator[Step, Any, T]


@dataclass(frozen=True)
class Prepared:
    """A step as the transport sends it, and the request its reply must answer."""

    method: str
    path: str
    body: Any
    headers: dict[str, str]
    request: Request
    deadline: float | None = None


def prepare(project: str, step: Call | Upload | Chunk) -> Prepared:
    """The path, body and headers a client sends for a step."""
    request_id = "r-" + uuid.uuid4().hex
    if isinstance(step, Chunk):
        end = step.start + len(step.data) - 1
        return Prepared(
            "POST",
            f"/v1/projects/{project}/uploads/{name(step.upload)}/bytes",
            step.data,
            {
                "Content-Type": "application/octet-stream",
                "Content-Range": f"bytes {step.start}-{end}/{step.total}",
                "X-GF-Project": project,
                "X-GF-Request-ID": request_id,
            },
            Request(Scope(project, request_id), Resource("corpus", step.upload), {}),
        )
    if isinstance(step, Upload):
        return Prepared(
            "POST",
            f"/v1/projects/{project}/snapshots",
            step.data,
            {
                "Content-Type": ARROW_STREAM,
                "X-GF-Project": project,
                "X-GF-Request-ID": request_id,
            },
            Request(Scope(project, request_id), Resource("context", "snapshots"), {}),
        )
    request = envelope(project, request_id, step.kind, step.resource, step.payload)
    prefix = "/v1" if step.global_route else f"/v1/projects/{project}"
    query = "?" + urlencode(dict(step.params)) if step.params else ""
    return Prepared(
        step.method,
        prefix + step.path + query,
        request.to_dict() if step.method == "POST" else None,
        {"X-GF-Request-ID": request_id, "X-GF-Project": project},
        request,
        step.deadline,
    )


def answer(prepared: Prepared, value: JsonObject) -> JsonObject:
    """The result of a reply, once it answers the request that was sent."""
    return unwrap(value, prepared.request)


def decoded[T](decode: Callable[[Any], T], value: Any, what: str) -> T:
    """A reply value decoded, or a `ProtocolError` naming what was malformed."""
    try:
        return decode(value)
    except (KeyError, TypeError, ValueError, PredictorError):
        raise ServiceError(
            "ProtocolError", f"invalid {what} in response", 502
        ) from None


# A key whose project is not configured is asked for it once, on a global
# route; that request names a placeholder project, and the key decides.
UNSCOPED = "local"
KEY_SCOPE = Call("GET", "/auth/status", "workspace", "auth", global_route=True)


def key_project(status: JsonObject) -> str:
    """The project an `auth.status` reply says the key is scoped to."""
    scoped = status.get("key")
    project = scoped.get("project") if isinstance(scoped, dict) else None
    if not isinstance(project, str):
        raise InputError(
            "ValidationError",
            "this API key is scoped to no project",
            param="project",
            hint="set GF_PROJECT, or pass project=",
        )
    return name(project)


class Context(Protocol):
    """What operations read from their client: the project, profile and handle types."""

    @property
    def project(self) -> str | None: ...

    profile: Profile
    _hosted_predictor: Any
    _live_context: Any


def predict(client: Context, **settings: Any) -> Operation[PredictionResult]:
    """`POST /predict`: an ad hoc predictor's manifest fragment and rows."""
    from generalforecast.predictor import PredictRequest

    request = yield Work(PredictRequest.build, kwargs=settings)
    result = yield Call(
        "POST", "/predict", "predictor", "ad-hoc", request.payload(), retry=PREDICT
    )
    return inference_result(result)


def upload_snapshot(client: Context, data: Any) -> Operation[JsonObject]:
    """`POST /snapshots`: the full normalized Arrow snapshot, its digest verified."""
    normalized = yield Work(table, (data,))
    encoded = yield Work(arrow_bytes, (normalized,))
    digest = yield Work(bytes_digest, (encoded,))
    receipt = yield Upload(encoded)
    if type(receipt.get("rows")) is not int or receipt != {
        "digest": digest,
        "rows": normalized.num_rows,
    }:
        raise ServiceError(
            "ProtocolError", "snapshot receipt does not match uploaded content", 502
        )
    return receipt


def listed(kind: ResourceKind, collection: str) -> Operation[list[JsonObject]]:
    """`GET /{collection}`: the project's resource metadata."""
    result = yield Call("GET", f"/{collection}", kind, collection)
    value = result.get(collection)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ServiceError("ProtocolError", "invalid collection response", 502)
    return value


def got(kind: ResourceKind, collection: str, resource: str) -> Operation[JsonObject]:
    """`GET /{collection}/{name}`: one project-qualified resource."""
    return (yield Call("GET", f"/{collection}/{name(resource)}", kind, resource))


# A catalog model id, `general-forecast/<name>`, with an optional variant.
_MODEL_ID = re.compile(rf"general-forecast/({MODEL_NAME_PATTERN})(?:@\S+)?")
# A project's fine-tune, `<project>/<name>`, or one version, `<project>/<name>@ft3`.
_FINE_TUNE_ID = re.compile(
    r"([a-z][a-z0-9-]{0,62})/([a-z][a-z0-9-]{0,62})(@ft[1-9][0-9]*)?"
)


def model_got(model: str, project: str | None = None) -> Operation[JsonObject]:
    """`GET /models/{name}`: a catalog model, or one of the project's fine-tunes.

    A catalog model is named by its canonical id, `general-forecast/<name>`;
    a fine-tune by its id, `<project>/<name>@ft<version>` for one version,
    or `<project>/<name>` for its latest version with every version under
    `versions`, newest first. The path takes the name alone, since a path
    segment holds no `/`, so a fine-tune is read from the client's own
    project, `project` when the client knows it. A bare name is refused with
    the canonical id, as every surface refuses it.
    """
    found = _MODEL_ID.fullmatch(model) if isinstance(model, str) else None
    if found is not None:
        model_name = found.group(1)
        return (yield Call("GET", f"/models/{model_name}", "model", model_name))
    tuned = _FINE_TUNE_ID.fullmatch(model) if isinstance(model, str) else None
    if tuned is None:
        bare = isinstance(model, str) and "/" not in model and bool(model)
        raise InputError(
            "ValidationError",
            f"`{model}` is not a catalog model id",
            param="model",
            hint=f"use general-forecast/{model}"
            if bare
            else "model ids are general-forecast/<name>, or a fine-tune's "
            "<project>/<name>[@ft<version>]",
        )
    owner, fine_tune, version = tuned.groups()

    def other(project: str) -> InputError:
        return InputError(
            "ValidationError",
            f"{model} is project {owner}'s; this client reads project {project}",
            param="model",
            hint=f"read it with a client for project {owner}",
        )

    if project is not None and project != owner:
        raise other(project)
    result = yield Call(
        "GET", f"/models/{fine_tune}{version or ''}", "model", fine_tune
    )
    # An unscoped key's project is the reply's, which names the fine-tune.
    named = result.get("name") if isinstance(result, dict) else None
    if isinstance(named, str) and "/" in named and named.split("/", 1)[0] != owner:
        raise other(named.split("/", 1)[0])
    return result


def posted(
    kind: ResourceKind,
    collection: str,
    resource: str,
    action: str,
    payload: JsonObject,
    *,
    retry: str | None = None,
) -> Operation[JsonObject]:
    """`POST /{collection}/{name}/{action}`; `retry` as `Call` takes it."""
    suffix = f"/{name(resource)}" + (f"/{action}" if action else "")
    return (
        yield Call(
            "POST", f"/{collection}{suffix}", kind, resource, payload, retry=retry
        )
    )


def created(
    kind: ResourceKind,
    collection: str,
    resource: str,
    payload: JsonObject,
    *,
    retry: str | None = None,
) -> Operation[JsonObject]:
    """`POST /{collection}`: a new resource; `retry` as `Call` takes it."""
    return (yield Call("POST", f"/{collection}", kind, resource, payload, retry=retry))


def run_create(name_: str, payload: JsonObject) -> Operation[JsonObject]:
    """`POST /runs`, retried: the service returns a run created again with the same spec."""
    return (yield from created("run", "runs", name_, payload, retry=WRITE))


def corpus_create(
    name_: str, members: JsonObject, parent: str | None
) -> Operation[JsonObject]:
    """`POST /corpora`, retried, taking its own earlier try's corpus as made.

    The service never reuses a corpus's name, so a try sent again after one
    whose answer was lost is refused 409 if that try made it. Only then is
    the corpus read back, and taken when it lists exactly these members and
    parent; any other conflict is raised as it is.
    """
    payload: JsonObject = {"name": name_, "members": members}
    if parent is not None:
        payload["parent"] = parent
    try:
        return (yield from created("corpus", "corpora", name_, payload, retry=WRITE))
    except ServiceError as error:
        if error.status != 409 or not after_unknown_outcome(error):
            raise
        conflict = error
    found = yield from got("corpus", "corpora", name_)
    if found.get("members") != json.loads(json.dumps(members)) or (
        found.get("parent") != parent
    ):
        raise conflict
    return found


def context_create(client: Context, config: JsonObject | ContextSpec) -> Operation[Any]:
    """Creates a live context from its schema."""
    value = config.to_dict() if isinstance(config, ContextSpec) else json_object(config)
    resource = name(value.get("name"))
    result = yield from created("context", "contexts", resource, value)
    return client._live_context.from_response(client, resource, result)


def context_get(client: Context, resource: str) -> Operation[Any]:
    """A detached live-context snapshot with typed write methods."""
    result = yield from got("context", "contexts", resource)
    return client._live_context.from_response(client, resource, result)


class PartialWrite(ServiceError):
    """A chunked append or label that failed after the service accepted some chunks.

    It is the failed chunk's error, with its code and status, so `except
    gf.CapacityExceeded` still catches a rate limit; the message and
    `because` say which chunks were accepted. Retrying with the same idempotency key is safe:
    the accepted chunks replay their receipts and the rest are sent.

    Attributes:
        accepted: The receipts of the accepted chunks, in order.
        chunks: How many chunks the write was sent in.
    """

    def __init__(
        self,
        failure: ServiceError,
        action: str,
        accepted: Sequence[WriteReceipt],
        chunks: int,
        records: int,
    ) -> None:
        """Wraps the failed chunk's error, naming what was accepted before it."""
        noun = "rows" if action == "append" else "labels"
        taken = min(len(accepted) * WRITE_CHUNK, records)
        ids = [receipt.write_id for receipt in accepted]
        named = f"`{ids[0]}`" if len(ids) == 1 else f"`{ids[0]}` to `{ids[-1]}`"
        super().__init__(
            failure.code,
            f"{action} failed at chunk {len(accepted) + 1} of {chunks}: "
            f"{failure.message}",
            failure.status,
            reason=failure.reason,
            param=failure.param,
            because=f"chunks 1 to {len(accepted)} of {chunks}, {taken} of "
            f"{records} {noun}, were accepted as write ids {named}",
            hint="retry with the same idempotency_key: accepted chunks replay "
            "their receipts and the rest are sent",
            limit=failure.limit,
            actual=failure.actual,
            request_id=failure.request_id,
        )
        self.accepted = tuple(accepted)
        self.chunks = chunks


def write_key(key: Any) -> str:
    """An idempotency key, checked as every write checks it.

    Raises:
        InputError: A key that is not 1 to `WRITE_ID_LENGTH` bytes of UTF-8
            text, or that is in the chunked writes' namespace.
    """
    try:
        size = len(key.encode()) if isinstance(key, str) else 0
    except UnicodeEncodeError:
        size = 0
    if not 0 < size <= WRITE_ID_LENGTH:
        raise InputError(
            "ValidationError",
            f"an idempotency key is 1 to {WRITE_ID_LENGTH} bytes of text",
            param="idempotency_key",
        )
    if key.startswith(CHUNK_PREFIX):
        raise InputError(
            "ValidationError",
            f"idempotency keys starting with `{CHUNK_PREFIX}` are reserved for "
            "chunked writes",
            param="idempotency_key",
        )
    return key


def chunk_write_id(key: str, index: int) -> str:
    """The write id of the `index`th chunk, from 0, of a write under `key`.

    `gf-chunk/<sha256 of key>/<n>`, in a namespace no caller's key can use,
    and of one length whatever the key: a retry under the same key sends the
    same ids, so the service replays each chunk it already accepted.
    """
    digest = hashlib.sha256(write_key(key).encode()).hexdigest()
    return f"{CHUNK_PREFIX}{digest}/{index + 1}"


def _context_write(
    context: str, action: str, payload: JsonObject, field: str, key: str
) -> Operation[WriteReceipt]:
    """Posts an append or label, in chunks of `WRITE_CHUNK` records when it has more.

    A write that fits one call is sent as it is, under `key`. A larger one
    is sent chunk by chunk under `chunk_write_id`s; its receipt is the last
    chunk's, under `key`, with every chunk's rows and receipts.

    Raises:
        PartialWrite: When a chunk fails after earlier chunks were accepted.
    """
    records = payload[field]
    if len(records) <= WRITE_CHUNK:
        result = yield from posted("context", "contexts", context, action, payload)
        return decoded(WriteReceipt.from_dict, result, "write receipt")
    chunks = -(-len(records) // WRITE_CHUNK)
    accepted: list[WriteReceipt] = []
    for index in range(chunks):
        part = payload | {
            "write_id": chunk_write_id(key, index),
            field: records[index * WRITE_CHUNK : (index + 1) * WRITE_CHUNK],
        }
        try:
            result = yield from posted("context", "contexts", context, action, part)
            accepted.append(decoded(WriteReceipt.from_dict, result, "write receipt"))
        except ServiceError as failure:
            if not accepted:
                raise
            raise PartialWrite(
                failure, action, accepted, chunks, len(records)
            ) from failure
    receipts = [receipt.to_dict() for receipt in accepted]

    def combined(receipts: list[JsonObject]) -> WriteReceipt:
        return WriteReceipt.from_dict(
            receipts[-1]
            | {
                "write_id": key,
                "rows": sum(receipt["rows"] for receipt in receipts),
                "durably_accepted": all(
                    receipt["durably_accepted"] for receipt in receipts
                ),
                "chunks": receipts,
            }
        )

    return decoded(combined, receipts, "write receipt")


def context_append(
    context: str, data: Any, *, id: str, idempotency_key: str
) -> Operation[WriteReceipt]:
    """Appends rows under a durable write id, in chunks the service accepts."""
    write_key(idempotency_key)
    payload = yield Work(
        append_payload, (data,), {"id": id, "write_id": idempotency_key}
    )
    return (
        yield from _context_write(context, "append", payload, "rows", idempotency_key)
    )


def context_label(
    context: str, data: Any, *, id: str, column: str, idempotency_key: str
) -> Operation[WriteReceipt]:
    """Attaches one target's values to rows already in the context, in chunks the service accepts."""
    write_key(idempotency_key)
    payload = yield Work(
        label_payload,
        (data,),
        {"id": id, "column": column, "write_id": idempotency_key},
    )
    return (
        yield from _context_write(context, "label", payload, "labels", idempotency_key)
    )


def context_import(reference: ContextImportRef) -> Operation[ImportReceipt]:
    """Binds a cloud or uploaded table to an existing live context."""
    result = yield from posted(
        "context", "contexts", reference.context, "imports", reference.to_dict()
    )
    return decoded(ImportReceipt.from_dict, result, "import receipt")


def context_import_file(
    client: Context, context: str, data: Any
) -> Operation[ImportReceipt]:
    """Uploads a local bounded table, then binds it to a live context."""
    receipt = yield from upload_snapshot(client, data)
    return (
        yield from context_import(
            ContextImportRef(
                context,
                "upload://" + receipt["digest"],
                "arrow",
                digest=receipt["digest"],
            )
        )
    )


def context_refresh(context: str) -> Operation[ImportReceipt]:
    """Requests an explicit provider refresh; the service answers with its import."""
    result = yield from posted("context", "contexts", context, "refresh", {})
    return decoded(ImportReceipt.from_dict, result, "import receipt")


def context_mutate(context: str, mutation: Mutation) -> Operation[AcceptedWrite]:
    """An atomic write; accepted is not yet serving. Preserve its write_id."""
    result = yield from posted(
        "context", "contexts", context, "mutations", dict(mutation)
    )
    return cast(AcceptedWrite, result)


def context_snapshot(
    context: str,
    *,
    target: str,
    knowledge_at: float | None,
    event_before: float | None,
) -> Operation[ContextSnapshot]:
    """Keeps knowledge (acceptance) time independent of event time."""
    result = yield from posted(
        "context",
        "contexts",
        context,
        "snapshot",
        {"target": target, "knowledge_at": knowledge_at, "event_before": event_before},
    )
    return cast(ContextSnapshot, result)


def source_pin(spec: JsonObject) -> Operation[SourcePin]:
    """Pins server-relative bytes; this does not upload a client-side path."""
    result = yield from created("source", "sources", "new", spec)
    return cast(SourcePin, result)


# Training uploads and runs that are done waiting.
UPLOAD_DONE = ("verified", "rejected", "expired")
RUN_DONE = ("finished", "cancelled", "failed")


def parquet_bytes(data: Any) -> bytes:
    """A training table as Parquet bytes: a path or bytes as given, else a table written."""
    import io

    import pyarrow.parquet as pq

    if isinstance(data, bytes | bytearray):
        return bytes(data)
    if isinstance(data, str | Path):
        return Path(data).read_bytes()
    sink = io.BytesIO()
    pq.write_table(table(data), sink)
    return sink.getvalue()


def put_chunk(url: str, start: int, data: bytes, total: int) -> int:
    """Sends one chunk to a resumable upload URL; returns the reply's status.

    308 means the upload wants more; 200 or 201 means it is complete.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={
            "Content-Range": f"bytes {start}-{start + len(data) - 1}/{total}",
            "Content-Type": "application/octet-stream",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            return reply.status
    except urllib.error.HTTPError as error:
        if error.code == 308:
            return 308
        raise ServiceError(
            "ProviderUnavailable",
            f"the upload's storage refused a chunk ({error.code})",
            503,
        ) from None


def training_upload(
    data: Any, metadata: JsonObject, *, wait: bool, poll: float = 1.0
) -> Operation[JsonObject]:
    """Uploads a training table in chunks and waits for the service to verify it.

    The upload opens with the table's size and metadata, so the service
    refuses a bad one before any bytes move. The chunks go where it says:
    straight to object storage when it returns a URL, else to the service.
    Once every declared byte is stored, the service verifies the upload on
    its own. A rejected upload raises with the reason the service gave.
    """
    body = yield Work(parquet_bytes, (data,))
    state = yield Call(
        "POST",
        "/uploads",
        "corpus",
        "new",
        {"bytes": len(body), "format": "parquet", "metadata": metadata},
    )
    upload, url, size = state["id"], state.get("url"), state["chunk_bytes"]
    for start in range(0, len(body), size):
        part = body[start : start + size]
        if url:
            yield Work(put_chunk, (url, start, part, len(body)))
        else:
            yield Chunk(upload, start, part, len(body))
    while wait and state["state"] not in UPLOAD_DONE:
        yield Pause(poll)
        state = yield from got("corpus", "uploads", upload)
    if state["state"] in ("rejected", "expired"):
        raise ServiceError(
            "ValidationError",
            f"upload {upload} was {state['state']}: {state.get('error') or ''}".strip(),
            422,
        )
    return state


def upload_metadata(
    target: str,
    task: str,
    *,
    column_types: Mapping[str, str] | None = None,
    time: str | None = None,
    group: str | None = None,
    origin: JsonObject | None = None,
) -> JsonObject:
    """A training upload's metadata, leaving out what was not given."""
    found: JsonObject = {"target": target, "task": task}
    if column_types is not None:
        found["column_types"] = dict(column_types)
    for key, value in (("time", time), ("group", group), ("origin", origin)):
        if value is not None:
            found[key] = value
    return found


def run_spec(name_: str, **settings: Any) -> JsonObject:
    """A run's `POST /runs` payload, leaving out settings that were not given."""
    return {"name": name_, **{k: v for k, v in settings.items() if v is not None}}


def run_list(state: str | None) -> Operation[list[JsonObject]]:
    """`GET /runs`: the project's runs by name, with `state` only those in it."""
    params = None if state is None else {"state": state}
    result = yield Call("GET", "/runs", "run", "runs", params=params)
    value = result.get("runs")
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ServiceError("ProtocolError", "invalid collection response", 502)
    return value


def run_cancel(run: str, *, save: bool) -> Operation[JsonObject]:
    """`POST /runs/{run}/cancel`; with `save`, a training run publishes what it trained."""
    require(type(save) is bool, "save must be true or false")
    return (
        yield from posted("run", "runs", run, "cancel", {"save": True} if save else {})
    )


def run_lineage(run: str) -> Operation[JsonObject]:
    """`GET /runs/{run}/lineage`: every change to a run, oldest first."""
    return (yield Call("GET", f"/runs/{name(run)}/lineage", "run", run))


def run_wait(run: str, *, timeout: float | None, poll: float) -> Operation[JsonObject]:
    """Polls a run until it finishes, is cancelled or fails."""
    deadline = None if timeout is None else time.monotonic() + timeout
    state = yield from got("run", "runs", run)
    while state["state"] not in RUN_DONE:
        if deadline is not None and time.monotonic() >= deadline:
            raise ServiceError(
                "WaitTimeout", f"run {run} is still {state['state']}", 504
            )
        yield Pause(poll)
        state = yield from got("run", "runs", run)
    return state


def _predictor(predictor: str, suffix: str = "") -> str:
    return f"/predictors/{name(predictor)}{suffix}"


def _environment_path(
    client: Context, predictor: str, env: str | None, tail: str = ""
) -> str:
    selected = environment_name(env, client.profile.environment)
    return _predictor(predictor, f"/environments/{selected}{tail}")


def predictor_get(client: Context, resource: str, env: str | None) -> Operation[Any]:
    """The predictor's facts and, with `env`, that environment's state."""
    state = yield from got("predictor", "predictors", resource)
    environment = (
        (yield from predictor_environment(client, resource, env))
        if env is not None
        else None
    )
    return client._hosted_predictor.from_response(client, resource, state, environment)


def predictor_create(client: Context, predictor: str) -> Operation[Any]:
    """Creates only the predictor name, not serving readiness."""
    result = yield from created(
        "predictor", "predictors", predictor, {"name": name(predictor)}, retry=WRITE
    )
    return client._hosted_predictor.from_response(client, predictor, result)


def predictor_revisions(predictor: str) -> Operation[tuple[Revision, ...]]:
    """Immutable revisions, without inferring environment readiness."""
    value = yield Call(
        "GET", _predictor(predictor, "/revisions"), "predictor", predictor
    )
    return decoded(
        lambda found: tuple(Revision.from_dict(row) for row in found["revisions"]),
        value,
        "revisions",
    )


def predictor_revision(predictor: str, number: int) -> Operation[RevisionDetails]:
    """The exact immutable compiled lock for a revision."""
    require(type(number) is int and number > 0, "invalid revision")
    value = yield Call(
        "GET", _predictor(predictor, f"/revisions/{number}"), "predictor", predictor
    )
    return decoded(
        lambda found: RevisionDetails(
            Revision.from_dict(found["revision"]), parse_lock(found["lock"])
        ),
        value,
        "revision",
    )


def predictor_environments(predictor: str) -> Operation[tuple[Environment, ...]]:
    """Runtime-bound environment pointers and their actual states."""
    value = yield Call(
        "GET", _predictor(predictor, "/environments"), "predictor", predictor
    )
    return decoded(
        lambda found: tuple(
            Environment.from_dict(row) for row in found["environments"]
        ),
        value,
        "environments",
    )


def predictor_environment(
    client: Context,
    predictor: str,
    env: str | None,
    deadline: float | None = None,
    retry: str | None = None,
) -> Operation[Environment]:
    """One environment with its revision and CAS generation; `retry` as `Call` takes it."""
    value = yield Call(
        "GET",
        _environment_path(client, predictor, env),
        "predictor",
        predictor,
        deadline=deadline,
        retry=retry,
    )
    return decoded(Environment.from_dict, value, "environment")


def predictor_create_environment(predictor: str, env: str) -> Operation[Environment]:
    """Creates an environment, a pointer to the revision it serves."""
    value = yield from posted(
        "predictor",
        "predictors",
        predictor,
        "environments",
        {"name": name(env)},
        retry=WRITE,
    )
    return decoded(Environment.from_dict, value, "environment")


def predictor_move(
    client: Context,
    predictor: str,
    action: str,
    *,
    revision: int,
    env: str | None,
    expected_generation: int | None,
) -> Operation[Environment]:
    """Promotes or rolls back to an explicit revision under a CAS guard."""
    selected = environment_name(env, client.profile.environment)
    generation = (
        (yield from predictor_environment(client, predictor, selected)).generation
        if expected_generation is None
        else expected_generation
    )
    try:
        value = yield from posted(
            "predictor",
            "predictors",
            predictor,
            f"environments/{selected}/{action}",
            PromoteRequest(revision, generation).to_dict(),
            retry=WRITE,
        )
    except ServiceError as error:
        # Every try carries one request id, which the service replays once it
        # has committed the move; a try that arrives while the first is still
        # committing loses its CAS instead. Only after a try whose outcome was
        # unknown is the environment read, and the move taken as made when it
        # points at this revision one generation on.
        if error.status != 409 or not after_unknown_outcome(error):
            raise
        conflict = error
    else:
        return decoded(Environment.from_dict, value, "environment")
    state = yield from predictor_environment(client, predictor, selected)
    if state.revision != revision or state.generation != generation + 1:
        raise conflict
    return state


def predictor_predict(
    client: Context,
    predictor: str,
    queries: Any,
    *,
    context: Any,
    as_of: Any,
    env: str | None,
) -> Operation[PredictionResult]:
    """Predicts with a declared predictor in an environment, the profile's by default.

    Services score `production` when a request names no environment, so
    production is named by omission; a standalone `gf serve`, whose
    predictors have no environments, refuses any other.
    """
    from generalforecast.predictor import PredictRequest
    from generalforecast.transport import DEFAULT_ENVIRONMENT

    payload = yield Work(
        PredictRequest.declared_payload,
        (queries,),
        {"context": context, "as_of": as_of},
    )
    selected = environment_name(env, client.profile.environment)
    result = yield Call(
        "POST",
        _predictor(predictor, "/predict"),
        "predictor",
        predictor,
        payload,
        params=None if selected == DEFAULT_ENVIRONMENT else {"env": selected},
        retry=PREDICT,
    )
    return inference_result(result)


def predictor_push(
    client: Context,
    predictor: str | Path,
    *,
    env: str | None,
    wait: bool,
    timeout: float,
) -> Operation[Revision]:
    """Registers a checked predictor's revision and points an environment at it."""
    checked = yield Work(pushed, (predictor,))
    lock = checked.lock
    resource = checked.name
    selected = environment_name(env, client.profile.environment)
    try:
        yield from got("predictor", "predictors", resource)
    except ServiceError as error:
        if error.status != 404:
            raise
        yield from predictor_create(client, resource)
    snapshot = None
    if checked.snapshot is not None:
        snapshot = (yield from upload_snapshot(client, checked.snapshot))["digest"]
        require(snapshot == lock.data_digest, "snapshot differs from gf.lock")
    value = yield from posted(
        "predictor",
        "predictors",
        resource,
        "revisions",
        {"lock": lock.to_dict(), "snapshot": snapshot, "manifest": checked.manifest},
        retry=WRITE,
    )
    revision = decoded(Revision.from_dict, value, "revision")
    try:
        environment = yield from predictor_environment(client, resource, selected)
    except ServiceError as error:
        if error.status != 404:
            raise
        environment = yield from predictor_create_environment(resource, selected)
    promoted = yield from predictor_move(
        client,
        resource,
        "promote",
        revision=revision.number,
        env=selected,
        expected_generation=environment.generation,
    )
    if wait:
        yield from predictor_wait(
            client,
            resource,
            env=selected,
            revision=revision.number,
            generation=promoted.generation,
            timeout=timeout,
            poll_interval=0.2,
        )
    return revision


def _wait_timeout() -> ServiceError:
    return ServiceError("WaitTimeout", "environment readiness timed out", 504)


def predictor_wait(
    client: Context,
    predictor: str,
    *,
    env: str | None,
    revision: int | None,
    generation: int | None,
    timeout: float,
    poll_interval: float,
) -> Operation[Environment]:
    """Polls an exact environment pointer until it is ready.

    A superseded pointer raises `Conflict` and a failed or stopped one
    `EnvironmentFailed`. Each poll is bounded by the deadline, and running
    out of time raises `WaitTimeout`, during a poll or between polls. A
    `timeout` of 0 checks once.
    """
    number(timeout, 0, 86400, "timeout")
    number(poll_interval, 0.001, 30, "poll_interval")
    deadline = time.monotonic() + timeout
    first = True
    while True:
        if not first and time.monotonic() >= deadline:
            raise _wait_timeout()
        # A wait with no time checks once: its poll is bounded by no deadline,
        # so it is not retried either. Any other poll is retried within it.
        bounded = bool(timeout) or not first
        try:
            state = yield from predictor_environment(
                client,
                predictor,
                env,
                deadline if bounded else None,
                None if bounded else ONCE,
            )
        except ServiceError as error:
            if error.code == "Timeout" and time.monotonic() >= deadline:
                raise _wait_timeout() from None
            raise
        first = False
        if (revision is not None and state.revision != revision) or (
            generation is not None and state.generation != generation
        ):
            raise ServiceError(
                "Conflict", "environment pointer changed while waiting", 409
            )
        revision, generation = state.revision, state.generation
        if state.status == "ready":
            return state
        if state.status in {"failed", "stopped"}:
            raise ServiceError(
                "EnvironmentFailed", "environment did not become ready", 409
            )
        if time.monotonic() >= deadline:
            raise _wait_timeout()
        yield Pause(min(poll_interval, max(0.0, deadline - time.monotonic())))


def predictor_report(
    client: Context, predictor: str, env: str | None, report: str
) -> Operation[Snapshot]:
    """An environment's `metrics` or `logs`, as reported, never fabricated."""
    value = yield Call(
        "GET",
        _environment_path(client, predictor, env, f"/{report}"),
        "predictor",
        predictor,
    )
    return decoded(Snapshot.from_dict, value, report)


def predictor_delete(predictor: str) -> Operation[Snapshot]:
    """Requests explicit stop and delete; unsupported cleanup is never success."""
    value = yield from posted("predictor", "predictors", predictor, "delete", {})
    return decoded(Snapshot.from_dict, value, "deletion")


# Session-only onboarding and workspace-admin actions. The server checks the
# signed-in Google session and live workspace role on every call; project API
# keys receive its typed SessionRequired denial, never a local short-circuit.
WORKSPACE_ACTIONS: dict[str, tuple[str, str, str]] = {
    "onboarding.get": ("GET", "/onboarding", "onboarding"),
    "workspaces.list": ("GET", "/workspaces", "workspaces"),
    "workspaces.create": ("POST", "/workspaces", "new"),
    "workspaces.get": ("GET", "/workspaces/{workspace}", "{workspace}"),
    "workspaces.update": ("POST", "/workspaces/{workspace}", "{workspace}"),
    "workspaces.join": ("POST", "/workspaces/{workspace}/join", "{workspace}"),
    "workspaces.members": ("GET", "/workspaces/{workspace}/members", "{workspace}"),
    "members.update": (
        "POST",
        "/workspaces/{workspace}/members/role",
        "{workspace}",
    ),
    "members.remove": (
        "POST",
        "/workspaces/{workspace}/members/remove",
        "{workspace}",
    ),
    "invitations.preview": ("POST", "/invitations/preview", "invitation"),
    "device.preview": ("POST", "/cli-logins/preview", "cli-login"),
    "device.approve": ("POST", "/cli-logins/approve", "cli-login"),
    "device.deny": ("POST", "/cli-logins/deny", "cli-login"),
    "invitations.accept": ("POST", "/invitations/accept", "invitation"),
    "invitations.list": ("GET", "/workspaces/{workspace}/invitations", "{workspace}"),
    "invitations.create": (
        "POST",
        "/workspaces/{workspace}/invitations",
        "{workspace}",
    ),
    "invitations.revoke": (
        "POST",
        "/workspaces/{workspace}/invitations/{id}/revoke",
        "{id}",
    ),
    "requests.list": ("GET", "/workspaces/{workspace}/join-requests", "{workspace}"),
    "requests.approve": (
        "POST",
        "/workspaces/{workspace}/join-requests/{id}/approve",
        "{id}",
    ),
    "requests.deny": (
        "POST",
        "/workspaces/{workspace}/join-requests/{id}/deny",
        "{id}",
    ),
    "requests.cancel": ("POST", "/join-requests/{id}/cancel", "{id}"),
    "waitlist.join": ("POST", "/waitlist", "waitlist"),
    "waitlist.get": ("GET", "/waitlist/{id}", "{id}"),
    "waitlist.approve": ("POST", "/waitlist/{id}/approve", "{id}"),
    "domains.list": ("GET", "/workspaces/{workspace}/domains", "{workspace}"),
    "domains.configure": ("POST", "/workspaces/{workspace}/domains", "{workspace}"),
}

# Global identity routes: (resource kind, path, resource name).
_GLOBAL_ACTIONS: dict[str, tuple[ResourceKind, str, str | None]] = {
    "workspace.get": ("workspace", "/workspace", "workspace"),
    "auth.status": ("workspace", "/auth/status", "auth"),
    "auth.logout": ("workspace", "/auth/logout", "auth"),
    "organizations.list": ("workspace", "/organizations", "organizations"),
    "organizations.create": ("workspace", "/organizations", None),
    "projects.list": ("project", "/projects", "projects"),
    "projects.create": ("project", "/projects", None),
}

# Project-scoped key and secret operations hosted serves. Keys are never read
# one at a time: `GET /keys/{id}` is not a route.
_SCOPED_OPERATIONS = {
    "keys": frozenset({"list", "create", "rotate", "revoke"}),
    "secrets": frozenset({"list", "create", "get", "rotate", "delete"}),
}

_ADMIN_ACTIONS = frozenset({"sso.get", "sso.configure", "audit.list"})


def workspace_request(
    action: str, payload: JsonObject
) -> tuple[str, str, str, JsonObject | None, JsonObject | None]:
    """Binds one workspace action to (method, path, resource, body, query).

    Path identities are repeated in POST bodies so the server can check them;
    GET identities move into the path and only join requests accept `state`.
    """
    method, path, resource = WORKSPACE_ACTIONS[action]
    keys = [k for k in ("workspace", "id") if "{" + k + "}" in path]
    values = {k: name(payload.get(k)) for k in keys}
    path = path.format(**values)
    resource = resource.format(**values)
    if method == "POST":
        return method, path, resource, payload, None
    extra = set(payload) - set(keys)
    require(
        not extra or (action == "requests.list" and extra == {"state"}),
        "GET operation accepts only path identities",
    )
    params = None
    if "state" in extra:
        require(payload["state"] in ("pending", "all"), "invalid join request state")
        params = {"state": payload["state"]}
    return method, path, resource, None, params


def manage(action: str, payload: JsonObject | None) -> Operation[JsonObject]:
    """Calls one allowlisted identity action, never a caller-chosen server method."""
    payload = {} if payload is None else json_object(payload)
    if action in WORKSPACE_ACTIONS:
        method, path, resource, body, params = workspace_request(action, payload)
        return (
            yield Call(
                method,
                path,
                "workspace",
                resource,
                body,
                global_route=True,
                params=params,
            )
        )
    if action in _GLOBAL_ACTIONS:
        kind, path, resource = _GLOBAL_ACTIONS[action]
        creating = action.endswith("create")
        require(creating or not payload, "GET operation requires empty payload")
        return (
            yield Call(
                "POST" if creating or action == "auth.logout" else "GET",
                path,
                kind,
                resource or payload.get("name", "new"),
                payload,
                global_route=True,
            )
        )
    domain, _, operation = action.partition(".")
    if operation in _SCOPED_OPERATIONS.get(domain, ()):
        if operation == "list":
            require(not payload, "GET operation requires empty payload")
            resource = domain
        elif domain == "keys":
            resource = "new" if operation == "create" else payload.get("id", "")
        else:
            resource = payload.get("name", "")
        path = f"/{domain}"
        if operation not in ("list", "create"):
            path += f"/{name(resource)}" + (
                "" if operation == "get" else f"/{operation}"
            )
        return (
            yield Call(
                "GET" if operation in ("list", "get") else "POST",
                path,
                "api-key" if domain == "keys" else "secret",
                resource,
                payload,
            )
        )
    if action in _ADMIN_ACTIONS:
        params = None
        reading = operation in ("get", "list")
        if action == "audit.list":
            require(set(payload) <= {"after", "limit"}, "invalid audit parameters")
            after, limit = payload.get("after", 0), payload.get("limit", 100)
            require(
                type(after) is int and 0 <= after <= 2**53 - 1,
                "invalid audit cursor",
            )
            require(type(limit) is int and 1 <= limit <= 100, "invalid audit limit")
            params = {"after": after, "limit": limit}
        elif reading:
            require(not payload, "GET operation requires empty payload")
        return (
            yield Call(
                "GET" if reading else "POST",
                "/" + domain,
                "workspace",
                domain,
                None if reading else payload,
                params=params,
            )
        )
    raise ServiceError("UnsupportedOperation", "identity action is not supported", 501)


def key_create(
    *, name: str, permissions: Sequence[str], expires_in: float | None
) -> Operation[JsonObject]:
    """Creates only the permissions the server allows the caller to grant."""
    payload: JsonObject = {"name": name, "permissions": list(permissions)}
    if expires_in is not None:
        payload["expires_in"] = expires_in
    return (yield from manage("keys.create", payload))
