# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Frames between the server and a worker: a JSON header plus raw arrays.

The layout matches `gf-server`'s `protocol.rs`: a u64 length, then the frame,
which is a u32 header length, the JSON header, a u32 buffer count, and each
buffer with a u64 length. Arrays are little-endian and read without copying.
"""

import dataclasses
import json
import mmap
import os
import socket
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np

from generalforecast._core import CacheKey

# What a worker does with a unit or page, and the tiers entries move between.
Action = Literal["encode", "reuse", "transient", "fuse", "extend"]
Tier = Literal["device", "host", "disk"]

# The most bytes one frame may claim, as in `protocol.rs`, so a damaged length
# never exhausts memory.
MAX_FRAME_BYTES = 1 << 33
# The most buffers one frame may claim.
MAX_BUFFERS = 1 << 20
# The version of the frames, which the server checks when a worker connects.
PROTOCOL_VERSION = 13


@dataclass(frozen=True)
class Fit:
    """Rows to fit a table's state on, which also embeds them into `pages`."""

    table: int
    epoch: int
    rows: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    pages: list[int]


@dataclass(frozen=True)
class Page:
    """One page's row representations.

    `features` and `targets` are set to embed the rows no source copies, and
    `copies` names cached pages to copy the rest from, each with (position in
    the source, position in `rows`) pairs.
    """

    table: int
    key: CacheKey
    rows: np.ndarray
    action: Action
    slots: list[int]
    features: np.ndarray | None
    targets: np.ndarray | None
    copies: list[tuple[CacheKey, list[tuple[int, int]]]] = field(default_factory=list)


@dataclass(frozen=True)
class Unit:
    """One unit of a step: rows to encode or reuse; `features` and `targets` are set to fit them.

    A `fuse` unit fits its table's state on its rows, then encodes them and
    decodes the queries that read it in the same call, in no slots.
    """

    table: int
    key: CacheKey
    rows: np.ndarray
    action: Action
    slots: list[int]
    features: np.ndarray | None = None
    targets: np.ndarray | None = None


@dataclass(frozen=True)
class QueryBatch:
    """One request's queries, and the unit each decodes against."""

    request: int
    table: int
    epoch: int
    features: np.ndarray
    units: list[int]


@dataclass(frozen=True)
class SeriesUnit:
    """A group of series as a grid to encode, reuse or forecast whole.

    Attributes:
        members: The group's series, in the order the grid's rows take, each
            with its target row then one row per feature column.
        values: The grid, `(rows, steps)`, 0 where unobserved.
        observed: Which values are observed.
        roles: Each row's role.
        action: `encode`, `reuse`, `extend` or `transient`; a unit forecast
            whole is transient.
        source: For an extension, the entry holding the shorter encoding it
            grows.
        grown: For an extension, how many steps at the grid's end are new.
    """

    table: int
    key: CacheKey
    members: np.ndarray
    cutoff: int
    values: np.ndarray
    observed: np.ndarray
    roles: np.ndarray
    action: Action
    slots: list[int]
    source: CacheKey | None = None
    grown: int = 0


@dataclass(frozen=True)
class Forecast:
    """One series' forecast: the unit it reads, its member, horizon and known future."""

    request: int
    table: int
    epoch: int
    unit: int
    target: int
    horizon: int
    future: np.ndarray
    future_observed: np.ndarray


@dataclass(frozen=True)
class Movement:
    """A copy or drop to make before running a step."""

    key: CacheKey
    source: Tier
    target: Tier | None
    slots: list[int]


@dataclass(frozen=True)
class Step:
    """What a worker runs, in order: movements, fits, pages, units, queries, forecasts."""

    movements: list[Movement]
    fits: list[Fit]
    pages: list[Page]
    units: list[Unit]
    queries: list[QueryBatch]
    series_units: list[SeriesUnit] = field(default_factory=list)
    forecasts: list[Forecast] = field(default_factory=list)


def _exactly(connection: socket.socket, count: int) -> bytes:
    chunks, remaining = [], count
    while remaining:
        chunk = connection.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("the server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(connection: socket.socket) -> tuple[dict[str, Any], list[memoryview]]:
    """Reads one frame: its header and buffers, which are views, not copies.

    A frame whose header names a shared file carries its buffers there: the file
    is mapped read-only and removed, and the buffers are views of the mapping,
    which stays valid while any of them is referenced.
    """
    (length,) = struct.unpack("<Q", _exactly(connection, 8))
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"a frame claims {length} bytes, past {MAX_FRAME_BYTES}")
    data = memoryview(_exactly(connection, length))

    def fits(offset: int, size: int) -> int:
        """The offset past `size` bytes at `offset`, which must lie in the frame."""
        if offset + size > length:
            raise ValueError("a frame's parts run past its end")
        return offset + size

    offset = fits(0, 4)
    (header_length,) = struct.unpack_from("<I", data, 0)
    end = fits(offset, header_length)
    header = json.loads(bytes(data[offset:end]))
    offset = fits(end, 4)
    (count,) = struct.unpack_from("<I", data, end)
    if count > MAX_BUFFERS:
        raise ValueError(f"a frame claims {count} buffers")
    buffers = []
    for _ in range(count):
        start = fits(offset, 8)
        (size,) = struct.unpack_from("<Q", data, offset)
        offset = fits(start, size)
        buffers.append(data[start:offset])
    if offset != length:
        raise ValueError("bytes after the frame")
    if "shared" in header:
        buffers = _shared(header.pop("shared"))
    return header, buffers


def _shared(shared: dict[str, Any]) -> list[memoryview]:
    """Maps a shared frame's file, removes it, and slices its buffers."""
    path, sizes = shared["path"], shared["sizes"]
    with open(path, "rb") as file:
        try:
            mapped = memoryview(
                mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                if sum(sizes)
                else b""
            )
        finally:
            os.unlink(path)
    if sum(sizes) != len(mapped):
        raise ValueError(f"a shared frame holds {len(mapped)} bytes, not {sum(sizes)}")
    buffers, offset = [], 0
    for size in sizes:
        buffers.append(mapped[offset : offset + size])
        offset += size
    return buffers


def write_frame(
    connection: socket.socket, header: dict[str, Any], buffers: Sequence[bytes] = ()
) -> None:
    """Writes one frame."""
    encoded = json.dumps(header).encode()
    parts = [struct.pack("<I", len(encoded)), encoded, struct.pack("<I", len(buffers))]
    for buffer in buffers:
        parts += [struct.pack("<Q", len(buffer)), buffer]
    body = b"".join(parts)
    connection.sendall(struct.pack("<Q", len(body)) + body)


def _key(value: list[Any]) -> CacheKey:
    """A cache key from its JSON: fingerprints come as strings."""
    model, table, epoch, kind, rows = value
    return CacheKey(model, table, epoch, kind, int(rows))


def _action(value: Any) -> Action:
    if value not in get_args(Action):
        raise ValueError(f"unknown action {value!r}")
    return value


def _tier(value: Any) -> Tier:
    if value not in get_args(Tier):
        raise ValueError(f"unknown tier {value!r}")
    return value


def _targets(entry: dict[str, Any], buffers: list[memoryview]) -> np.ndarray:
    dtype = "<i4" if entry["target_type"] == "classes" else "<f4"
    return np.frombuffer(buffers[entry["targets"]], dtype=dtype)


def _features(entry: dict[str, Any], buffers: list[memoryview]) -> np.ndarray:
    features = np.frombuffer(buffers[entry["features"]], dtype="<f4")
    return features.reshape(entry["count"], -1)


def _fit(entry: dict[str, Any], buffers: list[memoryview]) -> Fit:
    return Fit(
        table=entry["table"],
        epoch=entry["epoch"],
        rows=np.frombuffer(buffers[entry["rows"]], dtype="<u4"),
        features=_features(entry, buffers),
        targets=_targets(entry, buffers),
        pages=entry["pages"],
    )


def _unit(entry: dict[str, Any], buffers: list[memoryview]) -> Unit:
    fit = entry["features"] is not None
    return Unit(
        table=entry["table"],
        key=_key(entry["key"]),
        rows=np.frombuffer(buffers[entry["rows"]], dtype="<u4"),
        action=_action(entry["action"]),
        slots=entry["slots"],
        features=_features(entry, buffers) if fit else None,
        targets=_targets(entry, buffers) if fit else None,
    )


def _page(entry: dict[str, Any], buffers: list[memoryview]) -> Page:
    embed = entry["features"] is not None
    copies = [
        (_key(copy["source"]), [(at, to) for at, to in copy["positions"]])
        for copy in entry.get("copies", [])
    ]
    embedded = entry["count"] - sum(len(positions) for _, positions in copies)
    features = None
    if embed:
        features = np.frombuffer(buffers[entry["features"]], dtype="<f4")
        features = features.reshape(embedded, -1 if embedded else 0)
    return Page(
        table=entry["table"],
        key=_key(entry["key"]),
        rows=np.frombuffer(buffers[entry["rows"]], dtype="<u4"),
        action=_action(entry["action"]),
        slots=entry["slots"],
        features=features,
        targets=_targets(entry, buffers) if embed else None,
        copies=copies,
    )


def movements(entries: list[dict[str, Any]]) -> list[Movement]:
    """Movements from their JSON."""
    return [
        Movement(
            _key(entry["key"]),
            _tier(entry["source"]),
            None if entry["target"] is None else _tier(entry["target"]),
            entry["slots"],
        )
        for entry in entries
    ]


def step(header: dict[str, Any], buffers: list[memoryview]) -> Step:
    """A step from its frame."""
    queries = [
        QueryBatch(
            request=entry["request"],
            table=entry["table"],
            epoch=entry["epoch"],
            features=np.frombuffer(buffers[entry["features"]], dtype="<f4").reshape(
                entry["count"], -1
            ),
            units=entry["units"],
        )
        for entry in header["queries"]
    ]
    return Step(
        movements=movements(header["movements"]),
        fits=[_fit(entry, buffers) for entry in header["fits"]],
        pages=[_page(entry, buffers) for entry in header["pages"]],
        units=[_unit(entry, buffers) for entry in header["units"]],
        queries=queries,
        series_units=[
            _series_unit(entry, buffers) for entry in header.get("series_units", [])
        ],
        forecasts=[_forecast(entry, buffers) for entry in header.get("forecasts", [])],
    )


def _grid(
    entry: dict[str, Any], buffers: list[memoryview], steps: str
) -> tuple[np.ndarray, np.ndarray]:
    """A grid's values and observed mask from a frame, `(rows, steps)`."""
    shape = (entry["rows"], entry[steps])
    values = np.frombuffer(buffers[entry["values"]], dtype="<f4").reshape(shape)
    observed = np.frombuffer(buffers[entry["observed"]], dtype="u1").reshape(shape)
    return values, observed.view(bool)


def _series_unit(entry: dict[str, Any], buffers: list[memoryview]) -> SeriesUnit:
    values, observed = _grid(entry, buffers, "steps")
    return SeriesUnit(
        table=entry["table"],
        key=_key(entry["key"]),
        members=np.asarray(
            [int(member) for member in entry["members"]], dtype=np.uint64
        ),
        cutoff=int(entry["cutoff"]),
        values=values,
        observed=observed,
        roles=np.frombuffer(buffers[entry["roles"]], dtype="<i4"),
        action=_action(entry["action"]),
        slots=entry["slots"],
        source=_key(entry["source"]) if entry.get("source") is not None else None,
        grown=int(entry.get("grown", 0)),
    )


def _forecast(entry: dict[str, Any], buffers: list[memoryview]) -> Forecast:
    future, future_observed = _grid(entry, buffers, "horizon")
    return Forecast(
        request=entry["request"],
        table=entry["table"],
        epoch=entry["epoch"],
        unit=entry["unit"],
        target=entry["target"],
        horizon=entry["horizon"],
        future=future,
        future_observed=future_observed,
    )


@dataclass(frozen=True)
class Allocate:
    """Sizes the slot pools as the node's tiers."""

    device_slots: int
    host_slots: int
    disk_slots: int


@dataclass(frozen=True)
class Profile:
    """Asks what a model needs per context row, for a table of `features` columns."""

    model: str
    features: int


@dataclass(frozen=True)
class FitLimits:
    """Asks how many rows a model's fit takes within the call budget, for `features` columns."""

    model: str
    features: int


@dataclass(frozen=True)
class AddModel:
    """Loads a model the worker did not start with, under `name`, from `spec`.

    A control: a shard sends it in a `Controls` frame, ahead of the first
    table that names the model.
    """

    name: str
    spec: str


@dataclass(frozen=True)
class Check:
    """Asks whether a model can fit a table of this task and width."""

    model: str
    classes: int
    features: int
    # Series a forecast's group holds at most; tabular tables send 1.
    series: int


@dataclass(frozen=True)
class AddTable:
    """Serves a table; `categorical` names the columns holding category codes.

    `summary` is what the column summary is fit on: `table` or `context`.

    A table of `series` is forecast by a forecaster, its `features` being its
    covariate columns.
    """

    table: int
    model: str
    classes: int
    features: int
    categorical: tuple[int, ...]
    page_size: int = 1
    summary: str = "table"
    series: bool = False


@dataclass(frozen=True)
class RemoveTable:
    """Stops serving a table."""

    table: int


@dataclass(frozen=True)
class Apply:
    """Copies and drops between tiers."""

    movements: list[Movement]


@dataclass(frozen=True)
class Controls:
    """Models loaded, tables added and removed and tiers' copies and drops, in order.

    A shard sends what one tick asks besides its steps in one frame; the
    worker answers `Done`, or `Controlled` with the requests that failed.
    """

    requests: list[AddModel | AddTable | RemoveTable | Apply]


# The trace event a worker records around each step, with the step's number
# as `step_num`.
STEP_EVENT = "gf step"
# The step boundaries a worker writes beside its trace.
STEPS_FILE = "steps.json"


@dataclass(frozen=True)
class Record:
    """Samples the worker's threads for `seconds`, their stacks written collapsed to `path`.

    The window starts after the step in flight, if any, and `path` appears
    when it ends. With `trace`, JAX's profiler traces the same window into
    that directory, and the steps that end inside it are listed there in
    `STEPS_FILE`.
    """

    seconds: float
    path: str
    trace: str | None = None


Request = (
    Allocate
    | Profile
    | FitLimits
    | AddModel
    | Check
    | AddTable
    | RemoveTable
    | Apply
    | Controls
    | Record
    | Step
)


def request(header: dict[str, Any], buffers: list[memoryview]) -> Request:
    """The request a frame holds, checked as `gf-server`'s `WorkerRequest` is.

    Raises:
        ValueError: If the frame is no request, or its fields are not the request's.
    """
    kind = header.get("kind")
    if kind == "step":
        return step(header, buffers)
    if kind not in _REQUESTS:
        raise ValueError(f"unknown request {kind!r}")
    fields = {name: value for name, value in header.items() if name != "kind"}
    if kind == "apply":
        fields["movements"] = movements(fields.get("movements", []))
    if kind == "add_table":
        fields["categorical"] = tuple(fields.get("categorical", ()))
    if kind == "controls":
        requests = fields.get("requests")
        if type(requests) is not list or not all(
            type(control) is dict and control.get("kind") in _CONTROLS
            for control in requests
        ):
            raise ValueError(
                "a malformed controls request: its requests must be controls"
            )
        fields["requests"] = [request(control, []) for control in requests]
    try:
        return _REQUESTS[kind](**fields)
    except TypeError as error:
        raise ValueError(f"a malformed {kind} request: {error}") from None


_REQUESTS: dict[str, Any] = {
    "allocate": Allocate,
    "profile": Profile,
    "fit_limits": FitLimits,
    "add_model": AddModel,
    "check": Check,
    "add_table": AddTable,
    "remove_table": RemoveTable,
    "apply": Apply,
    "controls": Controls,
    "record": Record,
}
# The requests a `Controls` frame carries.
_CONTROLS = frozenset({"add_model", "add_table", "remove_table", "apply"})


@dataclass(frozen=True)
class Ready:
    """Sent once on connecting: the models loaded, in id order, and memory.

    Attributes:
        device_memory: Bytes of the worker's device.
        device_bytes: The device tier the worker's budget gives.
        host_bytes: The host tier its budget gives.
        row_memory_bytes: The memory for rows its budget gives.
    """

    protocol: int
    models: list[str]
    device_memory: int
    device_bytes: int
    host_bytes: int
    row_memory_bytes: int


@dataclass(frozen=True)
class Profiled:
    """What a model needs per table, for one feature count.

    For a forecaster, `bytes_per_row` is what a patch of its cache takes per
    padded row of a group and `patch_bytes` what it takes beside them;
    `outputs` is its quantile count, and the series fields describe it.
    `fit_bound` is the most rows a table's fit takes before the server asks
    for its `Limited`, or None where fits are not limited.
    """

    bytes_per_row: int
    representation_bytes: int
    regression: bool
    class_slots: int
    outputs: int
    context_rows: int | None
    state_bytes: int
    column_summary: bool
    categorical_state_bytes: int = 0
    series: bool = False
    patch_bytes: int = 0
    patch: int = 0
    output_patch: int = 0
    context_limit: int = 0
    pass_limit: int | None = None
    quantiles: list[float] | None = None
    reusable: bool = False
    max_rows: int | None = None
    extendable: bool = False
    unlabeled_context: bool = False
    fit_bound: int | None = None


@dataclass(frozen=True)
class Limited:
    """How many rows a model's fit takes within the call budget, for one feature count.

    Attributes:
        state_rows: The most rows a state alone is fit on, or None for any.
        fused_rows: The most a fit embeds in the call that fits its state,
            or None for any.
    """

    state_rows: int | None
    fused_rows: int | None


@dataclass(frozen=True)
class Done:
    """A request carried out."""


@dataclass(frozen=True)
class Warmed:
    """An allocation carried out, after compiling the programs earlier workers recorded.

    Attributes:
        compiles: Programs compiled before replying.
        seconds: Seconds they took.
    """

    compiles: int
    seconds: float


@dataclass(frozen=True)
class Progress:
    """Sent while a step runs, once per compile finished; answers no request."""


@dataclass(frozen=True)
class Compiled:
    """Sent once programs that steps deferred tables for are ready, by key; answers no request."""

    keys: list[str]


@dataclass(frozen=True)
class Invalid:
    """A check the model refuses, and why."""

    message: str


@dataclass(frozen=True)
class Failed:
    """A request that failed, and why."""

    message: str


@dataclass(frozen=True)
class ControlFailure:
    """A request of a `Controls` frame that failed: its index there, and why."""

    index: int
    message: str


@dataclass(frozen=True)
class Controlled:
    """The requests of a `Controls` frame that failed; the rest were carried out."""

    failures: list[ControlFailure]


Reply = (
    Ready
    | Profiled
    | Limited
    | Done
    | Warmed
    | Progress
    | Compiled
    | Invalid
    | Failed
    | Controlled
)

_REPLIES: dict[type, str] = {
    Ready: "ready",
    Profiled: "profile",
    Limited: "fit_limits",
    Done: "ok",
    Warmed: "warmed",
    Progress: "progress",
    Compiled: "compiled",
    Invalid: "invalid",
    Failed: "error",
    Controlled: "controlled",
}


_SERIES_FIELDS = (
    "series",
    "patch_bytes",
    "patch",
    "output_patch",
    "context_limit",
    "pass_limit",
    "quantiles",
    "reusable",
    "max_rows",
    "extendable",
)


def reply(answer: Reply) -> dict[str, Any]:
    """A reply's header, as `gf-server`'s `WorkerReply` reads it.

    A tabular model's profile leaves the series fields out, as the server
    reads it.
    """
    fields = dataclasses.asdict(answer)
    if isinstance(answer, Profiled) and not answer.series:
        fields = {
            key: value for key, value in fields.items() if key not in _SERIES_FIELDS
        }
    return {"kind": _REPLIES[type(answer)], **fields}


def predictions(
    results: dict[int, np.ndarray],
    failed: dict[int, str] | None = None,
    seconds: float = 0.0,
    stages: dict[str, tuple[int, float]] | None = None,
    parts: dict[str, tuple[int, float]] | None = None,
    deferred: dict[int, str] | None = None,
) -> tuple[dict[str, Any], list[bytes]]:
    """A frame of float32 predictions per request, and the tables that failed.

    A forecast's predictions are `(series, horizon, quantiles)` and carry
    their `horizon`; a table's are `(rows, columns)`. What the step cost
    may be set after the frame is built, with `timing`, so the building
    counts in it.

    Args:
        results: Each request's predictions.
        failed: Each failed table's error.
        deferred: Each table whose part waits for a program, by its key.
        seconds: The time the step took the worker.
        stages: The step's compiled calls and their seconds, by stage.
        parts: The step's parts, each run and their seconds; see
            `WorkerStats.parts`.

    Returns:
        The frame's header and buffers.
    """
    entries, buffers = [], []
    for request, values in results.items():
        values = np.ascontiguousarray(values, dtype="<f4")
        entry = {
            "request": request,
            "rows": values.shape[0],
            "columns": values.shape[-1],
            "buffer": len(buffers),
        }
        if values.ndim == 3:
            entry["horizon"] = values.shape[1]
        entries.append(entry)
        buffers.append(values.tobytes())
    failures = [
        {"table": table, "message": message}
        for table, message in (failed or {}).items()
    ]
    header = {
        "kind": "predictions",
        "results": entries,
        "failures": failures,
        "deferred": [
            {"table": table, "key": key} for table, key in (deferred or {}).items()
        ],
        **timing(seconds, stages, parts),
    }
    return header, buffers


def timing(
    seconds: float,
    stages: dict[str, tuple[int, float]] | None = None,
    parts: dict[str, tuple[int, float]] | None = None,
) -> dict[str, Any]:
    """The fields of a predictions header that say what the step cost; see `predictions`."""
    return {
        "seconds": seconds,
        "stages": {
            stage: {"calls": calls, "seconds": spent}
            for stage, (calls, spent) in (stages or {}).items()
        },
        "parts": {
            part: {"calls": calls, "seconds": spent}
            for part, (calls, spent) in (parts or {}).items()
        },
    }
