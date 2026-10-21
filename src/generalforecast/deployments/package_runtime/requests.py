"""Request-scoped package inputs shared by the hosted API and `gf serve`."""

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pyarrow as pa

from generalforecast.contracts import ContextVersion, PredictionResult
from generalforecast.deployments.boundary import JsonObject
from generalforecast.deployments.contracts import ServiceError, fields, require
from generalforecast.deployments.interfaces import PreparedExecution
from generalforecast.deployments.package_runtime.validation import MAX_QUERY_ROWS
from generalforecast.packages import Package, PackageLock, PackageSpec, PythonLock
from generalforecast.packages.validation import PackageError
from generalforecast.tabular import coerce, to_arrow

ONE_SHOT_OPTIONS = frozenset(
    {
        "id",
        "time",
        "series",
        "horizon",
        "frequency",
        "past_inputs",
        "future_inputs",
        "quantiles",
        "outputs",
    }
)


def request_table(data, lock, *, max_rows):
    """Keeps other input types strict; ISO strings are the JSON temporal encoding."""
    table = to_arrow(data, max_rows=max_rows)
    if isinstance(lock, PythonLock):
        return table
    try:
        return coerce(
            table,
            {
                column.name: column.type
                for column in lock.schema.columns
                if column.name in table.column_names
                and column.type in {"date", "timestamp"}
                and pa.types.is_string(table[column.name].type)
            },
        )
    except PackageError:
        raise ServiceError(
            "ValidationError", "invalid ISO temporal input", 422
        ) from None


def compile_request(authored, context):
    """Uses the standard offline compiler; writes no data or executable files."""
    PackageSpec.from_dict(authored)
    with tempfile.TemporaryDirectory(prefix="gf-request-") as root:
        (Path(root) / "gf.yaml").write_text(json.dumps(authored))
        package = Package.load(root).check(data=context, write=False, offline=True)
    assert package.lock is not None
    return package.lock


def bind_request(lock, payload, context):
    """Binds explicit per-call roles only when no physical schema was locked."""
    authored = lock.spec.to_dict()
    target = payload.get("target", lock.spec.target)
    require(type(target) is str, "request target is required")
    require(
        lock.spec.target is None or target == lock.spec.target,
        "target differs from lock",
    )
    authored["target"] = (
        {"column": target, "type": lock.spec.target_type}
        if lock.spec.target_type is not None
        else target
    )
    if "features" in payload:
        features = payload["features"]
        require(type(features) is list, "features must be a list")
        inputs = authored.get("inputs", {})
        require(not inputs.get("exclude"), "features differ from lock")
        require(
            not inputs.get("include") or features == inputs["include"],
            "features differ from lock",
        )
        authored["inputs"] = {**inputs, "include": features}
        authored["inputs"].pop("exclude", None)
    return compile_request(authored, context)


def one_shot_lock(payload: JsonObject) -> tuple[PackageLock, pa.Table]:
    """Compiles the Models API `/predictions` body into a private request package."""
    fields(
        payload,
        {"model", "task", "target", "features", "context", "queries", "options"},
    )
    options = fields(payload["options"], set(), set(ONE_SHOT_OPTIONS))
    require(type(payload["features"]) is list, "features must be a list")
    require(
        type(payload["queries"]) is list
        and 0 < len(payload["queries"]) <= MAX_QUERY_ROWS,
        "queries must contain 1..10000 rows",
    )
    authored = {
        "name": "request",
        "model": payload["model"],
        "task": payload["task"],
        "target": payload["target"],
        "context": "request",
        "inputs": {"include": payload["features"]},
        **options,
    }
    context = to_arrow(payload["context"])
    lock = compile_request(authored, context)
    assert isinstance(lock, PackageLock)
    return lock, context


def one_shot(
    payload: JsonObject, prepare: Callable[[PackageLock], PreparedExecution]
) -> JsonObject:
    """Runs a `/predictions` body; never writes context, queries or results."""
    lock, context = one_shot_lock(payload)
    prepared = prepare(lock)
    table = prepared.predict(
        request_table(payload["queries"], lock, max_rows=MAX_QUERY_ROWS),
        context=context,
    )
    return prediction(table, lock, context.num_rows)


def prediction(
    table: pa.Table,
    lock: PackageLock,
    rows: int,
    *,
    source: str = "request",
    version: str | int | None = None,
) -> JsonObject:
    """The shared `PredictionResult` wire value, with the context rows actually used."""
    metadata = ContextVersion(source, version, rows=rows, as_of=time.time())
    used_rows = (table.schema.metadata or {}).get(b"generalforecast.context_rows")
    if used_rows is not None:
        metadata = replace(metadata, rows=int(used_rows))
    return PredictionResult(table, lock.model.resolved, lock.task, metadata).to_dict()
