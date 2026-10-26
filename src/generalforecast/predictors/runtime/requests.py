# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request-scoped predictor inputs shared by the hosted API and `gf serve`."""

from collections.abc import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from generalforecast.predictors.runtime.compile import compile_plan
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.validation import PredictorError
from generalforecast.service.contracts import ServiceError, require
from generalforecast.tabular import coerce, to_arrow


def request_table(data, lock, *, max_rows):
    """Keeps other input types strict; ISO strings are the JSON temporal encoding."""
    table = to_arrow(data, max_rows=max_rows)
    return temporal(table, {column.name: column.type for column in lock.schema.columns})


def temporal(table: pa.Table, types: Mapping[str, str]) -> pa.Table:
    """Restores the date and timestamp columns JSON carried as ISO strings.

    A time without a zone, such as `2026-01-01` or `2026-01-01T05:00:00`,
    is read as UTC; other declared types are left to the schema check.
    """
    declared = {
        column: kind
        for column, kind in types.items()
        if kind in ("date", "timestamp")
        and column in table.column_names
        and pa.types.is_string(table[column].type)
    }
    for column, kind in declared.items():
        if kind != "timestamp":
            continue
        index = table.column_names.index(column)
        try:
            naive = pc.cast(table.column(index), pa.timestamp("us"), safe=True)
        except (pa.ArrowException, ValueError):
            continue
        table = table.set_column(
            index, table.schema.field(index).with_type(naive.type), naive
        )
    try:
        return coerce(table, declared)
    except PredictorError:
        raise ServiceError(
            "ValidationError", "invalid ISO temporal input", 422
        ) from None


def compile_request(authored, context):
    """Compiles the authored plan against the request's context; writes nothing."""
    return compile_plan(ExecutionPlan.from_dict(authored), context, offline=True)


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
