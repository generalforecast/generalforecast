# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request-scoped predictor inputs shared by the hosted API and `gf serve`."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from generalforecast.predictors.runtime.compile import compile_plan
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.validation import PredictorError
from generalforecast.service.contracts import ServiceError, require
from generalforecast.tabular import coerce, to_arrow


def request_table(data: Any, plan: CompiledPlan, *, max_rows: int) -> pa.Table:
    """A request's rows, with the dates and times JSON carries as text restored."""
    table = to_arrow(data, max_rows=max_rows)
    return temporal(table, {column.name: column.type for column in plan.schema.columns})


def temporal(table: pa.Table, types: Mapping[str, str]) -> pa.Table:
    """Restores the date and timestamp columns JSON carried as ISO strings.

    A time without a zone, such as `2026-01-01` or `2026-01-01T05:00:00`,
    is read as UTC, and one with a zone is converted to UTC; other declared
    types are left to the schema check.
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
        values = table.column(index)
        try:
            naive = pc.cast(values, pa.timestamp("us"), safe=True)
        except (pa.ArrowException, ValueError):
            try:
                zoned = pc.cast(values, pa.timestamp("us", tz="UTC"), safe=True)
            except (pa.ArrowException, ValueError):
                continue
            naive = pc.cast(zoned, pa.timestamp("us"))
        table = table.set_column(
            index, table.schema.field(index).with_type(naive.type), naive
        )
    try:
        return coerce(
            table, {key: kind for key, kind in declared.items() if kind == "date"}
        )
    except PredictorError:
        raise ServiceError(
            "ValidationError", "invalid ISO temporal input", 422
        ) from None


def bind_request(
    plan: CompiledPlan, payload: Mapping[str, Any], context: pa.Table
) -> CompiledPlan:
    """A schema-free request plan compiled against one call's context rows.

    The call's `target` and `features`, when it names them, must be the
    lock's; the rows give the schema.
    """
    target = payload.get("target", plan.spec.target)
    require(type(target) is str, "request target is required")
    require(
        plan.spec.target is None or target == plan.spec.target,
        "target differs from lock",
    )
    spec = plan.spec
    if "features" in payload:
        features = payload["features"]
        require(type(features) is list, "features must be a list")
        require(
            spec.inputs.include is None or tuple(features) == spec.inputs.include,
            "features differ from lock",
        )
        spec = replace(spec, inputs=replace(spec.inputs, include=tuple(features)))
    return compile_plan(
        replace(spec, target=target), context, lock_digest=plan.lock_digest
    )


__all__ = ["bind_request", "request_table", "temporal"]
