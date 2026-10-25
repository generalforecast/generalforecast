# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request-scoped predictor inputs shared by the hosted API and `gf serve`."""

import pyarrow as pa

from generalforecast.predictors.runtime.compile import compile_plan
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.validation import PredictorError
from generalforecast.service.contracts import ServiceError, require
from generalforecast.tabular import coerce, to_arrow


def request_table(data, lock, *, max_rows):
    """Keeps other input types strict; ISO strings are the JSON temporal encoding."""
    table = to_arrow(data, max_rows=max_rows)
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
