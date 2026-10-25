# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compiles an execution plan against its context into the lock the runtime binds."""

import os
from pathlib import Path
from typing import Any

import pyarrow as pa

from generalforecast.predictors.files import MAX_DATA_BYTES, parse_yaml, read
from generalforecast.predictors.models import CatalogResolver, ModelPin, ModelResolver
from generalforecast.predictors.runtime.capabilities import admit
from generalforecast.predictors.runtime.lock import ExecutionLock
from generalforecast.predictors.runtime.plan import ExecutionPlan, parse_plan
from generalforecast.predictors.runtime.schema import (
    infer_schema,
    select_context,
)
from generalforecast.predictors.validation import digest, ensure
from generalforecast.tabular import (
    file_format,
    physical_type,
    read_table,
    table_digest,
    to_arrow,
)


def compile_plan(
    plan: ExecutionPlan,
    table: pa.Table | None = None,
    *,
    resolver: ModelResolver | None = None,
    offline: bool = True,
    previous: ExecutionLock | None = None,
    upgrade: bool = False,
) -> ExecutionLock:
    """The lock for `plan` bound to `table`: its model pin, schema and digests.

    `table` is the snapshot or captured live context, or None for a plan whose
    context comes with each request. `previous` keeps an earlier lock's model
    pin for the same model and refuses a changed schema unless `upgrade`.
    """
    if previous is not None and previous.spec.model == plan.model and not upgrade:
        pin = previous.model
        if not offline:
            resolved = (resolver or CatalogResolver()).resolve(
                pin.resolved, offline=False
            )
            ensure(
                resolved == pin,
                "online model capabilities differ from locked metadata; explicit upgrade required",
                "ModelMismatch",
            )
    else:
        pin = (resolver or CatalogResolver()).resolve(plan.model, offline=offline)
    pin = ModelPin.from_dict(pin.to_dict())
    task = plan.task
    if task is None:
        ensure(
            len(pin.tasks) == 1,
            "model supports multiple tasks; declare task explicitly",
            "AmbiguousTask",
        )
        task = pin.tasks[0]
    outputs = plan.outputs or (
        ("label", "probabilities") if task == "classification" else ("value",)
    )
    admit(plan, pin, task, outputs)
    schema = infer_schema(plan, pin, task, table)
    if previous is not None and previous.spec == plan and not upgrade:
        ensure(
            (
                previous.schema.columns,
                previous.schema.classes,
                previous.schema.frequency,
            )
            == (schema.columns, schema.classes, schema.frequency),
            "schema differs from the previous lock; explicit upgrade is required",
            "SchemaChanged",
        )
    selected = select_context(plan, table, pin) if table is not None else None
    context_rows = (
        selected.num_rows
        if selected is not None and plan.context.mode == "snapshot"
        else None
    )
    data_digest = (
        table_digest(table)
        if table is not None and plan.context.mode == "snapshot"
        else None
    )
    return ExecutionLock.build(
        spec=plan,
        model=pin,
        task=task,
        schema=schema,
        source_digest=digest({"spec": plan.to_dict()}),
        data_digest=data_digest,
        outputs=outputs,
        context_rows=context_rows,
    )


def plan_from_dir(
    root: str | os.PathLike = ".",
) -> tuple[ExecutionPlan, pa.Table | None]:
    """A directory's `gf.yaml` as a plan, with its snapshot file read when declared."""
    base = Path(os.path.abspath(root))
    plan = parse_plan(parse_yaml(read(base, "gf.yaml")))
    table = None
    if plan.context.mode == "snapshot":
        assert plan.context.source is not None
        raw = read(base, plan.context.source, limit=MAX_DATA_BYTES)
        table = read_table(raw, file_format(plan.context.source))
    return plan, table


def compile_dir(root: str | os.PathLike = ".", **options: Any) -> ExecutionLock:
    """`compile_plan` over a directory's plan and snapshot."""
    plan, table = plan_from_dir(root)
    return compile_plan(plan, table, **options)


def validate_queries(lock: ExecutionLock, data: Any) -> pa.Table:
    """Validates a locked tabular request without silently redefining the schema."""
    ensure(
        bool(lock.schema.columns),
        "request-context predictors validate their schema per request",
        "SchemaRequired",
    )
    ensure(
        lock.task != "forecast",
        "forecast request alignment requires the series runtime validator",
        "UnsupportedCapability",
    )
    table = to_arrow(data)
    expected = {
        column.name: column
        for column in lock.schema.columns
        if column.role in ("input", "id")
    }
    required = {column.name for column in expected.values() if column.role == "input"}
    ensure(
        required <= set(table.column_names) <= expected.keys(),
        "query columns differ from locked inputs",
    )
    for field, array in zip(table.schema, table.columns, strict=True):
        column = expected[field.name]
        ensure(
            physical_type(field.type) == column.type,
            "query physical type differs from lock",
        )
        if column.role == "input":
            ensure(
                not array.null_count
                or (
                    lock.model.missing_values
                    and lock.spec.inputs.missing_values == "allow"
                ),
                "query has unsupported missing values",
            )
    return table.select([column for column in expected if column in table.column_names])


__all__ = ["compile_dir", "compile_plan", "plan_from_dir", "validate_queries"]
