# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compiles a lock, or a plan, against its context rows into the plan the runtime binds."""

from dataclasses import replace
from typing import Any

import pyarrow as pa

from generalforecast.predictors.models import CatalogResolver, ModelResolver
from generalforecast.predictors.runtime.capabilities import admit
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.context import conform
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.runtime.schema import (
    infer_schema,
    observed_frequency,
    select_context,
)
from generalforecast.predictors.validation import ensure
from generalforecast.tabular import physical_type, table_digest, to_arrow


def compile_plan(
    plan: ExecutionPlan,
    table: pa.Table | None = None,
    *,
    resolver: ModelResolver | None = None,
    offline: bool = True,
    lock_digest: str | None = None,
) -> CompiledPlan:
    """The compiled plan for `plan` bound to `table`: its model pin, schema and digests.

    `table` is the snapshot or captured live context, or None for a plan whose
    context comes with each request. A forecaster's frequency, when the plan
    declares none, is inferred from the rows by `frequency.infer` and kept on
    the compiled plan.
    """
    pin = (resolver or CatalogResolver()).resolve(plan.model, offline=offline)
    task = plan.task
    if task is None:
        ensure(
            len(pin.tasks) == 1,
            "model supports multiple tasks; declare task explicitly",
            "AmbiguousTask",
        )
        task = pin.tasks[0]
    plan = replace(plan.with_outputs(task), task=task)
    if task == "forecast" and table is not None and plan.frequency is None:
        plan = replace(plan, frequency=observed_frequency(plan, to_arrow(table)))
    admit(plan, pin, task, plan.outputs)
    schema = infer_schema(plan, pin, task, table)
    selected = select_context(plan, table, pin) if table is not None else None
    snapshot = selected is not None and plan.context.mode == "snapshot"
    return CompiledPlan.build(
        spec=plan,
        model=pin,
        task=task,
        schema=schema,
        data_digest=table_digest(table) if snapshot else None,
        outputs=plan.outputs,
        context_rows=selected.num_rows if snapshot and selected is not None else None,
        lock_digest=lock_digest,
    )


def compile_lock(
    lock: Any,
    table: pa.Table | None = None,
    *,
    outputs: tuple[str, ...] | list[str] | None = None,
    resolver: ModelResolver | None = None,
    manifest: dict[str, Any] | None = None,
) -> CompiledPlan:
    """The compiled plan for a `gf.lock` and its context rows.

    Every lock is held to its rows (a snapshot's row count, its task's
    outputs) and, given the normalized manifest `gf push` sends with it, to
    that manifest: `resolution.pushed_admission`, the hosted counterpart of
    the `verified` admission every local entry point applies.

    Args:
        lock: A `contracts.PredictorLock`, or a version 2 lock document.
        table: For a snapshot, the context rows the lock's context digest
            names (`context.served_context` reads them from a snapshot file);
            for a live context, its captured rows; None for a request
            context, whose rows come with each call.
        outputs: A call's own outputs, for an ad hoc request; the lock's
            otherwise. Outputs without `quantiles` predict no quantile
            levels, whatever levels the lock records.
        resolver: The trusted catalog; the built-in one by default.
        manifest: The normalized `gf.yaml` the lock was written for, as
            `gf push` sends it; a pushed lock is held to it field by field.

    Raises:
        PredictorError: `SnapshotMismatch` for rows other than the lock's,
            `SchemaChanged` for rows whose kinds differ from the lock's, or
            the admission and schema refusals of `compile_plan`.
    """
    from generalforecast.contracts import PredictorLock, parse_lock
    from generalforecast.predictors.resolution import pushed_admission

    locked = lock if isinstance(lock, PredictorLock) else parse_lock(lock)
    pushed_admission(locked.to_dict(), manifest, table)
    plan = ExecutionPlan.from_lock(locked)
    if outputs is not None:
        plan = replace(plan, outputs=tuple(outputs))
        if "quantiles" not in plan.outputs:
            plan = replace(plan, quantiles=())
    context = locked.context
    if locked.context_kind == "snapshot":
        ensure(
            table is not None,
            "a snapshot lock compiles with its rows",
            "ContextRequired",
        )
        assert table is not None
        ensure(
            table_digest(table) == context["digest"],
            "snapshot digest differs from lock",
            "SnapshotMismatch",
        )
    compiled = compile_plan(plan, table, resolver=resolver, lock_digest=locked.digest)
    pin = locked.predictor["model_digest"]
    ensure(
        compiled.model.digest == pin,
        "the catalog's model digest differs from the lock's; run `gf check --upgrade`",
        "ModelMismatch",
    )
    _agrees(locked.predictor, compiled)
    return compiled


def _agrees(predictor: dict, compiled: CompiledPlan) -> None:
    """Refuses a compiled schema whose classes or frequency differ from the lock's."""
    if predictor.get("classes") is not None and compiled.schema.classes:
        ensure(
            list(compiled.schema.classes) == list(predictor["classes"]),
            "context classes differ from lock",
            "SchemaChanged",
        )
    if predictor.get("frequency") is not None:
        ensure(
            compiled.spec.frequency == predictor["frequency"],
            "context frequency differs from lock",
            "SchemaChanged",
        )


def validate_queries(plan: CompiledPlan, data: Any) -> pa.Table:
    """Validates a compiled tabular plan's queries without redefining its schema."""
    ensure(
        bool(plan.schema.columns),
        "request-context predictors validate their schema per request",
        "SchemaRequired",
    )
    ensure(
        plan.task != "forecast",
        "forecast request alignment requires the series runtime validator",
        "UnsupportedCapability",
    )
    table = to_arrow(data)
    expected = {
        column.name: column
        for column in plan.schema.columns
        if column.role in ("input", "id")
    }
    required = {column.name for column in expected.values() if column.role == "input"}
    ensure(
        required <= set(table.column_names) <= expected.keys(),
        "query columns differ from locked inputs",
    )
    for index, (field, array) in enumerate(
        zip(table.schema, table.columns, strict=True)
    ):
        column = expected[field.name]
        conformed = conform(array, column.type)
        ensure(
            conformed is not None,
            f"query column {field.name!r} is {physical_type(field.type)}; "
            f"the lock reads {column.type}",
        )
        table = table.set_column(index, field.name, conformed)
        array = conformed
        if column.role == "input":
            ensure(
                not array.null_count
                or (
                    plan.model.missing_values
                    and plan.spec.inputs.missing_values == "allow"
                ),
                "query has unsupported missing values",
            )
    return table.select([column for column in expected if column in table.column_names])


__all__ = ["compile_lock", "compile_plan", "validate_queries"]
