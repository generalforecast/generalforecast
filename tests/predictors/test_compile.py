# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compiling a lock, or a plan, against its rows: offline, pinned and deterministic."""

import dataclasses
import socket
from typing import Any

import pyarrow as pa
import pytest

from generalforecast.predictors.models import CatalogResolver
from generalforecast.predictors.runtime import PredictorExecutor
from generalforecast.predictors.runtime.compile import (
    compile_lock,
    compile_plan,
    validate_queries,
)
from generalforecast.predictors.runtime.context import served_context
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.runtime.schema import select_context
from generalforecast.predictors.validation import PredictorError
from generalforecast.service.contracts import ServiceError
from generalforecast.tabular import table_digest
from tests.utils.locks import CLASSIFIER, directory, lock_of

ROWS = pa.table({"id": ["a", "b", "c"], "x": [1, 2, 3], "y": [False, True, False]})


def _resolver(**changes):
    """A resolver whose classifier pin has these capabilities changed."""
    pin = dataclasses.replace(CatalogResolver().resolve(CLASSIFIER), **changes)

    class Resolver:
        def resolve(self, requested, *, offline=True):
            return pin

    return Resolver()


def test_a_snapshot_lock_compiles_offline_and_deterministically(tmp_path, monkeypatch):
    lock = lock_of(directory(tmp_path, ROWS, id="id"))
    rows = served_context(lock, ROWS)
    assert rows is not None

    def forbidden(*args, **kwargs):
        pytest.fail("offline compilation attempted network or weight loading")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    plan = compile_lock(lock, rows)
    assert plan.task == "classification"
    assert plan.schema.features == ("x",)
    assert plan.schema.classes == (False, True)
    assert plan.schema.columns[0].role == "id"
    assert plan.model.resolved == f"{CLASSIFIER}@{lock.predictor['model_digest']}"
    assert compile_lock(lock, rows) == plan
    assert select_context(plan.spec, rows, plan.model).num_rows == 3
    assert plan.data_digest == table_digest(rows) == lock.data_digest
    assert plan.lock_digest == lock.digest
    digests = (plan.source_digest, plan.data_digest, plan.pin_digest, plan.plan_digest)
    assert len({*digests, plan.lock_digest}) == 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(plan, "task", "regression")  # noqa: B010
    wire = plan.to_dict()
    wire["spec"]["name"] = "changed"
    assert plan.spec.name == lock.name
    with pytest.raises(PredictorError) as caught:
        compile_lock(lock, rows.slice(0, 2))
    assert caught.value.code == "SnapshotMismatch"
    with pytest.raises(PredictorError) as caught:
        compile_lock(lock, None)
    assert caught.value.code == "ContextRequired"


def test_live_and_request_plans_bind_a_schema_not_rows(tmp_path, monkeypatch):
    live = lock_of(directory(tmp_path / "live", ROWS, context="live", id="id"))
    first = compile_lock(live, ROWS)
    other = pa.table({"id": ["d", "e"], "x": [7, 8], "y": [True, False]})
    assert compile_lock(live, other) == first
    assert first.data_digest is None and first.context_rows is None
    assert first.schema.rows is None
    request = lock_of(directory(tmp_path / "request", ROWS, context="request"))
    assert compile_lock(request, None).data_digest is None
    # The rows a live plan compiled with fix the kinds later rows must have.
    drifted = other.set_column(1, "x", pa.array([7.5, 8.5]))
    with pytest.raises(ServiceError) as caught:
        PredictorExecutor().prepare(first, drifted)
    assert caught.value.code == "SchemaChanged"


@pytest.mark.parametrize("role", ["target", "id", "time", "series"])
@pytest.mark.parametrize("inputs", ["past_inputs", "future_inputs"])
def test_forecast_inputs_cannot_take_a_reserved_role(role, inputs):
    value: dict[str, Any] = {
        "target": "y",
        role: ("reserved",) if role == "series" else "reserved",
    }
    with pytest.raises(PredictorError):
        ExecutionPlan(
            "forecast",
            "general-forecast/chronos-2",
            **value,
            **{inputs: ("reserved",)},
        )


def test_ambiguity_and_unsupported_tasks_fail():
    plan = ExecutionPlan.build("fraud", CLASSIFIER, target="y")
    both = _resolver(tasks=("classification", "regression"))
    with pytest.raises(PredictorError) as caught:
        compile_plan(plan, ROWS, resolver=both)
    assert caught.value.code == "AmbiguousTask"
    explicit = dataclasses.replace(plan, task="classification")
    assert compile_plan(explicit, ROWS, resolver=both).task == "classification"
    with pytest.raises(PredictorError) as caught:
        compile_plan(dataclasses.replace(plan, task="regression"), ROWS)
    assert caught.value.code == "UnsupportedTask"


def test_an_exact_context_past_the_model_cap_is_refused_not_truncated():
    plan = ExecutionPlan.build("fraud", CLASSIFIER, target="y")
    with pytest.raises(PredictorError) as caught:
        compile_plan(plan, ROWS, resolver=_resolver(max_context_rows=2))
    assert caught.value.code == "CapacityExceeded"


def test_query_validation_takes_the_locked_column_order(tmp_path):
    lock = lock_of(directory(tmp_path, ROWS, id="id"))
    plan = compile_lock(lock, served_context(lock, ROWS))
    assert validate_queries(plan, [{"x": 3, "id": "q"}]).column_names == ["id", "x"]
