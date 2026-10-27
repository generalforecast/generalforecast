# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Predictor execution contract tests. Injected numeric kernels are not model evidence."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import pyarrow as pa
import pytest

from generalforecast.engine import ModelOutputError
from generalforecast.predictors.runtime import PredictorExecutor
from generalforecast.predictors.selection import SelectionInfo
from generalforecast.predictors.validation import PredictorError
from generalforecast.service.contracts import ServiceError
from generalforecast.service.interfaces import PreparedExecution
from tests.utils.engine import FakeEngine
from tests.utils.locks import compiled


def checked(root, data, **changes):
    """The compiled plan of a directory checked with these manifest settings.

    `past_inputs` and `future_inputs` are a forecaster's `inputs.past` and
    `inputs.future`.
    """
    covariates = {
        key: changes.pop(f"{key}_inputs")
        for key in ("past", "future")
        if f"{key}_inputs" in changes
    }
    if covariates:
        changes["inputs"] = covariates
    return compiled(root, data, **changes)


@pytest.fixture
def data():
    return pa.table(
        {
            "id": ["a", "b", "c", "d"],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [False, False, True, True],
        }
    )


@pytest.fixture
def kernels(monkeypatch):
    return FakeEngine().install(monkeypatch).contexts


def test_prepare_warms_once_preserves_ids_outputs_and_immutable_state(
    tmp_path, data, kernels
):
    locked = checked(tmp_path, data, id="id", outputs=["probabilities"])
    prepared = PredictorExecutor().prepare(locked, data)
    assert prepared.context_rows == 4 and len(kernels) == 1
    assert isinstance(prepared, PreparedExecution) and prepared.usable
    q = pa.table({"id": ["q"], "x": [1.5]})
    assert prepared.predict(q).to_pylist() == [
        {"id": "q", "probabilities": {"false": 0.25, "true": 0.75}}
    ]
    prepared.predict(q)
    assert len(kernels) == 1
    assert prepared.predict(q).schema.metadata == {
        b"generalforecast.context_rows": b"4"
    }
    with pytest.raises(FrozenInstanceError):
        prepared.context_rows = 100  # ty: ignore[invalid-assignment]


def test_the_locked_selection_binds_and_the_full_snapshot_is_checked(
    tmp_path, data, kernels, monkeypatch
):
    fake = FakeEngine().install(monkeypatch)
    locked = checked(
        tmp_path,
        data,
        id="id",
        selection={"core": 2, "neighbors": 1, "seed": 3},
    )
    assert locked.spec.selection == SelectionInfo(
        "retrieved", 2, 1, "stratified", seed=3
    )
    prepared = PredictorExecutor().prepare(locked, data)
    assert prepared.context_rows == 4
    [strategy] = fake.strategies
    assert strategy is not None
    assert json.loads(strategy) == locked.spec.selection.strategy_spec()
    with pytest.raises(ServiceError, match="snapshot digest") as error:
        PredictorExecutor().prepare(locked, data.slice(1))
    assert error.value.code == "SnapshotMismatch"


def test_dev_replacement_does_not_mutate_existing_context(tmp_path, data, kernels):
    first = checked(tmp_path / "first", data, id="id")
    old = PredictorExecutor().prepare(first, data)
    changed = data.set_column(1, "x", pa.array([10.0, 11.0, 12.0, 13.0]))
    second = checked(tmp_path / "second", changed, id="id")
    new = PredictorExecutor().prepare(second, changed)
    assert old.lock != new.lock and old._context is not new._context
    assert kernels[0][:, 0].tolist() == [0, 1, 2, 3]
    assert kernels[1][:, 0].tolist() == [10, 11, 12, 13]
    assert old.predict(pa.table({"x": [1.0]})).num_rows == 1
    with pytest.raises(ServiceError):
        PredictorExecutor().prepare(second, data)
    assert old.context_rows == 4


@pytest.mark.parametrize("mode", ["request", "live"])
def test_dynamic_context_validation_and_no_override(tmp_path, data, kernels, mode):
    locked = checked(tmp_path, data, context=mode, id="id")
    executor = PredictorExecutor()
    prepared = executor.prepare(locked, None if mode == "request" else data)
    q = pa.table({"x": [1.0]})
    if mode == "request":
        with pytest.raises(ServiceError) as error:
            prepared.predict(q)
        assert error.value.code == "ContextRequired"
        prepared.predict(q, context=data)
        prepared.predict(q, context=data)
        assert len(kernels) == 2 and prepared._context is None
    else:
        with pytest.raises(ServiceError):
            prepared.predict(q, context=data)
        assert len(kernels) == 1


@pytest.mark.parametrize(
    "queries",
    [{"x": ["1"]}, {"x": [None]}, {"x": [1.0], "y": [True]}, {"unknown": [1.0]}],
)
def test_queries_reject_physical_drift_missing_leakage_and_unknown(
    tmp_path, data, kernels, queries
):
    locked = checked(tmp_path, data, id="id")
    prepared = PredictorExecutor().prepare(locked, data)
    with pytest.raises(ServiceError) as error:
        prepared.predict(pa.table(queries))
    assert error.value.status == 422


def test_forged_capability_rehashed_plan_still_fails_admission(tmp_path, data, kernels):
    locked = checked(tmp_path, data, id="id")
    forged = replace(
        locked,
        model=replace(locked.model, max_context_rows=locked.model.max_context_rows + 1),
    )
    with pytest.raises(ServiceError) as error:
        PredictorExecutor().prepare(forged, data)
    assert error.value.code == "ModelMismatch" and kernels == []


def test_missing_bound_context_and_unexpected_request_context(tmp_path, data, kernels):
    locked = checked(tmp_path, data, id="id")
    with pytest.raises(ServiceError):
        PredictorExecutor().prepare(locked, None)
    locked = checked(tmp_path, data, context="request", id="id")
    with pytest.raises(ServiceError):
        PredictorExecutor().prepare(locked, data)


def test_actual_labeled_rows_metadata_for_request_selection(tmp_path, data, kernels):
    data = data.set_column(2, "y", pa.array([None, False, True, True]))
    locked = checked(tmp_path, data, context="request", id="id")
    prepared = PredictorExecutor().prepare(locked, None)
    result = prepared.predict(pa.table({"x": [1.0]}), context=data)
    assert result.schema.metadata == {b"generalforecast.context_rows": b"3"}
    assert prepared.context_rows is None


def test_query_intermediates_batched_and_capacity_rejected_not_sampled(
    tmp_path, data, kernels
):
    locked = checked(tmp_path, data, id="id")
    prepared = PredictorExecutor().prepare(locked, data)
    q = pa.table({"id": [str(i) for i in range(257)], "x": [1.0] * 257})
    result = prepared.predict(q)
    assert result.num_rows == 257 and result["id"].equals(q["id"])
    assert len(kernels) == 1
    with pytest.raises(ServiceError) as error:
        prepared.predict(pa.table({"x": [1.0] * 10_001}))
    assert error.value.code == "CapacityExceeded"


def test_schema_free_request_and_regression_projection(tmp_path, kernels):
    data = pa.table({"x": [1.0, 2.0], "y": [3.0, 5.0]})
    locked = checked(
        tmp_path,
        data,
        context="request",
        model="general-forecast/tabicl-v2-regressor",
        outputs=["quantiles", "value"],
        quantiles=[0.1, 0.5, 0.9],
    )
    assert not locked.schema.columns
    prepared = PredictorExecutor().prepare(locked, None)
    result = prepared.predict(pa.table({"x": [1.5]}), context=data)
    assert result.column_names == ["quantiles", "value"]
    assert result.to_pylist() == [
        {"quantiles": {"0.1": 103.0, "0.5": 503.0, "0.9": 903.0}, "value": 503.0}
    ]


def test_concurrent_request_contexts_do_not_mutate_bound_production(tmp_path, kernels):
    first = pa.table({"x": [1.0, 2.0], "y": [3.0, 5.0]})
    second = first.set_column(1, "y", pa.array([7.0, 9.0]))
    request = checked(
        tmp_path / "request",
        first,
        context="request",
        model="general-forecast/tabicl-v2-regressor",
    )
    snapshot = checked(
        tmp_path / "snapshot", first, model="general-forecast/tabicl-v2-regressor"
    )
    executor = PredictorExecutor()
    remote = executor.prepare(request, None)
    production = executor.prepare(snapshot, first)
    queries = pa.table({"x": [1.5]})

    def predict(i):
        result = remote.predict(queries, context=first if i % 2 == 0 else second)
        return result["value"][0].as_py()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(predict, range(8))) == [503.0, 507.0] * 4
    assert production.predict(queries)["value"][0].as_py() == 503.0
    assert remote._context is None and remote.context_rows is None
    assert production.context_rows == 2


def test_a_request_plan_without_a_target_refuses_its_rows(tmp_path, data, kernels):
    locked = checked(tmp_path, data, context="request")
    lock = replace(locked, spec=replace(locked.spec, target=None))
    prepared = PredictorExecutor().prepare(lock, None)
    with pytest.raises(ServiceError, match="explicit target"):
        prepared.predict(pa.table({"x": [1.0]}), context=data)


@pytest.mark.parametrize("context", ["snapshot", "live"])
def test_a_bound_plan_names_its_target(tmp_path, data, kernels, context):
    locked = checked(tmp_path, data, context=context)
    with pytest.raises(PredictorError, match="explicit target"):
        replace(locked.spec, target=None)


def test_integer_queries_widen_to_a_float_input(tmp_path, data, kernels):
    locked = checked(tmp_path, data, id="id")
    prepared = PredictorExecutor().prepare(locked, data)
    assert prepared.predict(pa.table({"x": [1]})).num_rows == 1


def test_bad_native_output_and_private_error_are_typed(
    tmp_path, data, kernels, monkeypatch
):
    locked = checked(tmp_path, data, id="id")
    prepared = PredictorExecutor().prepare(locked, data)
    monkeypatch.undo()

    def invalid(*_args):
        raise ModelOutputError("the model returned invalid outputs")

    engine, _ = prepared._model
    monkeypatch.setattr(engine, "predict", invalid)
    monkeypatch.setattr(engine, "tabular", invalid)
    with pytest.raises(ServiceError) as error:
        prepared.predict(pa.table({"x": [1.0]}))
    assert error.value.code == "ModelOutputError"

    def private(*args):
        raise RuntimeError("credential=must-not-leak")

    monkeypatch.setattr("generalforecast.predictors.runtime._load", private)
    with pytest.raises(ServiceError) as error:
        PredictorExecutor().prepare(locked, data)
    assert error.value.code == "InferenceFailed" and "credential" not in str(
        error.value
    )
