# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real hash-verified CPU model execution. Requires preloaded HF artifacts."""

import datetime
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from generalforecast.predictors.runtime import PredictorExecutor
from generalforecast.predictors.runtime.forecast import ForecastContext
from tests.product_runtime.test_executor import checked

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_real_tabicl_prepare_predict_and_isolated_patch(tmp_path, task):
    context = pa.table(
        {
            "id": [str(i) for i in range(8)],
            "x": [float(i) for i in range(8)],
            "y": [i >= 4 for i in range(8)]
            if task == "classification"
            else [float(i * 2) for i in range(8)],
        }
    )
    locked = checked(
        tmp_path / "first",
        context,
        id="id",
        model="general-forecast/tabicl-v2-classifier"
        if task == "classification"
        else "general-forecast/tabicl-v2-regressor",
        outputs=["label", "probabilities"]
        if task == "classification"
        else ["value", "quantiles"],
    )
    executor = PredictorExecutor()
    old = executor.prepare(locked, context)
    q = pa.table({"id": ["query"], "x": [1.5]})
    before = old.predict(q)
    assert before["id"].to_pylist() == ["query"]
    if task == "classification":
        probabilities = before["probabilities"].to_pylist()[0]
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        new_y = pa.array([not value for value in context["y"].to_pylist()])
    else:
        values = list(before["quantiles"].to_pylist()[0].values())
        assert np.isfinite(values).all() and values == sorted(values)
        new_y = pa.array([value + 100.0 for value in context["y"].to_pylist()])
    changed = context.set_column(2, "y", new_y)
    second = checked(
        tmp_path / "patch",
        changed,
        id="id",
        model=locked.spec.model,
        outputs=list(locked.outputs),
    )
    patched = executor.prepare(second, changed)
    assert patched.predict(q).to_pylist() != before.to_pylist()
    assert old.predict(q).equals(before)


def test_real_grouped_chronos_known_future_and_production_isolation(tmp_path):
    context = pa.Table.from_pylist(
        [
            {
                "id": f"{s}-{i}",
                "series": s,
                "time": i,
                "y": float(10 + i * (1 if s == "a" else 3)),
                "promo": float(i % 2),
                "past": float(i / 10),
            }
            for s in ("a", "b")
            for i in range(16)
        ]
    )
    locked = checked(
        tmp_path,
        context,
        model="general-forecast/chronos-2",
        id="id",
        time="time",
        series=["series"],
        horizon=3,
        frequency=1,
        future_inputs=["promo"],
        past_inputs=["past"],
        outputs=["value", "quantiles"],
        quantiles=[0.1, 0.5, 0.9],
    )
    old = PredictorExecutor().prepare(locked, context)
    queries = pa.Table.from_pylist(
        [
            {"id": f"q-{s}-{i}", "series": s, "time": i, "promo": float(i % 2)}
            for s in ("b", "a")
            for i in range(16, 19)
        ]
    )
    before = old.predict(queries)
    assert before.column_names == ["id", "value", "quantiles"]
    assert before["id"].equals(queries["id"])
    assert np.isfinite(before["value"].to_numpy()).all()
    changed = queries.set_column(3, "promo", pa.array([100.0, -100.0, 100.0] * 2))
    assert old.predict(changed).to_pylist() != before.to_pylist()
    assert old.predict(queries).equals(before)


def test_real_schema_free_request_multiclass_categories_and_datetime(tmp_path):
    context = pa.table(
        {
            "id": [str(i) for i in range(9)],
            "country": ["a", "b", "c"] * 3,
            "when": [datetime.date(2026, 1, i + 1) for i in range(9)],
            "y": ["low", "medium", "high"] * 3,
        }
    )
    lock = checked(tmp_path, context, context="request", id="id")
    assert lock.schema.columns == ()
    prepared = PredictorExecutor().prepare(lock, None)
    queries = pa.table(
        {
            "id": ["known", "unseen"],
            "country": ["b", "new"],
            "when": [datetime.date(2026, 2, 1), datetime.date(2026, 2, 2)],
        }
    )
    output = prepared.predict(queries, context=context)
    assert output["id"].equals(queries["id"])
    assert set(output["label"].to_pylist()) <= {"low", "medium", "high"}
    for row in output["probabilities"].to_pylist():
        assert set(row) == {"low", "medium", "high"}
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-5)
    assert prepared._context is None


def test_real_grouped_forecast_overlapping_covariates_follow_native_channels(tmp_path):
    context = pa.Table.from_pylist(
        [
            {
                "series": s,
                "time": i,
                "y": float(i + 10),
                "driver": float(i % 2),
                "zz": float(i % 3),
            }
            for s in ("a", "b")
            for i in range(16)
        ]
    )
    locked = checked(
        tmp_path,
        context,
        model="general-forecast/chronos-2",
        time="time",
        series=["series"],
        horizon=2,
        future_inputs=["driver", "zz"],
    )
    prepared = PredictorExecutor().prepare(locked, context)
    assert isinstance(prepared._context, ForecastContext)
    overlap = replace(
        prepared.lock, spec=replace(prepared.lock.spec, past_inputs=("driver",))
    )
    queries = pa.Table.from_pylist(
        [
            {"series": s, "time": i, "driver": float(i % 2), "zz": float(i % 3)}
            for s in ("b", "a")
            for i in (16, 17)
        ]
    )
    before = prepared.predict(queries)
    actual = prepared._context.predict(overlap, prepared._model, queries)
    assert actual["value"] == before["value"].to_pylist()
    changed = queries.set_column(3, "zz", pa.array([100.0, -100.0] * 2))
    assert prepared._context.predict(overlap, prepared._model, changed) != actual
    assert prepared._context.predict(overlap, prepared._model, queries) == actual


def test_real_forecast_date_axis_preserves_out_of_order_rows(tmp_path):
    context = pa.table(
        {
            "time": [datetime.date(2026, 1, i + 1) for i in range(16)],
            "y": [float(i) for i in range(16)],
        }
    )
    locked = checked(
        tmp_path,
        context,
        model="general-forecast/chronos-2",
        time="time",
        horizon=2,
        frequency="1d",
    )
    prepared = PredictorExecutor().prepare(locked, context)
    queries = pa.table(
        {"time": [datetime.date(2026, 1, 18), datetime.date(2026, 1, 17)]}
    )
    output = prepared.predict(queries)
    assert output.column_names == ["value"] and output.num_rows == 2
    assert np.isfinite(output["value"].to_numpy()).all()
    chronological = prepared.predict(queries.take(pa.array([1, 0])))
    assert output["value"].to_pylist() == chronological["value"].to_pylist()[::-1]
