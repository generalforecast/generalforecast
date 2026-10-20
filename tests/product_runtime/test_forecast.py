"""Forecast alignment contract tests with explicitly substituted numeric kernels."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from generalforecast.deployments.contracts import ServiceError
from generalforecast.deployments.package_runtime import PackageExecutor
from generalforecast.models.catalog import CHRONOS2_QUANTILES
from tests.product_runtime.test_executor import checked


@pytest.fixture
def context():
    return pa.Table.from_pylist(
        [
            {
                "series": s,
                "time": i,
                "y": float(i * (1 if s == "a" else 2)),
                "promo": float(i % 2),
            }
            for s in ("a", "b")
            for i in range(4)
        ]
    )


@pytest.fixture
def engine(tmp_path, context, monkeypatch):
    properties = SimpleNamespace(patch=16, quantiles=CHRONOS2_QUANTILES)
    model = (SimpleNamespace(properties=lambda _: properties), None)
    monkeypatch.setattr(
        "generalforecast.deployments.package_runtime._load", lambda *a: model
    )

    def run(model, group, horizon):
        result = np.broadcast_to(
            np.asarray(group.future).sum() + np.arange(21),
            (group.values.shape[0], horizon, 21),
        ).copy()
        return result

    monkeypatch.setattr(
        "generalforecast.deployments.package_runtime.forecast._run", run
    )
    package = checked(
        tmp_path,
        context,
        model="chronos-2",
        time="time",
        series=["series"],
        horizon=2,
        future_inputs=["promo"],
        outputs=["value", "quantiles"],
    )
    return PackageExecutor().prepare(package.lock, context)


def queries():
    return pa.table(
        {
            "series": ["a", "a", "b", "b"],
            "time": [4, 5, 4, 5],
            "promo": [0.0, 1.0, 0.0, 1.0],
        }
    )


def test_known_future_rows_order_and_shared_values(engine):
    q = queries().take(pa.array([3, 0, 2, 1]))
    assert engine.predict(q).num_rows == 4
    changed = q.set_column(2, "promo", pa.array([2.0, 0.0, 0.0, 2.0]))
    assert engine.predict(changed).to_pylist() != engine.predict(q).to_pylist()
    assert engine.predict(q).equals(engine.predict(q))


def test_overlapping_past_future_covariates_use_native_unique_channel_order(
    tmp_path, context, engine
):
    context = context.append_column("zz", pa.array([2.0] * context.num_rows))
    package = checked(
        tmp_path,
        context,
        model="chronos-2",
        time="time",
        series=["series"],
        horizon=2,
        future_inputs=["promo", "zz"],
    )
    prepared = PackageExecutor().prepare(package.lock, context)
    assert prepared._context is not None
    # Isolate runtime ordering from the separately owned manifest role validator.
    overlap = replace(
        prepared.lock, spec=replace(prepared.lock.spec, past_inputs=("promo",))
    )
    query = queries().append_column("zz", pa.array([3.0] * 4))
    expected = prepared._context.predict(prepared.lock, prepared._model, query)
    assert prepared._context.predict(overlap, prepared._model, query) == expected


@pytest.mark.parametrize(
    "case",
    [
        "incomplete",
        "duplicate",
        "past",
        "outside",
        "unknown_series",
        "target_leak",
        "future_disagrees",
        "future_missing",
        "wrong_type",
    ],
)
def test_bad_forecast_queries_are_typed(engine, case):
    q = queries()
    if case == "incomplete":
        q = q.slice(0, 1)
    elif case == "duplicate":
        q = q.set_column(1, "time", pa.array([4, 4, 4, 5]))
    elif case == "past":
        q = q.set_column(1, "time", pa.array([3, 5, 4, 5]))
    elif case == "outside":
        q = q.set_column(1, "time", pa.array([4, 6, 4, 5]))
    elif case == "unknown_series":
        q = q.set_column(0, "series", pa.array(["a", "a", "c", "c"]))
    elif case == "target_leak":
        q = q.append_column("y", pa.array([1.0] * 4))
    elif case == "future_disagrees":
        q = q.set_column(2, "promo", pa.array([0.0, 1.0, 2.0, 1.0]))
    elif case == "future_missing":
        q = q.drop(["promo"])
    else:
        q = q.set_column(1, "time", pa.array([4.0, 5.0, 4.0, 5.0]))
    with pytest.raises(ServiceError) as error:
        engine.predict(q)
    assert error.value.status == 422


def test_irregular_context_and_covariate_disagreement_fail_before_ready(
    tmp_path, context, monkeypatch
):
    properties = SimpleNamespace(patch=16, quantiles=CHRONOS2_QUANTILES)
    model = (SimpleNamespace(properties=lambda _: properties), None)
    monkeypatch.setattr(
        "generalforecast.deployments.package_runtime._load", lambda *a: model
    )
    package = checked(
        tmp_path,
        context,
        context="dataset/events",
        model="chronos-2",
        time="time",
        series=["series"],
        horizon=2,
        future_inputs=["promo"],
    )
    changed = context.set_column(1, "time", pa.array([0, 1, 2, 4, 0, 1, 2, 3]))
    with pytest.raises(ServiceError) as error:
        PackageExecutor().prepare(package.lock, changed)
    assert error.value.code == "AmbiguousFrequency"
    changed = context.set_column(3, "promo", pa.array([0.0] * 4 + [1.0] * 4))
    with pytest.raises(ServiceError) as error:
        PackageExecutor().prepare(package.lock, changed)
    assert error.value.code == "UnsupportedCovariates"
