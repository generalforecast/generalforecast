"""Real optimized engine calls with tiny random weights, not catalog/cloud proof."""

import pickle

import numpy as np
import pyarrow as pa
import pytest
from sklearn.model_selection import cross_val_score

import generalforecast
from generalforecast.estimators import Forecaster, Regressor
from generalforecast.models.chronos2.model import chronos2_functions
from generalforecast.models.tabicl.model import tabicl_functions
from generalforecast.packages import PackageError
from tests.models.chronos2.tiny import tiny_chronos2
from tests.models.tabicl.tiny import tiny_tabicl

pytestmark = pytest.mark.slow


def test_regressor_cross_validation_uses_fold_local_context():
    features = np.random.default_rng(0).normal(size=(24, 3)).astype(np.float32)
    target = features.sum(axis=1)
    estimator = Regressor((tabicl_functions, tiny_tabicl(regression=True)))
    scores = cross_val_score(estimator, features, target, cv=2)
    assert scores.shape == (2,) and np.isfinite(scores).all()
    assert not estimator.__sklearn_is_fitted__()


@pytest.mark.parametrize(
    "covariates,shared", [(False, False), (True, False), (True, True)]
)
def test_forecaster_matches_native_grouped_series_and_preserves_context(
    tmp_path, covariates, shared
):
    model = (chronos2_functions, tiny_chronos2())
    context = pa.table(
        {
            "store": ["a"] * 8 + ["b"] * 8,
            "time": list(range(8)) * 2,
            "sales": list(np.arange(16, dtype=float)),
        }
    )
    future = pa.table({"store": ["b", "a", "b", "a"], "time": [9, 8, 8, 9]})
    if covariates:
        context = context.append_column(
            "price", pa.array(np.tile(np.arange(8, dtype=float) * 0.5, 2))
        )
        context = context.append_column("weather", pa.array(np.ones(16)))
        future = future.append_column("price", pa.array([13.0, 11.0, 11.0, 13.0]))
    estimator = Forecaster(
        model,
        target="sales",
        time="time",
        series="store",
        horizon=2,
        past_inputs=("weather", "price")
        if shared
        else ("weather",)
        if covariates
        else (),
        future_inputs=("price",) if covariates else (),
    ).fit(context)
    original_count = len(estimator.table_)
    quantiles = estimator.predict_quantiles(future)
    assert quantiles.shape == (4, 3) and np.isfinite(quantiles).all()
    assert len(estimator.table_) == original_count
    assert estimator.context_.equals(context)
    np.testing.assert_allclose(estimator.predict(future), quantiles[:, 1], atol=1e-6)
    client = generalforecast.local()
    table = client.create_series_table(
        "independent",
        model=model,
        series="store",
        time="time",
        target="sales",
        step=1,
        horizon=2,
        group=generalforecast.GroupStrategy.given(2),
        columns={"price": "numeric", "weather": "numeric"} if covariates else None,
        known=["price"] if covariates else [],
    )
    try:
        table.insert(context)
        if covariates:
            query = future.append_column(
                "sales", pa.nulls(4, pa.float64())
            ).append_column("weather", pa.nulls(4, pa.float64()))
            table.insert(query)
        independent = table.forecast_given(
            ["a", "b"],
            [["a", "b"], ["a", "b"]],
            before=8,
            horizon=2,
            quantiles=(0.1, 0.5, 0.9),
        )
        np.testing.assert_allclose(
            quantiles, independent[[1, 0, 1, 0], [1, 0, 0, 1]], rtol=1e-5, atol=1e-5
        )
        restored = pickle.loads(pickle.dumps(estimator))
        np.testing.assert_allclose(
            restored.predict_quantiles(future), quantiles, atol=1e-5
        )
        restored.close()
        with pytest.raises(PackageError, match="custom native"):
            estimator.to_package(tmp_path / "invalid", name="custom")
    finally:
        client.drop_table("independent")
        estimator.close()


def test_verified_catalog_loader_path_executes_real_engine(monkeypatch, tmp_path):
    from generalforecast.deployments import catalog_tabicl
    from generalforecast.estimators import Classifier
    from tests.models.tabicl.tiny import classifier

    # Loader substitution is explicit: exercise admission/export/native behavior
    # but do not claim the tiny checkpoint is the published catalog checkpoint.
    def model(pin, *, allow_downloads):
        assert pin.name == "tabicl-v2-classifier" and not allow_downloads
        return classifier()[1]

    monkeypatch.setattr(catalog_tabicl, "_model", model)
    caller_values = np.arange(16, dtype=float)
    X = pa.table({"x": caller_values, "group": ["north", "south"] * 8})
    y = np.array(["a", "b"] * 8)
    estimator = Classifier().fit(X, y)
    assert estimator.predict_proba(X.slice(0, 2)).shape == (2, 2)
    checked = estimator.to_package(tmp_path / "portable", name="portable")
    assert checked.validate_queries(X.slice(0, 2)).num_rows == 2
    assert checked.context_table().num_rows == 16
    query = pa.table({"x": [0.0, 1.0], "group": ["north", "south"]})
    expected = estimator.predict_proba(query)
    caller_values[:] = 100
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_allclose(restored.predict_proba(query), expected, atol=1e-6)
    restored.close()
    estimator.close()


@pytest.mark.parametrize("shared", [False, True])
def test_catalog_forecaster_uses_shared_covariate_rows(monkeypatch, tmp_path, shared):
    from generalforecast.deployments import catalog_forecast
    from tests.models.chronos2 import tiny

    monkeypatch.setattr(tiny, "PATCH", 16)
    monkeypatch.setattr(tiny, "QUANTILES", catalog_forecast.QUANTILES)
    model = tiny.tiny_chronos2()
    monkeypatch.setattr(catalog_forecast, "_model", lambda **kwargs: model)
    context = pa.table(
        {
            "store": ["a"] * 8 + ["b"] * 8,
            "time": list(range(8)) * 2,
            "sales": np.arange(16, dtype=float),
            "price": np.tile(np.arange(8, dtype=float), 2),
        }
    )
    query = pa.table(
        {
            "store": ["b", "a", "b", "a"],
            "time": [9, 8, 8, 9],
            "price": [13.0, 11.0, 11.0, 13.0],
        }
    )
    estimator = Forecaster(
        target="sales",
        series="store",
        horizon=2,
        past_inputs=("price",) if shared else (),
        future_inputs=("price",),
    ).fit(context)
    before = estimator.context_.to_pylist()
    values = estimator.predict_quantiles(query)
    changed = estimator.predict_quantiles(
        query.set_column(2, "price", pa.array([130.0, 110.0, 110.0, 130.0]))
    )
    assert np.isfinite(values).all() and not np.allclose(values, changed)
    np.testing.assert_allclose(values, estimator.predict_quantiles(query))
    assert estimator.context_.to_pylist() == before
    checked = estimator.to_package(tmp_path / "forecast", name="forecast")
    assert checked.lock is not None and checked.lock.task == "forecast"
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_allclose(values, restored.predict_quantiles(query))
    estimator.close()
    restored.close()


def test_local_context_beyond_package_capacity_runs_but_never_exports(
    monkeypatch, tmp_path
):
    from generalforecast.deployments import catalog_tabicl
    from tests.models.tabicl.tiny import regressor

    monkeypatch.setattr(catalog_tabicl, "_model", lambda pin, **_: regressor()[1])
    features = np.random.default_rng(1).normal(size=(1100, 2))
    estimator = Regressor().fit(features, features.sum(axis=1))
    assert np.isfinite(estimator.predict(features[:3])).all()
    with pytest.raises(PackageError) as error:
        estimator.to_package(tmp_path / "large", name="large")
    assert error.value.code == "CapacityExceeded"
    assert not (tmp_path / "large").exists()
    estimator.close()
