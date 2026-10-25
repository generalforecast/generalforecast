# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real optimized engine calls with tiny random weights, not catalog/cloud proof."""

import pickle
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import yaml
from pyarrow import parquet
from sklearn.model_selection import cross_val_score

from generalforecast.engine import LocalEngine
from generalforecast.estimators import Forecaster, Regressor
from generalforecast.predictor import Predictor
from generalforecast.predictors.validation import PredictorError
from tests.utils import engine_client as generalforecast

pytestmark = pytest.mark.slow

# The tiny model the engine's workers load for each catalog model, so catalog
# tests run its admission, export and native paths without its checkpoint.
CATALOG = {
    "tabicl-v2-classifier": "tests/tabicl-classifier",
    "tabicl-v2-regressor": "tests/tabicl-regressor",
    "chronos-2": "tests/chronos2-catalog-forecaster",
}


@pytest.fixture(scope="module")
def catalog() -> Iterator[None]:
    """The process engine, loading each catalog model's tiny spec in its workers.

    It does not claim the tiny checkpoints are the published ones; estimators
    given no client connect to it afresh, and never download.
    """
    tiny = LocalEngine(models=CATALOG)

    def local(surface: Any = "local", allow_downloads: bool = False) -> LocalEngine:
        assert not allow_downloads
        return tiny.gated(surface, allow_downloads)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("generalforecast.engine.local", local)
        patch.setattr("generalforecast.native.estimators._shared", {})
        yield
    tiny.close()


def test_regressor_cross_validation_uses_fold_local_context():
    features = np.random.default_rng(0).normal(size=(24, 3)).astype(np.float32)
    target = features.sum(axis=1)
    estimator = Regressor("tests/tabicl-regressor")
    scores = cross_val_score(estimator, features, target, cv=2)
    assert scores.shape == (2,) and np.isfinite(scores).all()
    assert not estimator.__sklearn_is_fitted__()


def test_byte_string_columns_predict_as_their_text():
    protocols = np.array(["tcp", "udp", "icmp"])[np.arange(24) % 3]
    numbers = np.random.default_rng(0).normal(size=24).astype(np.float32)
    text = pa.table({"protocol": protocols, "x": numbers})
    raw = text.set_column(
        0, "protocol", pa.array(np.char.encode(protocols), pa.binary())
    )
    target = numbers + np.arange(24) % 3
    model = "tests/tabicl-regressor"
    from_bytes = Regressor(model).fit(raw, target)
    from_text = Regressor(model).fit(text, target)
    np.testing.assert_allclose(
        from_bytes.predict(raw), from_text.predict(text), atol=1e-6
    )


@pytest.mark.parametrize(
    "covariates,shared", [(False, False), (True, False), (True, True)]
)
def test_forecaster_matches_native_grouped_series_and_preserves_context(
    tmp_path, covariates, shared
):
    model = "tests/chronos2-forecaster"
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
    quantiles = estimator.predict_quantiles(future)
    assert quantiles.shape == (4, 3) and np.isfinite(quantiles).all()
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
        with pytest.raises(PredictorError, match="custom native"):
            estimator.save(tmp_path / "invalid", name="custom")
    finally:
        client.drop_table("independent")
        estimator.close()


def test_verified_catalog_loader_path_executes_real_engine(catalog, tmp_path):
    from generalforecast.estimators import Classifier

    caller_values = np.arange(16, dtype=float)
    X = pa.table({"x": caller_values, "group": ["north", "south"] * 8})
    y = np.array(["a", "b"] * 8)
    estimator = Classifier().fit(X, y)
    assert estimator.predict_proba(X.slice(0, 2)).shape == (2, 2)
    checked = estimator.save(tmp_path / "portable", name="portable")
    assert isinstance(checked, Predictor) and checked.model.endswith(
        "tabicl-v2-classifier"
    )
    assert len(checked.predict(X.slice(0, 2).to_pandas())) == 2
    assert parquet.read_table(tmp_path / "portable" / "context.parquet").num_rows == 16
    assert (
        yaml.safe_load((tmp_path / "portable" / "gf.lock").read_text())["predictor"][
            "context"
        ]["rows"]
        == 16
    )
    query = pa.table({"x": [0.0, 1.0], "group": ["north", "south"]})
    expected = estimator.predict_proba(query)
    caller_values[:] = 100
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_allclose(restored.predict_proba(query), expected, atol=1e-6)
    restored.close()
    estimator.close()


@pytest.mark.parametrize("shared", [False, True])
def test_catalog_forecaster_uses_shared_covariate_rows(catalog, tmp_path, shared):
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
    checked = estimator.save(tmp_path / "forecast", name="forecast")
    assert isinstance(checked, Predictor) and checked.task == "forecast"
    assert (tmp_path / "forecast" / "gf.lock").is_file()
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_allclose(values, restored.predict_quantiles(query))
    estimator.close()
    restored.close()


def test_local_context_beyond_package_capacity_runs_but_never_exports(
    catalog, tmp_path
):
    features = np.random.default_rng(1).normal(size=(1100, 2))
    estimator = Regressor().fit(features, features.sum(axis=1))
    assert np.isfinite(estimator.predict(features[:3])).all()
    with pytest.raises(PredictorError) as error:
        estimator.save(tmp_path / "large", name="large")
    assert error.value.code == "CapacityExceeded"
    assert not (tmp_path / "large").exists()
    estimator.close()


def test_module_function_specs_are_not_catalog_models() -> None:
    import generalforecast as gf

    for estimator in (gf.Classifier, gf.Regressor, gf.Forecaster):
        with pytest.raises(
            gf.InvalidInput,
            match="is not a catalog model id; use general-forecast/<name>",
        ):
            estimator(model="os.path:join")
