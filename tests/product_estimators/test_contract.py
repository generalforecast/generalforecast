# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Estimator contracts without network access or downloaded model weights."""

import datetime
import socket

import numpy as np
import pyarrow as pa
import pytest
import yaml
from pyarrow import parquet
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from generalforecast import predictors
from generalforecast.contracts import PredictionResult
from generalforecast.errors import CapacityExceeded, InvalidInput, NotFound
from generalforecast.estimators import Classifier, Forecaster, Regressor
from generalforecast.models.catalog import V1_POLICY
from generalforecast.predictor import Predictor


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("estimator contract tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


class Backend:
    """Protocol test double, not engine inference evidence."""

    def __init__(self):
        self.requests = []
        self.result = None

    def predict(self, **request):
        self.requests.append(request)
        if self.result is not None:
            return self.result
        count = len(request["queries"])
        if request["task"] == "classification":
            output = {
                "label": ["a"] * count,
                "probabilities": [{"a": 0.75, "b": 0.25}] * count,
            }
        else:
            levels = request["options"].get("quantiles", [0.1, 0.5, 0.9])
            output = {
                "value": [2.0] * count,
                "quantiles": [{str(float(q)): float(q) for q in levels}] * count,
            }
        return PredictionResult(
            pa.table(output), model=request["model"], task=request["task"]
        )


def _saved(root) -> dict:
    """The predictor a save wrote, as `gf check --locked` reads it: a lock that holds."""
    spec = predictors.load(root)
    resolution = predictors.resolve(spec, root, locked=True)
    assert resolution.lock_changes == ()
    lock = yaml.safe_load((root / "gf.lock").read_text())
    assert lock["lock_version"] == 2 and lock["predictor"]["name"] == spec.name
    return lock["predictor"]


def history():
    return pa.table(
        {
            "store": ["a"] * 4 + ["b"] * 4,
            "time": [0, 1, 2, 3] * 2,
            "sales": [1.0, 2.0, 3.0, 4.0] * 2,
        }
    )


def future():
    # Interleaved and reversed rows must still map outputs to the original order.
    return pa.table({"store": ["b", "a", "b", "a"], "time": [5, 4, 4, 5]})


@pytest.mark.parametrize("estimator", [Classifier(), Regressor(), Forecaster()])
def test_clone_params_and_unfitted(estimator):
    copy = clone(estimator)
    assert copy.get_params() == estimator.get_params()
    assert copy.set_params(download=False) is copy
    with pytest.raises(NotFittedError):
        copy.predict(np.ones((2, 2)))
    with pytest.raises(NotFittedError):
        copy.save("unused", name="unused")


@pytest.mark.parametrize("adapter", ["arrow", "numpy", "pandas", "polars", "lazy"])
def test_tabular_adapters_and_target_names(tmp_path, adapter):
    backend = Backend()
    frame = pa.table({"price": [1.0, 2.0, 3.0, 4.0], "stock": [2.0, 3.0, 4.0, 5.0]})
    if adapter == "numpy":
        X, y = (
            np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
            np.array(["a", "b", "a", "b"]),
        )
    elif adapter == "pandas":
        pd = pytest.importorskip("pandas")
        X, y = frame.to_pandas(), pd.Series(["a", "b", "a", "b"], name="outcome")
    elif adapter in ("polars", "lazy"):
        pl = pytest.importorskip("polars")
        X, y = pl.from_arrow(frame), pl.Series("outcome", ["a", "b", "a", "b"])
        X = X.lazy() if adapter == "lazy" else X
    else:
        X, y = frame, np.array(["a", "b", "a", "b"])
    estimator = Classifier(backend=backend).fit(X, y)
    np.testing.assert_array_equal(estimator.predict(X), ["a"] * 4)
    assert estimator.score(X, y) == 0.5
    exported = estimator.save(tmp_path / adapter, name="portable")
    assert isinstance(exported, Predictor)
    assert exported.target == (
        "outcome" if adapter in ("pandas", "polars", "lazy") else "target"
    )
    lock = _saved(tmp_path / adapter)
    assert lock["target"] == exported.target
    assert lock["model"] == "general-forecast/tabicl-v2-classifier"
    assert lock["task"] == "classification" and lock["classes"] == ["a", "b"]
    assert lock["context"] == {
        "kind": "snapshot",
        "digest": lock["context"]["digest"],
        "rows": 4,
        "query_rows": 0,
        "path": "context.parquet",
    }
    assert parquet.read_table(tmp_path / adapter / "context.parquet").num_rows == 4


def test_export_keeps_the_codes_and_ensemble_variant(tmp_path):
    estimator = Classifier(
        model="general-forecast/tabicl-v2-classifier@4*2", backend=Backend()
    ).fit(np.ones((4, 2)), ["a", "b", "a", "b"])
    exported = estimator.save(tmp_path / "variant", name="variant")
    manifest = yaml.safe_load((tmp_path / "variant" / "gf.yaml").read_text())
    assert manifest["model"] == "general-forecast/tabicl-v2-classifier@4*2"
    assert exported.model == "general-forecast/tabicl-v2-classifier@4*2"
    lock = _saved(tmp_path / "variant")
    assert lock["model"] == "general-forecast/tabicl-v2-classifier@4*2"


def test_export_identity_owns_context_and_never_overwrites(tmp_path):
    features = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    target = np.array([1.0, 2.0, 3.0, 4.0])
    estimator = Regressor(backend=Backend()).fit(features, target)
    features[:] = 0
    target[:] = 0
    first = estimator.save(tmp_path / "one", name="portable")
    second = estimator.save(tmp_path / "two", name="portable")
    assert isinstance(first, Predictor) and isinstance(second, Predictor)
    assert _saved(tmp_path / "one") == _saved(tmp_path / "two")
    snapshot = parquet.read_table(tmp_path / "one" / "context.parquet")
    assert snapshot["f0"].to_pylist() == [1.0, 2.0, 3.0, 4.0]
    assert snapshot["target"].to_pylist() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(InvalidInput, match="already exists"):
        estimator.save(tmp_path / "one", name="portable")
    with pytest.raises(InvalidInput, match="saved as a predictor"):
        estimator.save(tmp_path / "three", name="portable", id="f0")
    estimator.set_params(page_size=64)
    with pytest.raises(NotFittedError):
        estimator.predict(np.ones((1, 2)))


def test_named_order_types_and_caller_buffers_are_preserved():
    a, b = np.arange(4.0), np.arange(4.0) + 2
    frame = pa.table({"a": a, "b": b})
    estimator = Regressor(backend=Backend()).fit(frame, np.arange(4.0))
    a[:] = 100
    assert estimator.context_["a"].to_pylist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(InvalidInput, match="names"):
        estimator.predict(frame.select(["b", "a"]))
    with pytest.raises(InvalidInput, match="types"):
        estimator.predict(pa.table({"a": [1, 2], "b": [2.0, 3.0]}))
    with pytest.raises(InvalidInput, match="frames"):
        estimator.predict(np.ones((2, 2)))


def test_byte_string_columns_are_read_as_text():
    backend = Backend()
    protocols = pa.array([b"tcp", b"udp", b"icmp", b"tcp"], pa.binary())
    raw = pa.table({"protocol": protocols, "bytes": [1.0, 2.0, 3.0, 4.0]})
    estimator = Regressor(backend=backend).fit(raw, np.arange(4.0))
    assert estimator.context_.schema.field("protocol").type == pa.string()
    assert estimator.context_["protocol"].to_pylist() == ["tcp", "udp", "icmp", "tcp"]
    estimator.predict(raw)
    assert backend.requests[-1]["queries"]["protocol"].to_pylist() == [
        "tcp",
        "udp",
        "icmp",
        "tcp",
    ]
    invalid = raw.set_column(0, "protocol", pa.array([b"\xff"] * 4, pa.binary()))
    with pytest.raises(InvalidInput, match="'protocol' holds bytes that are not UTF-8"):
        Regressor(backend=Backend()).fit(invalid, np.arange(4.0))
    rows = np.array([[b"tcp", 1.0], [b"udp", 2.0]], dtype=object)
    with pytest.raises(InvalidInput, match="pass text columns in a pandas or polars"):
        Regressor(backend=Backend()).fit(rows, np.arange(2.0))


@pytest.mark.parametrize(
    "fault", ["type", "model", "task", "rows", "classes", "probability"]
)
def test_backend_contract_failures_are_not_success(fault):
    backend = Backend()
    estimator = Classifier(backend=backend).fit(np.ones((4, 2)), ["a", "b", "a", "b"])
    table = pa.table({"probabilities": [{"a": 0.75, "b": 0.25}] * 2})
    model, task = estimator._spec.model, "classification"
    if fault == "model":
        model = "other"
    if fault == "task":
        task = "regression"
    if fault == "rows":
        table = table.slice(0, 1)
    if fault == "classes":
        table = pa.table({"probabilities": [{"a": 1.0}] * 2})
    if fault == "probability":
        table = pa.table({"probabilities": [{"a": -1.0, "b": 2.0}] * 2})
    backend.result = (
        {} if fault == "type" else PredictionResult(table, model=model, task=task)
    )
    with pytest.raises(InvalidInput) as error:
        estimator.predict(np.ones((2, 2)))
    assert error.value.code == "ProtocolError"


def test_byte_string_labels_are_read_as_text():
    X = pa.table({"x": [1.0, 2.0, 3.0, 4.0]})
    labels = np.array([b"a", b"b", b"a", b"b"], dtype=object)
    estimator = Classifier(backend=Backend()).fit(X, labels)
    np.testing.assert_array_equal(estimator.classes_, ["a", "b"])
    np.testing.assert_array_equal(estimator.predict(X), ["a"] * 4)
    assert estimator.score(X, labels) == 0.5
    with pytest.raises(InvalidInput, match="not UTF-8"):
        Classifier(backend=Backend()).fit(X, np.array([b"\xff", b"a", b"\xff", b"a"]))
    for target in (labels, labels.astype(str)):
        with pytest.raises(InvalidInput, match="use a Classifier to predict labels"):
            Regressor(backend=Backend()).fit(X, target)


def test_a_regressor_reads_numeric_text_and_forgets_its_table_on_a_text_refit():
    X = np.arange(4.0).reshape(-1, 1)
    numbers = np.array(["1", "2", "3", "4"], dtype=object)
    estimator = Regressor(backend=Backend()).fit(X, numbers)
    assert estimator.predict(X).shape == (4,)
    with pytest.raises(InvalidInput, match="use a Classifier to predict labels"):
        estimator.fit(X, np.array(["a", "b", "a", "b"], dtype=object))
    with pytest.raises(NotFittedError):
        estimator.predict(X)


def test_forecast_cadence_horizon_and_export(tmp_path):
    backend = Backend()
    estimator = Forecaster(
        target="sales", series="store", horizon=2, backend=backend
    ).fit(history())
    assert estimator.predict(future()).shape == (4,)
    request = backend.requests[-1]
    assert request["options"]["horizon"] == 2
    assert request["options"]["series"] == ["store"]
    exported = estimator.save(tmp_path / "forecast", name="forecast")
    assert isinstance(exported, Predictor)
    lock = _saved(tmp_path / "forecast")
    assert lock["task"] == "forecast" and lock["frequency"] == 1
    assert lock["horizon"] == 2 and lock["series"] == ["store"]
    assert lock["quantiles"] == [0.1, 0.5, 0.9]
    assert parquet.read_table(tmp_path / "forecast" / "context.parquet").equals(
        estimator.context_
    )
    assert yaml.safe_load((tmp_path / "forecast" / "gf.yaml").read_text()) == {
        "name": "forecast",
        "model": "general-forecast/chronos-2",
        "target": "sales",
        "time": "time",
        "series": ["store"],
        "horizon": 2,
        "quantiles": [0.1, 0.5, 0.9],
        "context": {"snapshot": "context.parquet"},
    }
    with pytest.raises(InvalidInput, match="horizon"):
        estimator.predict(future().slice(0, 1))
    with pytest.raises(InvalidInput, match="grid"):
        estimator.predict(pa.table({"store": ["a", "a"], "time": [5, 6]}))
    with pytest.raises(InvalidInput, match="absent"):
        estimator.predict(pa.table({"store": ["c", "c"], "time": [4, 5]}))
    estimator.set_params(horizon=3)
    with pytest.raises(NotFittedError):
        estimator.predict(future())


@pytest.mark.parametrize("times", [[0, 1, 3, 4], [0, 1, 1, 2]])
def test_forecaster_rejects_irregular_or_duplicate_context(times):
    with pytest.raises(InvalidInput):
        Forecaster(backend=Backend()).fit(
            pa.table({"time": times, "target": [1.0, 2.0, 3.0, 4.0]})
        )


def test_forecaster_datetime_and_separate_target():
    day = datetime.date(2026, 1, 1)
    features = pa.table({"time": [day + datetime.timedelta(days=i) for i in range(4)]})
    estimator = Forecaster(backend=Backend(), horizon=2).fit(
        features, [1.0, 2.0, 3.0, 4.0]
    )
    query = pa.table({"time": [day + datetime.timedelta(days=i) for i in (4, 5)]})
    assert estimator.predict(query).shape == (2,)
    assert estimator._frequency == 86400


def test_download_false_names_the_pull_command_when_not_cached(monkeypatch):
    monkeypatch.setattr("generalforecast.native.models.is_cached", lambda ref: False)
    with pytest.raises(NotFound) as caught:
        Classifier(download=False)
    assert str(caught.value) == (
        "error: model general-forecast/tabicl-v2-classifier is not cached\n"
        "  hint: run `gf model pull general-forecast/tabicl-v2-classifier`, "
        "or pass download=True"
    )


def test_package_capacity_binds_remote_and_export_not_local_paging(monkeypatch):
    def loading(entry, **_):
        raise RuntimeError("local loader reached")

    monkeypatch.setattr("generalforecast.models.artifacts.checkpoint", loading)
    count = V1_POLICY.context_rows + 1
    rows, target = np.ones((count, 2)), np.arange(float(count))
    with pytest.raises(RuntimeError, match="loader reached"):
        Regressor().fit(rows, target)
    with pytest.raises(CapacityExceeded) as error:
        Regressor(backend=Backend()).fit(rows, target)
    assert error.value.code == "CapacityExceeded"


def test_regression_backend_levels_and_unsupported_default():
    estimator = Regressor(backend=Backend()).fit(np.ones((4, 2)), np.arange(4.0))
    assert estimator.predict_quantiles(np.ones((2, 2)), [0.1, 0.5, 0.9]).shape == (2, 3)
    assert estimator.backend.requests[-1]["options"]["outputs"] == [
        "value",
        "quantiles",
    ]
    with pytest.raises(InvalidInput, match="explicit levels"):
        estimator.predict_quantiles(np.ones((2, 2)))
    with pytest.raises(InvalidInput, match="invalid quantile"):
        estimator.predict_quantiles(np.ones((2, 2)), [0.5, 0.1])
    with pytest.raises(InvalidInput, match="invalid quantile"):
        estimator.predict_quantiles(np.ones((2, 2)), [0.1234])


def test_failed_refit_invalidates_old_context_and_name_collisions():
    estimator = Classifier(backend=Backend()).fit(np.ones((4, 2)), ["a", "b"] * 2)
    with pytest.raises(InvalidInput):
        estimator.fit(np.ones((4, 2)), ["a"])
    assert not estimator.__sklearn_is_fitted__()
    frame = pa.table({"target": [1.0, 2.0, 3.0, 4.0]})
    estimator.fit(frame, ["a", "b"] * 2)
    assert estimator.target_ == "_target"


def test_forecaster_explicit_covariates_and_missing_future_rows():
    context = (
        history()
        .append_column("past", pa.array([1.0] * 8))
        .append_column("known", pa.array([2.0] * 8))
    )
    estimator = Forecaster(
        target="sales",
        series="store",
        horizon=2,
        backend=Backend(),
        past_inputs=("past",),
        future_inputs=("known",),
    ).fit(context)
    with pytest.raises(InvalidInput, match="known-future"):
        estimator.predict(future())
    query = future().append_column("known", pa.array([3.0] * 4))
    assert estimator.predict(query).shape == (4,)
    with pytest.raises(InvalidInput):
        Forecaster(target="sales", series="store", horizon=2, backend=Backend()).fit(
            context
        )


def test_related_forecasts_reject_nonportable_group_semantics():
    context = history().append_column("known", pa.array([1.0] * 4 + [2.0] * 4))
    with pytest.raises(InvalidInput, match="historical covariates"):
        Forecaster(
            target="sales",
            series="store",
            horizon=2,
            future_inputs=("known",),
            backend=Backend(),
        ).fit(context)
    context = history().append_column("known", pa.array([1.0] * 8))
    estimator = Forecaster(
        target="sales",
        series="store",
        horizon=2,
        future_inputs=("known",),
        backend=Backend(),
    ).fit(context)
    with pytest.raises(InvalidInput, match="future covariates"):
        estimator.predict(
            future().append_column("known", pa.array([1.0, 1.0, 2.0, 2.0]))
        )
