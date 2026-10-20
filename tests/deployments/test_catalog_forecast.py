import copy
from types import SimpleNamespace

import numpy as np
import pytest

from generalforecast.deployments import catalog_forecast
from generalforecast.deployments.boundary import Scope, json_object
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.contracts import ServiceError
from generalforecast.models.series import KNOWN, PAST, TARGET

SCOPE = Scope("forecast-project", "test-catalog")


MODEL = catalog_forecast.model


@pytest.fixture
def forecast():
    return {
        "model": "chronos-2",
        "task": "forecast",
        "target": "sales",
        "features": [],
        "context": [
            {
                "key": f"{name}-{i}",
                "series": name,
                "time": i,
                "value": base + i,
                "covariates": {"promo": i % 2, "past": i / 10},
            }
            for name, base in [("a", 10), ("b", 30)]
            for i in range(4)
        ],
        "queries": [
            {
                "key": "q-" + name,
                "series": name,
                "origin": 3,
                "group": ["a", "b"],
                "future_covariates": {"promo": [0, 1, 0]},
            }
            for name in ["a", "b"]
        ],
        "options": {
            "series": {
                "step": 1,
                "horizon": 3,
                "quantiles": [0.1, 0.5, 0.9],
                "known_covariates": ["promo"],
            }
        },
    }


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def refuse(**_kwargs):
        pytest.fail("invalid forecast must not load native weights")

    monkeypatch.setattr(catalog_forecast, "model", refuse)


@pytest.mark.parametrize(
    "path,value",
    [
        (("features",), ["x"]),
        (("options", "series", "step"), 0),
        (("options", "series", "step"), True),
        (("options", "series", "step"), 1.0),
        (("options", "series", "step"), 2),
        (("options", "series", "horizon"), 33),
        (("options", "series", "horizon"), 0),
        (("options", "series", "horizon"), 3.0),
        (("options", "series", "quantiles"), [0.5, 0.1]),
        (("options", "series", "known_covariates"), ["sales"]),
        (("options", "series", "known_covariates"), ["a"]),
        (("options", "series", "known_covariates"), ["promo", "promo"]),
        (("context", 0, "time"), 0.0),
        (("context", 0, "time"), -1024),
        (("context", 0, "time"), 4),
        (("context", 0, "time"), 1),
        (("context", 0, "key"), "a-1"),
        (("context", 0, "value"), True),
        (("context", 0, "value"), 1e13),
        (("context", 0, "covariates"), {"sales": 1}),
        (("context", 0, "covariates"), {"a": 1}),
        (("context", 0, "covariates"), {"promo": float("inf")}),
        (("context", 0, "covariates"), {} | {str(i): i for i in range(9)}),
        (("queries", 0, "origin"), 2),
        (("queries", 0, "origin"), 3.0),
        (("queries", 0, "origin"), 1030),
        (("queries", 0, "key"), "q-b"),
        (("queries", 0, "group"), ["a", "a"]),
        (("queries", 0, "group"), ["b"]),
        (("queries", 0, "group"), ["a", "missing"]),
        (("queries", 0, "future_covariates"), {"promo": [0]}),
        (("queries", 0, "future_covariates"), {"promo": [0, None, 1]}),
        (("queries", 0, "future_covariates"), {"promo": [0, True, 1]}),
        (("queries", 0, "future_covariates"), {"sales": [1, 2, 3]}),
        (("queries", 0, "future_covariates"), {"promo": [1, 2, 3], "past": [1, 2, 3]}),
    ],
)
def test_forecast_rejects_invalid_shapes_and_leakage(forecast, path, value):
    cursor = forecast
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert (caught.value.code, caught.value.status) == ("ValidationError", 422)


def test_unsupported_forecast_quantiles(forecast):
    forecast["options"]["series"]["quantiles"] = [0.123]
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert (caught.value.code, caught.value.status) == ("UnsupportedQuantiles", 422)


def test_forecast_times_must_stay_distinct_at_float_precision(forecast):
    origin = 1e12 - 2**-13
    forecast["context"] = forecast["context"][:1]
    forecast["context"][0]["time"] = origin
    forecast["queries"] = forecast["queries"][:1]
    forecast["queries"][0].update(origin=origin, group=["a"])
    forecast["options"]["series"]["step"] = 2**-14
    with pytest.raises(ServiceError, match="precision") as caught:
        CatalogService().infer(SCOPE, forecast)
    assert caught.value.code == "ValidationError"


@pytest.mark.parametrize("empty", ["all", "one", "no-rows"])
def test_empty_context(forecast, empty):
    for row in forecast["context"]:
        if empty == "all" or row["series"] == "b":
            row["value"] = None
    if empty == "no-rows":
        forecast["context"] = []
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert (caught.value.code, caught.value.status) == ("EmptyContext", 503)


def test_conflicting_group_covariates_are_not_silently_discarded(forecast):
    forecast["context"][0]["covariates"]["promo"] = 99
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert caught.value.code == "UnsupportedCovariates"


@pytest.mark.parametrize("floating_axis", [False, True])
def test_forecast_alignment_roles_and_no_request_persistence(
    forecast, monkeypatch, tmp_path, floating_axis
):
    if floating_axis:
        for row in forecast["context"]:
            row["time"] = 1_700_000_000.0 + row["time"] * 60
        for query in forecast["queries"]:
            query["origin"] = 1_700_000_180.0
        forecast["options"]["series"]["step"] = 60.0
    forecast["context"][1]["value"] = None
    forecast["context"].pop(6)
    before = copy.deepcopy(forecast)
    calls = []
    parameters = object()
    properties = SimpleNamespace(quantiles=catalog_forecast.QUANTILES, patch=16)
    model = (SimpleNamespace(properties=lambda _: properties), parameters)
    monkeypatch.setattr(catalog_forecast, "model", lambda **_: model)

    def run(actual_model, group, horizon):
        assert actual_model is parameters
        calls.append(group)
        return np.arange(group.values.shape[0] * horizon * 21).reshape(
            group.values.shape[0], horizon, 21
        )

    monkeypatch.setattr("generalforecast.models.inference._forecast", lambda _: run)
    monkeypatch.chdir(tmp_path)
    service = CatalogService()
    result = service.infer(SCOPE, forecast)
    assert (
        forecast == before
        and vars(service) == {"allow_downloads": False}
        and list(tmp_path.iterdir()) == []
    )
    assert len(calls) == 2
    first = calls[0]
    assert first.values.shape == (4, 16)
    np.testing.assert_array_equal(first.roles, [TARGET, TARGET, PAST, KNOWN])
    np.testing.assert_array_equal(first.observed[0, -4:], [True, False, True, True])
    np.testing.assert_array_equal(first.observed[1, -4:], [True, True, False, True])
    assert not np.asarray(first.observed[:, :-4]).any()
    np.testing.assert_array_equal(
        first.future_observed[:, 0], [False, False, False, True]
    )
    np.testing.assert_array_equal(first.future[-1], [0, 1, 0])
    assert result["outputs"][0]["quantiles"]["0.1"] == [2, 23, 44]
    assert result["outputs"][1]["quantiles"]["0.5"] == [73, 94, 115]
    assert result["outputs"][0]["times"] == (
        [1_700_000_240.0, 1_700_000_300.0, 1_700_000_360.0]
        if floating_axis
        else [4, 5, 6]
    )
    assert result["context_provenance"]["persisted"] is False
    assert result["artifact"]["verified"] is True
    json_object(dict(result))


@pytest.mark.parametrize("kind", ["shape", "finite", "capability", "exception"])
def test_forecast_native_output_validation(forecast, monkeypatch, kind):
    quantiles = (0.5,) if kind == "capability" else catalog_forecast.QUANTILES
    properties = SimpleNamespace(quantiles=quantiles, patch=16)
    model = (SimpleNamespace(properties=lambda _: properties), None)
    monkeypatch.setattr("generalforecast.models.registry.load", lambda *_, **__: model)
    monkeypatch.setattr(catalog_forecast, "model", MODEL)

    def run(*_):
        if kind == "exception":
            raise RuntimeError("private-request-token")
        return np.full(
            (4, 3, 1 if kind == "shape" else 21),
            float("nan") if kind == "finite" else 1,
        )

    monkeypatch.setattr("generalforecast.models.inference._forecast", lambda _: run)
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert caught.value.code == {
        "capability": "UnsupportedModel",
        "exception": "InferenceFailed",
    }.get(kind, "ModelOutputError")
    assert "private-request-token" not in str(caught.value)
