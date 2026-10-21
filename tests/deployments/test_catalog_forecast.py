# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest

from generalforecast.deployments import catalog_forecast
from generalforecast.deployments.boundary import Scope, json_object
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.contracts import ServiceError
from generalforecast.engine import Forecasts, ModelOutputError
from generalforecast.models import catalog
from generalforecast.models.outputs import quantile_index
from generalforecast.models.series import KNOWN, PAST, TARGET
from tests.utils.engine import FakeEngine

SCOPE = Scope("forecast-project", "test-catalog")


RUN = catalog_forecast.run
QUANTILES = catalog.ENTRIES["chronos-2"].capability.quantiles


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
    def refuse(*_args, **_kwargs):
        pytest.fail("invalid forecast must not load native weights")

    monkeypatch.setattr(catalog_forecast, "run", refuse)


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
    fake = FakeEngine(QUANTILES).install(monkeypatch)
    monkeypatch.setattr(catalog_forecast, "run", RUN)
    monkeypatch.chdir(tmp_path)
    service = CatalogService()
    result = service.infer(SCOPE, forecast)
    assert (
        forecast == before
        and vars(service)
        == {"allow_downloads": False, "runtime": "cpu", "surface": "remote"}
        and list(tmp_path.iterdir()) == []
    )
    # One validated request, every query with its aligned group.
    [request] = fake.forecasts
    assert request.known_covariates == ["promo"]
    assert [(q.series, q.members, q.member) for q in request.queries] == [
        ("a", ["a", "b"], 0),
        ("b", ["a", "b"], 1),
    ]
    group = request.queries[0].group
    np.testing.assert_array_equal(group.roles, [TARGET, TARGET, PAST, KNOWN])
    np.testing.assert_array_equal(group.values[:2], [[10, 0, 12, 13], [30, 31, 0, 33]])
    np.testing.assert_array_equal(group.observed[:2], [[1, 0, 1, 1], [1, 1, 0, 1]])
    np.testing.assert_array_equal(group.future[3], [0, 1, 0])
    low, median = quantile_index(0.1, QUANTILES), quantile_index(0.5, QUANTILES)
    width = len(QUANTILES)
    assert result["outputs"][0]["quantiles"]["0.1"] == [
        period * width + low for period in range(3)
    ]
    assert result["outputs"][1]["quantiles"]["0.5"] == [
        (3 + period) * width + median for period in range(3)
    ]
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
    quantiles = (0.5,) if kind == "capability" else QUANTILES
    monkeypatch.setattr(catalog_forecast, "run", RUN)

    class Engine:
        def forecast(self, ref, request):
            if kind == "exception":
                raise RuntimeError("private-request-token")
            if kind in ("shape", "finite"):
                raise ModelOutputError("the forecaster returned invalid forecasts")
            return Forecasts(np.ones((2, 3, len(quantiles))), quantiles)

    monkeypatch.setattr("generalforecast.engine.local", lambda *_: Engine())
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert caught.value.code == {
        "capability": "UnsupportedModel",
        "exception": "InferenceFailed",
    }.get(kind, "ModelOutputError")
    assert "private-request-token" not in str(caught.value)


def fake_forecaster(monkeypatch, levels=(0.1, 0.5, 0.9), max_rows=None):
    monkeypatch.setattr(catalog_forecast, "run", RUN)
    return FakeEngine(levels, max_rows).install(monkeypatch).loads


@pytest.mark.parametrize(
    "name", ["chronos-2", "timestral-28m", "timestral-209m", "timestral-28m-unmirrored"]
)
def test_each_forecaster_loads_and_reports_its_own_weights(forecast, monkeypatch, name):
    levels = QUANTILES if name == "chronos-2" else (0.1, 0.5, 0.9)
    loads = fake_forecaster(monkeypatch, levels)
    forecast["model"] = name
    result = CatalogService().infer(SCOPE, forecast)
    entry = catalog.ENTRIES[name]
    assert loads == [(name, "remote")]
    assert result["artifact"]["repository"] == entry.artifact.repository
    assert result["artifact"]["revision"] == entry.artifact.revision
    assert result["artifact"]["license"] == entry.artifact.license
    assert result["artifact"]["verified"] is True
    assert list(result["outputs"][0]["quantiles"]) == ["0.1", "0.5", "0.9"]
    json_object(dict(result))


def test_quantiles_and_row_limit_come_from_the_loaded_model(forecast, monkeypatch):
    forecast["model"] = "timestral-28m"
    loads = fake_forecaster(monkeypatch, (0.2, 0.5, 0.8))
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert caught.value.code == "UnsupportedQuantiles"
    assert loads == [("timestral-28m", "remote")]
    fake_forecaster(monkeypatch, max_rows=3)
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert caught.value.code == "CapacityExceeded"


def test_timesfm_forecasts_only_on_a_self_hosted_server(forecast, monkeypatch):
    forecast["model"] = "timesfm-3"
    loads = fake_forecaster(monkeypatch)
    with pytest.raises(ServiceError) as caught:
        CatalogService().infer(SCOPE, forecast)
    assert caught.value.code == "UnsupportedModel" and loads == []
    result = CatalogService(surface="self-hosted").infer(SCOPE, forecast)
    assert loads == [("timesfm-3", "self-hosted")]
    entry = catalog.ENTRIES["timesfm-3"]
    assert result["artifact"]["repository"] == entry.artifact.repository
    assert result["artifact"]["license"] == entry.artifact.license
