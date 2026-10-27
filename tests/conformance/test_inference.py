# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real inference over HTTP against the server GF_CONFORMANCE_URL names (`-m conformance`).

`test_openapi.py` replays the spec's examples and checks their shapes; these
check what the predictions mean. The server serves the predictor `churn` the
examples name (general-forecast/tabicl-v2-classifier, id `customer_id`,
inputs `spend` and `plan`), the ad hoc route for
general-forecast/tabicl-v2-classifier, and a forecast predictor `sales` on
general-forecast/chronos-2 (time `date`, series `sku`, frequency `1d`, the
future input `price` and the default quantiles), with contexts dated before
2026-01-31.
"""

import datetime

import pytest

from tests.conformance.openapi import Server, Validator, load

VALIDATOR = Validator(load())
pytestmark = pytest.mark.conformance


@pytest.fixture
def server():
    return Server()


def scored(server, path, name, payload):
    status, body = server.call("POST", path, "predictor", name, payload)
    assert status == 200, body
    VALIDATOR.validate(VALIDATOR.schema("ResultReply"), body)
    return body["result"]


def test_classifier_predictions_follow_query_order_with_ids_and_probabilities(server):
    queries = [
        {"customer_id": 9001, "spend": 82.5, "plan": "pro"},
        {"customer_id": 9002, "spend": 3.0, "plan": "free"},
    ]
    result = scored(server, "/predictors/churn/predict", "churn", {"queries": queries})
    assert result["task"] == "classification"
    # Each prediction carries its query's id under the id column's name.
    assert [row["customer_id"] for row in result["predictions"]] == [9001, 9002]
    for row in result["predictions"]:
        assert set(row) == {"customer_id", "label", "probabilities"}
        assert abs(sum(row["probabilities"].values()) - 1) < 1e-6
    reordered = scored(
        server,
        "/predictors/churn/predict",
        "churn",
        {"queries": list(reversed(queries))},
    )
    assert reordered["predictions"] == list(reversed(result["predictions"]))


def test_ad_hoc_classifier_uses_only_the_listed_inputs(server):
    # The unlisted column `note` is left out: only `spend` and `seats` are
    # inputs, so a query's `note` cannot change its prediction.
    context = [
        {
            "spend": float(i),
            "seats": i % 2,
            "note": str(i),
            "churned": i % 3 == 0,
        }
        for i in range(12)
    ]
    payload = {
        "model": "general-forecast/tabicl-v2-classifier",
        "target": "churned",
        "inputs": ["spend", "seats"],
        "context": context,
        "queries": [{"spend": 4.0, "seats": 1, "note": "x"}],
    }
    result = scored(server, "/predict", "ad-hoc", payload)
    assert (result["context"]["source"], result["context"]["rows"]) == ("request", 12)
    [row] = result["predictions"]
    noted = scored(
        server,
        "/predict",
        "ad-hoc",
        {**payload, "queries": [{"spend": 4.0, "seats": 1, "note": "other"}]},
    )
    [other] = noted["predictions"]
    assert other["label"] == row["label"]
    assert other["probabilities"] == pytest.approx(row["probabilities"], rel=1e-5)
    assert type(row["label"]) is bool and "id" not in row
    status, body = server.call(
        "POST",
        "/predict",
        "predictor",
        "ad-hoc",
        {**payload, "inputs": ["spend", "missing"]},
    )
    assert status == 422, body
    assert body["error"]["param"] == "inputs.missing", body


def test_ad_hoc_capacity_errors_name_the_limit_the_model_lists(server):
    model = "general-forecast/tabicl-v2-classifier"
    status, body = server.call("GET", "/models", "model", "models")
    assert status == 200, body
    [limits] = [m["limits"] for m in body["result"]["models"] if m["name"] == model]
    rows = [{"spend": 4.0, "seats": 1}]
    payload = {
        "model": model,
        "target": "churned",
        "context": [{"spend": 1.0, "seats": 0, "churned": True}] * 2,
        "queries": rows * (limits["queries"] + 1),
    }
    status, body = server.call("POST", "/predict", "predictor", "ad-hoc", payload)
    assert status == 422 and body["error"]["code"] == "CapacityExceeded", body
    assert body["error"]["limit"] == limits["queries"]
    assert body["error"]["actual"] == limits["queries"] + 1
    assert model in body["error"]["message"]


def test_health_answers_200_only_once_predictions_answer(server):
    status, body = server.request("GET", "/healthz")
    ready, _ = server.request("GET", "/readyz")
    assert ready == status
    if status == 200:
        VALIDATOR.validate(VALIDATOR.schema("Health"), body)
        queries = [{"customer_id": 9001, "spend": 82.5, "plan": "pro"}]
        scored(server, "/predictors/churn/predict", "churn", {"queries": queries})
    else:
        assert status == 503
        VALIDATOR.validate(VALIDATOR.schema("HealthError"), body)
        assert body["error"]["code"] == "ModelNotReady"


def test_forecast_predictions_carry_ordered_quantiles_per_future_row(server):
    start = datetime.date(2026, 1, 1)
    queries = [
        {
            "sku": sku,
            "date": (start + datetime.timedelta(days=30 + i)).isoformat(),
            "price": 9.0,
        }
        for sku in ("a", "b")
        for i in range(2)
    ]
    result = scored(server, "/predictors/sales/predict", "sales", {"queries": queries})
    assert result["task"] == "forecast"
    assert len(result["predictions"]) == len(queries)
    for row in result["predictions"]:
        assert type(row["value"]) is float
        levels = row["quantiles"]
        assert set(levels) == {"0.1", "0.5", "0.9"}
        assert levels["0.1"] <= levels["0.5"] <= levels["0.9"]
