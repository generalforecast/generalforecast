# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""api/openapi.yaml is the source of truth: its examples fit its schemas, the
generated module round-trips them, regenerating changes nothing, and (with
`GF_CONFORMANCE_URL`) a running gf serve answers them the same way."""

import subprocess
import sys
from typing import Any

import pytest

from api import generate
from generalforecast import _api
from tests.conformance.openapi import (
    ROOT,
    SERVED_TAGS,
    Example,
    SchemaError,
    Server,
    Validator,
    examples,
    load,
    pairs,
)

SPEC = load()
VALIDATOR = Validator(SPEC)
EXAMPLES = examples(SPEC)
PAIRS = pairs(EXAMPLES)
ROUTES = {route.name: route for route in _api.ROUTES}


def ident(example):
    return example.id if isinstance(example, Example) else example[0].id


REPLIES = {(reply.operation, reply.name): reply for _, reply in PAIRS}


@pytest.mark.parametrize("example", EXAMPLES, ids=ident)
def test_example_fits_its_schema(example):
    if not example.invalid:
        VALIDATOR.validate(example.schema, example.value, example.id)
        return
    reply = REPLIES[(example.operation, example.name)]
    param = reply.value["error"]["param"]
    with pytest.raises(SchemaError, match=rf"payload\.{param}"):
        VALIDATOR.validate(example.schema, example.value, example.id)


@pytest.mark.parametrize("example", EXAMPLES, ids=ident)
def test_example_shows_the_full_envelope(example):
    if "health" in example.tags:
        assert set(example.value) in ({"status"}, {"error"})
        return
    if not example.envelope:
        return
    body = (
        "payload"
        if example.section == "request"
        else ("error" if example.status and example.status >= 400 else "result")
    )
    assert set(example.value) == {
        "api_version",
        "project",
        "request_id",
        "resource",
        body,
    }, example.id
    assert example.value["api_version"] == _api.VERSION


def test_every_route_has_a_request_example_per_reply_example():
    for _, reply in PAIRS:
        assert reply.status is not None
    responses = {(e.operation, e.name) for e in EXAMPLES if e.section == "response"}
    for example in EXAMPLES:
        if example.section == "request":
            assert (example.operation, example.name) in responses, example.id


def test_error_codes_are_the_contract_set():
    assert [code.value for code in _api.ErrorCode] == [
        "ValidationError",
        "NotFound",
        "Unauthorized",
        "CapacityExceeded",
        "ModelNotReady",
        "Internal",
        "Conflict",
    ]
    for example in EXAMPLES:
        if example.status is not None and example.status >= 400:
            error = example.value["error"]
            assert error["code"] in _api.ErrorCode.__members__
            if error["code"] == "CapacityExceeded":
                assert {"limit", "actual"} <= set(error), example.id
                assert "general-forecast/" in error["message"], example.id


def test_manifest_fields_are_the_gf_yaml_fields_plus_rows():
    manifest = SPEC["components"]["schemas"]["Manifest"]
    assert list(manifest["properties"]) == [
        "model",
        "target",
        "id",
        "inputs",
        "time",
        "series",
        "related",
        "frequency",
        "horizon",
        "quantiles",
        "outputs",
        "selection",
        "context",
        "queries",
    ]
    assert manifest["required"] == ["model", "target", "context", "queries"]
    assert manifest["properties"]["quantiles"]["default"] == [0.1, 0.5, 0.9]
    assert _api.Manifest.DEFAULTS == {"quantiles": [0.1, 0.5, 0.9]}
    assert _api.Selection.DEFAULTS == {
        "core": 0.25,
        "neighbors": 0.75,
        "kind": "stratified",
    }
    for retired in ("task", "options", "features", "data", "dataset", "package"):
        assert retired not in manifest["properties"]


def test_routes_table_matches_the_spec():
    expected = {
        (method.upper(), path)
        for path, methods in SPEC["paths"].items()
        for method in methods
    }
    assert {(route.method, route.path) for route in _api.ROUTES} == expected
    public = {
        ("POST", "/v1/projects/{project}/predictors/{name}/predict"),
        ("POST", "/v1/projects/{project}/predict"),
        ("GET", "/v1/projects/{project}/predictors"),
        ("GET", "/v1/projects/{project}/predictors/{name}"),
        ("GET", "/v1/projects/{project}/models"),
        ("GET", "/healthz"),
        ("GET", "/readyz"),
    }
    assert public <= expected
    for route in _api.ROUTES:
        assert (route.body is None) == (route.method == "GET"), route.name
        assert (route.request is None) == (route.body != "application/json"), route.name
        assert issubclass(route.reply, _api.Payload)


@pytest.mark.parametrize("example", EXAMPLES, ids=ident)
def test_generated_module_round_trips(example):
    route = ROUTES[example.operation]
    if example.section == "request":
        cls = route.request
    elif example.status < 400:
        cls = route.reply
    else:
        cls = getattr(_api, example.schema["$ref"].rsplit("/", 1)[-1])
    assert cls is not None
    if example.invalid:
        reply = REPLIES[(example.operation, example.name)]
        with pytest.raises(_api.PayloadError) as refused:
            cls.from_payload(example.value)
        assert refused.value.param == "payload." + reply.value["error"]["param"]
        return
    decoded = cls.from_payload(example.value)
    assert decoded.to_payload() == example.value
    assert cls.from_payload(decoded.to_payload()) == decoded


def test_generated_module_names_the_bad_field():
    scored = next(p for p in PAIRS if p[0].id == "predict:request:scored")[0].value
    with pytest.raises(_api.PayloadError) as missing:
        _api.ManifestRequest.from_payload(
            {
                **scored,
                "payload": {
                    k: v for k, v in scored["payload"].items() if k != "target"
                },
            }
        )
    assert missing.value.param == "payload.target"
    with pytest.raises(_api.PayloadError) as unknown:
        _api.Queries.from_payload({"queries": [{"x": 1}], "target": "y"})
    assert unknown.value.param == "target"
    with pytest.raises(_api.PayloadError) as bare:
        _api.Manifest.from_payload(
            {**scored["payload"], "model": "tabicl-v2-classifier"}
        )
    assert bare.value.param == "model"
    with pytest.raises(_api.PayloadError) as wrong:
        _api.Queries.from_payload({"queries": [{"x": [1]}]})
    assert wrong.value.param == "queries[0].x"


MODEL = "general-forecast/tabicl-v2-classifier"
ROWS = [{"customer_id": 1041, "spend": 82.5, "plan": "pro"}]
CONTEXT = [
    {"customer_id": 1, "spend": 10.0, "plan": "free", "churned": 1},
    {"customer_id": 2, "spend": 95.0, "plan": "pro", "churned": 0},
]


def envelope(name, payload):
    return {
        "api_version": _api.VERSION,
        "project": "acme",
        "request_id": "r-payload",
        "resource": {"kind": "predictor", "name": name},
        "payload": payload,
    }


@pytest.mark.parametrize("field", ["inputs", "model", "task"])
def test_a_declared_predictors_payload_refuses_what_its_manifest_declares(field):
    payload = {"queries": ROWS, field: ["spend"] if field == "inputs" else "churned"}
    with pytest.raises(_api.PayloadError) as refused:
        _api.QueriesRequest.from_payload(envelope("churn", payload))
    assert refused.value.param == "payload." + field


def test_a_declared_predictors_payload_is_queries_context_and_as_of():
    request = _api.QueriesRequest.from_payload(
        envelope(
            "churn",
            {"queries": ROWS, "context": CONTEXT, "as_of": "2026-10-24T05:00:00Z"},
        )
    )
    assert request.payload.as_of == "2026-10-24T05:00:00Z"
    assert request.payload.to_payload()["context"] == CONTEXT
    with pytest.raises(_api.PayloadError) as empty:
        _api.Queries.from_payload({})
    assert empty.value.param == "queries"


@pytest.mark.parametrize("field", ["model", "context", "queries"])
def test_an_ad_hoc_payload_requires_model_context_and_queries(field):
    payload = {"model": MODEL, "target": "churned", "context": CONTEXT, "queries": ROWS}
    del payload[field]
    with pytest.raises(_api.PayloadError) as refused:
        _api.ManifestRequest.from_payload(envelope("ad-hoc", payload))
    assert refused.value.param == "payload." + field


@pytest.mark.parametrize("field", ["task", "options", "features", "data", "name"])
def test_an_ad_hoc_payload_has_no_task_options_or_retired_words(field):
    payload = {
        "model": MODEL,
        "target": "churned",
        "context": CONTEXT,
        "queries": ROWS,
        field: "classification" if field == "task" else {},
    }
    with pytest.raises(_api.PayloadError) as refused:
        _api.Manifest.from_payload(payload)
    assert refused.value.param == field


def test_an_ad_hoc_payload_takes_the_manifests_input_and_frequency_forms():
    manifest = _api.Manifest.from_payload(
        {
            "model": "general-forecast/chronos-2",
            "target": "units",
            "time": "date",
            "series": ["sku"],
            "frequency": "1d",
            "horizon": 7,
            "inputs": {"past": [], "future": ["price"]},
            "context": CONTEXT,
            "queries": ROWS,
        }
    )
    assert isinstance(manifest.inputs, _api.Inputs)
    assert manifest.inputs.future == ("price",)
    assert manifest.quantiles is None
    listed = _api.Manifest.from_payload(
        {**manifest.to_payload(), "inputs": ["price"], "frequency": 86400}
    )
    assert listed.inputs == ("price",) and listed.frequency == 86400


def test_health_and_readiness_answer_the_same_unenveloped_reply():
    assert ROUTES["health"].path == "/healthz"
    assert ROUTES["ready"].path == "/readyz"
    assert ROUTES["health"].reply is ROUTES["ready"].reply is _api.Health


def test_regenerating_changes_nothing(tmp_path):
    for path, text in generate.render().items():
        assert path.read_text() == text, f"{path.relative_to(ROOT)} is stale"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "api" / "generate.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_generated_python_is_formatted():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            str(generate.PYTHON),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


# -- replay against a running server -----------------------------------------


def served():
    return [
        (request, reply) for request, reply in PAIRS if set(request.tags) & SERVED_TAGS
    ] + [
        (None, reply)
        for reply in EXAMPLES
        if reply.section == "response"
        and reply.method == "GET"
        and set(reply.tags) & SERVED_TAGS
        and not ("health" in reply.tags and reply.status != 200)
    ]


def replay_id(pair):
    return pair[1].id


@pytest.mark.conformance
@pytest.mark.parametrize("pair", served(), ids=replay_id)
def test_server_answers_the_example(pair):
    request, reply = pair
    server = Server()
    path = reply.path.replace("{project}", server.project)
    if "{name}" in path or "{model}" in path:
        name = (request or reply).value["resource"]["name"]
        path = path.replace("{name}", name).replace("{model}", name)
    body = server.rewrite(request.value) if request is not None else None
    status, answer = server.request(reply.method, path, body)
    assert status == reply.status, answer
    VALIDATOR.validate(reply.schema, answer, reply.id)
    if "health" in reply.tags:
        return
    if status >= 400:
        assert answer["error"]["code"] == reply.value["error"]["code"]
        assert answer["error"].get("param") == reply.value["error"].get("param")
        return
    if body is not None:
        assert answer["request_id"] == body["request_id"]
        assert answer["resource"] == body["resource"]
    if body is not None and "predictions" in answer.get("result", {}):
        assert len(answer["result"]["predictions"]) == len(body["payload"]["queries"])


def _locks(node: Any) -> list[dict]:
    """Every `lock` document the spec's examples hold."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "lock" and isinstance(value, dict) and "predictor" in value:
                found.append(value)
            found.extend(_locks(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_locks(value))
    return found


def test_every_example_lock_is_one_the_sdk_accepts() -> None:
    from generalforecast.contracts import parse_lock

    locks = _locks(SPEC["paths"])
    assert locks
    for lock in locks:
        assert parse_lock(lock).digest == lock["digest"]
