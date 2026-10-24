# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""api/openapi.yaml is the source of truth: its examples fit its schemas, the
generated module round-trips them, regenerating changes nothing, and (with
`GF_CONFORMANCE_URL`) a running gf serve answers them the same way."""

import subprocess
import sys

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
        assert (route.request is None) == (route.method == "GET"), route.name
        assert issubclass(route.reply, _api.Payload)


@pytest.mark.parametrize("example", EXAMPLES, ids=ident)
def test_generated_module_round_trips(example):
    route = ROUTES[example.operation]
    if example.section == "request":
        cls = route.request
    elif example.status < 400:
        cls = route.reply
    elif "health" in example.tags:
        cls = _api.HealthError
    else:
        cls = _api.ErrorReply
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
