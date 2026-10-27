# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone `gf serve` endpoints through fake sync and async transports."""

import asyncio
import inspect
import io
import json
import re
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest

import generalforecast as gf
from generalforecast.async_client import AsyncClient
from generalforecast.client import Client
from generalforecast.service.boundary import Request, Resource, Scope, success
from generalforecast.service.contracts import ServiceError
from generalforecast.transport import (
    PLATFORM_PROBE,
    HostedOnlyError,
    Profile,
    hosted_only,
)

HOSTED = "https://hosted.example"
LOCAL = "http://127.0.0.1:8123"
# What gf-server's front resolves (crates/gf-server/src/http/incoming.rs).
_SERVED = (
    ("GET", re.compile(r"/v1/projects/([^/]+)/(models)(?:/([^/]+))?")),
    ("GET", re.compile(r"/v1/projects/([^/]+)/(predictors)(?:/([^/]+))?")),
    ("POST", re.compile(r"/v1/projects/([^/]+)/(predictors)/([^/]+)/predict")),
)
_PREDICTION = {
    "model": "general-forecast/tabicl-v2-classifier",
    "task": "classification",
    "predictions": [{"label": True, "probabilities": {"false": 0.1, "true": 0.9}}],
}


def _refused(message: str) -> tuple[int, dict[str, Any]]:
    return 422, {"error": {"code": "ValidationError", "message": message}}


_PREDICTORS = {
    "risk": {"name": "risk", "model": "general-forecast/tabicl-v2-classifier"}
}


class ModelServer:
    """Answers like a plain gf serve model server, or like the hosted platform."""

    def __init__(self, origin: str, *, platform: bool = False) -> None:
        self.origin, self.platform, self.seen = origin, platform, []

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> tuple[int, dict[str, Any]]:
        origin, _, rest = url.partition("/v1")
        path, _, query = ("/v1" + rest).partition("?")
        self.seen.append((method, origin, path, headers))
        assert origin == self.origin
        if path == PLATFORM_PROBE:
            if self.platform:
                return 401, {"error": {"code": "Unauthenticated", "message": "key"}}
            return 404, {"error": {"code": "NotFound", "message": "route not found"}}
        for verb, pattern in _SERVED:
            route = pattern.fullmatch(path)
            if route is not None and method == verb:
                break
        else:
            return 404, {"error": {"code": "NotFound", "message": "route not found"}}
        project, collection, item = route.groups()
        if method == "POST":
            # A served predictor has no environments; any query is refused.
            if query:
                return _refused(f"predictor {item} is served from its lock alone")
            if item not in _PREDICTORS:
                return 404, {
                    "error": {"code": "NotFound", "message": "predictor not found"}
                }
            request = Request.parse(json.loads(body))
            return 200, dict(success(request, _PREDICTION))
        if query:
            return _refused("query parameters are not supported")
        assert not body
        kind = "model" if collection == "models" else "predictor"
        if collection == "models":
            found = (
                {"name": item}
                if item
                else {"models": [{"name": "tabicl-v2-classifier"}]}
            )
        elif item is None:
            found = {"predictors": list(_PREDICTORS.values())}
        elif item in _PREDICTORS:
            found = _PREDICTORS[item]
        else:
            return 404, {
                "error": {"code": "NotFound", "message": "predictor not found"}
            }
        request = Request(
            Scope(project, headers["x-gf-request-id"]),
            Resource(kind, item or collection),
            {},
        )
        return 200, dict(success(request, found))


class _Reply(io.BytesIO):
    def __init__(self, status: int, value: dict[str, Any]) -> None:
        super().__init__(json.dumps(value).encode())
        self.code = status


def connect(asynchronous: bool, server: Callable[..., Any], **options):
    if asynchronous:

        def handler(request: httpx.Request) -> httpx.Response:
            status, value = server(
                request.method,
                str(request.url),
                {k.lower(): v for k, v in request.headers.items()},
                request.content,
            )
            return httpx.Response(status, json=value)

        return AsyncClient(http_transport=httpx.MockTransport(handler), **options)
    client = Client(**options)

    def open_request(request, **_kwargs):
        status, value = server(
            request.get_method(),
            request.full_url,
            {k.lower(): v for k, v in request.header_items()},
            request.data or b"",
        )
        return _Reply(status, value)

    cast(Any, client._transport._opener).open = open_request
    return client


def drive(client, *calls):
    async def run():
        results = []
        for call in calls:
            try:
                value = call(client)
                results.append(await value if inspect.isawaitable(value) else value)
            except ServiceError as error:
                results.append(error)
        close = getattr(client, "aclose", None) or client.close
        value = close()
        if inspect.isawaitable(value):
            await value
        return results

    return asyncio.run(run())


@pytest.fixture
def hosted_environment(monkeypatch, tmp_path):
    """Every ambient hosted credential and profile a local endpoint must ignore."""
    config = tmp_path / "profiles.json"
    config.write_text(
        json.dumps(
            {"default": {"url": HOSTED, "project": "acme", "key_env": "WORK_KEY"}}
        )
    )
    config.chmod(0o600)
    for variable, value in {
        "GF_CONFIG_PATH": str(config),
        "GF_URL": HOSTED,
        "GF_PROJECT": "acme",
        "GF_API_KEY": "hosted-api-key",
        "GF_TOKEN": "hosted-token",
        "WORK_KEY": "hosted-work-key",
        "GF_PLATFORM_TOKEN": "hosted-platform-token",
        "GF_MODAL_KEY": "hosted-modal-key",
        "GF_MODAL_SECRET": "hosted-modal-secret",
    }.items():
        monkeypatch.setenv(variable, value)
    return "hosted-"


def assert_no_hosted_secret(server: ModelServer, marker: str) -> None:
    for _method, _origin, _path, headers in server.seen:
        assert "cookie" not in headers
        assert not any(marker in value for value in headers.values())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_no_key_endpoint_lists_and_describes_without_authorization(
    asynchronous, hosted_environment
):
    server = ModelServer(LOCAL)
    client = connect(asynchronous, server, base_url=LOCAL)
    assert client.project == "local" and client.profile.key_env is None
    listed, model = drive(
        client,
        lambda c: c.models.list(),
        lambda c: c.models.get("tabicl-v2-classifier"),
    )
    assert listed == [{"name": "tabicl-v2-classifier"}]
    assert model == {"name": "tabicl-v2-classifier"}
    assert [(m, p) for m, _o, p, _h in server.seen] == [
        ("GET", "/v1/projects/local/models"),
        ("GET", "/v1/projects/local/models/tabicl-v2-classifier"),
    ]
    assert all("authorization" not in h for *_rest, h in server.seen)
    assert all(h["x-gf-project"] == "local" for *_rest, h in server.seen)
    assert_no_hosted_secret(server, hosted_environment)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_static_key_endpoint_sends_only_that_bearer_key(
    asynchronous, hosted_environment
):
    server = ModelServer(LOCAL)
    client = connect(
        asynchronous, server, base_url=LOCAL, api_key="serve-key", project="lab"
    )
    assert "serve-key" not in repr(client.profile)
    (described,) = drive(client, lambda c: c.models.get("tabicl-v2-classifier"))
    assert described == {"name": "tabicl-v2-classifier"}
    ((_method, _origin, path, headers),) = server.seen
    assert path == "/v1/projects/lab/models/tabicl-v2-classifier"
    assert headers["authorization"] == "Bearer serve-key"
    assert_no_hosted_secret(server, hosted_environment)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_hosted_only_operations_raise_typed_error_after_one_probe(
    asynchronous, hosted_environment
):
    server = ModelServer(LOCAL)
    client = connect(asynchronous, server, base_url=LOCAL)
    results = drive(
        client,
        lambda c: c.predictors.create("risk"),
        lambda c: c.keys.list(),
        lambda c: c.identity.manage("auth.status"),
        lambda c: c.models.list(),
    )
    *errors, listed = results
    assert listed == [{"name": "tabicl-v2-classifier"}]
    for error in errors:
        assert isinstance(error, HostedOnlyError)
        assert isinstance(error, ServiceError)
        assert (error.code, error.status) == ("UnsupportedOperation", 501)
        assert "standalone model server" in error.message
    assert errors[0].path == "/v1/projects/local/predictors"
    # One probe classifies the server; every request is still sent, so a
    # server that gains a route answers it.
    assert [p for _m, _o, p, _h in server.seen] == [
        "/v1/projects/local/predictors",
        PLATFORM_PROBE,
        "/v1/projects/local/keys",
        "/v1/auth/status",
        "/v1/projects/local/models",
    ]
    assert_no_hosted_secret(server, hosted_environment)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_missing_predictor_on_gf_serve_never_hides_its_other_routes(asynchronous):
    server = ModelServer(LOCAL)
    client = connect(asynchronous, server, base_url=LOCAL)
    missing, listed, found = drive(
        client,
        lambda c: c.predictors.get("typo"),
        lambda c: c.predictors.list(),
        lambda c: c.predictors.get("risk"),
    )
    assert isinstance(missing, gf.NotFound) and not isinstance(missing, HostedOnlyError)
    assert listed == [_PREDICTORS["risk"]]
    assert found.name == "risk"
    assert PLATFORM_PROBE not in [p for _m, _o, p, _h in server.seen]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_served_predictor_predicts_with_no_environment_named(asynchronous):
    server = ModelServer(LOCAL)
    client = connect(asynchronous, server, base_url=LOCAL)
    scored, staged = drive(
        client,
        lambda c: c.predictors.predict("risk", {"x": [1]}),
        lambda c: c.predictors.predict("risk", {"x": [1]}, env="staging"),
    )
    assert scored.table.num_rows == 1
    assert isinstance(staged, gf.InvalidInput) and "lock alone" in staged.message
    assert [p for _m, _o, p, _h in server.seen] == [
        "/v1/projects/local/predictors/risk/predict"
    ] * 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_probe_that_gets_no_answer_classifies_nothing(asynchronous):
    server = ModelServer(LOCAL)
    probes = []

    def answer(method, url, headers, body):
        if url.endswith(PLATFORM_PROBE) and not probes:
            probes.append(url)
            raise ServiceError("TransportError", "request failed", 503)
        return server(method, url, headers, body)

    client = connect(asynchronous, answer, base_url=LOCAL)
    first, second = drive(
        client, lambda c: c.keys.list(), lambda c: c.predictors.create("risk")
    )
    assert not isinstance(first, HostedOnlyError) and first.code == "NotFound"
    assert isinstance(second, HostedOnlyError)
    assert [p for _m, _o, p, _h in server.seen] == [
        "/v1/projects/local/keys",
        "/v1/projects/local/predictors",
        PLATFORM_PROBE,
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_hosted_missing_resource_stays_not_found(asynchronous):
    server = ModelServer(HOSTED, platform=True)
    client = connect(asynchronous, server, url=HOSTED, api_key="hosted-key")
    first, second = drive(
        client, lambda c: c.predictors.get("gone"), lambda c: c.keys.list()
    )
    for error in (first, second):
        assert isinstance(error, gf.NotFound) and error.code == "NotFound"
    assert [p for _m, _o, p, _h in server.seen] == [
        "/v1/projects/local/predictors/gone",
        "/v1/projects/local/keys",
        PLATFORM_PROBE,
    ]
    assert all(h["authorization"] == "Bearer hosted-key" for *_r, h in server.seen)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_hosted_profile_credentials_never_reach_another_origin(
    asynchronous, hosted_environment
):
    hosted = ModelServer(HOSTED, platform=True)
    profile_client = connect(asynchronous, hosted)
    assert profile_client.profile.url == HOSTED
    drive(profile_client, lambda c: c.models.list())
    assert hosted.seen[0][3]["authorization"] == "Bearer hosted-work-key"
    local = ModelServer(LOCAL)
    drive(connect(asynchronous, local, base_url=LOCAL), lambda c: c.models.list())
    assert local.seen and "authorization" not in local.seen[0][3]
    assert_no_hosted_secret(local, hosted_environment)


def test_base_url_is_explicit_and_keyless_profiles_round_trip(tmp_path):
    with pytest.raises(ServiceError):
        Client(base_url=LOCAL, url=HOSTED)
    with pytest.raises(ServiceError):
        Client(base_url=LOCAL, profile="default")
    with pytest.raises(ServiceError):
        Client(base_url=LOCAL, api_key="")
    config = tmp_path / "profiles.json"
    Profile(LOCAL, "local", key_env=None).save("serve", config)
    assert json.loads(config.read_text())["serve"]["key_env"] is None
    loaded = Profile.load("serve", config)
    assert loaded.key_env is None
    client = Client(profile=loaded)
    assert "Authorization" not in client._transport._headers


def test_portable_routes_are_exactly_what_gf_serve_resolves():
    portable = [
        ("GET", "/v1/projects/p/models"),
        ("GET", "/v1/projects/p/models/m"),
        ("GET", "/v1/projects/p/predictors"),
        ("GET", "/v1/projects/p/predictors/risk"),
        ("POST", "/v1/projects/p/predictors/risk/predict?env=staging"),
        ("POST", "/v1/projects/p/predict"),
        ("GET", "/livez"),
    ]
    platform = [
        ("POST", "/v1/projects/p/models"),
        ("POST", "/v1/projects/p/predictors"),
        ("POST", "/v1/projects/p/predictors/d/revisions"),
        ("GET", "/v1/projects/p/predictors/d/environments/production"),
        ("POST", "/v1/projects/p/predictions"),
        ("POST", "/v1/projects/p/models/m/infer"),
        ("GET", "/v1/projects/p/keys"),
        ("GET", "/v1/workspaces"),
        ("GET", PLATFORM_PROBE),
    ]
    assert not any(hosted_only(*route) for route in portable)
    assert all(hosted_only(*route) for route in platform)
