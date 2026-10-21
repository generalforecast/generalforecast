"""Standalone `gf serve` endpoints through fake sync and async transports."""

import asyncio
import inspect
import io
import json
import re
from typing import Any, cast

import httpx
import pytest

import generalforecast as gf
from generalforecast.deployments.boundary import Request, Resource, Scope, success
from generalforecast.deployments.interfaces import InferenceRequest
from generalforecast.transport import PLATFORM_PROBE, hosted_only

HOSTED = "https://hosted.example"
LOCAL = "http://127.0.0.1:8123"
INFERENCE = cast(
    InferenceRequest,
    {
        "model": "tabicl-v2-classifier",
        "task": "classification",
        "target": "y",
        "features": ["x"],
        "context": [{"x": 1, "y": "a"}],
        "queries": [{"x": 2}],
        "options": {},
    },
)
RESULT = {"model": "tabicl-v2-classifier", "predictions": [{"label": "a"}]}
_MODEL_ROUTE = re.compile(r"/v1alpha1/projects/([^/]+)/models(?:/([^/]+)(/infer)?)?")


class ModelServer:
    """Answers like a plain model server, or like the hosted platform."""

    def __init__(self, origin: str, *, platform: bool = False) -> None:
        self.origin, self.platform, self.seen = origin, platform, []

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> tuple[int, dict[str, Any]]:
        origin, _, rest = url.partition("/v1alpha1")
        path = "/v1alpha1" + rest
        self.seen.append((method, origin, path, headers))
        assert origin == self.origin
        if path == PLATFORM_PROBE:
            if self.platform:
                return 401, {"error": {"code": "Unauthenticated", "message": "key"}}
            return 404, {"error": {"code": "NotFound", "message": "route not found"}}
        route = _MODEL_ROUTE.fullmatch(path)
        if route is None:
            return 404, {"error": {"code": "NotFound", "message": "route not found"}}
        project, model, infer = route.groups()
        if method == "POST" and infer:
            request = Request.parse(json.loads(body))
            assert request.payload == dict(INFERENCE)
            return 200, dict(success(request, RESULT))
        assert method == "GET" and not infer and not body
        request = Request(
            Scope(project, headers["x-gf-request-id"]),
            Resource("model", model or "models"),
            {},
        )
        listing = {"models": [{"name": "tabicl-v2-classifier"}]}
        return 200, dict(success(request, {"name": model} if model else listing))


class _Reply(io.BytesIO):
    def __init__(self, status: int, value: dict[str, Any]) -> None:
        super().__init__(json.dumps(value).encode())
        self.code = status


def connect(asynchronous: bool, server: ModelServer, **options):
    if asynchronous:

        def handler(request: httpx.Request) -> httpx.Response:
            status, value = server(
                request.method,
                str(request.url),
                {k.lower(): v for k, v in request.headers.items()},
                request.content,
            )
            return httpx.Response(status, json=value)

        return gf.AsyncClient(http_transport=httpx.MockTransport(handler), **options)
    client = gf.Client(**options)

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
            except gf.ServiceError as error:
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
def test_no_key_endpoint_lists_and_infers_without_authorization(
    asynchronous, hosted_environment
):
    server = ModelServer(LOCAL)
    client = connect(asynchronous, server, base_url=LOCAL)
    assert client.project == "local" and client.profile.key_env is None
    listed, model, inferred = drive(
        client,
        lambda c: c.models.list(),
        lambda c: c.models.get("tabicl-v2-classifier"),
        lambda c: c.models.infer(INFERENCE),
    )
    assert listed == [{"name": "tabicl-v2-classifier"}]
    assert model == {"name": "tabicl-v2-classifier"}
    assert inferred == RESULT
    assert [(m, p) for m, _o, p, _h in server.seen] == [
        ("GET", "/v1alpha1/projects/local/models"),
        ("GET", "/v1alpha1/projects/local/models/tabicl-v2-classifier"),
        ("POST", "/v1alpha1/projects/local/models/tabicl-v2-classifier/infer"),
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
    (inferred,) = drive(client, lambda c: c.models.infer(INFERENCE))
    assert inferred == RESULT
    ((_method, _origin, path, headers),) = server.seen
    assert path == "/v1alpha1/projects/lab/models/tabicl-v2-classifier/infer"
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
        lambda c: c.deployments.list(),
        lambda c: c.dev_sessions.start("risk", base_revision=1, runtime="cpu"),
        lambda c: c.keys.list(),
        lambda c: c.identity.manage("auth.status"),
        lambda c: c.models.list(),
    )
    *errors, listed = results
    assert listed == [{"name": "tabicl-v2-classifier"}]
    for error in errors:
        assert isinstance(error, gf.HostedOnlyError)
        assert isinstance(error, gf.ServiceError)
        assert (error.code, error.status) == ("UnsupportedOperation", 501)
        assert "standalone model server" in error.message
    assert errors[0].path == "/v1alpha1/projects/local/deployments"
    assert [p for _m, _o, p, _h in server.seen] == [
        "/v1alpha1/projects/local/deployments",
        PLATFORM_PROBE,
        "/v1alpha1/projects/local/models",
    ]
    assert_no_hosted_secret(server, hosted_environment)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_hosted_missing_resource_stays_not_found(asynchronous):
    server = ModelServer(HOSTED, platform=True)
    client = connect(asynchronous, server, url=HOSTED, api_key="hosted-key")
    first, second = drive(
        client, lambda c: c.deployments.get("gone"), lambda c: c.keys.list()
    )
    for error in (first, second):
        assert isinstance(error, gf.ServiceError) and error.code == "NotFound"
    assert [p for _m, _o, p, _h in server.seen] == [
        "/v1alpha1/projects/local/deployments/gone",
        PLATFORM_PROBE,
        "/v1alpha1/projects/local/keys",
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
    with pytest.raises(gf.ServiceError):
        gf.Client(base_url=LOCAL, url=HOSTED)
    with pytest.raises(gf.ServiceError):
        gf.Client(base_url=LOCAL, profile="default")
    with pytest.raises(gf.ServiceError):
        gf.Client(base_url=LOCAL, api_key="")
    config = tmp_path / "profiles.json"
    gf.Profile(LOCAL, "local", key_env=None).save("serve", config)
    assert json.loads(config.read_text())["serve"]["key_env"] is None
    loaded = gf.Profile.load("serve", config)
    assert loaded.key_env is None
    client = gf.Client(profile=loaded)
    assert "Authorization" not in client._transport._headers


def test_portable_routes_are_exactly_model_listing_and_inference():
    portable = [
        ("GET", "/v1alpha1/projects/p/models"),
        ("GET", "/v1alpha1/projects/p/models/m?view=full"),
        ("POST", "/v1alpha1/projects/p/models/m/infer"),
        ("POST", "/v1alpha1/projects/p/predictions"),
        ("GET", "/livez"),
    ]
    platform = [
        ("POST", "/v1alpha1/projects/p/models"),
        ("POST", "/v1alpha1/projects/p/deployments/d/revisions"),
        ("POST", "/v1alpha1/projects/p/deployments/d/dev-sessions"),
        ("GET", "/v1alpha1/projects/p/keys"),
        ("POST", "/v1alpha1/projects/p/jobs"),
        ("GET", "/v1alpha1/workspaces"),
        ("GET", PLATFORM_PROBE),
    ]
    assert not any(hosted_only(*route) for route in portable)
    assert all(hosted_only(*route) for route in platform)
