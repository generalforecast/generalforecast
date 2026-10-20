import asyncio
import inspect
import json
import ssl
import time
from dataclasses import FrozenInstanceError
from typing import Any, cast

import httpx
import pytest

from generalforecast import AsyncClient, Client, Profile, ServiceError
from generalforecast.deployments.interfaces import InferenceRequest
from generalforecast.models.catalog import ENTRIES, parse
from generalforecast.transport import AsyncTransport, Transport, endpoint


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
@pytest.mark.parametrize("options", [{"url": ""}, {"project": ""}, {"profile": ""}])
def test_explicit_empty_configuration_never_selects_ambient_defaults(
    client_type, options
):
    with pytest.raises(ServiceError):
        client_type(api_key="fixture-key", **options)


def test_sdk_reads_canonical_cli_profile_location(tmp_path, monkeypatch):
    root = tmp_path / "gf"
    root.mkdir()
    config = root / "profiles.json"
    config.touch(mode=0o600)
    config.write_text(
        json.dumps(
            {
                "research": {
                    "url": "http://localhost:8123",
                    "project": "research",
                    "key_env": "RESEARCH_KEY",
                }
            }
        )
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("RESEARCH_KEY", "credential-for-test")
    monkeypatch.setenv("GF_PROFILE", "research")
    for key in ("GF_URL", "GF_PROJECT", "GF_CONFIG_PATH"):
        monkeypatch.delenv(key, raising=False)
    client = Client()
    assert (
        client.project == "research"
        and client._transport.url == "http://localhost:8123"
    )
    assert "credential-for-test" not in repr(client.profile)
    client.close()
    config.chmod(0o644)
    with pytest.raises(ServiceError, match="private regular file"):
        Client()


def test_profile_malformed_collections_raise_typed_errors(tmp_path):
    path = tmp_path / "profiles.json"
    path.touch(mode=0o600)
    path.write_text("[]")
    with pytest.raises(ServiceError):
        Profile.load(path=path)


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
def test_profile_auth_precedence_and_no_secret_persistence(
    tmp_path, monkeypatch, client_type
):
    monkeypatch.delenv("GF_PROFILE", raising=False)
    for key in ("GF_URL", "GF_PROJECT", "GF_ENV", "GF_RUNTIME"):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "config.toml"
    config.touch(mode=0o600)
    config.write_text(
        '[profiles.work]\nurl="https://service.example"\nproject="research"\nenvironment="staging"\nruntime="cpu"\nkey_env="SDK_FIXTURE_KEY"\n'
    )
    monkeypatch.setenv("SDK_FIXTURE_KEY", "reference-key")
    monkeypatch.setenv("GF_URL", "https://override.example")
    monkeypatch.setenv("GF_ENV", "production")
    client = client_type(profile="work", config_path=config, project="explicit")
    assert client.project == "explicit"
    assert client.profile.environment == "production"
    assert client._transport.url == "https://override.example"
    assert client._transport._headers["Authorization"] == "Bearer reference-key"
    assert (
        "reference-key" not in repr(client) + repr(client.profile) + config.read_text()
    )
    with pytest.raises(FrozenInstanceError):
        client.profile.project = "mutated"
    if client_type is AsyncClient:
        asyncio.run(client.aclose())
    else:
        client.close()
    config.write_text('[profiles.work]\ntoken="forbidden-key"\n')
    with pytest.raises(ServiceError) as error:
        client_type(profile="work", config_path=config)
    assert "forbidden-key" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example",
        "https://user:pass@remote.example",
        "https://remote.example/path",
        "https://remote.example?key=sensitive",
        "https://remote.example#fragment",
        "https://remote.example:99999",
        "https://remote.example\n",
        True,
    ],
)
def test_endpoint_fail_closed_without_reflecting_credentials(url):
    with pytest.raises(ServiceError) as error:
        endpoint(url)
    assert "sensitive" not in str(error.value) and "pass@" not in str(error.value)


@pytest.mark.parametrize("transport_type", [Transport, AsyncTransport])
def test_tls_cannot_be_disabled_and_provider_layers_stay_distinct(transport_type):
    unsafe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    unsafe.check_hostname = False
    unsafe.verify_mode = ssl.CERT_NONE
    for kwargs in (
        {"tls_context": unsafe},
        {"modal_key": "only-one"},
        {"platform_token": "wrong-layer"},
    ):
        with pytest.raises(ServiceError):
            transport_type("https://service.example", token="fixture-key", **kwargs)
    value = transport_type(
        "https://service.example",
        token="fixture-key",
        auth_mode="azure-easyauth-and-api-key",
        platform_token="platform-key",
    )
    assert value._headers["Authorization"] == "Bearer platform-key"
    assert value._headers["X-GF-API-Key"] == "fixture-key"
    assert "fixture-key" not in repr(value) and "platform-key" not in repr(value)
    if transport_type is AsyncTransport:
        asyncio.run(value.aclose())
    else:
        value.close()


def test_async_native_socket_request_and_scoped_envelope(wire, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("async request used synchronous network transport")

    monkeypatch.setattr(Transport, "send", forbidden)
    wire.expect("GET", "/deployments/risk", {"name": "risk"})

    async def run():
        async with AsyncClient(
            wire.url, project="test", api_key="fixture-key"
        ) as client:
            assert (await client.deployments.get("risk")).name == "risk"
        assert client._transport._http.is_closed
        with pytest.raises(ServiceError) as error:
            await client.deployments.get("risk")
        assert error.value.code == "ClientClosed"

    asyncio.run(run())


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield b'{"partial":'
        self.started.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def test_async_cancellation_and_deadlines_close_streams_without_blocking_event_loop():
    async def run():
        for cancel in (True, False):
            stream = BlockingStream()
            ticks = 0

            async def heartbeat():
                nonlocal ticks
                while True:
                    ticks += 1
                    await asyncio.sleep(0.001)

            async def handler(_request, stream=stream):
                return httpx.Response(200, stream=stream)

            transport = AsyncTransport(
                "https://service.example",
                token="fixture-key",
                timeout=0.05,
                http_transport=httpx.MockTransport(handler),
            )
            beat = asyncio.create_task(heartbeat())
            task = asyncio.create_task(transport.send("GET", "/test", None, {}))
            await stream.started.wait()
            if cancel:
                await asyncio.sleep(0.005)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(ServiceError) as error:
                    await task
                assert error.value.code == "Timeout"
            assert stream.closed and ticks >= 2
            beat.cancel()
            with pytest.raises(asyncio.CancelledError):
                await beat
            await transport.aclose()
            assert transport._http.is_closed

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("identity", [None, "write_id", "submission_id"])
def test_retries_preserve_guarded_write_identity_and_never_retry_unguarded_posts(
    monkeypatch, asynchronous, identity
):
    body = {"payload": {identity: "stable-id"} if identity else {}}
    sent = []

    async def run():
        async def handler(request):
            sent.append((request.content, request.headers["X-Request-ID"]))
            if len(sent) == 1:
                return httpx.Response(
                    503,
                    json={
                        "error": {
                            "code": "ServiceUnavailable",
                            "message": "fixture-key",
                        }
                    },
                )
            return httpx.Response(200, json={})

        class Reply:
            code = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read(self, _size):
                return b"{}"

        def open_request(request, **_kwargs):
            sent.append((request.data, request.get_header("X-request-id")))
            if len(sent) == 1:
                raise ServiceError("ServiceUnavailable", "fixture failure", 503)
            return Reply()

        if asynchronous:
            transport = AsyncTransport(
                "https://service.example",
                token="fixture-key",
                retries=2,
                retry_delay=0,
                http_transport=httpx.MockTransport(handler),
            )
        else:
            transport = Transport(
                "https://service.example", token="fixture-key", retries=2, retry_delay=0
            )
            monkeypatch.setattr(transport._opener, "open", open_request)
        try:
            if identity:
                value = transport.send(
                    "POST", "/test", body, {"X-Request-ID": "same-request"}
                )
                assert (await value if inspect.isawaitable(value) else value) == {}
                assert len(sent) == 2 and sent[0] == sent[1]
                assert json.loads(sent[0][0])["payload"][identity] == "stable-id"
            else:
                with pytest.raises(ServiceError):
                    value = transport.send(
                        "POST", "/test", body, {"X-Request-ID": "same-request"}
                    )
                    if inspect.isawaitable(value):
                        await value
                assert len(sent) == 1
        finally:
            if isinstance(transport, AsyncTransport):
                await transport.aclose()
            else:
                transport.close()

    asyncio.run(run())


def test_async_total_deadline_includes_retry_backoff():
    async def run():
        calls = []

        async def handler(request):
            calls.append(request)
            return httpx.Response(
                503,
                json={
                    "error": {"code": "ServiceUnavailable", "message": "unavailable"}
                },
            )

        transport = AsyncTransport(
            "https://service.example",
            token="fixture-key",
            timeout=0.5,
            retries=5,
            retry_delay=1,
            http_transport=httpx.MockTransport(handler),
        )
        start = time.monotonic()
        with pytest.raises(ServiceError) as error:
            await transport.send("GET", "/test", None, {}, deadline=start + 0.03)
        assert error.value.code == "Timeout" and len(calls) == 1
        assert time.monotonic() - start < 0.3
        await transport.aclose()

    asyncio.run(run())


def test_async_redirect_revocation_cookie_and_response_bounds(monkeypatch):
    import generalforecast.transport as module

    async def run():
        calls = []

        async def handler(request):
            calls.append(request)
            assert "Cookie" not in request.headers
            if request.url.path == "/redirect":
                return httpx.Response(
                    302,
                    headers={
                        "Location": "https://other.example",
                        "Set-Cookie": "session=untrusted",
                    },
                )
            if request.url.path == "/revoked":
                return httpx.Response(
                    401,
                    json={"error": {"code": "Unauthorized", "message": "fixture-key"}},
                )
            return httpx.Response(200, json={"data": "a" * 1000})

        transport = AsyncTransport(
            "https://service.example",
            token="fixture-key",
            retries=5,
            http_transport=httpx.MockTransport(handler),
        )
        for path, code in (
            ("/redirect", "RedirectRefused"),
            ("/revoked", "Unauthorized"),
            ("/large", "ProtocolError"),
        ):
            if path == "/large":
                monkeypatch.setattr(module, "MAX_BYTES", 20)
            with pytest.raises(ServiceError) as error:
                await transport.send("GET", path, None, {})
            assert error.value.code == code and "fixture-key" not in str(error.value)
        assert len(calls) == 3
        await transport.aclose()

    asyncio.run(run())


def test_closed_sync_client_and_invalid_profile_values_fail_typed():
    with Client(api_key="fixture-key") as client:
        assert not client._transport.closed
    with pytest.raises(ServiceError):
        client.models.list()
    with pytest.raises(ServiceError):
        Profile(url=cast(Any, True))


def _private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "environ,key_env,expected",
    [
        ({"GF_API_KEY": "primary", "GF_TOKEN": "fallback"}, "GF_API_KEY", "primary"),
        ({"GF_TOKEN": "fallback"}, "GF_API_KEY", "fallback"),
        ({"GF_API_KEY": "", "GF_TOKEN": "fallback"}, "GF_API_KEY", None),
        ({"WORK_KEY": "work", "GF_TOKEN": "fallback"}, "WORK_KEY", "work"),
        ({"GF_TOKEN": "fallback", "GF_API_KEY": "ambient"}, "WORK_KEY", None),
        ({"WORK_KEY": "", "GF_TOKEN": "fallback"}, "WORK_KEY", None),
    ],
)
def test_one_credential_rule_for_sdk_cli_and_transport(environ, key_env, expected):
    from generalforecast.transport import ConnectionSettings

    profile = Profile(key_env=key_env)
    if expected is None:
        with pytest.raises(ServiceError) as error:
            ConnectionSettings.resolve(profile=profile, environ=environ)
        assert error.value.code == "ValidationError"
        assert "fallback" not in str(error.value)
    else:
        settings = ConnectionSettings.resolve(profile=profile, environ=environ)
        assert settings.token == expected and expected not in repr(settings)
    explicit = ConnectionSettings.resolve(
        profile=profile, api_key="explicit", environ=environ
    )
    assert explicit.token == "explicit" and explicit.credential_env is None


def test_resolver_precedence_uses_explicit_mapping_not_process_env(
    tmp_path, monkeypatch
):
    from generalforecast.transport import ConnectionSettings

    config = tmp_path / "profiles.json"
    _private_json(
        config,
        {
            "work": {
                "url": "https://profile.example",
                "project": "stored",
                "environment": "staging",
                "runtime": "cpu",
                "key_env": "WORK_KEY",
            }
        },
    )
    monkeypatch.setenv("GF_URL", "https://process.example")
    environ = {
        "GF_CONFIG_PATH": str(config),
        "GF_PROFILE": "work",
        "GF_PROJECT": "from-env",
        "WORK_KEY": "work-key",
    }
    settings = ConnectionSettings.resolve(environ=environ)
    assert settings.profile == Profile(
        "https://profile.example", "from-env", "staging", "cpu", "WORK_KEY"
    )
    assert settings.profile_name == "work" and settings.credential_env == "WORK_KEY"
    explicit = ConnectionSettings.resolve(
        "https://explicit.example", "explicit", environ=environ
    )
    assert (explicit.profile.url, explicit.profile.project) == (
        "https://explicit.example",
        "explicit",
    )
    with pytest.raises(ServiceError):
        ConnectionSettings.resolve(profile="missing", environ=environ)
    assert (
        ConnectionSettings.resolve(
            profile="missing", new_profile=True, environ=environ | {"GF_API_KEY": "k"}
        ).profile_name
        == "missing"
    )


def test_profile_save_preserves_other_values_and_rejects_toml(tmp_path):
    config = tmp_path / "gf" / "profiles.json"
    _private_json(
        config,
        {"other": {"url": "https://other.example", "project": "p", "runtime": "gpu"}},
    )
    Profile("https://work.example", "work", "staging", key_env="WORK_KEY").save(
        "work", config
    )
    stored = json.loads(config.read_text())
    assert stored["other"] == {
        "url": "https://other.example",
        "project": "p",
        "runtime": "gpu",
    }
    assert stored["work"] == {
        "url": "https://work.example",
        "project": "work",
        "key_env": "WORK_KEY",
        "environment": "staging",
    }
    assert config.stat().st_mode & 0o777 == 0o600
    assert Profile.load("work", config).environment == "staging"
    toml = tmp_path / "profiles.toml"
    toml.touch(mode=0o600)
    with pytest.raises(ServiceError) as error:
        Profile().save("default", toml)
    assert error.value.code == "UnsupportedOperation"
    config.write_text(json.dumps({"bad": {"url": "https://x.example", "token": "x"}}))
    with pytest.raises(ServiceError):
        Profile().save("default", config)
    assert "bad" in config.read_text()


def test_concurrent_profile_saves_never_lose_updates(tmp_path):
    import threading

    config = tmp_path / "profiles.json"
    barrier = threading.Barrier(8)

    def save(index):
        barrier.wait()
        for round_ in range(5):
            Profile(project=f"p{index}-{round_}").save(f"profile-{index}", config)

    threads = [threading.Thread(target=save, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    profiles = Profile.load_all(config)
    assert {name: value["project"] for name, value in profiles.items()} == {
        f"profile-{i}": f"p{i}-4" for i in range(8)
    }


@pytest.mark.parametrize("locked", [False, True])
def test_profile_lock_prevents_reproducible_lost_update(tmp_path, monkeypatch, locked):
    import contextlib
    import threading

    from generalforecast import transport

    config = tmp_path / "profiles.json"
    Profile().save("seed", config)
    real_write = transport.write_private
    barrier = threading.Barrier(2, timeout=0.5)

    def interleaved_write(path, data):
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait()
        real_write(path, data)

    monkeypatch.setattr(transport, "write_private", interleaved_write)
    if not locked:
        monkeypatch.setattr(
            transport, "private_lock", lambda _path: contextlib.nullcontext()
        )
    threads = [
        threading.Thread(target=Profile().save, args=(f"p{i}", config))
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stored = set(Profile.load_all(config))
    if locked:
        assert stored == {"seed", "p0", "p1"}
    else:
        assert stored < {"seed", "p0", "p1"} and len(stored) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "model",
    [
        "tabicl-classifier@4",
        "tabicl-classifier@4*2",
        "tabicl-regressor*2",
        f"catalog/tabicl-v2-classifier@{ENTRIES['tabicl-v2-classifier'].artifact.digest}@4",
    ],
)
def test_variant_resource_binding_keeps_the_full_reference(
    monkeypatch, asynchronous, model
):
    request: InferenceRequest = {
        "model": model,
        "task": "regression" if "regressor" in model else "classification",
        "target": "y",
        "features": ["x"],
        "context": [],
        "queries": [],
        "options": {},
    }
    captured = []
    reply = {"outputs": []}

    def post(resource, action, payload):
        captured.append((resource, action, payload))
        return reply

    async def async_post(resource, action, payload):
        return post(resource, action, payload)

    if asynchronous:
        client = AsyncClient(url="http://localhost", project="local", token="fixture")
        monkeypatch.setattr(client.models, "_post", async_post)
        result = asyncio.run(client.models.infer(request))
    else:
        client = Client(url="http://localhost", project="local", token="fixture")
        monkeypatch.setattr(client.models, "_post", post)
        result = client.models.infer(request)
    assert result is reply
    resource, action, payload = captured.pop()
    assert (resource, action) == (parse(model).name, "infer")
    assert payload == request and payload is not request
