"""Independent regressions for canonical payloads, snapshot guards and transports."""

import asyncio
import json
import threading

import httpx
import pytest

import generalforecast as gf
from generalforecast.async_client import AsyncClient, AsyncDeployment
from generalforecast.transport import AsyncTransport, response_value
from tests.product_sdk.test_lifecycle import invoke


@pytest.mark.parametrize("asynchronous", [False, True])
def test_retired_deployment_plane_is_not_exposed(asynchronous):
    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(api_key="fixture-key")
        try:
            for attribute in (
                "apply",
                "score",
                "wait_ready",
                "effect_status",
                "wait_effect",
            ):
                assert not hasattr(client.deployments, attribute)
            assert not hasattr(gf, "ServingEffect")
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


def result(task="classification"):
    return gf.PredictionResult.from_dict(
        {
            "model": "catalog/chronos-2"
            if task == "forecast"
            else "catalog/tabicl-v2-classifier",
            "task": task,
            "context": None,
            "deployment": None,
            "revision": None,
            "environment": None,
            "predictions": [{"value": 2.0}]
            if task == "forecast"
            else [{"label": True}],
        }
    ).to_dict()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_one_shot_forecast_uses_declared_roles_and_nested_options(wire, asynchronous):
    options = {
        "time": "t",
        "series": ["s"],
        "horizon": 1,
        "frequency": 1,
        "past_inputs": ["past"],
        "future_inputs": ["future"],
        "quantiles": [0.5],
    }
    context = [{"t": 1, "s": "a", "y": 2.0, "past": 3.0, "future": 4.0}]
    queries = [{"t": 2, "s": "a", "future": 5.0}]
    wire.expect(
        "POST",
        "/predictions",
        result("forecast"),
        payload={
            "model": "chronos-2",
            "task": "forecast",
            "target": "y",
            "features": [],
            "context": context,
            "queries": queries,
            "options": options | {"outputs": ["value"]},
        },
    )

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            prediction = await invoke(
                client.predict,
                model="chronos-2",
                target="y",
                context=context,
                queries=queries,
                options=options,
                outputs=["value"],
            )
            assert prediction.task == "forecast"
            assert "outputs" not in options
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_named_prediction_retains_optional_request_bindings(wire, asynchronous):
    payload = {
        "queries": [{"x": 2}],
        "context": [{"x": 1, "y": True}],
        "target": "y",
        "features": ["x"],
    }
    for path in (
        "/deployments/risk/environments/staging/predict",
        "/deployments/risk/dev-sessions/dev-one/predict",
    ):
        wire.expect("POST", path, result(), payload=payload)

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            bindings = {
                "context": [{"x": 1, "y": True}],
                "target": "y",
                "features": ["x"],
            }
            await invoke(
                client.deployments.predict,
                "risk",
                {"x": [2]},
                env="staging",
                **bindings,
            )
            await invoke(
                client.dev_sessions.predict, "risk", "dev-one", {"x": [2]}, **bindings
            )
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_captured_environment_never_lends_other_environment_its_cas(wire, asynchronous):
    captured = gf.Environment("risk", "production", 1, 3, "local", "ready")
    staging = gf.Environment("risk", "staging", 2, 9, "local", "ready")
    base = "/deployments/risk/environments/staging"
    for action in ("promote", "rollback"):
        wire.expect("GET", base, staging.to_dict())
        wire.expect(
            "POST",
            base + "/" + action,
            staging.to_dict(),
            payload={"revision": 2, "expected_generation": 9},
        )
    wire.expect("GET", base, staging.to_dict())
    wire.expect("GET", "/deployments/risk/environments/production", captured.to_dict())
    moved = gf.Environment("risk", "production", 2, 4, "local", "ready")
    wire.expect("GET", "/deployments/risk/environments/production", moved.to_dict())
    wire.expect(
        "POST",
        "/deployments/risk/environments/production/promote",
        moved.to_dict(),
        payload={"revision": 2, "expected_generation": 3},
    )

    async def run():
        client = (gf.AsyncClient if asynchronous else gf.Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            handle = (AsyncDeployment if asynchronous else gf.Deployment).from_response(
                client, "risk", {"name": "risk"}, captured
            )
            for action in ("promote", "rollback"):
                assert (
                    await invoke(getattr(handle, action), env="staging", revision=2)
                ).generation == 9
            assert (await invoke(handle.wait, env="staging")).revision == 2
            assert (await invoke(handle.wait, env=None)).revision == 1
            with pytest.raises(gf.ServiceError) as error:
                await invoke(handle.wait, revision=None, generation=None)
            assert error.value.code == "Conflict"
            assert (
                await invoke(handle.promote, revision=2, expected_generation=None)
            ).generation == 4
            assert handle.environment == captured
        finally:
            await invoke(
                client.aclose if isinstance(client, gf.AsyncClient) else client.close
            )

    asyncio.run(run())


def test_error_code_is_not_an_untrusted_credential_echo():
    for code in (
        "fixture-secret",
        "Bearer fixture-secret",
        "UnpublishedCredentialValue",
    ):
        with pytest.raises(gf.ServiceError) as error:
            response_value(
                401, json.dumps({"error": {"code": code, "message": code}}).encode()
            )
        assert error.value.code == "RemoteError"
        assert code not in repr(error.value.body())


def test_async_refuses_compressed_bodies_before_consuming_or_decoding():
    class NeverRead(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            pytest.fail("compressed body must not be decoded")
            yield b""

        async def aclose(self):
            self.closed = True

    async def run():
        stream = NeverRead()

        async def handler(request):
            assert request.headers["Accept-Encoding"] == "identity"
            return httpx.Response(
                200, headers={"Content-Encoding": "gzip"}, stream=stream
            )

        transport = AsyncTransport(
            "https://service.example",
            token="fixture-key",
            http_transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(gf.ServiceError) as error:
                await transport.send("GET", "/test", None, {})
            assert error.value.code == "ProtocolError" and stream.closed
        finally:
            await transport.aclose()

    asyncio.run(run())


def test_async_snapshot_encoding_and_digest_do_not_block_event_loop(wire, monkeypatch):
    import generalforecast.async_client as module

    original_arrow, original_digest = module.arrow_bytes, module.table_digest
    main_thread = threading.get_ident()
    calls = []

    def off_loop(function, data):
        assert threading.get_ident() != main_thread
        calls.append(function)
        return function(data)

    monkeypatch.setattr(
        module, "arrow_bytes", lambda data: off_loop(original_arrow, data)
    )
    monkeypatch.setattr(
        module, "table_digest", lambda data: off_loop(original_digest, data)
    )
    wire.expect("POST", "/snapshots")

    async def run():
        async with gf.AsyncClient(
            wire.url, project="test", api_key="fixture-key"
        ) as client:
            assert (await client.upload_snapshot({"x": [1]}))["rows"] == 1

    asyncio.run(run())
    assert calls == [original_arrow, original_digest]
