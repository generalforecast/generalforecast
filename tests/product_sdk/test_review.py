# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent regressions for canonical payloads, snapshot guards and transports."""

import asyncio
import json
import threading

import httpx
import pytest

import generalforecast as gf
from generalforecast.async_client import AsyncClient, AsyncHostedPredictor
from generalforecast.client import Client
from generalforecast.contracts import Environment, PredictionResult
from generalforecast.service.contracts import ServiceError
from generalforecast.service.sdk import HostedPredictor
from generalforecast.transport import (
    AsyncTransport,
    RelayedError,
    printable,
    response_value,
)
from tests.product_sdk.test_lifecycle import invoke


@pytest.mark.parametrize("asynchronous", [False, True])
def test_retired_deployment_plane_is_not_exposed(asynchronous):
    async def run():
        client = (AsyncClient if asynchronous else Client)(api_key="fixture-key")
        try:
            for attribute in (
                "apply",
                "score",
                "wait_ready",
                "effect_status",
                "wait_effect",
            ):
                assert not hasattr(client.predictors, attribute)
            assert not hasattr(gf, "ServingEffect")
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


def result(task="classification"):
    return PredictionResult.from_dict(
        {
            "model": "general-forecast/chronos-2"
            if task == "forecast"
            else "general-forecast/tabicl-v2-classifier",
            "task": task,
            "context": None,
            "predictor": None,
            "revision": None,
            "environment": None,
            "predictions": [{"value": 2.0}]
            if task == "forecast"
            else [{"label": True}],
        }
    ).to_dict()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_one_shot_forecast_uses_declared_roles_and_nested_options(wire, asynchronous):
    context = [{"t": 1, "s": "a", "y": 2.0, "past": 3.0, "future": 4.0}]
    queries = [{"t": 2, "s": "a", "future": 5.0}]
    wire.expect(
        "POST",
        "/predict",
        result("forecast"),
        payload={
            "model": "general-forecast/chronos-2",
            "target": "y",
            "context": context,
            "queries": queries,
            "inputs": ["future"],
            "time": "t",
            "frequency": 1,
            "horizon": 1,
            "series": ["s"],
            "quantiles": [0.5],
            "outputs": ["value"],
        },
    )

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            prediction = await invoke(
                client.predict,
                "general-forecast/chronos-2",
                "y",
                context,
                queries,
                inputs=["future"],
                time="t",
                series=["s"],
                frequency=1,
                horizon=1,
                quantiles=[0.5],
                outputs=["value"],
            )
            assert prediction.task == "forecast"
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_named_prediction_retains_optional_request_bindings(wire, asynchronous):
    payload = {"queries": [{"x": 2}], "context": [{"x": 1, "y": True}]}
    wire.expect(
        "POST", "/predictors/risk/predict?env=staging", result(), payload=payload
    )

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            bindings = {"context": [{"x": 1, "y": True}]}
            await invoke(
                client.predictors.predict,
                "risk",
                {"x": [2]},
                env="staging",
                **bindings,
            )
            with pytest.raises(TypeError):
                await invoke(client.predictors.predict, "risk", {"x": [2]}, target="y")
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_captured_environment_never_lends_other_environment_its_cas(wire, asynchronous):
    captured = Environment("risk", "production", 1, 3, "local", "ready")
    staging = Environment("risk", "staging", 2, 9, "local", "ready")
    base = "/predictors/risk/environments/staging"
    for action in ("promote", "rollback"):
        wire.expect("GET", base, staging.to_dict())
        wire.expect(
            "POST",
            base + "/" + action,
            staging.to_dict(),
            payload={"revision": 2, "expected_generation": 9},
        )
    wire.expect("GET", base, staging.to_dict())
    wire.expect("GET", "/predictors/risk/environments/production", captured.to_dict())
    moved = Environment("risk", "production", 2, 4, "local", "ready")
    wire.expect("GET", "/predictors/risk/environments/production", moved.to_dict())
    wire.expect(
        "POST",
        "/predictors/risk/environments/production/promote",
        moved.to_dict(),
        payload={"revision": 2, "expected_generation": 3},
    )

    async def run():
        client = (AsyncClient if asynchronous else Client)(
            wire.url, project="test", api_key="fixture-key"
        )
        try:
            handle = (
                AsyncHostedPredictor if asynchronous else HostedPredictor
            ).from_response(client, "risk", {"name": "risk"}, captured)
            for action in ("promote", "rollback"):
                assert (
                    await invoke(getattr(handle, action), env="staging", revision=2)
                ).generation == 9
            assert (await invoke(handle.wait, env="staging")).revision == 2
            assert (await invoke(handle.wait, env=None)).revision == 1
            with pytest.raises(ServiceError) as error:
                await invoke(handle.wait, revision=None, generation=None)
            assert error.value.code == "Conflict"
            assert (
                await invoke(handle.promote, revision=2, expected_generation=None)
            ).generation == 4
            assert handle.environment == captured
        finally:
            await invoke(
                client.aclose if isinstance(client, AsyncClient) else client.close
            )

    asyncio.run(run())


def test_error_code_is_not_an_untrusted_credential_echo():
    for code in (
        "fixture-secret",
        "Bearer fixture-secret",
        "UnpublishedCredentialValue",
    ):
        with pytest.raises(ServiceError) as error:
            response_value(
                401,
                json.dumps(
                    {"error": {"code": code, "message": "permission denied"}}
                ).encode(),
            )
        assert error.value.code == "RemoteError"
        assert str(error.value) == "permission denied"
        assert code not in repr(error.value.body())


@pytest.mark.parametrize(
    "message,expected",
    [
        ("\x1b[31mred\x1b[0m", "red"),
        ("\x1b]0;title\x07ok", "ok"),
        ("\x9b2Jx", "x"),
        ("a\x00\u202eb\nc", "ab c"),
    ],
)
def test_printable_removes_terminal_escapes_and_controls(message, expected):
    assert printable(message) == expected


@pytest.mark.parametrize("message", ["x" * 513, "", "\x00\u202e"])
def test_response_value_falls_back_for_long_or_empty_printable_messages(message):
    raw = json.dumps({"error": {"code": "Forbidden", "message": message}}).encode()
    with pytest.raises(ServiceError) as error:
        response_value(403, raw)
    assert not isinstance(error.value, RelayedError)
    assert str(error.value) == "service rejected request"


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
            with pytest.raises(ServiceError) as error:
                await transport.send("GET", "/test", None, {})
            assert error.value.code == "ProtocolError" and stream.closed
        finally:
            await transport.aclose()

    asyncio.run(run())


def test_async_snapshot_encoding_and_digest_do_not_block_event_loop(wire, monkeypatch):
    import generalforecast._operations as module

    main_thread = threading.get_ident()
    calls = []
    for name in ("table", "arrow_bytes", "bytes_digest"):
        original = getattr(module, name)

        def off_loop(data, original=original):
            assert threading.get_ident() != main_thread
            calls.append(original.__name__)
            return original(data)

        monkeypatch.setattr(module, name, off_loop)
    wire.expect("POST", "/snapshots")

    async def run():
        async with AsyncClient(
            wire.url, project="test", api_key="fixture-key"
        ) as client:
            assert (await client.upload_snapshot({"x": [1]}))["rows"] == 1

    asyncio.run(run())
    assert calls == ["table", "arrow_bytes", "bytes_digest"]
