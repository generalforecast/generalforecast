# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Client behaviours that match the hosted handlers, in both clients."""

import asyncio
import inspect
import time

import pytest

import generalforecast as gf
from generalforecast.async_client import AsyncClient
from generalforecast.client import Client
from generalforecast.contracts import Environment
from generalforecast.service.contracts import ServiceError
from generalforecast.transport import Profile

PREDICTION = {
    "model": "general-forecast/tabicl-v2-classifier",
    "task": "classification",
    "predictions": [{"label": True, "probabilities": {"false": 0.1, "true": 0.9}}],
}


def run(asynchronous: bool, url: str, body, *, environment: str = "production"):
    async def go():
        profile = Profile(url, "test", environment=environment)
        client = (AsyncClient if asynchronous else Client)(
            profile=profile, api_key="fixture-key"
        )
        try:
            value = body(client)
            return await value if inspect.isawaitable(value) else value
        finally:
            if isinstance(client, AsyncClient):
                await client.aclose()
            else:
                client.close()

    return asyncio.run(go())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_predict_uses_the_profile_environment(wire, asynchronous):
    for path in ("?env=staging", "", ""):
        wire.expect(
            "POST",
            "/predictors/risk/predict" + path,
            PREDICTION,
            payload={"queries": [{"x": 3}]},
        )

    async def body(client):
        # The profile's staging, then production by name; production, the
        # services' default, is named by omission.
        for env in (None, "production"):
            value = client.predictors.predict("risk", {"x": [3]}, env=env)
            await value if inspect.isawaitable(value) else value

    run(asynchronous, wire.url, body, environment="staging")
    run(
        asynchronous,
        wire.url,
        lambda c: c.predictors.predict("risk", {"x": [3]}),
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_with_no_time_left_checks_once_then_times_out(wire, asynchronous):
    pending = Environment("risk", "production", 1, 1, "local", "pending")
    wire.expect("GET", "/predictors/risk/environments/production", pending.to_dict())
    with pytest.raises(ServiceError) as error:
        run(asynchronous, wire.url, lambda c: c.predictors.wait("risk", timeout=0))
    assert error.value.code == "WaitTimeout" and isinstance(
        error.value, gf.ModelNotReady
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_raises_wait_timeout_when_the_deadline_passes_during_a_poll(asynchronous):
    def slow(method, path, body, headers, *, deadline=None):
        assert deadline is not None
        time.sleep(max(0.0, deadline - time.monotonic()))
        raise ServiceError("Timeout", "request deadline exceeded", 504)

    async def slow_async(*args, **kwargs):
        return slow(*args, **kwargs)

    def body(client):
        client._transport.send = slow_async if asynchronous else slow
        return client.predictors.wait("risk", timeout=0.05)

    with pytest.raises(ServiceError) as error:
        run(asynchronous, "https://service.example", body)
    assert error.value.code == "WaitTimeout"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_predictor_reports_hosted_facts(wire, asynchronous):
    environment = Environment("risk", "production", 2, 3, "local", "ready")
    wire.expect(
        "GET",
        "/predictors/risk",
        {
            "name": "risk",
            "created_at": 1.0,
            "latest_revision": 2,
            "environments": [environment.to_dict()],
        },
    )
    predictor = run(asynchronous, wire.url, lambda c: c.predictors.get("risk"))
    assert predictor.revision == 2
    assert predictor.status is None
    assert predictor.environments == (environment,)


def test_keys_are_listed_never_read_one_at_a_time():
    client = Client("https://service.example", api_key="fixture-key")
    assert not hasattr(client.keys, "get")
    with pytest.raises(ServiceError) as error:
        client.identity.manage("keys.get", {"id": "key-1"})
    assert error.value.code == "UnsupportedOperation"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("action", "payload"),
    [
        ("memberships.list", None),
        ("memberships.add", {"issuer": "i", "subject": "s", "role": "admin"}),
        ("memberships.remove", {"issuer": "i", "subject": "s"}),
    ],
)
def test_the_retired_memberships_api_is_refused_before_any_request(
    asynchronous, action, payload
):
    # Workspace members is the only membership API; nothing reaches the wire.
    def body(client):
        client._transport.send = None
        return client.identity.manage(action, payload)

    with pytest.raises(ServiceError) as error:
        run(asynchronous, "https://service.example", body)
    assert (error.value.code, error.value.status) == ("UnsupportedOperation", 501)
