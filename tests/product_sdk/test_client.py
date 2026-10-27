# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Client behaviours that match the hosted handlers, in both clients."""

import asyncio
import inspect
import time
from pathlib import Path

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
    pending = Environment("risk", "production", 1, 1, "pending")
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
    environment = Environment("risk", "production", 2, 3, "ready")
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


def _receipt(write_id: str, operation: str, rows: int, sequence: int) -> dict:
    return {
        "context": "history",
        "write_id": write_id,
        "operation": operation,
        "rows": rows,
        "sequence": sequence,
        "accepted_at": float(sequence),
        "durably_accepted": True,
    }


def _rows(count: int) -> list[dict]:
    return [{"id": f"r{i}", "x": float(i), "y": i % 2} for i in range(count)]


def test_the_chunk_size_is_the_largest_write_the_spec_accepts():
    import yaml

    from generalforecast._operations import WRITE_CHUNK, WRITE_ID_LENGTH

    root = Path(__file__).resolve().parents[2]
    schemas = yaml.safe_load((root / "api" / "openapi.yaml").read_text())["components"][
        "schemas"
    ]
    append, label = schemas["LiveContextAppend"], schemas["LiveContextLabel"]
    assert append["properties"]["rows"]["maxItems"] == WRITE_CHUNK
    assert label["properties"]["labels"]["maxItems"] == WRITE_CHUNK
    for schema in (append, label):
        assert schema["properties"]["write_id"]["maxLength"] == WRITE_ID_LENGTH


def test_chunk_write_ids_derive_from_the_key_and_stay_within_the_limit():
    from generalforecast._operations import WRITE_ID_LENGTH, chunk_write_id

    assert [chunk_write_id("w", index) for index in range(3)] == ["w#1", "w#2", "w#3"]
    long = "k" * WRITE_ID_LENGTH
    derived = chunk_write_id(long, 11)
    assert len(derived) <= WRITE_ID_LENGTH and derived.endswith("#12")
    assert derived == chunk_write_id(long, 11) != chunk_write_id(long[:-1] + "j", 11)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_write_that_fits_one_call_keeps_its_key(wire, asynchronous):
    rows = _rows(100)
    wire.expect(
        "POST",
        "/contexts/history/append",
        _receipt("w", "append", 100, 7),
        payload={"write_id": "w", "rows": rows},
    )
    receipt = run(
        asynchronous,
        wire.url,
        lambda c: c.contexts.append("history", rows, id="id", idempotency_key="w"),
    )
    assert receipt.to_dict() == _receipt("w", "append", 100, 7)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_large_appends_and_labels_go_in_chunks_under_derived_write_ids(
    wire, asynchronous
):
    rows = _rows(250)
    for index, (start, stop) in enumerate(((0, 100), (100, 200), (200, 250))):
        wire.expect(
            "POST",
            "/contexts/history/append",
            _receipt(f"w#{index + 1}", "append", stop - start, index + 1),
            payload={"write_id": f"w#{index + 1}", "rows": rows[start:stop]},
        )
    labels = [{"id": row["id"], "value": row["y"]} for row in rows]
    for index, (start, stop) in enumerate(((0, 100), (100, 200), (200, 250))):
        wire.expect(
            "POST",
            "/contexts/history/label",
            _receipt(f"l#{index + 1}", "label", stop - start, index + 4),
            payload={
                "write_id": f"l#{index + 1}",
                "target": "y",
                "labels": labels[start:stop],
            },
        )

    async def body(client):
        appended = client.contexts.append("history", rows, id="id", idempotency_key="w")
        appended = await appended if inspect.isawaitable(appended) else appended
        labeled = client.contexts.label(
            "history", rows, id="id", column="y", idempotency_key="l"
        )
        return appended, await labeled if inspect.isawaitable(labeled) else labeled

    appended, labeled = run(asynchronous, wire.url, body)
    for receipt, key, operation, last in (
        (appended, "w", "append", 3),
        (labeled, "l", "label", 6),
    ):
        assert receipt.write_id == key and receipt.sequence == last
        assert receipt["rows"] == 250 and receipt["durably_accepted"] is True
        assert [chunk["write_id"] for chunk in receipt["chunks"]] == [
            f"{key}#1",
            f"{key}#2",
            f"{key}#3",
        ]
        assert receipt["operation"] == operation


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_failed_chunk_names_the_accepted_ones_and_a_retry_resumes(wire, asynchronous):
    from generalforecast.client import PartialWrite

    rows = _rows(250)

    def expect(index: int, result, status: int = 200) -> None:
        start = 100 * index
        wire.expect(
            "POST",
            "/contexts/history/append",
            result,
            payload={"write_id": f"w#{index + 1}", "rows": rows[start : start + 100]},
            status=status,
        )

    def append(client):
        return client.contexts.append("history", rows, id="id", idempotency_key="w")

    expect(0, _receipt("w#1", "append", 100, 1))
    expect(1, "CapacityExceeded", 429)
    with pytest.raises(PartialWrite) as caught:
        run(asynchronous, wire.url, append)
    error = caught.value
    assert isinstance(error, gf.CapacityExceeded) and error.status == 429
    assert [receipt.write_id for receipt in error.accepted] == ["w#1"]
    assert error.chunks == 3
    assert (
        error.message == "append failed at chunk 2 of 3: service rejected the request"
    )
    assert (
        error.because
        == "chunks 1 to 1 of 3, 100 of 250 rows, were accepted as write ids `w#1`"
    )
    assert "same idempotency_key" in (error.hint or "")

    # The same key again: the first chunk replays its receipt, the rest go on.
    expect(0, _receipt("w#1", "append", 100, 1))
    expect(1, _receipt("w#2", "append", 100, 2))
    expect(2, _receipt("w#3", "append", 50, 3))
    receipt = run(asynchronous, wire.url, append)
    assert receipt["rows"] == 250 and receipt.sequence == 3


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_first_chunk_that_fails_raises_its_own_error(wire, asynchronous):
    from generalforecast.client import PartialWrite

    rows = _rows(150)
    wire.expect(
        "POST",
        "/contexts/history/append",
        "NotFound",
        payload={"write_id": "w#1", "rows": rows[:100]},
        status=404,
    )
    with pytest.raises(ServiceError) as caught:
        run(
            asynchronous,
            wire.url,
            lambda c: c.contexts.append("history", rows, id="id", idempotency_key="w"),
        )
    assert not isinstance(caught.value, PartialWrite)
    assert isinstance(caught.value, gf.NotFound)
