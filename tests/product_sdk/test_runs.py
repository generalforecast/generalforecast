# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A run is created with what it needs, followed by its events, and ends naming its version."""

import inspect

import pytest

import generalforecast as gf
from generalforecast.async_client import Runs as AsyncRuns
from generalforecast.client import Runs
from generalforecast.service.contracts import ServiceError
from tests.product_sdk.test_client import run

BASE = "general-forecast/tabicl-v2-classifier"
PUBLISHED = {
    "model": "test/credit@ft2",
    "digest": "sha256:" + "9" * 64,
    "kept_step": 180,
    "validation_loss": 0.412,
    "base_validation_loss": 0.455,
}
CREATED = {"at": 1.0, "kind": "created", "body": {}}
SETTING_UP = {"at": 2.0, "kind": "state", "body": {"state": "setting_up"}}
TRAINING = {"at": 3.0, "kind": "state", "body": {"state": "training"}}
METRICS = {"at": 4.0, "kind": "metrics", "body": {"steps": 20, "loss": 0.5}}
NOTE = {"at": 5.0, "kind": "published", "body": PUBLISHED}
FINISHED = {"at": 6.0, "kind": "state", "body": {"state": "finished"}}


def _run(state, **fields):
    return {
        "model": "test/credit@ft2",
        "name": "credit",
        "version": 2,
        "state": state,
        **fields,
    }


def _lineage(*events):
    return {"model": "test/credit@ft2", "events": list(events)}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_run_is_created_by_its_fine_tune_name_and_base(wire, asynchronous):
    spec = {
        "name": "credit",
        "base": BASE,
        "data": ["credit-2026-02"],
        "method": {"type": "lora", "rank": 16},
        "plan": {"steps": 200},
    }
    wire.expect("POST", "/runs", _run("queued"), payload=spec)
    created = run(
        asynchronous,
        wire.url,
        lambda c: c.runs.create(
            "credit",
            base=BASE,
            data=["credit-2026-02"],
            method={"type": "lora", "rank": 16},
            plan={"steps": 200},
        ),
    )
    assert created["model"] == "test/credit@ft2" and created["version"] == 2


def test_a_run_takes_no_fine_tune_apart_from_its_name_and_its_evaluation():
    for runs in (Runs, AsyncRuns):
        parameters = inspect.signature(runs.create).parameters
        assert "fine_tune" not in parameters and "evaluation" in parameters
        assert "base" in parameters


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_run_is_read_by_its_model_or_its_fine_tune_name(wire, asynchronous):
    # The path names the project, so the run's segment leaves it out.
    wire.expect("GET", "/runs/credit", _run("training"))
    wire.expect("GET", "/runs/credit@ft2", _run("training"))
    wire.expect("GET", "/runs/credit", _run("training"))
    wire.expect("GET", "/runs/credit@ft2", _run("training"))
    read = [
        run(asynchronous, wire.url, lambda c, ref=ref: c.runs.get(ref))
        for ref in ("test/credit", "test/credit@ft2", "credit", "credit@ft2")
    ]
    assert all(one == read[0] for one in read)
    assert read[0]["model"] == "test/credit@ft2"
    with pytest.raises(ServiceError, match="is not a run") as refused:
        run(asynchronous, wire.url, lambda c: c.runs.get("credit@ft0"))
    assert "<project>/<name>@ft<version>" in (refused.value.hint or "")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_run_of_another_project_or_no_run_is_refused_before_any_request(
    wire, asynchronous
):
    # The wire answers nothing: any request would fail the test.
    for ref in ("acme/credit@ft2", "general-forecast/tabicl-v2-classifier", "a/b/c"):
        with pytest.raises(ServiceError) as refused:
            run(asynchronous, wire.url, lambda c, ref=ref: c.runs.get(ref))
        assert refused.value.code == "ValidationError" and refused.value.param == "run"
    with pytest.raises(ServiceError, match="project acme's; this client reads"):
        run(asynchronous, wire.url, lambda c: c.runs.cancel("acme/credit@ft2"))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_every_run_method_takes_the_run_by_its_model(wire, asynchronous):
    wire.expect("POST", "/runs/credit@ft2/cancel", _run("cancelled"), payload={})
    wire.expect("GET", "/runs/credit@ft2/lineage", _lineage(CREATED))
    cancelled = run(asynchronous, wire.url, lambda c: c.runs.cancel("test/credit@ft2"))
    lineage = run(asynchronous, wire.url, lambda c: c.runs.lineage("test/credit@ft2"))
    assert cancelled["state"] == "cancelled"
    assert lineage == _lineage(CREATED)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_wait_by_name_follows_the_run_that_was_latest_when_it_began(
    wire, asynchronous
):
    # The name is read once; then its run, by its model, until it ends.
    wire.expect("GET", "/runs/credit", _run("training", published=None))
    wire.expect("GET", "/runs/credit@ft2", _run("finished", published=PUBLISHED))
    ended = run(asynchronous, wire.url, lambda c: c.runs.wait("test/credit", poll=0.01))
    assert ended["model"] == "test/credit@ft2" and ended["state"] == "finished"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_returns_the_run_with_the_version_it_published(wire, asynchronous):
    wire.expect("GET", "/runs/credit@ft2", _run("training", published=None))
    wire.expect("GET", "/runs/credit@ft2", _run("finished", published=PUBLISHED))
    ended = run(asynchronous, wire.url, lambda c: c.runs.wait("credit@ft2", poll=0.01))
    assert ended["published"]["model"] == "test/credit@ft2"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_returns_the_run_as_the_service_answers_and_reads_no_lineage(
    wire, asynchronous
):
    # The wire answers only the run: a read of its lineage would fail.
    wire.expect(
        "GET", "/runs/credit@ft2", _run("failed", reason="data", published=None)
    )
    ended = run(asynchronous, wire.url, lambda c: c.runs.wait("credit@ft2"))
    assert ended["published"] is None and ended["reason"] == "data"


def _follow(wire):
    """A run seen three times: setting up, training, then finished and published."""
    wire.expect("GET", "/runs/credit@ft2", _run("setting_up", published=None))
    wire.expect("GET", "/runs/credit@ft2/lineage", _lineage(CREATED, SETTING_UP))
    wire.expect("GET", "/runs/credit@ft2", _run("training", published=None))
    wire.expect(
        "GET",
        "/runs/credit@ft2/lineage",
        _lineage(CREATED, SETTING_UP, TRAINING, METRICS),
    )
    wire.expect("GET", "/runs/credit@ft2", _run("finished", published=PUBLISHED))
    wire.expect(
        "GET",
        "/runs/credit@ft2/lineage",
        _lineage(CREATED, SETTING_UP, TRAINING, METRICS, NOTE, FINISHED),
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_watch_yields_each_event_once_until_the_run_ends(wire, asynchronous):
    _follow(wire)

    async def body(client):
        watched = client.runs.watch("credit@ft2", poll=0.01)
        if asynchronous:
            return [event async for event in watched]
        return list(watched)

    events = run(asynchronous, wire.url, body)
    assert events == [CREATED, SETTING_UP, TRAINING, METRICS, NOTE, FINISHED]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_passes_each_event_to_on_event_and_returns_the_run(wire, asynchronous):
    _follow(wire)
    seen = []
    ended = run(
        asynchronous,
        wire.url,
        lambda c: c.runs.wait("credit@ft2", poll=0.01, on_event=seen.append),
    )
    assert seen == [CREATED, SETTING_UP, TRAINING, METRICS, NOTE, FINISHED]
    assert ended["state"] == "finished" and ended["published"] == PUBLISHED


def test_the_async_client_awaits_a_coroutine_on_event(wire):
    _follow(wire)
    seen = []

    async def record(event):
        seen.append(event["kind"])

    ended = run(
        True,
        wire.url,
        lambda c: c.runs.wait("credit@ft2", poll=0.01, on_event=record),
    )
    assert seen == ["created", "state", "state", "metrics", "published", "state"]
    assert ended["published"]["model"] == "test/credit@ft2"
    assert inspect.iscoroutinefunction(record)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_watch_past_its_timeout_raises_and_leaves_the_run_going(wire, asynchronous):
    wire.expect("GET", "/runs/credit@ft2", _run("training", published=None))
    wire.expect("GET", "/runs/credit@ft2/lineage", _lineage(CREATED, TRAINING))

    async def body(client):
        watched = client.runs.watch("credit@ft2", timeout=0, poll=0.01)
        if asynchronous:
            return [event async for event in watched]
        return list(watched)

    with pytest.raises(ServiceError) as raised:
        run(asynchronous, wire.url, body)
    assert raised.value.code == "WaitTimeout"
    assert isinstance(raised.value, gf.ModelNotReady)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_an_on_event_that_raises_stops_the_wait(wire, asynchronous):
    wire.expect("GET", "/runs/credit@ft2", _run("training", published=None))
    wire.expect("GET", "/runs/credit@ft2/lineage", _lineage(CREATED, TRAINING))

    def stop(event):
        raise RuntimeError("seen enough")

    with pytest.raises(RuntimeError, match="seen enough"):
        run(
            asynchronous,
            wire.url,
            lambda c: c.runs.wait("credit@ft2", poll=0.01, on_event=stop),
        )
