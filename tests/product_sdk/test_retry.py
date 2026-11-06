# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Retries of transient failures, in both clients: which requests, how long, who is told."""

import asyncio
import inspect
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

import generalforecast as gf
from generalforecast import _operations
from generalforecast.async_client import AsyncClient
from generalforecast.client import Client
from generalforecast.contracts import Environment
from generalforecast.retry import (
    DEFAULT_RETRY,
    PREDICT,
    READ,
    RESPONSE_INTERRUPTED,
    WRITE,
    Backoff,
    RetryEvent,
    RetryPolicy,
    Tries,
    outcome_unknown,
    retry_after,
    retryable,
)
from generalforecast.service.contracts import ServiceError
from generalforecast.transport import AsyncTransport, Profile, Transport

PREDICTION = {
    "model": "general-forecast/tabicl-v2-classifier",
    "task": "classification",
    "predictions": [{"label": True, "probabilities": {"false": 0.1, "true": 0.9}}],
}
SNAPSHOT = "sha256:" + "a" * 64
MEMBERS = {"table": {"snapshots": [SNAPSHOT]}}


def fast(**changes) -> Backoff:
    """A backoff with no jitter and short waits, the sleeps faked."""
    settings: dict[str, Any] = {
        "attempts": 5,
        "deadline": 60.0,
        "initial": 1.0,
        "maximum": 4.0,
    }
    chosen: dict[str, Any] = settings | {"jitter": 0.0} | changes
    return Backoff(**chosen)


def policy(events: list | None = None, **changes) -> RetryPolicy:
    backoff = fast(**changes)
    return RetryPolicy(
        predict=backoff,
        control=backoff,
        on_retry=None if events is None else events.append,
    )


def unavailable(code: str = "ModelNotReady", status: int = 503, wait=None):
    error = ServiceError(code, "hosted inference is starting; retry shortly", status)
    error.retry_after = wait
    return error


def lost() -> ServiceError:
    return ServiceError(
        "TransportError",
        "request failed; outcome unknown; preserve write_id or submission_id",
        503,
    )


class Slow:
    """An outcome a try reaches only after `seconds`, as a fit on a cold GPU does."""

    def __init__(self, seconds: float, outcome: Any) -> None:
        self.seconds, self.outcome = seconds, outcome


class Script:
    """A transport that answers each request with the next outcome, and records it."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.sent: list[tuple[str, str, str, float | None]] = []

    def reply(self, method, path, body, headers, *, deadline=None):
        self.sent.append(
            (
                method,
                path.removeprefix("/v1/projects/test"),
                headers["X-GF-Request-ID"],
                deadline,
            )
        )
        assert self.outcomes, f"unexpected {method} {path}"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Slow):
            time.sleep(outcome.seconds)
            outcome = outcome.outcome
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def run(asynchronous: bool, script: Script, body, *, retry=DEFAULT_RETRY, monkeypatch):
    """Runs `body(client)` against `script`, the waits recorded instead of slept."""
    monkeypatch.setattr(_operations, "answer", lambda _prepared, value: value)
    slept: list[float] = []

    async def go():
        profile = Profile("https://service.example", "test")
        cls = AsyncClient if asynchronous else Client
        client: Any = cls(profile=profile, api_key="fixture-key", retry=retry)
        if asynchronous:

            async def send(*args, **kwargs):
                return script.reply(*args, **kwargs)

            async def sleep(seconds):
                slept.append(seconds)

        else:

            def send(*args, **kwargs):
                return script.reply(*args, **kwargs)

            def sleep(seconds):
                slept.append(seconds)

        client._transport.send = send
        client._retry_sleep = sleep
        try:
            value = body(client)
            return await value if inspect.isawaitable(value) else value
        finally:
            if isinstance(client, AsyncClient):
                await client.aclose()
            else:
                client.close()

    return asyncio.run(go()), slept


both = pytest.mark.parametrize("asynchronous", [False, True])


@both
def test_a_prediction_is_retried_through_a_starting_model(asynchronous, monkeypatch):
    events: list[RetryEvent] = []
    script = Script(
        unavailable(),
        unavailable("ModelUnavailable"),
        ServiceError("RequestTimeout", "inference timed out", 504),
        PREDICTION,
    )
    result, slept = run(
        asynchronous,
        script,
        lambda c: c.predictors.predict("risk", {"x": [3]}),
        retry=policy(events),
        monkeypatch=monkeypatch,
    )
    assert result.model == PREDICTION["model"]
    assert [sent[:2] for sent in script.sent] == [
        ("POST", "/predictors/risk/predict")
    ] * 4
    # Every try is the same request: the service can tell a try again.
    assert len({sent[2] for sent in script.sent}) == 1
    assert slept == [1.0, 2.0, 4.0]
    assert [(e.attempt, e.wait, e.retry_class) for e in events] == [
        (1, 1.0, PREDICT),
        (2, 2.0, PREDICT),
        (3, 4.0, PREDICT),
    ]
    assert not any(event.gave_up for event in events)
    assert events[0].request_id == script.sent[0][2]
    assert events[2].error.status == 504


@both
def test_an_ad_hoc_prediction_is_retried(asynchronous, monkeypatch):
    script = Script(lost(), PREDICTION)
    result, slept = run(
        asynchronous,
        script,
        lambda c: c.predict(
            "general-forecast/tabicl-v2-classifier",
            "y",
            {"x": [1, 2], "y": [True, False]},
            {"x": [3]},
        ),
        retry=policy(),
        monkeypatch=monkeypatch,
    )
    assert result.model == PREDICTION["model"] and slept == [1.0]
    assert [sent[1] for sent in script.sent] == ["/predict", "/predict"]


@both
def test_the_backoff_doubles_up_to_its_maximum(asynchronous, monkeypatch):
    script = Script(*[unavailable()] * 5, PREDICTION)
    _, slept = run(
        asynchronous,
        script,
        lambda c: c.predictors.predict("risk", {"x": [3]}),
        retry=policy(attempts=10, initial=0.5, maximum=3.0),
        monkeypatch=monkeypatch,
    )
    assert slept == [0.5, 1.0, 2.0, 3.0, 3.0]


def test_jitter_draws_each_wait_down_from_its_bound():
    backoff = Backoff(attempts=5, deadline=60, initial=2, maximum=30, jitter=0.5)
    assert backoff.wait(1, lambda: 0.0) == 2.0
    assert backoff.wait(1, lambda: 1.0) == 1.0
    assert backoff.wait(3, lambda: 0.5) == 8.0 * 0.75
    assert backoff.wait(10, lambda: 0.0) == 30.0


@both
def test_retry_after_is_the_wait_when_the_service_sends_one(asynchronous, monkeypatch):
    events: list[RetryEvent] = []
    script = Script(
        unavailable("RateLimited", 429, wait=7.0),
        unavailable("CapacityExceeded", 503, wait=0.0),
        PREDICTION,
    )
    _, slept = run(
        asynchronous,
        script,
        lambda c: c.predictors.predict("risk", {"x": [3]}),
        retry=policy(events),
        monkeypatch=monkeypatch,
    )
    assert slept == [7.0, 0.0]
    assert [event.retry_after for event in events] == [7.0, 0.0]


@both
def test_a_request_gives_up_after_its_tries_and_raises_the_last_error(
    asynchronous, monkeypatch
):
    events: list[RetryEvent] = []
    last = unavailable("ModelUnavailable")
    script = Script(unavailable(), unavailable(), last)
    with pytest.raises(gf.ModelNotReady) as raised:
        run(
            asynchronous,
            script,
            lambda c: c.predictors.predict("risk", {"x": [3]}),
            retry=policy(events, attempts=3),
            monkeypatch=monkeypatch,
        )
    assert raised.value is last
    assert len(script.sent) == 3
    assert [(e.attempt, e.gave_up) for e in events] == [
        (1, False),
        (2, False),
        (3, True),
    ]


@both
def test_a_request_gives_up_when_its_next_try_would_pass_its_deadline(
    asynchronous, monkeypatch
):
    events: list[RetryEvent] = []
    script = Script(unavailable(), unavailable(wait=30.0))
    with pytest.raises(gf.ModelNotReady):
        run(
            asynchronous,
            script,
            lambda c: c.predictors.predict("risk", {"x": [3]}),
            retry=policy(events, deadline=10.0),
            monkeypatch=monkeypatch,
        )
    # The second answer asks for longer than the deadline leaves.
    assert len(script.sent) == 2 and events[-1].gave_up
    # The retry deadline never bounds a try in flight: a call with no
    # deadline of its own leaves each try to the client's timeout.
    assert [sent[3] for sent in script.sent] == [None, None]


@both
@pytest.mark.parametrize(
    "error",
    [
        ServiceError("ValidationError", "queries must be rows", 422),
        ServiceError("NotFound", "predictor not found", 404),
        ServiceError("Conflict", "environment generation changed", 409),
        ServiceError("Unauthorized", "invalid key", 401),
        ServiceError("CapacityExceeded", "too many rows", 413),
        ServiceError("InternalError", "our fault", 500),
        ServiceError("RedirectRefused", "credential forwarding is disabled", 502),
    ],
    ids=lambda error: f"{error.status}-{error.code}",
)
def test_a_refusal_of_the_request_itself_is_raised_at_once(
    asynchronous, error, monkeypatch
):
    events: list[RetryEvent] = []
    script = Script(error)
    with pytest.raises(ServiceError) as raised:
        run(
            asynchronous,
            script,
            lambda c: c.predictors.predict("risk", {"x": [3]}),
            retry=policy(events),
            monkeypatch=monkeypatch,
        )
    assert raised.value is error and len(script.sent) == 1 and not events


@both
@pytest.mark.parametrize("retry", [None, 0, False, 1])
def test_retry_none_or_zero_sends_every_request_once(asynchronous, retry, monkeypatch):
    script = Script(unavailable())
    with pytest.raises(gf.ModelNotReady):
        run(
            asynchronous,
            script,
            lambda c: c.predictors.predict("risk", {"x": [3]}),
            retry=retry,
            monkeypatch=monkeypatch,
        )
    assert len(script.sent) == 1


@both
def test_reads_are_retried(asynchronous, monkeypatch):
    script = Script(
        lost(),
        unavailable("ServiceUnavailable", 502),
        {"runs": []},
        ServiceError("Timeout", "request deadline exceeded", 504),
        {"name": "r", "state": "finished"},
    )

    def body(client):
        if asynchronous:

            async def go():
                return await client.runs.list(), await client.runs.get("r")

            return go()
        return client.runs.list(), client.runs.get("r")

    result, slept = run(
        asynchronous, script, body, retry=policy(), monkeypatch=monkeypatch
    )
    assert result == ([], {"name": "r", "state": "finished"})
    assert slept == [1.0, 2.0, 1.0]


@both
def test_runs_are_created_again_past_the_run_limit(asynchronous, monkeypatch):
    run_limit = ServiceError(
        "RunLimitReached",
        "the project already has 4 runs queued or running; wait for one to end",
        429,
    )
    created = {"name": "ft", "state": "queued"}
    script = Script(run_limit, lost(), created)
    result, slept = run(
        asynchronous,
        script,
        lambda c: c.runs.create(
            "ft",
            base="general-forecast/tabicl-v2-classifier",
            corpus="c",
            changes=[],
            evaluation={},
        ),
        retry=policy(),
        monkeypatch=monkeypatch,
    )
    assert result == created and slept == [1.0, 2.0]
    assert [sent[:2] for sent in script.sent] == [("POST", "/runs")] * 3


@both
@pytest.mark.parametrize(
    ("call", "route"),
    [
        (lambda c: c.predictors.create("risk"), "/predictors"),
        (
            lambda c: c.predictors.create_environment("risk", env="staging"),
            "/predictors/risk/environments",
        ),
        (
            lambda c: c.predictors.promote(
                "risk", revision=2, env="staging", expected_generation=1
            ),
            "/predictors/risk/environments/staging/promote",
        ),
    ],
    ids=["predictor", "environment", "promote"],
)
def test_idempotent_writes_are_retried_under_one_request_id(
    asynchronous, call, route, monkeypatch
):
    answer = (
        {"name": "risk"}
        if route == "/predictors"
        else Environment("risk", "staging", 2, 2, "ready").to_dict()
    )
    script = Script(lost(), unavailable("SingleWriterUnavailable"), answer)
    run(asynchronous, script, call, retry=policy(), monkeypatch=monkeypatch)
    assert [sent[1] for sent in script.sent] == [route] * 3
    assert len({sent[2] for sent in script.sent}) == 1


@both
@pytest.mark.parametrize(
    ("call", "route"),
    [
        (
            lambda c: c.contexts.mutate("history", {"write_id": "w"}),
            "/contexts/history/mutations",
        ),
        (
            lambda c: c.contexts.append(
                "history", {"id": [1], "x": [2]}, id="id", idempotency_key="w"
            ),
            "/contexts/history/append",
        ),
        (lambda c: c.runs.cancel("ft"), "/runs/ft/cancel"),
        (lambda c: c.runs.train("ft", {"steps": 1}), "/runs/ft/train"),
        (lambda c: c.corpora.delete("c"), "/corpora/c/delete"),
        (lambda c: c.predictors.delete("risk"), "/predictors/risk/delete"),
        (lambda c: c.keys.create(name="k", permissions=[]), "/keys"),
        (lambda c: c.secrets.create(name="s", value="v"), "/secrets"),
    ],
    ids=[
        "mutate",
        "append",
        "cancel",
        "train",
        "corpus-delete",
        "delete",
        "key",
        "secret",
    ],
)
def test_other_writes_are_sent_once(asynchronous, call, route, monkeypatch):
    events: list[RetryEvent] = []
    script = Script(lost())
    with pytest.raises(gf.Unavailable):
        run(asynchronous, script, call, retry=policy(events), monkeypatch=monkeypatch)
    assert [sent[1] for sent in script.sent] == [route] and not events


@both
def test_a_corpus_its_own_lost_try_made_is_returned(asynchronous, monkeypatch):
    made = {
        "name": "c",
        "digest": "sha256:" + "b" * 64,
        "members": MEMBERS,
        "parent": None,
    }
    conflict = ServiceError("Conflict", "a corpus never changes", 409)
    script = Script(lost(), conflict, made)
    result, _ = run(
        asynchronous,
        script,
        lambda c: c.corpora.create("c", MEMBERS),
        retry=policy(),
        monkeypatch=monkeypatch,
    )
    assert result == made
    assert [sent[:2] for sent in script.sent] == [
        ("POST", "/corpora"),
        ("POST", "/corpora"),
        ("GET", "/corpora/c"),
    ]


@both
def test_a_corpus_of_other_members_is_still_a_conflict(asynchronous, monkeypatch):
    other = {"table": {"snapshots": ["sha256:" + "c" * 64]}}
    conflict = ServiceError("Conflict", "a corpus never changes", 409)
    script = Script(lost(), conflict, {"name": "c", "members": other, "parent": None})
    with pytest.raises(gf.Conflict) as raised:
        run(
            asynchronous,
            script,
            lambda c: c.corpora.create("c", MEMBERS),
            retry=policy(),
            monkeypatch=monkeypatch,
        )
    assert raised.value is conflict


@both
def test_a_corpus_conflict_on_the_first_try_is_raised_unread(asynchronous, monkeypatch):
    conflict = ServiceError("Conflict", "a corpus never changes", 409)
    script = Script(conflict)
    with pytest.raises(gf.Conflict):
        run(
            asynchronous,
            script,
            lambda c: c.corpora.create("c", MEMBERS),
            retry=policy(),
            monkeypatch=monkeypatch,
        )
    assert [sent[1] for sent in script.sent] == ["/corpora"]


@both
def test_a_corpus_conflict_after_a_rate_limit_is_raised_unread(
    asynchronous, monkeypatch
):
    # A 429 refuses before anything is kept, so the conflict is someone else's.
    conflict = ServiceError("Conflict", "a corpus never changes", 409)
    script = Script(unavailable("RateLimited", 429), conflict)
    with pytest.raises(gf.Conflict):
        run(
            asynchronous,
            script,
            lambda c: c.corpora.create("c", MEMBERS),
            retry=policy(),
            monkeypatch=monkeypatch,
        )
    assert [sent[1] for sent in script.sent] == ["/corpora", "/corpora"]


@both
def test_a_promotion_its_own_lost_try_made_is_returned(asynchronous, monkeypatch):
    moved = Environment("risk", "staging", 2, 2, "pending")
    script = Script(
        lost(),
        ServiceError("Conflict", "environment generation changed", 409),
        moved.to_dict(),
    )
    result, _ = run(
        asynchronous,
        script,
        lambda c: c.predictors.promote(
            "risk", revision=2, env="staging", expected_generation=1
        ),
        retry=policy(),
        monkeypatch=monkeypatch,
    )
    assert result == moved
    assert [sent[:2] for sent in script.sent] == [
        ("POST", "/predictors/risk/environments/staging/promote"),
        ("POST", "/predictors/risk/environments/staging/promote"),
        ("GET", "/predictors/risk/environments/staging"),
    ]


@both
def test_a_promotion_someone_else_moved_is_still_a_conflict(asynchronous, monkeypatch):
    conflict = ServiceError("Conflict", "environment generation changed", 409)
    script = Script(
        lost(), conflict, Environment("risk", "staging", 3, 2, "pending").to_dict()
    )
    with pytest.raises(gf.Conflict) as raised:
        run(
            asynchronous,
            script,
            lambda c: c.predictors.promote(
                "risk", revision=2, env="staging", expected_generation=1
            ),
            retry=policy(),
            monkeypatch=monkeypatch,
        )
    assert raised.value is conflict


@both
def test_an_exception_from_on_retry_ends_the_call(asynchronous, monkeypatch):
    class Stop(Exception):
        pass

    def stop(_event):
        raise Stop

    script = Script(unavailable(), PREDICTION)
    with pytest.raises(Stop):
        run(
            asynchronous,
            script,
            lambda c: c.predictors.predict("risk", {"x": [3]}),
            retry=RetryPolicy(predict=fast(), control=fast(), on_retry=stop),
            monkeypatch=monkeypatch,
        )
    assert len(script.sent) == 1


@both
def test_retries_are_logged(asynchronous, monkeypatch, caplog):
    script = Script(unavailable(), PREDICTION)
    with caplog.at_level("INFO", logger="generalforecast.retries"):
        run(
            asynchronous,
            script,
            lambda c: c.predictors.predict("risk", {"x": [3]}),
            retry=policy(),
            monkeypatch=monkeypatch,
        )
    assert [record.getMessage() for record in caplog.records] == [
        (
            "POST /v1/projects/test/predictors/risk/predict retrying in 1.0s "
            "on try 1: 503 ModelNotReady"
        )
    ]


def test_which_errors_are_retryable_and_which_leave_the_outcome_unknown():
    assert retryable(lost()) and outcome_unknown(lost())
    timeout = ServiceError("Timeout", "request deadline exceeded", 504)
    assert retryable(timeout) and outcome_unknown(timeout)
    for status in (502, 503, 504):
        assert retryable(unavailable(status=status))
    limited = unavailable("RateLimited", 429)
    assert retryable(limited) and not outcome_unknown(limited)
    for refused in (
        ServiceError("ValidationError", "bad", 422),
        ServiceError("Conflict", "moved", 409),
        ServiceError("InternalError", "ours", 500),
        ServiceError("RedirectRefused", "credential forwarding is disabled", 502),
        ServiceError("ClientClosed", "client is closed", 400),
        ValueError("not a service error"),
    ):
        assert not retryable(refused)
    # A reply that arrived is never asked for again: malformed, oversized or
    # in the wrong envelope whatever its status, or cut off once it began.
    for status in (429, 502, 503, 504):
        assert not retryable(
            ServiceError("ProtocolError", "invalid service response", status)
        )
    for code, status in (("TransportError", 503), ("Timeout", 504)):
        interrupted = ServiceError(
            code, "response interrupted", status, reason=RESPONSE_INTERRUPTED
        )
        assert not retryable(interrupted) and not outcome_unknown(interrupted)


def test_a_retry_argument_is_a_policy_a_number_of_tries_or_none():
    assert RetryPolicy.of(None) is None and RetryPolicy.of(0) is None
    assert RetryPolicy.of(False) is None and RetryPolicy.of(1) is None
    assert RetryPolicy.of(True) == RetryPolicy()
    three = RetryPolicy.of(3)
    assert three is not None
    assert three.predict.attempts == 3 and three.control.attempts == 3
    assert three.predict.deadline == 120.0 and three.control.deadline == 120.0
    custom = RetryPolicy(on_retry=print)
    assert RetryPolicy.of(custom) is custom
    wrongs: list[Any] = [-1, 2.5, "3"]
    for wrong in wrongs:
        with pytest.raises(ValueError, match="retry is"):
            RetryPolicy.of(wrong)
    assert DEFAULT_RETRY.backoff(PREDICT).deadline == 120.0
    assert DEFAULT_RETRY.backoff(READ) is DEFAULT_RETRY.backoff(WRITE)


@pytest.mark.parametrize(
    "settings",
    [
        {"attempts": 0},
        {"attempts": 1.5},
        {"deadline": -1},
        {"multiplier": 0.5},
        {"jitter": 2},
    ],
)
def test_a_backoff_refuses_bounds_that_mean_nothing(settings):
    with pytest.raises(ValueError):
        Backoff(
            **({"attempts": 3, "deadline": 1, "initial": 1, "maximum": 1} | settings)
        )


def test_retry_after_is_read_as_seconds_or_an_http_date():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert retry_after("5") == 5.0
    assert retry_after(" 0 ") == 0.0
    later = format_datetime(now + timedelta(seconds=30), usegmt=True)
    assert retry_after(later, lambda: now) == 30.0
    earlier = format_datetime(now - timedelta(seconds=30), usegmt=True)
    assert retry_after(earlier, lambda: now) == 0.0
    assert retry_after(None) is None
    assert retry_after("soon") is None
    assert retry_after("-3") is None
    assert retry_after("99999999") == 86400.0


@pytest.fixture
def busy():
    """A server that answers 503 with a Retry-After, then 200."""
    answers = [
        (
            503,
            {"error": {"code": "ModelNotReady", "message": "the model is restarting"}},
            {"Retry-After": "2"},
        ),
        (
            429,
            {"error": {"code": "CapacityExceeded", "message": "slow down"}},
            {},
        ),
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def do_GET(self):
            status, body, headers = answers.pop(0)
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join()


@both
def test_the_transport_carries_a_replys_retry_after_on_its_error(asynchronous, busy):
    async def go():
        transport: Any = (AsyncTransport if asynchronous else Transport)(busy)
        errors = []
        for _ in range(2):
            try:
                value = transport.send("GET", "/v1/projects/test/runs")
                await value if inspect.isawaitable(value) else value
            except ServiceError as error:
                errors.append(error)
        if asynchronous:
            await transport.aclose()
        return errors

    first, second = asyncio.run(go())
    assert first.status == 503 and first.retry_after == 2.0
    assert second.status == 429 and second.retry_after is None
    # The wire error object is unchanged.
    assert "retry_after" not in first.to_dict()


@pytest.fixture
def replies():
    """A server whose every reply is spoiled the way `.mode` names, its requests counted."""

    class State:
        mode = "invalid"
        requests = 0
        url = ""

    state = State()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def send(self, status: int, data: bytes, length: int | None = None):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header(
                "Content-Length", str(len(data) if length is None else length)
            )
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state.requests += 1
            if state.mode == "silent":
                # The connection closes before any status line.
                self.close_connection = True
                return
            if state.mode == "invalid":
                return self.send(200, b"not-json")
            if state.mode == "oversized":
                return self.send(200, b"x" * (1_048_576 + 1))
            if state.mode == "bad-error":
                return self.send(503, b"<html>bad gateway</html>")
            if state.mode == "interrupted":
                self.close_connection = True
                return self.send(200, b'{"ob', length=100)
            raise AssertionError(state.mode)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()
    worker.join()


def listed(asynchronous: bool, url: str, **options):
    """`models.list()` through a real transport; the error and the waits slept."""
    slept: list[float] = []

    async def go():
        cls = AsyncClient if asynchronous else Client
        client: Any = cls(
            url, project="test", token="test-only-token", retry=policy(), **options
        )

        if asynchronous:

            async def sleep(seconds):
                slept.append(seconds)

        else:

            def sleep(seconds):
                slept.append(seconds)

        client._retry_sleep = sleep
        try:
            value = client.models.list()
            if inspect.isawaitable(value):
                await value
        except ServiceError as error:
            return error
        finally:
            if isinstance(client, AsyncClient):
                await client.aclose()
            else:
                client.close()
        raise AssertionError("the request succeeded")

    return asyncio.run(go()), slept


@both
@pytest.mark.parametrize("mode", ["invalid", "oversized", "bad-error"])
def test_a_malformed_reply_is_sent_for_once(asynchronous, mode, replies):
    replies.mode = mode
    error, slept = listed(asynchronous, replies.url)
    assert error.code == "ProtocolError"
    assert replies.requests == 1 and slept == []


@both
def test_a_reply_cut_off_once_it_began_is_sent_for_once(asynchronous, replies):
    replies.mode = "interrupted"
    error, slept = listed(asynchronous, replies.url)
    assert error.code == "TransportError" and error.reason == RESPONSE_INTERRUPTED
    assert replies.requests == 1 and slept == []


@both
def test_a_connection_closed_before_any_reply_is_retried(asynchronous, replies):
    replies.mode = "silent"
    error, slept = listed(asynchronous, replies.url)
    assert error.code == "TransportError" and error.reason != RESPONSE_INTERRUPTED
    assert replies.requests == 5 and len(slept) == 4


def test_a_sync_body_read_that_raises_is_typed_and_sent_once(monkeypatch):
    from http.client import IncompleteRead

    class Interrupted:
        code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self, _limit):
            raise IncompleteRead(b"", 10)

    calls = []
    client = Client(token="test-only-token")

    def open_once(*_args, **_kwargs):
        calls.append(True)
        return Interrupted()

    monkeypatch.setattr(client._transport._opener, "open", open_once)
    with pytest.raises(ServiceError) as caught:
        client.models.list()
    assert caught.value.code == "TransportError"
    assert len(calls) == 1


class _Broken(httpx.AsyncByteStream):
    """A body that yields one chunk, then fails or stalls."""

    def __init__(self, stall: bool) -> None:
        self.stall = stall

    async def __aiter__(self):
        yield b'{"ob'
        if self.stall:
            await asyncio.sleep(60)
        raise httpx.ReadError("connection reset")


@pytest.mark.parametrize("stall", [False, True])
def test_an_async_body_that_fails_or_stalls_is_sent_once(stall):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=_Broken(stall))

    error, slept = listed(
        True,
        "https://service.example",
        http_transport=httpx.MockTransport(handler),
        timeout=0.2,
    )
    assert error.code == ("Timeout" if stall else "TransportError")
    assert error.reason == RESPONSE_INTERRUPTED
    assert len(calls) == 1 and slept == []


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
def test_an_async_failure_before_any_reply_is_retried(failure):
    calls = []

    def handler(request):
        calls.append(request)
        raise failure("no reply", request=request)

    error, slept = listed(
        True, "https://service.example", http_transport=httpx.MockTransport(handler)
    )
    assert error.code in ("TransportError", "Timeout")
    assert error.reason != RESPONSE_INTERRUPTED
    assert len(calls) == 5 and len(slept) == 4


def request_limit() -> ServiceError:
    """The 504 a server answers when a request outlasts its own limit."""
    return ServiceError("Timeout", "the request exceeded the server's limit", 504)


@both
def test_a_slow_first_try_past_the_deadline_completes(asynchronous, monkeypatch):
    events: list[RetryEvent] = []
    script = Script(Slow(0.3, PREDICTION))
    result, slept = run(
        asynchronous,
        script,
        lambda c: c.predictors.predict("risk", {"x": [3]}),
        retry=policy(events, deadline=0.1),
        monkeypatch=monkeypatch,
    )
    assert result.model == PREDICTION["model"]
    assert len(script.sent) == 1 and slept == [] and events == []
    assert script.sent[0][3] is None


@both
def test_a_request_limit_after_a_long_first_try_is_retried_once(
    asynchronous, monkeypatch
):
    events: list[RetryEvent] = []
    # The first call fits past the budget and the server's limit; the fit
    # it kept answers the second at once.
    script = Script(Slow(0.3, request_limit()), PREDICTION)
    result, slept = run(
        asynchronous,
        script,
        lambda c: c.predictors.predict("risk", {"x": [3]}),
        retry=policy(events, deadline=0.1),
        monkeypatch=monkeypatch,
    )
    assert result.model == PREDICTION["model"]
    assert len(script.sent) == 2 and slept == [1.0]
    assert [event.wait for event in events] == [1.0]
    # The two tries are one request.
    assert script.sent[0][2] == script.sent[1][2]


@both
def test_no_try_starts_past_the_deadline_after_the_one_grace_retry(
    asynchronous, monkeypatch
):
    events: list[RetryEvent] = []
    script = Script(Slow(0.3, request_limit()), request_limit())
    with pytest.raises(ServiceError) as caught:
        run(
            asynchronous,
            script,
            lambda c: c.predictors.predict("risk", {"x": [3]}),
            retry=policy(events, deadline=0.1),
            monkeypatch=monkeypatch,
        )
    assert caught.value.status == 504
    assert len(script.sent) == 2 and events[-1].gave_up and events[-1].attempt == 2


@both
def test_a_later_try_past_the_deadline_is_not_followed_by_another(
    asynchronous, monkeypatch
):
    events: list[RetryEvent] = []
    script = Script(unavailable(wait=0.0), Slow(0.3, request_limit()))
    with pytest.raises(ServiceError):
        run(
            asynchronous,
            script,
            lambda c: c.predictors.predict("risk", {"x": [3]}),
            retry=policy(events, deadline=0.1),
            monkeypatch=monkeypatch,
        )
    assert len(script.sent) == 2 and events[-1].gave_up


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def tries(clock: Clock, limit: float | None = None, **changes) -> Tries:
    return Tries(
        policy(**changes), PREDICT, "POST", "/predict", "r-1", limit, clock=clock
    )


def test_the_grace_retry_waits_at_most_the_maximum_and_never_past_the_calls_own_deadline():
    clock = Clock()
    budget = tries(clock, deadline=120.0)
    clock.now += 146.0
    # A Retry-After longer than the backoff's maximum is cut to it.
    assert budget.failed(unavailable("Timeout", 504, wait=60.0)) == 4.0
    clock.now += 4.0
    assert budget.failed(request_limit()) is None

    clock = Clock()
    own = tries(clock, limit=clock.now + 147.0, deadline=120.0)
    clock.now += 146.0
    assert own.failed(request_limit()) is None


def test_a_wait_never_carries_a_try_past_the_deadline():
    clock = Clock()
    budget = tries(clock, deadline=120.0)
    clock.now += 118.0
    # The first try ended inside the budget, so no grace: a wait past it gives up.
    assert budget.failed(unavailable(wait=5.0)) is None
