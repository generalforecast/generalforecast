# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Retries of transient failures, for the requests that are safe to send again.

The hosted service answers conditions that pass on their own with statuses a
caller may retry: 503 while a model starts or restarts or a device is full,
429 for a rate or run limit, 504 when inference outlasts one request (a
predictor's first fit, which the service keeps, so the next try is answered
from it), 502 when a proxy in front of it has no answer. A request can also
fail on the way, a connection reset or a reply cut short, with its outcome
unknown. `Client` and `AsyncClient` try such a request again, under a
`RetryPolicy`, when sending it again cannot apply it twice:

- `PREDICT`: predictions, which read and keep nothing but a fit;
- `READ`: every GET;
- `WRITE`: the writes the service applies once whatever the number of
  times they are sent: creating a predictor, a run or an environment by
  name, registering a revision by its lock, uploading a snapshot by its
  content, and promoting or rolling back under the same request id, which
  the service replays.

Every other write, such as an append without a stable write id, is sent
once. A refusal of the request itself, a 4xx other than 429, is never
retried. Each retry, and each give-up, is told to the policy's `on_retry`
and logged at INFO on `generalforecast.retries`.

Public, and stable across releases: `RetryPolicy`, `Backoff`, `RetryEvent`,
`retryable` and `outcome_unknown`.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final

from generalforecast.errors import Error

__all__ = [
    "ONCE",
    "PREDICT",
    "READ",
    "RESPONSE_INTERRUPTED",
    "WRITE",
    "Backoff",
    "RetryEvent",
    "RetryPolicy",
    "outcome_unknown",
    "retryable",
]

PREDICT: Final = "predict"
READ: Final = "read"
WRITE: Final = "write"
# A request sent once whatever its method, such as a poll with no time left.
ONCE: Final = "once"

# Statuses a service answers for a condition that passes on its own.
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
# Codes the SDK raises for a request whose outcome it could not learn.
_NO_ANSWER = frozenset({"TransportError", "Timeout"})
# Codes that carry a retryable status but name a failure no retry mends: a
# reply that is malformed, oversized or in the wrong envelope is never sent for
# again, whatever its status.
_PERMANENT = frozenset({"RedirectRefused", "ClientClosed", "ProtocolError"})
# The reason a transport gives a `TransportError` or `Timeout` raised once a
# reply's status line had arrived: the service answered, so the client fails
# closed rather than send the request again.
RESPONSE_INTERRUPTED: Final = "response_interrupted"
# The longest Retry-After believed, in seconds.
MAX_RETRY_AFTER = 86400.0

RETRIES = logging.getLogger("generalforecast.retries")


def retryable(error: BaseException) -> bool:
    """Whether a failed request may succeed if it is sent again unchanged.

    True for a request that got no answer (`TransportError`, `Timeout`
    before any reply arrived) and for a 429, 502, 503 or 504 with a
    well-formed error object; never for a reply that was interrupted,
    malformed or oversized once it began (the client fails closed), and never
    for any other refusal.
    """
    if not isinstance(error, Error):
        return False
    if error.code in _NO_ANSWER:
        return error.reason != RESPONSE_INTERRUPTED
    return error.code not in _PERMANENT and error.status in RETRYABLE_STATUSES


def outcome_unknown(error: BaseException) -> bool:
    """Whether a failed request may have been applied all the same.

    A request that got no answer, or a 5xx from a proxy or a service handing
    over, may have been; a 429 refuses before anything is kept.
    """
    return retryable(error) and getattr(error, "status", None) != 429


def retry_after(
    value: str | None, now: Callable[[], datetime] | None = None
) -> float | None:
    """The seconds a Retry-After header asks for: a delay or an HTTP date; None if neither."""
    if value is None:
        return None
    text = value.strip()
    if text.isascii() and text.isdigit():
        return min(float(text), MAX_RETRY_AFTER)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    current = now() if now is not None else datetime.now(UTC)
    return min(max(0.0, (when - current).total_seconds()), MAX_RETRY_AFTER)


@dataclass(frozen=True)
class Backoff:
    """How long one class of requests is retried, and how long between tries.

    Attributes:
        attempts: The most tries, the first included; 1 never retries.
        deadline: The most seconds from the first try to the end of the last.
        initial: Seconds before the first retry.
        maximum: The longest wait between tries.
        multiplier: How much longer each wait is than the one before.
        jitter: The fraction of each wait drawn at random, 0 to 1, so many
            callers retrying together spread out.
    """

    attempts: int
    deadline: float
    initial: float
    maximum: float
    multiplier: float = 2.0
    jitter: float = 0.5

    def __post_init__(self) -> None:
        """Checks the bounds."""
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("attempts must be an integer of at least 1")
        for label in ("deadline", "initial", "maximum"):
            value = getattr(self, label)
            if not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{label} must be a number of at least 0")
        if not isinstance(self.multiplier, int | float) or self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if not isinstance(self.jitter, int | float) or not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")

    def wait(self, retry: int, draw: Callable[[], float] = random.random) -> float:
        """The wait before the `retry`th retry, from 1, jittered down from its bound."""
        bound = min(self.maximum, self.initial * self.multiplier ** (retry - 1))
        return bound * (1 - self.jitter * draw())


@dataclass(frozen=True)
class RetryEvent:
    """One failed try of a request that was retryable, as `on_retry` is told it.

    Attributes:
        retry_class: `PREDICT`, `READ` or `WRITE`.
        method: The request's method.
        path: The request's path under the endpoint, with any query.
        request_id: The request id every try of it is sent with.
        attempt: The try that failed, from 1.
        error: What it failed with.
        elapsed: Seconds since the first try began.
        wait: Seconds before the next try; None when the request gives up,
            out of tries or time, and `error` is raised.
        retry_after: The seconds the service asked for, if it did.
    """

    retry_class: str
    method: str
    path: str
    request_id: str | None
    attempt: int
    error: Error
    elapsed: float
    wait: float | None
    retry_after: float | None

    @property
    def gave_up(self) -> bool:
        """Whether this was the last try and its error is raised."""
        return self.wait is None


def _predict_backoff() -> Backoff:
    return Backoff(attempts=20, deadline=120.0, initial=2.0, maximum=30.0)


def _control_backoff() -> Backoff:
    return Backoff(attempts=6, deadline=120.0, initial=1.0, maximum=20.0)


@dataclass(frozen=True)
class RetryPolicy:
    """Which requests a client retries, for how long, and who is told.

    The default retries a prediction, and every other retryable request,
    for up to two minutes. A wait is the service's Retry-After when it sends
    one, else the class's backoff; a request whose next try would start past
    its deadline, or the deadline the call itself has, gives up and raises
    the last error.

    Attributes:
        predict: The backoff of predictions.
        control: The backoff of reads and of idempotent writes.
        on_retry: Called with a `RetryEvent` for each failed try that was
            retryable, the last one included, in the calling thread (on the
            event loop for `AsyncClient`); an exception it raises ends the
            call with that exception.
    """

    predict: Backoff = field(default_factory=_predict_backoff)
    control: Backoff = field(default_factory=_control_backoff)
    on_retry: Callable[[RetryEvent], Any] | None = None

    @classmethod
    def of(cls, retry: RetryPolicy | int | bool | None) -> RetryPolicy | None:
        """A client's `retry` argument as a policy; None when it retries nothing.

        A policy is itself; None, False or 0 retries nothing; True is the
        default; a positive integer is the default with that many tries at
        most, the first included, for every class.
        """
        if isinstance(retry, RetryPolicy):
            return retry
        if retry is None or retry is False:
            return None
        if retry is True:
            return cls()
        if type(retry) is not int or retry < 0:
            raise ValueError(
                "retry is a RetryPolicy, a number of tries, True, or None/0 for none"
            )
        if retry <= 1:
            return None
        default = cls()
        return replace(
            default,
            predict=replace(default.predict, attempts=retry),
            control=replace(default.control, attempts=retry),
        )

    def backoff(self, retry_class: str) -> Backoff:
        """The backoff of a retry class."""
        return self.predict if retry_class == PREDICT else self.control


@dataclass
class Tries:
    """One request's tries under a policy: each try's deadline, and whether to try again.

    The clients drive it: they send with `deadline`, hand a failure to
    `failed`, and wait the seconds it returns or raise when it returns None.
    """

    policy: RetryPolicy
    retry_class: str
    method: str
    path: str
    request_id: str | None
    limit: float | None = None
    clock: Callable[[], float] = time.monotonic
    draw: Callable[[], float] = random.random
    attempt: int = 0
    unknown: bool = False
    started: float = 0.0
    end: float = 0.0

    def __post_init__(self) -> None:
        """Starts the request's clock; its end is its class's deadline or the call's own."""
        self.started = self.clock()
        self.end = self.started + self.policy.backoff(self.retry_class).deadline
        if self.limit is not None:
            self.end = min(self.end, self.limit)

    @property
    def deadline(self) -> float:
        """The `time.monotonic()` moment every try must finish by."""
        return self.end

    def failed(self, error: BaseException) -> float | None:
        """The seconds to wait before trying again after `error`; None to raise it.

        Marks `error` with whether an earlier try's outcome was unknown, so an
        operation can tell its own earlier try from someone else's write.
        """
        self.attempt += 1
        if isinstance(error, Error):
            vars(error)["_after_unknown_outcome"] = self.unknown
        if not retryable(error):
            return None
        assert isinstance(error, Error)
        self.unknown = self.unknown or outcome_unknown(error)
        backoff = self.policy.backoff(self.retry_class)
        asked = error.retry_after
        wait: float | None = (
            asked if asked is not None else backoff.wait(self.attempt, self.draw)
        )
        now = self.clock()
        if self.attempt >= backoff.attempts or now + (wait or 0.0) >= self.end:
            wait = None
        event = RetryEvent(
            self.retry_class,
            self.method,
            self.path,
            self.request_id,
            self.attempt,
            error,
            now - self.started,
            wait,
            asked,
        )
        RETRIES.info(
            "%s %s %s on try %d: %s %s",
            self.method,
            self.path.split("?", 1)[0],
            "gave up" if wait is None else f"retrying in {wait:.1f}s",
            self.attempt,
            error.status,
            error.code,
        )
        if self.policy.on_retry is not None:
            self.policy.on_retry(event)
        return wait


def after_unknown_outcome(error: BaseException) -> bool:
    """Whether an error came after a try of the same request whose outcome was unknown."""
    return bool(getattr(error, "_after_unknown_outcome", False))


# What a client retries under when it is given no `retry`.
DEFAULT_RETRY: Final = RetryPolicy()
