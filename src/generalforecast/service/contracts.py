# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The alpha predictor contract, independent of runtime provisioning."""

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Required, TypedDict

# The wire vocabulary lives with `Error`; services import it from here too.
from generalforecast.errors import (  # noqa: F401
    WIRE_ERROR_CODES,
    Error,
    public_code,
    reason_of,
)

VERSION = "gf/v1"
MODEL = "general-forecast/tabicl-v2-classifier"
PIN = "5fb6a4c6f24bd5fb3623f97e3c62186e12e28e5b"
NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")


class ServiceError(Error):
    """A public error with no prediction or private runtime details.

    An `Error` raised from a code and an HTTP status, as services and clients
    raise them: it is also an error of the category its code names, so
    `except gf.NotFound` catches `ServiceError("NotFound", ...)`.
    """

    def __init__(
        self, code: str, message: str, status: int = 400, **details: Any
    ) -> None:
        """Constructs the safe public error envelope.

        A code outside the OpenAPI enum folds into it on the wire and leaves
        its own name as the `reason`, in snake case; an enum code has none.
        `details` are the optional `Error` fields, such as `param` or `hint`.
        """
        super().__init__(code, message, status, **details)

    def __str__(self) -> str:
        """The message alone, as services embed it in replies and logs."""
        return self.message


class InputError(ServiceError):
    """A request error that names its field and, for a limit, its bound."""

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 422,
        *,
        param: str | None = None,
        hint: str | None = None,
        limit: int | None = None,
        actual: int | None = None,
    ) -> None:
        """Keeps the OpenAPI error object's optional fields; None leaves one out."""
        super().__init__(
            code, message, status, param=param, hint=hint, limit=limit, actual=actual
        )


SESSION_REASONS = {
    code: reason_of(code)
    for code in (
        "ReauthenticationRequired",
        "LinkConfirmationRequired",
        "SessionRequired",
        "WaitlistRequired",
    )
}
"""Browser-session refusals: `Unauthorized` on the wire, with these reasons."""


def require(condition: bool, message: str) -> None:
    """Rejects malformed input before it reaches persistence or inference."""
    if not condition:
        raise ServiceError("ValidationError", message, 422)


def fields(value: Any, required: set[str], optional: set[str] | None = None) -> dict:
    """Checks object shape, including unknown keys."""
    require(isinstance(value, dict), "expected an object")
    require(required <= value.keys(), f"required fields: {sorted(required)}")
    require(not value.keys() - required - (optional or set()), "unknown fields")
    return value


MODEL_ID = re.compile(
    r"[a-z][a-z0-9-]{0,62}/[a-z0-9][a-z0-9.-]{0,62}(?:@sha256:[0-9a-f]{64})?(?:@\d+)?(?:\*\d+)?"
)


def model_id(value: Any) -> str:
    """Validates a canonical `general-forecast/<name>` model id."""
    require(
        isinstance(value, str) and MODEL_ID.fullmatch(value) is not None,
        "invalid model id",
    )
    return value


def name(value: Any) -> str:
    """Validates a stable resource name."""
    require(
        isinstance(value, str) and NAME.fullmatch(value) is not None, "invalid name"
    )
    return value


def text(value: Any, field: str) -> str:
    """Checks a bounded, nonempty identity string."""
    require(isinstance(value, str) and 0 < len(value) <= 256, f"invalid {field}")
    return value


def number(value: Any, low: float, high: float, field: str) -> float:
    """Checks a finite number without treating booleans as numbers."""
    require(
        type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
        f"{field} must be between {low} and {high}",
    )
    return value


@dataclass(frozen=True)
class ModelPin:
    """A curated model and its immutable Hugging Face commit."""

    repository: str
    revision: str

    @classmethod
    def parse(cls, value: Any) -> "ModelPin":
        """Rejects arbitrary code paths and mutable aliases."""
        fields(value, {"repository", "revision"})
        require(
            value["repository"] == MODEL,
            "only the native TabICL classifier is supported",
        )
        require(
            isinstance(value["revision"], str)
            and re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is not None,
            "model.revision must be a 40-character lowercase commit",
        )
        return cls(**value)


class Row(TypedDict, total=False):
    """Immutable feature observation; omitted label is unknown."""

    id: Required[str]
    features: Required[dict[str, float]]
    label: bool
    source_at: str
    label_available_at: str


class Label(TypedDict, total=False):
    """A delayed outcome bound to an existing immutable observation."""

    id: Required[str]
    label: Required[bool]
    label_available_at: str


class Freshness(TypedDict):
    """Accepted-update progress, separate from source/context age."""

    accepted_sequence: int
    processed_sequence: int
    pending_updates: int
    oldest_pending_at: float | None
    update_lag_seconds: float
    max_update_lag_seconds: float
    context_age_seconds: float | None
    max_label_delay_seconds: float | None


class Score(TypedDict):
    """A score and provenance, never a business action."""

    id: str
    probabilities: dict[str, float]
    predictor_revision: int
    model: dict[str, str]
    context: str
    serving_sequence: int
    freshness: Freshness


class RateLimit:
    """A token bucket: `rate` events per second with bursts up to `burst`."""

    def __init__(self, rate: float, burst: int, clock: Any = time.monotonic) -> None:
        """A full bucket that refills at `rate` per second on `clock`."""
        self.rate, self.burst, self.clock = rate, burst, clock
        self.tokens, self.at = float(burst), clock()
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Spends one token, or refuses when the bucket is empty."""
        with self._lock:
            now = self.clock()
            self.tokens = min(self.burst, self.tokens + (now - self.at) * self.rate)
            self.at = now
            if self.tokens < 1:
                return False
            self.tokens -= 1
            return True
