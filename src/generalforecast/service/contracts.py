# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The service contract: wire errors, field checks and request limits.

Every service surface, `gf serve`, the hosted plane and the SDK, raises and
reads these errors and validates request fields with these checks. Names
follow `generalforecast.predictors.validation.NAME`, the one rule for the
names people choose.
"""

import math
import re
import threading
import time
from typing import Any

from generalforecast.predictors.validation import NAME

# The wire vocabulary lives with `Error`; services import it from here too.
from generalforecast.errors import (  # noqa: F401
    WIRE_ERROR_CODES,
    Error,
    public_code,
    reason_of,
)

VERSION = "gf/v1"


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
    """Validates a chosen name by the one rule, `predictors.validation.NAME`."""
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
