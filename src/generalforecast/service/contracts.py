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

VERSION = "gf/v1"


class ServiceError(Exception):
    """A public error with no prediction or private runtime details."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        """Constructs the safe public error envelope.

        A code outside the OpenAPI enum folds into it on the wire and leaves
        its own name as the `reason`, in snake case; an enum code has none.
        """
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.reason = None if code in WIRE_ERROR_CODES else reason_of(code)

    @property
    def public(self) -> str:
        """The wire code, one of the OpenAPI `ErrorCode` enum."""
        return public_code(self.code, self.status)

    def details(self) -> dict[str, Any]:
        """Extra safe fields of the error object, none by default."""
        return {}

    def error(self) -> dict[str, Any]:
        """The wire error object: the OpenAPI code, message and safe details."""
        error: dict[str, Any] = {"code": self.public, "message": self.message}
        if self.reason is not None:
            error["reason"] = self.reason
        return error | self.details()

    def public_body(self) -> dict[str, Any]:
        """The API's bare error envelope, with the OpenAPI code."""
        return {"error": self.error()}

    body = public_body


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
        super().__init__(code, message, status)
        self.param, self.hint, self.limit, self.actual = param, hint, limit, actual

    def details(self) -> dict[str, Any]:
        """`param`, `hint`, `limit` and `actual`, those that are set."""
        return {
            key: value
            for key in ("param", "hint", "limit", "actual")
            if (value := getattr(self, key)) is not None
        }


WIRE_ERROR_CODES = frozenset(
    [
        "ValidationError",
        "NotFound",
        "Unauthorized",
        "CapacityExceeded",
        "ModelNotReady",
        "Internal",
        "Conflict",
    ]
)
"""The OpenAPI `ErrorCode` enum: the only codes a GF service puts on the wire."""


def reason_of(code: str) -> str:
    """A service code's wire `reason`: its name in snake case."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", code).lower()


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

_PUBLIC_BY_CODE = {
    **dict.fromkeys(
        (
            "Forbidden",
            "AccessDenied",
            "InvalidLogin",
            "ExpiredToken",
            "NotLoggedIn",
            "SessionRequired",
            "ReauthenticationRequired",
            "LinkConfirmationRequired",
            "WaitlistRequired",
            "CredentialInactive",
            "AuthorizationPending",
        ),
        "Unauthorized",
    ),
    **dict.fromkeys(
        (
            "AlreadyExists",
            "WriteConflict",
            "StaleComponent",
            "OperationSuperseded",
        ),
        "Conflict",
    ),
    **dict.fromkeys(("RateLimited", "ContextCapacityExceeded"), "CapacityExceeded"),
    **dict.fromkeys(
        (
            "NotReady",
            "ModelUnavailable",
            "RuntimeUnavailable",
            "ProviderUnavailable",
            "SingleWriterUnavailable",
            "Timeout",
            "WaitTimeout",
            "PredictorTimeout",
        ),
        "ModelNotReady",
    ),
    "InternalError": "Internal",
}


def public_code(code: str, status: int) -> str:
    """The wire code for an internal failure code raised with `status`."""
    if code in WIRE_ERROR_CODES:
        return code
    if code in _PUBLIC_BY_CODE:
        return _PUBLIC_BY_CODE[code]
    if code.endswith("NotFound") or status == 404:
        return "NotFound"
    if status in (401, 403):
        return "Unauthorized"
    if status == 409:
        return "Conflict"
    if status == 429:
        return "CapacityExceeded"
    if status == 503:
        return "ModelNotReady"
    if status >= 500:
        return "Internal"
    return "ValidationError"


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
