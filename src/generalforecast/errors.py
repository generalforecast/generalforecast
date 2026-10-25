# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The errors every General Forecast surface raises, shaped as the CLI prints them.

One hierarchy for the SDK, the CLI and the engine boundary. The HTTP error
object (section 5 of the interface contract) maps onto the same fields, so a
server reply and a local mistake render identically.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def _wrap(label: str, text: str) -> str:
    head = f"  {label}: "
    pad = " " * len(head)
    lines = text.splitlines() or [""]
    return head + ("\n" + pad).join(lines)


class Error(Exception):
    """A General Forecast error with the fields the CLI and the API render.

    Attributes:
        code: The stable error code, such as `ValidationError`.
        message: One lowercase line naming the mistake, without a trailing period.
        reason: The service's finer reason behind `code`, in snake case, if any.
        param: The field or column the error refers to, if any.
        because: The cause, when the message alone does not explain it.
        hint: What to do instead, if there is a short answer.
        limit: The bound a limit error enforces, if any.
        actual: The value that exceeded the bound, if any.
        request_id: The server request id, if the error came from a request.
        path: The file the error points at, if any.
        line: The 1-based `gf.yaml` line the error points at, if any.
        column: The 1-based column on that line, if any.
    """

    code = "Internal"

    def __init__(
        self,
        *args: str,
        code: str | None = None,
        message: str | None = None,
        reason: str | None = None,
        param: str | None = None,
        because: str | None = None,
        hint: str | None = None,
        limit: int | None = None,
        actual: int | None = None,
        request_id: str | None = None,
        path: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        """Builds an error from `(message)` or `(code, message)`; `code` defaults to the class's code."""
        if len(args) == 2:
            code, message = args[0], args[1]
        elif len(args) == 1:
            message = args[0]
        elif args:
            raise TypeError("Error takes a message, or a code and a message")
        if code is not None:
            self.code = _CODES.get(code, code)
        self.message = message or ""
        self.reason = reason
        self.path = path
        self.param = param
        self.because = because
        self.hint = hint
        self.limit = limit
        self.actual = actual
        self.request_id = request_id
        self.line = line
        self.column = column
        super().__init__(self.message)

    def __str__(self) -> str:
        """The CLI's plain shape, without colour: error, because, hint, request id."""
        lines = [f"error: {self.message}"]
        if self.because:
            lines.append(_wrap("because", self.because))
        if self.hint:
            lines.append(_wrap("hint", self.hint))
        if self.request_id:
            lines.append(f"  request {self.request_id}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The `--json` and HTTP error object."""
        value: dict[str, Any] = {"code": self.code, "message": self.message}
        for key in ("param", "hint", "limit", "actual", "request_id"):
            if (item := getattr(self, key)) is not None:
                value[key] = item
        return value

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, request_id: str | None = None
    ) -> Error:
        """The error class for a server error object."""
        code = str(value.get("code", "Internal"))
        klass = _BY_CODE.get(code, Internal)
        return klass(
            str(value.get("message", "")),
            code=code,
            param=value.get("param"),
            hint=value.get("hint"),
            limit=value.get("limit"),
            actual=value.get("actual"),
            request_id=request_id or value.get("request_id"),
        )


class InvalidInput(Error):
    """A manifest, flag, column, reference or request value is wrong."""

    code = "ValidationError"


class NotFound(Error):
    """A predictor, model, context, revision or file that is not there."""

    code = "NotFound"


class Unauthorized(Error):
    """No credential, a bad credential or the wrong project."""

    code = "Unauthorized"


class CapacityExceeded(Error):
    """A request is larger than the model or the plan allows; see `limit` and `actual`."""

    code = "CapacityExceeded"


class ModelNotReady(Error):
    """The model is still loading or compiling; retry shortly."""

    code = "ModelNotReady"


class AmbiguousReference(Error):
    """A reference names both a local predictor and a hosted one."""

    code = "AmbiguousReference"


class Conflict(Error):
    """A compare-and-swap lost: the generation moved since it was read."""

    code = "Conflict"


class Internal(Error):
    """Our fault; always carries a request id when it came from a server."""

    code = "Internal"


_CODES: dict[str, str] = {
    "InvalidInput": "ValidationError",
    "NotFound": "NotFound",
    "Unauthorized": "Unauthorized",
    "CapacityExceeded": "CapacityExceeded",
    "ModelNotReady": "ModelNotReady",
    "AmbiguousReference": "AmbiguousReference",
    "Conflict": "Conflict",
    "Internal": "Internal",
}

_BY_CODE: dict[str, type[Error]] = {
    klass.code: klass
    for klass in (
        InvalidInput,
        NotFound,
        Unauthorized,
        CapacityExceeded,
        ModelNotReady,
        AmbiguousReference,
        Conflict,
        Internal,
    )
}

__all__ = [
    "AmbiguousReference",
    "CapacityExceeded",
    "Conflict",
    "Error",
    "Internal",
    "InvalidInput",
    "ModelNotReady",
    "NotFound",
    "Unauthorized",
]


_ALIASES: dict[str, type[Error]] = {
    "InvalidInput": InvalidInput,
    "InvalidTable": InvalidInput,
    "InvalidModel": InvalidInput,
    "InvalidPredictor": InvalidInput,
    "InvalidContext": InvalidInput,
    "SchemaRequired": InvalidInput,
    "UnsupportedCapability": InvalidInput,
    "ModelMismatch": InvalidInput,
    "AmbiguousTask": InvalidInput,
    "ContextRequired": InvalidInput,
    "EmptyContext": InvalidInput,
    "UnknownModel": NotFound,
    "ModelNotFound": NotFound,
    "NotReady": ModelNotReady,
    "RuntimeUnavailable": ModelNotReady,
    "WriteConflict": Conflict,
    "Unauthenticated": Unauthorized,
    "AccessDenied": Unauthorized,
    "CredentialInactive": Unauthorized,
    "CredentialUnavailable": Unauthorized,
    "CredentialsUnavailable": Unauthorized,
    "ContextCapacityExceeded": CapacityExceeded,
    "QueryCapacityExceeded": CapacityExceeded,
}

_BY_STATUS: dict[int, type[Error]] = {
    400: InvalidInput,
    401: Unauthorized,
    403: Unauthorized,
    404: NotFound,
    409: Conflict,
    413: CapacityExceeded,
    422: InvalidInput,
    429: CapacityExceeded,
    503: ModelNotReady,
}


def _class_for(code: str, status: int | None) -> type[Error]:
    found = _BY_CODE.get(code) or _ALIASES.get(code)
    if found is not None:
        return found
    if status is not None and status in _BY_STATUS:
        return _BY_STATUS[status]
    return Internal


def from_wire(
    error: dict[str, Any], *, status: int | None = None, request_id: str | None = None
) -> Error:
    """The error for a reply's error object, falling back on the HTTP status."""
    code = str(error.get("code", "Internal"))
    return _class_for(code, status)(
        str(error.get("message", "service rejected request")),
        code=code,
        param=error.get("param"),
        hint=error.get("hint"),
        limit=error.get("limit"),
        actual=error.get("actual"),
        request_id=error.get("request_id", request_id),
    )


def convert(exc: BaseException) -> Error | None:
    """The public error for an exception another layer raised, or None.

    Maps the service error, the predictor errors, model reference errors,
    Arrow and Flight failures, and the `ValueError`/`TypeError` the native
    tables raise for a caller's mistake. Returns None for errors that are not
    the caller's.
    """
    if isinstance(exc, Error):
        return exc
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    if isinstance(code, str):
        status = getattr(exc, "status", None)
        found = _BY_CODE.get(code) or _ALIASES.get(code)
        if found is None:
            if isinstance(status, int) and status in _BY_STATUS:
                found = _BY_STATUS[status]
            elif isinstance(exc, (ValueError, TypeError)):
                found = InvalidInput
            else:
                found = Internal
        return found(
            _lower(message),
            code=code,
            param=getattr(exc, "param", None),
            hint=getattr(exc, "hint", None),
            limit=getattr(exc, "limit", None),
            actual=getattr(exc, "actual", None),
            request_id=getattr(exc, "request_id", None),
        )
    name = type(exc).__module__ + "." + type(exc).__name__
    if name.startswith("pyarrow._flight."):
        return _flight(exc, message)
    if name.startswith("pyarrow.lib.Arrow") or isinstance(exc, (ValueError, TypeError)):
        return _by_text(_lower(_strip(message)))
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return NotFound(_lower(_strip(message)))
    if isinstance(exc, PermissionError):
        return Unauthorized(_lower(_strip(message)))
    if isinstance(exc, (OSError, OverflowError, ArithmeticError)):
        return InvalidInput(_lower(_strip(message)))
    return None


def _by_text(text: str) -> Error:
    if "not found" in text or ("unknown" in text and "table" in text):
        return NotFound(text)
    if "capacity" in text or "too many" in text or "exceed" in text:
        return CapacityExceeded(text)
    return InvalidInput(text)


def _flight(exc: BaseException, message: str) -> Error:
    by_name: dict[str, type[Error]] = {
        "FlightUnauthenticatedError": Unauthorized,
        "FlightUnauthorizedError": Unauthorized,
        "FlightUnavailableError": ModelNotReady,
        "FlightTimedOutError": ModelNotReady,
        "FlightCancelledError": ModelNotReady,
        "FlightInternalError": Internal,
        "FlightServerError": Internal,
    }
    text = _lower(_strip(message))
    if (found := by_name.get(type(exc).__name__)) is not None:
        return found(text)
    return _by_text(text)


def _strip(message: str) -> str:
    """Drops Arrow and Flight framing and gRPC detail from a message."""
    text = message.strip()
    for prefix in ("Flight returned invalid argument error: ", "Flight returned "):
        text = text.removeprefix(prefix)
    for marker in (". Detail:", ". gRPC client debug context:", "Detail: "):
        cut = text.find(marker)
        if cut > 0:
            text = text[:cut]
    if text.lower().startswith("invalid: "):
        text = text[9:]
    return text.strip().rstrip(".")


def _lower(message: str) -> str:
    return message[:1].lower() + message[1:] if message else message


@contextmanager
def boundary() -> Iterator[None]:
    """Raises every mapped failure inside the block as a public error."""
    try:
        yield
    except Error:
        raise
    except Exception as exc:
        mapped = convert(exc)
        if mapped is None:
            raise
        raise mapped from exc


__all__ += ["boundary", "convert", "from_wire"]
