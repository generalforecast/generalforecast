# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The errors every General Forecast surface raises, shaped as the CLI prints them.

One hierarchy for the SDK, the clients, the CLI, the engine boundary and the
services. Every error is an `Error` of one category: `InvalidInput`,
`NotFound`, `Unauthorized`, `CapacityExceeded`, `ModelNotReady`, `Conflict`,
`Internal`, `AmbiguousReference`, `Unavailable` or `Unsupported`. An error made from a code,
such as `ServiceError("WriteConflict", ..., 409)`, takes the category its
code and status name, so `except gf.Conflict` catches it whichever layer
raised it. The HTTP error object (section 5 of the interface contract) maps
onto the same fields, so a server reply and a local mistake render alike.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

WIRE_ERROR_CODES = frozenset(
    [
        "ValidationError",
        "NotFound",
        "Unauthorized",
        "CapacityExceeded",
        "ModelNotReady",
        "Internal",
        "Conflict",
        "Unsupported",
    ]
)
"""The OpenAPI `ErrorCode` enum: the only codes a GF service puts on the wire."""

# Codes that name their wire code whatever status they are raised with. This
# is the services' wire contract: a change here changes what they send.
_PUBLIC_BY_CODE: dict[str, str] = {
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
            "WaitTimeout",
            "PredictorTimeout",
        ),
        "ModelNotReady",
    ),
    "InternalError": "Internal",
}

# Categories of codes raised on this side of the wire, which a service never
# sends: an unreachable or silent service, a closed client, a model's own
# fault, and the engine's refusals named by what they refuse. Every other
# code takes its wire code's category.
_LOCAL_CATEGORY: dict[str, str] = {
    "TransportError": "Unavailable",
    "Timeout": "Unavailable",
    "ClientClosed": "InvalidInput",
    "AmbiguousReference": "AmbiguousReference",
    "ModelOutputError": "Internal",
    "Unauthenticated": "Unauthorized",
    "UnknownModel": "NotFound",
    "QueryCapacityExceeded": "CapacityExceeded",
    **dict.fromkeys(
        (
            "InvalidInput",
            "InvalidTable",
            "InvalidModel",
            "InvalidPredictor",
            "InvalidContext",
            "SchemaRequired",
            "UnsupportedCapability",
            "ModelMismatch",
            "AmbiguousTask",
            "ContextRequired",
            "EmptyContext",
        ),
        "InvalidInput",
    ),
}


def reason_of(code: str) -> str:
    """A service code's wire `reason`: its name in snake case."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", code).lower()


def public_code(code: str, status: int | None) -> str:
    """The wire code for a failure code raised with `status`."""
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
    if status == 501:
        return "Unsupported"
    # A service that is down or answered too late: retry shortly.
    if status in (503, 504):
        return "ModelNotReady"
    if status is not None and status >= 500:
        return "Internal"
    return "ValidationError"


# Category names that are spelled differently as codes.
_CODE_ALIASES = {"InvalidInput": "ValidationError"}

# Codes that are their own reason: the wire codes.
_UNREASONED = WIRE_ERROR_CODES

# Codes only a client raises about its own side of the wire; a server that
# sends one is not believed.
_CLIENT_ONLY = frozenset(
    {"TransportError", "ClientClosed", "RedirectRefused", "ModelOutputError"}
)


def known_code(code: str) -> bool:
    """Whether a code names its category without a status."""
    return (
        code in WIRE_ERROR_CODES or code in _PUBLIC_BY_CODE or code in _LOCAL_CATEGORY
    )


_ESCAPES = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])|\x9b[0-?]*[ -/]*[@-~]"
)


def printable(message: str) -> str:
    """Removes terminal escapes and non-printable characters from a message."""
    return "".join(
        " " if character in "\t\n\r" else character
        for character in _ESCAPES.sub("", message)
        if character in "\t\n\r" or character.isprintable()
    ).strip()


def _wrap(label: str, text: str) -> str:
    head = f"  {label}: "
    pad = " " * len(head)
    lines = text.splitlines() or [""]
    return head + ("\n" + pad).join(lines)


# The optional wire fields of an error object, in the order they are sent.
_DETAILS = ("param", "hint", "limit", "actual")


class Error(Exception):
    """A General Forecast error with the fields the CLI and the API render.

    Attributes:
        code: The stable error code, such as `ValidationError`.
        message: One lowercase line naming the mistake, without a trailing period.
        status: The HTTP status the error answers with.
        reason: The finer reason behind `code`, in snake case, if any.
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
    # The status an error of this category answers with when none is given.
    default_status: ClassVar[int] = 500
    # Set on the category classes; every other class takes one from its code.
    _category: ClassVar[bool] = False

    def __init__(
        self,
        *args: Any,
        code: str | None = None,
        message: str | None = None,
        status: int | None = None,
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
        """Builds an error from `(message)`, `(code, message)` or `(code, message, status)`.

        `code` defaults to the class's code, and an error of a class that is
        not a category, such as `ServiceError`, becomes one of its category.
        """
        if len(args) == 3:
            code, message, status = args
        elif len(args) == 2:
            code, message = args
        elif len(args) == 1:
            message = args[0]
        elif args:
            raise TypeError("Error takes a message, a code and a message, and a status")
        self.code = (
            type(self).code
            if code is None
            # A category's name is spelled as its code; a service's own code
            # stays as it was raised.
            else _CODE_ALIASES.get(code, code)
            if type(self) is Error or type(self)._category
            else code
        )
        self.message = "" if message is None else message
        self.reason = (
            reason
            if reason is not None or self.code in _UNREASONED
            else reason_of(self.code)
        )
        self.param = param
        self.because = because
        self.hint = hint
        self.limit = limit
        self.actual = actual
        self.request_id = request_id
        self.path = path
        self.line = line
        self.column = column
        super().__init__(self.message)
        self.__class__ = _classified(type(self), self.code, status)
        self.status = type(self).default_status if status is None else status

    @property
    def public(self) -> str:
        """The wire code, one of the OpenAPI `ErrorCode` enum."""
        return public_code(self.code, self.status)

    def details(self) -> dict[str, Any]:
        """The optional wire fields that are set: `param`, `hint`, `limit` and `actual`."""
        return {
            key: value
            for key in _DETAILS
            if (value := getattr(self, key, None)) is not None
        }

    def error(self) -> dict[str, Any]:
        """The wire error object: the OpenAPI code, message, reason and details."""
        error: dict[str, Any] = {"code": self.public, "message": self.message}
        if self.reason is not None:
            error["reason"] = self.reason
        return error | self.details()

    def public_body(self) -> dict[str, Any]:
        """The API's bare error envelope, with the OpenAPI code."""
        return {"error": self.error()}

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
        """The `--json` error object, with the error's own code."""
        value: dict[str, Any] = {"code": self.code, "message": self.message}
        for key in ("reason", *_DETAILS, "request_id"):
            if (item := getattr(self, key, None)) is not None:
                value[key] = item
        return value

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickles the declared class and fields; unpickling classifies again."""
        return (_restore, (_declared(type(self)), self.args), self.__dict__)

    def __setstate__(self, state: dict[str, Any] | None, /) -> None:
        """Restores the fields and the category they name."""
        self.__dict__.update(state or {})
        self.__class__ = _classified(type(self), self.code, self.status)


class InvalidInput(Error):
    """A manifest, flag, column, reference or request value is wrong."""

    code = "ValidationError"
    default_status = 422
    _category = True


class NotFound(Error):
    """A predictor, model, context, revision or file that is not there."""

    code = "NotFound"
    default_status = 404
    _category = True


class Unauthorized(Error):
    """No credential, a bad credential, a missing permission or the wrong project."""

    code = "Unauthorized"
    default_status = 401
    _category = True


class CapacityExceeded(Error):
    """A request is larger than the model or the plan allows; see `limit` and `actual`."""

    code = "CapacityExceeded"
    default_status = 413
    _category = True


class ModelNotReady(Error):
    """The model or environment is still loading or compiling; retry shortly."""

    code = "ModelNotReady"
    default_status = 503
    _category = True


class AmbiguousReference(Error):
    """A reference names both a local predictor and a hosted one."""

    code = "AmbiguousReference"
    default_status = 422
    _category = True


class Conflict(Error):
    """A compare-and-swap lost, or a resource already exists or moved meanwhile."""

    code = "Conflict"
    default_status = 409
    _category = True


class Unavailable(Error):
    """The service could not be reached or did not answer in time.

    The outcome of a write is unknown: retry it with the same `write_id` or
    `submission_id`.
    """

    code = "Unavailable"
    default_status = 503
    _category = True


class Unsupported(Error):
    """An operation this server or platform does not offer, such as a hosted route on gf serve."""

    code = "Unsupported"
    default_status = 501
    _category = True


class Internal(Error):
    """Our fault; always carries a request id when it came from a server."""

    code = "Internal"
    default_status = 500
    _category = True


_BY_WIRE: dict[str, type[Error]] = {
    "ValidationError": InvalidInput,
    "NotFound": NotFound,
    "Unauthorized": Unauthorized,
    "CapacityExceeded": CapacityExceeded,
    "ModelNotReady": ModelNotReady,
    "Conflict": Conflict,
    "Internal": Internal,
    "Unsupported": Unsupported,
}
_BY_NAME: dict[str, type[Error]] = {
    "InvalidInput": InvalidInput,
    "AmbiguousReference": AmbiguousReference,
    "Unavailable": Unavailable,
    **{klass.__name__: klass for klass in _BY_WIRE.values()},
}


def category(code: str, status: int | None = None) -> type[Error]:
    """The category of an error raised with `code` and `status`."""
    if status == 501:
        return Unsupported
    local = _LOCAL_CATEGORY.get(code)
    if local is not None:
        return _BY_NAME[local]
    if code in _BY_NAME:
        return _BY_NAME[code]
    if status == 413:
        return CapacityExceeded
    if status is None and not known_code(code) and not code.endswith("NotFound"):
        return InvalidInput
    return _BY_WIRE[public_code(code, status)]


_MIXED: dict[tuple[type[Error], type[Error]], type[Error]] = {}


def _classified(cls: type[Error], code: str, status: int | None) -> type[Error]:
    """The class an error of `cls` raised with `code` and `status` is.

    A category class stays itself. `Error` becomes the category; another
    class, such as `ServiceError`, becomes a subclass of both, so it is
    caught both as itself and as its category.
    """
    cls = _declared(cls)
    if cls._category:
        return cls
    kind = category(code, status)
    if cls is Error:
        return kind
    key = (cls, kind)
    if key not in _MIXED:
        _MIXED[key] = type(
            cls.__name__,
            (cls, kind),
            {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__doc__": cls.__doc__,
                "_declared_class": cls,
            },
        )
    return _MIXED[key]


def _declared(cls: type[Error]) -> type[Error]:
    """The class an error was raised as, before its category was mixed in."""
    return cls.__dict__.get("_declared_class", cls)


def _restore(cls: type[Error], args: tuple[Any, ...]) -> Error:
    error = Exception.__new__(cls)
    error.args = args
    return error


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
    "Unavailable",
    "Unsupported",
    "boundary",
    "category",
    "convert",
    "from_wire",
    "wire_fields",
]


_ERROR_KEYS = frozenset({"code", "message", "reason", *_DETAILS, "request_id"})
# The longest server message relayed as it is; longer ones are replaced.
MAX_MESSAGE = 512


def wire_fields(
    detail: Any, *, status: int | None = None, request_id: str | None = None
) -> dict[str, Any] | None:
    """The fields of a reply's error object, made printable, or None if it is malformed.

    A code outside the known vocabulary becomes `RemoteError`, so an untrusted
    server's codes are never echoed; its category then follows the status.
    An empty or overlong message is left empty.
    """
    if (
        not isinstance(detail, Mapping)
        or not {"code", "message"} <= set(detail)
        or not set(detail) <= _ERROR_KEYS
        or any(type(detail[key]) is not str for key in ("code", "message"))
        or any(
            detail.get(key) is not None and type(detail[key]) is not str
            for key in ("reason", "param", "hint", "request_id")
        )
        or any(
            detail.get(key) is not None
            and (type(detail[key]) not in (int, float) or type(detail[key]) is bool)
            for key in ("limit", "actual")
        )
    ):
        return None
    code = detail["code"]
    message = detail["message"]
    fields: dict[str, Any] = {
        "code": code
        if known_code(code) and code not in _CLIENT_ONLY
        else "RemoteError",
        "message": printable(message) if len(message) <= MAX_MESSAGE else "",
        "status": status,
        "request_id": detail.get("request_id") or request_id,
    }
    for key in ("reason", *_DETAILS):
        value = detail.get(key)
        if value is not None:
            fields[key] = printable(value) if type(value) is str else value
    if fields["request_id"] is not None:
        fields["request_id"] = printable(fields["request_id"])
    return fields


def from_wire(
    error: Any, *, status: int | None = None, request_id: str | None = None
) -> Error:
    """The error a reply's error object describes, its category from its code and status.

    A malformed error object is an `Internal` `ProtocolError`.
    """
    fields = wire_fields(error, status=status, request_id=request_id)
    if fields is None:
        return Error(
            "ProtocolError",
            "invalid service error",
            status=status,
            request_id=request_id,
        )
    if not fields["message"]:
        fields["message"] = "service rejected request"
    return Error(**fields)


# Where an uncoded ValueError, TypeError or KeyError says the caller's input
# or the engine's answer was refused: the engine, and the data libraries it
# hands the caller's rows to. Raised there, its type is the documented signal
# (pyarrow raises gRPC NOT_FOUND as a KeyError and INVALID_ARGUMENT as a
# ValueError; sklearn refuses inconsistent arrays with a ValueError). Raised
# anywhere else it is a bug, and stays itself.
_ENGINE_MODULES = (
    "generalforecast.native",
    "generalforecast.keys",
    "generalforecast.columns",
    "generalforecast.engine",
    "generalforecast.tabular",
    "generalforecast.models",
    "generalforecast.predictors",
    "pyarrow",
    "numpy",
    "pandas",
    "sklearn",
)


def _engine_raised(exc: BaseException) -> bool:
    """Whether the innermost frame of an exception's traceback is engine code."""
    trace = exc.__traceback__
    if trace is None:
        return False
    while trace.tb_next is not None:
        trace = trace.tb_next
    module = str(trace.tb_frame.f_globals.get("__name__", ""))
    return any(
        module == name or module.startswith(name + ".") for name in _ENGINE_MODULES
    )


def convert(exc: BaseException) -> Error | None:
    """The public error for an exception another layer raised, or None.

    Maps by code first: an exception with a string `code` (a predictor,
    model reference or artifact error) or the engine's coded `RequestError`.
    Then by cause: a Flight status, an Arrow error about the caller's data, a
    file the caller named, a refused credential, or a `ValueError`,
    `TypeError` or `KeyError` the engine raised for the caller's input.
    Returns None for anything else, such as a bug, which stays itself.
    """
    if isinstance(exc, Error):
        return exc
    code = getattr(exc, "code", None)
    if not isinstance(code, str) and type(exc).__name__ == "RequestError":
        parts = exc.args
        if len(parts) == 3 and isinstance(parts[0], str) and isinstance(parts[2], int):
            return Error(parts[0], _lower(str(parts[1])), status=parts[2])
    message = getattr(exc, "message", None) or _text(exc)
    if isinstance(code, str):
        status = getattr(exc, "status", None)
        return Error(
            code,
            _lower(str(message)),
            status=status if isinstance(status, int) else None,
            reason=getattr(exc, "reason", None),
            param=getattr(exc, "param", None),
            hint=getattr(exc, "hint", None),
            limit=getattr(exc, "limit", None),
            actual=getattr(exc, "actual", None),
            request_id=getattr(exc, "request_id", None),
        )
    name = type(exc).__module__ + "." + type(exc).__name__
    if name.startswith("pyarrow._flight."):
        return _flight(exc, message)
    if _FRAMING.match(message.strip()):
        return _framed(message)
    if name.startswith("pyarrow.lib.Arrow"):
        return _framed(message)
    if isinstance(exc, PermissionError) and _engine_raised(exc):
        return Unauthorized(_lower(_strip(message)))
    if isinstance(exc, OSError) and getattr(exc, "filename", None) is not None:
        text = _lower(_strip(exc.strerror or message))
        named = f"{text}: {exc.filename}"
        if isinstance(exc, FileNotFoundError):
            return NotFound(named)
        return InvalidInput(named)
    if isinstance(exc, KeyError) and _engine_raised(exc):
        return NotFound(_lower(_strip(message)))
    if isinstance(exc, (ValueError, TypeError, ArithmeticError)) and _engine_raised(
        exc
    ):
        return _framed(message)
    return None


def _text(exc: BaseException) -> str:
    """An exception's message; a KeyError's without the quotes `str` adds."""
    if (
        isinstance(exc, KeyError)
        and len(exc.args) == 1
        and isinstance(exc.args[0], str)
    ):
        return exc.args[0] or type(exc).__name__
    return str(exc) or type(exc).__name__


def _by_text(text: str) -> Error:
    if "not found" in text or "does not exist" in text:
        return NotFound(text)
    if "unknown" in text and "table" in text:
        return NotFound(text)
    if "capacity" in text or "too many" in text or "exceed" in text:
        return CapacityExceeded(text)
    return InvalidInput(text)


def _flight(exc: BaseException, message: str) -> Error:
    by_name: dict[str, type[Error]] = {
        "FlightUnauthenticatedError": Unauthorized,
        "FlightUnauthorizedError": Unauthorized,
        "FlightUnavailableError": Unavailable,
        "FlightTimedOutError": Unavailable,
        "FlightCancelledError": Unavailable,
        "FlightInternalError": Internal,
        "FlightServerError": Internal,
    }
    text = _lower(_strip(message))
    if (found := by_name.get(type(exc).__name__)) is not None:
        return found(text)
    return _framed(message)


# How pyarrow frames a server's status: `Flight returned <status> error: ...`
# or `Flight returned <status> error, with message: ...`.
_FRAMING = re.compile(r"^Flight returned ([a-z ]+) error(?:, with message)?:\s*")
_BY_FLIGHT: dict[str, type[Error]] = {
    "already exists": Conflict,
    "not found": NotFound,
    "unauthenticated": Unauthorized,
    "unauthorized": Unauthorized,
    "resource exhausted": CapacityExceeded,
    "unavailable": Unavailable,
    "internal": Internal,
}


def _framed(message: str) -> Error:
    """The error a Flight status names, else the one its text suggests."""
    found = _FRAMING.match(message.strip())
    text = _lower(_strip(message))
    if found is not None and (kind := _BY_FLIGHT.get(found.group(1))) is not None:
        return kind(text)
    return _by_text(text)


def _strip(message: str) -> str:
    """Drops Arrow and Flight framing and gRPC detail from a message."""
    text = _FRAMING.sub("", message.strip(), count=1)
    text = text.removeprefix("Flight returned ")
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
