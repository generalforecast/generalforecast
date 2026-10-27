# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `gf/v1` request and result envelope, shared by HTTP and in-process adapters.

`gf serve`, hosted serving and the SDK parse and write the same envelope.
Parsing is not authorization or domain admission; each adapter authorizes
after parsing.
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypedDict

from generalforecast.service.contracts import (
    VERSION,
    ServiceError,
    fields,
    model_name,
    name,
    require,
    text,
)

MAX_BYTES = 1_048_576
MAX_NODES = 65_536
MAX_DEPTH = 16
MAX_INTEGER = 2**53 - 1
JsonObject = dict[str, Any]
ResourceKind = Literal[
    "workspace",
    "project",
    "context",
    "predictor",
    "source",
    "model",
    "operator",
    "api-key",
    "secret",
]
RESOURCE_KINDS = frozenset(ResourceKind.__args__)


def json_object(value: Any) -> JsonObject:
    """Copies bounded portable JSON, rejecting lossy/coercible Python values."""
    return json.loads(json_bytes(value))


def json_bytes(value: Any) -> bytes:
    """Encodes bounded portable JSON, rejecting lossy/coercible Python values.

    The bytes are what `json_object` measures, so native validation of them
    holds a payload to the same bounds.
    """
    count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal count
        count += 1
        require(depth <= MAX_DEPTH and count <= MAX_NODES, "JSON structure too large")
        if type(item) is dict:
            for key, child in item.items():
                require(type(key) is str, "JSON object keys must be strings")
                visit(child, depth + 1)
        elif type(item) is list:
            for child in item:
                visit(child, depth + 1)
        elif type(item) is int:
            require(abs(item) <= MAX_INTEGER, "integer exceeds portable JSON range")
        elif type(item) is float:
            require(math.isfinite(item), "JSON numbers must be finite")
        else:
            require(item is None or type(item) in (str, bool), "expected JSON value")

    require(type(value) is dict, "expected JSON object")
    visit(value, 0)
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, UnicodeError) as error:
        raise ServiceError("ValidationError", "invalid JSON encoding", 422) from error
    require(len(encoded) <= MAX_BYTES, "JSON body too large")
    return encoded


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Rejects ambiguous JSON instead of silently accepting the final key."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def parse_body(body: bytes) -> dict[str, Any]:
    """Parses bounded unambiguous JSON without retaining caller bytes."""
    require(0 < len(body) <= MAX_BYTES, "bounded JSON body required")
    try:
        value = json.loads(body, object_pairs_hook=unique_object)
        require(type(value) is dict, "expected JSON object")
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise ServiceError("ValidationError", "invalid JSON body", 422) from None


@dataclass(frozen=True)
class Scope:
    """Request correlation and project selection, never an authorization proof."""

    project: str
    request_id: str

    def __post_init__(self) -> None:
        """Validates in-process calls as well as wire parsing."""
        name(self.project)
        text(self.request_id, "request_id")


@dataclass(frozen=True)
class Resource:
    """Project-qualified identity; model repositories remain in model payloads."""

    kind: ResourceKind
    name: str

    def __post_init__(self) -> None:
        """Rejects unknown nouns rather than routing to a retired resource."""
        require(
            isinstance(self.kind, str) and self.kind in RESOURCE_KINDS,
            "invalid resource kind",
        )
        # A model is named by its catalog name, which may hold dots; every
        # chosen resource name keeps the strict rule.
        (model_name if self.kind == "model" else name)(self.name)


@dataclass(frozen=True)
class Request:
    """Versioned request whose payload still needs owner-specific validation."""

    scope: Scope
    resource: Resource
    payload: JsonObject

    @classmethod
    def parse(cls, value: Any) -> "Request":
        """Validates the complete envelope before domain operations can run."""
        value = json_object(value)
        fields(value, {"api_version", "project", "request_id", "resource", "payload"})
        require(value["api_version"] == VERSION, "unsupported api_version")
        resource = fields(value["resource"], {"kind", "name"})
        return cls(
            Scope(value["project"], value["request_id"]),
            Resource(**resource),
            json_object(value["payload"]),
        )

    def to_dict(self) -> JsonObject:
        """Serializes the same envelope for web, Rust and Python clients."""
        return json_object(
            {
                "api_version": VERSION,
                **asdict(self.scope),
                "resource": asdict(self.resource),
                "payload": self.payload,
            }
        )


class ResultEnvelope(TypedDict):
    """Successful domain result; acceptance is not implicitly readiness."""

    api_version: str
    project: str
    request_id: str
    resource: JsonObject
    result: JsonObject


class ErrorEnvelope(TypedDict):
    """Same safe error leaf as the alpha, without any partial score/result."""

    api_version: str
    project: str
    request_id: str
    resource: JsonObject
    error: dict[str, Any]


def success(request: Request, result: JsonObject) -> ResultEnvelope:
    """Builds a bounded wire response without retaining the request payload."""
    response: ResultEnvelope = {
        "api_version": VERSION,
        "project": request.scope.project,
        "request_id": request.scope.request_id,
        "resource": asdict(request.resource),
        "result": json_object(result),
    }
    json_object(dict(response))
    return response


def failure(request: Request, error: ServiceError) -> ErrorEnvelope:
    """Wraps only an explicitly safe ServiceError, never arbitrary exceptions.

    Every envelope carries the OpenAPI `ErrorCode`; a refused browser session
    adds the `reason` the console's sign-in and membership flows branch on.
    """
    return {
        "api_version": VERSION,
        "project": request.scope.project,
        "request_id": request.scope.request_id,
        "resource": asdict(request.resource),
        "error": error.error(),
    }


@dataclass(frozen=True)
class Capability:
    """An honest gate, not proof that an entire journey has been completed."""

    name: str
    state: Literal["supported", "unsupported", "unconfigured"]
    code: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Requires actionable reasons for unavailable capabilities."""
        name(self.name)
        require(
            self.state in ("supported", "unsupported", "unconfigured"),
            "invalid capability state",
        )
        if self.state == "supported":
            require(
                self.code is None and self.reason is None,
                "supported capability has no gate",
            )
        else:
            text(self.code, "capability code")
            text(self.reason, "capability reason")

    def enforce(self) -> None:
        """Raises the same typed unavailable error in local and HTTP journeys."""
        if self.state != "supported":
            raise ServiceError(
                self.code or "Unsupported", self.reason or "unavailable", 501
            )


@dataclass(frozen=True)
class Artifact:
    """Content identity and license assertion supplied by a verified adapter."""

    repository: str
    revision: str
    digest: str
    license: str

    def __post_init__(self) -> None:
        """Validates identity syntax, not content integrity or license approval."""
        text(self.repository, "repository")
        text(self.revision, "revision")
        text(self.license, "license")
        require(
            isinstance(self.digest, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest) is not None,
            "digest must be sha256:<64 lowercase hex>",
        )
