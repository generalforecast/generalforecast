"""The HTTP API's request and reply types, generated from api/openapi.yaml.

Do not edit: run `python api/generate.py`. Every schema is a frozen dataclass
(or a type alias for scalars, arrays and mappings) with `from_payload`, which
checks the JSON shape and names the offending field in `PayloadError.param`,
and `to_payload`, which writes the JSON back without the optional fields left
as None. `ROUTES` lists every route with its request and reply types, and
the media type of its request body.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal

VERSION = "gf/v1"


class PayloadError(ValueError):
    """A payload that does not match the spec; `param` names the field."""

    def __init__(self, message: str, param: str) -> None:
        """Keep the message and the dotted path of the field."""
        super().__init__(f"{param}: {message}" if param else message)
        self.message, self.param = message, param


Decoder = Callable[[Any, str], Any]


def _at(param: str, key: str) -> str:
    return f"{param}.{key}" if param else key


def _text(value: Any, param: str) -> str:
    if type(value) is not str:
        raise PayloadError("expected a string", param)
    return value


def _integer(value: Any, param: str) -> int:
    if type(value) is not int:
        raise PayloadError("expected an integer", param)
    return value


def _number(value: Any, param: str) -> float:
    if type(value) not in (int, float):
        raise PayloadError("expected a number", param)
    return value


def _boolean(value: Any, param: str) -> bool:
    if type(value) is not bool:
        raise PayloadError("expected a boolean", param)
    return value


def _null(value: Any, param: str) -> None:
    if value is not None:
        raise PayloadError("expected null", param)


def _any(value: Any, param: str) -> Any:
    return value


def _pattern(expression: str) -> Decoder:
    compiled = re.compile(expression)

    def decode(value: Any, param: str) -> str:
        text = _text(value, param)
        if compiled.search(text) is None:
            raise PayloadError(f"expected a string matching {expression}", param)
        return text

    return decode


def _literal(*options: Any) -> Decoder:
    def decode(value: Any, param: str) -> Any:
        if type(value) is bool and bool not in {type(option) for option in options}:
            raise PayloadError(f"expected one of {list(options)}", param)
        if value not in options:
            raise PayloadError(f"expected one of {list(options)}", param)
        return value

    return decode


def _items(item: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> tuple:
        if type(value) is not list:
            raise PayloadError("expected an array", param)
        return tuple(item(element, f"{param}[{i}]") for i, element in enumerate(value))

    return decode


def _mapping(item: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> dict:
        if type(value) is not dict:
            raise PayloadError("expected an object", param)
        return {key: item(element, _at(param, key)) for key, element in value.items()}

    return decode


def _union(*options: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> Any:
        errors = []
        for option in options:
            try:
                return option(value, param)
            except PayloadError as error:
                errors.append(error.message)
        raise PayloadError(" or ".join(errors), param)

    return decode


def _record(cls: type[Payload]) -> Decoder:
    return cls.from_payload


def _enum(cls: type[StrEnum]) -> Decoder:
    def decode(value: Any, param: str) -> StrEnum:
        try:
            return cls(_text(value, param))
        except ValueError:
            raise PayloadError(
                f"expected one of {[m.value for m in cls]}", param
            ) from None

    return decode


def _encode(value: Any) -> Any:
    if isinstance(value, Payload):
        return value.to_payload()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Field:
    """How one JSON key of a record maps to a dataclass field."""

    key: str
    name: str
    required: bool
    decode: Decoder


@dataclass(frozen=True, kw_only=True)
class Payload:
    """A record schema: JSON in through `from_payload`, out through `to_payload`."""

    FIELDS: ClassVar[tuple[Field, ...]] = ()
    DEFAULTS: ClassVar[dict[str, Any]] = {}
    EXTRA: ClassVar[bool] = False

    @classmethod
    def from_payload(cls, value: Any, param: str = ""):
        """Decode a JSON object, naming the first field that does not fit."""
        if type(value) is not dict:
            raise PayloadError("expected an object", param)
        known = {spec.key for spec in cls.FIELDS}
        kwargs: dict[str, Any] = {}
        for spec in cls.FIELDS:
            if spec.key in value:
                kwargs[spec.name] = spec.decode(value[spec.key], _at(param, spec.key))
            elif spec.required:
                raise PayloadError("required", _at(param, spec.key))
        unknown = [key for key in value if key not in known]
        if cls.EXTRA:
            kwargs["extra"] = {key: value[key] for key in unknown}
        elif unknown:
            raise PayloadError("unknown field", _at(param, unknown[0]))
        return cls(**kwargs)

    def to_payload(self) -> dict[str, Any]:
        """The JSON object, without optional fields left as None."""
        payload: dict[str, Any] = {}
        for spec in self.FIELDS:
            value = getattr(self, spec.name)
            if value is None and not spec.required:
                continue
            payload[spec.key] = _encode(value)
        if self.EXTRA:
            payload.update(_encode(self.__dict__["extra"]))
        return payload


@dataclass(frozen=True)
class Route:
    """A route: its request and reply types, and its request body's media type.

    `request` is None for a GET, which has no body, and for a body that is not
    JSON, such as an Arrow stream, which `body` names.
    """

    name: str
    method: str
    path: str
    request: type[Payload] | None
    reply: type[Payload]
    body: str | None


Name = str
_decode_Name: Decoder = _pattern("^[a-z][a-z0-9-]{0,62}$")


ModelId = str
_decode_ModelId: Decoder = _pattern(
    "^general-forecast/[a-z0-9][a-z0-9.-]*(@[^@\\s]+)?$",
)


Digest = str
_decode_Digest: Decoder = _pattern("^sha256:[0-9a-f]{64}$")


Task = Literal["classification", "regression", "forecast"]
_decode_Task: Decoder = _literal("classification", "regression", "forecast")


ResourceKind = Literal[
    "predictor",
    "model",
    "project",
    "api-key",
    "secret",
    "context",
    "source",
    "workspace",
]
_decode_ResourceKind: Decoder = _literal(
    "predictor",
    "model",
    "project",
    "api-key",
    "secret",
    "context",
    "source",
    "workspace",
)


@dataclass(frozen=True, kw_only=True)
class Resource(Payload):
    """What the envelope is about; GET lists use the collection name."""

    kind: ResourceKind
    name: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _decode_ResourceKind),
        Field("name", "name", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class Envelope(Payload):
    """The fields every request and reply carries."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
    )


class ErrorCode(StrEnum):
    """The ErrorCode values.

    ValidationError 422, NotFound 404, Unauthorized 401, CapacityExceeded 422,
    ModelNotReady 503, Internal 500, Conflict 409 (a compare-and-swap on a stale
    generation), Unsupported 501 (an operation this server does not offer, such as a
    hosted route on gf serve)
    """

    ValidationError = "ValidationError"
    NotFound = "NotFound"
    Unauthorized = "Unauthorized"
    CapacityExceeded = "CapacityExceeded"
    ModelNotReady = "ModelNotReady"
    Internal = "Internal"
    Conflict = "Conflict"
    Unsupported = "Unsupported"


@dataclass(frozen=True, kw_only=True)
class Error(Payload):
    """The Error schema.

    `param` names the payload field (`target`, `inputs.plan`, `queries`); every limit
    error carries `limit` and `actual` and names the model in `message`. The request id
    is in the envelope.
    """

    code: ErrorCode
    message: str
    reason: str | None = None
    param: str | None = None
    hint: str | None = None
    limit: int | None = None
    actual: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("code", "code", True, _enum(ErrorCode)),
        Field("message", "message", True, _text),
        Field("reason", "reason", False, _pattern("^[a-z][a-z0-9_]{0,63}$")),
        Field("param", "param", False, _text),
        Field("hint", "hint", False, _text),
        Field("limit", "limit", False, _integer),
        Field("actual", "actual", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class ErrorReply(Payload):
    """The ErrorReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    error: Error

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("error", "error", True, _record(Error)),
    )


Row = dict[str, str | float | bool | None]
_decode_Row: Decoder = _mapping(_union(_text, _number, _boolean, _null))


Kind = Literal["int", "float", "0/1", "category", "text", "timestamp", "bool"]
_decode_Kind: Decoder = _literal(
    "int",
    "float",
    "0/1",
    "category",
    "text",
    "timestamp",
    "bool",
)


@dataclass(frozen=True, kw_only=True)
class Inputs(Payload):
    """The Inputs schema.

    inputs by rule rather than by list: `exclude` removes columns from the inferred
    set, `types` overrides inferred types; forecast predictors use `past` and `future`
    instead
    """

    exclude: tuple[str, ...] | None = None
    types: dict[str, Kind] | None = None
    past: tuple[str, ...] | None = None
    future: tuple[str, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("exclude", "exclude", False, _items(_text)),
        Field("types", "types", False, _mapping(_decode_Kind)),
        Field("past", "past", False, _items(_text)),
        Field("future", "future", False, _items(_text)),
    )


ColumnType = Literal["number", "category", "text", "boolean", "date", "timestamp"]
_decode_ColumnType: Decoder = _literal(
    "number",
    "category",
    "text",
    "boolean",
    "date",
    "timestamp",
)


@dataclass(frozen=True, kw_only=True)
class Selection(Payload):
    """The Selection schema.

    which context rows a query sees; fractions below 1 are of the model's row cap,
    integers are rows; `core: null` is the whole table
    """

    core: float | None = None
    neighbors: float | None = None
    kind: Literal["stratified", "random"] | None = None
    recent: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("core", "core", False, _union(_number, _null)),
        Field("neighbors", "neighbors", False, _number),
        Field("kind", "kind", False, _literal("stratified", "random")),
        Field("recent", "recent", False, _union(_integer, _null)),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "core": 0.25,
        "neighbors": 0.75,
        "kind": "stratified",
    }


@dataclass(frozen=True, kw_only=True)
class Manifest(Payload):
    """The Manifest schema.

    the ad hoc predictor: the gf.yaml fields, with `context` and `queries` as rows
    instead of a context declaration; `name` and `resources` are file-only
    """

    model: ModelId
    target: str
    context: tuple[Row, ...]
    queries: tuple[Row, ...]
    id: str | None = None
    inputs: tuple[str, ...] | Inputs | None = None
    time: str | None = None
    series: tuple[str, ...] | None = None
    related: Literal["all"] | tuple[str, ...] | None = None
    frequency: str | float | None = None
    horizon: int | None = None
    quantiles: tuple[float, ...] | None = None
    outputs: (
        tuple[Literal["label", "value", "probabilities", "quantiles"], ...] | None
    ) = None
    selection: Selection | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelId),
        Field("target", "target", True, _text),
        Field("id", "id", False, _text),
        Field("inputs", "inputs", False, _union(_items(_text), _record(Inputs))),
        Field("time", "time", False, _text),
        Field("series", "series", False, _items(_text)),
        Field("related", "related", False, _union(_literal("all"), _items(_text))),
        Field(
            "frequency",
            "frequency",
            False,
            _union(_pattern("^[1-9][0-9]*(s|min|h|d|w)$"), _number),
        ),
        Field("horizon", "horizon", False, _integer),
        Field("quantiles", "quantiles", False, _items(_number)),
        Field(
            "outputs",
            "outputs",
            False,
            _items(_literal("label", "value", "probabilities", "quantiles")),
        ),
        Field("selection", "selection", False, _record(Selection)),
        Field("context", "context", True, _items(_decode_Row)),
        Field("queries", "queries", True, _items(_decode_Row)),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {"quantiles": [0.1, 0.5, 0.9]}


@dataclass(frozen=True, kw_only=True)
class ManifestRequest(Payload):
    """The ManifestRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Manifest

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Manifest)),
    )


AsOfRequest = str | float
_decode_AsOfRequest: Decoder = _union(_text, _number)


@dataclass(frozen=True, kw_only=True)
class Queries(Payload):
    """The Queries schema.

    the declared predictor payload; `context` only when the manifest declares `context:
    {request: ...}`
    """

    queries: tuple[Row, ...]
    context: tuple[Row, ...] | None = None
    as_of: AsOfRequest | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("queries", "queries", True, _items(_decode_Row)),
        Field("context", "context", False, _items(_decode_Row)),
        Field("as_of", "as_of", False, _decode_AsOfRequest),
    )


@dataclass(frozen=True, kw_only=True)
class QueriesRequest(Payload):
    """The QueriesRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Queries

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Queries)),
    )


ContextKind = Literal["snapshot", "request", "live"]
_decode_ContextKind: Decoder = _literal("snapshot", "request", "live")


AsOf = str | None
_decode_AsOf: Decoder = _union(_pattern("Z$"), _null)


@dataclass(frozen=True, kw_only=True)
class ContextInfo(Payload):
    """The ContextInfo schema.

    the context a prediction actually used, the same shape from gf serve and the hosted
    platform. `source` is the context's kind; `name` names a live context; `version` is
    a live context's accepted sequence; `digest` is a snapshot's content digest; `rows`
    counts the rows read; `as_of` is the instant a live context was read as of. Each
    field a source does not have is null.
    """

    source: ContextKind
    name: Name | None
    version: int | None
    digest: Digest | None
    rows: int
    as_of: AsOf

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("source", "source", True, _decode_ContextKind),
        Field("name", "name", True, _union(_decode_Name, _null)),
        Field("version", "version", True, _union(_integer, _null)),
        Field("digest", "digest", True, _union(_decode_Digest, _null)),
        Field("rows", "rows", True, _integer),
        Field("as_of", "as_of", True, _decode_AsOf),
    )


@dataclass(frozen=True, kw_only=True)
class Prediction(Payload):
    """The Prediction schema.

    the query's id under the manifest's id column name when it declares one (omitted
    when it does not); `label` and `probabilities` for classification, `value` for
    regression, `value` and `quantiles` for forecasts
    """

    label: str | int | bool | None = None
    value: float | None = None
    probabilities: dict[str, float] | None = None
    quantiles: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("label", "label", False, _union(_text, _integer, _boolean)),
        Field("value", "value", False, _number),
        Field("probabilities", "probabilities", False, _mapping(_number)),
        Field("quantiles", "quantiles", False, _mapping(_number)),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class Result(Payload):
    """The Result schema."""

    model: ModelId
    task: Task
    context: ContextInfo
    predictions: tuple[Prediction, ...]
    predictor: Name | None = None
    revision: int | None = None
    environment: Name | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelId),
        Field("task", "task", True, _decode_Task),
        Field("context", "context", True, _record(ContextInfo)),
        Field("predictor", "predictor", False, _union(_decode_Name, _null)),
        Field("revision", "revision", False, _union(_integer, _null)),
        Field("environment", "environment", False, _union(_decode_Name, _null)),
        Field("predictions", "predictions", True, _items(_record(Prediction))),
    )


@dataclass(frozen=True, kw_only=True)
class ResultReply(Payload):
    """The ResultReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Result

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Result)),
    )


@dataclass(frozen=True, kw_only=True)
class Limits(Payload):
    """The most a model accepts in one call."""

    context_rows: int
    queries: int
    inputs: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context_rows", "context_rows", True, _integer),
        Field("queries", "queries", True, _integer),
        Field("inputs", "inputs", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class Shape(Payload):
    """A compiled program's shape."""

    context_rows: int
    queries: int
    inputs: int
    classes: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context_rows", "context_rows", True, _integer),
        Field("queries", "queries", True, _integer),
        Field("inputs", "inputs", True, _integer),
        Field("classes", "classes", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class LockInput(Payload):
    """The LockInput schema.

    an input column as gf.lock pins it: its kind (null when a request or live context's
    rows were not seen) and its role, `input` for a tabular model, `past` or `future`
    for a forecaster's covariates
    """

    name: str
    type: Kind | None
    role: Literal["input", "past", "future"]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field("type", "type", True, _union(_decode_Kind, _null)),
        Field("role", "role", True, _literal("input", "past", "future")),
    )


@dataclass(frozen=True, kw_only=True)
class LockHoldout(Payload):
    """A seeded random fraction of the labeled rows are queries."""

    fraction: float
    seed: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("fraction", "fraction", True, _number),
        Field("seed", "seed", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class ContextLock(Payload):
    """The ContextLock schema.

    the context as gf.lock pins it: a snapshot's file, its context rows' digest and
    count and the split that made its queries; a live context's name; a request
    context's most rows per call
    """

    kind: ContextKind
    rows: int | None = None
    query_rows: int | None = None
    digest: Digest | None = None
    path: str | None = None
    name: Name | None = None
    max_rows: int | None = None
    until: str | None = None
    holdout: LockHoldout | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _decode_ContextKind),
        Field("rows", "rows", False, _integer),
        Field("query_rows", "query_rows", False, _integer),
        Field("digest", "digest", False, _decode_Digest),
        Field("path", "path", False, _text),
        Field("name", "name", False, _decode_Name),
        Field("max_rows", "max_rows", False, _integer),
        Field("until", "until", False, _text),
        Field("holdout", "holdout", False, _record(LockHoldout)),
    )


@dataclass(frozen=True, kw_only=True)
class ResolvedSelection(Payload):
    """The ResolvedSelection schema.

    the selection as gf.lock pins it, in rows: `exact` reads the whole context,
    `retrieved` a core and each query's nearest rows, and `adaptive` resolves per call
    for a context only calls carry
    """

    strategy: Literal["exact", "adaptive", "retrieved"]
    core: int | None
    neighbors: int
    kind: Literal["stratified", "random", "balanced", "diverse"]
    recent: int | None
    seed: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("strategy", "strategy", True, _literal("exact", "adaptive", "retrieved")),
        Field("core", "core", True, _union(_integer, _null)),
        Field("neighbors", "neighbors", True, _integer),
        Field(
            "kind",
            "kind",
            True,
            _literal("stratified", "random", "balanced", "diverse"),
        ),
        Field("recent", "recent", True, _union(_integer, _null)),
        Field("seed", "seed", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class Resolution(Payload):
    """The resolved predictor gf.lock pins."""

    name: Name
    model: ModelId
    model_digest: Digest
    task: Task
    target: str
    target_type: Kind | None
    outputs: tuple[str, ...]
    inputs: tuple[LockInput, ...]
    context: ContextLock
    selection: ResolvedSelection
    classes: tuple[str | int | bool, ...] | None = None
    id: str | None = None
    id_type: Kind | None = None
    time: str | None = None
    time_type: Kind | None = None
    series: tuple[str, ...] | None = None
    series_types: tuple[Kind | None, ...] | None = None
    related: tuple[str, ...] | None = None
    frequency: str | float | None = None
    horizon: int | None = None
    quantiles: tuple[float, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("model", "model", True, _decode_ModelId),
        Field("model_digest", "model_digest", True, _decode_Digest),
        Field("task", "task", True, _decode_Task),
        Field("target", "target", True, _text),
        Field("target_type", "target_type", True, _union(_decode_Kind, _null)),
        Field("classes", "classes", False, _items(_union(_text, _integer, _boolean))),
        Field("id", "id", False, _text),
        Field("id_type", "id_type", False, _union(_decode_Kind, _null)),
        Field("time", "time", False, _text),
        Field("time_type", "time_type", False, _union(_decode_Kind, _null)),
        Field("series", "series", False, _items(_text)),
        Field(
            "series_types",
            "series_types",
            False,
            _items(_union(_decode_Kind, _null)),
        ),
        Field("related", "related", False, _items(_text)),
        Field("frequency", "frequency", False, _union(_text, _number)),
        Field("horizon", "horizon", False, _integer),
        Field("quantiles", "quantiles", False, _items(_number)),
        Field("outputs", "outputs", True, _items(_text)),
        Field("inputs", "inputs", True, _items(_record(LockInput))),
        Field("context", "context", True, _record(ContextLock)),
        Field("selection", "selection", True, _record(ResolvedSelection)),
    )


@dataclass(frozen=True, kw_only=True)
class Serving(Payload):
    """Whether this server has the predictor's model loaded and compiled."""

    loaded: bool
    compiled: bool
    warm_shapes: tuple[Shape, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("loaded", "loaded", True, _boolean),
        Field("compiled", "compiled", True, _boolean),
        Field("warm_shapes", "warm_shapes", True, _items(_record(Shape))),
    )


State = Literal["pending", "building", "ready", "failed", "stopped"]
_decode_State: Decoder = _literal("pending", "building", "ready", "failed", "stopped")


@dataclass(frozen=True, kw_only=True)
class Environment(Payload):
    """A pointer to a revision; `generation` guards promote and rollback."""

    predictor: Name
    name: Name
    revision: int | None
    generation: int
    status: State

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("predictor", "predictor", True, _decode_Name),
        Field("name", "name", True, _decode_Name),
        Field("revision", "revision", True, _union(_integer, _null)),
        Field("generation", "generation", True, _integer),
        Field("status", "status", True, _decode_State),
    )


@dataclass(frozen=True, kw_only=True)
class Predictor(Payload):
    """A predictor by name, with its environments.

    Its resolution (the fields `Resolution` names, `limits` and `serving`) is there
    once it resolves: always for a predictor gf serve declares, and for a hosted one
    once it has a revision, from its latest. `created_at` and `latest_revision` come
    from the hosted journal; a declared predictor has neither.
    """

    name: Name
    environments: tuple[Environment, ...]
    created_at: float | None = None
    latest_revision: int | None = None
    lock_digest: Digest | None = None
    model: ModelId | None = None
    model_digest: Digest | None = None
    task: Task | None = None
    target: str | None = None
    target_type: Kind | None = None
    classes: tuple[str | int | bool, ...] | None = None
    id: str | None = None
    time: str | None = None
    series: tuple[str, ...] | None = None
    inputs: tuple[LockInput, ...] | None = None
    context: ContextLock | None = None
    selection: ResolvedSelection | None = None
    limits: Limits | None = None
    serving: Serving | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("created_at", "created_at", False, _number),
        Field("latest_revision", "latest_revision", False, _union(_integer, _null)),
        Field("environments", "environments", True, _items(_record(Environment))),
        Field("lock_digest", "lock_digest", False, _decode_Digest),
        Field("model", "model", False, _decode_ModelId),
        Field("model_digest", "model_digest", False, _decode_Digest),
        Field("task", "task", False, _decode_Task),
        Field("target", "target", False, _text),
        Field("target_type", "target_type", False, _union(_decode_Kind, _null)),
        Field("classes", "classes", False, _items(_union(_text, _integer, _boolean))),
        Field("id", "id", False, _text),
        Field("time", "time", False, _text),
        Field("series", "series", False, _items(_text)),
        Field("inputs", "inputs", False, _items(_record(LockInput))),
        Field("context", "context", False, _record(ContextLock)),
        Field("selection", "selection", False, _record(ResolvedSelection)),
        Field("limits", "limits", False, _record(Limits)),
        Field("serving", "serving", False, _record(Serving)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorReply(Payload):
    """The PredictorReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Predictor

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Predictor)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorList(Payload):
    """The PredictorList schema."""

    predictors: tuple[Predictor, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("predictors", "predictors", True, _items(_record(Predictor))),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorListReply(Payload):
    """The PredictorListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: PredictorList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(PredictorList)),
    )


@dataclass(frozen=True, kw_only=True)
class Artifact(Payload):
    """The Artifact schema."""

    repository: str
    revision: str
    digest: Digest
    license: str
    files: dict[str, str]
    digest_format: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("repository", "repository", True, _text),
        Field("revision", "revision", True, _text),
        Field("digest", "digest", True, _decode_Digest),
        Field("license", "license", True, _text),
        Field("digest_format", "digest_format", False, _text),
        Field("files", "files", True, _mapping(_text)),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class Capability(Payload):
    """The Capability schema."""

    name: str
    state: Literal["supported", "unsupported", "unconfigured"]
    code: str | None = None
    reason: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field(
            "state",
            "state",
            True,
            _literal("supported", "unsupported", "unconfigured"),
        ),
        Field("code", "code", False, _union(_text, _null)),
        Field("reason", "reason", False, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Model(Payload):
    """The Model schema.

    the catalog's description of a served model: `model` is the resolved id with its
    content digest, `limits` what one call may send here, `warm_shapes` the shapes
    compiled at startup
    """

    model: str
    name: ModelId
    task: Task
    repository: str
    revision: str
    license: str
    runtime: Literal["cpu", "gpu"]
    artifact: Artifact
    bounds: dict[str, int | float | None]
    capabilities: tuple[Capability, ...]
    limits: Limits
    warm_shapes: tuple[Shape, ...]
    artifact_verification: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _text),
        Field("name", "name", True, _decode_ModelId),
        Field("task", "task", True, _decode_Task),
        Field("repository", "repository", True, _text),
        Field("revision", "revision", True, _text),
        Field("license", "license", True, _text),
        Field("runtime", "runtime", True, _literal("cpu", "gpu")),
        Field("artifact", "artifact", True, _record(Artifact)),
        Field("artifact_verification", "artifact_verification", False, _text),
        Field("bounds", "bounds", True, _mapping(_union(_integer, _number, _null))),
        Field("capabilities", "capabilities", True, _items(_record(Capability))),
        Field("limits", "limits", True, _record(Limits)),
        Field("warm_shapes", "warm_shapes", True, _items(_record(Shape))),
    )


@dataclass(frozen=True, kw_only=True)
class ModelList(Payload):
    """The ModelList schema."""

    models: tuple[Model, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("models", "models", True, _items(_record(Model))),
    )


@dataclass(frozen=True, kw_only=True)
class ModelReply(Payload):
    """The ModelReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Model

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Model)),
    )


@dataclass(frozen=True, kw_only=True)
class ModelListReply(Payload):
    """The ModelListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ModelList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ModelList)),
    )


@dataclass(frozen=True, kw_only=True)
class Health(Payload):
    """The Health schema."""

    status: Literal["ok"] = "ok"

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("status", "status", True, _literal("ok")),
    )


@dataclass(frozen=True, kw_only=True)
class HealthError(Payload):
    """The HealthError schema."""

    error: Error

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("error", "error", True, _record(Error)),
    )


@dataclass(frozen=True, kw_only=True)
class Empty(Payload):
    """The Empty schema."""

    FIELDS: ClassVar[tuple[Field, ...]] = ()


@dataclass(frozen=True, kw_only=True)
class EmptyRequest(Payload):
    """The EmptyRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Empty

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Empty)),
    )


@dataclass(frozen=True, kw_only=True)
class Project(Payload):
    """The Project schema."""

    name: Name
    organization: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("organization", "organization", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectReply(Payload):
    """The ProjectReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Project

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Project)),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectList(Payload):
    """The ProjectList schema."""

    projects: tuple[Project, ...]
    organizations: tuple[Name, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("projects", "projects", True, _items(_record(Project))),
        Field("organizations", "organizations", True, _items(_decode_Name)),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectListReply(Payload):
    """The ProjectListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ProjectList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ProjectList)),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectCreate(Payload):
    """The ProjectCreate schema."""

    name: Name
    organization: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("organization", "organization", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class ProjectCreateRequest(Payload):
    """The ProjectCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ProjectCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ProjectCreate)),
    )


Permission = Literal[
    "predictors:read",
    "predictors:write",
    "predictors:predict",
    "contexts:read",
    "contexts:write",
    "contexts:history",
    "sources:read",
    "sources:write",
    "models:read",
    "keys:read",
    "keys:write",
    "secrets:read",
    "secrets:write",
    "members:read",
    "members:write",
    "sso:read",
    "sso:write",
    "audit:read",
    "operators:read",
    "operators:write",
    "projects:read",
]
_decode_Permission: Decoder = _literal(
    "predictors:read",
    "predictors:write",
    "predictors:predict",
    "contexts:read",
    "contexts:write",
    "contexts:history",
    "sources:read",
    "sources:write",
    "models:read",
    "keys:read",
    "keys:write",
    "secrets:read",
    "secrets:write",
    "members:read",
    "members:write",
    "sso:read",
    "sso:write",
    "audit:read",
    "operators:read",
    "operators:write",
    "projects:read",
)


@dataclass(frozen=True, kw_only=True)
class Key(Payload):
    """`token` is present only in the create and rotate replies."""

    id: str
    project: Name
    name: Name
    permissions: tuple[Permission, ...]
    created_at: float
    expires_at: float
    revoked_at: float | None
    token: str | None = None
    replaces: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _pattern("^key-[0-9a-f]{20,24}$")),
        Field("project", "project", True, _decode_Name),
        Field("name", "name", True, _decode_Name),
        Field("permissions", "permissions", True, _items(_decode_Permission)),
        Field("created_at", "created_at", True, _number),
        Field("expires_at", "expires_at", True, _number),
        Field("revoked_at", "revoked_at", True, _union(_number, _null)),
        Field("token", "token", False, _pattern("^gf_[A-Za-z0-9_-]{32,}$")),
        Field("replaces", "replaces", False, _pattern("^key-[0-9a-f]{20,24}$")),
    )


@dataclass(frozen=True, kw_only=True)
class KeyReply(Payload):
    """The KeyReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Key

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Key)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyList(Payload):
    """The KeyList schema."""

    keys: tuple[Key, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("keys", "keys", True, _items(_record(Key))),
    )


@dataclass(frozen=True, kw_only=True)
class KeyListReply(Payload):
    """The KeyListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: KeyList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(KeyList)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyCreate(Payload):
    """The KeyCreate schema."""

    name: Name
    permissions: tuple[Permission, ...]
    expires_in: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("permissions", "permissions", True, _items(_decode_Permission)),
        Field("expires_in", "expires_in", False, _number),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {"expires_in": 2592000}


@dataclass(frozen=True, kw_only=True)
class KeyCreateRequest(Payload):
    """The KeyCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: KeyCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(KeyCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyRef(Payload):
    """The KeyRef schema."""

    id: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _pattern("^key-[0-9a-f]{20,24}$")),
    )


@dataclass(frozen=True, kw_only=True)
class KeyRefRequest(Payload):
    """The KeyRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: KeyRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(KeyRef)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyRevoked(Payload):
    """The KeyRevoked schema."""

    id: str
    revoked: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _text),
        Field("revoked", "revoked", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class KeyRevokedReply(Payload):
    """The KeyRevokedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: KeyRevoked

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(KeyRevoked)),
    )


@dataclass(frozen=True, kw_only=True)
class Secret(Payload):
    """Metadata only; a value is never returned."""

    project: Name
    name: Name
    version: int
    created_at: float
    updated_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("project", "project", True, _decode_Name),
        Field("name", "name", True, _decode_Name),
        Field("version", "version", True, _integer),
        Field("created_at", "created_at", True, _number),
        Field("updated_at", "updated_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class SecretReply(Payload):
    """The SecretReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Secret

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Secret)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretList(Payload):
    """The SecretList schema."""

    secrets: tuple[Secret, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("secrets", "secrets", True, _items(_record(Secret))),
    )


@dataclass(frozen=True, kw_only=True)
class SecretListReply(Payload):
    """The SecretListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SecretList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SecretList)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretCreate(Payload):
    """The SecretCreate schema."""

    name: Name
    value: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("value", "value", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class SecretCreateRequest(Payload):
    """The SecretCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SecretCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SecretCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretRotate(Payload):
    """The SecretRotate schema."""

    name: Name
    value: str
    expected_version: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("value", "value", True, _text),
        Field("expected_version", "expected_version", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class SecretRotateRequest(Payload):
    """The SecretRotateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SecretRotate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SecretRotate)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretRef(Payload):
    """The SecretRef schema."""

    name: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("name", "name", True, _decode_Name),)


@dataclass(frozen=True, kw_only=True)
class SecretRefRequest(Payload):
    """The SecretRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SecretRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SecretRef)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretDeleted(Payload):
    """The SecretDeleted schema."""

    name: Name
    deleted: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("deleted", "deleted", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class SecretDeletedReply(Payload):
    """The SecretDeletedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SecretDeleted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SecretDeleted)),
    )


@dataclass(frozen=True, kw_only=True)
class Revision(Payload):
    """An immutable pushed predictor."""

    predictor: Name
    number: int
    predictor_digest: Digest
    model: ModelId
    created_at: float
    status: State

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("predictor", "predictor", True, _decode_Name),
        Field("number", "number", True, _integer),
        Field("predictor_digest", "predictor_digest", True, _decode_Digest),
        Field("model", "model", True, _decode_ModelId),
        Field("created_at", "created_at", True, _number),
        Field("status", "status", True, _decode_State),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionReply(Payload):
    """The RevisionReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Revision

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Revision)),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionList(Payload):
    """The RevisionList schema."""

    revisions: tuple[Revision, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("revisions", "revisions", True, _items(_record(Revision))),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionListReply(Payload):
    """The RevisionListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: RevisionList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(RevisionList)),
    )


@dataclass(frozen=True, kw_only=True)
class LockedModel(Payload):
    """The LockedModel schema."""

    files: dict[str, str]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("files", "files", True, _mapping(_pattern("^[0-9a-f]{64}$"))),
    )


@dataclass(frozen=True, kw_only=True)
class LockResources(Payload):
    """The hosted serving resources gf.yaml declares."""

    accelerator: Literal["L4", "A10G", "L40S", "cpu"] | None = None
    replicas_min: int | None = None
    replicas_max: int | None = None
    idle_seconds: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "accelerator",
            "accelerator",
            False,
            _literal("L4", "A10G", "L40S", "cpu"),
        ),
        Field("replicas_min", "replicas_min", False, _integer),
        Field("replicas_max", "replicas_max", False, _integer),
        Field("idle_seconds", "idle_seconds", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class Lock(Payload):
    """Gf.lock, version 2: what gf check resolved gf.yaml into.

    `digest` is the canonical JSON digest of the document without `digest`, the one
    identity of a predictor revision; `manifest` the digest of the normalized gf.yaml
    it resolves, null for an upgraded version 1 lock
    """

    lock_version: Literal[2] = 2
    digest: Digest
    manifest: Digest | None
    predictor: Resolution
    model: LockedModel
    resources: LockResources | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("lock_version", "lock_version", True, _literal(2)),
        Field("digest", "digest", True, _decode_Digest),
        Field("manifest", "manifest", True, _union(_decode_Digest, _null)),
        Field("predictor", "predictor", True, _record(Resolution)),
        Field("resources", "resources", False, _record(LockResources)),
        Field("model", "model", True, _record(LockedModel)),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionRegister(Payload):
    """The RevisionRegister schema."""

    lock: Lock
    snapshot: Digest | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("lock", "lock", True, _record(Lock)),
        Field("snapshot", "snapshot", True, _union(_decode_Digest, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionRegisterRequest(Payload):
    """The RevisionRegisterRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: RevisionRegister

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(RevisionRegister)),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionDetails(Payload):
    """The RevisionDetails schema."""

    revision: Revision
    lock: Lock

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("revision", "revision", True, _record(Revision)),
        Field("lock", "lock", True, _record(Lock)),
    )


@dataclass(frozen=True, kw_only=True)
class RevisionDetailsReply(Payload):
    """The RevisionDetailsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: RevisionDetails

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(RevisionDetails)),
    )


@dataclass(frozen=True, kw_only=True)
class EnvironmentReply(Payload):
    """The EnvironmentReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Environment

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Environment)),
    )


@dataclass(frozen=True, kw_only=True)
class EnvironmentList(Payload):
    """The EnvironmentList schema."""

    environments: tuple[Environment, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("environments", "environments", True, _items(_record(Environment))),
    )


@dataclass(frozen=True, kw_only=True)
class EnvironmentListReply(Payload):
    """The EnvironmentListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: EnvironmentList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(EnvironmentList)),
    )


@dataclass(frozen=True, kw_only=True)
class EnvironmentCreate(Payload):
    """The EnvironmentCreate schema."""

    name: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("name", "name", True, _decode_Name),)


@dataclass(frozen=True, kw_only=True)
class EnvironmentCreateRequest(Payload):
    """The EnvironmentCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: EnvironmentCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(EnvironmentCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class Promote(Payload):
    """The Promote schema."""

    revision: int
    expected_generation: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("revision", "revision", True, _integer),
        Field("expected_generation", "expected_generation", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class PromoteRequest(Payload):
    """The PromoteRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Promote

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Promote)),
    )


@dataclass(frozen=True, kw_only=True)
class ServingContext(Payload):
    """The ServingContext schema.

    the context an environment serves: `source` is `snapshot`, `request` or
    `live/<context>`; the version fields and `freshness_lag_seconds` are set for live
    contexts
    """

    source: str
    version: int | str | None
    rows: int
    as_of: AsOf
    accepted_version: int | str | None
    serving_version: int | str | None
    freshness_lag_seconds: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("source", "source", True, _text),
        Field("version", "version", True, _union(_integer, _text, _null)),
        Field("rows", "rows", True, _integer),
        Field("as_of", "as_of", True, _decode_AsOf),
        Field(
            "accepted_version",
            "accepted_version",
            True,
            _union(_integer, _text, _null),
        ),
        Field(
            "serving_version",
            "serving_version",
            True,
            _union(_integer, _text, _null),
        ),
        Field(
            "freshness_lag_seconds",
            "freshness_lag_seconds",
            True,
            _union(_number, _null),
        ),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class Metrics(Payload):
    """The Metrics schema.

    serving facts: `context` is the context the environment is serving now,
    `accepted_version`/`serving_version` compare the newest accepted write of a live
    context with the one being served, `fresh` says whether they agree (null without a
    context)
    """

    status: State
    context: ServingContext | None
    accepted_version: int | str | None
    serving_version: int | str | None
    fresh: bool | None
    requests: int
    errors: int
    mean_latency_seconds: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("status", "status", True, _decode_State),
        Field("context", "context", True, _union(_record(ServingContext), _null)),
        Field(
            "accepted_version",
            "accepted_version",
            True,
            _union(_integer, _text, _null),
        ),
        Field(
            "serving_version",
            "serving_version",
            True,
            _union(_integer, _text, _null),
        ),
        Field("fresh", "fresh", True, _union(_boolean, _null)),
        Field("requests", "requests", True, _integer),
        Field("errors", "errors", True, _integer),
        Field(
            "mean_latency_seconds",
            "mean_latency_seconds",
            True,
            _union(_number, _null),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class MetricsReply(Payload):
    """The MetricsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Metrics

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Metrics)),
    )


@dataclass(frozen=True, kw_only=True)
class LogEntry(Payload):
    """The LogEntry schema."""

    at: float
    code: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("at", "at", True, _number),
        Field("code", "code", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class Logs(Payload):
    """The Logs schema."""

    logs: tuple[LogEntry, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("logs", "logs", True, _items(_record(LogEntry))),
    )


@dataclass(frozen=True, kw_only=True)
class LogsReply(Payload):
    """The LogsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Logs

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Logs)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorDeleted(Payload):
    """The PredictorDeleted schema."""

    name: Name
    deleted: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("deleted", "deleted", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorDeletedReply(Payload):
    """The PredictorDeletedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: PredictorDeleted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(PredictorDeleted)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContext(Payload):
    """A hosted, appendable context."""

    name: Name
    inputs: tuple[str, ...]
    targets: dict[str, ColumnType]
    max_rows: int
    rows: int
    labeled_rows: int
    sequence: int
    accepted_at: float | None
    id: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("inputs", "inputs", True, _items(_text)),
        Field("targets", "targets", True, _mapping(_decode_ColumnType)),
        Field("max_rows", "max_rows", True, _integer),
        Field("rows", "rows", True, _integer),
        Field("labeled_rows", "labeled_rows", True, _integer),
        Field("sequence", "sequence", True, _integer),
        Field("accepted_at", "accepted_at", True, _union(_number, _null)),
        Field("id", "id", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextReply(Payload):
    """The LiveContextReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: LiveContext

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(LiveContext)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextList(Payload):
    """The LiveContextList schema."""

    contexts: tuple[LiveContext, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("contexts", "contexts", True, _items(_record(LiveContext))),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextListReply(Payload):
    """The LiveContextListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: LiveContextList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(LiveContextList)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextCreate(Payload):
    """The LiveContextCreate schema."""

    name: Name
    inputs: tuple[str, ...]
    targets: dict[str, ColumnType]
    max_rows: int | None = None
    id: str | None = None
    types: dict[str, ColumnType] | None = None
    max_writes: int | None = None
    max_pending_labels: int | None = None
    pending_ttl_seconds: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("inputs", "inputs", True, _items(_text)),
        Field("targets", "targets", True, _mapping(_decode_ColumnType)),
        Field("max_rows", "max_rows", False, _integer),
        Field("id", "id", False, _text),
        Field("types", "types", False, _mapping(_decode_ColumnType)),
        Field("max_writes", "max_writes", False, _integer),
        Field("max_pending_labels", "max_pending_labels", False, _integer),
        Field("pending_ttl_seconds", "pending_ttl_seconds", False, _number),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "max_rows": 100000,
        "max_writes": 10000,
        "max_pending_labels": 0,
        "pending_ttl_seconds": 3600,
    }


@dataclass(frozen=True, kw_only=True)
class LiveContextCreateRequest(Payload):
    """The LiveContextCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: LiveContextCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(LiveContextCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextAppend(Payload):
    """The LiveContextAppend schema."""

    write_id: str
    rows: tuple[Row, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("write_id", "write_id", True, _text),
        Field("rows", "rows", True, _items(_decode_Row)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextAppendRequest(Payload):
    """The LiveContextAppendRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: LiveContextAppend

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(LiveContextAppend)),
    )


@dataclass(frozen=True, kw_only=True)
class Label(Payload):
    """The Label schema."""

    id: str | int
    value: str | float | bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _union(_text, _integer)),
        Field("value", "value", True, _union(_text, _number, _boolean)),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextLabel(Payload):
    """The LiveContextLabel schema."""

    write_id: str
    target: str
    labels: tuple[Label, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("write_id", "write_id", True, _text),
        Field("target", "target", True, _text),
        Field("labels", "labels", True, _items(_record(Label))),
    )


@dataclass(frozen=True, kw_only=True)
class LiveContextLabelRequest(Payload):
    """The LiveContextLabelRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: LiveContextLabel

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(LiveContextLabel)),
    )


@dataclass(frozen=True, kw_only=True)
class Receipt(Payload):
    """Durable acceptance of an append or label write."""

    context: Name
    write_id: str
    operation: Literal["append", "label"]
    rows: int
    sequence: int
    accepted_at: float
    durably_accepted: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context", "context", True, _decode_Name),
        Field("write_id", "write_id", True, _text),
        Field("operation", "operation", True, _literal("append", "label")),
        Field("rows", "rows", True, _integer),
        Field("sequence", "sequence", True, _integer),
        Field("accepted_at", "accepted_at", True, _number),
        Field("durably_accepted", "durably_accepted", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class ReceiptReply(Payload):
    """The ReceiptReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Receipt

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Receipt)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorCreate(Payload):
    """The PredictorCreate schema."""

    name: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("name", "name", True, _decode_Name),)


@dataclass(frozen=True, kw_only=True)
class PredictorCreateRequest(Payload):
    """The PredictorCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: PredictorCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(PredictorCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class PredictorCreatedReply(Payload):
    """The PredictorCreatedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: PredictorCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(PredictorCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextImport(Payload):
    """Where a live context's rows come from, and how its columns map."""

    context: Name
    source: str
    format: Literal["csv", "tsv", "parquet", "arrow", "json", "jsonl", "table"]
    credential_ref: str | None = None
    digest: Digest | None = None
    mapping: tuple[tuple[str, ...], ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context", "context", True, _decode_Name),
        Field("source", "source", True, _text),
        Field(
            "format",
            "format",
            True,
            _literal("csv", "tsv", "parquet", "arrow", "json", "jsonl", "table"),
        ),
        Field(
            "credential_ref",
            "credential_ref",
            False,
            _union(_pattern("^secret/[a-z][a-z0-9-]{0,62}$"), _null),
        ),
        Field("digest", "digest", False, _union(_decode_Digest, _null)),
        Field("mapping", "mapping", False, _items(_items(_text))),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {"mapping": []}


@dataclass(frozen=True, kw_only=True)
class ContextImportRequest(Payload):
    """The ContextImportRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ContextImport

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ContextImport)),
    )


WriteOperation = Literal[
    "create_features",
    "attach_labels",
    "correct_features",
    "correct_labels",
    "resupply_labels",
    "import_record",
    "checkpoint",
]
_decode_WriteOperation: Decoder = _literal(
    "create_features",
    "attach_labels",
    "correct_features",
    "correct_labels",
    "resupply_labels",
    "import_record",
    "checkpoint",
)


@dataclass(frozen=True, kw_only=True)
class Observation(Payload):
    """One observation's components; corrections name the versions they replace."""

    key: str
    features: dict[str, str | float | bool | None] | None = None
    labels: dict[str, str | float | bool] | None = None
    event_at: float | None = None
    expected_feature_version: int | None = None
    expected_label_versions: dict[str, int] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("key", "key", True, _text),
        Field(
            "features",
            "features",
            False,
            _mapping(_union(_text, _number, _boolean, _null)),
        ),
        Field("labels", "labels", False, _mapping(_union(_text, _number, _boolean))),
        Field("event_at", "event_at", False, _number),
        Field("expected_feature_version", "expected_feature_version", False, _integer),
        Field(
            "expected_label_versions",
            "expected_label_versions",
            False,
            _mapping(_integer),
        ),
    )


MutationOperation = Literal[
    "create_features",
    "attach_labels",
    "correct_features",
    "correct_labels",
    "resupply_labels",
]
_decode_MutationOperation: Decoder = _literal(
    "create_features",
    "attach_labels",
    "correct_features",
    "correct_labels",
    "resupply_labels",
)


@dataclass(frozen=True, kw_only=True)
class Mutation(Payload):
    """The Mutation schema."""

    write_id: str
    operation: MutationOperation
    records: tuple[Observation, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("write_id", "write_id", True, _text),
        Field("operation", "operation", True, _decode_MutationOperation),
        Field("records", "records", True, _items(_record(Observation))),
    )


@dataclass(frozen=True, kw_only=True)
class MutationRequest(Payload):
    """The MutationRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Mutation

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Mutation)),
    )


@dataclass(frozen=True, kw_only=True)
class ComponentVersions(Payload):
    """The ComponentVersions schema."""

    key: str
    feature_version: int | None
    label_versions: dict[str, int]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("key", "key", True, _text),
        Field("feature_version", "feature_version", True, _union(_integer, _null)),
        Field("label_versions", "label_versions", True, _mapping(_integer)),
    )


@dataclass(frozen=True, kw_only=True)
class AcceptedWrite(Payload):
    """A write's durable receipt, the same for every retry of its write id."""

    project: Name
    context: Name
    write_id: str
    operation: WriteOperation
    sequence: int
    accepted_at: float
    durably_accepted: bool
    changes: tuple[ComponentVersions, ...]
    retry_valid_until: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("project", "project", True, _decode_Name),
        Field("context", "context", True, _decode_Name),
        Field("write_id", "write_id", True, _text),
        Field("operation", "operation", True, _decode_WriteOperation),
        Field("sequence", "sequence", True, _integer),
        Field("accepted_at", "accepted_at", True, _number),
        Field("durably_accepted", "durably_accepted", True, _boolean),
        Field("changes", "changes", True, _items(_record(ComponentVersions))),
        Field("retry_valid_until", "retry_valid_until", True, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class AcceptedWriteReply(Payload):
    """The AcceptedWriteReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: AcceptedWrite

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(AcceptedWrite)),
    )


@dataclass(frozen=True, kw_only=True)
class ImportReceipt(Payload):
    """The ImportReceipt schema."""

    context: Name
    digest: Digest
    rows: int
    sequence: int
    accepted_at: float
    receipt: AcceptedWrite
    mode: Literal["explicit-refresh"] = "explicit-refresh"
    continuous: Literal[False] = False

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("context", "context", True, _decode_Name),
        Field("digest", "digest", True, _decode_Digest),
        Field("rows", "rows", True, _integer),
        Field("sequence", "sequence", True, _integer),
        Field("accepted_at", "accepted_at", True, _number),
        Field("receipt", "receipt", True, _record(AcceptedWrite)),
        Field("mode", "mode", True, _literal("explicit-refresh")),
        Field("continuous", "continuous", True, _literal(False)),
    )


@dataclass(frozen=True, kw_only=True)
class ImportReceiptReply(Payload):
    """The ImportReceiptReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ImportReceipt

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ImportReceipt)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextSnapshotQuery(Payload):
    """The ContextSnapshotQuery schema."""

    target: str
    knowledge_at: float | None = None
    event_before: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("target", "target", True, _text),
        Field("knowledge_at", "knowledge_at", False, _union(_number, _null)),
        Field("event_before", "event_before", False, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextSnapshotQueryRequest(Payload):
    """The ContextSnapshotQueryRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: ContextSnapshotQuery

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(ContextSnapshotQuery)),
    )


@dataclass(frozen=True, kw_only=True)
class SnapshotRow(Payload):
    """The SnapshotRow schema."""

    key: str
    features: dict[str, str | float | bool | None]
    label: str | float | bool
    feature_version: int
    label_version: int
    feature_accepted_at: float
    label_accepted_at: float
    event_at: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("key", "key", True, _text),
        Field(
            "features",
            "features",
            True,
            _mapping(_union(_text, _number, _boolean, _null)),
        ),
        Field("label", "label", True, _union(_text, _number, _boolean)),
        Field("feature_version", "feature_version", True, _integer),
        Field("label_version", "label_version", True, _integer),
        Field("feature_accepted_at", "feature_accepted_at", True, _number),
        Field("label_accepted_at", "label_accepted_at", True, _number),
        Field("event_at", "event_at", True, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class ContextSnapshot(Payload):
    """The ContextSnapshot schema."""

    project: Name
    context: Name
    target: str
    features: tuple[str, ...]
    through_sequence: int
    knowledge_at: float
    event_before: float | None
    rows: tuple[SnapshotRow, ...]
    source_wait_count: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("project", "project", True, _decode_Name),
        Field("context", "context", True, _decode_Name),
        Field("target", "target", True, _text),
        Field("features", "features", True, _items(_text)),
        Field("through_sequence", "through_sequence", True, _integer),
        Field("knowledge_at", "knowledge_at", True, _number),
        Field("event_before", "event_before", True, _union(_number, _null)),
        Field("rows", "rows", True, _items(_record(SnapshotRow))),
        Field("source_wait_count", "source_wait_count", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class ContextSnapshotReply(Payload):
    """The ContextSnapshotReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: ContextSnapshot

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(ContextSnapshot)),
    )


@dataclass(frozen=True, kw_only=True)
class SourceMapping(Payload):
    """The SourceMapping schema.

    which columns of a pinned file are an observation's key, features and labels; empty
    for JSONL already in that shape
    """

    key: str | None = None
    features: dict[str, str] | None = None
    labels: dict[str, str] | None = None
    event_at: str | None = None
    label_types: dict[str, Literal["boolean", "numeric"]] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("key", "key", False, _text),
        Field("features", "features", False, _mapping(_text)),
        Field("labels", "labels", False, _mapping(_text)),
        Field("event_at", "event_at", False, _text),
        Field(
            "label_types",
            "label_types",
            False,
            _mapping(_literal("boolean", "numeric")),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePinSpec(Payload):
    """The SourcePinSpec schema."""

    path: str
    format: Literal["jsonl", "csv"]
    mapping: SourceMapping

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("path", "path", True, _text),
        Field("format", "format", True, _literal("jsonl", "csv")),
        Field("mapping", "mapping", True, _record(SourceMapping)),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePinRequest(Payload):
    """The SourcePinRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SourcePinSpec

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SourcePinSpec)),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePin(Payload):
    """The SourcePin schema."""

    source_id: Name
    digest: Digest
    size_bytes: int
    format: Literal["jsonl", "csv"]
    mapping: SourceMapping

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("source_id", "source_id", True, _decode_Name),
        Field("digest", "digest", True, _decode_Digest),
        Field("size_bytes", "size_bytes", True, _integer),
        Field("format", "format", True, _literal("jsonl", "csv")),
        Field("mapping", "mapping", True, _record(SourceMapping)),
    )


@dataclass(frozen=True, kw_only=True)
class SourcePinReply(Payload):
    """The SourcePinReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SourcePin

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SourcePin)),
    )


@dataclass(frozen=True, kw_only=True)
class SourceList(Payload):
    """The SourceList schema."""

    sources: tuple[SourcePin, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("sources", "sources", True, _items(_record(SourcePin))),
    )


@dataclass(frozen=True, kw_only=True)
class SourceListReply(Payload):
    """The SourceListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: SourceList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(SourceList)),
    )


@dataclass(frozen=True, kw_only=True)
class UploadReceipt(Payload):
    """The UploadReceipt schema."""

    digest: Digest
    rows: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("digest", "digest", True, _decode_Digest),
        Field("rows", "rows", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class UploadReceiptReply(Payload):
    """The UploadReceiptReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: UploadReceipt

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(UploadReceipt)),
    )


@dataclass(frozen=True, kw_only=True)
class PermissionGrant(Payload):
    """A permission keys may hold; one that is not enforced authorizes nothing yet."""

    name: Permission
    description: str
    enforced: bool
    roles: tuple[Literal["admin", "editor", "viewer", "scorer"], ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Permission),
        Field("description", "description", True, _text),
        Field("enforced", "enforced", True, _boolean),
        Field(
            "roles",
            "roles",
            True,
            _items(_literal("admin", "editor", "viewer", "scorer")),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class KeyIdentity(Payload):
    """The KeyIdentity schema."""

    id: str
    name: Name
    project: Name
    expires_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _pattern("^key-[0-9a-f]{20,24}$")),
        Field("name", "name", True, _decode_Name),
        Field("project", "project", True, _decode_Name),
        Field("expires_at", "expires_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceIdentity(Payload):
    """The WorkspaceIdentity schema."""

    id: str
    display_name: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _text),
        Field("display_name", "display_name", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class Principal(Payload):
    """The credential's kind and id, with what the server knows of it."""

    kind: str
    id: str
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _text),
        Field("id", "id", True, _text),
    )
    EXTRA: ClassVar[bool] = True


@dataclass(frozen=True, kw_only=True)
class AuthStatus(Payload):
    """The AuthStatus schema."""

    authenticated: Literal[True] = True
    principal: Principal
    google: Literal["configured", "unconfigured"]
    sign_in: tuple[Literal["google", "microsoft", "github"], ...]
    secrets: Literal["configured", "unconfigured"]
    capabilities: tuple[Capability, ...]
    permission_inventory: tuple[PermissionGrant, ...]
    bootstrap: Literal["unconfigured", "available", "consumed"]
    key: KeyIdentity | None = None
    email: str | None = None
    machine: str | None = None
    workspace: WorkspaceIdentity | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("authenticated", "authenticated", True, _literal(True)),
        Field("principal", "principal", True, _record(Principal)),
        Field("google", "google", True, _literal("configured", "unconfigured")),
        Field(
            "sign_in",
            "sign_in",
            True,
            _items(_literal("google", "microsoft", "github")),
        ),
        Field("secrets", "secrets", True, _literal("configured", "unconfigured")),
        Field("capabilities", "capabilities", True, _items(_record(Capability))),
        Field(
            "permission_inventory",
            "permission_inventory",
            True,
            _items(_record(PermissionGrant)),
        ),
        Field(
            "bootstrap",
            "bootstrap",
            True,
            _literal("unconfigured", "available", "consumed"),
        ),
        Field("key", "key", False, _record(KeyIdentity)),
        Field("email", "email", False, _union(_text, _null)),
        Field("machine", "machine", False, _union(_text, _null)),
        Field(
            "workspace",
            "workspace",
            False,
            _union(_record(WorkspaceIdentity), _null),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class AuthStatusReply(Payload):
    """The AuthStatusReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: AuthStatus

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(AuthStatus)),
    )


@dataclass(frozen=True, kw_only=True)
class LoggedOut(Payload):
    """The LoggedOut schema."""

    revoked: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("revoked", "revoked", True, _pattern("^key-[0-9a-f]{20,24}$")),
    )


@dataclass(frozen=True, kw_only=True)
class LoggedOutReply(Payload):
    """The LoggedOutReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: LoggedOut

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(LoggedOut)),
    )


@dataclass(frozen=True, kw_only=True)
class Organization(Payload):
    """The Organization schema."""

    name: Name
    account: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("account", "account", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class OrganizationReply(Payload):
    """The OrganizationReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Organization

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Organization)),
    )


@dataclass(frozen=True, kw_only=True)
class OrganizationCreateRequest(Payload):
    """The OrganizationCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: Organization

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(Organization)),
    )


@dataclass(frozen=True, kw_only=True)
class OrganizationList(Payload):
    """The OrganizationList schema."""

    organizations: tuple[Organization, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("organizations", "organizations", True, _items(_record(Organization))),
    )


@dataclass(frozen=True, kw_only=True)
class OrganizationListReply(Payload):
    """The OrganizationListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: OrganizationList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(OrganizationList)),
    )


@dataclass(frozen=True, kw_only=True)
class Sso(Payload):
    """The Sso schema."""

    organization: Name
    provider: Literal["google"] = "google"
    providers: tuple[Literal["google", "microsoft", "github"], ...]
    enabled: bool
    configured: bool
    membership_required: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("organization", "organization", True, _decode_Name),
        Field("provider", "provider", True, _literal("google")),
        Field(
            "providers",
            "providers",
            True,
            _items(_literal("google", "microsoft", "github")),
        ),
        Field("enabled", "enabled", True, _boolean),
        Field("configured", "configured", True, _boolean),
        Field("membership_required", "membership_required", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class SsoReply(Payload):
    """The SsoReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Sso

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Sso)),
    )


@dataclass(frozen=True, kw_only=True)
class SsoConfigure(Payload):
    """The SsoConfigure schema."""

    enabled: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("enabled", "enabled", True, _boolean),)


@dataclass(frozen=True, kw_only=True)
class SsoConfigureRequest(Payload):
    """The SsoConfigureRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: SsoConfigure

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(SsoConfigure)),
    )


@dataclass(frozen=True, kw_only=True)
class AuditEvent(Payload):
    """The AuditEvent schema."""

    sequence: int
    at: float
    actor: str
    project: Name
    action: str
    target: str
    outcome: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("sequence", "sequence", True, _integer),
        Field("at", "at", True, _number),
        Field("actor", "actor", True, _text),
        Field("project", "project", True, _decode_Name),
        Field("action", "action", True, _text),
        Field("target", "target", True, _text),
        Field("outcome", "outcome", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class AuditLog(Payload):
    """The AuditLog schema."""

    events: tuple[AuditEvent, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("events", "events", True, _items(_record(AuditEvent))),
    )


@dataclass(frozen=True, kw_only=True)
class AuditLogReply(Payload):
    """The AuditLogReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: AuditLog

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(AuditLog)),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginStart(Payload):
    """The CliLoginStart schema."""

    machine: str

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("machine", "machine", True, _text),)


@dataclass(frozen=True, kw_only=True)
class CliLoginGrant(Payload):
    """The CliLoginGrant schema."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "device_code",
            "device_code",
            True,
            _pattern("^gf_dc_[A-Za-z0-9_-]{43}$"),
        ),
        Field("user_code", "user_code", True, _pattern("^[A-Z0-9]{4}-[A-Z0-9]{4}$")),
        Field("verification_uri", "verification_uri", True, _text),
        Field("verification_uri_complete", "verification_uri_complete", True, _text),
        Field("expires_in", "expires_in", True, _integer),
        Field("interval", "interval", True, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginPoll(Payload):
    """The CliLoginPoll schema."""

    device_code: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "device_code",
            "device_code",
            True,
            _pattern("^gf_dc_[A-Za-z0-9_-]{43}$"),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginToken(Payload):
    """The CliLoginToken schema."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    key: KeyIdentity
    email: str
    workspace: WorkspaceIdentity
    project: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("access_token", "access_token", True, _text),
        Field("token_type", "token_type", True, _literal("bearer")),
        Field("key", "key", True, _record(KeyIdentity)),
        Field("email", "email", True, _text),
        Field("workspace", "workspace", True, _record(WorkspaceIdentity)),
        Field("project", "project", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class LoginErrorReply(Payload):
    """Why a CLI login cannot go on, outside the envelope as the login is.

    `reason` says which: `authorization_pending` (`Unauthorized`, poll again),
    `slow_down` (`ValidationError`, poll less often), `expired_token` and
    `access_denied` (`Unauthorized`, start again) or `rate_limited`
    (`CapacityExceeded`).
    """

    error: Error

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("error", "error", True, _record(Error)),
    )


WorkspaceRole = Literal["admin", "editor", "viewer", "scorer"]
_decode_WorkspaceRole: Decoder = _literal("admin", "editor", "viewer", "scorer")


JoinRequestId = str
_decode_JoinRequestId: Decoder = _pattern("^jr-[0-9a-f]{16}$")


InvitationId = str
_decode_InvitationId: Decoder = _pattern("^inv-[0-9a-f]{16}$")


WaitlistId = str
_decode_WaitlistId: Decoder = _pattern("^wl-[0-9a-f]{16}$")


WorkspaceName = str
_decode_WorkspaceName: Decoder = _text


UserCode = str
_decode_UserCode: Decoder = _text


@dataclass(frozen=True, kw_only=True)
class WorkspaceSummary(Payload):
    """A workspace the caller belongs to: its role there and the projects it may use."""

    workspace: Name
    display_name: str
    role: WorkspaceRole | None
    admin: bool
    default_project: Name | None
    projects: tuple[Name, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field("role", "role", True, _union(_decode_WorkspaceRole, _null)),
        Field("admin", "admin", True, _boolean),
        Field("default_project", "default_project", True, _union(_decode_Name, _null)),
        Field("projects", "projects", True, _items(_decode_Name)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceDetails(Payload):
    """The WorkspaceDetails schema."""

    workspace: Name
    display_name: str
    role: WorkspaceRole | None
    admin: bool
    default_project: Name | None
    projects: tuple[Name, ...]
    organization: Name
    created_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field("role", "role", True, _union(_decode_WorkspaceRole, _null)),
        Field("admin", "admin", True, _boolean),
        Field("default_project", "default_project", True, _union(_decode_Name, _null)),
        Field("projects", "projects", True, _items(_decode_Name)),
        Field("organization", "organization", True, _decode_Name),
        Field("created_at", "created_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequest(Payload):
    """A request to join a workspace, as the person who asked sees it."""

    id: JoinRequestId
    workspace: Name
    display_name: str
    state: Literal["pending", "approved", "denied", "cancelled"]
    created_at: float
    decided_at: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_JoinRequestId),
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field(
            "state",
            "state",
            True,
            _literal("pending", "approved", "denied", "cancelled"),
        ),
        Field("created_at", "created_at", True, _number),
        Field("decided_at", "decided_at", True, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestAdmin(Payload):
    """A request to join a workspace, as its admins see it."""

    id: JoinRequestId
    workspace: Name
    subject: str
    email: str
    name: str | None
    domain: str
    state: Literal["pending", "approved", "denied", "cancelled"]
    role: WorkspaceRole | None
    created_at: float
    decided_at: float | None
    decided_by: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_JoinRequestId),
        Field("workspace", "workspace", True, _decode_Name),
        Field("subject", "subject", True, _text),
        Field("email", "email", True, _text),
        Field("name", "name", True, _union(_text, _null)),
        Field("domain", "domain", True, _text),
        Field(
            "state",
            "state",
            True,
            _literal("pending", "approved", "denied", "cancelled"),
        ),
        Field("role", "role", True, _union(_decode_WorkspaceRole, _null)),
        Field("created_at", "created_at", True, _number),
        Field("decided_at", "decided_at", True, _union(_number, _null)),
        Field("decided_by", "decided_by", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class OnboardingProfile(Payload):
    """The signed-in person's verified email, all null until they sign in again."""

    email: str | None
    email_domain: str | None
    display_name: str | None
    personal_domain: bool | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("email", "email", True, _union(_text, _null)),
        Field("email_domain", "email_domain", True, _union(_text, _null)),
        Field("display_name", "display_name", True, _union(_text, _null)),
        Field("personal_domain", "personal_domain", True, _union(_boolean, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class DiscoveredWorkspace(Payload):
    """A workspace the person's email domain may ask to join."""

    workspace: Name
    display_name: str
    domain: str
    request: JoinRequest | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field("domain", "domain", True, _text),
        Field("request", "request", True, _union(_record(JoinRequest), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Onboarding(Payload):
    """The Onboarding schema.

    where a signed-in person stands: their workspaces, those their domain may join, and
    whether they may create one
    """

    state: Literal["member", "onboarding", "reauthentication_required"]
    profile: OnboardingProfile
    workspaces: tuple[WorkspaceSummary, ...]
    discovered: tuple[DiscoveredWorkspace, ...]
    requests: tuple[JoinRequest, ...]
    can_create_workspace: bool
    signup: Literal["open", "waitlist", "pending"]
    workspace_name: str | None
    sign_in_url: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "state",
            "state",
            True,
            _literal("member", "onboarding", "reauthentication_required"),
        ),
        Field("profile", "profile", True, _record(OnboardingProfile)),
        Field("workspaces", "workspaces", True, _items(_record(WorkspaceSummary))),
        Field("discovered", "discovered", True, _items(_record(DiscoveredWorkspace))),
        Field("requests", "requests", True, _items(_record(JoinRequest))),
        Field("can_create_workspace", "can_create_workspace", True, _boolean),
        Field("signup", "signup", True, _literal("open", "waitlist", "pending")),
        Field("workspace_name", "workspace_name", True, _union(_text, _null)),
        Field("sign_in_url", "sign_in_url", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceList(Payload):
    """The WorkspaceList schema."""

    workspaces: tuple[WorkspaceSummary, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspaces", "workspaces", True, _items(_record(WorkspaceSummary))),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceCreate(Payload):
    """The WorkspaceCreate schema."""

    display_name: WorkspaceName

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("display_name", "display_name", True, _decode_WorkspaceName),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceCreated(Payload):
    """`created` is false when the request id already made this workspace."""

    workspace: WorkspaceSummary
    organization: Name
    created: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _record(WorkspaceSummary)),
        Field("organization", "organization", True, _decode_Name),
        Field("created", "created", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceRef(Payload):
    """The WorkspaceRef schema."""

    workspace: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceRename(Payload):
    """The WorkspaceRename schema."""

    workspace: Name
    display_name: WorkspaceName

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _decode_WorkspaceName),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceJoin(Payload):
    """The WorkspaceJoin schema.

    `workspace` is set once the caller is a member, `request` while a join request is
    pending or was denied lately
    """

    status: Literal["already_member", "pending", "denied"]
    workspace: WorkspaceSummary | None
    request: JoinRequest | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "status",
            "status",
            True,
            _literal("already_member", "pending", "denied"),
        ),
        Field("workspace", "workspace", True, _union(_record(WorkspaceSummary), _null)),
        Field("request", "request", True, _union(_record(JoinRequest), _null)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceMember(Payload):
    """One membership in the workspace, its organizations or projects."""

    issuer: str
    subject: str
    email: str | None
    name: str | None
    scope_kind: Literal["account", "organization", "project"]
    scope_id: str
    role: WorkspaceRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("email", "email", True, _union(_text, _null)),
        Field("name", "name", True, _union(_text, _null)),
        Field(
            "scope_kind",
            "scope_kind",
            True,
            _literal("account", "organization", "project"),
        ),
        Field("scope_id", "scope_id", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceMemberList(Payload):
    """The WorkspaceMemberList schema."""

    members: tuple[WorkspaceMember, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("members", "members", True, _items(_record(WorkspaceMember))),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRoleChange(Payload):
    """The MemberRoleChange schema."""

    workspace: Name
    issuer: str
    subject: str
    role: WorkspaceRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRole(Payload):
    """The MemberRole schema."""

    issuer: str
    subject: str
    role: WorkspaceRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRoleChanged(Payload):
    """The MemberRoleChanged schema."""

    member: MemberRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("member", "member", True, _record(MemberRole)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRemove(Payload):
    """The MemberRemove schema."""

    workspace: Name
    issuer: str
    subject: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class MemberGone(Payload):
    """The MemberGone schema."""

    issuer: str
    subject: str
    removed: Literal[True] = True

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("removed", "removed", True, _literal(True)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRemoved(Payload):
    """The MemberRemoved schema."""

    member: MemberGone

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("member", "member", True, _record(MemberGone)),
    )


@dataclass(frozen=True, kw_only=True)
class Invitation(Payload):
    """An invitation to a workspace; `state` is `expired` once a pending one lapses."""

    id: InvitationId
    workspace: Name
    email: str
    role: WorkspaceRole
    state: Literal["pending", "expired", "accepted", "revoked"]
    created_at: float
    expires_at: float
    created_by: str
    accepted_at: float | None
    revoked_at: float | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_InvitationId),
        Field("workspace", "workspace", True, _decode_Name),
        Field("email", "email", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
        Field(
            "state",
            "state",
            True,
            _literal("pending", "expired", "accepted", "revoked"),
        ),
        Field("created_at", "created_at", True, _number),
        Field("expires_at", "expires_at", True, _number),
        Field("created_by", "created_by", True, _text),
        Field("accepted_at", "accepted_at", True, _union(_number, _null)),
        Field("revoked_at", "revoked_at", True, _union(_number, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationList(Payload):
    """The InvitationList schema."""

    invitations: tuple[Invitation, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("invitations", "invitations", True, _items(_record(Invitation))),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationCreate(Payload):
    """`role` is `viewer` and `expires_in` a week unless given."""

    workspace: Name
    email: str
    role: WorkspaceRole | None = None
    expires_in: float | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("email", "email", True, _text),
        Field("role", "role", False, _decode_WorkspaceRole),
        Field("expires_in", "expires_in", False, _number),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationCreated(Payload):
    """The InvitationCreated schema.

    `token` and `invitation_url` only when the server has no mailer and `delivery` is
    `link`; `replaces` names a pending invitation to the same email it revoked
    """

    invitation: Invitation
    delivery: Literal["email", "link"]
    replaces: InvitationId | None
    token: str | None = None
    invitation_url: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("invitation", "invitation", True, _record(Invitation)),
        Field("delivery", "delivery", True, _literal("email", "link")),
        Field("replaces", "replaces", True, _union(_decode_InvitationId, _null)),
        Field("token", "token", False, _pattern("^gf_inv_[A-Za-z0-9_-]{43}$")),
        Field("invitation_url", "invitation_url", False, _text),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationRevoke(Payload):
    """The InvitationRevoke schema."""

    workspace: Name
    id: InvitationId

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("id", "id", True, _decode_InvitationId),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationView(Payload):
    """The InvitationView schema."""

    invitation: Invitation

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("invitation", "invitation", True, _record(Invitation)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationToken(Payload):
    """The token from the invitation link."""

    token: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("token", "token", True, _pattern("^gf_inv_[A-Za-z0-9_-]{43}$")),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationPreview(Payload):
    """The InvitationPreview schema."""

    workspace: Name
    display_name: str
    role: WorkspaceRole
    expires_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
        Field("role", "role", True, _decode_WorkspaceRole),
        Field("expires_at", "expires_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationAccepted(Payload):
    """The InvitationAccepted schema."""

    status: Literal["joined", "already_member"]
    workspace: WorkspaceSummary

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("status", "status", True, _literal("joined", "already_member")),
        Field("workspace", "workspace", True, _record(WorkspaceSummary)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestRef(Payload):
    """The JoinRequestRef schema."""

    id: JoinRequestId

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_JoinRequestId),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestView(Payload):
    """The JoinRequestView schema."""

    request: JoinRequest

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("request", "request", True, _record(JoinRequest)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestList(Payload):
    """The JoinRequestList schema."""

    requests: tuple[JoinRequestAdmin, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("requests", "requests", True, _items(_record(JoinRequestAdmin))),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestApprove(Payload):
    """`role` is `viewer` unless given."""

    workspace: Name
    id: JoinRequestId
    role: WorkspaceRole | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("id", "id", True, _decode_JoinRequestId),
        Field("role", "role", False, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestDeny(Payload):
    """The JoinRequestDeny schema."""

    workspace: Name
    id: JoinRequestId

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("id", "id", True, _decode_JoinRequestId),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceMembership(Payload):
    """The WorkspaceMembership schema.

    the account membership an approval granted, or the one the person already held
    """

    issuer: str
    subject: str
    scope_kind: Literal["account"] = "account"
    scope_id: Name
    role: WorkspaceRole

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("issuer", "issuer", True, _text),
        Field("subject", "subject", True, _text),
        Field("scope_kind", "scope_kind", True, _literal("account")),
        Field("scope_id", "scope_id", True, _decode_Name),
        Field("role", "role", True, _decode_WorkspaceRole),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestApproved(Payload):
    """The JoinRequestApproved schema."""

    request: JoinRequestAdmin
    membership: WorkspaceMembership

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("request", "request", True, _record(JoinRequestAdmin)),
        Field("membership", "membership", True, _record(WorkspaceMembership)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestDenied(Payload):
    """The JoinRequestDenied schema."""

    request: JoinRequestAdmin

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("request", "request", True, _record(JoinRequestAdmin)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomain(Payload):
    """An email domain whose people may ask to join the workspace."""

    workspace: Name
    domain: str
    enabled_by: str
    enabled_at: float

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("domain", "domain", True, _text),
        Field("enabled_by", "enabled_by", True, _text),
        Field("enabled_at", "enabled_at", True, _number),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainList(Payload):
    """The JoinDomainList schema.

    `email_domain` is the caller's own company domain, the one it may enable; null for
    a personal one
    """

    domains: tuple[JoinDomain, ...]
    email_domain: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("domains", "domains", True, _items(_record(JoinDomain))),
        Field("email_domain", "email_domain", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainConfigure(Payload):
    """Enabling takes only the domain of the admin's own verified email."""

    workspace: Name
    domain: str
    enabled: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("domain", "domain", True, _text),
        Field("enabled", "enabled", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainConfigured(Payload):
    """The JoinDomainConfigured schema."""

    workspace: Name
    domain: str
    enabled: bool

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("domain", "domain", True, _text),
        Field("enabled", "enabled", True, _boolean),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistJoin(Payload):
    """`display_name` is the workspace the person means to create."""

    display_name: WorkspaceName | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("display_name", "display_name", False, _decode_WorkspaceName),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistStatus(Payload):
    """The WaitlistStatus schema."""

    status: Literal["approved", "pending"]
    message: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("status", "status", True, _literal("approved", "pending")),
        Field("message", "message", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistEntry(Payload):
    """The WaitlistEntry schema.

    a waitlist sign-up; approving one from a company domain approves the whole domain
    """

    id: WaitlistId
    email: str
    display_name: str | None
    state: Literal["pending", "approved"]
    approves: Literal["person", "domain"]
    domain: str
    created_at: float
    decided_at: float | None
    workspace_name: str | None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_WaitlistId),
        Field("email", "email", True, _text),
        Field("display_name", "display_name", True, _union(_text, _null)),
        Field("state", "state", True, _literal("pending", "approved")),
        Field("approves", "approves", True, _literal("person", "domain")),
        Field("domain", "domain", True, _text),
        Field("created_at", "created_at", True, _number),
        Field("decided_at", "decided_at", True, _union(_number, _null)),
        Field("workspace_name", "workspace_name", True, _union(_text, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistRef(Payload):
    """The WaitlistRef schema."""

    id: WaitlistId

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("id", "id", True, _decode_WaitlistId),)


@dataclass(frozen=True, kw_only=True)
class WaitlistEntryView(Payload):
    """The WaitlistEntryView schema."""

    entry: WaitlistEntry

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("entry", "entry", True, _record(WaitlistEntry)),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginCode(Payload):
    """The CliLoginCode schema."""

    user_code: UserCode

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("user_code", "user_code", True, _decode_UserCode),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginWorkspace(Payload):
    """The CliLoginWorkspace schema."""

    workspace: Name
    display_name: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("workspace", "workspace", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginPreview(Payload):
    """The CliLoginPreview schema.

    what the sign-in page confirms: the machine asking and the workspaces with a
    project the key may serve
    """

    user_code: str
    machine: str
    expires_at: float
    email: str
    workspaces: tuple[CliLoginWorkspace, ...]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "user_code",
            "user_code",
            True,
            _pattern("^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$"),
        ),
        Field("machine", "machine", True, _text),
        Field("expires_at", "expires_at", True, _number),
        Field("email", "email", True, _text),
        Field("workspaces", "workspaces", True, _items(_record(CliLoginWorkspace))),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginApprove(Payload):
    """The CliLoginApprove schema."""

    user_code: UserCode
    workspace: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("user_code", "user_code", True, _decode_UserCode),
        Field("workspace", "workspace", True, _decode_Name),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginGranted(Payload):
    """The CliLoginGranted schema."""

    id: Name
    display_name: str

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("id", "id", True, _decode_Name),
        Field("display_name", "display_name", True, _text),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginApproved(Payload):
    """The CliLoginApproved schema."""

    user_code: str
    state: Literal["approved"] = "approved"
    workspace: CliLoginGranted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "user_code",
            "user_code",
            True,
            _pattern("^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$"),
        ),
        Field("state", "state", True, _literal("approved")),
        Field("workspace", "workspace", True, _record(CliLoginGranted)),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginDenied(Payload):
    """The CliLoginDenied schema."""

    user_code: str
    state: Literal["denied"] = "denied"

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field(
            "user_code",
            "user_code",
            True,
            _pattern("^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$"),
        ),
        Field("state", "state", True, _literal("denied")),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginApproveRequest(Payload):
    """The CliLoginApproveRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: CliLoginApprove

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(CliLoginApprove)),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginCodeRequest(Payload):
    """The CliLoginCodeRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: CliLoginCode

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(CliLoginCode)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationCreateRequest(Payload):
    """The InvitationCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: InvitationCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(InvitationCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationRevokeRequest(Payload):
    """The InvitationRevokeRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: InvitationRevoke

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(InvitationRevoke)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationTokenRequest(Payload):
    """The InvitationTokenRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: InvitationToken

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(InvitationToken)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainConfigureRequest(Payload):
    """The JoinDomainConfigureRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: JoinDomainConfigure

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(JoinDomainConfigure)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestApproveRequest(Payload):
    """The JoinRequestApproveRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: JoinRequestApprove

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(JoinRequestApprove)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestDenyRequest(Payload):
    """The JoinRequestDenyRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: JoinRequestDeny

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(JoinRequestDeny)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestRefRequest(Payload):
    """The JoinRequestRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: JoinRequestRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(JoinRequestRef)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRemoveRequest(Payload):
    """The MemberRemoveRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: MemberRemove

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(MemberRemove)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRoleChangeRequest(Payload):
    """The MemberRoleChangeRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: MemberRoleChange

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(MemberRoleChange)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistJoinRequest(Payload):
    """The WaitlistJoinRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WaitlistJoin

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WaitlistJoin)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistRefRequest(Payload):
    """The WaitlistRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WaitlistRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WaitlistRef)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceCreateRequest(Payload):
    """The WorkspaceCreateRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WorkspaceCreate

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WorkspaceCreate)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceRefRequest(Payload):
    """The WorkspaceRefRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WorkspaceRef

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WorkspaceRef)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceRenameRequest(Payload):
    """The WorkspaceRenameRequest schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    payload: WorkspaceRename

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("payload", "payload", True, _record(WorkspaceRename)),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginApprovedReply(Payload):
    """The CliLoginApprovedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: CliLoginApproved

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(CliLoginApproved)),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginDeniedReply(Payload):
    """The CliLoginDeniedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: CliLoginDenied

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(CliLoginDenied)),
    )


@dataclass(frozen=True, kw_only=True)
class CliLoginPreviewReply(Payload):
    """The CliLoginPreviewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: CliLoginPreview

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(CliLoginPreview)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationAcceptedReply(Payload):
    """The InvitationAcceptedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationAccepted

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationAccepted)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationCreatedReply(Payload):
    """The InvitationCreatedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationCreated

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationCreated)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationListReply(Payload):
    """The InvitationListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationList)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationPreviewReply(Payload):
    """The InvitationPreviewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationPreview

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationPreview)),
    )


@dataclass(frozen=True, kw_only=True)
class InvitationViewReply(Payload):
    """The InvitationViewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: InvitationView

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(InvitationView)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainConfiguredReply(Payload):
    """The JoinDomainConfiguredReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinDomainConfigured

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinDomainConfigured)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinDomainListReply(Payload):
    """The JoinDomainListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinDomainList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinDomainList)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestApprovedReply(Payload):
    """The JoinRequestApprovedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinRequestApproved

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinRequestApproved)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestDeniedReply(Payload):
    """The JoinRequestDeniedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinRequestDenied

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinRequestDenied)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestListReply(Payload):
    """The JoinRequestListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinRequestList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinRequestList)),
    )


@dataclass(frozen=True, kw_only=True)
class JoinRequestViewReply(Payload):
    """The JoinRequestViewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: JoinRequestView

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(JoinRequestView)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRemovedReply(Payload):
    """The MemberRemovedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: MemberRemoved

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(MemberRemoved)),
    )


@dataclass(frozen=True, kw_only=True)
class MemberRoleChangedReply(Payload):
    """The MemberRoleChangedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: MemberRoleChanged

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(MemberRoleChanged)),
    )


@dataclass(frozen=True, kw_only=True)
class OnboardingReply(Payload):
    """The OnboardingReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: Onboarding

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(Onboarding)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistEntryViewReply(Payload):
    """The WaitlistEntryViewReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WaitlistEntryView

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WaitlistEntryView)),
    )


@dataclass(frozen=True, kw_only=True)
class WaitlistStatusReply(Payload):
    """The WaitlistStatusReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WaitlistStatus

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WaitlistStatus)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceCreatedReply(Payload):
    """The WorkspaceCreatedReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceCreated

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceCreated)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceDetailsReply(Payload):
    """The WorkspaceDetailsReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceDetails

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceDetails)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceJoinReply(Payload):
    """The WorkspaceJoinReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceJoin

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceJoin)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceListReply(Payload):
    """The WorkspaceListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceList)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceMemberListReply(Payload):
    """The WorkspaceMemberListReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceMemberList

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceMemberList)),
    )


@dataclass(frozen=True, kw_only=True)
class WorkspaceSummaryReply(Payload):
    """The WorkspaceSummaryReply schema."""

    api_version: Literal["gf/v1"] = "gf/v1"
    project: Name
    request_id: str
    resource: Resource
    result: WorkspaceSummary

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("api_version", "api_version", True, _literal("gf/v1")),
        Field("project", "project", True, _decode_Name),
        Field("request_id", "request_id", True, _pattern("^[\\x20-\\x7e]+$")),
        Field("resource", "resource", True, _record(Resource)),
        Field("result", "result", True, _record(WorkspaceSummary)),
    )


ROUTES: tuple[Route, ...] = (
    Route(
        "predictors.predict",
        "POST",
        "/v1/projects/{project}/predictors/{name}/predict",
        QueriesRequest,
        ResultReply,
        "application/json",
    ),
    Route(
        "predict",
        "POST",
        "/v1/projects/{project}/predict",
        ManifestRequest,
        ResultReply,
        "application/json",
    ),
    Route(
        "predictors.list",
        "GET",
        "/v1/projects/{project}/predictors",
        None,
        PredictorListReply,
        None,
    ),
    Route(
        "predictors.create",
        "POST",
        "/v1/projects/{project}/predictors",
        PredictorCreateRequest,
        PredictorCreatedReply,
        "application/json",
    ),
    Route(
        "predictors.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}",
        None,
        PredictorReply,
        None,
    ),
    Route(
        "models.list",
        "GET",
        "/v1/projects/{project}/models",
        None,
        ModelListReply,
        None,
    ),
    Route(
        "models.get",
        "GET",
        "/v1/projects/{project}/models/{model}",
        None,
        ModelReply,
        None,
    ),
    Route("health", "GET", "/healthz", None, Health, None),
    Route("ready", "GET", "/readyz", None, Health, None),
    Route("live", "GET", "/livez", None, Health, None),
    Route("projects.list", "GET", "/v1/projects", None, ProjectListReply, None),
    Route(
        "projects.create",
        "POST",
        "/v1/projects",
        ProjectCreateRequest,
        ProjectReply,
        "application/json",
    ),
    Route("keys.list", "GET", "/v1/projects/{project}/keys", None, KeyListReply, None),
    Route(
        "keys.create",
        "POST",
        "/v1/projects/{project}/keys",
        KeyCreateRequest,
        KeyReply,
        "application/json",
    ),
    Route(
        "keys.revoke",
        "POST",
        "/v1/projects/{project}/keys/{key}/revoke",
        KeyRefRequest,
        KeyRevokedReply,
        "application/json",
    ),
    Route(
        "keys.rotate",
        "POST",
        "/v1/projects/{project}/keys/{key}/rotate",
        KeyRefRequest,
        KeyReply,
        "application/json",
    ),
    Route(
        "secrets.list",
        "GET",
        "/v1/projects/{project}/secrets",
        None,
        SecretListReply,
        None,
    ),
    Route(
        "secrets.create",
        "POST",
        "/v1/projects/{project}/secrets",
        SecretCreateRequest,
        SecretReply,
        "application/json",
    ),
    Route(
        "secrets.get",
        "GET",
        "/v1/projects/{project}/secrets/{secret}",
        None,
        SecretReply,
        None,
    ),
    Route(
        "secrets.rotate",
        "POST",
        "/v1/projects/{project}/secrets/{secret}/rotate",
        SecretRotateRequest,
        SecretReply,
        "application/json",
    ),
    Route(
        "secrets.delete",
        "POST",
        "/v1/projects/{project}/secrets/{secret}/delete",
        SecretRefRequest,
        SecretDeletedReply,
        "application/json",
    ),
    Route(
        "revisions.list",
        "GET",
        "/v1/projects/{project}/predictors/{name}/revisions",
        None,
        RevisionListReply,
        None,
    ),
    Route(
        "revisions.register",
        "POST",
        "/v1/projects/{project}/predictors/{name}/revisions",
        RevisionRegisterRequest,
        RevisionReply,
        "application/json",
    ),
    Route(
        "revisions.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}/revisions/{number}",
        None,
        RevisionDetailsReply,
        None,
    ),
    Route(
        "environments.list",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments",
        None,
        EnvironmentListReply,
        None,
    ),
    Route(
        "environments.create",
        "POST",
        "/v1/projects/{project}/predictors/{name}/environments",
        EnvironmentCreateRequest,
        EnvironmentReply,
        "application/json",
    ),
    Route(
        "environments.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}",
        None,
        EnvironmentReply,
        None,
    ),
    Route(
        "environments.promote",
        "POST",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/promote",
        PromoteRequest,
        EnvironmentReply,
        "application/json",
    ),
    Route(
        "environments.rollback",
        "POST",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/rollback",
        PromoteRequest,
        EnvironmentReply,
        "application/json",
    ),
    Route(
        "environments.metrics",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/metrics",
        None,
        MetricsReply,
        None,
    ),
    Route(
        "environments.logs",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/logs",
        None,
        LogsReply,
        None,
    ),
    Route(
        "predictors.delete",
        "POST",
        "/v1/projects/{project}/predictors/{name}/delete",
        EmptyRequest,
        PredictorDeletedReply,
        "application/json",
    ),
    Route(
        "contexts.list",
        "GET",
        "/v1/projects/{project}/contexts",
        None,
        LiveContextListReply,
        None,
    ),
    Route(
        "contexts.create",
        "POST",
        "/v1/projects/{project}/contexts",
        LiveContextCreateRequest,
        LiveContextReply,
        "application/json",
    ),
    Route(
        "contexts.show",
        "GET",
        "/v1/projects/{project}/contexts/{name}",
        None,
        LiveContextReply,
        None,
    ),
    Route(
        "contexts.append",
        "POST",
        "/v1/projects/{project}/contexts/{name}/append",
        LiveContextAppendRequest,
        ReceiptReply,
        "application/json",
    ),
    Route(
        "contexts.label",
        "POST",
        "/v1/projects/{project}/contexts/{name}/label",
        LiveContextLabelRequest,
        ReceiptReply,
        "application/json",
    ),
    Route(
        "contexts.imports",
        "POST",
        "/v1/projects/{project}/contexts/{name}/imports",
        ContextImportRequest,
        ImportReceiptReply,
        "application/json",
    ),
    Route(
        "contexts.refresh",
        "POST",
        "/v1/projects/{project}/contexts/{name}/refresh",
        EmptyRequest,
        ImportReceiptReply,
        "application/json",
    ),
    Route(
        "contexts.mutate",
        "POST",
        "/v1/projects/{project}/contexts/{name}/mutations",
        MutationRequest,
        AcceptedWriteReply,
        "application/json",
    ),
    Route(
        "contexts.snapshot",
        "POST",
        "/v1/projects/{project}/contexts/{name}/snapshot",
        ContextSnapshotQueryRequest,
        ContextSnapshotReply,
        "application/json",
    ),
    Route(
        "sources.list",
        "GET",
        "/v1/projects/{project}/sources",
        None,
        SourceListReply,
        None,
    ),
    Route(
        "sources.pin",
        "POST",
        "/v1/projects/{project}/sources",
        SourcePinRequest,
        SourcePinReply,
        "application/json",
    ),
    Route(
        "sources.get",
        "GET",
        "/v1/projects/{project}/sources/{source}",
        None,
        SourcePinReply,
        None,
    ),
    Route(
        "snapshots.upload",
        "POST",
        "/v1/projects/{project}/snapshots",
        None,
        UploadReceiptReply,
        "application/vnd.apache.arrow.stream",
    ),
    Route("auth.status", "GET", "/v1/auth/status", None, AuthStatusReply, None),
    Route(
        "auth.logout",
        "POST",
        "/v1/auth/logout",
        EmptyRequest,
        LoggedOutReply,
        "application/json",
    ),
    Route("workspace.get", "GET", "/v1/workspace", None, ProjectListReply, None),
    Route(
        "organizations.list",
        "GET",
        "/v1/organizations",
        None,
        OrganizationListReply,
        None,
    ),
    Route(
        "organizations.create",
        "POST",
        "/v1/organizations",
        OrganizationCreateRequest,
        OrganizationReply,
        "application/json",
    ),
    Route("sso.get", "GET", "/v1/projects/{project}/sso", None, SsoReply, None),
    Route(
        "sso.configure",
        "POST",
        "/v1/projects/{project}/sso",
        SsoConfigureRequest,
        SsoReply,
        "application/json",
    ),
    Route(
        "audit.list",
        "GET",
        "/v1/projects/{project}/audit",
        None,
        AuditLogReply,
        None,
    ),
    Route(
        "logins.start",
        "POST",
        "/v1/cli-logins",
        CliLoginStart,
        CliLoginGrant,
        "application/json",
    ),
    Route(
        "logins.token",
        "POST",
        "/v1/cli-logins/token",
        CliLoginPoll,
        CliLoginToken,
        "application/json",
    ),
    Route("onboarding.get", "GET", "/v1/onboarding", None, OnboardingReply, None),
    Route("workspaces.list", "GET", "/v1/workspaces", None, WorkspaceListReply, None),
    Route(
        "workspaces.create",
        "POST",
        "/v1/workspaces",
        WorkspaceCreateRequest,
        WorkspaceCreatedReply,
        "application/json",
    ),
    Route(
        "workspaces.get",
        "GET",
        "/v1/workspaces/{workspace}",
        None,
        WorkspaceDetailsReply,
        None,
    ),
    Route(
        "workspaces.update",
        "POST",
        "/v1/workspaces/{workspace}",
        WorkspaceRenameRequest,
        WorkspaceSummaryReply,
        "application/json",
    ),
    Route(
        "workspaces.join",
        "POST",
        "/v1/workspaces/{workspace}/join",
        WorkspaceRefRequest,
        WorkspaceJoinReply,
        "application/json",
    ),
    Route(
        "workspaces.members",
        "GET",
        "/v1/workspaces/{workspace}/members",
        None,
        WorkspaceMemberListReply,
        None,
    ),
    Route(
        "members.update",
        "POST",
        "/v1/workspaces/{workspace}/members/role",
        MemberRoleChangeRequest,
        MemberRoleChangedReply,
        "application/json",
    ),
    Route(
        "members.remove",
        "POST",
        "/v1/workspaces/{workspace}/members/remove",
        MemberRemoveRequest,
        MemberRemovedReply,
        "application/json",
    ),
    Route(
        "invitations.list",
        "GET",
        "/v1/workspaces/{workspace}/invitations",
        None,
        InvitationListReply,
        None,
    ),
    Route(
        "invitations.create",
        "POST",
        "/v1/workspaces/{workspace}/invitations",
        InvitationCreateRequest,
        InvitationCreatedReply,
        "application/json",
    ),
    Route(
        "invitations.revoke",
        "POST",
        "/v1/workspaces/{workspace}/invitations/{id}/revoke",
        InvitationRevokeRequest,
        InvitationViewReply,
        "application/json",
    ),
    Route(
        "requests.list",
        "GET",
        "/v1/workspaces/{workspace}/join-requests",
        None,
        JoinRequestListReply,
        None,
    ),
    Route(
        "requests.approve",
        "POST",
        "/v1/workspaces/{workspace}/join-requests/{id}/approve",
        JoinRequestApproveRequest,
        JoinRequestApprovedReply,
        "application/json",
    ),
    Route(
        "requests.deny",
        "POST",
        "/v1/workspaces/{workspace}/join-requests/{id}/deny",
        JoinRequestDenyRequest,
        JoinRequestDeniedReply,
        "application/json",
    ),
    Route(
        "domains.list",
        "GET",
        "/v1/workspaces/{workspace}/domains",
        None,
        JoinDomainListReply,
        None,
    ),
    Route(
        "domains.configure",
        "POST",
        "/v1/workspaces/{workspace}/domains",
        JoinDomainConfigureRequest,
        JoinDomainConfiguredReply,
        "application/json",
    ),
    Route(
        "invitations.preview",
        "POST",
        "/v1/invitations/preview",
        InvitationTokenRequest,
        InvitationPreviewReply,
        "application/json",
    ),
    Route(
        "invitations.accept",
        "POST",
        "/v1/invitations/accept",
        InvitationTokenRequest,
        InvitationAcceptedReply,
        "application/json",
    ),
    Route(
        "requests.cancel",
        "POST",
        "/v1/join-requests/{id}/cancel",
        JoinRequestRefRequest,
        JoinRequestViewReply,
        "application/json",
    ),
    Route(
        "waitlist.join",
        "POST",
        "/v1/waitlist",
        WaitlistJoinRequest,
        WaitlistStatusReply,
        "application/json",
    ),
    Route(
        "waitlist.get",
        "GET",
        "/v1/waitlist/{id}",
        None,
        WaitlistEntryViewReply,
        None,
    ),
    Route(
        "waitlist.approve",
        "POST",
        "/v1/waitlist/{id}/approve",
        WaitlistRefRequest,
        WaitlistEntryViewReply,
        "application/json",
    ),
    Route(
        "device.preview",
        "POST",
        "/v1/cli-logins/preview",
        CliLoginCodeRequest,
        CliLoginPreviewReply,
        "application/json",
    ),
    Route(
        "device.approve",
        "POST",
        "/v1/cli-logins/approve",
        CliLoginApproveRequest,
        CliLoginApprovedReply,
        "application/json",
    ),
    Route(
        "device.deny",
        "POST",
        "/v1/cli-logins/deny",
        CliLoginCodeRequest,
        CliLoginDeniedReply,
        "application/json",
    ),
)
