"""The HTTP API's request and reply types, generated from api/openapi.yaml.

Do not edit: run `python api/generate.py`. Every schema is a frozen dataclass
(or a type alias for scalars, arrays and mappings) with `from_payload`, which
checks the JSON shape and names the offending field in `PayloadError.param`,
and `to_payload`, which writes the JSON back without the optional fields left
as None. `ROUTES` lists every route with its request and reply types.
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
    """A route: its request envelope type (None for GET) and its reply type."""

    name: str
    method: str
    path: str
    request: type[Payload] | None
    reply: type[Payload]


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


ResourceKind = Literal["predictor", "model", "project", "api-key", "secret", "context"]
_decode_ResourceKind: Decoder = _literal(
    "predictor",
    "model",
    "project",
    "api-key",
    "secret",
    "context",
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
    generation)
    """

    ValidationError = "ValidationError"
    NotFound = "NotFound"
    Unauthorized = "Unauthorized"
    CapacityExceeded = "CapacityExceeded"
    ModelNotReady = "ModelNotReady"
    Internal = "Internal"
    Conflict = "Conflict"


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
class Inputs(Payload):
    """The Inputs schema.

    inputs by rule rather than by list: `exclude` removes columns from the inferred
    set, `types` overrides inferred types; forecast predictors use `past` and `future`
    instead
    """

    exclude: tuple[str, ...] | None = None
    types: dict[str, ColumnType] | None = None
    past: tuple[str, ...] | None = None
    future: tuple[str, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("exclude", "exclude", False, _items(_text)),
        Field("types", "types", False, _mapping(_decode_ColumnType)),
        Field("past", "past", False, _items(_text)),
        Field("future", "future", False, _items(_text)),
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


@dataclass(frozen=True, kw_only=True)
class Queries(Payload):
    """The Queries schema.

    the declared predictor payload; `context` only when the manifest declares `context:
    {request: ...}`
    """

    queries: tuple[Row, ...]
    context: tuple[Row, ...] | None = None
    as_of: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("queries", "queries", True, _items(_decode_Row)),
        Field("context", "context", False, _items(_decode_Row)),
        Field("as_of", "as_of", False, _text),
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


@dataclass(frozen=True, kw_only=True)
class ContextInfo(Payload):
    """The context a prediction actually used."""

    kind: ContextKind
    rows: int
    digest: Digest | None = None
    as_of: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _decode_ContextKind),
        Field("rows", "rows", True, _integer),
        Field("digest", "digest", False, _decode_Digest),
        Field("as_of", "as_of", False, _text),
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
    revision: int | None = None
    environment: Name | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("model", "model", True, _decode_ModelId),
        Field("task", "task", True, _decode_Task),
        Field("context", "context", True, _record(ContextInfo)),
        Field("revision", "revision", False, _integer),
        Field("environment", "environment", False, _decode_Name),
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
class Input(Payload):
    """The Input schema."""

    name: str
    type: ColumnType

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field("type", "type", True, _decode_ColumnType),
    )


@dataclass(frozen=True, kw_only=True)
class ContextLock(Payload):
    """The context as gf.lock pins it."""

    kind: ContextKind
    rows: int | None = None
    query_rows: int | None = None
    digest: Digest | None = None
    live: Name | None = None
    max_rows: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("kind", "kind", True, _decode_ContextKind),
        Field("rows", "rows", False, _integer),
        Field("query_rows", "query_rows", False, _integer),
        Field("digest", "digest", False, _decode_Digest),
        Field("live", "live", False, _decode_Name),
        Field("max_rows", "max_rows", False, _integer),
    )


@dataclass(frozen=True, kw_only=True)
class ResolvedSelection(Payload):
    """The selection as gf.lock pins it, in rows."""

    core: int | None
    neighbors: int
    kind: Literal["stratified", "random"]
    recent: int | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("core", "core", True, _union(_integer, _null)),
        Field("neighbors", "neighbors", True, _integer),
        Field("kind", "kind", True, _literal("stratified", "random")),
        Field("recent", "recent", False, _union(_integer, _null)),
    )


@dataclass(frozen=True, kw_only=True)
class Resolution(Payload):
    """The resolved predictor gf check prints and gf.lock pins."""

    name: Name
    model: ModelId
    model_digest: Digest
    task: Task
    target: str
    target_type: ColumnType
    inputs: tuple[Input, ...]
    context: ContextLock
    selection: ResolvedSelection
    classes: tuple[str, ...] | None = None
    id: str | None = None
    time: str | None = None
    series: tuple[str, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("model", "model", True, _decode_ModelId),
        Field("model_digest", "model_digest", True, _decode_Digest),
        Field("task", "task", True, _decode_Task),
        Field("target", "target", True, _text),
        Field("target_type", "target_type", True, _decode_ColumnType),
        Field("classes", "classes", False, _items(_text)),
        Field("id", "id", False, _text),
        Field("time", "time", False, _text),
        Field("series", "series", False, _items(_text)),
        Field("inputs", "inputs", True, _items(_record(Input))),
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
    runtime: Name
    status: State

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("predictor", "predictor", True, _decode_Name),
        Field("name", "name", True, _decode_Name),
        Field("revision", "revision", True, _union(_integer, _null)),
        Field("generation", "generation", True, _integer),
        Field("runtime", "runtime", True, _decode_Name),
        Field("status", "status", True, _decode_State),
    )


@dataclass(frozen=True, kw_only=True)
class Predictor(Payload):
    """The Predictor schema."""

    name: Name
    model: ModelId
    model_digest: Digest
    task: Task
    target: str
    target_type: ColumnType
    inputs: tuple[Input, ...]
    context: ContextLock
    selection: ResolvedSelection
    limits: Limits
    environments: tuple[Environment, ...]
    serving: Serving
    classes: tuple[str, ...] | None = None
    id: str | None = None
    time: str | None = None
    series: tuple[str, ...] | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("model", "model", True, _decode_ModelId),
        Field("model_digest", "model_digest", True, _decode_Digest),
        Field("task", "task", True, _decode_Task),
        Field("target", "target", True, _text),
        Field("target_type", "target_type", True, _decode_ColumnType),
        Field("classes", "classes", False, _items(_text)),
        Field("id", "id", False, _text),
        Field("time", "time", False, _text),
        Field("series", "series", False, _items(_text)),
        Field("inputs", "inputs", True, _items(_record(Input))),
        Field("context", "context", True, _record(ContextLock)),
        Field("selection", "selection", True, _record(ResolvedSelection)),
        Field("limits", "limits", True, _record(Limits)),
        Field("environments", "environments", True, _items(_record(Environment))),
        Field("serving", "serving", True, _record(Serving)),
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
    state: Literal["supported", "unsupported"]
    code: str | None = None
    reason: str | None = None

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _text),
        Field("state", "state", True, _literal("supported", "unsupported")),
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

    files: dict[str, Digest]

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("files", "files", True, _mapping(_decode_Digest)),
    )


@dataclass(frozen=True, kw_only=True)
class Lock(Payload):
    """Gf.lock."""

    lock_version: Literal[1] = 1
    predictor: Resolution
    model: LockedModel

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("lock_version", "lock_version", True, _literal(1)),
        Field("predictor", "predictor", True, _record(Resolution)),
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
    runtime: Name

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("runtime", "runtime", True, _decode_Name),
    )


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
    as_of: float | None
    accepted_version: int | str | None
    serving_version: int | str | None
    freshness_lag_seconds: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("source", "source", True, _text),
        Field("version", "version", True, _union(_integer, _text, _null)),
        Field("rows", "rows", True, _integer),
        Field("as_of", "as_of", True, _union(_number, _null)),
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

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("inputs", "inputs", True, _items(_text)),
        Field("targets", "targets", True, _mapping(_decode_ColumnType)),
        Field("max_rows", "max_rows", True, _integer),
        Field("rows", "rows", True, _integer),
        Field("labeled_rows", "labeled_rows", True, _integer),
        Field("sequence", "sequence", True, _integer),
        Field("accepted_at", "accepted_at", True, _union(_number, _null)),
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

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("name", "name", True, _decode_Name),
        Field("inputs", "inputs", True, _items(_text)),
        Field("targets", "targets", True, _mapping(_decode_ColumnType)),
        Field("max_rows", "max_rows", False, _integer),
    )
    DEFAULTS: ClassVar[dict[str, Any]] = {"max_rows": 100000}


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


ROUTES: tuple[Route, ...] = (
    Route(
        "predictors.predict",
        "POST",
        "/v1/projects/{project}/predictors/{name}/predict",
        QueriesRequest,
        ResultReply,
    ),
    Route(
        "predict",
        "POST",
        "/v1/projects/{project}/predict",
        ManifestRequest,
        ResultReply,
    ),
    Route(
        "predictors.list",
        "GET",
        "/v1/projects/{project}/predictors",
        None,
        PredictorListReply,
    ),
    Route(
        "predictors.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}",
        None,
        PredictorReply,
    ),
    Route("models.list", "GET", "/v1/projects/{project}/models", None, ModelListReply),
    Route(
        "models.get",
        "GET",
        "/v1/projects/{project}/models/{model}",
        None,
        ModelReply,
    ),
    Route("health", "GET", "/healthz", None, Health),
    Route("ready", "GET", "/readyz", None, Health),
    Route("live", "GET", "/livez", None, Health),
    Route("projects.list", "GET", "/v1/projects", None, ProjectListReply),
    Route(
        "projects.create",
        "POST",
        "/v1/projects",
        ProjectCreateRequest,
        ProjectReply,
    ),
    Route("keys.list", "GET", "/v1/projects/{project}/keys", None, KeyListReply),
    Route(
        "keys.create",
        "POST",
        "/v1/projects/{project}/keys",
        KeyCreateRequest,
        KeyReply,
    ),
    Route(
        "keys.revoke",
        "POST",
        "/v1/projects/{project}/keys/{key}/revoke",
        KeyRefRequest,
        KeyRevokedReply,
    ),
    Route(
        "keys.rotate",
        "POST",
        "/v1/projects/{project}/keys/{key}/rotate",
        KeyRefRequest,
        KeyReply,
    ),
    Route(
        "secrets.list",
        "GET",
        "/v1/projects/{project}/secrets",
        None,
        SecretListReply,
    ),
    Route(
        "secrets.create",
        "POST",
        "/v1/projects/{project}/secrets",
        SecretCreateRequest,
        SecretReply,
    ),
    Route(
        "secrets.get",
        "GET",
        "/v1/projects/{project}/secrets/{secret}",
        None,
        SecretReply,
    ),
    Route(
        "secrets.rotate",
        "POST",
        "/v1/projects/{project}/secrets/{secret}/rotate",
        SecretRotateRequest,
        SecretReply,
    ),
    Route(
        "secrets.delete",
        "POST",
        "/v1/projects/{project}/secrets/{secret}/delete",
        SecretRefRequest,
        SecretDeletedReply,
    ),
    Route(
        "revisions.list",
        "GET",
        "/v1/projects/{project}/predictors/{name}/revisions",
        None,
        RevisionListReply,
    ),
    Route(
        "revisions.register",
        "POST",
        "/v1/projects/{project}/predictors/{name}/revisions",
        RevisionRegisterRequest,
        RevisionReply,
    ),
    Route(
        "revisions.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}/revisions/{number}",
        None,
        RevisionDetailsReply,
    ),
    Route(
        "environments.list",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments",
        None,
        EnvironmentListReply,
    ),
    Route(
        "environments.create",
        "POST",
        "/v1/projects/{project}/predictors/{name}/environments",
        EnvironmentCreateRequest,
        EnvironmentReply,
    ),
    Route(
        "environments.get",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}",
        None,
        EnvironmentReply,
    ),
    Route(
        "environments.promote",
        "POST",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/promote",
        PromoteRequest,
        EnvironmentReply,
    ),
    Route(
        "environments.rollback",
        "POST",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/rollback",
        PromoteRequest,
        EnvironmentReply,
    ),
    Route(
        "environments.metrics",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/metrics",
        None,
        MetricsReply,
    ),
    Route(
        "environments.logs",
        "GET",
        "/v1/projects/{project}/predictors/{name}/environments/{environment}/logs",
        None,
        LogsReply,
    ),
    Route(
        "predictors.delete",
        "POST",
        "/v1/projects/{project}/predictors/{name}/delete",
        EmptyRequest,
        PredictorDeletedReply,
    ),
    Route(
        "contexts.list",
        "GET",
        "/v1/projects/{project}/contexts",
        None,
        LiveContextListReply,
    ),
    Route(
        "contexts.create",
        "POST",
        "/v1/projects/{project}/contexts",
        LiveContextCreateRequest,
        LiveContextReply,
    ),
    Route(
        "contexts.show",
        "GET",
        "/v1/projects/{project}/contexts/{name}",
        None,
        LiveContextReply,
    ),
    Route(
        "contexts.append",
        "POST",
        "/v1/projects/{project}/contexts/{name}/append",
        LiveContextAppendRequest,
        ReceiptReply,
    ),
    Route(
        "contexts.label",
        "POST",
        "/v1/projects/{project}/contexts/{name}/label",
        LiveContextLabelRequest,
        ReceiptReply,
    ),
)
