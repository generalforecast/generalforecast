# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The alpha deployment contract, independent of runtime provisioning."""

import math
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Required, TypedDict

import yaml

from generalforecast.models.catalog import V1_POLICY

VERSION = "gf/v1"
MODEL = "general-forecast/tabicl-v2-classifier"
PIN = "5fb6a4c6f24bd5fb3623f97e3c62186e12e28e5b"
NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")


class ServiceError(Exception):
    """A public error with no prediction or private runtime details."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        """Constructs the safe public error envelope."""
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

    @property
    def public(self) -> str:
        """The wire code, one of the OpenAPI `ErrorCode` enum."""
        return public_code(self.code, self.status)

    def body(self) -> dict[str, Any]:
        """The console session routes' error envelope, with the typed code."""
        return {"error": {"code": self.code, "message": self.message}}

    def public_body(self) -> dict[str, Any]:
        """The API's bare error envelope, with the OpenAPI code."""
        return {"error": {"code": self.public, "message": self.message}}


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
            "LinkConfirmationRequired",
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


@dataclass(frozen=True)
class PredictionSpec:
    """Portable native prediction definition, not a managed resource."""

    api_version: str
    dataset: str
    model: ModelPin
    target: str
    context: str
    max_update_lag_seconds: float

    @classmethod
    def parse(cls, value: Any) -> "PredictionSpec":
        """Validates the spec without contacting a service or loading weights."""
        fields(value, set(cls.__dataclass_fields__))
        require(value["api_version"] == VERSION, "unsupported prediction api_version")
        name(value["dataset"])
        name(value["target"])
        require(value["context"] == "exact", "only exact labeled context is supported")
        number(value["max_update_lag_seconds"], 0, 86400, "max_update_lag_seconds")
        return cls(
            **(
                value
                | {
                    "model": ModelPin.parse(value["model"]),
                    "max_update_lag_seconds": float(value["max_update_lag_seconds"]),
                }
            )
        )


@dataclass(frozen=True)
class DeploymentConfig:
    """A versioned desired deployment configuration."""

    api_version: str
    kind: str
    name: str
    prediction: PredictionSpec

    @classmethod
    def parse(cls, value: Any) -> "DeploymentConfig":
        """Rejects unknown fields and unsupported versions locally."""
        fields(value, set(cls.__dataclass_fields__))
        require(value["api_version"] == VERSION, "unsupported deployment api_version")
        require(value["kind"] == "deployment", "kind must be deployment")
        name(value["name"])
        return cls(
            **(value | {"prediction": PredictionSpec.parse(value["prediction"])})
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the same contract all clients submit."""
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "DeploymentConfig":
        """Reads a YAML or JSON deployment file."""
        return cls.parse(yaml.safe_load(Path(path).read_text()))


@dataclass(frozen=True)
class DatasetConfig:
    """Provisional fixture bootstrap only, not the future data API."""

    api_version: str
    kind: str
    name: str
    features: list[str]
    target: str
    max_rows: int
    max_writes: int

    @classmethod
    def parse(cls, value: Any) -> "DatasetConfig":
        """Checks the bounded numeric/boolean bootstrap capability."""
        fields(value, set(cls.__dataclass_fields__))
        require(value["api_version"] == VERSION, "unsupported dataset api_version")
        require(value["kind"] == "bootstrap-dataset", "kind must be bootstrap-dataset")
        name(value["name"])
        name(value["target"])
        features = value["features"]
        require(
            isinstance(features, list) and 0 < len(features) <= 32,
            "expected 1..32 features",
        )
        for feature in features:
            name(feature)
        require(len(set(features)) == len(features), "duplicate features")
        require(value["target"] not in features, "target cannot be a feature")
        for field, high in (
            ("max_rows", V1_POLICY.context_rows),
            ("max_writes", 10000),
        ):
            require(type(value[field]) is int, f"{field} must be an integer")
            number(value[field], 1, high, field)
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


class Receipt(TypedDict):
    """Durable acceptance, explicitly not serving visibility."""

    write_id: str
    sequence: int
    accepted_at: float
    durably_accepted: bool


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


class DeploymentStatus(TypedDict):
    """Desired/effective revisions and current process readiness."""

    name: str
    desired_revision: int
    effective_revision: int | None
    desired_config: dict[str, Any]
    effective_config: dict[str, Any] | None
    phase: str
    ready: bool
    runtime_ready: bool
    labeled_rows: int
    last_error: dict[str, str] | None
    freshness: Freshness
    capabilities: dict[str, Any]


class Score(TypedDict):
    """A score and provenance, never a business action."""

    id: str
    probabilities: dict[str, float]
    deployment_revision: int
    model: dict[str, str]
    dataset: str
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
