"""The alpha deployment contract, independent of runtime provisioning."""

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Required, TypedDict

import yaml

VERSION = "gf/v1alpha1"
MODEL = "general-forecast/tabicl-v2-classifier"
PIN = "5fb6a4c6f24bd5fb3623f97e3c62186e12e28e5b"
NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")


class ServiceError(Exception):
    """A public error with no prediction or private runtime details."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        """Constructs the safe public error envelope."""
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

    def body(self) -> dict[str, Any]:
        """Returns the shared wire error envelope."""
        return {"error": {"code": self.code, "message": self.message}}


PUBLIC_ERROR_CODES = frozenset(
    [
        "AlreadyExists",
        "AmbiguousFrequency",
        "AmbiguousTask",
        "ArtifactIntegrityError",
        "ArtifactMismatch",
        "ArtifactUnavailable",
        "ArtifactUnsupported",
        "BackupIntegrityError",
        "BackupUnavailable",
        "CapacityExceeded",
        "ConfigurationError",
        "Conflict",
        "ContextCapacityExceeded",
        "ContextRequired",
        "CredentialInactive",
        "CredentialUnavailable",
        "CredentialsUnavailable",
        "DomainUnverified",
        "DomainVerificationFailed",
        "DurabilityUnavailable",
        "DurableStateUnavailable",
        "EffectUnavailable",
        "EmptyContext",
        "EmptySource",
        "EnterpriseSSOUnavailable",
        "Forbidden",
        "FreshnessExceeded",
        "HostAuthorityUnavailable",
        "ImmutableRevision",
        "InferenceFailed",
        "InstanceClosed",
        "InternalError",
        "InvalidContext",
        "InvalidLogin",
        "InvalidPackage",
        "InvalidTable",
        "InvitationEmailMismatch",
        "InvitationUnavailable",
        "IsolationUnavailable",
        "JobBusy",
        "JobCapacityExceeded",
        "JobExists",
        "JobIntegrityError",
        "JobStorageUnavailable",
        "JobTerminal",
        "LastAdministrator",
        "LinkConfirmationRequired",
        "LockMismatch",
        "LockOutdated",
        "LockRequired",
        "MethodNotAllowed",
        "MissingFile",
        "MissingValues",
        "ModelMismatch",
        "ModelNotFound",
        "ModelOutputError",
        "ModelResolutionRequired",
        "ModelRuntimeUnavailable",
        "ModelUnavailable",
        "NotFound",
        "NotReady",
        "NothingToRetry",
        "OperationSuperseded",
        "OperationUnresolved",
        "OwnershipMismatch",
        "PackageOutputExceeded",
        "PackageProtocolError",
        "PackageTestFailed",
        "PackageTestTimeout",
        "PendingExpired",
        "PersistenceUnavailable",
        "PersonalEmailDomain",
        "PredictorFailed",
        "PredictorTimeout",
        "ProtocolError",
        "ProviderAuthenticationUnavailable",
        "ProviderConflict",
        "ProviderError",
        "ProviderNotFound",
        "ProviderOperationFailed",
        "ProviderPending",
        "ProviderPrerequisiteMissing",
        "ProviderProtocolError",
        "ProviderProvisioningFailed",
        "ProviderRuntimeError",
        "ProviderTimeout",
        "ProviderTransportError",
        "ProviderUnavailable",
        "RateLimited",
        "ReauthenticationRequired",
        "RebuildRequired",
        "RecoveryUnavailable",
        "RedeployRequiresStop",
        "RedirectRefused",
        "RegistryAuthenticationUnavailable",
        "RequestDisconnected",
        "RequestTimeout",
        "RequestTooLarge",
        "ResidencyMismatch",
        "RevisionConflict",
        "RevisionSuperseded",
        "RuntimeUnavailable",
        "SSOUnconfigured",
        "SchemaChanged",
        "SchemaMismatch",
        "SchemaRequired",
        "SchemaVersionUnsupported",
        "ScopeMismatch",
        "SecretUnavailable",
        "SecretsUnconfigured",
        "ServiceUnavailable",
        "SessionRequired",
        "SingleWriterUnavailable",
        "SnapshotMismatch",
        "SnapshotNotFound",
        "SourceChanged",
        "SourceNotConfigured",
        "SourceUnavailable",
        "StaleComponent",
        "StateBusy",
        "SubmissionConflict",
        "Unauthorized",
        "UnknownRow",
        "UnsafePath",
        "Unsupported",
        "UnsupportedCapability",
        "UnsupportedClassCount",
        "UnsupportedCovariates",
        "UnsupportedDeploymentTask",
        "UnsupportedFormat",
        "UnsupportedJobTask",
        "UnsupportedLockVersion",
        "UnsupportedMode",
        "UnsupportedModel",
        "UnsupportedOperation",
        "UnsupportedProvider",
        "UnsupportedQuantiles",
        "UnsupportedTask",
        "UnsupportedType",
        "ValidationError",
        "WaitTimeout",
        "WriteConflict",
    ]
)
"""Codes a GF service may return; clients map any other code to RemoteError."""


def public_code(code: str) -> str:
    """Classifies an untrusted remote code without echoing unknown values."""
    return code if code in PUBLIC_ERROR_CODES else "RemoteError"


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
        """Rejects arbitrary code packages and mutable aliases."""
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
        for field, high in (("max_rows", 1024), ("max_writes", 10000)):
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
