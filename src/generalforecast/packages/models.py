"""Immutable catalog metadata and explicit online qualification, never weight loading."""

import re
from dataclasses import dataclass
from typing import Any, Protocol

from generalforecast.models.catalog import SERIES_TYPES, TABULAR_TYPES
from generalforecast.packages.validation import (
    PackageError,
    digest,
    ensure,
    fields,
    hash_value,
    integer,
    names,
    text,
)

# Pin fields that record the serving policy applied, not what the model is.
POLICY_FIELDS = ("max_context_rows", "max_features", "max_horizon", "cpu", "memory")


@dataclass(frozen=True)
class ModelPin:
    """An exact model artifact and variant, its capabilities and the applied policy.

    Identity is the artifact, variant and intrinsic capability; the request
    bounds and compute are the serving policy a lock was compiled under, which
    the whole lock's digest covers but the model's identity does not.
    """

    name: str
    repository: str
    revision: str
    digest: str
    files: tuple[tuple[str, str], ...]
    tasks: tuple[str, ...]
    outputs: tuple[str, ...]
    input_types: tuple[str, ...]
    missing_values: bool
    max_context_rows: int
    max_features: int
    max_classes: int | None = None
    max_horizon: int | None = None
    quantiles: tuple[float, ...] = ()
    past_inputs: bool = False
    future_inputs: bool = False
    cpu: int = 2
    memory: str = "8Gi"
    quantile_step: float | None = None
    codes: int | None = None
    estimators: int | None = None

    @property
    def resolved(self) -> str:
        """A portable content-addressed model reference, never a moving alias."""
        return (
            f"catalog/{self.name}@{self.digest}"
            + (f"@{self.codes}" if self.codes is not None else "")
            + (f"*{self.estimators}" if self.estimators is not None else "")
        )

    def to_dict(self) -> dict:
        """Serializes exact capability metadata, provenance and applied policy."""
        return {
            "name": self.name,
            "repository": self.repository,
            "revision": self.revision,
            "digest": self.digest,
            "files": dict(self.files),
            "codes": self.codes,
            "estimators": self.estimators,
            "tasks": list(self.tasks),
            "outputs": list(self.outputs),
            "input_types": list(self.input_types),
            "missing_values": self.missing_values,
            "max_context_rows": self.max_context_rows,
            "max_features": self.max_features,
            "max_classes": self.max_classes,
            "max_horizon": self.max_horizon,
            "quantiles": list(self.quantiles),
            "past_inputs": self.past_inputs,
            "future_inputs": self.future_inputs,
            "cpu": self.cpu,
            "memory": self.memory,
            "quantile_step": self.quantile_step,
        }

    def identity(self) -> dict:
        """The artifact, variant and capabilities, without the applied policy."""
        value = self.to_dict()
        for key in POLICY_FIELDS:
            value.pop(key)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> "ModelPin":
        """Validates serialized pins before they can influence compilation."""
        required = {
            "name",
            "repository",
            "revision",
            "digest",
            "files",
            "codes",
            "estimators",
            "tasks",
            "outputs",
            "input_types",
            "missing_values",
            "max_context_rows",
            "max_features",
            "max_classes",
            "max_horizon",
            "quantiles",
            "past_inputs",
            "future_inputs",
            "cpu",
            "memory",
            "quantile_step",
        }
        fields(value, required)
        for key in ("name", "repository", "revision", "memory"):
            text(value[key])
        ensure(
            re.fullmatch(r"[0-9a-f]{40}", value["revision"]),
            "model revision must be an immutable upstream commit",
        )
        hash_value(value["digest"])
        ensure(
            type(value["files"]) is dict and bool(value["files"]),
            "model requires artifact file hashes",
        )
        from generalforecast.packages.files import relative

        for path, hashed in value["files"].items():
            relative(path)
            hash_value("sha256:" + hashed)
        expected = digest(value["files"])
        ensure(
            expected == value["digest"],
            "model artifact digest does not match file manifest",
            "ModelMismatch",
        )
        for key in ("tasks", "outputs", "input_types"):
            names(value[key])
            ensure(bool(value[key]), "model capabilities cannot be empty")
        ensure(
            set(value["tasks"]) <= {"classification", "regression", "forecast"},
            "unsupported model task",
        )
        for key in ("missing_values", "past_inputs", "future_inputs"):
            ensure(type(value[key]) is bool, "model capability must be boolean")
        for key in ("max_context_rows", "max_features", "cpu"):
            integer(value[key])
        for key in ("max_classes", "max_horizon", "codes", "estimators"):
            if value[key] is not None:
                integer(value[key])
        for key in ("codes", "estimators"):
            ensure(
                value[key] is None or value[key] > 0,
                "model variant counts are positive",
            )
        ensure(
            type(value["quantiles"]) is list
            and all(type(q) in (float, int) and 0 < q < 1 for q in value["quantiles"]),
            "invalid model quantiles",
        )
        ensure(
            value["quantile_step"] is None
            or (
                type(value["quantile_step"]) in (int, float)
                and 0 < value["quantile_step"] < 1
            ),
            "invalid model quantile step",
        )
        normalized: dict[str, Any] = dict(value)
        normalized["files"] = tuple(sorted(value["files"].items()))
        for key in ("tasks", "outputs", "input_types", "quantiles"):
            normalized[key] = tuple(value[key])
        return cls(**normalized)


def project(entry: Any, ref: Any, policy: Any) -> ModelPin:
    """A catalog entry and variant as a pin, under a serving policy's bounds."""
    capability = entry.capability
    task = capability.task
    return ModelPin(
        entry.name,
        entry.artifact.repository,
        entry.artifact.revision,
        entry.artifact.digest,
        tuple(sorted(entry.artifact.files.items())),
        (task,),
        ("label", "probabilities")
        if task == "classification"
        else ("value", "quantiles"),
        TABULAR_TYPES if capability.tabular else SERIES_TYPES,
        True,
        policy.context_rows,
        policy.features if capability.tabular else policy.covariates,
        entry.capacity(ref),
        policy.horizon if task == "forecast" else None,
        capability.quantiles,
        capability.past_inputs,
        capability.future_inputs,
        policy.cpu,
        policy.memory,
        capability.quantile_step,
        ref.codes,
        ref.estimators,
    )


def describe(requested: str) -> ModelPin:
    """Any pinned catalog model's metadata for trusted local use, whatever its gates.

    Packages and servers resolve through `CatalogResolver`, which also requires
    package qualification; local estimators validate intent with this instead.
    """
    from generalforecast.models.catalog import V1_POLICY, ModelRefError, parse

    try:
        ref = parse(requested)
    except ModelRefError as error:
        raise PackageError(error.code, str(error)) from None
    ensure(
        bool(ref.entry.artifact.files),
        f"{ref.entry.name} has no pinned files to describe",
        "ModelResolutionRequired",
    )
    return project(ref.entry, ref, V1_POLICY)


class ModelResolver(Protocol):
    """A capability resolver must honor offline and return a content-addressed pin."""

    def resolve(self, requested: str, *, offline: bool = True) -> ModelPin:
        """Resolves metadata only; loading and proving engine artifacts is separate."""
        ...


class CatalogResolver:
    """Uses curated, already pinned metadata offline; checks upstream only on opt-in."""

    def resolve(self, requested: str, *, offline: bool = True) -> ModelPin:
        """Refuses unknown names and pin mismatches instead of falling back or upgrading."""
        from generalforecast.models.catalog import V1_POLICY, ModelRefError, parse

        text(requested)
        try:
            ref = parse(requested)
        except ModelRefError as error:
            code = {
                "UnknownModel": "ModelResolutionRequired",
                "UntrustedModel": "ModelResolutionRequired",
                "InvalidModel": "InvalidPackage",
            }.get(error.code, error.code)
            message = (
                "model is not in the pinned offline catalog; supply an explicit capability resolver"
                if code == "ModelResolutionRequired"
                else str(error)
            )
            raise PackageError(code, message) from None
        entry = ref.entry
        if not entry.gates.allows("package"):
            raise PackageError(
                "ModelResolutionRequired",
                f"{entry.name} is not qualified for packages: {entry.gates.evidence}",
            )
        ensure(
            entry.gates.allows("package", variant=ref.variant),
            f"{ref} variants are not qualified for packages",
            "UnsupportedCapability",
        )
        refused = V1_POLICY.admits(ref)
        ensure(refused is None, f"serving policy allows {refused}", "CapacityExceeded")
        result = project(entry, ref, V1_POLICY)
        if not offline:
            from huggingface_hub import HfApi

            try:
                info = HfApi().model_info(result.repository, revision=result.revision)
            except Exception:  # noqa: BLE001
                raise PackageError(
                    "ModelUnavailable", "online model metadata qualification failed"
                ) from None
            ensure(
                info.sha == result.revision,
                "upstream model revision differs from pin",
                "ModelMismatch",
            )
        return ModelPin.from_dict(result.to_dict())


def trusted(
    spec: str,
    task: str,
    *,
    max_context_rows: int,
    max_features: int,
    max_horizon: int,
    quantiles: tuple[float, ...],
) -> ModelPin:
    """A trusted `module:function` model, bounded by what its local caller asks.

    It has no catalog files or published bounds: the caller's context, inputs,
    horizon and levels are its limits, and the workers that import it check
    the rest.
    """
    return ModelPin(
        name=spec,
        repository="",
        revision="",
        digest="",
        files=(),
        tasks=(task,),
        outputs=("value", "quantiles"),
        input_types=("int64", "float64"),
        missing_values=True,
        max_context_rows=max_context_rows,
        max_features=max_features,
        max_horizon=max_horizon,
        quantiles=quantiles,
        past_inputs=True,
        future_inputs=True,
    )
