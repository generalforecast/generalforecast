"""Strict Python serving intent, independent of catalog capabilities and execution."""

from dataclasses import dataclass
from typing import Any, Literal

from generalforecast.packages.validation import (
    PackageError,
    ensure,
    fields,
    name,
    text,
)
from generalforecast.python_packages.api import TableSchema
from generalforecast.python_packages.compiler import VERSION, PythonSpec


@dataclass(frozen=True)
class NoContext:
    """Python predictors have no training-context or live-dataset binding."""

    @property
    def mode(self) -> Literal["none"]:
        """Discriminate this from catalog request context."""
        return "none"

    def to_dict(self) -> dict:
        """Return the only supported Python context declaration."""
        return {"source": "none"}


@dataclass(frozen=True)
class PythonPackageSpec:
    """Strict authored Python serving intent, without fabricated catalog metadata."""

    name: str
    python: PythonSpec

    @property
    def task(self) -> str:
        """Return the explicitly declared predictor task."""
        return self.python.task

    @property
    def context(self) -> NoContext:
        """Return the distinct no-context contract."""
        return NoContext()

    @property
    def cpu(self) -> int:
        """Match the isolated worker's fixed admission policy."""
        return 1

    @property
    def memory(self) -> str:
        """Match the isolated worker's fixed memory limit."""
        return "256Mi"

    @property
    def min_replicas(self) -> int:
        """Do not claim autoscaling or scale-to-zero support."""
        return 1

    @property
    def max_replicas(self) -> int:
        """Admit exactly one isolated worker."""
        return 1

    @classmethod
    def from_dict(cls, value: Any) -> "PythonPackageSpec":
        """Parse explicit files and scalar schemas without importing customer code."""
        fields(
            value,
            {"name", "model", "task", "context"},
            {"package_version", "compute", "scaling"},
        )
        ensure(
            type(value.get("package_version", 1)) is int
            and value.get("package_version", 1) == 1,
            "unsupported package version",
        )
        model = fields(
            value["model"],
            {"kind", "code", "dependencies", "artifacts", "inputs", "outputs"},
        )
        ensure(model["kind"] == "python", "unsupported package model kind")
        context = fields(value["context"], {"source"})
        ensure(context["source"] == "none", "Python packages require no context")
        compute = fields(
            value.get("compute", {"cpu": 1, "memory": "256Mi"}), {"cpu", "memory"}
        )
        ensure(
            type(compute["cpu"]) is int and compute == {"cpu": 1, "memory": "256Mi"},
            "Python worker requires cpu=1 and memory=256Mi",
            "UnsupportedCapability",
        )
        scaling = fields(
            value.get("scaling", {"min_replicas": 1, "max_replicas": 1}),
            {"min_replicas", "max_replicas"},
        )
        ensure(
            all(type(item) is int and item == 1 for item in scaling.values()),
            "Python worker requires exactly one replica",
            "UnsupportedCapability",
        )
        try:
            python = PythonSpec.from_dict(
                {
                    **{key: child for key, child in model.items() if key != "kind"},
                    "version": VERSION,
                    "task": value["task"],
                }
            )
        except PackageError:
            raise
        except (ValueError, TypeError, KeyError):
            raise PackageError(
                "InvalidPackage", "invalid Python table schema"
            ) from None
        ensure("probe.json" in python.artifacts, "probe.json is a required artifact")
        ensure(
            all(
                not any(c in path for c in "*?[]")
                for path in python.code + python.dependencies + python.artifacts
            ),
            "file allowlists cannot contain globs",
        )
        for field in python.inputs.fields + python.outputs.fields:
            text(field.name)
        return cls(name(value["name"]), python)

    def to_dict(self) -> dict:
        """Normalize intent while retaining ordered scalar fields and explicit files."""
        model = self.python.to_dict()
        model.pop("version")
        model.pop("task", None)
        return {
            "package_version": 1,
            "name": self.name,
            "task": self.task,
            "model": {"kind": "python", **model},
            "context": self.context.to_dict(),
            "compute": {"cpu": self.cpu, "memory": self.memory},
            "scaling": {
                "min_replicas": self.min_replicas,
                "max_replicas": self.max_replicas,
            },
        }


@dataclass(frozen=True)
class PythonSchema:
    """Exact input and output scalar schemas, not inferred training metadata."""

    inputs: TableSchema
    outputs: TableSchema

    def to_dict(self) -> dict:
        """Return independent ordered scalar field declarations."""
        return {"inputs": self.inputs.to_dict(), "outputs": self.outputs.to_dict()}
