# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Immutable authored intent, independent of runtime, credentials and environment."""

import re
from dataclasses import dataclass
from typing import Any

from generalforecast.predictors import frequency as _frequency
from generalforecast.predictors.files import relative
from generalforecast.predictors.validation import (
    PredictorError,
    ensure,
    fields,
    integer,
    name,
    names,
    text,
)
from generalforecast.tabular import DECLARED_TYPES

TASKS = frozenset({"classification", "regression", "forecast"})


@dataclass(frozen=True)
class PlanContext:
    """Request, immutable snapshot or live context binding."""

    mode: str = "request"
    source: str | None = None
    order_by: str | None = None
    selection: str = "all"
    max_rows: int | None = None

    @classmethod
    def from_dict(cls, value: Any = None) -> "PlanContext":
        """Normalizes short context syntax without fetching a source."""
        if value is None:
            return cls()
        if type(value) is str:
            value = {"source": value}
        fields(value, {"source"}, {"order_by", "selection", "max_rows"})
        source = text(value["source"])
        if source == "request":
            ensure(len(value) == 1, "request context has no selection policy")
            return cls()
        if source.startswith("live/"):
            name(source.removeprefix("live/"))
            mode = "live"
        else:
            source = relative(source)
            mode = "snapshot"
        order = text(value["order_by"]) if "order_by" in value else None
        selection = value.get("selection", "all")
        ensure(selection in ("all", "most_recent"), "unsupported context selection")
        maximum = integer(value["max_rows"]) if "max_rows" in value else None
        ensure(
            (selection == "all" and maximum is None)
            or (
                selection == "most_recent" and order is not None and maximum is not None
            ),
            "bounded context requires explicit most_recent ordering and max_rows",
        )
        return cls(mode, source, order, selection, maximum)

    def to_dict(self) -> dict:
        """Serializes normalized portable source intent."""
        if self.mode == "request":
            return {"source": "request"}
        result = {"source": self.source or "request", "selection": self.selection}
        if self.order_by is not None:
            result["order_by"] = self.order_by
        if self.max_rows is not None:
            result["max_rows"] = self.max_rows
        return result


@dataclass(frozen=True)
class PlanInputs:
    """Feature selection and explicit semantic overrides, not semantic guesses."""

    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    types: tuple[tuple[str, str], ...] = ()
    missing_values: str = "allow"

    @classmethod
    def from_dict(cls, value: Any) -> "PlanInputs":
        """Rejects contradictory selections and unknown physical or semantic types."""
        fields(value, set(), {"include", "exclude", "types", "missing_values"})
        ensure(
            not ("include" in value and "exclude" in value),
            "choose include or exclude, not both",
        )
        types = value.get("types", {})
        ensure(type(types) is dict, "input types must be an object")
        for column, kind in types.items():
            text(column)
            ensure(
                type(kind) is str and kind in DECLARED_TYPES, "unsupported input type"
            )
        missing = value.get("missing_values", "allow")
        ensure(missing in ("allow", "reject"), "unsupported missing-value policy")
        return cls(
            names(value["include"]) if "include" in value else None,
            names(value.get("exclude", [])),
            tuple(sorted(types.items())),
            missing,
        )

    def to_dict(self) -> dict:
        """Copies the feature contract into JSON-compatible values."""
        result: dict = {
            "types": dict(self.types),
            "missing_values": self.missing_values,
        }
        if self.include is not None:
            result["include"] = list(self.include)
        elif self.exclude:
            result["exclude"] = list(self.exclude)
        return result


@dataclass(frozen=True)
class ExecutionPlan:
    """A normalized functional predictor intent; build via from_dict for validation."""

    name: str
    model: str
    task: str | None = None
    target: str | None = None
    target_type: str | None = None
    id: str | None = None
    context: PlanContext = PlanContext()
    inputs: PlanInputs = PlanInputs()
    time: str | None = None
    series: tuple[str, ...] = ()
    horizon: int | None = None
    frequency: str | int | float | None = None
    past_inputs: tuple[str, ...] = ()
    future_inputs: tuple[str, ...] = ()
    quantiles: tuple[float, ...] = ()
    outputs: tuple[str, ...] = ()
    cpu: int | None = None
    memory: str | None = None
    min_replicas: int = 1
    max_replicas: int = 1
    region: str | None = None
    residency: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Any) -> "ExecutionPlan":
        """Validates every level with no credentials, executable code or runtime binding."""
        fields(
            value,
            {"name", "model"},
            {
                "plan_version",
                "task",
                "target",
                "id",
                "context",
                "inputs",
                "time",
                "series",
                "horizon",
                "frequency",
                "past_inputs",
                "future_inputs",
                "quantiles",
                "outputs",
                "compute",
                "scaling",
                "placement",
            },
        )
        ensure(
            type(value.get("plan_version", 1)) is int
            and value.get("plan_version", 1) == 1,
            "unsupported plan version",
        )
        task = value.get("task")
        ensure(
            task is None or (type(task) is str and task in TASKS), "unsupported task"
        )
        target = value.get("target")
        target_type = None
        if type(target) is dict:
            fields(target, {"column"}, {"type"})
            target_type = target.get("type")
            ensure(
                target_type is None
                or (type(target_type) is str and target_type in DECLARED_TYPES),
                "unsupported target type",
            )
            target = target["column"]
        if target is not None:
            text(target)
        compute = fields(value.get("compute", {}), set(), {"cpu", "memory"})
        cpu = integer(compute["cpu"], 1, 256) if "cpu" in compute else None
        memory = compute.get("memory")
        ensure(
            memory is None
            or (type(memory) is str and re.fullmatch(r"[1-9][0-9]*(?:Mi|Gi)", memory)),
            "memory must be a positive Mi or Gi quantity",
        )
        scaling = fields(
            value.get("scaling", {}), set(), {"min_replicas", "max_replicas"}
        )
        minimum = integer(scaling.get("min_replicas", 1), 0, 100)
        maximum = integer(scaling.get("max_replicas", 1), 1, 100)
        ensure(minimum <= maximum, "min_replicas exceeds max_replicas")
        placement = fields(value.get("placement", {}), set(), {"region", "residency"})
        region = text(placement["region"]) if "region" in placement else None
        residency = names(placement.get("residency", []))
        ensure(
            not residency or region is None or region in residency,
            "placement region conflicts with residency",
        )
        levels = value.get("quantiles", [])
        ensure(
            type(levels) is list
            and len(levels) <= 99
            and all(type(q) in (int, float) and 0 < q < 1 for q in levels),
            "quantiles must be inside (0, 1)",
        )
        ensure(levels == sorted(set(levels)), "quantiles must be unique and sorted")
        frequency = value.get("frequency")
        if frequency is not None:
            ensure(
                type(frequency) in (float, int, str)
                and (type(frequency) is str or 0 < frequency < 1e12),
                "frequency must be a positive period step or fixed duration",
            )
            try:
                frequency = _frequency.normalize(frequency)
            except ValueError:
                raise PredictorError(
                    "ValidationError",
                    "frequency must be a positive period step or fixed duration",
                ) from None
            value = {**value, "frequency": frequency}
        result = cls(
            name(value["name"]),
            text(value["model"]),
            task,
            target,
            target_type,
            text(value["id"]) if "id" in value else None,
            PlanContext.from_dict(value.get("context")),
            PlanInputs.from_dict(value.get("inputs", {})),
            text(value["time"]) if "time" in value else None,
            names(value.get("series", [])),
            integer(value["horizon"]) if "horizon" in value else None,
            frequency,
            names(value.get("past_inputs", [])),
            names(value.get("future_inputs", [])),
            tuple(float(q) for q in levels),
            names(value.get("outputs", [])),
            cpu,
            memory,
            minimum,
            maximum,
            region,
            residency,
        )
        roles = [
            column
            for column in (
                result.target,
                result.id,
                result.time,
                *result.series,
            )
            if column is not None
        ]
        ensure(len(set(roles)) == len(roles), "column roles must be disjoint")
        ensure(
            not set(roles) & set(result.forecast_inputs),
            "forecast inputs cannot overlap target, ID, time or series roles",
        )
        ensure(
            result.context.mode == "request" or result.target is not None,
            "snapshot and live context require an explicit target",
        )
        return result

    @property
    def forecast_inputs(self) -> tuple[str, ...]:
        """Lists each forecast channel once, in past then future declaration order."""
        return tuple(dict.fromkeys((*self.past_inputs, *self.future_inputs)))

    def to_dict(self) -> dict:
        """Returns a fresh normalized manifest, suitable for stable content hashing."""
        result: dict = {
            "plan_version": 1,
            "name": self.name,
            "model": self.model,
            "context": self.context.to_dict(),
            "inputs": self.inputs.to_dict(),
            "scaling": {
                "min_replicas": self.min_replicas,
                "max_replicas": self.max_replicas,
            },
        }
        for key in ("task", "id", "time", "horizon", "frequency"):
            if (value := getattr(self, key)) is not None:
                result[key] = value
        if self.target is not None:
            result["target"] = {
                "column": self.target,
                **({"type": self.target_type} if self.target_type else {}),
            }
        for key in ("series", "past_inputs", "future_inputs", "quantiles", "outputs"):
            if value := getattr(self, key):
                result[key] = list(value)
        if self.cpu is not None or self.memory is not None:
            result["compute"] = {
                key: value
                for key, value in {"cpu": self.cpu, "memory": self.memory}.items()
                if value is not None
            }
        if self.region is not None or self.residency:
            result["placement"] = {
                **({"region": self.region} if self.region else {}),
                **({"residency": list(self.residency)} if self.residency else {}),
            }
        return result


def parse_plan(value: Any) -> "ExecutionPlan":
    """Parses an authored catalog spec; a mapping `model` is not a catalog model."""
    if type(value) is dict and type(value.get("model")) is dict:
        raise PredictorError(
            "InvalidPredictor",
            "model must be a catalog model id such as general-forecast/x",
        )
    return ExecutionPlan.from_dict(value)
