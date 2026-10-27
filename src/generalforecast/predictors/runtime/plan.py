# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The execution plan: what the runtime binds, derived from a lock, never parsed from YAML.

`gf.yaml` is parsed once, by `generalforecast.predictors.spec`; `gf check`
resolves it into `gf.lock`. `ExecutionPlan.from_lock` reads the plan the
runtime compiles from that lock, so `gf.Predictor`, `gf serve` and hosted
serving bind the same plan from the same document. The estimators build
plans directly for their own fitted contexts.
"""

from dataclasses import dataclass, field, replace
from typing import Any

from generalforecast.predictors import frequency as _frequency
from generalforecast.predictors.selection import SelectionInfo
from generalforecast.predictors.validation import (
    PredictorError,
    ensure,
    name,
    text,
)
from generalforecast.predictors.validation import column as column_name

TASKS = frozenset({"classification", "regression", "forecast"})
CONTEXT_MODES = ("snapshot", "request", "live")
OUTPUTS = {
    "classification": ("label", "probabilities"),
    "regression": ("value", "quantiles"),
    "forecast": ("value", "quantiles"),
}


@dataclass(frozen=True)
class PlanContext:
    """Where the plan's context rows come from.

    Attributes:
        mode: `snapshot`, `request` or `live`.
        name: A live context's name.
        max_rows: The most context rows one request may carry.
    """

    mode: str = "request"
    name: str | None = None
    max_rows: int | None = None

    def __post_init__(self) -> None:
        """Checks the mode and its fields."""
        ensure(self.mode in CONTEXT_MODES, "unsupported context mode")
        ensure(
            (self.mode == "live") == (self.name is not None),
            "a live context, and only one, is named",
        )
        if self.name is not None:
            name(self.name)
        ensure(
            self.max_rows is None or (type(self.max_rows) is int and self.max_rows > 0),
            "max_rows must be a positive integer",
        )
        ensure(
            self.max_rows is None or self.mode == "request",
            "max_rows bounds request contexts only",
        )

    def to_dict(self) -> dict:
        """The context as the plan digest hashes it."""
        return {"mode": self.mode, "name": self.name, "max_rows": self.max_rows}


@dataclass(frozen=True)
class PlanInputs:
    """A tabular plan's input columns and their declared kinds.

    Attributes:
        include: The input columns in order; None takes every column without
            a role, less `exclude`.
        exclude: Columns left out of the inferred inputs.
        types: Each column's kind, in the vocabulary of
            `generalforecast.predictors.tables`, as the lock records it.
        missing_values: `allow` or `reject` missing input values.
    """

    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    types: tuple[tuple[str, str], ...] = ()
    missing_values: str = "allow"

    def __post_init__(self) -> None:
        """Rejects contradictory selections and unknown kinds."""
        from generalforecast.predictors.tables import PHYSICAL

        ensure(
            self.include is None or not self.exclude,
            "choose include or exclude, not both",
        )
        for column, kind in self.types:
            column_name(column)
            ensure(kind in PHYSICAL, "unsupported input type")
        ensure(
            self.missing_values in ("allow", "reject"),
            "unsupported missing-value policy",
        )

    def to_dict(self) -> dict:
        """The inputs as the plan digest hashes them."""
        return {
            "include": None if self.include is None else list(self.include),
            "exclude": list(self.exclude),
            "types": dict(self.types),
            "missing_values": self.missing_values,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """A normalized predictor intent, validated on construction.

    Attributes:
        name: The predictor's name.
        model: The catalog model, as `ModelPin.resolved` or a canonical id.
        task: `classification`, `regression` or `forecast`, or None to take
            the model's one task.
        target: The target column.
        target_type: The target's kind, when the lock knows it.
        id: The id column, which predictions echo.
        context: Where context rows come from.
        inputs: The tabular input columns.
        time: The time column.
        series: A forecaster's series columns.
        horizon: A forecaster's steps.
        frequency: A forecaster's normalized frequency.
        past_inputs: A forecaster's covariates known up to the origin.
        future_inputs: A forecaster's covariates each query brings ahead.
        quantiles: The output levels of a regressor or forecaster.
        outputs: The outputs predictions carry; the task's when empty.
        selection: The resolved context selection; exact when None.
        role_types: The kinds of the id, time and series columns.
        related: A forecaster's declared groups: None for independent series,
            `()` for every series as one group, else the series columns whose
            values name each group.
    """

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
    selection: SelectionInfo | None = None
    role_types: tuple[tuple[str, str], ...] = field(default=())
    related: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validates every field; no credentials, code or runtime binding."""
        name(self.name)
        text(self.model)
        ensure(self.task is None or self.task in TASKS, "unsupported task")
        for column in (self.target, self.id, self.time, *self.series):
            if column is not None:
                column_name(column)
        ensure(
            self.horizon is None
            or (type(self.horizon) is int and 0 < self.horizon <= 10_000),
            "horizon must be a positive integer",
        )
        levels = self.quantiles
        ensure(
            len(levels) <= 99
            and all(type(q) in (int, float) and 0 < q < 1 for q in levels),
            "quantiles must be inside (0, 1)",
        )
        ensure(
            list(levels) == sorted(set(levels)), "quantiles must be unique and sorted"
        )
        if self.frequency is not None:
            try:
                normalized = _frequency.normalize(self.frequency)
            except (TypeError, ValueError):
                raise PredictorError(
                    "ValidationError",
                    "frequency must be a positive period step or fixed duration",
                ) from None
            object.__setattr__(self, "frequency", normalized)
        roles = [
            column
            for column in (self.target, self.id, self.time, *self.series)
            if column is not None
        ]
        ensure(len(set(roles)) == len(roles), "column roles must be disjoint")
        ensure(
            not set(roles) & set(self.forecast_inputs),
            "forecast inputs cannot overlap target, ID, time or series roles",
        )
        ensure(
            not set(self.forecast_inputs) & set(self.inputs.include or ()),
            "a column is a tabular input or a forecast covariate, not both",
        )
        ensure(
            self.context.mode == "request" or self.target is not None,
            "snapshot and live context require an explicit target",
        )
        ensure(
            self.related is None or set(self.related) <= set(self.series),
            "related groups are named by series columns",
        )

    @classmethod
    def from_lock(cls, lock: Any) -> "ExecutionPlan":
        """The plan a `gf.lock` (version 2) describes.

        Args:
            lock: A `contracts.PredictorLock` or its document.

        Each input's role is the lock's: `input` for a tabular model, `past`
        or `future` for a forecaster, so no consumer re-derives roles.
        """
        document = lock.to_dict() if hasattr(lock, "to_dict") else lock
        predictor = document["predictor"]
        inputs = predictor["inputs"]
        context = predictor["context"]
        task = predictor["task"]
        role_types = [
            (column, kind)
            for column, kind in (
                (predictor.get("id"), predictor.get("id_type")),
                (predictor.get("time"), predictor.get("time_type")),
                *zip(
                    predictor.get("series") or (),
                    predictor.get("series_types") or (),
                    strict=False,
                ),
            )
            if column is not None and kind is not None
        ]
        return cls(
            name=predictor["name"],
            model=predictor["model"],
            task=task,
            target=predictor["target"],
            target_type=predictor.get("target_type"),
            id=predictor.get("id"),
            context=PlanContext(
                context["kind"],
                context.get("name") if context["kind"] == "live" else None,
                context.get("max_rows"),
            ),
            inputs=PlanInputs(
                include=tuple(
                    column["name"] for column in inputs if column["role"] == "input"
                ),
                types=tuple(
                    (column["name"], column["type"])
                    for column in inputs
                    if column.get("type") is not None
                ),
            ),
            time=predictor.get("time"),
            series=tuple(predictor.get("series") or ()),
            horizon=predictor.get("horizon"),
            frequency=predictor.get("frequency"),
            past_inputs=tuple(
                column["name"] for column in inputs if column["role"] == "past"
            ),
            future_inputs=tuple(
                column["name"] for column in inputs if column["role"] == "future"
            ),
            quantiles=tuple(float(q) for q in predictor.get("quantiles") or ()),
            outputs=tuple(predictor["outputs"]),
            selection=SelectionInfo.from_dict(predictor["selection"]),
            role_types=tuple(role_types),
            related=tuple(predictor["related"]) if "related" in predictor else None,
        )

    @classmethod
    def build(
        cls,
        name: str,
        model: str,
        *,
        task: str | None = None,
        target: str | None = None,
        mode: str = "snapshot",
        inputs: Any = None,
        time: str | None = None,
        series: Any = (),
        horizon: int | None = None,
        frequency: Any = None,
        past_inputs: Any = (),
        future_inputs: Any = (),
        quantiles: Any = (),
        outputs: Any = (),
    ) -> "ExecutionPlan":
        """A plan from Python values, as the estimators describe their fitted contexts.

        `inputs` is a list of tabular input columns, or None for every column
        without a role.
        """
        return cls(
            name=name,
            model=model,
            task=task,
            target=target,
            context=PlanContext(mode),
            inputs=PlanInputs(include=None if inputs is None else tuple(inputs)),
            time=time,
            series=tuple(series),
            horizon=horizon,
            frequency=frequency,
            past_inputs=tuple(past_inputs),
            future_inputs=tuple(future_inputs),
            quantiles=tuple(float(q) for q in quantiles),
            outputs=tuple(outputs),
        )

    @property
    def forecast_inputs(self) -> tuple[str, ...]:
        """Lists each forecast channel once, in past then future declaration order."""
        return tuple(dict.fromkeys((*self.past_inputs, *self.future_inputs)))

    def with_outputs(self, task: str) -> "ExecutionPlan":
        """The plan with its task's outputs when it names none."""
        return self if self.outputs else replace(self, outputs=OUTPUTS[task])

    def to_dict(self) -> dict:
        """A fresh normalized plan, the content its digest hashes."""
        return {
            "plan_version": 2,
            "name": self.name,
            "model": self.model,
            "task": self.task,
            "target": self.target,
            "target_type": self.target_type,
            "id": self.id,
            "context": self.context.to_dict(),
            "inputs": self.inputs.to_dict(),
            "time": self.time,
            "series": list(self.series),
            "horizon": self.horizon,
            "frequency": self.frequency,
            "past_inputs": list(self.past_inputs),
            "future_inputs": list(self.future_inputs),
            "quantiles": list(self.quantiles),
            "outputs": list(self.outputs),
            "selection": None if self.selection is None else self.selection.to_dict(),
            "role_types": dict(self.role_types),
            "related": None if self.related is None else list(self.related),
        }


__all__ = ["OUTPUTS", "ExecutionPlan", "PlanContext", "PlanInputs"]
