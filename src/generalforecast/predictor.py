# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`generalforecast.Predictor`: one object for local, ad hoc, served and hosted predictors.

A predictor is a catalog model, a target, inputs and a context. This module
resolves a reference to one, builds the request every path validates the
same way, runs it in process through the engine or over HTTP, and scores a
held-out split.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as _compute

from generalforecast import tabular
from generalforecast.errors import (
    AmbiguousReference,
    Error,
    Internal,
    InvalidInput,
    NotFound,
    boundary,
    from_wire,
)
from generalforecast.native import models as registry

pc: Any = _compute
ENVIRONMENTS = ("staging", "production")
NAME = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_URL = re.compile(r"^https?://", re.IGNORECASE)
_HOLDOUT_FRACTION = 0.2


def _records(table: pa.Table) -> list[dict[str, Any]]:
    rows = table.to_pylist()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                row[key] = None
            elif hasattr(value, "isoformat"):
                row[key] = value.isoformat()
            elif isinstance(value, bytes):
                row[key] = value.decode("utf-8", "replace")
    return rows


def _did_you_mean(name: str, names: Iterable[str]) -> str | None:
    import difflib

    close = difflib.get_close_matches(name, list(names), n=1, cutoff=0.6)
    return close[0] if close else None


def _empty(column: pa.ChunkedArray) -> pa.ChunkedArray:
    """A mask of the rows whose target is empty: null, NaN or blank text."""
    mask = pc.is_null(column, nan_is_null=True)
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        blank = pc.equal(pc.utf8_trim_whitespace(pc.fill_null(column, "")), "")
        mask = pc.or_(mask, blank)
    return mask


@dataclass(frozen=True)
class PredictRequest:
    """A validated prediction request, the same for every path.

    Built once from the caller's rows and settings, then run in process or
    sent as the ad hoc payload; the checks run here so a mistake reads the
    same wherever it is made.
    """

    model: str
    task: str
    target: str
    context: pa.Table
    queries: pa.Table
    inputs: tuple[str, ...]
    id: str | None = None
    time: str | None = None
    series: tuple[str, ...] = ()
    frequency: Any = None
    horizon: int | None = None
    quantiles: tuple[float, ...] | None = None
    outputs: tuple[str, ...] | None = None
    selection: Mapping[str, Any] | None = None
    as_of: Any = None
    classes: tuple[Any, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        model: str,
        target: str,
        context: Any,
        queries: Any,
        inputs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        id: str | None = None,
        time: str | None = None,
        series: Sequence[str] | str | None = None,
        frequency: Any = None,
        horizon: int | None = None,
        quantiles: Sequence[float] | None = None,
        outputs: Sequence[str] | None = None,
        selection: Mapping[str, Any] | None = None,
        as_of: Any = None,
    ) -> PredictRequest:
        """Checks the settings against the rows and the model's limits.

        Raises:
            InvalidInput: For a missing target or input column, no inputs
                left, or a forecaster without a time column.
            CapacityExceeded: For more queries, context rows or inputs than
                the model takes in one call.
        """
        canonical = registry.canonical(model)
        task = registry.task_of(canonical)
        if not isinstance(target, str) or not target:
            raise InvalidInput("target is required", param="target")
        if context is None:
            raise InvalidInput(
                "context is required",
                param="context",
                hint="pass the rows the model learns from",
            )
        context_table = tabular.frame(context)
        query_table = tabular.frame(queries)
        if target not in context_table.column_names:
            near = _did_you_mean(target, context_table.column_names)
            raise InvalidInput(
                f"target column `{target}` is not in the context",
                param="target",
                hint=f"did you mean `{near}`?"
                if near
                else "the context has " + ", ".join(context_table.column_names),
            )
        series_names = (
            (series,)
            if isinstance(series, str)
            else tuple(series)
            if series is not None
            else ()
        )
        roles = {target, *(name for name in (id, time) if name), *series_names}
        for role, name in (("id", id), ("time", time)):
            if name is not None and name not in context_table.column_names:
                raise InvalidInput(
                    f"{role} column `{name}` is not in the context",
                    param=role,
                    hint="the context has " + ", ".join(context_table.column_names),
                )
        for name in series_names:
            if name not in context_table.column_names:
                raise InvalidInput(
                    f"series column `{name}` is not in the context",
                    param="series",
                    hint="the context has " + ", ".join(context_table.column_names),
                )
        if inputs is not None:
            if isinstance(inputs, str):
                raise InvalidInput(
                    "inputs must be a list of column names", param="inputs"
                )
            if exclude:
                raise InvalidInput(
                    "give either `inputs` or `exclude`, not both",
                    param="inputs",
                    hint="a list of inputs already replaces inference",
                )
            chosen = tuple(inputs)
            for name in chosen:
                if name not in context_table.column_names:
                    raise InvalidInput(
                        f"input column `{name}` is not in the context",
                        param=f"inputs.{name}",
                        hint="the context has " + ", ".join(context_table.column_names),
                    )
                if name in roles:
                    raise InvalidInput(
                        f"input column `{name}` is the target or has a role",
                        param=f"inputs.{name}",
                    )
        else:
            left = set(exclude)
            chosen = tuple(
                name
                for name in context_table.column_names
                if name not in roles and name not in left
            )
        if task == "forecast":
            if time is None:
                raise InvalidInput(
                    f"{canonical} forecasts series and no `time` column is given",
                    param="time",
                    hint="pass time=<column> and frequency, or pick a tabular model",
                )
        elif not chosen:
            raise InvalidInput(
                "no input columns are left after the target and roles",
                param="inputs",
                hint="the context needs at least one input column",
            )
        for name in chosen:
            if name not in query_table.column_names:
                raise InvalidInput(
                    f"input column `{name}` is not in the queries",
                    param=f"inputs.{name}",
                    hint="the queries have " + ", ".join(query_table.column_names),
                )
        limits = registry.limits(canonical)
        if task != "forecast" and len(query_table) > limits.queries:
            raise _capacity("queries", canonical, limits.queries, len(query_table))
        if (
            task != "forecast"
            and len(context_table) > limits.context_rows
            and not selection
        ):
            raise _capacity(
                "context", canonical, limits.context_rows, len(context_table)
            )
        if len(chosen) > limits.inputs:
            raise _capacity("inputs", canonical, limits.inputs, len(chosen))
        classes: tuple[Any, ...] = ()
        if task == "classification":
            from generalforecast.predictors.tables import whole_numbers

            context_table = _whole_target(context_table, target, whole_numbers)
            query_table = _whole_target(query_table, target, whole_numbers)
            column = context_table.column(target)
            labels = column.filter(pc.invert(_empty(column))).to_pylist()
            classes = tuple(
                sorted(set(labels), key=lambda item: (str(type(item)), item))
            )
            if len(classes) < 2:
                raise InvalidInput(
                    f"target column `{target}` has {len(classes)} class in the "
                    "context; a classifier needs at least two",
                    param="target",
                )
        if quantiles is not None:
            levels = tuple(float(level) for level in quantiles)
            if any(not 0 < level < 1 for level in levels):
                raise InvalidInput(
                    "quantiles must lie strictly between 0 and 1", param="quantiles"
                )
        else:
            levels = None
        if horizon is not None and (not isinstance(horizon, int) or horizon <= 0):
            raise InvalidInput("horizon must be a positive integer", param="horizon")
        return cls(
            model=canonical,
            task=task,
            target=target,
            context=context_table,
            queries=query_table,
            inputs=chosen,
            id=id,
            time=time,
            series=series_names,
            frequency=frequency,
            horizon=horizon,
            quantiles=levels,
            outputs=tuple(outputs) if outputs is not None else None,
            selection=dict(selection) if selection else None,
            as_of=as_of,
            classes=classes,
        )

    def payload(self) -> dict[str, Any]:
        """The ad hoc request payload of section 5."""
        payload: dict[str, Any] = {
            "model": self.model,
            "target": self.target,
            "context": _records(self.context),
            "queries": _records(self.queries),
            "inputs": list(self.inputs),
        }
        for key in ("id", "time", "frequency", "horizon", "as_of"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = (
                    value if not hasattr(value, "isoformat") else value.isoformat()
                )
        if self.series:
            payload["series"] = list(self.series)
        if self.quantiles is not None:
            payload["quantiles"] = list(self.quantiles)
        if self.outputs is not None:
            payload["outputs"] = list(self.outputs)
        if self.selection:
            payload["selection"] = dict(self.selection)
        return payload

    @staticmethod
    def declared_payload(
        queries: Any, *, context: Any = None, as_of: Any = None
    ) -> dict[str, Any]:
        """The declared-predictor payload: queries, and context or as_of if given."""
        payload: dict[str, Any] = {"queries": _records(tabular.frame(queries))}
        if context is not None:
            payload["context"] = _records(tabular.frame(context))
        if as_of is not None:
            payload["as_of"] = (
                as_of if not hasattr(as_of, "isoformat") else as_of.isoformat()
            )
        return payload


def _whole_target(table: pa.Table, target: str, whole: Any) -> pa.Table:
    """The table with a float target that holds only whole numbers cast to int64."""
    if target not in table.column_names:
        return table
    cast = whole(table.column(target))
    if cast is None:
        return table
    return table.set_column(table.schema.get_field_index(target), target, cast)


def _capacity(what: str, model: str, limit: int, actual: int) -> Error:
    from generalforecast.errors import CapacityExceeded

    unit = {
        "queries": ("accepts", "queries per call"),
        "context": ("reads", "context rows"),
        "inputs": ("reads", "inputs"),
    }[what]
    hints = {
        "queries": "split the queries, or push a predictor and call it in batches",
        "context": "pass a selection with core and neighbors, or push a predictor",
        "inputs": "list the inputs the model should read, or exclude columns",
    }
    return CapacityExceeded(
        f"{what} exceed the limit for {model}"
        if what != "context"
        else f"context exceeds the limit for {model}",
        because=f"this model {unit[0]} {limit:,} {unit[1]}; the request sent {actual:,}",
        hint=hints[what],
        limit=limit,
        actual=actual,
        param=what,
    )


def run_local(request: PredictRequest, *, download: bool = True) -> Any:
    """Runs a request in this process's engine and returns the result frame."""
    from generalforecast.native import tables

    name = f"predictor-{uuid.uuid4().hex}"
    if request.task == "forecast":
        schema = tables.SeriesSchema(
            target=request.target,
            time=request.time or "time",
            series=request.series or _single_series(request),
            frequency=request.frequency or _infer_frequency(request),
            known=request.inputs,
        )
        series_table = tables.SeriesTable(
            name,
            model=request.model,
            schema=schema,
            horizon=request.horizon,
            quantiles=request.quantiles,
            download=download,
        )
        try:
            series_table.insert(request.context)
            return series_table.predict(request.queries, request.as_of)
        finally:
            series_table.drop()
    limits = registry.limits(request.model)
    selection = tables.retrieved_of(
        request.selection,
        limits.context_rows,
        timed=request.time is not None,
        rows=len(request.context),
    )
    table = tables.Table(
        name,
        model=request.model,
        schema=tables.Schema(
            target=request.target,
            id=request.id,
            time=request.time,
            inputs=request.inputs,
        ),
        selection=selection,
        download=download,
    )
    try:
        table.insert(request.context)
        frame = table.predict(
            request.queries, request.as_of, quantiles=request.quantiles
        )
    finally:
        table.drop()
    if request.id is not None and request.id in frame.columns:
        frame = frame.rename(columns={request.id: "id"})
    return frame


def _single_series(request: PredictRequest) -> tuple[str, ...]:
    raise InvalidInput(
        f"{request.model} forecasts series and no `series` column is given",
        param="series",
        hint="pass series=[<column>] naming each series",
    )


def _infer_frequency(request: PredictRequest) -> Any:
    assert request.time is not None
    times = request.context.column(request.time)
    values = pc.unique(times).sort().to_pylist()
    if len(values) < 2:
        raise InvalidInput(
            "frequency is required: the context has too few times to infer it",
            param="frequency",
        )
    step = values[1] - values[0]
    return step.total_seconds() if hasattr(step, "total_seconds") else step


@dataclass(frozen=True)
class Evaluation:
    """Scores of one or more models on the same held-out rows.

    Attributes:
        metrics: `{model_id: {metric: value}}`.
        rows: The held-out rows: the id (if any), the target and one
            prediction column per model.
    """

    metrics: dict[str, dict[str, float]]
    rows: Any

    def __str__(self) -> str:
        """The metric table as `gf eval` prints it."""
        models = list(self.metrics)
        names = list(
            dict.fromkeys(key for item in self.metrics.values() for key in item)
        )
        width = max((len(name) for name in names), default=0)
        heads = [registry.short(model) for model in models]
        many = len(models) > 1
        widths = [max(len(head), 5) if many else 5 for head in heads]
        lines = []
        if many:
            lines.append(
                " " * 13
                + " " * width
                + "   "
                + "   ".join(
                    head.rjust(w) for head, w in zip(heads, widths, strict=True)
                )
            )
        for name in names:
            cells = []
            for model, w in zip(models, widths, strict=True):
                value = self.metrics[model].get(name)
                cells.append(("" if value is None else f"{value:.3f}").rjust(w))
            lines.append(" " * 13 + name.ljust(width) + "   " + "   ".join(cells))
        return "\n".join(lines)


def classification_metrics(
    truth: Sequence[Any],
    labels: Sequence[Any],
    probabilities: Mapping[Any, Sequence[float]],
) -> dict[str, float]:
    """`accuracy`, `log loss`, and `auroc` (binary) or `macro f1` (multiclass)."""
    classes = list(probabilities)
    y = list(truth)
    n = len(y)
    if n == 0:
        raise InvalidInput("no rows to evaluate")
    accuracy = sum(1 for a, b in zip(y, labels, strict=True) if a == b) / n
    eps = 1e-15
    losses = []
    for index, value in enumerate(y):
        p = probabilities[value][index] if value in probabilities else 0.0
        losses.append(-math.log(min(max(p, eps), 1 - eps)))
    out = {"accuracy": accuracy, "log loss": sum(losses) / n}
    if len(classes) == 2:
        positive = classes[1]
        scores = list(probabilities[positive])
        out["auroc"] = _auroc([1 if value == positive else 0 for value in y], scores)
    else:
        f1s = []
        for label in classes:
            tp = sum(
                1 for a, b in zip(y, labels, strict=True) if a == label and b == label
            )
            fp = sum(
                1 for a, b in zip(y, labels, strict=True) if a != label and b == label
            )
            fn = sum(
                1 for a, b in zip(y, labels, strict=True) if a == label and b != label
            )
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        out["macro f1"] = sum(f1s) / len(f1s)
    return out


def _auroc(truth: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(truth)
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    positive_ranks = sum(
        rank for rank, label in zip(ranks, truth, strict=True) if label
    )
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def regression_metrics(
    truth: Sequence[float],
    values: Sequence[float],
    quantiles: Mapping[float, Sequence[float]] | None = None,
) -> dict[str, float]:
    """`mae`, `rmse` and `pinball q` per quantile."""
    y = np.asarray(truth, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(y) == 0:
        raise InvalidInput("no rows to evaluate")
    out = {
        "mae": float(np.mean(np.abs(y - v))),
        "rmse": float(math.sqrt(np.mean((y - v) ** 2))),
    }
    for level, predicted in (quantiles or {}).items():
        out[f"pinball {level:g}"] = _pinball(
            y, np.asarray(predicted, dtype=float), level
        )
    return out


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    diff = y - q
    return float(np.mean(np.where(diff >= 0, level * diff, (level - 1) * diff)))


def forecast_metrics(
    truth: Sequence[float],
    quantiles: Mapping[float, Sequence[float]],
    *,
    history: Mapping[Any, Sequence[float]],
    series: Sequence[Any],
) -> dict[str, float]:
    """`mase` (scaled by each series' naive in-sample error) and `wql`."""
    y = np.asarray(truth, dtype=float)
    levels = sorted(quantiles)
    median = min(levels, key=lambda level: abs(level - 0.5))
    point = np.asarray(quantiles[median], dtype=float)
    scales = {}
    for key, values in history.items():
        past = np.asarray(values, dtype=float)
        scale = np.mean(np.abs(np.diff(past))) if len(past) > 1 else float("nan")
        scales[key] = scale if scale and not math.isnan(scale) else float("nan")
    errors = []
    for value, predicted, key in zip(y, point, series, strict=True):
        scale = scales.get(key, float("nan"))
        if scale and not math.isnan(scale):
            errors.append(abs(value - predicted) / scale)
    mase = float(np.mean(errors)) if errors else float("nan")
    total = float(np.sum(np.abs(y)))
    losses = sum(
        2 * _pinball(y, np.asarray(quantiles[level], dtype=float), level) * len(y)
        for level in levels
    ) / len(levels)
    wql = float(losses / total) if total else float("nan")
    return {"mase": mase, "wql": wql}


@dataclass(frozen=True)
class Context:
    """What a predictor learns from: its kind and how many rows it holds."""

    kind: str
    rows: int | None = None
    name: str | None = None
    at: Any = None
    _owner: Any = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:  # noqa: D105
        rows = "" if self.rows is None else f" {self.rows:,} rows"
        return f"Context({self.kind}{rows})"

    def _live(self) -> Any:
        if self.kind != "live" or self._owner is None:
            raise InvalidInput(
                f"a {self.kind} context cannot change",
                hint="only a live context takes append, label and as_of",
            )
        return self._owner

    def append(self, rows: Any) -> Any:
        """Appends rows to a live context."""
        owner = self._live()
        with boundary():
            return owner._client().contexts.append(self.name, tabular.frame(rows))

    def label(self, ids: Iterable[Any], values: Iterable[Any]) -> Any:
        """Sets the target of rows in a live context, by id."""
        owner = self._live()
        ids, values = list(ids), list(values)
        if len(ids) != len(values):
            raise InvalidInput("ids and values must have the same length")
        with boundary():
            return owner._client().contexts.label(self.name, ids, values)

    def as_of(self, ts: Any) -> Predictor:
        """The same predictor reading the live context as of `ts`."""
        owner = self._live()
        return owner._with_as_of(ts)


@dataclass(frozen=True)
class Resolution:
    """The resolved schema of a predictor, as `gf check` prints it."""

    # contract: provided by session B (generalforecast.predictors.Resolution);
    # this is the SDK's own view when a predictor is ad hoc or hosted.
    name: str
    model: str
    task: str
    target: str
    target_type: str | None = None
    classes: tuple[Any, ...] = ()
    id: str | None = None
    time: str | None = None
    series: tuple[str, ...] = ()
    inputs: tuple[tuple[str, str], ...] = ()
    context: Context | None = None
    selection: Mapping[str, Any] | None = None
    inferred: Mapping[str, str] = field(default_factory=dict)


class Predictor:
    """A predictor by reference: a local directory, a hosted name, a catalog model or a URL.

    Args:
        ref: What to predict with. A path to a directory holding `gf.yaml`
            (anything with a path separator, starting with `.`, or naming an
            existing directory with `gf.yaml`) is local and runs in process.
            A catalog id such as `general-forecast/tabicl-v2-classifier` is
            an ad hoc predictor: give `target` and the context. A URL is a
            `gf serve` elsewhere. Anything else is a hosted predictor's name.
        env: The hosted environment, `production` by default.
        target: An ad hoc predictor's target column.
        inputs: An ad hoc predictor's input columns; inferred if None.
        exclude: Columns an ad hoc predictor leaves out of the inferred inputs.
        id: The id column.
        time: The time column.
        series: A forecaster's series columns.
        frequency: A forecaster's step.
        horizon: A forecaster's steps.
        quantiles: A regressor's or forecaster's levels.
        context: An ad hoc predictor's context rows, so `predict` can omit them.
        selection: How each query's context is drawn: core, neighbors, kind.
        client: The hosted client to use; one from the profile if None.
        download: Whether a model's files may download if not cached.
    """

    def __init__(
        self,
        ref: str | os.PathLike[str],
        *,
        env: str | None = None,
        target: str | None = None,
        inputs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        id: str | None = None,
        time: str | None = None,
        series: Sequence[str] | str | None = None,
        frequency: Any = None,
        horizon: int | None = None,
        quantiles: Sequence[float] | None = None,
        context: Any = None,
        selection: Mapping[str, Any] | None = None,
        client: Any = None,
        download: bool = True,
    ) -> None:
        """Resolves the reference; nothing runs until `predict` or `eval`."""
        self._kind, self._where = resolve_reference(ref)
        self._client_given = client
        self._download = download
        self._as_of: Any = None
        self._root: Path | None = None
        self._spec: Any = None
        self._context_frame: pa.Table | None = None
        self._query_frame: pa.Table | None = None
        self.target = target
        self.inputs: tuple[str, ...] | None = (
            tuple(inputs) if inputs is not None else None
        )
        self.exclude = tuple(exclude)
        self.id = id
        self.time = time
        self.series: tuple[str, ...] = (
            (series,) if isinstance(series, str) else tuple(series or ())
        )
        self.frequency = frequency
        self.horizon = horizon
        self.quantiles = tuple(quantiles) if quantiles is not None else None
        self.selection = dict(selection) if selection else None
        self._env: str | None = None
        if self._kind == "local":
            if env is not None:
                raise InvalidInput(
                    "a local predictor has no environment",
                    hint="pass the hosted name without a path to use an environment",
                )
            self._root = Path(self._where)
            self.name = self._root.name
            self._load_local()
        elif self._kind == "catalog":
            if env is not None:
                raise InvalidInput("an ad hoc predictor has no environment")
            if not target:
                raise InvalidInput(
                    f"an ad hoc predictor on {self._where} needs a target",
                    param="target",
                    hint='pass target="<column>"',
                )
            self.model = self._where
            self.name = "ad-hoc"
            self.task = registry.task_of(self.model)
            if context is not None:
                self._context_frame = tabular.frame(context)
        elif self._kind == "served":
            if env is not None:
                raise InvalidInput("a served predictor's environment is its URL")
            self._base, self.name = self._where
            self._env = self._base
            self.model = "?"
            self.task = "?"
        else:
            if env is not None and env not in ENVIRONMENTS:
                raise InvalidInput(
                    f"environment `{env}` is not one of staging, production",
                    param="env",
                )
            self.name = self._where
            self._env = env or "production"
            self.model = "?"
            self.task = "?"
            self._facts: dict[str, Any] | None = None

    # -- resolution ---------------------------------------------------------

    def _load_local(self) -> None:
        assert self._root is not None
        spec = _load_spec(self._root)
        self._spec = spec
        if spec.get("name"):
            self.name = spec["name"]
        self.model = registry.canonical(spec["model"])
        self.task = registry.task_of(self.model)
        self.target = self.target or spec.get("target")
        if not self.target:
            raise InvalidInput(
                f"predictor {self.name} has no target",
                hint="add `target: <column>` to gf.yaml",
            )
        if self.inputs is None and spec.get("inputs") is not None:
            self.inputs = tuple(spec["inputs"])
        if not self.exclude:
            self.exclude = tuple(spec.get("exclude") or ())
        self.id = self.id or spec.get("id")
        self.time = self.time or spec.get("time")
        self.series = self.series or tuple(spec.get("series") or ())
        self.frequency = self.frequency or spec.get("frequency")
        self.horizon = self.horizon or spec.get("horizon")
        if self.quantiles is None and spec.get("quantiles"):
            self.quantiles = tuple(spec["quantiles"])
        if self.selection is None and spec.get("selection"):
            self.selection = dict(spec["selection"])
        context = spec.get("context") or {}
        self._context_kind = context.get("kind", "snapshot")
        if self._context_kind == "snapshot":
            source = self._root / context["source"]
            if not source.is_file():
                raise NotFound(
                    f"context file `{context['source']}` is not in {self._root}",
                    hint="put the snapshot next to gf.yaml, or fix `context.snapshot`",
                )
            frame = tabular.frame(str(source))
            self._context_frame, self._query_frame = _split(frame, self, context)
        elif self._context_kind == "live":
            self._live_name = context.get("name")

    # -- facts --------------------------------------------------------------

    @property
    def env(self) -> str | None:
        """None for local and ad hoc, `staging`/`production`, or the served URL."""
        return self._env

    @property
    def context(self) -> Context:
        """The predictor's context: its kind and rows, and a live context's handles."""
        if self._kind in ("local", "catalog"):
            if self._kind == "local" and self._context_kind == "live":
                return Context("live", None, self._live_name, self._as_of, self)
            if self._kind == "local" and self._context_kind == "request":
                return Context("request", None)
            rows = None if self._context_frame is None else len(self._context_frame)
            return Context("snapshot" if self._kind == "local" else "request", rows)
        facts = self._hosted_facts()
        info = facts.get("context") or {}
        return Context(
            str(info.get("kind", "live")),
            info.get("rows"),
            info.get("name", self.name),
            self._as_of,
            self,
        )

    def _id_name(self) -> str | None:
        """The id column, asked of a served or hosted predictor once."""
        if self.id is None and self._kind in ("served", "hosted"):
            self.id = self._hosted_facts().get("id") or None
        return self.id

    def _hosted_facts(self) -> dict[str, Any]:
        if self._kind == "served":
            return self._served_get(f"/predictors/{self.name}")
        if self._facts is None:
            with boundary():
                found = self._client().predictors.get(self.name, env=self._env)
            data = getattr(found, "data", None)
            self._facts = dict(data) if isinstance(data, Mapping) else {}
            self.model = str(self._facts.get("model", self.model))
            self.task = str(self._facts.get("task", self.task))
        return self._facts

    def _client(self) -> Any:
        if self._client_given is None:
            from generalforecast.client import Client

            with boundary():
                self._client_given = Client()
        return self._client_given

    def __repr__(self) -> str:  # noqa: D105
        inputs = "?" if self.inputs is None else str(len(self.inputs))
        if (
            self._kind == "catalog"
            and self.inputs is None
            and self._context_frame is not None
        ):
            roles = {self.target, self.id, self.time, *self.series, *self.exclude}
            inputs = str(
                sum(1 for name in self._context_frame.column_names if name not in roles)
            )
        try:
            context = (
                repr(self.context)
                if self._kind in ("local", "catalog")
                else "Context(?)"
            )
        except Error:
            context = "Context(?)"
        return (
            f"Predictor({self.name}, model={self.model}, inputs={inputs}, "
            f"context={context[len('Context(') : -1]}, env={self._env!r})"
        )

    # -- prediction ---------------------------------------------------------

    def request(
        self, queries: Any, context: Any = None, *, as_of: Any = None
    ) -> PredictRequest:
        """The validated request `predict` runs for these rows."""
        if context is None:
            context = self._context_frame
        if context is None:
            raise InvalidInput(
                "context is required",
                param="context",
                hint="pass context= to predict, or give the predictor a context",
            )
        assert self.target is not None
        return PredictRequest.build(
            model=self.model,
            target=self.target,
            context=context,
            queries=queries,
            inputs=self.inputs,
            exclude=self.exclude,
            id=self.id,
            time=self.time,
            series=self.series,
            frequency=self.frequency,
            horizon=self.horizon,
            quantiles=self.quantiles,
            selection=self.selection,
            as_of=as_of if as_of is not None else self._as_of,
        )

    def predict(self, queries: Any, context: Any = None, *, as_of: Any = None) -> Any:
        """Scores rows.

        Args:
            queries: Rows to score: a frame, list of dicts, Arrow table, or a
                path or URL to a .csv, .parquet or .json file.
            context: Rows the model learns from, for an ad hoc predictor or a
                predictor whose context kind is `request`.
            as_of: For a live context, the time to read it as of.

        Returns:
            A pandas DataFrame: `id` (if any), `label` and `p_<class>`
            columns for a classifier, or `value` and `q_<level>` columns for
            a regressor or forecaster.
        """
        if self._kind in ("local", "catalog") and not (
            self._kind == "local" and self._context_kind == "live"
        ):
            request = self.request(queries, context, as_of=as_of)
            with boundary():
                return run_local(request, download=self._download)
        if self._kind == "served":
            payload = PredictRequest.declared_payload(
                queries, context=context, as_of=as_of or self._as_of
            )
            value = self._served_post(f"/predictors/{self.name}/predict", payload)
            return _frame_of_result(value, self._id_name())
        with boundary():
            result = self._client().predictors.predict(
                self.name,
                queries,
                context=context,
                as_of=as_of if as_of is not None else self._as_of,
                env=self._env,
            )
        return _frame_of_result(result, self._id_name())

    def _with_as_of(self, ts: Any) -> Predictor:
        clone = Predictor.__new__(Predictor)
        clone.__dict__.update(self.__dict__)
        clone._as_of = ts
        return clone

    # -- served HTTP --------------------------------------------------------

    def _served_post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        request_id = "r-" + uuid.uuid4().hex
        body = {
            "api_version": "gf/v1",
            "project": "local",
            "request_id": request_id,
            "resource": {"kind": "predictor", "name": self.name},
            "payload": dict(payload),
        }
        return self._served(path, json.dumps(body).encode(), request_id)

    def _served_get(self, path: str) -> dict[str, Any]:
        return self._served(path, None, "r-" + uuid.uuid4().hex)

    def _served(self, path: str, data: bytes | None, request_id: str) -> dict[str, Any]:
        url = f"{self._base}/v1/projects/local{path}"
        request = urllib.request.Request(
            url,
            data=data,
            method="POST" if data is not None else "GET",
            headers={"Content-Type": "application/json", "X-GF-Request-ID": request_id},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            raw = error.read()
            status = error.code
        except OSError as error:
            raise Internal(
                f"cannot reach {self._base}: {error}".rstrip("."),
                request_id=request_id,
            ) from None
        try:
            value = json.loads(raw)
        except ValueError:
            raise Internal(
                f"{self._base} answered {status} without a JSON body",
                request_id=request_id,
            ) from None
        if not isinstance(value, Mapping):
            raise Internal(
                f"{self._base} answered with no envelope", request_id=request_id
            )
        if "error" in value and value["error"] is not None:
            raise from_wire(value["error"], status=status, request_id=request_id)
        if status >= 400:
            raise Internal(f"{self._base} answered {status}", request_id=request_id)
        return dict(value.get("result", value))

    # -- evaluation ---------------------------------------------------------

    def eval(self, queries: Any = None, *, models: Sequence[str] = ()) -> Evaluation:
        """Scores held-out rows whose target is known.

        Args:
            queries: Rows with the target, to score instead of the
                predictor's own query rows (a snapshot's empty-target rows,
                rows after `until`, or the holdout draw).
            models: Other catalog models to score on the same split.

        Returns:
            The metrics per model and the scored rows.
        """
        if self._kind not in ("local", "catalog"):
            raise InvalidInput(
                "eval runs in process; use a local or ad hoc predictor",
                hint="gf.Predictor('./dir').eval()",
            )
        context = self._context_frame
        if context is None:
            raise InvalidInput(
                "eval needs a context", hint="give the predictor a context"
            )
        if queries is not None:
            held = tabular.frame(queries)
        elif self._query_frame is not None and len(self._query_frame):
            held = self._query_frame
        elif self._kind == "local":
            raise InvalidInput(
                f"predictor {self.name} has no query rows to score",
                param="queries",
                because="every row of the context has a target, and gf.yaml "
                "sets no `until` or `holdout`",
                hint="pass rows with known targets: `gf eval --queries FILE` or "
                "`.eval(queries)`; or set `holdout` or `until` under `context`",
            )
        else:
            context, held = _holdout(context, self.target, seed=0)
        assert self.target is not None
        if self.target not in held.column_names:
            raise InvalidInput(
                f"query rows have no `{self.target}` column to score against",
                param="target",
            )
        truth_column = held.column(self.target)
        if _empty(truth_column).to_numpy(zero_copy_only=False).any():
            raise InvalidInput(
                f"query rows have empty `{self.target}` values; eval needs the truth",
                param="target",
            )
        truth = truth_column.to_pylist()
        rows: dict[str, Any] = {}
        if self.id is not None and self.id in held.column_names:
            rows["id"] = held.column(self.id).to_pylist()
        rows[self.target] = truth
        metrics: dict[str, dict[str, float]] = {}
        import pandas as pd

        for ref in (self.model, *models):
            model = registry.canonical(ref)
            task = registry.task_of(model)
            cap = registry.limits(model).queries if task != "forecast" else len(held)
            frames = []
            for start in range(0, max(len(held), 1), max(cap, 1)):
                batch = dataclasses.replace(
                    self.request(held.slice(start, cap), context),
                    model=model,
                    task=task,
                )
                if start == 0:
                    request = batch
                with boundary():
                    frames.append(run_local(batch, download=self._download))
            frame = (
                frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
            )
            metrics[model] = self._score(request, frame, truth, held)
            key = (
                registry.short(model)
                if (len(models) or model != self.model)
                else "prediction"
            )
            rows[key] = frame[
                "label" if self.task == "classification" else "value"
            ].tolist()
        return Evaluation(metrics, pd.DataFrame(rows))

    def _score(
        self, request: PredictRequest, frame: Any, truth: list[Any], held: pa.Table
    ) -> dict[str, float]:
        if request.task == "classification":
            probabilities = {
                label: frame[f"p_{label}"].tolist()
                for label in request.classes
                if f"p_{label}" in frame.columns
            }
            return classification_metrics(truth, frame["label"].tolist(), probabilities)
        quantiles = {
            float(column[2:]): frame[column].tolist()
            for column in frame.columns
            if column.startswith("q_")
        }
        if request.task == "regression":
            return regression_metrics(truth, frame["value"].tolist(), quantiles or None)
        series_column = request.series[0] if request.series else None
        history: dict[Any, list[float]] = {}
        context = request.context
        keys = (
            context.column(series_column).to_pylist()
            if series_column is not None
            else [None] * len(context)
        )
        for key, value in zip(
            keys, context.column(request.target).to_pylist(), strict=True
        ):
            if value is not None:
                history.setdefault(key, []).append(float(value))
        held_keys = (
            held.column(series_column).to_pylist()
            if series_column is not None
            else [None] * len(held)
        )
        if not quantiles:
            quantiles = {0.5: frame["value"].tolist()}
        return forecast_metrics(truth, quantiles, history=history, series=held_keys)

    # -- lifecycle ----------------------------------------------------------

    def explain(self) -> Any:
        """The resolved schema: model, task, target, inputs, context and selection."""
        if self._kind == "local":
            assert self._root is not None
            try:
                from generalforecast import predictors as _predictors

                spec = _predictors.load(self._root / "gf.yaml")
                return _predictors.resolve(spec, self._root)
            except ImportError:
                pass
        if self._kind in ("local", "catalog"):
            context = self._context_frame
            inputs: tuple[tuple[str, str], ...] = ()
            classes: tuple[Any, ...] = ()
            if context is not None:
                from generalforecast.columns import infer

                names = self.inputs
                if names is None:
                    roles = {
                        self.target,
                        self.id,
                        self.time,
                        *self.series,
                        *self.exclude,
                    }
                    names = tuple(n for n in context.column_names if n not in roles)
                kinds = infer(
                    context.select([n for n in names if n in context.column_names])
                )
                inputs = tuple((n, str(kinds.get(n, "?"))) for n in names)
                if (
                    self.task == "classification"
                    and self.target in context.column_names
                ):
                    column = context.column(self.target)
                    classes = tuple(
                        sorted(
                            set(column.filter(pc.invert(_empty(column))).to_pylist()),
                            key=lambda item: (str(type(item)), item),
                        )
                    )
            assert self.target is not None
            return Resolution(
                name=self.name,
                model=self.model,
                task=self.task,
                target=self.target,
                classes=classes,
                id=self.id,
                time=self.time,
                series=self.series,
                inputs=inputs,
                context=self.context,
                selection=self.selection,
            )
        facts = self._hosted_facts()
        return Resolution(
            name=self.name,
            model=str(facts.get("model", self.model)),
            task=str(facts.get("task", self.task)),
            target=str(facts.get("target", self.target or "?")),
            id=facts.get("id"),
            time=facts.get("time"),
            series=tuple(facts.get("series") or ()),
            inputs=tuple(
                (item["name"], item.get("type", "?"))
                if isinstance(item, Mapping)
                else (str(item), "?")
                for item in facts.get("inputs") or ()
            ),
            context=self.context,
            selection=facts.get("selection"),
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Writes `gf.yaml` (and the snapshot) for an ad hoc or local predictor at `path`."""
        if self._kind not in ("local", "catalog"):
            raise InvalidInput(
                "only a local or ad hoc predictor can be saved",
                hint="a hosted predictor's manifest is its revision",
            )
        import yaml

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        name = root.resolve().name if self._kind == "catalog" else self.name
        if not NAME.match(name):
            raise InvalidInput(
                f"`{name}` is not a predictor name",
                hint="use lowercase letters, digits and dashes, starting with a letter",
            )
        manifest: dict[str, Any] = {
            "name": name,
            "model": self.model,
            "target": self.target,
        }
        if self.id:
            manifest["id"] = self.id
        if self.inputs is not None:
            manifest["inputs"] = list(self.inputs)
        elif self.exclude:
            manifest["inputs"] = {"exclude": list(self.exclude)}
        if self.time:
            manifest["time"] = self.time
        if self.series:
            manifest["series"] = list(self.series)
        if self.frequency is not None:
            manifest["frequency"] = self.frequency
        if self.horizon is not None:
            manifest["horizon"] = self.horizon
        if self.quantiles is not None:
            manifest["quantiles"] = list(self.quantiles)
        if self.selection:
            manifest["selection"] = dict(self.selection)
        if self._context_frame is not None:
            from pyarrow import parquet

            frame = self._context_frame
            if self._query_frame is not None and len(self._query_frame):
                frame = pa.concat_tables(
                    [frame, self._query_frame], promote_options="default"
                )
            parquet.write_table(frame, root / "context.parquet")
            manifest["context"] = {"snapshot": "context.parquet"}
        elif self._kind == "local" and self._spec is not None:
            manifest["context"] = self._spec.get("context_raw") or {
                "request": {"max_rows": 10000}
            }
        else:
            manifest["context"] = {"request": {"max_rows": 10000}}
        (root / "gf.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        return root

    def push(self, env: str = "staging") -> Predictor:
        """Deploys a local predictor to a hosted environment and returns it."""
        if env not in ENVIRONMENTS:
            raise InvalidInput(
                f"environment `{env}` is not one of staging, production", param="env"
            )
        if self._kind != "local":
            raise InvalidInput(
                "only a local predictor can be pushed",
                hint="save it first with .save(path), then push that directory",
            )
        assert self._root is not None
        client = self._client()
        with boundary():
            client.predictors.push(self._root, env=env)
        return Predictor(self.name, env=env, client=client)


def _frame_of_result(result: Any, id_name: str | None) -> Any:
    import pandas as pd

    if isinstance(result, Mapping):
        predictions = result.get("predictions") or []
    else:
        table = getattr(result, "table", None)
        predictions = table.to_pylist() if table is not None else []
    rows: list[dict[str, Any]] = []
    for item in predictions:
        row: dict[str, Any] = {}
        if "id" in item:
            row["id"] = item["id"]
        elif id_name and id_name in item:
            row["id"] = item[id_name]
        if "label" in item:
            row["label"] = item["label"]
        if "value" in item:
            row["value"] = item["value"]
        probabilities = item.get("probabilities")
        if isinstance(probabilities, Mapping):
            for label, p in probabilities.items():
                row[f"p_{label}"] = p
        quantiles = item.get("quantiles")
        if isinstance(quantiles, Mapping):
            for level, q in quantiles.items():
                row[f"q_{float(level):g}"] = q
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_reference(ref: Any) -> tuple[str, Any]:
    """What a reference names: `("local", path)`, `("catalog", id)`, `("served", (base, name))` or `("hosted", name)`.

    Raises:
        AmbiguousReference: When a bare name is both a directory with
            `gf.yaml` and so a hosted name.
    """
    if isinstance(ref, os.PathLike):
        return "local", _local_root(os.fspath(ref))
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidInput(
            "a predictor reference must be a path, a name, a model id or a URL"
        )
    text = ref.strip()
    if _URL.match(text):
        base, _, name = text.rstrip("/").rpartition("/")
        if not name or "://" in name or not base:
            raise InvalidInput(
                f"`{text}` names no predictor",
                hint="use http://host:port/<predictor>",
            )
        return "served", (base, name)
    if registry.is_catalog_id(text) and registry.known_org(text):
        return "catalog", registry.canonical(text)
    if registry.is_bare_name(text):
        registry.canonical(text)  # a bare model name: rejected with the canonical id
    has_separator = "/" in text or os.sep in text or text.startswith(".")
    is_dir = (Path(text) / "gf.yaml").is_file()
    if has_separator:
        return "local", _local_root(text)
    if is_dir:
        raise AmbiguousReference(
            f"`{text}` is both a directory with gf.yaml and a hosted predictor name",
            hint=f"pass ./{text} for the local predictor, or env= for the hosted one",
        )
    if not NAME.match(text):
        raise InvalidInput(
            f"`{text}` is not a predictor name, path, model id or URL",
            hint="names are lowercase letters, digits and dashes",
        )
    return "hosted", text


def _local_root(path: str) -> Path:
    root = Path(path).expanduser()
    if root.is_file() and root.name == "gf.yaml":
        root = root.parent
    if not root.is_dir():
        raise NotFound(f"no directory at `{path}`")
    if not (root / "gf.yaml").is_file():
        raise NotFound(
            f"no gf.yaml in `{path}`", hint="run `gf init` there to write one"
        )
    return root


def _load_spec(root: Path) -> dict[str, Any]:
    """The manifest's fields as the SDK reads them, through session B's loader when present."""
    try:
        from generalforecast import predictors as _predictors
    except ImportError:
        _predictors = None
    if _predictors is not None and hasattr(_predictors, "load"):
        spec = _predictors.load(root / "gf.yaml")
        return _spec_fields(spec)
    import yaml

    with boundary():
        raw = yaml.safe_load((root / "gf.yaml").read_text()) or {}
    if not isinstance(raw, Mapping):
        raise InvalidInput(f"{root / 'gf.yaml'} is not a mapping")
    return _manifest_fields(raw)


def _spec_fields(spec: Any) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        return _manifest_fields(spec)
    if hasattr(spec, "to_dict"):
        return _manifest_fields(spec.to_dict())
    return _manifest_fields(
        {name: getattr(spec, name) for name in dir(spec) if not name.startswith("_")}
    )


def _manifest_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": raw.get("name"),
        "model": raw.get("model"),
        "target": raw.get("target"),
        "id": raw.get("id"),
        "time": raw.get("time"),
        "series": raw.get("series"),
        "frequency": raw.get("frequency"),
        "horizon": raw.get("horizon"),
        "quantiles": raw.get("quantiles"),
        "selection": raw.get("selection"),
        "inputs": None,
        "exclude": (),
    }
    if not out["model"]:
        raise InvalidInput(
            "gf.yaml has no `model`", hint="add `model: general-forecast/<name>`"
        )
    inputs = raw.get("inputs")
    if isinstance(inputs, Mapping):
        if "include" in inputs and inputs["include"] is not None:
            out["inputs"] = tuple(inputs["include"])
        out["exclude"] = tuple(inputs.get("exclude") or ())
        if out["inputs"] is not None and out["exclude"]:
            raise InvalidInput(
                "gf.yaml gives both an `inputs` list and `inputs.exclude`",
                hint="a list of inputs already replaces inference",
            )
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        out["inputs"] = tuple(inputs)
    elif inputs is not None:
        raise InvalidInput("gf.yaml `inputs` must be a list or a mapping")
    context = raw.get("context") or {}
    info: dict[str, Any] = {}
    if isinstance(context, Mapping):
        kinds = [
            key
            for key in ("snapshot", "request", "live", "source", "mode")
            if key in context
        ]
        if "snapshot" in context:
            info = {"kind": "snapshot", "source": context["snapshot"]}
        elif "live" in context:
            info = {"kind": "live", "name": context["live"]}
        elif "request" in context:
            info = {"kind": "request", **(context["request"] or {})}
        elif context.get("mode") == "snapshot" or "source" in context:
            info = {"kind": "snapshot", "source": context.get("source")}
        elif context.get("mode") == "live":
            info = {"kind": "live", "name": context.get("source")}
        elif context.get("mode") == "request":
            info = {"kind": "request"}
        elif kinds:
            info = {"kind": kinds[0]}
        if "until" in context and "holdout" in context:
            raise InvalidInput(
                "gf.yaml sets both `context.until` and `context.holdout`",
                hint="keep one of them",
            )
        info["until"] = context.get("until")
        info["holdout"] = context.get("holdout")
    elif isinstance(context, str):
        info = {"kind": "snapshot", "source": context}
    out["context"] = info
    out["context_raw"] = dict(context) if isinstance(context, Mapping) else context
    return out


def _split(
    frame: pa.Table, predictor: Predictor, context: Mapping[str, Any]
) -> tuple[pa.Table, pa.Table]:
    """Context rows and query rows of a snapshot, through session B's `split` when present."""
    try:
        from generalforecast import predictors as _predictors

        if hasattr(_predictors, "split") and predictor._root is not None:
            spec = _predictors.load(predictor._root / "gf.yaml")
            with boundary():
                left, right = _predictors.split(frame, spec)
            return tabular.frame(left), tabular.frame(right)
    except ImportError:
        pass
    target = predictor.target
    assert target is not None
    if target not in frame.column_names:
        near = _did_you_mean(target, frame.column_names)
        raise InvalidInput(
            f"target column `{target}` is not in the context",
            param="target",
            hint=f"did you mean `{near}`?"
            if near
            else "the context has " + ", ".join(frame.column_names),
        )
    empty = _empty(frame.column(target))
    if pc.any(empty).as_py():
        return frame.filter(pc.invert(empty)), frame.filter(empty)
    until = context.get("until")
    if until is not None and predictor.time:
        times = frame.column(predictor.time)
        boundary_value = (
            pa.scalar(until, type=times.type)
            if not isinstance(until, str)
            else pa.scalar(until).cast(times.type)
        )
        after = pc.greater_equal(times, boundary_value)
        return frame.filter(pc.invert(after)), frame.filter(after)
    holdout = context.get("holdout")
    if holdout:
        fraction = (
            float(holdout.get("fraction", _HOLDOUT_FRACTION))
            if isinstance(holdout, Mapping)
            else float(holdout)
        )
        seed = int(holdout.get("seed", 0)) if isinstance(holdout, Mapping) else 0
        return _holdout(frame, target, fraction=fraction, seed=seed)
    return frame, frame.slice(0, 0)


def _holdout(
    frame: pa.Table,
    target: str | None,
    *,
    fraction: float = _HOLDOUT_FRACTION,
    seed: int = 0,
) -> tuple[pa.Table, pa.Table]:
    """A deterministic random split: the context and `fraction` of rows held out."""
    count = len(frame)
    held = max(1, round(fraction * count)) if count > 1 else 0
    order = np.random.default_rng(seed).permutation(count)
    mask = np.zeros(count, dtype=bool)
    mask[order[:held]] = True
    return frame.filter(pa.array(~mask)), frame.filter(pa.array(mask))


__all__ = [
    "Context",
    "Evaluation",
    "PredictRequest",
    "Predictor",
    "Resolution",
    "classification_metrics",
    "forecast_metrics",
    "regression_metrics",
    "resolve_reference",
    "run_local",
]
