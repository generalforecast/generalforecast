# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`generalforecast.Predictor`: one object for local, ad hoc, served and hosted predictors.

A predictor is a catalog model, a target, inputs and a context. A local
predictor's `gf.yaml` resolves as `gf check` resolves it, and an ad hoc
predictor's settings resolve as the hosted ad hoc route resolves them, into
a version 2 lock document. `run_local` compiles that document and binds its
context rows with `PredictorExecutor`, the executor hosted serving runs, so
types, past and future covariates, `max_rows` and selection are honoured
exactly as `gf serve` and hosted serving honour them. A served or hosted
predictor is called over HTTP.
"""

from __future__ import annotations

import dataclasses
import math
import os
import urllib.parse
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
    InvalidInput,
    NotFound,
    boundary,
)
from generalforecast.native import models as registry
from generalforecast.predictors.validation import NAME

pc: Any = _compute
ENVIRONMENTS = ("staging", "production")
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
    future: tuple[str, ...] = ()
    lock: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)

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
        future: Sequence[str] = (),
    ) -> PredictRequest:
        """Checks the settings against the rows and the model's limits.

        A forecaster's inputs are its covariates: `future` names those whose
        values each query brings ahead, and every other input is a past
        covariate, the rule `gf check` applies.

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
            future=tuple(future),
        )

    def manifest(self) -> dict[str, Any]:
        """The ad hoc request as a manifest, as the hosted ad hoc route resolves it."""
        manifest: dict[str, Any] = {
            "name": "ad-hoc",
            "model": self.model,
            "target": self.target,
            "context": {"request": {}},
        }
        if self.task == "forecast":
            past = [name for name in self.inputs if name not in self.future]
            sections = {"past": past, "future": list(self.future)}
            if past or self.future:
                manifest["inputs"] = {
                    key: value for key, value in sections.items() if value
                }
        else:
            manifest["inputs"] = list(self.inputs)
        for key in ("id", "time", "frequency", "horizon"):
            if getattr(self, key) is not None:
                manifest[key] = getattr(self, key)
        if self.series:
            manifest["series"] = list(self.series)
        if self.quantiles is not None and self.task != "classification":
            manifest["quantiles"] = list(self.quantiles)
        if self.selection:
            manifest["selection"] = dict(self.selection)
        return manifest

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
    """Runs a request in this process, as hosted serving runs the same lock, and returns a frame.

    A local predictor's request carries its lock document; an ad hoc
    request resolves its settings against its context rows first, as the
    hosted ad hoc route does. The lock is compiled with `compile_lock` and
    bound by `PredictorExecutor`; the executor's results become the frame.
    """
    from generalforecast.contracts import parse_lock
    from generalforecast.predictors.models import CatalogResolver
    from generalforecast.predictors.runtime import PredictorExecutor
    from generalforecast.predictors.runtime.compile import compile_lock
    from generalforecast.predictors.runtime.context import served_context

    with boundary():
        lock = parse_lock(dict(request.lock or _ad_hoc_lock(request)))
        kind = lock.context_kind
        rows = served_context(lock, request.context)
        plan = compile_lock(lock, rows, resolver=CatalogResolver("local"))
        prepared = PredictorExecutor(allow_downloads=download, surface="local").prepare(
            plan, rows if kind != "request" else None
        )
        queries = _query_rows(plan, request.queries)
        table = prepared.predict(queries, context=rows if kind == "request" else None)
    return _frame_of_predictions(plan, queries, table)


def _ad_hoc_lock(request: PredictRequest) -> dict[str, Any]:
    """The lock document of an ad hoc request, resolved against its context rows."""
    from generalforecast.predictors import lock, resolve
    from generalforecast.predictors.spec import PredictorSpec

    spec = PredictorSpec.from_dict(request.manifest())
    return lock.document(resolve(spec, rows=request.context))


def _query_rows(plan: Any, queries: pa.Table) -> pa.Table:
    """The query columns the plan reads, in its order: its id and inputs, or a forecast's roles.

    Other columns a frame carries, such as the target of rows held out for
    `eval`, are left out, as a client would leave them out of a request.
    """
    spec = plan.spec
    if plan.task == "forecast":
        wanted = [*spec.series, spec.time, *spec.future_inputs]
        if spec.id is not None:
            wanted.append(spec.id)
    else:
        wanted = [
            column.name
            for column in plan.schema.columns
            if column.role in ("input", "id")
        ] or [*(spec.inputs.include or ()), *([spec.id] if spec.id else [])]
    missing = [name for name in wanted if name not in queries.column_names]
    if missing:
        raise InvalidInput(
            f"input column `{missing[0]}` is not in the queries",
            param=f"inputs.{missing[0]}",
            hint="the queries have " + ", ".join(queries.column_names),
        )
    return queries.select(list(dict.fromkeys(wanted)))


def _frame_of_predictions(plan: Any, queries: pa.Table, table: pa.Table) -> Any:
    """The executor's predictions as a frame: id, a forecast's series and time, then outputs."""
    import pandas as pd

    spec = plan.spec
    frame = _frame_of_result({"predictions": table.to_pylist()}, spec.id)
    if plan.task == "forecast":
        keys = pd.DataFrame(
            {
                name: queries.column(name).to_pylist()
                for name in (*spec.series, spec.time)
            }
        )
        frame = pd.concat([keys, frame.drop(columns=["id"], errors="ignore")], axis=1)
        if spec.id is not None and spec.id in queries.column_names:
            frame.insert(0, "id", queries.column(spec.id).to_pylist())
    return frame


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

    def append(self, rows: Any, *, idempotency_key: str | None = None) -> Any:
        """Appends rows to a live context, each carrying the predictor's id column.

        Args:
            rows: The rows to append.
            idempotency_key: The write's durable id, so a retried append is
                applied once; a fresh one if None.
        """
        owner = self._live()
        identity = owner._id_name()
        if identity is None:
            raise InvalidInput(
                "a live context's rows need an id column",
                hint="declare `id:` in gf.yaml so appended rows can be labeled later",
            )
        with boundary():
            return owner._client().contexts.append(
                self.name,
                tabular.frame(rows),
                id=identity,
                idempotency_key=idempotency_key or "w-" + uuid.uuid4().hex,
            )

    def label(
        self,
        ids: Iterable[Any],
        values: Iterable[Any],
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        """Sets the target of rows in a live context, by id."""
        owner = self._live()
        ids, values = list(ids), list(values)
        if len(ids) != len(values):
            raise InvalidInput("ids and values must have the same length")
        identity = owner._id_name() or "id"
        target = owner.target
        if target is None and owner._kind in ("served", "hosted"):
            target = owner._hosted_facts().get("target")
        if target is None:
            raise InvalidInput("the predictor names no target to label")
        rows = [
            {identity: key, target: value}
            for key, value in zip(ids, values, strict=True)
        ]
        with boundary():
            return owner._client().contexts.label(
                self.name,
                rows,
                id=identity,
                column=target,
                idempotency_key=idempotency_key or "w-" + uuid.uuid4().hex,
            )

    def as_of(self, ts: Any) -> Predictor:
        """The same predictor reading the live context as of `ts`."""
        owner = self._live()
        return owner._with_as_of(ts)


@dataclass(frozen=True)
class Resolution:
    """The resolved schema of a predictor, as `gf check` prints it."""

    # A served or hosted predictor's view, from what its server describes;
    # local and ad hoc predictors explain with `predictors.Resolution`.
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
        future: A forecaster's inputs whose values each query brings ahead;
            its other inputs are past covariates.
        client: The hosted client to use; one from the profile if None.
        download: Whether a model's files may download if not cached.
        project: A served predictor's project; `local` if None.
        api_key: A served predictor's key; `GF_API_KEY` if None.
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
        future: Sequence[str] = (),
        client: Any = None,
        download: bool = True,
        project: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Resolves the reference; nothing runs until `predict` or `eval`."""
        self._kind, self._where = resolve_reference(ref, env=env)
        self._client_given = client
        self._download = download
        self._as_of: Any = None
        self._root: Path | None = None
        self._spec: Any = None
        self._resolution: Any = None
        self._lock: dict[str, Any] | None = None
        self._snapshot: pa.Table | None = None
        self._context_frame: pa.Table | None = None
        self._query_frame: pa.Table | None = None
        self._context_kind = "request"
        self._live_name: str | None = None
        self._project = project
        self._api_key = api_key
        self._transport: Any = None
        self.future = tuple(future)
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

    @classmethod
    def local(cls, path: str | os.PathLike[str]) -> Predictor:
        """The local predictor in directory `path`, never a hosted, catalog or served one.

        A bare name such as `churn` is the directory `churn` here, so
        commands given a directory never read it as a hosted name.

        Raises:
            NotFound: If `path` is no directory or holds no `gf.yaml`.
        """
        return cls(Path(os.fspath(path)))

    def _load_local(self) -> None:
        """Resolves `gf.yaml` as `gf check` does, without writing `gf.lock`."""
        from generalforecast.predictors import lock, resolve, verified
        from generalforecast.predictors.spec import load

        assert self._root is not None
        with boundary():
            spec = load(self._root)
            # A gf.lock beside the manifest is what gf serve and gf push run;
            # predicting from anything else would answer differently, so it
            # must be exactly the lock gf check writes now. Without one, the
            # manifest resolves as gf check would resolve it.
            if (self._root / lock.LOCK).exists():
                resolution = verified(spec, self._root)
            else:
                resolution = resolve(spec, self._root)
        self._spec, self._resolution = spec, resolution
        self._lock = lock.document(resolution)
        self.name = spec.name
        self.model = spec.model
        self.task = resolution.task
        self.target = spec.target
        self.inputs = resolution.input_names
        self.exclude = spec.exclude
        self.id, self.time, self.series = spec.id, spec.time, spec.series
        self.frequency, self.horizon = resolution.frequency, spec.horizon
        self.quantiles = resolution.quantiles or None
        self.future = spec.future
        self.selection = resolution.selection.to_dict()
        self._context_kind = spec.context.kind
        self._live_name = spec.context.live
        self._context_frame = resolution.context_table
        self._query_frame = resolution.query_table

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
            self._facts = dict(found) if isinstance(found, Mapping) else {}
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
        if self._kind == "local":
            return self._local_request(queries, context, as_of=as_of)
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
            future=self.future,
        )

    def _local_request(
        self, queries: Any, context: Any = None, *, as_of: Any = None
    ) -> PredictRequest:
        """A local predictor's request: its lock, and the context rows the lock names."""
        assert self._lock is not None and self.target is not None
        if self._context_kind == "request":
            if context is None:
                raise InvalidInput(
                    "context is required",
                    param="context",
                    hint=f"predictor {self.name} takes its context with each call",
                )
            rows = tabular.frame(context)
        else:
            if context is not None:
                raise InvalidInput(
                    f"predictor {self.name} has a {self._context_kind} context",
                    param="context",
                    hint="its context is the one gf.yaml declares; pass queries only",
                )
            rows = self._context_frame
        if rows is None:
            raise InvalidInput(
                "a live context is read where it lives",
                hint="push the predictor and predict with the hosted name",
            )
        resolution = self._resolution
        return PredictRequest(
            model=self.model,
            task=self.task,
            target=self.target,
            context=rows,
            queries=tabular.frame(queries),
            inputs=resolution.input_names,
            id=self.id,
            time=self.time,
            series=self.series,
            frequency=self.frequency,
            horizon=self.horizon,
            quantiles=self.quantiles,
            selection=self.selection,
            as_of=as_of if as_of is not None else self._as_of,
            classes=tuple(resolution.classes or ()),
            future=self.future,
            lock=self._lock,
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

    def _served_transport(self) -> Any:
        """The SDK transport to a `gf serve`, with its key, made once."""
        if self._transport is None:
            from generalforecast.transport import Transport, credential_from_env

            key = self._api_key
            if key is None:
                value, _ = credential_from_env()
                key = value or None
            with boundary():
                self._transport = Transport(self._base, token=key, timeout=120)
        return self._transport

    def _served_post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._served("POST", path, payload)

    def _served_get(self, path: str) -> dict[str, Any]:
        return self._served("GET", path, None)

    def _served(
        self, method: str, path: str, payload: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """One call to a `gf serve`, enveloped and unwrapped as the SDK client does."""
        from generalforecast.service.boundary import Request, Resource, Scope
        from generalforecast.service.sdk import unwrap

        project = self._project or "local"
        request = Request(
            Scope(project, "r-" + uuid.uuid4().hex),
            Resource("predictor", self.name),
            dict(payload or {}),
        )
        with boundary():
            value = self._served_transport().send(
                method,
                f"/v1/projects/{project}{path}",
                request.to_dict() if method == "POST" else None,
                {
                    "X-GF-Request-ID": request.scope.request_id,
                    "X-GF-Project": project,
                },
            )
            return dict(unwrap(value, request))

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
            from generalforecast.predictors.tables import split_by

            assert self.target is not None
            context, held = split_by(
                context,
                self.target,
                holdout={"fraction": _HOLDOUT_FRACTION, "seed": 0},
            )
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
            lock = self._lock if model == self.model else self._lock_for(model)
            for start in range(0, max(len(held), 1), max(cap, 1)):
                given = None if self._kind == "local" else context
                batch = dataclasses.replace(
                    self.request(held.slice(start, cap), given),
                    model=model,
                    task=task,
                    lock=lock,
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

    def _lock_for(self, model: str) -> dict[str, Any] | None:
        """A local predictor's lock with another model, resolved as `gf check` would."""
        if self._kind != "local" or self._spec is None:
            return None
        from generalforecast.predictors import lock, resolve

        with boundary():
            spec = dataclasses.replace(self._spec, model=model)
            return lock.document(resolve(spec, self._root))

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
        """The resolved schema: model, task, target, inputs, context and selection.

        A local predictor's is the `predictors.Resolution` `gf check` prints;
        an ad hoc one's is the same, resolved against its context rows; a
        served or hosted one's is what its server describes.
        """
        if self._kind == "local":
            return self._resolution
        if self._kind == "catalog":
            from generalforecast.predictors import resolve
            from generalforecast.predictors.spec import PredictorSpec

            context = self._context_frame
            if context is None:
                raise InvalidInput(
                    "an ad hoc predictor explains against its context",
                    hint="give the predictor a context",
                )
            request = self.request(context.slice(0, 1), context)
            with boundary():
                return resolve(
                    PredictorSpec.from_dict(request.manifest()), rows=context
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
        """Writes `gf.yaml`, its snapshot and `gf.lock` for a local or ad hoc predictor at `path`.

        A local predictor's manifest is kept whole, its split, types, past
        and future covariates, selection and resources included, with the
        snapshot copied beside it; an ad hoc predictor's settings become a
        manifest with its context rows as the snapshot. The directory checks
        as the original does: `gf.Predictor(path)` predicts the same.
        """
        if self._kind not in ("local", "catalog"):
            raise InvalidInput(
                "only a local or ad hoc predictor can be saved",
                hint="a hosted predictor's manifest is its revision",
            )
        from pyarrow import parquet

        from generalforecast.predictors import resolve, write_lock, write_manifest
        from generalforecast.predictors.spec import Context as SpecContext
        from generalforecast.predictors.spec import PredictorSpec

        root = Path(path)
        name = root.resolve().name if self._kind == "catalog" else self.name
        if not NAME.fullmatch(name):
            raise InvalidInput(
                f"`{name}` is not a predictor name",
                hint="use lowercase letters, digits and dashes, starting with a letter",
            )
        snapshot = "context.parquet"
        with boundary():
            if self._kind == "local":
                assert self._spec is not None and self._root is not None
                spec = dataclasses.replace(self._spec, name=name, marks={})
                rows = None
                if spec.context.kind == "snapshot":
                    from generalforecast.predictors.tables import read_frame

                    assert spec.context.snapshot is not None
                    rows = read_frame(self._root, spec.context.snapshot)
                    spec = dataclasses.replace(
                        spec,
                        context=dataclasses.replace(spec.context, snapshot=snapshot),
                    )
            else:
                rows = self._context_frame
                if rows is None:
                    raise InvalidInput(
                        "an ad hoc predictor saves with its context rows",
                        hint="give the predictor a context",
                    )
                request = self.request(rows.slice(0, 1), rows)
                manifest = request.manifest()
                manifest["name"] = name
                manifest.pop("context")
                spec = dataclasses.replace(
                    PredictorSpec.from_dict(manifest),
                    context=SpecContext("snapshot", snapshot=snapshot),
                )
            root.mkdir(parents=True, exist_ok=True)
            if rows is not None:
                parquet.write_table(rows, root / snapshot)
            write_manifest(spec, root, force=True)
            write_lock(resolve(spec, root), root)
        return root

    def push(self, env: str = "staging") -> Predictor:
        """Deploys a local predictor to a hosted environment and returns it.

        Args:
            env: The environment, `staging` or `production`.
        """
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


def resolve_reference(ref: Any, *, env: str | None = None) -> tuple[str, Any]:
    """What a reference names: `("local", path)`, `("catalog", id)`, `("served", (base, name))` or `("hosted", name)`.

    An environment names a hosted predictor, so a bare name with one is hosted
    even where a directory of that name holds a `gf.yaml`.

    Raises:
        AmbiguousReference: When a bare name is both a directory with
            `gf.yaml` and so a hosted name, and no environment says which.
    """
    if isinstance(ref, os.PathLike):
        return "local", _local_root(os.fspath(ref))
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidInput(
            "a predictor reference must be a path, a name, a model id or a URL"
        )
    text = ref.strip()
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 1 or parsed.query or parsed.fragment:
            raise InvalidInput(
                f"`{text}` names no predictor",
                hint="use http://host:port/<predictor>",
            )
        base = f"{parsed.scheme}://{parsed.netloc}"
        return "served", (base, parts[0])
    if registry.is_catalog_id(text) and registry.known_org(text):
        return "catalog", registry.canonical(text)
    if registry.is_bare_name(text):
        registry.canonical(text)  # a bare model name: rejected with the canonical id
    has_separator = "/" in text or os.sep in text or text.startswith(".")
    is_dir = (Path(text) / "gf.yaml").is_file()
    if has_separator:
        return "local", _local_root(text)
    if is_dir and env is None:
        raise AmbiguousReference(
            f"`{text}` is both a directory with gf.yaml and a hosted predictor name",
            hint=f"pass ./{text} for the local predictor, or env= for the hosted one",
        )
    if not NAME.fullmatch(text):
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
