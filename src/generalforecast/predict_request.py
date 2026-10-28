# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A local or ad hoc prediction's request, run in process as hosted serving runs its lock.

`PredictRequest` is a call's validated settings and rows; `run_local`
compiles its lock and binds its context with `PredictorExecutor`, and
`_frame_of_result` and `_with_forecast_keys` shape every path's answers
alike. `generalforecast.predictor` builds on these.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as _compute

from generalforecast import tabular
from generalforecast.errors import (
    Error,
    InvalidInput,
    boundary,
)
from generalforecast.native import models as registry

pc: Any = _compute


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
        for key in ("id", "time", "frequency", "horizon"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = (
                    value if not hasattr(value, "isoformat") else value.isoformat()
                )
        if self.as_of is not None:
            from generalforecast import as_of as instants

            payload["as_of"] = instants.wire(self.as_of)
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
            from generalforecast import as_of as instants

            payload["as_of"] = instants.wire(as_of)
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
    spec = plan.spec
    frame = _frame_of_result({"predictions": table.to_pylist()}, spec.id)
    if plan.task == "forecast":
        assert spec.time is not None
        frame = _with_forecast_keys(frame, queries, spec.series, spec.time, spec.id)
    return frame


def _with_forecast_keys(
    frame: Any, queries: pa.Table, series: Sequence[str], time: str, id: str | None
) -> Any:
    """A forecast frame led by its queries' id (if any), series and time, one row per query.

    Every path, in process, the CLI, gf serve and hosted serving, gives a
    forecaster's predictions this shape.
    """
    import pandas as pd

    keys = pd.DataFrame(
        {name: queries.column(name).to_pylist() for name in (*series, time)}
    )
    frame = pd.concat(
        [keys, frame.drop(columns=["id"], errors="ignore").reset_index(drop=True)],
        axis=1,
    )
    if id is not None and id in queries.column_names:
        frame.insert(0, "id", queries.column(id).to_pylist())
    return frame


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
