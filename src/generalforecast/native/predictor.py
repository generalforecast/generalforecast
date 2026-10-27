# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Portable fitted context and faithful declarative export for native estimators."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from generalforecast.contracts import PredictionResult
from generalforecast.predictors import frequency as _frequency
from generalforecast.predictors.models import CatalogResolver
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.runtime.schema import infer_schema
from generalforecast.predictors.validation import ensure
from generalforecast.tabular import arrow_bytes, read_table, to_arrow


def copy_table(table: pa.Table) -> pa.Table:
    """Owns immutable buffers, independent of caller-owned NumPy/Arrow storage."""
    return read_table(arrow_bytes(table), "arrow")


def forecast(
    spec: ExecutionPlan,
    context: pa.Table,
    queries: pa.Table,
    frequency: float | str,
    *,
    allow_downloads: bool = True,
) -> np.ndarray:
    """Uses the admitted native grouped kernel with one shared row per covariate.

    `frequency` is the plan's frequency when it declares none: a number of
    periods or seconds, or a spelling. Times sit on its axis as
    `frequency.coordinate` places them, in UTC: a calendar alias such as
    `MS` steps through period ordinals one at a time.
    """
    from generalforecast.models.catalog import parse
    from generalforecast.models.outputs import quantile_index
    from generalforecast.service.catalog import forecast_request
    from generalforecast.service.catalog_forecast import checked

    cadence = spec.frequency if spec.frequency is not None else frequency

    def instant(value):
        return _frequency.coordinate(value, cadence)

    history, rows = context.to_pylist(), queries.to_pylist()
    identities = list(
        dict.fromkeys(tuple(row[name] for name in spec.series) for row in history)
    )

    def identity(row):
        return f"gfseries:{identities.index(tuple(row[name] for name in spec.series))}"

    origin = max(instant(row[spec.time]) for row in history)
    step: float | int = _frequency.step(cadence)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(identity(row), []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: rows[index][spec.time])
    future = [
        {
            "key": key,
            "series": key,
            "origin": origin,
            "group": [f"gfseries:{i}" for i in range(len(identities))],
            "future_covariates": {
                name: [rows[index][name] for index in indices]
                for name in spec.future_inputs
            },
        }
        for key, indices in groups.items()
    ]
    request = {
        "model": spec.model,
        "task": "forecast",
        "target": spec.target,
        "features": [],
        "context": [
            {
                "key": str(index),
                "series": identity(row),
                "time": instant(row[spec.time]),
                "value": row[spec.target],
                "covariates": {name: row[name] for name in spec.forecast_inputs},
            }
            for index, row in enumerate(history)
        ],
        "queries": future,
        "options": {
            "series": {
                "step": step,
                "horizon": spec.horizon,
                "quantiles": list(spec.quantiles),
                "known_covariates": list(spec.future_inputs),
            }
        },
    }
    ref = parse(spec.model)
    parsed = forecast_request(request, ref.entry.capability.quantiles)
    values, levels = checked(
        ref, parsed, surface="local", allow_downloads=allow_downloads
    )
    columns = [quantile_index(level, levels) for level in spec.quantiles]
    output = np.empty((len(rows), len(spec.quantiles)), dtype=float)
    for query, target in zip(parsed.queries, values, strict=True):
        for offset, index in enumerate(groups[query.series]):
            output[index] = target[offset, columns]
    return output


def context_table(rows: Any, targets: Any, names: list[str], target: str) -> pa.Table:
    """Copies exactly the rows used by fit, including the original target labels."""
    table = to_arrow(rows, columns=names if isinstance(rows, np.ndarray) else None)
    ensure(target not in table.column_names, "target name conflicts with an input")
    return copy_table(
        to_arrow(table.append_column(target, pa.array(np.asarray(targets))))
    )


def target_name(y: Any, features: list[str]) -> str:
    """Keeps an explicitly named target, or chooses an unambiguous reserved name."""
    name = getattr(y, "name", None)
    if isinstance(name, str) and name:
        ensure(name not in features, "target name conflicts with an input")
        return name
    name = "target"
    while name in features:
        name = "_" + name
    return name


def admitted_spec(
    model: Any, task: str, table: pa.Table, *, target: str, **options: Any
) -> ExecutionPlan:
    """Resolves metadata and validates all context rows without executing weights."""
    pin = CatalogResolver().resolve(model, offline=True)
    ensure(task in pin.tasks, "model does not support estimator task")
    options.setdefault(
        "outputs",
        ["label", "probabilities"]
        if task == "classification"
        else ["value", "quantiles"],
    )
    spec = ExecutionPlan.build(
        "estimator", pin.resolved, task=task, target=target, **options
    )
    infer_schema(spec, pin, task, table)
    ensure(
        table.num_rows <= pin.max_context_rows,
        "context capacity exceeded",
        "CapacityExceeded",
    )
    return spec


def export_predictor(
    path: str | os.PathLike,
    *,
    name: str,
    spec: ExecutionPlan,
    context: pa.Table,
) -> Path:
    """Writes a fitted estimator as a predictor directory: gf.yaml, gf.lock, snapshot.

    The directory is what `gf.Predictor(...).save` writes and `gf check`,
    `gf eval`, `gf serve` and `gf.Predictor(path)` read. It is staged next to
    `path` and moved there whole once its manifest resolves; an existing
    `path` is never overwritten.
    """
    from pyarrow import parquet

    from generalforecast.models.catalog import ModelRef, parse
    from generalforecast.predictors import resolve, write_lock, write_manifest
    from generalforecast.predictors.spec import PredictorSpec

    # The manifest names the model with its `@K`/`*N` variant; the lock holds
    # the digest the fit pinned.
    ref = parse(spec.model)
    manifest: dict[str, Any] = {
        "name": name,
        "model": str(ModelRef(ref.name, ref.codes, ref.estimators)),
    }
    if spec.target is not None:
        manifest["target"] = spec.target
    if spec.task == "forecast":
        manifest["time"] = spec.time
        if spec.series:
            manifest["series"] = list(spec.series)
        if spec.frequency is not None:
            manifest["frequency"] = spec.frequency
        manifest["horizon"] = spec.horizon
        if spec.quantiles:
            manifest["quantiles"] = list(spec.quantiles)
        covariates = {
            key: list(columns)
            for key, columns in (
                ("past", spec.past_inputs),
                ("future", spec.future_inputs),
            )
            if columns
        }
        if covariates:
            manifest["inputs"] = covariates
    elif spec.inputs.include is not None:
        manifest["inputs"] = list(spec.inputs.include)
    manifest["context"] = {"snapshot": SNAPSHOT}
    root = Path(os.path.abspath(path))
    ensure(
        not root.exists() and not root.is_symlink(), "export destination already exists"
    )
    ensure(
        root.parent.is_dir() and not root.parent.is_symlink(),
        "export parent must be an existing directory",
    )
    staging = Path(tempfile.mkdtemp(prefix=".gf-export-", dir=root.parent))
    try:
        parquet.write_table(context, staging / SNAPSHOT)
        os.chmod(staging / SNAPSHOT, 0o600)
        predictor = PredictorSpec.from_dict(manifest)
        write_manifest(predictor, staging)
        write_lock(resolve(predictor, staging), staging)
        staging.rename(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return root


SNAPSHOT = "context.parquet"


def remote_result(
    backend: Any,
    *,
    spec: ExecutionPlan,
    context: pa.Table,
    queries: pa.Table,
    features: list[str],
    options: dict,
) -> PredictionResult:
    """Uses the SDK's genuine synchronous prediction port, never another transport."""
    result = backend.predict(
        model=spec.model,
        task=spec.task,
        target=spec.target,
        context=context,
        queries=queries,
        features=features,
        options={"outputs": list(spec.outputs), **options},
    )
    ensure(
        isinstance(result, PredictionResult),
        "backend must return PredictionResult",
        "ProtocolError",
    )
    ensure(
        result.model == spec.model and result.task == spec.task,
        "backend model/task differs from fitted estimator",
        "ProtocolError",
    )
    ensure(
        result.table.num_rows == queries.num_rows,
        "backend prediction count differs from queries",
        "ProtocolError",
    )
    return result
