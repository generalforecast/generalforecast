"""Portable fitted context and faithful declarative export for native estimators."""

import datetime
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import yaml

from generalforecast.contracts import PredictionResult
from generalforecast.packages import CatalogResolver, Package, PackageSpec
from generalforecast.packages.schema import infer_schema
from generalforecast.packages.validation import ensure
from generalforecast.tabular import arrow_bytes, read_table, to_arrow


def copy_table(table: pa.Table) -> pa.Table:
    """Owns immutable buffers, independent of caller-owned NumPy/Arrow storage."""
    return read_table(arrow_bytes(table), "arrow")


def forecast(
    spec: PackageSpec,
    context: pa.Table,
    queries: pa.Table,
    frequency: float,
    *,
    allow_downloads: bool = False,
) -> np.ndarray:
    """Uses the admitted native grouped kernel with one shared row per covariate."""
    from generalforecast.deployments.catalog import forecast_request
    from generalforecast.deployments.catalog_forecast import checked
    from generalforecast.local_models import code_digest, custom_model
    from generalforecast.models.catalog import parse
    from generalforecast.models.outputs import quantile_index

    def instant(value):
        if isinstance(value, datetime.datetime):
            return float(value.timestamp())
        if isinstance(value, datetime.date):
            return float((value - datetime.date(1970, 1, 1)).days * 86400)
        return value

    history, rows = context.to_pylist(), queries.to_pylist()
    identities = list(
        dict.fromkeys(tuple(row[name] for name in spec.series) for row in history)
    )

    def identity(row):
        return f"gfseries:{identities.index(tuple(row[name] for name in spec.series))}"

    origin = max(instant(row[spec.time]) for row in history)
    step = (
        int(frequency) if type(origin) is int and frequency.is_integer() else frequency
    )
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
    custom = custom_model(spec.model)
    ref = parse(spec.model, trusted=custom)
    parsed = forecast_request(
        request,
        spec.quantiles if custom else ref.entry.capability.quantiles,
        "trusted" if custom else "catalog",
        code_digest(spec.model) if custom else None,
    )
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
) -> PackageSpec:
    """Resolves metadata and validates all context rows without executing weights."""
    from generalforecast.local_models import custom_model

    ensure(
        not custom_model(model),
        "custom native models cannot be exported",
        "UnsupportedCapability",
    )
    pin = CatalogResolver().resolve(model, offline=True)
    ensure(task in pin.tasks, "model does not support estimator task")
    spec = PackageSpec.from_dict(
        {
            "name": "estimator",
            "model": pin.resolved,
            "task": task,
            "target": target,
            "context": "context.arrow",
            "outputs": ["label", "probabilities"]
            if task == "classification"
            else ["value", "quantiles"],
            **options,
        }
    )
    infer_schema(spec, pin, task, table)
    ensure(
        table.num_rows <= pin.max_context_rows,
        "context capacity exceeded",
        "CapacityExceeded",
    )
    return spec


def export_package(
    path: str | os.PathLike,
    *,
    name: str,
    spec: PackageSpec,
    context: pa.Table,
) -> Package:
    """Writes one checked immutable package without overwriting existing content."""
    value = spec.to_dict()
    value["name"] = name
    spec = PackageSpec.from_dict(value)
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
        authored = yaml.safe_dump(spec.to_dict(), sort_keys=False).encode()
        for filename, content in (
            ("gf.yaml", authored),
            ("context.arrow", arrow_bytes(context)),
        ):
            with (staging / filename).open("xb") as stream:
                os.chmod(staging / filename, 0o600)
                stream.write(content)
        Package.load(staging).check(offline=True)
        staging.rename(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return Package.load(root).check(frozen=True)


def remote_result(
    backend: Any,
    *,
    spec: PackageSpec,
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
