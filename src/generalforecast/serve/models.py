# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What `gf serve` serves: a predictor's lock, or a catalog model, as gf-server's models file lists it."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from generalforecast.deployments.boundary import Scope
from generalforecast.deployments.catalog import CatalogService, ServingSurface
from generalforecast.deployments.contracts import ServiceError, require
from generalforecast.models.artifacts import ArtifactError, checkpoint
from generalforecast.models.catalog import (
    ModelRef,
    ModelRefError,
    known_as,
    not_canonical,
    parse,
)
from generalforecast.models.outputs import DEFAULT_QUANTILES
from generalforecast.predictors.models import describe
from generalforecast.predictors.validation import PredictorError, ensure
from generalforecast.tabular import arrow_bytes

ORG = "general-forecast"
LOCK_VERSION = 1
LOCK_BYTES = 1 << 20
# The lock's column types, as gf-server's `Schema` spells them physically.
PHYSICAL = {
    "float": "float64",
    "float64": "float64",
    "int": "int64",
    "int64": "int64",
    "0/1": "int64",
    "bool": "boolean",
    "boolean": "boolean",
    "category": "string",
    "text": "string",
    "string": "string",
    "date": "date",
    "datetime": "timestamp",
    "timestamp": "timestamp",
}
SEMANTIC = {"category": "category", "date": "datetime", "datetime": "datetime"}
INPUT_TYPES = {"0/1": "float"}
OUTPUTS = {
    "classification": ("label", "probabilities"),
    "regression": ("value",),
    "forecast": ("value", "quantiles"),
}


def canonical(reference: str) -> ModelRef:
    """The catalog model a canonical id names; a bare name gets the id to use.

    Raises:
        PredictorError: `ValidationError` with the hint for a bare or `catalog/`
            spelling, `ModelNotFound` for a name the catalog lacks.
    """
    try:
        return parse(reference)
    except ModelRefError as error:
        if (known := known_as(reference.removeprefix("catalog/"))) is not None:
            raise PredictorError(
                "ValidationError", not_canonical(reference, known)
            ) from None
        raise PredictorError("ModelNotFound", str(error)) from None


def resolved_id(ref: ModelRef) -> str:
    """The pinned id results report, `general-forecast/<name>@sha256:<digest>`."""
    pin = describe(str(ref)).resolved
    return f"{ORG}/{pin.split('/', 1)[1]}" if "/" in pin else f"{ORG}/{pin}"


def served(
    target: str, directory: Path, *, allow_downloads: bool = True
) -> dict[str, Any]:
    """The served model's entry in gf-server's models file, with its weights verified.

    The target is a predictor directory, its `gf.lock`, or a canonical model
    id. A snapshot predictor's context rows go to `directory`, where
    gf-server binds them once into a durable table; a request-context
    predictor binds each request's rows into an ephemeral table.

    Raises:
        PredictorError: If the target is not a predictor or catalog model this
            server may serve, or its weights are missing or differ from
            their pins.
    """
    require(type(target) is str and bool(target), "serve target is required")
    path = Path(target)
    try:
        if path.is_dir() or path.is_file():
            return _predictor(path, directory, allow_downloads=allow_downloads)
    except ServiceError as error:
        raise PredictorError(error.code, error.message) from None
    except OSError as error:
        raise PredictorError("MissingFile", f"cannot read {target!r}") from error
    if (
        target.startswith((".", "/", "~"))
        or target.endswith((".lock", ".yaml"))
        or (target.count("/") > 1)
    ):
        raise PredictorError(
            "NotFound",
            f"{target!r} is not a predictor directory, a gf.lock or a model id",
        )
    ref = canonical(target)
    return catalog_model(ref, surface="self-hosted", allow_downloads=allow_downloads)


def catalog_model(
    ref: ModelRef, *, surface: ServingSurface, allow_downloads: bool
) -> dict[str, Any]:
    """A catalog model's entry in gf-server's models file, its weights verified.

    `gf serve` passes the `self-hosted` surface, where the operator may serve
    pinned models whose license forbids hosting them on General Forecast
    infrastructure; the hosted pool passes `remote`, which keeps those out.

    Raises:
        PredictorError: If the surface may not serve the model, or its weights
            are missing or differ from their pins.
    """
    try:
        description = CatalogService(surface=surface).get(
            Scope("local", "load"), str(ref)
        )
    except ServiceError as error:
        raise PredictorError(error.code, error.message) from None
    _verified(ref, surface=surface, allow_downloads=allow_downloads)
    pin = describe(str(ref))
    description = dict(description, model=resolved_id(ref))
    return {
        "name": ref.entry.name,
        "reference": str(ref),
        "worker": ref.entry.id,
        "resolved": resolved_id(ref),
        "outputs": list(pin.outputs),
        "description": description,
        "license_source": ref.entry.artifact.license_source,
    }


def _verified(
    ref: ModelRef, *, surface: ServingSurface = "self-hosted", allow_downloads: bool
) -> None:
    try:
        checkpoint(ref.entry, surface=surface, allow_downloads=allow_downloads)
    except ArtifactError as error:
        raise PredictorError(error.code, str(error)) from None


def read_lock(path: Path) -> dict[str, Any]:
    """A predictor's `gf.lock`, checked to be a version-1 lock.

    Raises:
        PredictorError: If the file is not a lock this server reads.
    """
    import yaml

    raw = path.read_bytes()
    ensure(len(raw) <= LOCK_BYTES, "gf.lock is too large")
    try:
        lock = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise PredictorError(
            "ValidationError", f"{path} is not valid YAML: {error}"
        ) from None
    ensure(isinstance(lock, dict), f"{path} is not a gf.lock")
    ensure(
        lock.get("lock_version") == LOCK_VERSION,
        f"{path} has lock_version {lock.get('lock_version')!r}; run gf check to write version {LOCK_VERSION}",
        "LockRequired",
    )
    predictor = lock.get("predictor")
    ensure(isinstance(predictor, dict), f"{path} has no predictor")
    for field in ("name", "model", "target", "inputs", "context"):
        ensure(field in predictor, f"{path} lacks predictor.{field}; run gf check")
    ensure(
        isinstance(predictor["inputs"], list) and len(predictor["inputs"]) > 0,
        f"{path} resolves zero input columns; run gf check",
    )
    ensure(
        predictor["context"].get("kind") in ("snapshot", "request"),
        "live contexts need the hosted plane; serve a snapshot or request predictor",
        "UnsupportedCapability",
    )
    lock["lock_digest"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    return lock


def _predictor(path: Path, directory: Path, *, allow_downloads: bool) -> dict[str, Any]:
    """A locked predictor's entry, with a snapshot's context rows written out."""
    if path.is_dir():
        root, lock_path = path, path / "gf.lock"
        ensure(
            lock_path.is_file(),
            "gf.lock is required; run gf check first",
            "LockRequired",
        )
    else:
        root, lock_path = path.parent, path
    lock = read_lock(lock_path)
    predictor = lock["predictor"]
    ref = canonical(predictor["model"])
    _verified(ref, allow_downloads=allow_downloads)
    snapshot = None
    if predictor["context"]["kind"] == "snapshot":
        ensure(
            path.is_dir(),
            "a lone gf.lock serves only request-context predictors; serve the predictor directory instead",
            "ContextRequired",
        )
        snapshot = directory / "context.arrow"
        snapshot.write_bytes(arrow_bytes(_snapshot_rows(root, predictor)))
    return served_entry(lock, ref, snapshot)


def _context_rows(root: Path, table: Any, predictor: dict[str, Any]) -> Any:
    """The snapshot rows the manifest keeps as context.

    With `gf.yaml` beside the lock this is the split `gf check` resolved: rows
    with a target, less the rows `until` or `holdout` make queries. A lock
    without its manifest keeps every row with a target. A classifier's target
    keeps the integer labels the lock's classes have.
    """
    target = predictor["target"]
    if not (root / "gf.yaml").is_file():
        return table.filter(table[target].is_valid())
    from generalforecast.errors import InvalidInput
    from generalforecast.predictors import load, split
    from generalforecast.predictors.tables import whole_numbers

    try:
        context, _ = split(table, load(root))
    except InvalidInput as error:
        raise PredictorError("InvalidPredictor", error.message) from None
    whole = whole_numbers(context[target]) if predictor.get("classes") else None
    if whole is not None:
        index = context.schema.get_field_index(target)
        context = context.set_column(index, target, whole)
    return context


def _snapshot_rows(root: Path, predictor: dict[str, Any]):
    """A snapshot's context rows: the file the lock names, split as the manifest says."""
    import pyarrow as pa

    context = predictor["context"]
    named = context.get("path")
    candidates = (
        [root / named]
        if named
        else [
            root / f"context.{suffix}" for suffix in ("parquet", "csv", "json", "arrow")
        ]
    )
    file = next((candidate for candidate in candidates if candidate.is_file()), None)
    ensure(
        file is not None,
        f"snapshot context not found in {root}; the lock names "
        f"{named or 'context.parquet, context.csv, context.json or context.arrow'}",
        "MissingFile",
    )
    assert file is not None
    if file.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(file)
    elif file.suffix == ".arrow":
        with pa.ipc.open_stream(file.read_bytes()) as reader:
            table = reader.read_all()
    elif file.suffix == ".csv":
        from pyarrow import csv

        table = csv.read_csv(file)
    else:
        import pyarrow.json as pj

        table = pj.read_json(file)
    target = predictor["target"]
    ensure(
        target in table.column_names, f"target column {target!r} is not in {file.name}"
    )
    table = _context_rows(root, table, predictor)
    columns = [column["name"] for column in predictor["inputs"]]
    for role in ("id", "time"):
        if predictor.get(role):
            columns.append(predictor[role])
    columns.extend(predictor.get("series") or [])
    columns.append(target)
    missing = [column for column in columns if column not in table.column_names]
    ensure(not missing, f"columns {missing} are not in {file.name}")
    if context.get("rows") is not None:
        ensure(
            table.num_rows == context["rows"],
            f"{file.name} has {table.num_rows} context rows; the lock says {context['rows']}; run gf check",
        )
    return pa.table({column: table[column] for column in dict.fromkeys(columns)})


def _column(
    name: str, kind: str, role: str, *, nullable: bool = False
) -> dict[str, Any]:
    physical = PHYSICAL.get(kind)
    ensure(
        physical is not None,
        f"column {name!r} has an unknown type {kind!r}; run gf check",
    )
    return {
        "name": name,
        "type": physical,
        "nullable": nullable,
        "role": role,
        "semantic": SEMANTIC.get(kind),
    }


def served_entry(
    lock: dict[str, Any], ref: ModelRef, snapshot: Path | None = None
) -> dict[str, Any]:
    """A predictor's entry in gf-server's models file, from its version-1 lock.

    `snapshot` names the file a snapshot's rows were written to, if any were.
    """
    predictor = lock["predictor"]
    task = ref.entry.capability.task
    target_type = predictor.get("target_type") or (
        "float" if task != "classification" else "0/1"
    )
    frequency = predictor.get("frequency")
    columns = [
        _column(
            column["name"], INPUT_TYPES.get(column["type"], column["type"]), "input"
        )
        for column in predictor["inputs"]
    ]
    if predictor.get("id"):
        columns.insert(
            0, _column(predictor["id"], predictor.get("id_type") or "text", "id")
        )
    if predictor.get("time"):
        stepped = isinstance(frequency, (int, float)) and not isinstance(
            frequency, bool
        )
        columns.append(
            _column(predictor["time"], "int" if stepped else "datetime", "time")
        )
    for series in predictor.get("series") or []:
        columns.append(_column(series, "category", "series"))
    columns.append(_column(predictor["target"], target_type, "target", nullable=True))
    context = predictor["context"]
    schema = {
        "columns": columns,
        "rows": context.get("rows"),
        "classes": predictor.get("classes"),
        "frequency": frequency if isinstance(frequency, (int, float)) else None,
    }
    quantiles = (
        list(predictor.get("quantiles") or DEFAULT_QUANTILES)
        if task != "classification"
        else []
    )
    inputs = [column["name"] for column in predictor["inputs"]]
    resolved = resolved_id(ref)
    return {
        "name": predictor["name"],
        "reference": str(ref),
        "worker": ref.entry.id,
        "resolved": resolved,
        "outputs": list(predictor.get("outputs") or OUTPUTS[task]),
        "license_source": ref.entry.artifact.license_source,
        "description": {
            "name": predictor["name"],
            "kind": "predictor",
            "model": resolved,
            "task": task,
            "target": predictor["target"],
            "inputs": inputs,
            "context": context["kind"],
            "context_rows": context.get("rows")
            if context["kind"] == "snapshot"
            else None,
            "lock_digest": lock["lock_digest"],
        },
        "predictor": {
            "schema": schema,
            "task": task,
            "target": predictor["target"],
            "inputs": inputs,
            "id": predictor.get("id"),
            "context": context["kind"],
            "quantiles": quantiles,
            "context_digest": context.get("digest"),
            "lock_digest": lock["lock_digest"],
            "snapshot": None if snapshot is None else str(snapshot),
            "forecast": None
            if task != "forecast"
            else {
                "series": list(predictor.get("series") or []),
                "time": predictor["time"],
                "horizon": predictor["horizon"],
                "past_inputs": [
                    column["name"]
                    for column in predictor["inputs"]
                    if column.get("role", "past") == "past"
                ],
                "future_inputs": [
                    column["name"]
                    for column in predictor["inputs"]
                    if column.get("role") == "future"
                ],
                "frequency": schema["frequency"],
                "declared": schema["frequency"],
            },
        },
    }


def published_entry(lock: Any) -> dict[str, Any]:
    """A hosted revision's entry, from the control plane's `ExecutionLock`.

    The control plane holds execution locks; this reads the fields the
    predictor lock carries from it, so the hosted and local paths serve the
    same entry.
    """
    ref = parse(lock.model.resolved)
    schema = lock.schema.to_dict()
    columns = {column["name"]: column for column in schema.get("columns", [])}
    spec = lock.spec
    target = spec.target
    target_type = columns.get(target, {}).get("type") or "float"
    inputs = []
    for name in lock.schema.features:
        column = columns.get(name, {})
        kind = column.get("semantic") or column.get("type") or "float"
        role = {"past": "past", "future": "future"}.get(
            "future" if name in (getattr(spec, "future_inputs", None) or ()) else "past"
        )
        inputs.append({"name": name, "type": kind, "role": role})
    frequency = getattr(lock.schema, "frequency", None)
    predictor = {
        "name": spec.name,
        "model": f"{ORG}/{ref}",
        "target": target,
        "target_type": target_type,
        "classes": schema.get("classes"),
        "id": spec.id,
        "id_type": columns.get(spec.id, {}).get("type") if spec.id else None,
        "time": getattr(spec, "time", None),
        "series": list(getattr(spec, "series", None) or []),
        "frequency": frequency,
        "horizon": getattr(spec, "horizon", None),
        "quantiles": list(spec.quantiles) if getattr(spec, "quantiles", None) else None,
        "inputs": inputs,
        "context": {
            "kind": spec.context.mode,
            "digest": lock.data_digest,
            "rows": lock.context_rows if spec.context.mode == "snapshot" else None,
        },
    }
    return served_entry(
        {
            "lock_version": LOCK_VERSION,
            "predictor": predictor,
            "lock_digest": lock.lock_digest,
        },
        ref,
    )
