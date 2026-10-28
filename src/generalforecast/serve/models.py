# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What `gf serve` serves: a predictor's lock, or a catalog model, as gf-server's models file lists it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from generalforecast.errors import Error
from generalforecast.models.artifacts import ArtifactError, checkpoint
from generalforecast.models.catalog import (
    ModelRef,
    ModelRefError,
    known_as,
    not_canonical,
    parse,
)
from generalforecast.models.outputs import DEFAULT_QUANTILES
from generalforecast.predictors.files import parse_yaml
from generalforecast.predictors.models import describe
from generalforecast.predictors.validation import PredictorError, ensure
from generalforecast.service.boundary import Scope
from generalforecast.service.catalog import CatalogService, ServingSurface
from generalforecast.service.contracts import ServiceError, require
from generalforecast.tabular import arrow_bytes

ORG = "general-forecast"
LOCK_BYTES = 1 << 20


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
    """A predictor's `gf.lock`, validated as the current format.

    Raises:
        PredictorError: `LockRequired` for a lock of another version, with
            the hint to run `gf check`, or the field that is out of shape.
    """
    from generalforecast.predictors.lock import validate

    raw = path.read_bytes()
    ensure(len(raw) <= LOCK_BYTES, "gf.lock is too large")
    try:
        lock = parse_yaml(raw)
    except PredictorError:
        raise PredictorError("ValidationError", f"{path} is not valid YAML") from None
    try:
        return validate(lock)
    except PredictorError as error:
        code = "LockRequired" if error.code == "UnsupportedLockVersion" else error.code
        raise PredictorError(code, f"{path}: {error.message}") from None


def _predictor(path: Path, directory: Path, *, allow_downloads: bool) -> dict[str, Any]:
    """A locked predictor's entry, with a snapshot's context rows written out.

    A directory needs its `gf.yaml`, and its lock must be exactly the one
    `gf check` writes for it now (`resolution.verified`, as `gf push` and
    `gf.Predictor` require). A lone `gf.lock` file has no manifest to hold
    it to; it serves a request-context predictor, admitted as hosted
    serving admits a pushed lock. The entry is the one hosted serving
    publishes for the same lock and rows.
    """
    from generalforecast.predictors.resolution import verified
    from generalforecast.predictors.runtime.context import served_context
    from generalforecast.predictors.spec import load
    from generalforecast.predictors.tables import read_frame

    if path.is_file() and path.name != "gf.yaml":
        root, lock_path = path.parent, path
    else:
        root = path if path.is_dir() else path.parent
        lock_path = root / "gf.lock"
        ensure(
            lock_path.is_file(),
            "gf.lock is required; run gf check first",
            "LockRequired",
        )
        ensure(
            (root / "gf.yaml").is_file(),
            f"no gf.yaml beside {lock_path}; serve the gf.lock file itself to "
            "serve a lone request-context lock",
            "LockRequired",
        )
    lock = read_lock(lock_path)
    predictor = lock["predictor"]
    from generalforecast.predictors.models import qualification

    runs = qualification(predictor["model"])
    if not runs.serve:
        raise runs.refusal("serve")
    if (root / "gf.yaml").is_file():
        try:
            verified(load(root), root)
        except Error as error:
            code = "LockRequired" if error.param == "lock" else "InvalidPredictor"
            if error.reason == "snapshot_changed":
                code = "SnapshotMismatch"
            raise PredictorError(code, error.message) from None
    ref = canonical(predictor["model"])
    _verified(ref, allow_downloads=allow_downloads)
    rows = None
    snapshot = None
    if predictor["context"]["kind"] == "snapshot":
        ensure(
            path.is_dir(),
            "a lone gf.lock serves only request-context predictors; serve the predictor directory instead",
            "ContextRequired",
        )
        try:
            table = read_frame(root, predictor["context"]["path"])
        except Error as error:
            raise PredictorError("MissingFile", error.message) from None
        rows = served_context(lock, table)
    ensure(
        predictor["context"]["kind"] in ("snapshot", "request"),
        "live contexts need the hosted plane; serve a snapshot or request predictor",
        "UnsupportedCapability",
    )
    plan = compiled(lock, rows)
    if rows is not None:
        from generalforecast.predictors.runtime.tabular import bound_order

        snapshot = directory / "context.arrow"
        snapshot.write_bytes(arrow_bytes(bound_order(plan, rows)))
    return published_entry(plan, snapshot=snapshot)


def compiled(lock: Any, rows: Any) -> Any:
    """The compiled plan of a lock and its context rows, with refusals as `PredictorError`."""
    from generalforecast.predictors.models import CatalogResolver
    from generalforecast.predictors.runtime.compile import compile_lock

    try:
        return compile_lock(lock, rows, resolver=CatalogResolver("local"))
    except Error as error:
        raise PredictorError(error.code, error.message) from None


def published_entry(plan: Any, *, snapshot: Path | None = None) -> dict[str, Any]:
    """A predictor's entry in gf-server's models file, from its compiled plan.

    `gf serve` and hosted serving both publish this for a lock and its
    context rows (`compile_lock`), so gf-server binds the same predictor in
    both: the same schema, inputs, past and future covariates, frequency,
    selection strategy and lock digest.

    Args:
        plan: A `CompiledPlan`.
        snapshot: The file a snapshot's bound rows were written to, if any.

    The entry's `predictor.context` is `snapshot` or `request`; hosted
    serving binds a live context's captured rows as a snapshot. A forecast
    entry's `frequency` is the step on the time axis, `declared` the same
    step when the lock declares it, and `calendar` a calendar frequency's
    alias (`MS`, `2QE`, `W-MON`), whose UTC periods gf-server counts;
    `related` the lock's declared groups, every series on its own when None.
    `types` maps each column whose kind the lock records to that kind.
    """
    from generalforecast.predictors import frequency as _frequency

    spec = plan.spec
    ref = parse(plan.model.resolved)
    resolved = resolved_id(ref)
    schema = plan.schema.to_dict()
    task = plan.task
    inputs = [
        column["name"] for column in schema["columns"] if column["role"] == "input"
    ] or list(spec.inputs.include or ())
    context = spec.context.mode
    quantiles = (
        list(spec.quantiles or DEFAULT_QUANTILES) if task != "classification" else []
    )
    forecast = None
    if task == "forecast":
        assert spec.frequency is not None
        step = float(_frequency.step(spec.frequency))
        forecast = {
            "series": list(spec.series),
            "time": spec.time,
            "horizon": spec.horizon,
            "past_inputs": list(spec.past_inputs),
            "future_inputs": list(spec.future_inputs),
            "frequency": step,
            "declared": step,
            "calendar": spec.frequency
            if _frequency.is_calendar(spec.frequency)
            else None,
            "related": None if spec.related is None else list(spec.related),
        }
        inputs = list(spec.forecast_inputs)
    selection = spec.selection
    lock_digest = plan.lock_digest or plan.plan_digest
    # The kinds the lock declares, by column, as `context.typed` reads a
    # request context with them; gf-server types a context the lock holds
    # no schema for by the same map.
    types = {**dict(spec.inputs.types), **dict(spec.role_types)}
    if spec.target_type is not None:
        types[spec.target] = spec.target_type
    if (
        task == "forecast"
        and spec.time is not None
        and spec.time not in types
        and isinstance(spec.frequency, str)
    ):
        types[spec.time] = "timestamp"
    return {
        "name": spec.name,
        "reference": str(ref),
        "worker": ref.entry.id,
        "resolved": resolved,
        "outputs": list(plan.outputs),
        "license_source": ref.entry.artifact.license_source,
        "description": {
            "name": spec.name,
            "kind": "predictor",
            "model": resolved,
            "task": task,
            "target": spec.target,
            "inputs": inputs,
            "context": context,
            "context_rows": plan.context_rows if context == "snapshot" else None,
            "lock_digest": lock_digest,
        },
        "predictor": {
            "schema": schema,
            "task": task,
            "target": spec.target,
            "inputs": inputs,
            "id": spec.id,
            "context": context if context != "live" else "snapshot",
            "quantiles": quantiles,
            "context_digest": plan.data_digest,
            "lock_digest": lock_digest,
            "snapshot": None if snapshot is None else str(snapshot),
            "forecast": forecast,
            "strategy": None
            if selection is None or task == "forecast"
            else selection.strategy_spec(),
            "max_rows": spec.context.max_rows,
            "types": types,
        },
    }
