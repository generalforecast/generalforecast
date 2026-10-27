# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resolving a manifest against its context: types, split, selection and pin."""

import dataclasses
import difflib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.compute as _compute

from generalforecast import tabular
from generalforecast.errors import InvalidInput
from generalforecast.predictors import _registry, models
from generalforecast.predictors.selection import SelectionInfo
from generalforecast.predictors.selection import resolve as resolve_selection
from generalforecast.predictors.spec import (
    DEFAULT_QUANTILES,
    PredictorSpec,
    validate,
)
from generalforecast.predictors.tables import (
    TableInfo,
    inspect_frame,
    labels,
    read_frame,
    to_table,
)

pc: Any = _compute  # ty cannot see Arrow's generated compute functions
NUMERIC = ("int", "float", "0/1")


@dataclass(frozen=True)
class ContextInfo:
    """The resolved context.

    Attributes:
        kind: `snapshot`, `request` or `live`.
        rows: Context rows, when known.
        query_rows: Query rows split off a snapshot, when known.
        digest: The context rows' digest, when known.
    """

    kind: str
    rows: int | None = None
    query_rows: int | None = None
    digest: str | None = None


@dataclass(frozen=True)
class Resolution:
    """A manifest resolved against its context and the model catalog.

    Attributes:
        name: The predictor name.
        model: The canonical model id, without a digest.
        model_digest: The pinned artifact digest, `sha256:...`, if the model has one.
        task: `classification`, `regression` or `forecast`.
        target: The target column.
        target_type: The target's kind, when the context is known.
        classes: A classifier's classes in the context, when known.
        id: The id role column, if any.
        time: The time role column, if any.
        series: The series role columns.
        frequency: A forecaster's cadence.
        horizon: A forecaster's steps.
        quantiles: A forecaster's output levels.
        inputs: Each input column with its type (None when unknown).
        inferred: Each inferred field with the reason it was chosen.
        context: The resolved context.
        selection: The resolved selection.
        lock_changes: `(op, key, old, new)` against the previous lock, `op` in `+ - ~`.
        files: The pinned model files by name.
        spec: The manifest this resolves.
        input_roles: Each input's role, aligned with `inputs`: `input` for
            a tabular model; `past` or `future` for a forecaster, whose
            inputs the manifest does not mark `future` are `past`.
        time_type: The time column's kind.
        series_types: Each series column's kind.
        outputs: What predictions carry.
        resources: The hosted serving resources the manifest declares.
        manifest: The digest of the normalized manifest.
        related: A forecaster's declared groups: None for independent series,
            `()` for one group of every series, else the series columns
            whose values name each group.
        context_table: The context rows the lock's digest names, when known.
        query_table: A snapshot's query rows, when known.
    """

    name: str
    model: str
    model_digest: str | None
    task: str
    target: str
    target_type: str | None
    classes: tuple[Any, ...] | None
    id: str | None
    time: str | None
    series: tuple[str, ...]
    inputs: tuple[tuple[str, str | None], ...]
    inferred: Mapping[str, str]
    context: ContextInfo
    selection: SelectionInfo
    lock_changes: tuple[tuple[str, str, Any, Any], ...] = ()
    frequency: str | int | float | None = None
    horizon: int | None = None
    id_type: str | None = None
    quantiles: tuple[float, ...] = ()
    files: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    spec: PredictorSpec | None = field(default=None, compare=False, repr=False)
    input_roles: tuple[str, ...] = ()
    time_type: str | None = None
    series_types: tuple[str | None, ...] = ()
    outputs: tuple[str, ...] = ()
    resources: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    manifest: str | None = None
    related: tuple[str, ...] | None = None
    context_table: Any = field(default=None, compare=False, repr=False)
    query_table: Any = field(default=None, compare=False, repr=False)

    @property
    def pinned(self) -> str:
        """The model id with its digest, `general-forecast/<name>@sha256:...`."""
        if self.model_digest is None:
            return self.model
        base, _, variant = self.model.partition("@")
        return f"{base}@{self.model_digest}" + (f"@{variant}" if variant else "")

    @property
    def input_names(self) -> tuple[str, ...]:
        """The input column names in order."""
        return tuple(name for name, _ in self.inputs)


def resolve(
    spec: PredictorSpec,
    root: str | Path | None = None,
    *,
    online: bool = False,
    upgrade: bool = False,
    locked: bool = False,
    rows: Any = None,
) -> Resolution:
    """Resolves a manifest in its directory, or against the rows of a request.

    The resolution is compiled as the runtime compiles it, so a manifest
    `gf check` accepts is one `gf.Predictor`, `gf serve` and hosted serving
    accept, and the other way round.

    Args:
        spec: The manifest.
        root: The predictor directory, where the snapshot and `gf.lock` live;
            None for a manifest without one, such as an ad hoc call.
        online: Confirm the model pin against the upstream registry.
        upgrade: Accept a model digest that differs from the previous lock.
        locked: Fail if the result differs from the previous lock.
        rows: A request context's rows, when the call carries them; types,
            classes and frequency are then inferred as from a snapshot.

    Returns:
        The resolution, with `lock_changes` against any previous `gf.lock`.

    Raises:
        InvalidInput: On any contract violation, pointing at the manifest line
            when there is one.
    """
    from generalforecast.predictors import lock
    from generalforecast.predictors.runtime.context import (
        context_columns,
        project,
        snapshot_context,
        typed,
    )

    validate(spec)
    assert spec.target is not None
    inferred: dict[str, str] = {}
    if spec.marks and "name" not in spec.marks:
        inferred["name"] = "the directory name"
    task = _registry.task_of(spec.model)
    inferred["task"] = f"{spec.model} is a {task} model"
    pin = _pin(spec, online=online)
    limits = _registry.limits(spec.model)
    previous = lock.read_lock(root) if root is not None else None
    digest = _model_digest(spec, pin, previous, upgrade=upgrade)

    table: pa.Table | None = None
    info: TableInfo | None = None
    if spec.context.kind == "snapshot":
        assert spec.context.snapshot is not None
        if root is None:
            raise spec.error(
                f"snapshot `{spec.context.snapshot}` needs the predictor directory",
                param="context.snapshot",
                hint="resolve the manifest in its directory",
            )
        table = read_frame(Path(root), spec.context.snapshot)
        info = inspect_frame(table, path=spec.context.snapshot)
    elif rows is not None:
        table = to_table(rows)
        info = inspect_frame(table, path="the context")
    elif spec.inputs is None and not (spec.past or spec.future):
        raise spec.error(
            f"`inputs` must be listed for a {spec.context.kind} context",
            param="inputs",
            hint=f"a {spec.context.kind} context has no rows to infer from; "
            "write `inputs: [spend, plan]`",
        )
    if info is not None:
        _check_columns(spec, info)
    inputs, roles = _inputs(spec, info, limits, inferred, task)
    kinds = info.kinds() if info is not None else dict(spec.types)
    draft = {
        "task": task,
        "target": spec.target,
        "id": spec.id,
        "time": spec.time,
        "series": list(spec.series),
        "inputs": [{"name": name, "type": kind} for name, kind in inputs],
        "id_type": kinds.get(spec.id) if spec.id else None,
        "time_type": kinds.get(spec.time) if spec.time else None,
        "series_types": [kinds.get(column) for column in spec.series],
        "context": {
            "until": spec.context.until,
            "holdout": None
            if spec.context.holdout is None
            else {
                "fraction": spec.context.holdout.fraction,
                "seed": spec.context.holdout.seed,
            },
        },
    }
    frame: pa.Table | None = None
    queries: pa.Table | None = None
    context_rows = query_rows = context_digest = None
    if spec.context.kind == "snapshot":
        assert table is not None
        frame, queries = snapshot_context(table, draft, error=spec.error)
        if frame.num_rows == 0:
            raise spec.error(
                f"every row of {spec.context.snapshot} is a query",
                param="context",
                because="no row has a target value to learn from",
                hint="fill in `target` for the rows the model should learn from",
            )
        context_rows, query_rows = frame.num_rows, queries.num_rows
        inferred["queries"] = _query_reason(spec, query_rows)
        context_digest = tabular.table_digest(frame)
    elif table is not None:
        frame = typed(
            project(
                table,
                context_columns(draft),
                spec.target,
                classifier=task == "classification",
            ),
            draft,
        )
        context_rows = frame.num_rows

    target_type, classes = _target(spec, task, frame, pin, inferred)
    frequency = spec.frequency
    quantiles: tuple[float, ...] = ()
    if task == "forecast":
        if frame is not None:
            frequency = _frequency(spec, frame, inferred)
        elif frequency is None:
            raise spec.error(
                f"a {spec.context.kind} context has no rows to infer `frequency` from",
                param="frequency",
                hint="add `frequency: 1d`, or the step your times follow",
            )
    if task != "classification":
        # Levels are a set; the lock keeps them in ascending order.
        quantiles = tuple(sorted(spec.quantiles)) or DEFAULT_QUANTILES
        if not spec.quantiles:
            inferred["quantiles"] = "the default levels"
    labeled = None
    if frame is not None:
        labeled = frame.num_rows - frame[spec.target].null_count
    selection, reason = _selection(spec, labeled, limits.context_rows)
    if reason is not None:
        inferred["selection"] = reason
    if selection.recent is not None and spec.time is not None and selection.neighbors:
        inferred.setdefault(
            "selection.recent",
            f"`time` is {spec.time}, so the nearest rows lean to the newest "
            f"{selection.recent:,}",
        )

    from generalforecast.predictors.runtime.plan import OUTPUTS

    resolution = Resolution(
        name=spec.name,
        model=spec.model,
        model_digest=digest,
        task=task,
        target=spec.target,
        target_type=target_type,
        classes=classes,
        id=spec.id,
        id_type=kinds.get(spec.id) if spec.id else None,
        time=spec.time,
        time_type=kinds.get(spec.time) if spec.time else None,
        series=spec.series,
        series_types=tuple(kinds.get(column) for column in spec.series),
        inputs=inputs,
        input_roles=roles,
        inferred=MappingProxyType(inferred),
        context=ContextInfo(
            spec.context.kind, context_rows, query_rows, context_digest
        ),
        selection=selection,
        frequency=frequency,
        horizon=spec.horizon,
        related=spec.related,
        quantiles=quantiles,
        outputs=OUTPUTS[task],
        files=MappingProxyType(dict(pin.files)),
        resources=MappingProxyType(_resources(spec)),
        manifest=lock.manifest_digest(spec),
        spec=spec,
        context_table=frame,
        query_table=queries,
    )
    _admitted(spec, resolution, frame)
    changes = lock.changes(previous, lock.document(resolution))
    if locked and (changes or previous is None):
        raise InvalidInput(
            message="gf.lock is out of date"
            if previous is not None
            else "gf.lock does not exist",
            param="lock",
            hint="run `gf check` without '--locked' to write it",
        )
    return dataclasses.replace(resolution, lock_changes=changes)


def _resources(spec: PredictorSpec) -> dict[str, Any]:
    """The manifest's hosted resources, as the lock records them."""
    resources = spec.resources
    if resources is None:
        return {}
    values = {
        "accelerator": resources.accelerator,
        "replicas_min": resources.replicas_min,
        "replicas_max": resources.replicas_max,
        "idle_seconds": resources.idle_seconds,
    }
    return {key: value for key, value in values.items() if value is not None}


def _admitted(spec: PredictorSpec, resolution: Resolution, frame: Any) -> None:
    """Compiles the resolution as the runtime does, so check refuses what serving would.

    The same `capabilities.admit` and schema checks run here, at
    `compile_lock` and when `gf serve` or hosted serving binds a revision.
    """
    from generalforecast.predictors import lock
    from generalforecast.predictors.runtime.compile import compile_plan
    from generalforecast.predictors.runtime.plan import ExecutionPlan
    from generalforecast.predictors.validation import PredictorError

    document = lock.document(resolution)
    try:
        plan = ExecutionPlan.from_lock(document)
        compile_plan(plan, frame, resolver=_registry.resolver())
    except PredictorError as error:
        raise spec.error(
            f"{spec.model} cannot serve this predictor: {error.message}",
            param=_param(error.code),
            hint=_hint(error.code),
        ) from None


def _param(code: str) -> str:
    return {
        "UnsupportedCapability": "model",
        "CapacityExceeded": "inputs",
        "AmbiguousFrequency": "frequency",
        "FrequencyMismatch": "frequency",
        "DuplicateTime": "time",
        "ModelResolutionRequired": "model",
        "UnsupportedTask": "model",
    }.get(code, "model")


def _hint(code: str) -> str | None:
    return {
        "ModelResolutionRequired": "pick a model `gf model list` shows for predictors",
        "AmbiguousFrequency": "resample the context to one step, or fill its gaps",
        "DuplicateTime": "keep one row per series and time",
    }.get(code)


def _pin(spec: PredictorSpec, *, online: bool) -> models.ModelPin:
    try:
        return _registry.pin(spec.model, online=online)
    except models.PredictorError as error:
        if error.code == "ModelUnavailable" or online:
            raise spec.error(
                f"{spec.model} could not be confirmed upstream",
                param="model",
                hint=str(error),
            ) from None
        raise spec.error(
            f"{spec.model} cannot be locked: {error.message}",
            param="model",
            hint="pick a model `gf model list` shows for predictors",
        ) from None


def _model_digest(
    spec: PredictorSpec,
    pin: models.ModelPin,
    previous: dict | None,
    *,
    upgrade: bool,
) -> str | None:
    digest = pin.digest or None
    if previous is None or upgrade:
        return digest
    before = previous.get("predictor", {})
    pinned = before.get("model_digest")
    if before.get("model") == spec.model and pinned not in (None, digest):
        raise spec.error(
            f"the catalog's {spec.model} is not the one gf.lock pinned",
            param="model",
            because=f"gf.lock pins {_short(pinned)}; this release ships {_short(digest)}",
            hint="run `gf check --upgrade` to pin the new artifact",
        )
    return digest


def _short(digest: str | None) -> str:
    if digest is None:
        return "no digest"
    return digest.removeprefix("sha256:")[:7]


def _check_columns(spec: PredictorSpec, info: TableInfo) -> None:
    assert spec.target is not None
    names = info.names
    if spec.target not in names:
        raise spec.error(
            f"target column `{spec.target}` is not in the context",
            param="target",
            hint=_did_you_mean(spec.target, names)
            or f"{info.path} has {', '.join(names)}",
            because=f"{info.path} has {', '.join(names)}",
        )
    for column, role in spec.roles.items():
        if column not in names:
            param = role if role in ("id", "time", "series") else f"inputs.{role}"
            raise spec.error(
                f"{role} column `{column}` is not in the context",
                param=param,
                hint=_did_you_mean(column, names)
                or f"{info.path} has {', '.join(names)}",
            )
    for column in spec.exclude:
        if column not in names:
            raise spec.error(
                f"`inputs.exclude` names `{column}`, which is not in the context",
                param="inputs.exclude",
                hint=_did_you_mean(column, names)
                or f"{info.path} has {', '.join(names)}",
            )
    for column, _ in spec.types:
        if column not in names:
            raise spec.error(
                f"`inputs.types` names `{column}`, which is not in the context",
                param="inputs.types",
                hint=_did_you_mean(column, names)
                or f"{info.path} has {', '.join(names)}",
            )
    if spec.inputs is not None:
        for column in spec.inputs:
            if column not in names:
                raise spec.error(
                    f"input `{column}` is not in the context",
                    param="inputs",
                    hint=_did_you_mean(column, names)
                    or f"{info.path} has {', '.join(names)}",
                )


def _did_you_mean(name: str, names: tuple[str, ...]) -> str | None:
    close = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    return f"did you mean `{close[0]}`?" if close else None


def _inputs(
    spec: PredictorSpec,
    info: TableInfo | None,
    limits: _registry.Limits,
    inferred: dict[str, str],
    task: str,
) -> tuple[tuple[tuple[str, str | None], ...], tuple[str, ...]]:
    """The inputs with their kinds, and each one's role.

    A tabular model's inputs have the role `input`. A forecaster's inputs
    are covariates: `future` when the manifest lists them under
    `inputs.future`, and otherwise `past`, the one rule every consumer of
    the lock reads instead of deriving roles again.
    """
    roles = spec.roles
    kinds = info.kinds() if info is not None else {}
    overrides = dict(spec.types)
    if spec.inputs is not None:
        names = list(spec.inputs)
    elif info is None:
        names = []
    else:
        names = [
            name
            for name in info.names
            if name not in roles and name not in spec.exclude
        ]
        skipped = [f"{name} ({role})" for name, role in roles.items()]
        reason = f"every column of {info.path} except " + ", ".join(skipped)
        if spec.exclude:
            reason += ", minus " + ", ".join(spec.exclude)
        inferred["inputs"] = reason
    if not names and task != "forecast":
        raise spec.error(
            f"predictor {spec.name} has no input columns",
            param="inputs",
            because=(
                f"every column of {info.path} has a role or is the target"
                if info is not None and not spec.exclude
                else "every column has a role, is the target or is excluded"
            ),
            hint="add columns to the context, or list `inputs`",
        )
    if len(names) > limits.inputs:
        raise InvalidInput(
            message=f"predictor {spec.name} has {len(names):,} inputs; "
            f"{spec.model} reads at most {limits.inputs:,}",
            param="inputs",
            hint="exclude some columns with `inputs: {exclude: [...]}`",
            limit=limits.inputs,
            actual=len(names),
        )
    forecast_roles = {"past", "future"}
    if spec.past or spec.future:
        names = [name for name in names if roles.get(name) not in forecast_roles]
        names += [name for name in (*spec.past, *spec.future) if name not in names]
    inputs = []
    for name in names:
        kind = overrides.get(name, kinds.get(name))
        if name in overrides and kinds.get(name) not in (None, overrides[name]):
            inferred[f"inputs.{name}"] = (
                f"declared {overrides[name]}; {info.path if info else 'the context'} "
                f"looks like {kinds[name]}"
            )
        inputs.append((name, kind))
    if task == "forecast":
        future = set(spec.future)
        marked = tuple("future" if name in future else "past" for name in names)
    else:
        marked = ("input",) * len(names)
    return tuple(inputs), marked


def _target(
    spec: PredictorSpec,
    task: str,
    frame: pa.Table | None,
    pin: models.ModelPin,
    inferred: dict[str, str],
) -> tuple[str | None, tuple[Any, ...] | None]:
    assert spec.target is not None
    if frame is None:
        return None, None
    column = frame[spec.target]
    info = inspect_frame(frame.select([spec.target]), path=spec.context.snapshot or "")
    kind = info.column(spec.target).kind
    source = spec.context.snapshot or "the context"
    inferred["target_type"] = f"{spec.target} has {kind} values in {source}"
    if task == "classification":
        distinct = info.column(spec.target).distinct
        if kind in ("float", "text") or (kind == "int" and distinct > 50):
            raise spec.error(
                f"target `{spec.target}` has {kind} values; a classifier needs classes"
                if kind != "int"
                else f"target `{spec.target}` has {distinct:,} distinct int values; "
                "a classifier needs classes",
                param="target",
                because=f"{spec.model} classifies",
                hint="pick a regression model with `gf model list --task regression`",
            )
        values = pc.unique(labels(column)).to_pylist()
        classes = tuple(sorted(values, key=lambda item: (str(type(item)), item)))
        inferred["classes"] = f"{len(classes):,} classes in {source}"
        if len(classes) < 2:
            raise spec.error(
                f"target `{spec.target}` has one class",
                param="target",
                hint="the context needs rows of at least two classes",
            )
        if pin.max_classes is not None and len(classes) > pin.max_classes:
            raise InvalidInput(
                message=f"target `{spec.target}` has {len(classes):,} classes; "
                f"{spec.model} serves at most {pin.max_classes:,}",
                param="target",
                hint="pick a model with more class slots, or a variant with codes such as `@4`",
                limit=pin.max_classes,
                actual=len(classes),
            )
        return kind, classes
    if kind not in NUMERIC:
        raise spec.error(
            f"target `{spec.target}` has {kind} values; "
            f"{'a regressor' if task == 'regression' else 'a forecaster'} needs numbers",
            param="target",
            because=f"{spec.model} predicts a number",
            hint="pick a classifier with `gf model list --task classification`",
        )
    return kind, None


def _frequency(
    spec: PredictorSpec, frame: pa.Table, inferred: dict[str, str]
) -> str | int | float:
    """The context's frequency by the one rule, `frequency.infer` or `frequency.check`."""
    from generalforecast.predictors import frequency

    assert spec.time is not None
    times: dict[tuple, list] = {}
    for row in frame.select([*spec.series, spec.time]).to_pylist():
        key = tuple(row[column] for column in spec.series)
        times.setdefault(key, []).append(row[spec.time])
    try:
        if spec.frequency is not None:
            frequency.check(times, spec.frequency)
            return spec.frequency
        found = frequency.infer(times)
    except frequency.FrequencyError as error:
        raise spec.error(
            error.message,
            param="frequency" if error.code != "DuplicateTime" else "time",
            because=error.because,
            hint={
                "AmbiguousFrequency": "resample to one step per series, or fill its gaps",
                "DuplicateTime": "keep one row per series and time",
                "TooFewTimes": "add `frequency: 1d`, or the step your times follow",
                "FrequencyMismatch": "fix `frequency`, or resample the context",
            }.get(error.code),
        ) from None
    inferred["frequency"] = f"the times in {spec.time} step by {found}"
    return found


def _selection(
    spec: PredictorSpec, rows: int | None, cap: int
) -> tuple[SelectionInfo, str | None]:
    """The lock's selection: the manifest's against the context's size and the row cap."""
    return resolve_selection(
        spec.selection,
        rows=rows,
        cap=cap,
        timed=spec.time is not None,
        model=spec.model,
    )


def _query_reason(spec: PredictorSpec, query_rows: int) -> str:
    if spec.context.until is not None:
        return f"{query_rows:,} rows at or after {spec.context.until}"
    if spec.context.holdout is not None:
        return f"{query_rows:,} rows drawn at random, fraction {spec.context.holdout.fraction}"
    if query_rows:
        return f"{query_rows:,} rows with no target"
    return "no rows; every row has a target and no split is declared"


__all__ = ["ContextInfo", "Resolution", "SelectionInfo", "resolve"]
