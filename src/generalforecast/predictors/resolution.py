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
    split,
    to_table,
)

pc: Any = _compute  # ty cannot see Arrow's generated compute functions
CORE_SHARE = 0.25
RECENT_FACTOR = 4
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
class SelectionInfo:
    """The resolved context selection, as integers.

    Attributes:
        core: Rows every query sees; None means the whole context.
        neighbors: Nearest rows per query.
        kind: How the core is drawn, `stratified` or `random`.
        recent: How many newest rows the nearest-row search leans on.
    """

    core: int | None
    neighbors: int
    kind: str
    recent: int | None = None


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

    frame: pa.Table | None = None
    queries: pa.Table | None = None
    info: TableInfo | None = None
    context_rows = query_rows = None
    context_digest = None
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
        _check_columns(spec, info)
        frame, queries = split(table, spec)
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
    elif rows is not None:
        frame = to_table(rows)
        info = inspect_frame(frame, path="the context")
        _check_columns(spec, info)
        context_rows = frame.num_rows
    elif spec.inputs is None:
        raise spec.error(
            f"`inputs` must be listed for a {spec.context.kind} context",
            param="inputs",
            hint=f"a {spec.context.kind} context has no rows to infer from; "
            "write `inputs: [spend, plan]`",
        )

    inputs = _inputs(spec, info, limits, inferred, task)
    target_type, classes = _target(spec, task, frame, pin, inferred)
    frequency = spec.frequency
    quantiles = spec.quantiles
    if task == "forecast":
        if frequency is None and frame is not None:
            frequency = _frequency(spec, frame, inferred)
        if not quantiles:
            quantiles = DEFAULT_QUANTILES
            inferred["quantiles"] = "the default levels"
        elif pin.quantiles and not set(quantiles) <= set(pin.quantiles):
            unsupported = sorted(set(quantiles) - set(pin.quantiles))
            raise spec.error(
                f"{spec.model} does not output quantile {unsupported[0]}",
                param="quantiles",
                hint="pick levels from " + ", ".join(str(q) for q in pin.quantiles),
            )
    selection = _selection(spec, context_rows, limits.context_rows, inferred)

    resolution = Resolution(
        name=spec.name,
        model=spec.model,
        model_digest=digest,
        task=task,
        target=spec.target,
        target_type=target_type,
        classes=classes,
        id=spec.id,
        id_type=info.column(spec.id).kind if spec.id and info is not None else None,
        time=spec.time,
        series=spec.series,
        inputs=inputs,
        inferred=MappingProxyType(inferred),
        context=ContextInfo(
            spec.context.kind, context_rows, query_rows, context_digest
        ),
        selection=selection,
        frequency=frequency,
        horizon=spec.horizon,
        quantiles=quantiles,
        files=MappingProxyType(dict(pin.files)),
        spec=spec,
    )
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


def _pin(spec: PredictorSpec, *, online: bool) -> models.ModelPin:
    try:
        return _registry.pin(spec.model, online=online)
    except models.PredictorError as error:
        raise spec.error(
            f"{spec.model} could not be confirmed upstream",
            param="model",
            hint=str(error),
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
    if before.get("model") == spec.model and pinned is not None:
        return pinned
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
) -> tuple[tuple[str, str | None], ...]:
    roles = spec.roles
    kinds = info.kinds() if info is not None else {}
    overrides = dict(spec.types)
    if spec.inputs is not None:
        names = list(spec.inputs)
    else:
        assert info is not None
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
    return tuple(inputs)


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
) -> str | float | None:
    assert spec.time is not None
    if pa.types.is_integer(frame[spec.time].type):
        periods = sorted(pc.unique(pc.drop_null(frame[spec.time])).to_pylist())
        if len(periods) < 2:
            raise spec.error(
                f"cannot infer `frequency` from one distinct period in {spec.context.snapshot}",
                param="frequency",
                hint="add `frequency: 1`",
            )
        import statistics
        from itertools import pairwise

        step = int(statistics.median([b - a for a, b in pairwise(periods)]))
        inferred["frequency"] = (
            f"the median gap between periods in {spec.time} is {step}"
        )
        return step
    try:
        stamps = pc.cast(frame[spec.time], pa.timestamp("us"))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        raise spec.error(
            f"time column `{spec.time}` has no timestamps",
            param="time",
            hint="store times as ISO dates or timestamps",
        ) from None
    unique = pc.unique(pc.drop_null(stamps))
    if len(unique) < 2:
        raise spec.error(
            f"cannot infer `frequency` from one distinct time in {spec.context.snapshot}",
            param="frequency",
            hint="add `frequency: 1d`",
        )
    ordered = pc.sort_indices(unique)
    values = unique.take(ordered).cast(pa.int64()).to_numpy()
    gaps = values[1:] - values[:-1]
    import numpy as np

    seconds = float(np.median(gaps)) / 1_000_000
    frequency = _format_seconds(seconds)
    inferred["frequency"] = (
        f"the median gap between times in {spec.time} is {frequency}"
    )
    return frequency


def _format_seconds(seconds: float) -> str | float:
    for unit, size in (("w", 604800), ("d", 86400), ("h", 3600), ("min", 60), ("s", 1)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return seconds


def _selection(
    spec: PredictorSpec,
    rows: int | None,
    cap: int,
    inferred: dict[str, str],
) -> SelectionInfo:
    declared = spec.selection
    kind = declared.kind if declared and declared.kind else "stratified"
    undeclared = declared is None or (
        declared.core is None and declared.neighbors is None
    )
    if undeclared and (rows is None or rows <= cap):
        core: int | None = None
        neighbors = 0
        if rows is not None:
            inferred["selection"] = (
                f"{rows:,} rows fit under the {cap:,} row cap of {spec.model}, "
                "so every query sees the whole context"
            )
    else:
        core = _count(declared.core if declared else None, cap)
        if (
            core is None
            and declared is not None
            and declared.core is None
            and declared.neighbors is not None
        ):
            core = None
        elif core is None:
            core = round(CORE_SHARE * cap)
        neighbors_given = declared.neighbors if declared else None
        neighbors = _count(neighbors_given, cap)
        if neighbors is None:
            neighbors = max(cap - (core or 0), 0)
        if rows is not None and (
            declared is None or (declared.core is None and declared.neighbors is None)
        ):
            inferred["selection"] = (
                f"{rows:,} rows exceed the {cap:,} row cap of {spec.model}, so each query "
                f"sees a {kind} core of {core:,} rows and its {neighbors:,} nearest rows"
            )
    recent = declared.recent if declared and declared.recent is not None else None
    if recent is None and spec.time is not None and neighbors:
        recent = neighbors * RECENT_FACTOR
        inferred["selection.recent"] = (
            f"`time` is {spec.time}, so the nearest rows lean to the newest {recent:,}"
        )
    return SelectionInfo(core, neighbors, kind, recent)


def _count(value: float | None, cap: int) -> int | None:
    if value is None:
        return None
    if type(value) is float:
        return round(value * cap)
    return int(value)


def _query_reason(spec: PredictorSpec, query_rows: int) -> str:
    if spec.context.until is not None:
        return f"{query_rows:,} rows at or after {spec.context.until}"
    if spec.context.holdout is not None:
        return f"{query_rows:,} rows drawn at random, fraction {spec.context.holdout.fraction}"
    if query_rows:
        return f"{query_rows:,} rows with no target"
    return "no rows; every row has a target and no split is declared"


__all__ = ["ContextInfo", "Resolution", "SelectionInfo", "resolve"]
