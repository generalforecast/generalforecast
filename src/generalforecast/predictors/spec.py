# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The predictor manifest, `gf.yaml`: parsing, validation and writing.

The manifest holds intent only. Inferred types, resolved selection integers and
digests belong to `gf.lock` (see `generalforecast.predictors.lock`).
"""

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import yaml

from generalforecast.errors import InvalidInput
from generalforecast.predictors import _registry, files, frequency
from generalforecast.predictors.validation import NAME, PredictorError

MANIFEST = "gf.yaml"
FREQUENCY = frequency.SPAN
KINDS = ("int", "float", "0/1", "category", "text", "timestamp", "bool")
ACCELERATORS = ("L4", "A10G", "L40S", "cpu")
SELECTION_KINDS = ("stratified", "random")
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
ContextKind = Literal["snapshot", "request", "live"]

# Each retired key, with where it lived and the hint that names its replacement.
RETIRED: dict[tuple[str, str], str] = {
    ("", "features"): "use `inputs`: a list of columns, or `{exclude: [...]}`",
    ("", "task"): "remove `task`; the model determines it",
    ("", "data"): "use `context: {snapshot: FILE}`",
    ("", "strategy"): "use `selection: {core, neighbors, kind}`",
    ("", "step"): "use `frequency`, such as `1d` or `1h`",
    ("", "classes"): "remove `classes`; `gf check` reads them from the context",
    ("", "target_type"): "remove `target_type`; `gf check` infers it from the context",
    ("", "past_inputs"): "use `inputs: {past: [...]}`",
    ("", "future_inputs"): "use `inputs: {future: [...]}`",
    ("", "source"): "use `context: {snapshot: FILE}`",
    (
        "",
        "mode",
    ): "use `context: {snapshot: FILE}`, `{request: {max_rows: N}}` or `{live: NAME}`",
    ("", "cpu"): "use `resources: {accelerator: L4}`",
    ("", "memory"): "use `resources: {accelerator: L4}`",
    ("", "min_replicas"): "use `resources: {replicas: {min: 0, max: 2}}`",
    ("", "max_replicas"): "use `resources: {replicas: {min: 0, max: 2}}`",
    ("", "compute"): "use `resources: {accelerator: L4}`",
    ("", "scaling"): "use `resources: {replicas: {min: 0, max: 2}}`",
    ("", "outputs"): "remove `outputs`; the model determines them",
    ("context", "source"): "use `context: {snapshot: FILE}`",
    (
        "context",
        "mode",
    ): "use `context: {snapshot: FILE}`, `{request: {max_rows: N}}` or `{live: NAME}`",
    ("context", "selection"): "use the top-level `selection`",
    ("context", "order_by"): "use `time`; recent rows are selected by the time role",
    ("context", "max_rows"): "use `context: {request: {max_rows: N}}`",
    ("inputs", "include"): "use `inputs: [column, ...]`; a list replaces inference",
    ("inputs", "features"): "use `inputs: [column, ...]`",
    ("inputs", "missing_values"): "remove `missing_values`; empty cells are allowed",
    ("selection", "strategy"): "use `selection: {core, neighbors, kind}`",
    ("resources", "cpu"): "use `resources: {accelerator: L4}`",
    ("resources", "memory"): "use `resources: {accelerator: L4}`",
    ("resources", "min_replicas"): "use `resources: {replicas: {min: 0, max: 2}}`",
    ("resources", "max_replicas"): "use `resources: {replicas: {min: 0, max: 2}}`",
}

ALLOWED: dict[str, tuple[str, ...]] = {
    "": (
        "name",
        "model",
        "target",
        "id",
        "time",
        "series",
        "related",
        "frequency",
        "horizon",
        "quantiles",
        "inputs",
        "context",
        "selection",
        "resources",
    ),
    "context": ("snapshot", "request", "live", "until", "holdout"),
    "context.request": ("max_rows",),
    "context.holdout": ("fraction", "seed"),
    "inputs": ("exclude", "types", "past", "future"),
    "selection": ("core", "neighbors", "kind", "recent", "seed"),
    "resources": ("accelerator", "replicas", "idle_seconds"),
    "resources.replicas": ("min", "max"),
}


@dataclass(frozen=True)
class Mark:
    """A 1-based position in `gf.yaml`."""

    line: int
    column: int


Marks = Mapping[str, Mark]
NO_MARKS: Marks = MappingProxyType({})


@dataclass(frozen=True)
class Holdout:
    """A seeded random draw of snapshot rows used as queries."""

    fraction: float
    seed: int = 0


@dataclass(frozen=True)
class Context:
    """Where a predictor's context rows come from.

    Attributes:
        kind: `snapshot` (a file in the predictor directory), `request` (sent
            with each call) or `live` (a hosted live context).
        snapshot: The snapshot path, relative to the predictor directory.
        live: The live context name.
        max_rows: The most rows a request may carry.
        until: Snapshot rows at or after this time are queries.
        holdout: A random draw of snapshot rows that are queries.
    """

    kind: ContextKind = "request"
    snapshot: str | None = None
    live: str | None = None
    max_rows: int | None = None
    until: str | None = None
    holdout: Holdout | None = None


@dataclass(frozen=True)
class Selection:
    """Declared context selection; unset fields are chosen at `gf check`.

    Attributes:
        core: Rows every query sees, as a fraction of the cap or a count.
        neighbors: Nearest rows per query, as a fraction of the cap or a count.
        kind: How the core is drawn, `stratified` or `random`.
        recent: How many newest rows the nearest-row search leans on.
        seed: The draw's seed.
    """

    core: float | int | None = None
    neighbors: float | int | None = None
    kind: str | None = None
    recent: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class Resources:
    """Hosted serving resources."""

    accelerator: str | None = None
    replicas_min: int | None = None
    replicas_max: int | None = None
    idle_seconds: int | None = None


@dataclass(frozen=True)
class PredictorSpec:
    """A parsed `gf.yaml`.

    Attributes:
        name: The predictor name, `[a-z][a-z0-9-]{0,62}`.
        model: The canonical catalog id, `general-forecast/<name>`.
        target: The column to predict.
        id: The id role column, if any.
        time: The time role column, if any.
        series: The series role columns.
        related: Which series a forecaster forecasts jointly: None (each
            series on its own, the default), `()` for every series as one
            group (`related: all`), or the series columns whose values name
            each group (`related: [store]`). A group holds at most
            `ServingPolicy.group_series` series, which share one origin and
            their known-future values.
        frequency: A forecaster's cadence, such as `1d`, or seconds as a number.
        horizon: How many steps a forecaster predicts.
        quantiles: A forecaster's or regressor's output levels.
        inputs: The input columns, when listed; a list replaces inference.
        exclude: Columns removed from the inferred inputs.
        types: Declared column types, overriding inference.
        past: A forecaster's past-only input columns.
        future: A forecaster's known-future input columns.
        context: Where context rows come from and how queries are split off.
        selection: Declared context selection.
        resources: Hosted serving resources.
        marks: Where each key sits in the manifest text, for error pointers.
    """

    name: str
    model: str
    target: str | None = None
    id: str | None = None
    time: str | None = None
    series: tuple[str, ...] = ()
    related: tuple[str, ...] | None = None
    frequency: str | int | float | None = None
    horizon: int | None = None
    quantiles: tuple[float, ...] = ()
    inputs: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    types: tuple[tuple[str, str], ...] = ()
    past: tuple[str, ...] = ()
    future: tuple[str, ...] = ()
    context: Context = Context()
    selection: Selection | None = None
    resources: Resources | None = None
    marks: Marks = field(default=NO_MARKS, compare=False, repr=False, hash=False)

    @property
    def roles(self) -> dict[str, str]:
        """Each role column by name: target, id, time, series, past, future."""
        roles: dict[str, str] = {}
        for column, role in (
            (self.target, "target"),
            (self.id, "id"),
            (self.time, "time"),
            *((column, "series") for column in self.series),
            *((column, "past") for column in self.past),
            *((column, "future") for column in self.future),
        ):
            if column is not None:
                roles.setdefault(column, role)
        return roles

    def mark(self, path: str) -> Mark | None:
        """Where a dotted key path's value sits in the manifest, if known."""
        return self.marks.get(path)

    def error(
        self,
        message: str,
        *,
        param: str,
        hint: str | None = None,
        because: str | None = None,
        at: str | None = None,
    ) -> InvalidInput:
        """An `InvalidInput` pointing at a manifest key, when its position is known."""
        mark = self.mark(at or param)
        return InvalidInput(
            message=message,
            param=param,
            hint=hint,
            because=because,
            line=mark.line if mark else None,
            column=mark.column if mark else None,
        )

    @classmethod
    def from_dict(
        cls, value: Any, *, default_name: str | None = None, marks: Marks = NO_MARKS
    ) -> "PredictorSpec":
        """Parses manifest values, such as a loaded `gf.yaml` or an API body."""
        return _parse(value, default_name, marks)

    def to_dict(self) -> dict[str, Any]:
        """The manifest as `gf.yaml` holds it, with unset fields left out."""
        value: dict[str, Any] = {"name": self.name, "model": self.model}
        if self.target is not None:
            value["target"] = self.target
        for key in ("id", "time"):
            if getattr(self, key) is not None:
                value[key] = getattr(self, key)
        if self.series:
            value["series"] = list(self.series)
        if self.related is not None:
            value["related"] = list(self.related) if self.related else "all"
        if self.frequency is not None:
            value["frequency"] = self.frequency
        if self.horizon is not None:
            value["horizon"] = self.horizon
        if self.quantiles:
            value["quantiles"] = list(self.quantiles)
        if self.inputs is not None:
            value["inputs"] = list(self.inputs)
        elif self.exclude or self.types or self.past or self.future:
            inputs: dict[str, Any] = {}
            if self.exclude:
                inputs["exclude"] = list(self.exclude)
            if self.types:
                inputs["types"] = dict(self.types)
            if self.past:
                inputs["past"] = list(self.past)
            if self.future:
                inputs["future"] = list(self.future)
            value["inputs"] = inputs
        context: dict[str, Any] = {}
        if self.context.kind == "snapshot":
            context["snapshot"] = self.context.snapshot
        elif self.context.kind == "live":
            context["live"] = self.context.live
        else:
            context["request"] = (
                {"max_rows": self.context.max_rows}
                if self.context.max_rows is not None
                else {}
            )
        if self.context.until is not None:
            context["until"] = self.context.until
        if self.context.holdout is not None:
            context["holdout"] = {
                "fraction": self.context.holdout.fraction,
                "seed": self.context.holdout.seed,
            }
        value["context"] = context
        if self.selection is not None:
            selection = {
                key: getattr(self.selection, key)
                for key in ("core", "neighbors", "kind", "recent")
                if getattr(self.selection, key) is not None
            }
            if self.selection.seed:
                selection["seed"] = self.selection.seed
            if selection:
                value["selection"] = selection
        if self.resources is not None:
            resources: dict[str, Any] = {}
            if self.resources.accelerator is not None:
                resources["accelerator"] = self.resources.accelerator
            if self.resources.replicas_max is not None:
                resources["replicas"] = {
                    "min": self.resources.replicas_min,
                    "max": self.resources.replicas_max,
                }
            if self.resources.idle_seconds is not None:
                resources["idle_seconds"] = self.resources.idle_seconds
            if resources:
                value["resources"] = resources
        return value

    def to_yaml(self) -> str:
        """The manifest text `write_manifest` writes."""
        return yaml.safe_dump(
            self.to_dict(), sort_keys=False, default_flow_style=None, width=88
        )


class _Fields:
    """A mapping being consumed key by key, with its manifest positions."""

    def __init__(self, value: dict[str, Any], prefix: str, marks: Marks):
        self.value = dict(value)
        self.prefix = prefix
        self.marks = marks

    def path(self, key: str) -> str:
        return f"{self.prefix}.{key}" if self.prefix else key

    def fail(
        self,
        key: str,
        message: str,
        *,
        hint: str | None = None,
        at_key: bool = False,
        path: str | None = None,
    ) -> InvalidInput:
        path = path or self.path(key)
        mark = self.marks.get(f"{path}:key" if at_key else path)
        return InvalidInput(
            message=message,
            param=path,
            hint=hint,
            line=mark.line if mark else None,
            column=mark.column if mark else None,
        )

    def reject_retired(self) -> None:
        section = self.prefix.split(".")[0] if self.prefix else ""
        for key in self.value:
            if (hint := RETIRED.get((section, str(key)))) is not None:
                raise self.fail(
                    str(key),
                    f"`{key}` is no longer a manifest key",
                    hint=hint,
                    at_key=True,
                )

    def reject_unknown(self) -> None:
        allowed = ALLOWED[self.prefix]
        for key in self.value:
            if key not in allowed:
                where = f" in `{self.prefix}`" if self.prefix else ""
                raise self.fail(
                    str(key),
                    f"unknown key `{key}`{where}",
                    hint=f"{self.prefix or 'gf.yaml'} takes {', '.join(allowed)}",
                    at_key=True,
                )

    def has(self, key: str) -> bool:
        return key in self.value

    def get(self, key: str) -> Any:
        return self.value.get(key)

    def text(self, key: str, what: str = "a column name") -> str | None:
        value = self.value.get(key)
        if value is None:
            return None
        if type(value) is not str or not value or len(value) > 512:
            raise self.fail(key, f"`{self.path(key)}` must be {what}")
        if any(ord(c) < 32 for c in value):
            raise self.fail(key, f"`{self.path(key)}` has control characters")
        return value

    def texts(self, key: str, what: str = "a list of column names") -> tuple[str, ...]:
        value = self.value.get(key)
        if value is None:
            return ()
        if type(value) is not list or not all(
            type(item) is str and item for item in value
        ):
            raise self.fail(key, f"`{self.path(key)}` must be {what}")
        if len(set(value)) != len(value):
            duplicate = next(item for item in value if value.count(item) > 1)
            raise self.fail(
                key,
                f"`{self.path(key)}` lists `{duplicate}` twice",
                hint="name each column once",
            )
        return tuple(value)

    def integer(self, key: str, low: int, high: int = 10**9) -> int | None:
        value = self.value.get(key)
        if value is None:
            return None
        if type(value) is not int or not low <= value <= high:
            raise self.fail(
                key, f"`{self.path(key)}` must be an integer from {low} to {high:,}"
            )
        return value

    def number(self, key: str, low: float, high: float, *, open_low: bool = False):
        value = self.value.get(key)
        if value is None:
            return None
        if (
            type(value) not in (int, float)
            or not (low < value if open_low else low <= value)
            or value > high
        ):
            bracket = "(" if open_low else "["
            raise self.fail(
                key, f"`{self.path(key)}` must be a number in {bracket}{low}, {high}]"
            )
        return value

    def mapping(self, key: str) -> "_Fields | None":
        value = self.value.get(key)
        if value is None:
            return None
        if type(value) is not dict:
            raise self.fail(key, f"`{self.path(key)}` must be a mapping")
        return _Fields(value, self.path(key), self.marks)


def _parse(value: Any, default_name: str | None, marks: Marks) -> PredictorSpec:
    if type(value) is not dict:
        raise InvalidInput(
            message="gf.yaml must be a mapping of keys to values",
            param="",
            hint="start with `model:` and `target:`; `gf init` writes one",
        )
    top = _Fields(value, "", marks)
    top.reject_retired()
    for section in ("context", "inputs", "selection", "resources"):
        if type(top.get(section)) is dict:
            _Fields(top.get(section), section, marks).reject_retired()
    top.reject_unknown()

    if not top.has("model"):
        raise top.fail(
            "model",
            "gf.yaml has no `model`",
            hint=f"add `model: {_registry.ORGANIZATION}/tabicl-v2-classifier`, or run `gf model list`",
        )
    model = top.text("model", "a catalog id")
    try:
        model = _registry.canonical(model)
    except InvalidInput as error:
        raise top.fail("model", error.message, hint=error.hint) from None

    name = top.text("name", "a predictor name")
    if name is None:
        name = default_name
        if name is None or not NAME.fullmatch(name):
            raise top.fail(
                "name",
                "gf.yaml has no `name`"
                if name is None
                else f"directory name `{name}` is not a predictor name",
                hint="add `name:` with lowercase letters, digits and hyphens",
            )
    elif not NAME.fullmatch(name):
        raise top.fail(
            "name",
            f"name `{name}` is not a predictor name",
            hint="use lowercase letters, digits and hyphens, starting with a letter",
        )

    if not top.has("target"):
        raise top.fail(
            "target",
            "gf.yaml has no `target`",
            hint="add `target: <column>`; `gf init` picks one from the context",
        )
    target = top.text("target")
    id_ = top.text("id")
    time = top.text("time")
    series = top.texts("series")
    related = _related(top)
    frequency = _frequency(top)
    horizon = top.integer("horizon", 1, 10_000)
    quantiles = _quantiles(top)

    inputs: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    types: tuple[tuple[str, str], ...] = ()
    past: tuple[str, ...] = ()
    future: tuple[str, ...] = ()
    if top.has("inputs"):
        raw = top.get("inputs")
        if type(raw) is list:
            inputs = top.texts("inputs")
        elif type(raw) is dict:
            section = _Fields(raw, "inputs", marks)
            section.reject_unknown()
            exclude = section.texts("exclude")
            types = _types(section)
            past = section.texts("past")
            future = section.texts("future")
        else:
            raise top.fail(
                "inputs",
                "`inputs` must be a list of columns or a mapping",
                hint="write `inputs: [spend, plan]` or `inputs: {exclude: [note]}`",
            )

    context = _context(top)
    selection = _selection(top)
    resources = _resources(top)

    spec = PredictorSpec(
        name=name,
        model=model,
        target=target,
        id=id_,
        time=time,
        series=series,
        related=related,
        frequency=frequency,
        horizon=horizon,
        quantiles=quantiles,
        inputs=inputs,
        exclude=exclude,
        types=types,
        past=past,
        future=future,
        context=context,
        selection=selection,
        resources=resources,
        marks=marks,
    )
    validate(spec)
    return spec


def _related(top: _Fields) -> tuple[str, ...] | None:
    value = top.get("related")
    if value is None:
        return None
    if value == "all":
        return ()
    if type(value) is list:
        return top.texts("related", "`all` or a list of series columns")
    raise top.fail(
        "related",
        "`related` must be `all` or a list of series columns",
        hint="write `related: all`, or `related: [store]` to forecast each store's series together",
    )


def _frequency(top: _Fields) -> str | int | float | None:
    value = top.get("frequency")
    if value is None:
        return None
    if type(value) in (int, float, str):
        try:
            return frequency.normalize(value)
        except ValueError:
            pass
    raise top.fail(
        "frequency",
        "`frequency` must be a span such as `1d`, `1h` or `7d`, a pandas offset "
        "alias such as `W-MON` or `MS`, or seconds as a number",
    )


def _quantiles(top: _Fields) -> tuple[float, ...]:
    value = top.get("quantiles")
    if value is None:
        return ()
    if (
        type(value) is not list
        or not value
        or not all(type(item) in (int, float) and 0 < item < 1 for item in value)
    ):
        raise top.fail(
            "quantiles",
            "`quantiles` must be a list of levels between 0 and 1",
            hint="the default is [0.1, 0.5, 0.9]",
        )
    if len(set(value)) != len(value):
        raise top.fail("quantiles", "`quantiles` lists a level twice")
    return tuple(float(item) for item in value)


def _types(section: _Fields) -> tuple[tuple[str, str], ...]:
    value = section.get("types")
    if value is None:
        return ()
    if type(value) is not dict or not all(type(key) is str and key for key in value):
        raise section.fail("types", "`inputs.types` must map column names to types")
    for column, kind in value.items():
        if kind not in KINDS:
            raise section.fail(
                "types",
                f"`inputs.types.{column}` must be one of {', '.join(KINDS)}",
                path=f"inputs.types.{column}",
            )
    return tuple(value.items())


def _context(top: _Fields) -> Context:
    raw = top.get("context")
    if raw is None:
        return Context()
    if type(raw) is str:
        raise top.fail(
            "context",
            "`context` must say which kind it is",
            hint=f"write `context: {{snapshot: {raw}}}`",
        )
    if type(raw) is not dict:
        raise top.fail(
            "context",
            "`context` must be a mapping",
            hint="write `context: {snapshot: context.csv}`",
        )
    section = _Fields(raw, "context", top.marks)
    section.reject_unknown()
    kinds = [key for key in ("snapshot", "request", "live") if section.has(key)]
    if len(kinds) != 1:
        raise top.fail(
            "context",
            "`context` must have exactly one of `snapshot`, `request` or `live`"
            if kinds
            else "`context` must have one of `snapshot`, `request` or `live`",
            hint="write `context: {snapshot: context.csv}`",
        )
    kind = kinds[0]
    snapshot = live = None
    max_rows = None
    if kind == "snapshot":
        snapshot = section.text(
            "snapshot", "a file path inside the predictor directory"
        )
        try:
            snapshot = files.relative(snapshot)
        except PredictorError:
            raise section.fail(
                "snapshot",
                f"snapshot `{snapshot}` is outside the predictor directory",
                hint="copy the file into the predictor directory and name it relative to gf.yaml",
            ) from None
    elif kind == "live":
        live = section.text("live", "a live context name")
        if not NAME.fullmatch(live or ""):
            raise section.fail(
                "live",
                f"live context `{live}` is not a context name",
                hint="use lowercase letters, digits and hyphens, starting with a letter",
            )
    else:
        request = section.get("request")
        if request is None or request == {}:
            request = {}
        if type(request) is not dict:
            raise section.fail(
                "request",
                "`context.request` must be a mapping",
                hint="write `request: {max_rows: 10000}` or `request: {}`",
            )
        inner = _Fields(request, "context.request", top.marks)
        inner.reject_unknown()
        max_rows = inner.integer("max_rows", 1, 10**9)
    until = _until(section)
    holdout = _holdout(section)
    if until is not None and holdout is not None:
        raise section.fail(
            "holdout",
            "`until` and `holdout` cannot both be set",
            hint="pick one way to split queries from the snapshot",
        )
    if kind != "snapshot":
        for key in ("until", "holdout"):
            if section.has(key):
                raise section.fail(
                    key,
                    f"`{key}` needs a snapshot context",
                    hint=f"a {kind} context has no rows to split; remove `{key}`",
                )
    return Context(
        kind=cast(ContextKind, kind),
        snapshot=snapshot,
        live=live,
        max_rows=max_rows,
        until=until,
        holdout=holdout,
    )


def _until(section: _Fields) -> str | None:
    value = section.get("until")
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if type(value) is str:
        try:
            parsed = dt.datetime.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is not None:
            return value
    raise section.fail(
        "until",
        "`context.until` must be a date or timestamp",
        hint="write `until: 2025-09-01` or `until: 2025-09-01T00:00:00`",
    )


def until_timestamp(value: str | None) -> dt.datetime | None:
    """A declared `until` as a naive or aware datetime, or None."""
    if value is None:
        return None
    return dt.datetime.fromisoformat(value)


def _holdout(section: _Fields) -> Holdout | None:
    value = section.get("holdout")
    if value is None:
        return None
    if type(value) in (int, float):
        value = {"fraction": value}
    if type(value) is not dict:
        raise section.fail(
            "holdout",
            "`context.holdout` must be a mapping",
            hint="write `holdout: {fraction: 0.2, seed: 0}`",
        )
    inner = _Fields(value, "context.holdout", section.marks)
    inner.reject_unknown()
    fraction = inner.get("fraction")
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise inner.fail(
            "fraction",
            "`context.holdout.fraction` must be a number between 0 and 1",
            hint="write `holdout: {fraction: 0.2}`",
        )
    seed = inner.integer("seed", 0, 2**32 - 1)
    return Holdout(float(fraction), seed if seed is not None else 0)


def _selection(top: _Fields) -> Selection | None:
    section = top.mapping("selection")
    if section is None:
        return None
    section.reject_unknown()
    core = _share(section, "core")
    neighbors = _share(section, "neighbors")
    kind = section.text("kind", "a selection kind")
    if kind is not None and kind not in SELECTION_KINDS:
        raise section.fail(
            "kind",
            f"`selection.kind` must be one of {', '.join(SELECTION_KINDS)}",
        )
    recent = section.integer("recent", 0, 10**9)
    seed = section.integer("seed", 0, 2**32 - 1)
    return Selection(core, neighbors, kind, recent, seed if seed is not None else 0)


def _share(section: _Fields, key: str) -> float | int | None:
    value = section.get(key)
    if value is None:
        return None
    if type(value) is int and value >= 0:
        return value
    if type(value) is float and 0 <= value <= 1:
        return value
    raise section.fail(
        key,
        f"`selection.{key}` must be a fraction of the model's row cap or a row count",
        hint=f"write `{key}: 0.25` or `{key}: 1000`",
    )


def _resources(top: _Fields) -> Resources | None:
    section = top.mapping("resources")
    if section is None:
        return None
    section.reject_unknown()
    accelerator = section.text("accelerator", "an accelerator")
    if accelerator is not None and accelerator not in ACCELERATORS:
        raise section.fail(
            "accelerator",
            f"`resources.accelerator` must be one of {', '.join(ACCELERATORS)}",
        )
    low = high = None
    if (replicas := section.mapping("replicas")) is not None:
        replicas.reject_unknown()
        low = replicas.integer("min", 0, 1000)
        high = replicas.integer("max", 1, 1000)
        if high is None:
            raise replicas.fail("max", "`resources.replicas` needs `max`")
        if low is None:
            low = 0
        if low > high:
            raise replicas.fail("min", "`resources.replicas.min` must not exceed `max`")
    idle = section.integer("idle_seconds", 0, 86400)
    return Resources(accelerator, low, high, idle)


def validate(spec: PredictorSpec) -> None:
    """Checks the rules that hold between fields, whatever built the spec.

    Raises:
        InvalidInput: On the first broken rule, pointing at its manifest line
            when the spec was loaded from text.
    """
    if not NAME.fullmatch(spec.name):
        raise spec.error(
            f"name `{spec.name}` is not a predictor name",
            param="name",
            hint="use lowercase letters, digits and hyphens, starting with a letter",
        )
    if spec.target is None:
        raise spec.error(
            "gf.yaml has no `target`",
            param="target",
            hint="add `target: <column>`; `gf init` picks one from the context",
        )
    if spec.inputs is not None and spec.exclude:
        raise spec.error(
            "`inputs` lists columns and also excludes some",
            param="inputs",
            hint="list the inputs you want, or exclude from the inferred set, not both",
        )
    roles = spec.roles
    for column, role in (
        (spec.id, "id"),
        (spec.time, "time"),
        *((column, "series") for column in spec.series),
        *((column, "past") for column in spec.past),
        *((column, "future") for column in spec.future),
    ):
        if column is not None and roles[column] != role:
            raise spec.error(
                f"column `{column}` is both `{roles[column]}` and `{role}`",
                param=role,
                hint="give each column one role",
            )
    for column in spec.exclude:
        if column in roles:
            raise spec.error(
                f"cannot exclude `{column}`; it is the {roles[column]}",
                param="inputs.exclude",
                hint="remove it from `inputs.exclude`",
            )
    if spec.inputs is not None:
        for column in spec.inputs:
            if column in roles:
                raise spec.error(
                    f"`{column}` is the {roles[column]}, not an input",
                    param="inputs",
                    hint="remove it from `inputs`; role columns are never inputs",
                )
    for column, _ in spec.types:
        if column == spec.target:
            raise spec.error(
                f"`{column}` is the target; its type is read from the context",
                param="inputs.types",
                hint="remove it from `inputs.types`",
            )
    if spec.context.until is not None and spec.time is None:
        raise spec.error(
            "`until` needs a `time` column to compare against",
            param="context.until",
            hint="add `time: <column>`, or use `holdout` for a random split",
        )
    task = _registry.task_of(spec.model)
    if task == "forecast":
        if spec.time is None:
            raise spec.error(
                f"predictor {spec.name} has no cadence",
                param="time",
                because=f"{spec.model} forecasts series, and gf.yaml\n"
                "declares no `time` column",
                hint="add `time: date` and `frequency: 1d`, or pick a tabular model\n"
                "with `gf model list --task classification`",
            )
        if spec.related:
            outside = [column for column in spec.related if column not in spec.series]
            if outside:
                raise spec.error(
                    f"`related` names `{outside[0]}`, which is not a series column",
                    param="related",
                    hint="name series columns, or write `related: all`",
                )
        if spec.horizon is None:
            raise spec.error(
                f"predictor {spec.name} has no `horizon`",
                param="horizon",
                hint="add `horizon: 7` for seven steps ahead",
                at="model",
            )
    else:
        if spec.quantiles and task == "classification":
            raise spec.error(
                "`quantiles` is for regression and forecast models",
                param="quantiles",
                because=f"{spec.model} is a classification model",
                hint="remove `quantiles`; a classifier outputs class probabilities",
            )
        for key in ("horizon", "frequency", "past", "future"):
            if getattr(spec, key) not in (None, ()):
                raise spec.error(
                    f"`{key}` is for forecast models",
                    param=key if key not in ("past", "future") else f"inputs.{key}",
                    because=f"{spec.model} is a {task} model",
                    hint=f"remove `{key}`, or pick a forecast model with "
                    "`gf model list --task forecast`",
                )
        if spec.related is not None:
            raise spec.error(
                "`related` is for forecast models",
                param="related",
                because=f"{spec.model} is a {task} model",
                hint="remove `related`",
            )
        if spec.series:
            raise spec.error(
                "`series` is for forecast models",
                param="series",
                because=f"{spec.model} is a {task} model",
                hint="remove `series`, or pick a forecast model with "
                "`gf model list --task forecast`",
            )


def _marks(data: bytes) -> Marks:
    """Where each key path's key and value start, 1-based, in the YAML text."""
    marks: dict[str, Mark] = {}

    def walk(node: yaml.Node, path: str) -> None:
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if not isinstance(key, yaml.ScalarNode):
                    continue
                child = f"{path}.{key.value}" if path else str(key.value)
                marks[f"{child}:key"] = Mark(
                    key.start_mark.line + 1, key.start_mark.column + 1
                )
                marks[child] = Mark(
                    value.start_mark.line + 1, value.start_mark.column + 1
                )
                walk(value, child)
        elif isinstance(node, yaml.SequenceNode):
            for index, item in enumerate(node.value):
                child = f"{path}[{index}]"
                marks[child] = Mark(
                    item.start_mark.line + 1, item.start_mark.column + 1
                )
                walk(item, child)

    try:
        root = yaml.compose(data, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return NO_MARKS
    if root is not None:
        walk(root, "")
    return MappingProxyType(marks)


def parse_text(data: bytes | str, *, default_name: str | None = None) -> PredictorSpec:
    """Parses manifest text, keeping line and column marks for errors."""
    if isinstance(data, str):
        data = data.encode()
    try:
        value = files.parse_yaml(data)
    except PredictorError as error:
        raise InvalidInput(
            message=f"gf.yaml is not valid YAML: {error}".rstrip("."),
            param="",
            hint="check the indentation and quoting",
        ) from None
    return _parse(value, default_name, _marks(data))


def root_of(path: str | Path) -> Path:
    """The predictor directory a path names, whether the directory or its gf.yaml."""
    path = Path(path)
    if path.name == MANIFEST and not path.is_dir():
        return path.parent
    return path


def load(path: str | Path) -> PredictorSpec:
    """Loads and validates the `gf.yaml` in a predictor directory.

    Args:
        path: The directory, or its `gf.yaml`.

    Returns:
        The parsed spec, named after the directory when `gf.yaml` has no `name`.

    Raises:
        InvalidInput: If the manifest is missing, unreadable or invalid.
    """
    root = root_of(path)
    try:
        if not root.is_dir():
            raise PredictorError("MissingFile", "no manifest")
        data = files.read(root, MANIFEST)
    except PredictorError as error:
        if error.code == "MissingFile":
            raise InvalidInput(
                message=f"no gf.yaml in {_display(root)}",
                param="path",
                hint="run `gf init` to write one",
            ) from None
        raise InvalidInput(
            message=f"cannot read gf.yaml in {_display(root)}",
            param="path",
            hint=str(error),
        ) from None
    return parse_text(data, default_name=root.resolve().name)


def write_manifest(
    spec: PredictorSpec, root: str | Path, *, force: bool = False
) -> Path:
    """Writes `gf.yaml` for a spec, creating the directory only once it is valid.

    Args:
        spec: The manifest to write.
        root: The predictor directory; created if missing.
        force: Overwrite an existing `gf.yaml`.

    Returns:
        The written path.

    Raises:
        InvalidInput: If the spec is invalid, the snapshot is not a readable
            file inside `root`, or `gf.yaml` exists and `force` is false.
    """
    root = Path(root)
    validate(spec)
    if spec.context.kind == "snapshot":
        assert spec.context.snapshot is not None
        snapshot = root / spec.context.snapshot
        if not snapshot.is_file():
            raise InvalidInput(
                message=f"snapshot `{spec.context.snapshot}` is not a file in {_display(root)}",
                param="context.snapshot",
                hint="copy the file into the predictor directory and name it relative to gf.yaml",
            )
    target = root / MANIFEST
    if target.exists() and not force:
        raise InvalidInput(
            message=f"gf.yaml already exists in {_display(root)}",
            param="path",
            hint="pass --force to overwrite it, or run `gf check` to validate it",
        )
    text = spec.to_yaml()
    root.mkdir(parents=True, exist_ok=True)
    try:
        files.atomic_write(root, MANIFEST, text.encode())
    except PredictorError as error:
        raise InvalidInput(
            message=f"cannot write gf.yaml in {_display(root)}",
            param="path",
            hint=str(error),
        ) from None
    return target


def _display(root: Path) -> str:
    text = str(root)
    if root.is_absolute():
        try:
            text = str(root.relative_to(Path.cwd()))
        except ValueError:
            return text
    return text if text.startswith((".", "/")) else f"./{text}"


FIELDS = tuple(item.name for item in fields(PredictorSpec) if item.name != "marks")
