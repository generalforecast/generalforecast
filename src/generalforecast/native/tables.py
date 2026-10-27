# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The public engine tables: `gf.engine.Table` and `gf.engine.SeriesTable`.

A table holds context rows in this process's engine, or on a server it is
connected to, and predicts queries against them. Its settings come in four
groups: the `Schema` (which column is the target and which have roles),
the `Retrieved` selection (how each query's context is drawn), `TimeOptions`
(how time orders rows) and `TableOptions` (operational knobs).
"""

from __future__ import annotations

import datetime
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Self

import numpy as np
import pyarrow as pa

from generalforecast import tabular
from generalforecast._core import DEFAULT_FIT_ROWS, GroupStrategy, Strategy
from generalforecast.errors import Conflict, InvalidInput, NotFound, boundary
from generalforecast.native import models as registry
from generalforecast.native.client import (
    Described,
    PastCap,
    Rebuild,
    RebuildWait,
    RefitCutoffs,
    RetrievalCategoricals,
    Roles,
    SeriesDefinition,
    Summary,
)
from generalforecast.native.client import SeriesTable as _NativeSeriesTable
from generalforecast.native.client import Table as _NativeTable
from generalforecast.predictors import frequency

Kind = Literal["random", "stratified", "balanced", "diverse"]
Exists = Literal["error", "open"]

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d|w)?\s*$")
_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def seconds(value: Any, what: str = "duration") -> float | None:
    """Seconds of a duration such as `7d`, `1h`, `90s`, a number or a timedelta."""
    if value is None:
        return None
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, bool):
        raise InvalidInput(f"{what} must be a duration such as `7d`, not a boolean")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        match = _DURATION.match(value)
        if match is not None:
            return float(match.group(1)) * _UNITS[match.group(2) or "s"]
        try:
            return frequency.seconds(value)
        except ValueError:
            pass
    raise InvalidInput(
        f"{what} `{value}` is not a duration",
        hint="use a number of seconds or a string such as `1d`, `1h` or `7d`",
    )


@dataclass(frozen=True)
class Schema:
    """Which columns a table reads, and how.

    Args:
        target: The column to predict.
        id: A column holding each row's id; the frame's index if None.
        time: A column holding each row's time, if rows are ordered by time.
        inputs: The input columns; a list replaces inference from the frame.
        exclude: Columns to leave out of the inferred inputs.
        types: Column kinds that override inference, by name: `numeric`,
            `categorical`, `boolean`, `datetime` or `retrieval`.
    """

    target: str
    id: str | None = None
    time: str | None = None
    inputs: Sequence[str] | None = None
    exclude: Sequence[str] = ()
    types: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Checks the roles are distinct names."""
        if not isinstance(self.target, str) or not self.target:
            raise InvalidInput("schema needs a target column name")
        roles = [self.target, self.id, self.time]
        named = [role for role in roles if role is not None]
        if len(set(named)) != len(named):
            raise InvalidInput("target, id and time must be different columns")
        if self.inputs is not None and self.exclude:
            raise InvalidInput(
                "give either `inputs` or `exclude`, not both",
                hint="a list of inputs already replaces inference",
            )
        if self.inputs is not None:
            object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "exclude", tuple(self.exclude))
        object.__setattr__(self, "types", dict(self.types))

    @property
    def roles(self) -> frozenset[str]:
        """The columns with a role or the target, which are never inputs."""
        return frozenset(
            name for name in (self.target, self.id, self.time) if name is not None
        )


@dataclass(frozen=True)
class Retrieved:
    """How each query's context is drawn from the table.

    A context is a core of rows every query shares plus the query's nearest
    rows. Integer counts are rows; floats are fractions of the model's row
    cap; `core=None` is the whole table with no neighbors.

    Args:
        core: Rows of the shared core, a fraction of the model's row cap, or
            None for the whole table.
        neighbors: The nearest rows added per query, or a fraction of the cap.
        kind: How the core is drawn: `random`, `stratified` (by class or
            target bin), `balanced` or `diverse`.
        recent: Pick the neighbors as the newest of this many nearest rows.
        last: Take the whole context from the table's newest rows, this many.
        span: Take the whole context from rows at most this long before the
            newest row's time.
        seed: The draw's seed.
        categoricals: How categorical columns place rows for retrieval:
            `codes`, `one_hot` or `target`.
        regions: Live rows per region whose queries share a context, if any.
        scale_rows: Rows the retrieval scaling stays fixed after, if set.
    """

    core: float | None = 0.25
    neighbors: float = 0.75
    kind: Kind = "stratified"
    recent: float | None = None
    last: int | None = None
    span: Any = None
    seed: int = 0
    categoricals: RetrievalCategoricals = "codes"
    regions: int | None = None
    scale_rows: int | None = None

    def __post_init__(self) -> None:
        """Checks the counts."""
        for name in ("core", "neighbors", "recent"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise InvalidInput(f"selection {name} must be a count or a fraction")
            if value < 0 or (isinstance(value, float) and value > 1):
                raise InvalidInput(
                    f"selection {name} must be a row count or a fraction in [0, 1]"
                )
        if self.core is None and self.neighbors:
            raise InvalidInput(
                "a selection with `core=None` takes the whole table and no neighbors",
                hint="set neighbors=0, or give the core a size",
            )
        if self.kind not in ("random", "stratified", "balanced", "diverse"):
            raise InvalidInput(
                f"selection kind `{self.kind}` is not one of random, stratified, "
                "balanced, diverse"
            )

    @property
    def whole(self) -> bool:
        """Whether every query reads the whole table."""
        return self.core is None

    def resolved(self, cap: int) -> Retrieved:
        """The same selection with fractions turned into row counts for `cap`."""

        def rows(value: float | None) -> int | None:
            if value is None:
                return None
            if isinstance(value, float):
                return round(value * cap)
            return int(value)

        return replace(
            self,
            core=rows(self.core),
            neighbors=rows(self.neighbors) or 0,
            recent=rows(self.recent),
        )

    def strategy(self, cap: int) -> Strategy:
        """The engine's strategy for this selection, at a model's row cap."""
        with boundary():
            fixed = self.resolved(cap)
            assert fixed.neighbors is not None
            if fixed.core is None:
                return Strategy.exact()
            core, neighbors = int(fixed.core), int(fixed.neighbors)
            if neighbors == 0 and fixed.last is None and fixed.span is None:
                if core == 0:
                    return Strategy.exact()
                return Strategy.subsample(core, self.seed, self.kind)
            return Strategy.retrieved(
                neighbors,
                core,
                self.seed,
                self.kind,
                among=None if fixed.recent is None else int(fixed.recent),
                last=self.last,
                span=self.span,
            )

    def describe(self) -> str:
        """One line as `gf check` prints it."""
        if self.core is None:
            return "whole table"
        core = f"{self.core:,}" if isinstance(self.core, int) else f"{self.core:g}"
        parts = [f"core {core} {self.kind}"]
        if self.neighbors:
            neighbors = (
                f"{self.neighbors:,}"
                if isinstance(self.neighbors, int)
                else f"{self.neighbors:g}"
            )
            parts.append(f"{neighbors} nearest per query")
        return "retrieved  " + ", ".join(parts)


def default_selection(rows: int, cap: int, *, timed: bool = False) -> Retrieved:
    """The selection `gf check` and `Table` choose for a table of `rows` rows.

    Up to the model's row cap, every query reads the whole table. Past it,
    a stratified core of a quarter of the cap plus the nearest rows that fill
    the rest; with a time role the neighbors lean to the newest rows. The
    engine applies the same rule to a table's live rows at each prediction,
    so a `Table` without a selection follows its size as it grows or shrinks.
    """
    return selection_of(Strategy.adaptive().resolve(rows, cap, timed))


def selection_of(strategy: Strategy) -> Retrieved:
    """The selection an engine strategy spells, with its counts in rows.

    An adaptive strategy has no counts until a table's size resolves it;
    `default_selection` gives the resolution for a size.

    Raises:
        InvalidInput: For an adaptive strategy.
    """
    spec = json.loads(strategy.spec())
    match spec.get("kind"):
        case "exact":
            return Retrieved(core=None, neighbors=0)
        case "retrieved":
            core = spec.get("core") or {}
            recency = spec.get("recency") or {}
            return Retrieved(
                core=int(core["rows"]) if core else 0,
                neighbors=int(spec.get("neighbors", 0)),
                kind=core.get("kind", "stratified"),
                recent=spec.get("among"),
                last=recency.get("last"),
                span=recency.get("span"),
                seed=int(core.get("seed", 0)),
            )
    raise InvalidInput(f"strategy {strategy!r} resolves to no selection")


@dataclass(frozen=True)
class TimeOptions:
    """How time orders a table's rows.

    Args:
        column: The time column; the schema's `time` role if None.
        bucket: Rows within one bucket of time share a page: `1d`, `1h`, a
            number of seconds, or None for none.
        refit_after: Refit the model's state once this much time of rows
            arrived after the last fit; `7d`, seconds, or None for never.
        refit_shift: Offset of refits within `refit_after`, if any.
        cutoffs: When a query's cutoff calls for a new state: `each`,
            `changed` or `never`.
        past: What a context longer than the model was trained on does:
            `clamp` cuts it to the model's row cap, `warn` reads it whole and
            warns.
    """

    column: str | None = None
    bucket: Any = None
    refit_after: Any = None
    refit_shift: Any = None
    cutoffs: RefitCutoffs = "each"
    past: PastCap = "clamp"

    def __post_init__(self) -> None:
        """Checks the durations parse."""
        for name in ("bucket", "refit_after", "refit_shift"):
            seconds(getattr(self, name), name)
        if self.cutoffs not in ("each", "changed", "never"):
            raise InvalidInput("cutoffs must be `each`, `changed` or `never`")
        if self.past not in ("warn", "clamp"):
            raise InvalidInput("past must be `warn` or `clamp`")


@dataclass(frozen=True)
class TableOptions:
    """Operational settings of a table.

    Args:
        page_size: Rows per page of the table's cache.
        partitions: Tables to split the table into through a router.
        rebuild: When a region's context is built again: `always`, `never`
            or a number of seconds.
        rebuild_wait: How long a context due a rebuild is read stale.
        fit_rows: Most rows the model's state is fit on.
        summary: What the state is fit on: `table` or `context`.
        route_by: A column that sends rows and queries to one partition.
    """

    page_size: int = 128
    partitions: int = 1
    rebuild: Rebuild = "always"
    rebuild_wait: RebuildWait | None = None
    fit_rows: int = DEFAULT_FIT_ROWS
    summary: Summary = "table"
    route_by: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Checks the counts."""
        if not isinstance(self.page_size, int) or self.page_size <= 0:
            raise InvalidInput("page_size must be a positive integer")
        if not isinstance(self.partitions, int) or self.partitions <= 0:
            raise InvalidInput("partitions must be a positive integer")
        if self.summary not in ("table", "context"):
            raise InvalidInput("summary must be `table` or `context`")


@dataclass(frozen=True)
class Grouping:
    """How a series table forecasts related series together.

    Exactly one of `alone`, `random` or `given`: each series on its own, random
    groups of `random` series, or the groups the caller gives, of `given`.
    """

    alone: bool = False
    random: int | None = None
    given: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Checks exactly one way is chosen."""
        chosen = [self.alone, self.random is not None, self.given is not None]
        if sum(chosen) != 1:
            raise InvalidInput(
                "a grouping is one of alone=True, random=size or given=size"
            )

    def strategy(self) -> GroupStrategy:
        """The engine's group strategy."""
        with boundary():
            if self.random is not None:
                return GroupStrategy.random(self.random, self.seed)
            if self.given is not None:
                return GroupStrategy.given(self.given)
            return GroupStrategy.alone()


def _client(client: Any, download: bool) -> Any:
    if client is not None:
        return client
    from generalforecast.native.estimators import _local

    return _local(download)


def _column(table: pa.Table, name: str, what: str) -> pa.ChunkedArray:
    if name not in table.column_names:
        raise InvalidInput(
            f"{what} column `{name}` is not in the rows",
            param=name,
            hint=f"the rows have {', '.join(table.column_names) or 'no columns'}",
        )
    return table.column(name)


def _pending_target(rows: Any, target: str) -> Any:
    """`rows` with a target column of only missing values made floats: all pending.

    A frame whose every target is None has a column of no type; as NaN floats
    it says what it means, that none of the labels are known yet.
    """
    if isinstance(rows, pa.Table):
        if target in rows.column_names and pa.types.is_null(
            rows.schema.field(target).type
        ):
            index = rows.column_names.index(target)
            return rows.set_column(index, target, pa.nulls(rows.num_rows, pa.float64()))
        return rows
    if isinstance(rows, dict):
        values = rows.get(target)
        if (
            isinstance(values, list | tuple)
            and values
            and all(v is None for v in values)
        ):
            return {**rows, target: [math.nan] * len(values)}
        return rows
    column = getattr(rows, "columns", None)
    if column is not None and target in column and hasattr(rows, "assign"):
        values = rows[target]
        if (
            len(values)
            and getattr(values.dtype, "kind", "") == "O"
            and all(v is None for v in values.tolist())
        ):
            return rows.assign(**{target: np.full(len(values), math.nan)})
    return rows


def _as_objects(column: pa.ChunkedArray) -> np.ndarray:
    values = column.to_pandas() if hasattr(column, "to_pandas") else None
    if values is not None:
        return np.asarray(values.to_numpy(dtype=object, na_value=None), dtype=object)
    return np.asarray(column.to_pylist(), dtype=object)


def _exists(exists: Any) -> Exists:
    if exists not in ("error", "open"):
        raise InvalidInput('exists must be "error" or "open"')
    return exists


def _opened(client: Any, name: str, *, series: bool) -> Any:
    """The engine's handle on an existing table, or the public error for none."""
    opener, other, this = (
        ("SeriesTable", "Table", "rows")
        if series
        else ("Table", "SeriesTable", "series")
    )
    try:
        with boundary():
            return client.series_table(name) if series else client.table(name)
    except NotFound as exc:
        raise NotFound(
            f"table {name!r} does not exist",
            hint="check the name, or make the table and insert rows first",
        ) from exc
    except InvalidInput as exc:
        if not isinstance(exc.__cause__, TypeError):
            raise
        raise InvalidInput(
            f"table {name!r} holds {this}, not what gf.engine.{opener} opens",
            hint=f"open it with gf.engine.{other}.open({name!r})",
        ) from exc


def _conflict(name: str, what: str) -> Conflict:
    return Conflict(
        f"table {name!r} exists",
        hint=f'pass exists="open" or {what}.open({name!r})',
    )


def _mismatch(name: str, field: str, theirs: Any, ours: Any) -> InvalidInput:
    return InvalidInput(
        f"table {name!r} exists with {field} {theirs!r}, not {ours!r}",
        param=field,
        hint="open it as it is with open(name), or pick another name",
    )


def _route_of(route_by: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A route as the engine describes it, for comparing two."""
    if route_by is None:
        return None
    route = {"column": route_by["column"], "kind": route_by.get("kind", "hashed")}
    bounds = [float(bound) for bound in route_by.get("bounds", ())]
    return {**route, "bounds": bounds} if bounds else route


def _retrieval_of(described: Described) -> Retrieved:
    """The retrieval settings a table was made with, as a whole-table selection.

    `categoricals`, `regions` and `scale_rows` are set when a table is made
    and stay, whatever its strategy; a table made anew keeps them.
    """
    return Retrieved(
        core=None,
        neighbors=0,
        categoricals=described.retrieval_categoricals,
        regions=described.region_rows,
        scale_rows=described.scale_rows,
    )


def _retrieved_of(described: Described) -> Retrieved | None:
    """The selection a table was made with, read back from its strategy.

    None for an adaptive strategy: the table follows its size, and `selection`
    reads the current resolution from the engine.
    """
    spec = json.loads(described.strategy.spec())
    if spec["kind"] == "adaptive":
        return None
    core = spec.get("core") or {}
    recency = spec.get("recency") or {}
    whole = spec["kind"] != "retrieved"
    return replace(
        _retrieval_of(described),
        core=None if whole else int(core.get("rows", 0)),
        neighbors=0 if whole else int(spec.get("neighbors", 0)),
        kind="stratified" if whole else core.get("kind", "random"),
        recent=None if whole else spec.get("among"),
        last=None if whole else recency.get("last"),
        span=None if whole else recency.get("span"),
        seed=int(core.get("seed", 0)),
    )


def _roles_of(described: Described) -> Roles:
    if described.roles is None:
        raise InvalidInput(
            f"table {described.name!r} was not made by gf.engine.Table",
            hint="it names no target, id or time columns to open with",
        )
    return described.roles


class Table:
    """Context rows in the engine, predicted against as they are.

    Args:
        name: The table's name, unique in its engine.
        model: A catalog id such as `general-forecast/tabicl-v2-classifier`.
        target: The target column; shorthand for `Schema(target=...)`.
        schema: Which columns the table reads.
        selection: How each query's context is drawn. None follows the
            table's size: every query reads the whole table up to the model's
            row cap, and past it a stratified core plus nearest rows, resolved
            anew at each prediction as rows come and go (`default_selection`).
        time: How time orders rows.
        ops: Operational settings.
        client: The engine to hold the table; this process's if None.
        download: Whether the model's files may download if not cached.
        exists: What the first insert does when the engine already holds a
            table of this name: `error` raises `Conflict`; `open` attaches to
            it, if its model, target and schema agree with these, and raises
            `InvalidInput` naming the first that does not.
    """

    def __init__(
        self,
        name: str,
        *,
        model: str,
        target: str | None = None,
        schema: Schema | None = None,
        selection: Retrieved | None = None,
        time: TimeOptions | None = None,
        ops: TableOptions | None = None,
        client: Any = None,
        download: bool = True,
        exists: Exists = "error",
    ) -> None:
        """Checks the settings; the engine's table is made at the first insert."""
        if schema is None:
            if target is None:
                raise InvalidInput("a table needs a target: pass target= or schema=")
            schema = Schema(target=target)
        elif target is not None and target != schema.target:
            raise InvalidInput("target and schema.target disagree")
        if not isinstance(name, str) or not name:
            raise InvalidInput("a table needs a name")
        self.name = name
        self.model = registry.canonical(model)
        self.schema = schema
        self._selection = selection
        self._retrieval = Retrieved(core=None, neighbors=0)
        self.time = time or TimeOptions()
        self.ops = ops or TableOptions()
        if self.time.column is not None and self.schema.time is None:
            self.schema = replace(self.schema, time=self.time.column)
        self.task = registry.task_of(self.model)
        if self.task == "forecast":
            raise InvalidInput(
                f"{self.model} forecasts series",
                hint="use gf.engine.SeriesTable for a forecaster",
            )
        self._download = download
        self._client = client
        self._exists = _exists(exists)
        self._table: _NativeTable | None = None
        self._inputs: tuple[str, ...] | None = (
            tuple(schema.inputs) if schema.inputs is not None else None
        )
        self._classes: list[Any] = []
        self._rows = 0

    @classmethod
    def open(cls, name: str, *, client: Any = None, download: bool = True) -> Self:
        """The table of this name the engine already holds.

        The schema, classes, selection and settings come from the engine, as
        the table was made, so the table takes inserts, labels, deletions and
        predictions at once.

        Args:
            name: The table's name.
            client: The engine holding the table; this process's if None.
            download: Whether the model's files may download if not cached.

        Returns:
            The table.

        Raises:
            NotFound: If the engine holds no table of this name.
            InvalidInput: If the name holds series, which `SeriesTable.open`
                opens, or a table made without `gf.engine.Table`.
        """
        engine = _client(client, download)
        handle = _opened(engine, name, series=False)
        with boundary():
            described = handle.describe()
        roles = _roles_of(described)
        table = cls(
            name,
            model=described.model or "",
            schema=Schema(
                target=roles.target,
                id=roles.id,
                time=roles.time,
                inputs=tuple(column.name for column in described.columns),
                types={column.name: column.kind for column in described.columns},
            ),
            selection=_retrieved_of(described),
            time=TimeOptions(
                bucket=described.time_bucket,
                refit_after=described.refit_after,
                refit_shift=described.refit_shift,
                cutoffs=described.refit_cutoffs,
                past=described.past_cap,
            ),
            ops=TableOptions(
                page_size=described.page_size,
                partitions=described.partitions,
                rebuild=described.rebuild,
                rebuild_wait=described.rebuild_wait,
                fit_rows=described.fit_rows,
                summary=described.summary,
                route_by=described.route_by,
            ),
            client=client,
            download=download,
        )
        table._attach(handle, described)
        table._retrieval = _retrieval_of(described)
        return table

    def _attach(self, handle: _NativeTable, described: Described) -> None:
        """Takes the engine's table as this one, with its classes and row count."""
        roles = _roles_of(described)
        self._inputs = tuple(column.name for column in described.columns)
        if described.classes is None:
            self._classes = []
        elif roles.classes:
            self._classes = list(roles.classes)
        else:
            self._classes = list(described.labels or range(described.classes))
        self._rows = described.rows
        self._table = handle

    def _agrees(self, described: Described, kinds: Mapping[str, str]) -> None:
        """Checks an existing table is the one these settings describe."""
        name, schema, roles = self.name, self.schema, _roles_of(described)
        if described.model != self.model:
            raise _mismatch(name, "model", described.model, self.model)
        for role in ("target", "id", "time"):
            if getattr(roles, role) != getattr(schema, role):
                raise _mismatch(name, role, getattr(roles, role), getattr(schema, role))
        theirs = {column.name: column.kind for column in described.columns}
        if set(theirs) != set(kinds):
            raise _mismatch(name, "inputs", sorted(theirs), sorted(kinds))
        for column in sorted(theirs):
            if column in schema.types and theirs[column] != kinds[column]:
                raise _mismatch(
                    name, f"kind of {column}", theirs[column], kinds[column]
                )
        task = "regression" if described.classes is None else "classification"
        if task != self.task:
            raise _mismatch(name, "task", task, self.task)
        if described.partitions != self.ops.partitions:
            raise _mismatch(
                name, "partitions", described.partitions, self.ops.partitions
            )
        if _route_of(described.route_by) != _route_of(self.ops.route_by):
            raise _mismatch(name, "route_by", described.route_by, self.ops.route_by)

    def _open_existing(self, client: Any, kinds: Mapping[str, str]) -> bool:
        """Attaches to the engine's table of this name, if there is one."""
        try:
            handle = _opened(client, self.name, series=False)
        except NotFound:
            return False
        with boundary():
            described = handle.describe()
        self._agrees(described, kinds)
        self._attach(handle, described)
        return True

    @property
    def selection(self) -> Retrieved | None:
        """How each query's context is drawn now.

        The selection given, with its counts in rows once the table exists;
        without one, the selection the table's size resolves to at its next
        prediction, or None before the first rows arrive.
        """
        if self._selection is not None or self._table is None:
            return self._selection
        with boundary():
            return selection_of(self._table.describe().resolved_strategy)

    def set_selection(self, selection: Retrieved | None) -> None:
        """Changes how each query's context is drawn from here on.

        None returns to following the table's size. The engine logs the
        change with the table's writes, so it holds across restarts, and
        draws the ranking, cores and regions for it at the next prediction.
        The selection's `regions`, `scale_rows` and `categoricals` are set
        when the table is created and stay.
        """
        if self._table is None:
            self._selection = selection
            return
        cap = registry.limits(self.model).context_rows
        if selection is None:
            strategy = Strategy.adaptive()
        else:
            selection = selection.resolved(cap)
            strategy = selection.strategy(cap)
        with boundary():
            self._table.set_strategy(strategy)
        self._selection = selection

    @property
    def inputs(self) -> tuple[str, ...] | None:
        """The input columns, once known."""
        return self._inputs

    @property
    def classes(self) -> list[Any]:
        """A classifier's classes, in code order."""
        return list(self._classes)

    def __len__(self) -> int:  # noqa: D105
        return self._rows if self._table is None else len(self._table)

    def __repr__(self) -> str:  # noqa: D105
        inputs = "?" if self._inputs is None else str(len(self._inputs))
        return (
            f"Table({self.name!r}, model={self.model!r}, "
            f"target={self.schema.target!r}, inputs={inputs}, rows={len(self):,})"
        )

    def _frame(self, rows: Any) -> pa.Table:
        return tabular.frame(_pending_target(rows, self.schema.target))

    def _infer_inputs(self, table: pa.Table) -> tuple[str, ...]:
        excluded = set(self.schema.exclude) | self.schema.roles
        return tuple(name for name in table.column_names if name not in excluded)

    def _kinds(self, table: pa.Table) -> dict[str, str]:
        from generalforecast.columns import column_kind, infer

        assert self._inputs is not None
        for name in self._inputs:
            _column(table, name, "input")
        inferred = infer(table.select(list(self._inputs)))
        return {
            name: column_kind(name, self.schema.types.get(name, inferred.get(name)))
            for name in self._inputs
        }

    def _create(self, table: pa.Table, targets: np.ndarray) -> None:
        if self._inputs is None:
            self._inputs = self._infer_inputs(table)
        if not self._inputs:
            raise InvalidInput(
                "no input columns are left after the target and roles",
                hint="give the rows at least one input column",
            )
        kinds = self._kinds(table)
        client = _client(self._client, self._download)
        if self._exists == "open" and self._open_existing(client, kinds):
            return
        limits = registry.limits(self.model)
        if len(self._inputs) > limits.inputs:
            raise InvalidInput(
                f"{len(self._inputs):,} inputs exceed the limit for {self.model}",
                because=f"this model reads {limits.inputs} inputs; the rows have "
                f"{len(self._inputs):,}",
                limit=limits.inputs,
                actual=len(self._inputs),
                hint="drop columns with Schema(exclude=[...]) or list the inputs",
            )
        classes: int | None = None
        if self.task == "classification":
            labels = [value for value in targets.tolist() if value is not None]
            self._classes = sorted(
                set(labels), key=lambda item: (str(type(item)), item)
            )
            classes = max(len(self._classes), 2)
        # Without a selection the engine follows the table's size, with the
        # retrieval settings the table was opened with, or their defaults.
        selection = self._selection or self._retrieval
        if self._selection is None:
            strategy = Strategy.adaptive()
        else:
            selection = selection.resolved(limits.context_rows)
            strategy = selection.strategy(limits.context_rows)
            self._selection = selection
        times = "datetime" if self.schema.time is not None else "integer"
        ops, time = self.ops, self.time
        wait = {"rebuild_wait": ops.rebuild_wait} if ops.rebuild_wait else {}
        roles = Roles(
            target=self.schema.target,
            id=self.schema.id,
            time=self.schema.time,
            classes=tuple(self._classes),
        )
        try:
            with boundary():
                self._table = client._create_table(
                    self.name,
                    model=self.model,
                    columns=kinds,
                    classes=classes,
                    strategy=strategy,
                    page_size=ops.page_size,
                    time_bucket=_bucket(time.bucket, times),
                    refit_after=seconds(time.refit_after, "refit_after"),
                    refit_shift=seconds(time.refit_shift, "refit_shift"),
                    times=times,
                    fit_rows=ops.fit_rows,
                    past_cap=time.past,
                    region_rows=selection.regions,
                    rebuild=ops.rebuild,
                    refit_cutoffs=time.cutoffs,
                    scale_rows=selection.scale_rows,
                    summary=ops.summary,
                    retrieval_categoricals=selection.categoricals,
                    partitions=ops.partitions,
                    route_by=ops.route_by,
                    roles=roles,
                    **wait,
                )
        except Conflict as exc:
            # Another client made the table since the check above.
            if self._exists == "open" and self._open_existing(client, kinds):
                return
            raise _conflict(self.name, "Table") from exc

    def _split(
        self, table: pa.Table, *, need_target: bool
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None]:
        schema = self.schema
        ids = None
        if schema.id is not None:
            ids = _as_objects(_column(table, schema.id, "id"))
        targets = None
        if schema.target in table.column_names:
            targets = _as_objects(table.column(schema.target))
        elif need_target:
            raise InvalidInput(
                f"target column `{schema.target}` is not in the rows",
                param=schema.target,
                hint=f"the rows have {', '.join(table.column_names)}",
            )
        times = None
        if schema.time is not None:
            times = _as_objects(_column(table, schema.time, "time"))
        if self._inputs is None:
            self._inputs = self._infer_inputs(table)
        for name in self._inputs:
            _column(table, name, "input")
        inputs = np.empty((len(table), len(self._inputs)), dtype=object)
        for index, name in enumerate(self._inputs):
            inputs[:, index] = _as_objects(table.column(name))
        return ids, inputs, targets, times

    def insert(self, rows: Any) -> None:
        """Adds context rows.

        Args:
            rows: A frame, list of dicts, Arrow table or file with the target,
                the role columns and the inputs. Ids come from the `id` role,
                else from the row's position after the rows already inserted.
        """
        table = self._frame(rows)
        if len(table) == 0:
            return
        ids, inputs, targets, times = self._split(table, need_target=True)
        assert targets is not None
        if self._table is None:
            self._create(table, targets)
        assert self._table is not None
        if ids is None:
            ids = np.arange(self._rows, self._rows + len(table))
        if self.task == "classification":
            targets = self._coded(targets)
        with boundary():
            self._table.upsert(ids, inputs, targets, times)
        self._rows += len(table)

    def _coded(self, targets: np.ndarray) -> np.ndarray:
        codes = {label: index for index, label in enumerate(self._classes)}
        out = np.empty(len(targets), dtype=object)
        for index, value in enumerate(targets.tolist()):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                out[index] = None
            elif value in codes:
                out[index] = codes[value]
            else:
                raise InvalidInput(
                    f"target value `{value}` is not one of the classes from context",
                    param=self.schema.target,
                    hint="classes are " + ", ".join(map(str, self._classes)),
                )
        return out

    def label(self, ids: Iterable[Any], values: Any) -> None:
        """Sets the target of rows already inserted, by id."""
        self._ready()
        values = np.asarray(list(values), dtype=object)
        if self.task == "classification":
            values = self._coded(values)
        with boundary():
            assert self._table is not None
            self._table.label(list(ids), values)

    def delete(self, ids: Iterable[Any]) -> None:
        """Removes rows by id."""
        self._ready()
        with boundary():
            assert self._table is not None
            self._table.delete(list(ids), missing_ok=True)

    def truncate(self) -> None:
        """Removes every row."""
        if self._table is not None:
            with boundary():
                self._table.truncate()
        self._rows = 0

    def refit(self) -> None:
        """Fits the model's state again on the rows now in the table."""
        self._ready()
        with boundary():
            assert self._table is not None
            self._table.refit()

    def _ready(self) -> None:
        if self._table is None:
            raise InvalidInput(
                f"table {self.name} has no rows yet", hint="insert context rows first"
            )

    def predict(
        self, rows: Any, as_of: Any = None, *, quantiles: Sequence[float] | None = None
    ) -> Any:
        """Predicts query rows.

        Args:
            rows: Query rows with the inputs, and the `id` and `time` roles
                if the schema has them.
            as_of: A time cutoff for every query, or one per query: each
                query's context holds the rows from before it.
            quantiles: For a regressor, the levels to predict instead of the
                model's own.

        Returns:
            A pandas DataFrame with the id (if the schema has one), `label`
            and `p_<class>` columns for a classifier, or `value` and
            `q_<level>` columns for a regressor.
        """
        self._ready()
        table = self._frame(rows)
        ids, inputs, _, times = self._split(table, need_target=False)
        if as_of is None and times is not None:
            as_of = times
        with boundary():
            assert self._table is not None
            outputs = self._table.predict(inputs, as_of, quantiles=quantiles)
        return self._result(ids, outputs, quantiles)

    def _result(
        self, ids: np.ndarray | None, outputs: np.ndarray, quantiles: Any
    ) -> Any:
        import pandas as pd

        columns: dict[str, Any] = {}
        if ids is not None and self.schema.id is not None:
            columns[self.schema.id] = ids
        outputs = np.asarray(outputs)
        if self.task == "classification":
            classes = self._classes or list(range(outputs.shape[1]))
            probabilities = outputs[:, : len(classes)]
            columns["label"] = [classes[index] for index in probabilities.argmax(1)]
            for index, label in enumerate(classes):
                columns[f"p_{label}"] = probabilities[:, index]
        else:
            levels = (
                list(quantiles)
                if quantiles is not None
                else _implied_levels(outputs.shape[1] if outputs.ndim == 2 else 1)
            )
            if outputs.ndim == 1:
                columns["value"] = outputs
            else:
                median = _nearest(levels, 0.5)
                columns["value"] = outputs[:, median]
                for index, level in enumerate(levels):
                    columns[f"q_{level:g}"] = outputs[:, index]
        return pd.DataFrame(columns)

    def drop(self) -> None:
        """Removes the table from its engine."""
        if self._table is not None:
            client = _client(self._client, self._download)
            with boundary():
                client.drop_table(self.name)
            self._table = None
            self._rows = 0

    def close(self) -> None:
        """Alias of `drop`."""
        self.drop()

    def __enter__(self) -> Self:  # noqa: D105
        return self

    def __exit__(self, *_args: object) -> None:  # noqa: D105
        self.drop()


def _implied_levels(count: int) -> list[float]:
    if count <= 1:
        return [0.5]
    return [round((index + 1) / (count + 1), 4) for index in range(count)]


def _nearest(levels: Sequence[float], level: float) -> int:
    return min(range(len(levels)), key=lambda index: abs(levels[index] - level))


def _bucket(width: Any, times: str) -> Any:
    value = seconds(width, "bucket")
    if value is None:
        return None
    if times == "datetime":
        return datetime.timedelta(seconds=value)
    return int(value)


@dataclass(frozen=True)
class SeriesSchema:
    """Which columns a series table reads.

    Args:
        target: The observed value.
        time: The time of each observation.
        series: The column that names each series.
        frequency: The step between observations: `1d`, `1h`, seconds.
        known: Covariates known into the future, such as a price or a holiday.
        past: Covariates known only up to the forecast start.
        types: Column kinds that override inference.
    """

    target: str
    time: str
    series: Sequence[str] = ()
    frequency: Any = None
    known: Sequence[str] = ()
    past: Sequence[str] = ()
    types: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Checks the roles."""
        series = (self.series,) if isinstance(self.series, str) else tuple(self.series)
        if len(series) != 1:
            raise InvalidInput(
                "an engine series table takes exactly one series column",
                hint="combine several keys into one column before inserting",
            )
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "known", tuple(self.known))
        object.__setattr__(self, "past", tuple(self.past))
        object.__setattr__(self, "types", dict(self.types))
        if self.frequency is None:
            raise InvalidInput(
                "a series schema needs a frequency",
                hint="pass frequency='1d', '1h' or a number of seconds",
            )
        seconds(self.frequency, "frequency")


class SeriesTable:
    """Series in the engine, forecast past their latest observations.

    Args:
        name: The table's name.
        model: A catalog forecaster such as `general-forecast/chronos-2`.
        schema: Which columns the table reads.
        horizon: Steps each forecast covers by default.
        grouping: How related series are forecast together.
        quantiles: The levels each forecast reports; the model's if None.
        client: The engine to hold the table; this process's if None.
        download: Whether the model's files may download if not cached.
        exists: What the first insert does when the engine already holds a
            table of this name: `error` raises `Conflict`; `open` attaches to
            it, if its model and schema agree with these, and raises
            `InvalidInput` naming the first that does not.
    """

    def __init__(
        self,
        name: str,
        *,
        model: str,
        schema: SeriesSchema,
        horizon: int | None = None,
        grouping: Grouping | None = None,
        quantiles: Sequence[float] | None = None,
        client: Any = None,
        download: bool = True,
        exists: Exists = "error",
    ) -> None:
        """Checks the settings; the engine's table is made at the first insert."""
        if not isinstance(name, str) or not name:
            raise InvalidInput("a table needs a name")
        self.name = name
        self.model = registry.canonical(model)
        if registry.task_of(self.model) != "forecast":
            raise InvalidInput(
                f"{self.model} does not forecast series",
                hint="use gf.engine.Table for a classifier or regressor",
            )
        self.schema = schema
        self.horizon = horizon
        self.grouping = grouping or Grouping(alone=True)
        self.quantiles = tuple(quantiles) if quantiles is not None else None
        self._levels = registry.quantiles(self.model)
        self._client = client
        self._download = download
        self._exists = _exists(exists)
        self._table: _NativeSeriesTable | None = None
        self._times: str | None = None
        self._attached = False

    @classmethod
    def open(cls, name: str, *, client: Any = None, download: bool = True) -> Self:
        """The series table of this name the engine already holds.

        The schema, horizon and grouping come from the engine, as the table
        was made, so the table takes inserts and forecasts at once. Forecasts
        report the model's own quantile levels. The engine keeps series by
        digest, so an opened table does not know the keys of the series it
        holds: `forecast` needs them named.

        Args:
            name: The table's name.
            client: The engine holding the table; this process's if None.
            download: Whether the model's files may download if not cached.

        Returns:
            The table.

        Raises:
            NotFound: If the engine holds no table of this name.
            InvalidInput: If the name holds rows, which `Table.open` opens.
        """
        engine = _client(client, download)
        handle = _opened(engine, name, series=True)
        definition: SeriesDefinition = handle.definition
        table = cls(
            name,
            model=definition.model or "",
            schema=_series_schema_of(definition),
            horizon=definition.horizon,
            grouping=_grouping_of(definition.group.spec()),
            client=client,
            download=download,
        )
        table._attach(handle)
        return table

    def _agrees(
        self, definition: SeriesDefinition, kinds: Mapping[str, str], times: str
    ) -> None:
        """Checks an existing table is the one these settings describe."""
        name, schema = self.name, self.schema
        if definition.model != self.model:
            raise _mismatch(name, "model", definition.model, self.model)
        if definition.times != times:
            raise _mismatch(name, "time kind", definition.times, times)
        theirs = _series_schema_of(definition)
        for role in ("series", "time", "target"):
            if getattr(theirs, role) != getattr(schema, role):
                raise _mismatch(
                    name, role, getattr(theirs, role), getattr(schema, role)
                )
        for group in ("known", "past"):
            if sorted(getattr(theirs, group)) != sorted(getattr(schema, group)):
                raise _mismatch(
                    name,
                    f"{group} covariates",
                    sorted(getattr(theirs, group)),
                    sorted(getattr(schema, group)),
                )
        for column in definition.columns:
            if column.name in schema.types and column.kind != kinds[column.name]:
                raise _mismatch(
                    name, f"kind of {column.name}", column.kind, kinds[column.name]
                )
        if seconds(theirs.frequency) != seconds(schema.frequency, "frequency"):
            raise _mismatch(name, "frequency", theirs.frequency, schema.frequency)
        if definition.horizon != self.horizon:
            raise _mismatch(name, "horizon", definition.horizon, self.horizon)
        if definition.group != self.grouping.strategy():
            raise _mismatch(
                name, "grouping", _grouping_of(definition.group.spec()), self.grouping
            )

    def _open_existing(self, client: Any, kinds: Mapping[str, str], times: str) -> bool:
        """Attaches to the engine's table of this name, if there is one."""
        try:
            handle = _opened(client, self.name, series=True)
        except NotFound:
            return False
        self._agrees(handle.definition, kinds, times)
        self._attach(handle)
        return True

    def _attach(self, handle: Any) -> None:
        """Takes the engine's table as this one; its series stay unnamed."""
        self._table = handle
        self._times = handle.definition.times
        self._attached = True

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SeriesTable({self.name!r}, model={self.model!r}, "
            f"target={self.schema.target!r}, series={self.schema.series[0]!r}, "
            f"horizon={self.horizon!r})"
        )

    def _create(self, table: pa.Table) -> None:
        from generalforecast.columns import column_kind, infer

        schema = self.schema
        covariates = [*schema.known, *schema.past]
        for name in (schema.target, schema.time, schema.series[0], *covariates):
            _column(table, name, "schema")
        inferred = infer(table.select(covariates)) if covariates else {}
        columns = {
            name: column_kind(name, schema.types.get(name, inferred.get(name)))
            for name in covariates
        }
        times = (
            "datetime"
            if pa.types.is_temporal(table.schema.field(schema.time).type)
            else "integer"
        )
        step = seconds(schema.frequency, "frequency")
        assert step is not None
        client = _client(self._client, self._download)
        if self._exists == "open" and self._open_existing(client, columns, times):
            return
        try:
            with boundary():
                self._table = client._create_series_table(
                    self.name,
                    model=self.model,
                    series=schema.series[0],
                    time=schema.time,
                    target=schema.target,
                    step=datetime.timedelta(seconds=step)
                    if times == "datetime"
                    else step,
                    columns=columns or None,
                    known=schema.known,
                    group=self.grouping.strategy(),
                    horizon=self.horizon,
                    times=times,
                )
        except Conflict as exc:
            # Another client made the table since the check above.
            if self._exists == "open" and self._open_existing(client, columns, times):
                return
            raise _conflict(self.name, "SeriesTable") from exc
        self._times = times

    def insert(self, rows: Any) -> None:
        """Adds observations: series, time, target and covariate columns."""
        table = tabular.frame(rows)
        if len(table) == 0:
            return
        if self._table is None:
            self._create(table)
        assert self._table is not None
        with boundary():
            self._table.upsert(table)

    def forecast(
        self,
        series: Iterable[Any] | None = None,
        horizon: int | None = None,
        *,
        as_of: Any = None,
    ) -> Any:
        """Forecasts series over the steps after their latest observations.

        Args:
            series: The series to forecast, by key; every series if None.
            horizon: Steps to forecast; the table's if None.
            as_of: The time each forecast starts at, one for all or one per
                series.

        Returns:
            A pandas DataFrame with the series column, `step` (1-based) and
            `q_<level>` columns, plus `value` at the median.
        """
        if self._table is None:
            raise InvalidInput(
                f"table {self.name} has no rows yet", hint="insert observations first"
            )
        keys = list(series) if series is not None else self._all_series()
        with boundary():
            outputs = self._table.forecast(
                keys, horizon, as_of, quantiles=self.quantiles
            )
        return self._result(keys, np.asarray(outputs))

    def predict(self, rows: Any, as_of: Any = None) -> Any:
        """Forecasts the future rows: a time grid with known covariates.

        Rows carry the series and time columns, and the known covariates'
        future values, which are inserted ahead of the series before the
        forecast. The forecast covers the steps the rows span.
        """
        table = tabular.frame(rows)
        schema = self.schema
        if self._table is None:
            raise InvalidInput(
                f"table {self.name} has no rows yet", hint="insert observations first"
            )
        if schema.known:
            self.insert(table)
        keys_column = _column(table, schema.series[0], "series")
        keys = list(dict.fromkeys(keys_column.to_pylist()))
        counts = {key: 0 for key in keys}
        for key in keys_column.to_pylist():
            counts[key] += 1
        horizon = max(counts.values())
        with boundary():
            assert self._table is not None
            outputs = self._table.forecast(
                keys, horizon, as_of, quantiles=self.quantiles
            )
        return self._result(keys, np.asarray(outputs))

    def _all_series(self) -> list[Any]:
        assert self._table is not None
        if self._attached:
            raise InvalidInput(
                f"table {self.name} was opened, so name the series to forecast",
                hint="the engine keeps series by digest, and an opened table "
                "learns their keys only from the rows written through it",
            )
        with boundary():
            keys = self._table.known_series()
        if not keys:
            raise InvalidInput(
                f"table {self.name} has no series yet", hint="insert observations first"
            )
        return keys

    def _result(self, keys: Sequence[Any], outputs: np.ndarray) -> Any:
        import pandas as pd

        count, horizon, width = outputs.shape
        levels = list(self.quantiles or self._levels)
        if len(levels) != width:
            levels = _implied_levels(width)
        median = _nearest(levels, 0.5)
        columns: dict[str, Any] = {
            self.schema.series[0]: np.repeat(np.asarray(keys, dtype=object), horizon),
            "step": np.tile(np.arange(1, horizon + 1), count),
            "value": outputs[:, :, median].reshape(-1),
        }
        for index, level in enumerate(levels):
            columns[f"q_{level:g}"] = outputs[:, :, index].reshape(-1)
        return pd.DataFrame(columns)

    def __len__(self) -> int:  # noqa: D105
        return 0 if self._table is None else len(self._table)

    def drop(self) -> None:
        """Removes the table from its engine."""
        if self._table is not None:
            client = _client(self._client, self._download)
            with boundary():
                client.drop_table(self.name)
            self._table = None
            self._attached = False

    def close(self) -> None:
        """Alias of `drop`."""
        self.drop()


def _series_schema_of(definition: SeriesDefinition) -> SeriesSchema:
    """The schema a series table was made with, read back from its definition."""
    step = definition.step
    frequency = step.total_seconds() if isinstance(step, datetime.timedelta) else step
    return SeriesSchema(
        target=definition.target,
        time=definition.time,
        series=definition.series,
        frequency=frequency,
        known=definition.known,
        past=tuple(
            column.name
            for column in definition.columns
            if column.name not in definition.known
        ),
        types={column.name: column.kind for column in definition.columns},
    )


def _grouping_of(spec: str) -> Grouping:
    """The grouping behind a group strategy's spec."""
    group = json.loads(spec)
    match group["kind"]:
        case "random":
            return Grouping(random=int(group["size"]), seed=int(group.get("seed", 0)))
        case "given":
            return Grouping(given=int(group["size"]))
        case _:
            return Grouping(alone=True)


def retrieved_of(
    spec: Mapping[str, Any] | None, cap: int, *, timed: bool, rows: int
) -> Retrieved:
    """A `Retrieved` from a manifest's `selection` mapping, or the defaults."""
    if not spec:
        return default_selection(rows, cap, timed=timed)
    known = {"core", "neighbors", "kind", "recent", "last", "span", "seed"}
    extra = set(spec) - known
    if extra:
        raise InvalidInput(
            f"selection has no field `{min(extra)}`",
            param=f"selection.{min(extra)}",
            hint="selection takes core, neighbors, kind, recent, last, span, seed",
        )
    return Retrieved(**{key: spec[key] for key in spec}).resolved(cap)


__all__ = [
    "Grouping",
    "Retrieved",
    "Schema",
    "SeriesSchema",
    "SeriesTable",
    "Table",
    "TableOptions",
    "TimeOptions",
    "default_selection",
    "retrieved_of",
    "seconds",
]
