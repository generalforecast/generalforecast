# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The handles on an engine's tables: their definitions, and what every client's handle does.

`RemoteTable` and `RemoteSeriesTable` implement them over Flight, for a
server, a router or this process's engine; `gf.engine.Table` and
`gf.engine.SeriesTable` hold one each.
"""

import abc
import contextlib
import datetime
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self, get_args

import numpy as np

from generalforecast._core import DEFAULT_FIT_ROWS, GroupStrategy, Strategy
from generalforecast.columns import (
    Column,
    column_kind,
    moments,
    pick,
)
from generalforecast.keys import Key, keys_of, series_keys

# Names the node gives series in its errors, replaced by the keys given.
_SERIES = re.compile(r"series (\d+)")


# What a table's row times and cutoffs are: integers of the caller's unit, or
# dates and times.
Times = Literal["integer", "datetime"]
# What a table does with a context larger than its model was trained on.
PastCap = Literal["warn", "clamp"]
# When a query's time cutoff calls for a state fit on the rows before it.
RefitCutoffs = Literal["each", "changed", "never"]
# How categorical columns place rows for retrieval.
RetrievalCategoricals = Literal["codes", "one_hot", "target"]
# When a region's context is built again: after any change, never, once
# changed rows reach a share of its rows, or once a period has passed.
Rebuild = Literal["always", "never"] | float | datetime.timedelta

# What a table's column summary is fit on: the table, or each context.
Summary = Literal["table", "context"]

# The writes that carry rows; deletes carry keys alone.
WriteKind = Literal["insert", "update", "upsert"]


@dataclass(frozen=True)
class Answer:
    """What answered a table's predictions.

    Attributes:
        epoch: The epoch of the table's state that answered, which rises with
            each refit, so a stream can mark its refits; None from a
            partitioned table, whose partitions' states differ.
        gpu_seconds: The predictions' share of the worker time of the steps
            that answered them.
    """

    epoch: int | None
    gpu_seconds: float


@dataclass(frozen=True)
class RebuildWait:
    """How long a region's context due a rebuild is read stale while queries wait.

    Its rebuild takes a step ahead of the waiting queries once either limit
    is reached, and the next step once none waits.

    Attributes:
        steps: Steps of queries it waits through at most.
        seconds: How long it waits at most, or None to count steps alone, so
            two runs of the same writes and queries rebuild at the same points
            however long their steps take.
    """

    steps: int = 8
    seconds: float | None = 1.0


# What a table's regions wait, unless it says otherwise.
_REBUILD_WAIT = RebuildWait()


@dataclass(frozen=True)
class Roles:
    """The frame columns a table's rows carry besides the feature columns.

    `gf.engine.Table` reads the target, and the id and time if the frame has
    them, from these columns and keeps them with the table so it opens again.

    Attributes:
        target: The column the target comes from.
        id: The column holding each row's id, if the frame has one.
        time: The column holding each row's time, if the frame has one.
        classes: A classifier's class values in code order, as the target
            column holds them; empty for a regressor's table.
    """

    target: str
    id: str | None = None
    time: str | None = None
    classes: tuple[Any, ...] = ()

    def spec(self) -> dict[str, Any]:
        """The roles as a server's `create_table` takes them."""
        spec: dict[str, Any] = {"target": self.target}
        if self.id is not None:
            spec["id"] = self.id
        if self.time is not None:
            spec["time"] = self.time
        if self.classes:
            spec["classes"] = [_plain(value) for value in self.classes]
        return spec

    @classmethod
    def of(cls, spec: Mapping[str, Any] | None) -> Self | None:
        """The roles a server describes, or None for a table made without them."""
        if not spec:
            return None
        return cls(
            target=str(spec["target"]),
            id=spec.get("id"),
            time=spec.get("time"),
            classes=tuple(spec.get("classes") or ()),
        )


def _plain(value: Any) -> Any:
    """A class value as JSON carries it: a NumPy scalar as its Python value."""
    return value.item() if hasattr(value, "item") else value


@dataclass(frozen=True)
class Definition:
    """What a table is, as it was made.

    Attributes:
        name: The table's name.
        columns: The feature columns, with their kinds.
        classes: Number of classes, or None for a regressor's table.
        outputs: Columns in each prediction: the classes, or the model's quantiles.
        times: What row times and cutoffs are: `integer`, in a unit of the
            caller's, or `datetime`.
        model: The model's name, or None for one given as its calls and parameters.
        strategy: How each query's context is chosen.
        page_size: Rows per page.
        time_bucket: The window of time pages never straddle, if any: in the
            units of row times, or a timedelta for a datetime table.
        refit_after: The share of changed rows that refits the table, if set.
        refit_shift: The shift of a column's mean, in deviations, that refits
            the table, if set.
        fit_rows: The most rows the table's state is fit on.
        refit_cutoffs: When a query's cutoff inside the table's history calls
            for a new state: `each`, `changed` or `never`. A live stream's
            cutoffs, after every row, refit only through `refit_after` or
            `refit_shift`.
        scale_rows: Rows the retrieval scaling stays fixed after, once fit on
            at least that many, if set.
        retrieval_categoricals: How categorical columns place rows for
            retrieval: `codes`, `one_hot` or `target`.
        past_cap: What a context larger than the model was trained on does:
            `warn`, once, or `clamp`, to that length.
        context_rows: The most context rows the model was trained on, if it says.
        region_rows: Live rows per region whose queries share a context, if the
            table has regions.
        rebuild: When a region's context is built again.
        rebuild_wait: How long a context due a rebuild is read stale while
            queries wait.
        summary: What the state, preprocessing and column summary, is fit
            on: the table's rows, `table`, or each context's own, `context`.
        quantile_levels: The level of each of a regressor's prediction
            columns, increasing, or None for a classifier's table.
        roles: The frame columns around the feature columns, for a table
            `gf.engine.Table` made; None for one made without them.
    """

    name: str
    columns: tuple[Column, ...]
    classes: int | None
    outputs: int
    times: Times = "integer"
    model: str | None = None
    strategy: Strategy = field(default_factory=Strategy.exact)
    page_size: int = 128
    time_bucket: Any = None
    refit_after: float | None = None
    refit_shift: float | None = None
    fit_rows: int = DEFAULT_FIT_ROWS
    refit_cutoffs: RefitCutoffs = "each"
    scale_rows: int | None = None
    retrieval_categoricals: RetrievalCategoricals = "codes"
    past_cap: PastCap = "warn"
    context_rows: int | None = None
    region_rows: int | None = None
    rebuild: Rebuild = "always"
    rebuild_wait: RebuildWait = _REBUILD_WAIT
    summary: Summary = "table"
    quantile_levels: tuple[float, ...] | None = None
    roles: Roles | None = None
    partitions: int = 1
    route_by: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Described(Definition):
    """A table as it is now: its definition, rows, vocabularies and labels.

    Attributes:
        rows: Number of live rows.
        resolved_strategy: The strategy the next prediction chooses contexts
            with: an adaptive `strategy` resolved against the live rows, any
            other as given.
        effective_fit_rows: The most rows the table's state is fit on: its
            `fit_rows`, or fewer where the worker's call budget takes fewer at
            the table's width. A table holding more fits on a seeded sample.
            None for a table that fits each context's own state.
        vocabularies: Each categorical column's words, in code order.
        labels: A classifier's labels in code order, if its targets are strings.
        region_stats: For a table with regions, how many it has, their mean
            rows and the levels of k-means that drew them; queries read from a
            built context (`hits`), from one due a rebuild (`stale`), waiting
            for one (`misses`) and with a cutoff (`fresh`), which add up to
            the queries answered; contexts built again (`rebuilds`) and their
            rows, and the seconds they spent due.
    """

    rows: int = 0
    resolved_strategy: Strategy = field(default_factory=Strategy.exact)
    effective_fit_rows: int | None = None
    vocabularies: dict[str, list[str]] = field(default_factory=dict)
    labels: list[str] | None = None
    region_stats: dict[str, float] | None = None

    @classmethod
    def of(cls, reply: Mapping[str, Any]) -> Self:
        """A description from a server's `describe` reply, checked.

        Raises:
            ValueError: If the reply is not a table's description.
        """
        try:
            columns = tuple(
                Column(column["name"], column_kind(column["name"], column["kind"]))
                for column in reply["columns"]
            )
            times = reply.get("times", "integer")
            if times not in get_args(Times):
                raise ValueError(f"unknown time kind {times!r}")
            summary = reply.get("summary", "table")
            if summary not in get_args(Summary):
                raise ValueError(f"unknown summary {summary!r}")
            bucket = reply.get("time_bucket")
            if bucket is not None and times == "datetime":
                bucket = datetime.timedelta(microseconds=int(bucket))
            # Servers store a regressor's table as 0 classes.
            classes, outputs = int(reply["classes"]) or None, int(reply["outputs"])
            levels = reply.get("quantile_levels")
            if levels is not None:
                levels = tuple(float(level) for level in levels)
            elif classes is None:
                levels = _implied(outputs)
            return cls(
                name=str(reply["name"]),
                columns=columns,
                classes=classes,
                outputs=outputs,
                times=times,
                model=str(reply["model"]),
                strategy=_strategy_of(reply.get("strategy") or {"kind": "exact"}),
                page_size=int(reply.get("page_size", 128)),
                time_bucket=bucket,
                refit_after=reply.get("refit_after"),
                refit_shift=reply.get("refit_shift"),
                fit_rows=int(reply.get("fit_rows") or DEFAULT_FIT_ROWS),
                refit_cutoffs=reply.get("refit_cutoffs", "each"),
                scale_rows=reply.get("scale_rows"),
                retrieval_categoricals=reply.get("retrieval_categoricals", "codes"),
                past_cap=reply.get("past_cap", "warn"),
                context_rows=reply.get("context_rows"),
                region_rows=(reply.get("regions") or {}).get("rows"),
                rebuild=_rebuild_of((reply.get("regions") or {}).get("rebuild")),
                rebuild_wait=_wait_of((reply.get("regions") or {}).get("wait")),
                region_stats=reply.get("region_stats"),
                summary=summary,
                quantile_levels=levels,
                roles=Roles.of(reply.get("roles")),
                partitions=int(reply.get("partitions") or 1),
                route_by=reply.get("route_by"),
                rows=int(reply["rows"]),
                resolved_strategy=_strategy_of(
                    reply.get("resolved_strategy")
                    or reply.get("strategy")
                    or {"kind": "exact"}
                ),
                effective_fit_rows=reply.get("effective_fit_rows"),
                vocabularies=dict(reply.get("vocabularies", {})),
                labels=list(reply.get("labels") or []) or None,
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"a malformed table description: {error!r}") from None


def _strategy_of(spec: Mapping[str, Any]) -> Strategy:
    """A strategy from the JSON `Strategy.spec` gives and servers describe.

    Raises:
        ValueError: If the spec names no strategy.
    """
    match spec.get("kind"):
        case "exact":
            return Strategy.exact()
        case "adaptive":
            return Strategy.adaptive()
        case "subsample":
            return Strategy.subsample(int(spec["rows"]), int(spec.get("seed", 0)))
        case "neighbors":
            return Strategy.neighbors(int(spec["rows"]))
        case "retrieved":
            core = spec.get("core") or {}
            recency = spec.get("recency") or {}
            return Strategy.retrieved(
                int(spec.get("neighbors", 0)),
                int(core.get("rows", 0)),
                int(core.get("seed", 0)),
                core.get("kind", "random"),
                among=spec.get("among"),
                last=recency.get("last"),
                span=recency.get("span"),
            )
    raise ValueError(f"unknown strategy {spec!r}")


def _rebuild_of(spec: Any) -> Rebuild:
    """A rebuild policy from the JSON servers store: a name, or a one-key object."""
    match spec:
        case None | "always":
            return "always"
        case "never":
            return "never"
        case {"churn": share}:
            return float(share)
        case {"every": seconds}:
            return datetime.timedelta(seconds=float(seconds))
    raise ValueError(f"unknown rebuild policy {spec!r}")


def _wait_of(spec: Any) -> RebuildWait:
    """A rebuild wait from the JSON servers store, which leaves out the default."""
    if spec is None:
        return _REBUILD_WAIT
    seconds = spec.get("seconds")
    return RebuildWait(int(spec["steps"]), None if seconds is None else float(seconds))


def _rebuild_spec(rebuild: Rebuild) -> Any:
    """A rebuild policy as servers store it."""
    if isinstance(rebuild, datetime.timedelta):
        return {"every": rebuild.total_seconds()}
    if isinstance(rebuild, int | float):
        return {"churn": float(rebuild)}
    return rebuild


def _filled(targets: Any, unknown: float) -> Any:
    """Targets with each null `unknown`: a label not known yet.

    Nulls are Arrow nulls, pandas missing values, and None or NaN among the
    values of a list or an object array, such as a frame's object column.
    """
    if hasattr(targets, "fill_null"):
        return targets.fill_null(unknown)
    if hasattr(targets, "fillna") and hasattr(targets, "isna") and targets.isna().any():
        return targets.astype("float64" if np.isnan(unknown) else "Int64").fillna(
            unknown
        )
    if isinstance(targets, list | tuple) or (
        isinstance(targets, np.ndarray) and targets.dtype.kind == "O"
    ):
        values = np.asarray(targets, dtype=object).ravel()
        missing = np.fromiter(
            (
                value is None or (isinstance(value, float) and np.isnan(value))
                for value in values.tolist()
            ),
            dtype=bool,
            count=len(values),
        )
        if missing.any():
            values = values.copy()
            values[missing] = unknown
            return values.astype("float64")
    return targets


def _vector(values: Any, dtype: type) -> np.ndarray:
    """A column of values as a contiguous array, from any array-like or Arrow array.

    Integer columns must hold whole numbers that fit `dtype`, rather than wrap
    or truncate.
    """
    if isinstance(values, range):
        values = np.arange(values.start, values.stop, values.step)
    elif hasattr(values, "to_numpy") and not isinstance(values, np.ndarray):
        values = values.to_numpy()
    elif not hasattr(values, "__len__") and hasattr(values, "__iter__"):
        # A generator or other one-shot iterable, as the hints allow.
        values = list(values)
    array = np.asarray(values)
    if np.issubdtype(dtype, np.integer) and array.size:
        if array.dtype.kind == "f" and (
            not np.isfinite(array).all() or (array != np.round(array)).any()
        ):
            raise ValueError(f"expected whole numbers, got {array.ravel()[:4]} ...")
        try:
            # Numbers too large for any NumPy integer arrive as objects.
            converted = array.astype(dtype)
            fits = not (converted != array).any()
        except (OverflowError, TypeError, ValueError):
            fits = False
        if not fits:
            raise ValueError(f"expected whole numbers that fit {np.dtype(dtype).name}")
        return np.ascontiguousarray(converted)
    return np.ascontiguousarray(array, dtype=dtype)


class Table(abc.ABC):
    """A table: insert, update and delete rows, and predict queries.

    Attributes:
        definition: What the table is, as it was made.
        name: The table's name.
        columns: The feature columns, with their kinds.
        classes: Number of classes, or None for a regressor's table.
        outputs: Columns in each prediction: the classes, or the model's quantiles.
        times: What row times and cutoffs are: `integer`, in a unit of the
            caller's, or `datetime`.
        model: The model's name, or None for one given as its calls and parameters.
        strategy: How each query's context is chosen.
        last_answer: What answered the table's latest predictions, None before any.
    """

    def __init__(self, definition: Definition) -> None:
        """Wraps a table so defined; of a description, it keeps the definition alone."""
        definition = Definition(
            **{
                name: getattr(definition, name)
                for name in Definition.__dataclass_fields__
            }
        )
        self.definition = definition
        self.name, self.columns = definition.name, definition.columns
        self.classes, self.outputs = definition.classes, definition.outputs
        self.times, self.model = definition.times, definition.model
        self.strategy = definition.strategy
        self.last_answer: Answer | None = None

    def __repr__(self) -> str:  # noqa: D105
        task = "values" if self.classes is None else f"{self.classes} classes"
        model = "" if self.model is None else f", model {self.model!r}"
        return (
            f"<{type(self).__name__} {self.name!r}: {self.features} columns, "
            f"{task}{model}>"
        )

    @property
    def features(self) -> int:
        """Number of feature columns."""
        return len(self.columns)

    def _empty(self) -> np.ndarray:
        """Predictions for no queries."""
        return np.empty((0, self.outputs), dtype=np.float32)

    def _targets(self, targets: Any) -> np.ndarray | list[str | None]:
        """Regression targets, class codes, or class labels as strings."""
        if self.classes is not None:
            labels = _labels_of(targets)
            if labels is not None:
                return labels
            return _vector(_filled(targets, -1), np.int32)
        values = _vector(_filled(targets, np.nan), np.float32)
        if np.isinf(values).any():
            raise ValueError(
                "regression targets are finite, or NaN for rows not labeled yet"
            )
        return values

    @abc.abstractmethod
    def insert(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        """Adds labeled rows.

        Every write carries an id of its own, so one a server took but could
        not acknowledge, such as through a router whose replicas were out of
        reach, applies once however often it is retried. A write of no rows
        does nothing.

        Args:
            keys: Each row's key in the source table: integers, strings, bytes
                or UUIDs; see `generalforecast.keys`.
            features: Rows with the table's columns: an array, a frame or an
                Arrow table, with NaN or null for missing values. Strings in
                categorical columns take codes from the table's vocabulary.
            targets: Class labels `0 .. classes - 1`, or regression targets. A
                row whose label is not known yet takes NaN, or for a class -1,
                or None among string labels; see `label`.
            times: Each row's time as an integer, such as epoch seconds, for
                queries with a cutoff. Rows without one are visible to every query.
        """

    @abc.abstractmethod
    def update(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        """Changes existing rows, and their times if given. Contexts holding them are encoded again."""

    @abc.abstractmethod
    def upsert(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        """Adds rows, replacing the rows of keys the table already holds."""

    @abc.abstractmethod
    def label(self, keys: Iterable[Key], targets: Any) -> None:
        """Sets rows' targets, keeping their features and times.

        Rows inserted before their labels are known stay out of the table's
        fits and of every context until they are labeled, unless the model
        reads unlabeled context rows (`TabularProperties.unlabeled_context`).
        Scaling and drift read labeled rows alone either way.

        Args:
            keys: The rows' keys.
            targets: Their targets, as `insert` takes them.
        """

    @abc.abstractmethod
    def delete(self, keys: Iterable[Key], missing_ok: bool = False) -> None:
        """Removes rows; with `missing_ok`, keys the table does not hold are skipped."""

    @abc.abstractmethod
    def truncate(self) -> None:
        """Deletes every row, as when the source table is truncated."""

    @abc.abstractmethod
    def predict(
        self,
        queries: Any,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Predicts query rows.

        Args:
            queries: Rows to predict, with the table's columns as `insert` takes
                them; a category the table has not seen counts as missing.
            before: A time cutoff per query, or one for all: each query's
                context holds only rows from before it, and so does the table's
                state, such as TabICL's preprocessing and column summaries: it
                is fit on the rows before the request's earliest cutoff, or kept
                from an earlier cutoff as `refit_cutoffs` allows. A cutoff after
                every row reads the state fit on every row. On a partitioned
                table only the context keeps to the cutoffs: the state is the
                first partition's, fit on its rows whatever the cutoffs.
            quantiles: For a regressor's table, the quantile levels to predict,
                each strictly between 0 and 1, such as `[0.05, 0.5, 0.95]`;
                every quantile the model predicts if None. Each lies between
                the two of the model's evenly spaced quantiles around it, and a
                server sends only these.

        Returns:
            Class probabilities, one column per class in code order, or
            quantiles on the original target scale, one column per level.
        """

    @abc.abstractmethod
    def predict_given(
        self,
        queries: Any,
        contexts: Sequence[Iterable[Key]],
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Predicts query rows, each against exactly the rows of the keys given it.

        The table chooses no rows: each query's context is its given rows,
        encoded together, such as an external index's candidates for a table
        larger than its own index. The table's state is fit as for `predict`,
        or under `summary="context"` on each context's rows, and a context
        past the model's trained length warns, or under `past_cap="clamp"`
        keeps its first keys up to that length, in the order given: an
        index's ranking, best first. Rows deleted since are left out, as are
        rows without a label for a model that reads none, and a query left
        with no rows is refused.

        Args:
            queries: Rows to predict, as `predict` takes them.
            contexts: The keys of each query's context rows, one list per query.
            quantiles: Quantile levels, as `predict` takes them.

        Returns:
            Predictions, as `predict` returns them.

        Raises:
            ValueError: If there is not one context per query, a context is
                empty, or a key names no row.
        """

    @abc.abstractmethod
    def contexts(self, queries: Any, before: Any = None) -> list[np.ndarray]:
        """The keys of the rows each query's context holds, without predicting.

        The rows are those `predict` would read for the queries now, the
        state's refit included, with nothing encoded.

        Args:
            queries: Rows, as `predict` takes them.
            before: Time cutoffs, as `predict` takes them.

        Returns:
            Per query, its context's keys in the order the model reads them,
            ascending as unsigned 128-bit integers: an (n, 2) uint64 array of
            each key's low and high 64 bits, as `generalforecast.keys` holds them. An
            integer key from 0 to 2**64 - 1 is its low half, with a high half
            of 0; a negative one is two's complement, with a high half of all
            ones, so it comes after every key that is not negative.
        """

    @abc.abstractmethod
    def region_churn(self, queries: Any) -> np.ndarray:
        """How much each query's region has changed since its context was built.

        Its rows inserted, updated or deleted since the build, and its
        context's rows a recency window has since left behind, as a share of
        the context's rows: what a `rebuild` share is compared with.

        Args:
            queries: Rows, as `predict` takes them.

        Returns:
            Per query, its region's churn; NaN where the region has no built
            context or its context is no longer cached, as the next read
            builds it again, and for every query of a table without regions.
        """

    @abc.abstractmethod
    def refit(self) -> None:
        """Refits the table's state on its live rows at the next prediction."""

    @abc.abstractmethod
    def set_strategy(self, strategy: Strategy) -> None:
        """Changes how each query's context is chosen from here on.

        Logged like a write, so the table keeps it across restarts. The
        ranking, cores and regions drawn for the old strategy are drawn anew
        at the next prediction; cached encodings stay.
        """

    @abc.abstractmethod
    def compact(self) -> None:
        """Rewrites the table's log as its live rows, so old values leave the disk."""

    @abc.abstractmethod
    def vocabularies(self) -> dict[str, list[str]]:
        """Each categorical column's words, in code order."""

    @abc.abstractmethod
    def describe(self) -> Described:
        """The table as it is now: its definition, rows, vocabularies and labels."""

    @property
    @abc.abstractmethod
    def labels(self) -> list[str] | None:
        """A classifier's labels in the order of its prediction columns.

        A table whose targets are strings codes them in the order it first saw
        them, up to its classes; one whose targets are codes `0 ..`, and a
        regressor's, has none.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of live rows."""


@dataclass(frozen=True)
class SeriesDefinition:
    """What a table of series is, as it was made.

    Attributes:
        name: The table's name.
        series: The column naming each observation's series.
        time: The column holding each observation's time.
        target: The column holding the value forecast.
        columns: The covariate columns, with their kinds.
        known: The covariate columns known over the horizon, by name.
        step: The grid's step: an integer of the table's time unit, or a
            timedelta for a datetime table.
        times: What times and cutoffs are: `integer`, in a unit of the
            caller's, or `datetime`.
        model: The model's name, or None for one given as its calls and parameters.
        group: Which other series join a series in its group.
        horizon: Steps a forecast covers when its call names none, if set.
        quantile_levels: The level of each forecast's last axis, increasing.
    """

    name: str
    series: str
    time: str
    target: str
    columns: tuple[Column, ...]
    known: tuple[str, ...]
    step: Any
    times: Times
    model: str | None
    group: GroupStrategy
    horizon: int | None
    quantile_levels: tuple[float, ...]


class SeriesTable(abc.ABC):
    """A table of series: insert observations, and forecast series from a time.

    Rows are observations: a series, a time on the series' grid, the target's
    value there, and the covariates' values. Frames of rows carry these as
    columns named as the table was made.

    Attributes:
        definition: What the table is, as it was made.
        name: The table's name.
        series: The column naming each observation's series.
        time: The column holding each observation's time.
        target: The column holding the value forecast.
        columns: The covariate columns, with their kinds.
        known: The covariate columns known over the horizon, by name.
        step: The grid's step.
        times: What times and cutoffs are: `integer` or `datetime`.
        model: The model's name, or None for one given as its calls and parameters.
        group: Which other series join a series in its group.
        horizon: Steps a forecast covers when its call names none, if set.
        quantile_levels: The level of each forecast's last axis.
        last_answer: What answered the table's latest forecasts, None before any.
    """

    def __init__(self, definition: SeriesDefinition) -> None:
        """Wraps a table so defined."""
        self.definition = definition
        self.name, self.series = definition.name, definition.series
        self.time, self.target = definition.time, definition.target
        self.columns, self.known = definition.columns, definition.known
        self.step, self.times = definition.step, definition.times
        self.model, self.group = definition.model, definition.group
        self.horizon = definition.horizon
        self.quantile_levels = definition.quantile_levels
        self.last_answer: Answer | None = None
        # Each series key as given, by the 64 bits the node knows it by.
        self._names: dict[int, Any] = {}

    def known_series(self) -> list[Any]:
        """The keys of every series this table has taken observations for, as given.

        The table keeps the keys of the series written through it, in the order
        it first saw them.
        """
        return list(self._names.values())

    def _keys(self, series: Any) -> np.ndarray:
        """Series keys as the node knows them: the low 64 bits of each digest."""
        keys = keys_of(series)
        digests = np.ascontiguousarray(keys.halves[:, 0])
        for digest, given in zip(digests.tolist(), keys.given.tolist(), strict=True):
            self._names.setdefault(digest, given)
        return digests

    def _placed(self, rows: Any) -> tuple[np.ndarray, np.ndarray]:
        """Each observation's time and row key, from a frame."""
        (series, times), count = pick(rows, (self.series, self.time))
        stamps = _times(times, count, self.times)
        assert stamps is not None
        digests, keys = series_keys(series, stamps)
        for digest, given in zip(
            digests.tolist(), keys_of(series).given.tolist(), strict=True
        ):
            self._names.setdefault(digest, given)
        return stamps, keys

    @contextlib.contextmanager
    def _naming(self) -> Iterator[None]:
        """Re-raises a key or value error naming series as the caller gave them."""
        try:
            yield
        except (KeyError, ValueError) as error:
            message = str(error.args[0]) if error.args else str(error)

            def given(match: re.Match[str]) -> str:
                key = self._names.get(int(match.group(1)))
                return match.group(0) if key is None else f"series {key!r}"

            raise type(error)(_SERIES.sub(given, message)) from None

    def _shaped(
        self, forecasts: np.ndarray, levels: Sequence[float] | None
    ) -> np.ndarray:
        """Forecasts at the levels asked, from the model's."""
        if levels is None:
            return forecasts
        return _at_levels(forecasts, levels, self.quantile_levels)

    def __repr__(self) -> str:  # noqa: D105
        model = "" if self.model is None else f", model {self.model!r}"
        return (
            f"<{type(self).__name__} {self.name!r}: {len(self.columns)} covariates, "
            f"step {self.step!r}{model}>"
        )

    def _empty(self, horizon: int, levels: int) -> np.ndarray:
        """Forecasts of no series."""
        return np.empty((0, horizon, levels), dtype=np.float32)

    def _horizon(self, horizon: int | None) -> int:
        """The steps a call forecasts, checked.

        Raises:
            ValueError: If neither the call nor the table gives a positive one.
        """
        horizon = self.horizon if horizon is None else horizon
        if horizon is None:
            raise ValueError(
                f"table {self.name!r} was made without a horizon; give one to forecast"
            )
        if horizon < 1:
            raise ValueError(f"a horizon is at least one step, not {horizon}")
        return int(horizon)

    @abc.abstractmethod
    def insert(self, rows: Any) -> None:
        """Adds observations.

        Args:
            rows: A frame, Arrow table or mapping of columns with the table's
                series, time, target and covariate columns. A missing target,
                NaN or null, is an unobserved step: a covariate's future
                value, inserted ahead of the series, or a gap. Times lie on
                each series' grid, whole steps apart; a time a series already
                holds is refused, as `upsert` replaces it.
        """

    @abc.abstractmethod
    def upsert(self, rows: Any) -> None:
        """Adds observations, replacing those at times the series already hold."""

    @abc.abstractmethod
    def delete(self, rows: Any, missing_ok: bool = False) -> None:
        """Removes observations by their series and time columns.

        With `missing_ok`, observations the table does not hold are skipped.
        """

    @abc.abstractmethod
    def truncate(self) -> None:
        """Deletes every observation."""

    @abc.abstractmethod
    def forecast(
        self,
        series: Iterable[Key],
        horizon: int | None = None,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Forecasts series over the steps after a time.

        Args:
            series: The series to forecast, by key.
            horizon: Steps to forecast; the table's if None.
            before: The time each forecast starts at, one per series or one
                for all: the context holds the observations before it, and
                known covariates' values from it on. A time off a series'
                grid starts at the next point on it. The step after each
                series' latest observed value if None.
            quantiles: The levels to forecast, each strictly between 0 and 1,
                such as `[0.1, 0.5, 0.9]`; every level the model predicts if
                None. Each lies linearly between the two of the model's levels
                around it, and one beyond the model's outermost level takes
                that level.

        Returns:
            Forecasts `(series, horizon, levels)` on the target's scale.
        """

    @abc.abstractmethod
    def forecast_given(
        self,
        series: Iterable[Key],
        groups: Sequence[Iterable[Key]],
        horizon: int | None = None,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Forecasts series, each in the group of series given it.

        For a table made with `GroupStrategy.given(size)`: the table draws no
        group, each forecast reads exactly the members named, such as the
        nearest series an index outside the table chose, encoded together,
        and two forecasts naming one group at one time share its encoding. A
        series the table never saw is forecast through its group, as a cold
        start.

        Args:
            series: The series to forecast, by key.
            groups: Each series' group, by key, the series itself in it or
                not: one list per series, at most the table's size in all.
            horizon: Steps to forecast; the table's if None.
            before: The time each forecast starts at, as `forecast` takes it.
            quantiles: Quantile levels, as `forecast` takes them.

        Returns:
            Forecasts, as `forecast` returns them.

        Raises:
            ValueError: If there is not one group per series, a group has
                more series than the table's size, or the table draws its
                own groups.
            KeyError: If a group names a series the table does not hold.
        """

    @abc.abstractmethod
    def vocabularies(self) -> dict[str, list[str]]:
        """Each categorical covariate's words, in code order."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of observations."""


def _series_definition_of(reply: Mapping[str, Any]) -> SeriesDefinition:
    """A series table's definition from a server's `describe` reply, checked.

    Raises:
        ValueError: If the reply does not describe a table of series.
    """
    try:
        series = reply["series"]
        if not series:
            raise ValueError("the table holds rows, not series")
        times = reply.get("times", "integer")
        if times not in get_args(Times):
            raise ValueError(f"unknown time kind {times!r}")
        step: Any = int(series["step"])
        if times == "datetime":
            step = datetime.timedelta(microseconds=step)
        group = series.get("group") or {"kind": "alone"}
        match group.get("kind"):
            case "alone":
                strategy = GroupStrategy.alone()
            case "random":
                strategy = GroupStrategy.random(int(group["size"]), int(group["seed"]))
            case "given":
                strategy = GroupStrategy.given(int(group["size"]))
            case kind:
                raise ValueError(f"unknown group strategy {kind!r}")
        horizon = series.get("horizon")
        levels = reply.get("quantile_levels") or ()
        return SeriesDefinition(
            name=str(reply["name"]),
            series=str(series["series"]),
            time=str(series["time"]),
            target=str(series["target"]),
            columns=tuple(
                Column(column["name"], column_kind(column["name"], column["kind"]))
                for column in reply["columns"]
            ),
            known=tuple(str(name) for name in series.get("known") or ()),
            step=step,
            times=times,
            model=str(reply["model"]),
            group=strategy,
            horizon=None if horizon is None else int(horizon),
            quantile_levels=tuple(float(level) for level in levels),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"a malformed table description: {error!r}") from None


def _implied(columns: int) -> tuple[float, ...]:
    """The levels of a tabular regressor's evenly spaced quantile columns."""
    return tuple((column + 1) / (columns + 1) for column in range(columns))


def _given(contexts: Sequence[Iterable[Key]], count: int) -> list[np.ndarray]:
    """Each query's context keys as an (n, 2) uint64 array of key halves.

    Raises:
        ValueError: If there is not one context per query, or one is empty.
    """
    keys = [keys_of(context).halves for context in contexts]
    if len(keys) != count:
        raise ValueError(f"expected {count} contexts, one per query, got {len(keys)}")
    if any(len(context) == 0 for context in keys):
        raise ValueError("every query's context needs a row")
    return keys


def _levels(
    name: str, known: Sequence[float] | None, quantiles: Sequence[float] | None
) -> list[float] | None:
    """Quantile levels to predict, checked, or None for every quantile.

    Raises:
        ValueError: If the table predicts no quantiles, or a level is not
            strictly between 0 and 1.
    """
    if quantiles is None:
        return None
    if known is None:
        raise ValueError(f"table {name!r} predicts classes, not quantiles")
    levels = [float(level) for level in quantiles]
    if not levels or not all(0.0 < level < 1.0 for level in levels):
        raise ValueError("quantile levels lie strictly between 0 and 1")
    return levels


def _at_levels(
    predictions: np.ndarray, levels: Sequence[float], known: Sequence[float]
) -> np.ndarray:
    """Predictions at quantile levels, between the model's levels around each.

    The last axis of `predictions` holds the `known` levels; a level lies
    linearly between the two around it, or takes the outermost beyond them,
    as servers compute it.
    """
    positions = np.interp(levels, known, np.arange(len(known)))
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(known) - 1)
    weight = positions - lower
    values = predictions.astype(np.float64)
    picked = values[..., lower] * (1.0 - weight) + values[..., upper] * weight
    return picked.astype(np.float32)


def _labels_of(targets: Any) -> list[str | None] | None:
    """Class labels as strings, None for a label not known yet, or None for codes.

    Raises:
        ValueError: If strings and numbers mix.
    """
    if hasattr(targets, "to_pylist"):
        targets = targets.to_pylist()
    elif hasattr(targets, "to_numpy") and not isinstance(targets, np.ndarray):
        targets = targets.to_numpy()
    array = np.asarray(list(targets) if not hasattr(targets, "__len__") else targets)
    if array.dtype.kind in "biuf":
        return None
    values = array.ravel().tolist()
    if array.dtype.kind == "S":
        values = [value.decode() for value in values]
    if not any(isinstance(value, str) for value in values):
        return None
    if not all(value is None or isinstance(value, str) for value in values):
        raise ValueError("class labels are strings or codes 0 .., not both")
    return values


def _times(values: Any, count: int, kind: Times) -> np.ndarray | None:
    """Row times or cutoffs as int64, one per row, from an array or one value.

    A table's times are integers of the caller's unit, or for a datetime table
    dates and times, held as microseconds since 1970-01-01; each kind refuses
    the other's values, so a table never mixes units.
    """
    if values is None:
        return None
    if np.ndim(values) == 0:
        values = [values] * count
    elif not hasattr(values, "__len__") and hasattr(values, "__iter__"):
        values = list(values)
    to_numpy = getattr(values, "to_numpy", None)
    if to_numpy is not None and not isinstance(values, np.ndarray):
        values = to_numpy()
    if kind == "datetime":
        stamps = moments("times", values, "us")
        if np.isnat(stamps).any():
            raise ValueError("times cannot be missing")
        times = stamps.astype(np.int64)
    else:
        array = np.asarray(values)
        temporal = array.dtype.kind in "Mm" or (
            array.dtype.kind == "O"
            and any(
                isinstance(value, datetime.date | np.datetime64)
                for value in array.ravel()
            )
        )
        if temporal:
            raise ValueError(
                "this table's times are integers; create it with times='datetime' "
                "to give dates and times"
            )
        times = _vector(array, np.int64)
    if len(times) != count:
        raise ValueError(f"expected {count} times, got {len(times)}")
    return times


def _check_times(times: str) -> None:
    """Refuses a time type other than `integer` or `datetime`."""
    if times not in ("integer", "datetime"):
        raise ValueError(f"times is 'integer' or 'datetime', not {times!r}")


def _bucket(width: Any, kind: Times) -> int | None:
    """A time bucket's width in the table's time units: microseconds for datetime tables."""
    return None if width is None else _duration(width, kind, "time_bucket")


def _duration(width: Any, kind: Times, what: str) -> int:
    """A span of time in the table's time units: microseconds for datetime tables.

    Raises:
        TypeError: If a datetime table's span is not a timedelta, or an
            integer table's is one.
    """
    if kind == "datetime":
        if not isinstance(width, datetime.timedelta | np.timedelta64):
            raise TypeError(f"a datetime table's {what} is a timedelta")
        return int(np.timedelta64(width, "us").astype(np.int64))
    if isinstance(width, datetime.timedelta | np.timedelta64):
        raise TypeError(
            f"this table's times are integers, so its {what} is one too; "
            "create it with times='datetime' to give a timedelta"
        )
    return int(width)


def _check_classes(classes: int | None) -> None:
    """Refuses a class count no classifier's table can have."""
    if classes is not None and classes < 2:
        # sklearn's checks look for "1 class" in what a one-class fit raises.
        noun = "class" if classes == 1 else "classes"
        raise ValueError(
            f"a classifier tells at least 2 classes apart, not {classes} {noun}; "
            "a regressor's table takes none"
        )
