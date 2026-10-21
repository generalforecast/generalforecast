# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""In-process native table engine; remote services use generalforecast.Client."""

import abc
import contextlib
import datetime
import functools
import re
import shutil
import sys
import tempfile
import threading
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, get_args

import numpy as np

from generalforecast._core import DEFAULT_FIT_ROWS, GroupStrategy, Node, Step, Strategy
from generalforecast.columns import (
    Column,
    categorical,
    column_kind,
    columns_of,
    encode,
    moments,
    offsets,
    pick,
    prepare,
    retrieval,
    width,
)
from generalforecast.keys import Key, keys_of, series_keys
from generalforecast.models import registry
from generalforecast.models.contract import TabularFunctions, TabularProperties
from generalforecast.models.series import SeriesFunctions
from generalforecast.runtime.budget import MemoryBudget
from generalforecast.runtime.device import DeviceSettings, default_settings
from generalforecast.worker import Worker, WorkerStats, check_summary

TabularModel = tuple[TabularFunctions, Any]
SeriesModel = tuple[SeriesFunctions, Any]

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
class Described(Definition):
    """A table as it is now: its definition, rows, vocabularies and labels.

    Attributes:
        rows: Number of live rows.
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
                rows=int(reply["rows"]),
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


def _rebuild_arguments(rebuild: Rebuild) -> dict[str, Any]:
    """A rebuild policy as the node takes it: a kind, and its share or period."""
    if isinstance(rebuild, datetime.timedelta):
        return {"rebuild": "every", "rebuild_at": rebuild.total_seconds()}
    if isinstance(rebuild, int | float):
        return {"rebuild": "churn", "rebuild_at": float(rebuild)}
    return {"rebuild": rebuild}


def _rebuild_spec(rebuild: Rebuild) -> Any:
    """A rebuild policy as servers store it."""
    if isinstance(rebuild, datetime.timedelta):
        return {"every": rebuild.total_seconds()}
    if isinstance(rebuild, int | float):
        return {"churn": float(rebuild)}
    return rebuild


def _filled(targets: Any, unknown: float) -> Any:
    """A frame's or Arrow array's targets, its nulls `unknown`: not known yet."""
    if hasattr(targets, "fill_null"):
        return targets.fill_null(unknown)
    if hasattr(targets, "fillna") and hasattr(targets, "isna") and targets.isna().any():
        return targets.astype("float64" if np.isnan(unknown) else "Int64").fillna(
            unknown
        )
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


class LocalTable(Table):
    """A table on a local client."""

    def __init__(
        self, client: "LocalClient", table: int, definition: Definition
    ) -> None:
        """Wraps a table the client created."""
        super().__init__(definition)
        self._client, self._id = client, table
        self._vocabularies: list[dict[str, int]] = [{} for _ in self.columns]
        self._labels: dict[str, int] = {}

    def _live(self) -> None:
        """Refuses a handle to a table dropped since, by its name."""
        if self._client._tables.get(self.name) is not self:
            raise KeyError(f"table {self.name!r} does not exist")

    def _encode(
        self, rows: Any, assign: bool
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        """Rows as a float32 matrix, and the words that took new codes."""
        prepared, _ = prepare(rows, self.columns)
        return encode(prepared, self.columns, self._vocabularies, assign)

    def _learn(self, added: dict[int, list[str]]) -> None:
        """Keeps the codes a write's new words took, once its rows are in."""
        for index, words in added.items():
            vocabulary = self._vocabularies[index]
            for word in words:
                vocabulary[word] = len(vocabulary)

    def _change(
        self,
        kind: WriteKind,
        keys: Iterable[Key],
        features: Any,
        targets: Any,
        times: Any,
    ) -> None:
        wide = keys_of(keys)
        if not len(wide):
            return
        targets = self._targets(targets)
        stamps = _times(times, len(wide), self.times)
        with self._client._lock, wide.naming():
            self._live()
            rows, added = self._encode(features, assign=True)
            labels, new = self._codes(targets)
            node = self._client._node
            match kind:
                case "insert":
                    write = node.insert
                case "update":
                    write = node.update
                case "upsert":
                    write = node.upsert
            write(self._id, wide.halves, rows, labels, stamps)
            self._learn(added)
            for label in new:
                self._labels[label] = len(self._labels)
            self._client._purge()

    def insert(  # noqa: D102
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._change("insert", keys, features, targets, times)

    def update(  # noqa: D102
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._change("update", keys, features, targets, times)

    def upsert(  # noqa: D102
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._change("upsert", keys, features, targets, times)

    def label(self, keys: Iterable[Key], targets: Any) -> None:  # noqa: D102
        wide = keys_of(keys)
        if not len(wide):
            return
        targets = self._targets(targets)
        with self._client._lock, wide.naming():
            self._live()
            labels, new = self._codes(targets)
            self._client._node.label(self._id, wide.halves, labels)
            for label in new:
                self._labels[label] = len(self._labels)
            self._client._purge()

    def delete(self, keys: Iterable[Key], missing_ok: bool = False) -> None:  # noqa: D102
        wide = keys_of(keys)
        if not len(wide):
            return
        with self._client._lock, wide.naming():
            self._live()
            halves = wide.halves
            if missing_ok:
                halves = self._client._node.present(self._id, halves)
            self._client._node.delete(self._id, halves)
            self._client._purge()

    def truncate(self) -> None:  # noqa: D102
        with self._client._lock:
            self._live()
            self._client._node.truncate(self._id)
            self._client._purge()

    def predict(  # noqa: D102
        self,
        queries: Any,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        self._live()
        levels = _levels(self.name, self.definition.quantile_levels, quantiles)
        cutoffs = None if before is None else {self.name: before}
        predictions = self._client.predict({self.name: queries}, cutoffs)[self.name]
        if levels is None:
            return predictions
        assert self.definition.quantile_levels is not None
        return _at_levels(predictions, levels, self.definition.quantile_levels)

    def predict_given(  # noqa: D102
        self,
        queries: Any,
        contexts: Sequence[Iterable[Key]],
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        known = self.definition.quantile_levels
        levels = _levels(self.name, known, quantiles)
        with self._client._lock:
            self._live()
            rows, _ = self._encode(queries, assign=False)
            if len(rows) == 0:
                predictions = self._empty()
            else:
                keys = _given(contexts, len(rows))
                submit = functools.partial(
                    self._client._node.submit_keys, self._id, rows, keys
                )
                predictions = self._client._answer([(self.name, submit)])[self.name]
        if levels is None:
            return predictions
        assert known is not None
        return _at_levels(predictions, levels, known)

    def contexts(self, queries: Any, before: Any = None) -> list[np.ndarray]:  # noqa: D102
        with self._client._lock:
            self._live()
            rows, _ = self._encode(queries, assign=False)
            if len(rows) == 0:
                return []
            cutoffs = _times(before, len(rows), self.times)
            return self._client._node.contexts(self._id, rows, cutoffs)

    def region_churn(self, queries: Any) -> np.ndarray:  # noqa: D102
        with self._client._lock:
            self._live()
            rows, _ = self._encode(queries, assign=False)
            if len(rows) == 0:
                return np.empty(0)
            return self._client._node.region_churn(self._id, rows)

    def _codes(
        self, targets: np.ndarray | list[str | None]
    ) -> tuple[np.ndarray, list[str]]:
        """Targets as the node takes them, and the labels that took new codes."""
        if isinstance(targets, np.ndarray):
            if targets.dtype == np.int32 and self._labels:
                raise ValueError(
                    f"table {self.name!r}'s classes are labels {list(self._labels)}; "
                    "give its targets as labels too"
                )
            return targets, []
        if not self._labels and self._client._node.len(self._id):
            raise ValueError(
                f"table {self.name!r}'s classes are codes 0 .. {(self.classes or 1) - 1}; "
                "give its targets as codes too"
            )
        codes, new = _code_labels(targets, self._labels, self.classes or 0, self.name)
        return codes, new

    @property
    def labels(self) -> list[str] | None:  # noqa: D102
        with self._client._lock:
            return list(self._labels) or None

    def describe(self) -> Described:  # noqa: D102
        with self._client._lock:
            self._live()
            return Described(
                **{
                    name: getattr(self.definition, name)
                    for name in Definition.__dataclass_fields__
                },
                rows=self._client._node.len(self._id),
                effective_fit_rows=None
                if self.definition.summary == "context"
                else self._client._node.effective_fit_rows(self._id),
                vocabularies=self.vocabularies(),
                labels=self.labels,
                region_stats=self._client._node.region_stats(self._id),
            )

    def vocabularies(self) -> dict[str, list[str]]:  # noqa: D102
        with self._client._lock:
            self._live()
            return {
                column.name: list(vocabulary)
                for column, vocabulary in zip(
                    self.columns, self._vocabularies, strict=True
                )
                if column.kind == "categorical"
            }

    def refit(self) -> None:  # noqa: D102
        with self._client._lock:
            self._live()
            for movement in self._client._node.refit(self._id):
                self._client._worker.apply(movement)

    def compact(self) -> None:
        """Does nothing: local tables keep no log."""

    def __len__(self) -> int:  # noqa: D105
        with self._client._lock:
            self._live()
            return self._client._node.len(self._id)


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


class LocalSeriesTable(SeriesTable):
    """A table of series on a local client."""

    def __init__(
        self, client: "LocalClient", table: int, definition: SeriesDefinition
    ) -> None:
        """Wraps a table the client created."""
        super().__init__(definition)
        self._client, self._id = client, table
        self._vocabularies: list[dict[str, int]] = [{} for _ in self.columns]

    def _live(self) -> None:
        """Refuses a handle to a table dropped since, by its name."""
        if self._client._tables.get(self.name) is not self:
            raise KeyError(f"table {self.name!r} does not exist")

    def _change(self, kind: Literal["insert", "upsert"], rows: Any) -> None:
        stamps, keys = self._placed(rows)
        if not len(keys):
            return
        prepared, count = prepare(rows, (*self.columns, Column(self.target)))
        targets = np.ascontiguousarray(prepared[-1], dtype=np.float32)
        with self._client._lock, self._naming():
            self._live()
            if self.columns:
                matrix, added = encode(
                    prepared[:-1], self.columns, self._vocabularies, assign=True
                )
            else:
                matrix, added = np.empty((count, 0), dtype=np.float32), {}
            node = self._client._node
            write = node.insert if kind == "insert" else node.upsert
            write(self._id, keys, matrix, targets, stamps)
            for index, words in added.items():
                vocabulary = self._vocabularies[index]
                for word in words:
                    vocabulary[word] = len(vocabulary)
            self._client._purge()

    def insert(self, rows: Any) -> None:  # noqa: D102
        self._change("insert", rows)

    def upsert(self, rows: Any) -> None:  # noqa: D102
        self._change("upsert", rows)

    def delete(self, rows: Any, missing_ok: bool = False) -> None:  # noqa: D102
        _, keys = self._placed(rows)
        if not len(keys):
            return
        with self._client._lock, self._naming():
            self._live()
            if missing_ok:
                keys = self._client._node.present(self._id, keys)
            self._client._node.delete(self._id, keys)
            self._client._purge()

    def truncate(self) -> None:  # noqa: D102
        with self._client._lock:
            self._live()
            self._client._node.truncate(self._id)
            self._client._purge()

    def forecast(  # noqa: D102
        self,
        series: Iterable[Key],
        horizon: int | None = None,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        return self._forecast(series, None, horizon, before, quantiles)

    def forecast_given(  # noqa: D102
        self,
        series: Iterable[Key],
        groups: Sequence[Iterable[Key]],
        horizon: int | None = None,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        return self._forecast(series, groups, horizon, before, quantiles)

    def _forecast(
        self,
        series: Iterable[Key],
        groups: Sequence[Iterable[Key]] | None,
        horizon: int | None,
        before: Any,
        quantiles: Sequence[float] | None,
    ) -> np.ndarray:
        """Forecasts series, in the groups given them if any."""
        horizon = self._horizon(horizon)
        levels = _levels(self.name, self.quantile_levels, quantiles)
        digests = self._keys(series)
        if not len(digests):
            return self._empty(horizon, len(levels or self.quantile_levels))
        cutoffs = _times(before, len(digests), self.times)
        members = None if groups is None else self._groups(groups, len(digests))
        with self._client._lock, self._naming():
            self._live()
            submit = functools.partial(
                self._client._node.submit_forecast,
                self._id,
                digests,
                horizon,
                cutoffs,
                members,
            )
            forecasts = self._client._answer([(self.name, submit)])[self.name]
        return self._shaped(forecasts, levels)

    def _groups(self, groups: Sequence[Iterable[Key]], count: int) -> list[np.ndarray]:
        """Each series' group as the node's digests, one group per series.

        Raises:
            ValueError: If there is not one group per series.
        """
        found = [self._keys(group) for group in groups]
        if len(found) != count:
            raise ValueError(
                f"expected {count} groups, one per series, got {len(found)}"
            )
        return found

    def vocabularies(self) -> dict[str, list[str]]:  # noqa: D102
        with self._client._lock:
            self._live()
            return {
                column.name: list(vocabulary)
                for column, vocabulary in zip(
                    self.columns, self._vocabularies, strict=True
                )
                if column.kind == "categorical"
            }

    def __len__(self) -> int:  # noqa: D105
        with self._client._lock:
            self._live()
            return self._client._node.len(self._id)


class LocalClient:
    """A node and a worker in this process.

    Safe to share between threads: calls take turns, so one thread's
    prediction waits for another's rather than joining its step. A server
    batches its callers' queries into shared steps; here, one `predict` call
    with many tables shares a step.
    """

    def __init__(
        self,
        device_bytes: int | None = None,
        host_bytes: int | None = None,
        slot_bytes: int = 1 << 20,
        settings: DeviceSettings | None = None,
        bucket_minimum: int = 8,
        disk_bytes: int = 0,
        disk_path: Path | None = None,
        compile_cache: bool = True,
        step_bytes: int | None = None,
        eviction: Literal["gdsf", "lru"] = "gdsf",
        compile_ahead: bool = True,
    ) -> None:
        """Sets up an empty node and worker.

        Args:
            device_bytes: Device memory for encoded contexts; a quarter of the
                device's by default, at most 4 GiB, since a local client
                shares its machine with the programs around it.
            host_bytes: Host memory for contexts evicted from the device; as
                much as the device tier by default.
            slot_bytes: Granularity of cache memory.
            disk_bytes: Disk for contexts evicted from host memory; none by default.
            disk_path: The disk tier's file; a new temporary one by default.
            settings: Device settings for precision and attention; if None, this
                device's settings in the mode `GF_PRECISION` names, as a
                server's workers use.
            bucket_minimum: Smallest padded size for compiled calls.
            compile_cache: Keep compiled programs in `~/.cache/generalforecast/jax`, so a
                later process skips most compiling. This turns on JAX's
                process-wide compilation cache, unless one is configured
                already; pass False to leave JAX's configuration alone.
            step_bytes: Context bytes one step may newly encode; `device_bytes`
                if None. Tables past it wait for a later step, so a large
                table that must encode holds up the rest for less time.
            eviction: Which cached context a full tier gives up first: `gdsf`
                keeps what is costly to recompute per byte and often read,
                `lru` keeps what was read most recently.
            compile_ahead: Compile the sizes a table may need next (pages a
                write changes, larger query batches) on a thread while the
                client is idle. Pass False when only the sizes called are
                wanted, as when timing calls: each then compiles in the call
                that first needs it, and nothing compiles beside a later one.
        """
        # How to make this client again, as unpickling does.
        self._options = {
            "device_bytes": device_bytes,
            "host_bytes": host_bytes,
            "slot_bytes": slot_bytes,
            "settings": settings,
            "bucket_minimum": bucket_minimum,
            "disk_bytes": disk_bytes,
            "disk_path": disk_path,
            "compile_cache": compile_cache,
            "step_bytes": step_bytes,
            "eviction": eviction,
            "compile_ahead": compile_ahead,
        }
        if settings is None:
            settings = default_settings()
        budget = MemoryBudget.measure(local=True)
        if device_bytes is None:
            device_bytes = budget.device_tier
        if host_bytes is None:
            host_bytes = device_bytes
        self._temporary: Path | None = None
        if disk_bytes and disk_path is None:
            self._temporary = Path(tempfile.mkdtemp(prefix="gf-"))
            disk_path = self._temporary / "cache.slots"
        # Removes the disk tier's directory on close, or once the client is
        # collected or the process exits without closing it.
        self._cleanup = weakref.finalize(
            self, shutil.rmtree, self._temporary or "", ignore_errors=True
        )
        if self._temporary is None:
            self._cleanup.detach()
        self._node = Node(
            device_bytes, host_bytes, slot_bytes, disk_bytes, step_bytes, eviction
        )
        self._worker = Worker(
            settings,
            bucket_minimum,
            slot_bytes,
            disk_path if disk_bytes else None,
            compile_cache,
            budget,
            compile_ahead,
        )
        self._worker.allocate(
            device_bytes // slot_bytes,
            host_bytes // slot_bytes,
            disk_bytes // slot_bytes,
        )
        self._minimum = bucket_minimum
        self._tables: dict[str, LocalTable | LocalSeriesTable] = {}
        self._models: dict[int, tuple[int, Any]] = {}
        # Models loaded by spec, so tables naming one share its weights and calls.
        self._loaded: dict[str, registry.Model] = {}
        # Each tabular table's model and feature count, which measure its fits.
        self._fitting: dict[int, tuple[int, int]] = {}
        # Reentrant, so a table dropped by the garbage collector mid-call cannot deadlock.
        self._lock = threading.RLock()

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickles as the settings to make a new, empty client with."""
        return functools.partial(LocalClient, **self._options), ()

    def __repr__(self) -> str:  # noqa: D105
        return f"<LocalClient: {len(self._tables)} tables>"

    def close(self) -> None:
        """Drops every table, frees the device pool and removes any temporary disk tier."""
        with self._lock:
            for name in self.tables():
                self.drop_table(name)
            self._worker.close()
            self._cleanup()
            self._temporary = None

    def __enter__(self) -> Self:  # noqa: D105
        return self

    def __exit__(self, *_: object) -> None:  # noqa: D105
        self.close()

    def models(self) -> list[str]:
        """The model specs tables can name, sorted.

        These are the built-in models, which load on first use, and any other
        spec loaded so far.
        """
        with self._lock:
            return sorted(set(registry.BUILT_IN) | set(self._loaded))

    def _purge(self) -> None:
        """Drops cached entries that held rows since changed or deleted."""
        for movement in self._node.take_drops():
            self._worker.apply(movement)

    @property
    def stats(self) -> WorkerStats:
        """The worker's counters."""
        return self._worker.stats

    def create_table(
        self,
        name: str,
        *,
        model: str | TabularModel,
        features: int | None = None,
        columns: Any = None,
        classes: int | None = None,
        strategy: Strategy | None = None,
        page_size: int = 128,
        time_bucket: Any = None,
        refit_after: float | None = None,
        refit_shift: float | None = None,
        times: Times = "integer",
        fit_rows: int = DEFAULT_FIT_ROWS,
        past_cap: PastCap = "warn",
        region_rows: int | None = None,
        rebuild: Rebuild = "always",
        rebuild_wait: RebuildWait = _REBUILD_WAIT,
        refit_cutoffs: RefitCutoffs = "each",
        scale_rows: int | None = None,
        summary: Summary = "table",
        retrieval_categoricals: RetrievalCategoricals = "codes",
    ) -> LocalTable:
        """Adds an empty table.

        Args:
            name: The table's name, unique on this client.
            model: A model name or spec for `generalforecast.models.registry`, or the model's
                contract calls and parameters.
            features: Number of numeric feature columns, named `f0 ..`; or give
                `columns`.
            columns: The feature columns by name, each with its kind: `numeric`,
                `categorical` (strings, or codes `0 ..`, which models such as
                Kumo Tabular embed apart), `boolean`, `datetime` or
                `retrieval`: numbers that place rows and queries for retrieval,
                such as embeddings, which the model never reads and which
                follow every other column; the retrieval space is then exactly
                these columns, unscaled. Names alone make numeric columns. Or a
                frame, such as the rows to insert: a pandas or polars frame, an
                Arrow table or a mapping of columns, whose column types give the
                kinds; see `generalforecast.columns.infer`.
            classes: Number of classes, for a classifier's table. A regressor's
                table takes none: the model decides what a table predicts.
            strategy: How each query's context is chosen; exact by default.
            page_size: Rows per page.
            time_bucket: A window of time that pages never straddle, in the
                units of row times, or a timedelta for a datetime table; with
                cutoffs, pages wholly before one keep their caches as later rows
                arrive.
            refit_after: Refit the table's state on its own once rows inserted,
                updated or deleted since the last fit reach this fraction of the
                rows it was fit on. A query with a cutoff reads a state fit only
                on rows before it; with this set, a state fit before an earlier
                cutoff also serves later ones, with their caches, until the rows
                between reach this fraction, and without it each cutoff reads a
                state fit on all the rows before it.
            refit_shift: Refit once a column's mean moves by more than this many
                of its standard deviations at the last fit.
            times: What row times and cutoffs are: `integer`, in a unit of your
                choosing, or `datetime`, for dates and times.
            fit_rows: The most rows the table's state is fit on. A table holding
                more fits on a fixed sample of this many and embeds each page
                only once a query reads it, so its fits stay bounded. A wide
                table whose state on this many rows would not fit a worker's
                call budget fits on a sample of the rows it takes, the
                description's `effective_fit_rows`. A fit on more rows than
                one call embeds with its state also embeds each page only
                once a query reads it.
            past_cap: What a context larger than the model was trained on does:
                `warn` reads it whole and warns once; `clamp` cuts it to that
                length, every row to a random sample of it and a retrieved
                context to its core and fewer neighbors.
            region_rows: Group queries into regions of about this many live
                rows, by k-means over the retrieval space, each region's
                queries sharing one context: the strategy's core plus the
                rows nearest the region's center. Up to 1,024 regions are one
                level of k-means; past that, the leaves of two. Regions are
                drawn at each refit and each time the retrieval scaling is
                fit again, so their number follows the rows until the scaling
                stops (see `fit_rows`), and each draw builds every context
                anew. A strategy without neighbors has one region. Without
                `region_rows`, each query reads a context of its own.
            rebuild: When a region's context is built again, in a step of its
                own while queries keep reading the old one: `always` after any
                change to its rows, `never`, a float for once changed rows
                reach that share of its rows, or a timedelta for once that
                long has passed since it was built.
            rebuild_wait: How long a context due a rebuild is read stale while
                queries wait, in steps of queries and, unless its `seconds`
                is None, seconds; past either, its rebuild takes a step ahead
                of them. Steps alone make two runs of the same writes and
                queries read the same contexts.
            refit_cutoffs: When a query's cutoff inside the table's history
                calls for a state fit on the rows before it: `each` new cutoff,
                once rows between the state's cutoff and the query's reach
                `refit_after`'s share (`changed`), or `never`. A state fit on a
                row from at or after a query's cutoff is fit again whatever the
                setting. Under `each`, queries that alternate between a cutoff
                and none refit at every switch; under `never` a historical
                cutoff refits too, and later queries without one keep that
                state until `refit_after`, `refit_shift` or `refit` fits it
                again. A cutoff after every
                row, as a live stream's are, reads the state fit on every row,
                which only `refit_after`, `refit_shift` or `refit` fit again.
            scale_rows: Rows the retrieval scaling stays fixed after, once fit
                on at least that many. Without it, the scaling is fit again
                each time the rows double, up to `fit_rows`, and pages are cut
                and regions drawn anew each time. Each fit reads the labeled
                rows at that moment, so two tables of the same rows whose
                labels came in a different order may scale, and so retrieve,
                differently until the scaling stops; samples and ties do not
                depend on the order rows came in. Retrieval columns are never
                scaled, so it does nothing on a table with them.
            summary: What the model's state, preprocessing and column
                summary, is fit on: `table` fits it once on the table's rows,
                so row representations are cached per page; `context` fits the
                unit's own state on each context's rows, as the model's own
                package does, and embeds that context's queries against it.
                Only models whose state pools over rows, such as TabICL,
                TabPFN and Kumo Tabular, take `context`. Such a table's units
                depend on their rows alone, so refit policies and cutoffs
                refit nothing.
            retrieval_categoricals: How categorical columns place rows for
                retrieval, on a table without retrieval columns: as their
                codes, standardized like numbers (`codes`); one-hot over each
                column's 15 most frequent categories and one dimension for
                the rest, a missing value at none (`one_hot`), each dimension
                1/sqrt(2) so two categories lie one deviation apart, as two
                numbers one deviation apart do; or by the targets of each
                category's rows (`target`): its standardized mean target, or
                its rate of each class, averaged with 10 rows of the table's
                so a rare category is not placed by a label or two, an unseen
                or missing category taking the table's. Encodings are fit on
                table rows only, never a query's, and only with the retrieval
                scaling, as the rows double up to `scale_rows`: labels that
                change in between move nothing until then.

        Returns:
            The table.
        """
        made = columns_of(features, columns)
        _check_times(times)
        with self._lock:
            if name in self._tables:
                raise ValueError(f"table {name!r} exists")
            named = model if isinstance(model, str) else None
            served = named or "the model"
            functions, parameters = self._load(model)
            if not isinstance(functions, TabularFunctions):
                raise TypeError(
                    f"{served} is a forecaster; make a series table with it instead"
                )
            strategy = strategy or Strategy.exact()
            properties = functions.properties(parameters)
            count = _check_task(served, properties, classes)
            check_summary(properties, summary)
            model_id = self._model_id(functions, parameters)
            # Datetime columns become more than one model feature each, and
            # retrieval columns none.
            features = width(made) - retrieval(made)
            profile = self._worker.profile(model_id, features)
            self._worker.layout(model_id, features)
            self._worker.check(model_id, count, features)
            bound = self._worker.fit_bound(model_id, features)
            table_id = self._node.create_table(
                model_id,
                width(made),
                count,
                strategy,
                profile.cache_bytes_per_row,
                bytes_per_representation=profile.row_bytes,
                page_size=page_size,
                bucket_minimum=self._minimum,
                time_bucket=_bucket(time_bucket, times),
                refit_after=refit_after,
                refit_shift=refit_shift,
                fit_rows=fit_rows,
                context_rows=properties.context_rows,
                past_cap=past_cap,
                region_rows=region_rows,
                **_rebuild_arguments(rebuild),
                rebuild_steps=rebuild_wait.steps,
                rebuild_seconds=rebuild_wait.seconds,
                refit_cutoffs=refit_cutoffs,
                scale_rows=scale_rows,
                retrieval=retrieval(made),
                summary=summary,
                state_bytes=profile.table_state_bytes(bool(categorical(made))),
                categorical=categorical(made),
                retrieval_categoricals=retrieval_categoricals,
                unlabeled_context=properties.unlabeled_context,
                fit_bound=None if bound >= sys.maxsize else bound,
            )
            self._fitting[table_id] = (model_id, features)
            self._worker.add_table(
                table_id,
                model_id,
                count,
                features,
                categorical(made),
                page_size,
                summary,
            )
            definition = Definition(
                name=name,
                columns=made,
                classes=classes,
                outputs=count or properties.num_outputs,
                times=times,
                model=named,
                strategy=strategy,
                page_size=page_size,
                time_bucket=time_bucket,
                refit_after=refit_after,
                refit_shift=refit_shift,
                fit_rows=fit_rows,
                past_cap=past_cap,
                context_rows=properties.context_rows,
                region_rows=region_rows,
                rebuild=rebuild,
                rebuild_wait=rebuild_wait,
                refit_cutoffs=refit_cutoffs,
                scale_rows=scale_rows,
                summary=summary,
                retrieval_categoricals=retrieval_categoricals,
                quantile_levels=None if count else _implied(properties.num_outputs),
            )
            table = self._tables[name] = LocalTable(self, table_id, definition)
            return table

    def create_series_table(
        self,
        name: str,
        *,
        model: str | SeriesModel,
        series: str,
        time: str,
        target: str,
        step: Any,
        columns: Any = None,
        known: Iterable[str] = (),
        group: GroupStrategy | None = None,
        horizon: int | None = None,
        times: Times = "integer",
    ) -> LocalSeriesTable:
        """Adds an empty table of series.

        Args:
            name: The table's name, unique on this client.
            model: A forecaster's name or spec for `generalforecast.models.registry`, or
                its contract calls and parameters.
            series: The column naming each observation's series.
            time: The column holding each observation's time.
            target: The column holding the value forecast.
            step: The grid's step: an integer of the table's time unit, or a
                timedelta for a datetime table. Each series' times are whole
                steps apart, and a forecast covers whole steps. A calendar
                frequency that is not regular in the time unit, such as
                months, quarters or business days, takes period numbers as
                times, with `step=1` and integer times.
            columns: The covariate columns by name, each with its kind as
                `create_table` takes them, or a frame such as the rows to
                insert, whose other columns give the kinds. None for a table
                of targets alone.
            known: The covariate columns known over the horizon, such as
                prices or holidays: a forecast reads their values from the
                observations at and after its time, and the other covariates
                from before it alone.
            group: Which other series join a series in its group, forecast
                jointly; each alone by default.
            horizon: Steps a forecast covers when its call names none.
            times: What times and cutoffs are: `integer`, in a unit of your
                choosing, or `datetime`, for dates and times.

        Returns:
            The table.
        """
        _check_times(times)
        fixed = (series, time, target)
        if len(set(fixed)) != len(fixed):
            raise ValueError(
                "the series, time and target columns need names of their own"
            )
        made = () if columns is None else columns_of(None, columns)
        made = tuple(column for column in made if column.name not in fixed)
        names = [column.name for column in made]
        known = list(known)
        unknown = [column for column in known if column not in names]
        if unknown:
            raise ValueError(f"known columns {unknown} are not covariate columns")
        group = group or GroupStrategy.alone()
        if horizon is not None and horizon < 1:
            raise ValueError(f"a horizon is at least one step, not {horizon}")
        with self._lock:
            if name in self._tables:
                raise ValueError(f"table {name!r} exists")
            named = model if isinstance(model, str) else None
            functions, parameters = self._load(model)
            if not isinstance(functions, SeriesFunctions):
                raise TypeError(
                    f"{named or 'the model'} is not a forecaster; "
                    "make a table with it instead"
                )
            properties = functions.properties(parameters)
            model_id = self._model_id(functions, parameters)
            features = width(made)
            # A known datetime column's parts are all known.
            starts = [*offsets(made), features]
            known_features = [
                feature
                for index, column in enumerate(made)
                if column.name in known
                for feature in range(starts[index], starts[index + 1])
            ]
            sizes = self._worker.series_profile(model_id)
            # A group as large as the table's strategy draws, with the covariates.
            self._worker.check_series(model_id, features, rows=group.size())
            table_id = self._node.create_series_table(
                model_id,
                features,
                _duration(step, times, "step"),
                properties.patch,
                properties.context_limit,
                properties.output_patch,
                group,
                sizes.per_row,
                pass_limit=properties.pass_limit,
                patch_bytes=sizes.per_patch,
                known=known_features,
                max_rows=properties.max_rows,
                reusable=properties.reusable,
                bucket_minimum=self._minimum,
                extendable=functions.extend is not None,
            )
            self._worker.add_table(table_id, model_id, 0, features, series=True)
            definition = SeriesDefinition(
                name=name,
                series=series,
                time=time,
                target=target,
                columns=made,
                known=tuple(column for column in names if column in known),
                step=step,
                times=times,
                model=named,
                group=group,
                horizon=horizon,
                quantile_levels=tuple(properties.quantiles),
            )
            table = self._tables[name] = LocalSeriesTable(self, table_id, definition)
            return table

    def _load(self, model: str | TabularModel | SeriesModel) -> registry.Model:
        """A model's calls and parameters, loading a spec once."""
        if not isinstance(model, str):
            return model
        if model not in self._loaded:
            self._loaded[model] = registry.load(model)
        return self._loaded[model]

    def _model_id(self, functions: Any, parameters: Any) -> int:
        """The worker's id for a model, loading it there on first use."""
        if id(parameters) not in self._models:
            # Holding the parameters keeps their id from being reused.
            self._models[id(parameters)] = (len(self._models), parameters)
            self._worker.add_model(len(self._models) - 1, functions, parameters)
        return self._models[id(parameters)][0]

    def table(self, name: str) -> LocalTable:
        """A table by name.

        Raises:
            KeyError: If there is no such table.
            TypeError: If the table holds series; `series_table` gives it.
        """
        table = self._any_table(name)
        if not isinstance(table, LocalTable):
            raise TypeError(f"table {name!r} holds series; use series_table")
        return table

    def series_table(self, name: str) -> LocalSeriesTable:
        """A table of series by name.

        Raises:
            KeyError: If there is no such table.
            TypeError: If the table holds rows to predict from; `table` gives it.
        """
        table = self._any_table(name)
        if not isinstance(table, LocalSeriesTable):
            raise TypeError(f"table {name!r} holds rows, not series; use table")
        return table

    def _any_table(self, name: str) -> LocalTable | LocalSeriesTable:
        if name not in self._tables:
            raise KeyError(f"table {name!r} does not exist")
        return self._tables[name]

    def drop_table(self, name: str) -> None:
        """Removes a table, its rows and its caches."""
        with self._lock:
            table = self._tables.pop(name)
            for movement in self._node.remove_table(table._id):
                self._worker.apply(movement)
            self._worker.remove_table(table._id)
            self._fitting.pop(table._id, None)

    def tables(self) -> list[str]:
        """Every table's name, sorted."""
        return sorted(self._tables)

    def predict(
        self, queries: Mapping[str, Any], before: Mapping[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        """Predicts queries for many tables, in as few steps as the node's budget allows.

        Args:
            queries: Rows to predict, by table name.
            before: Time cutoffs, by table name, as `Table.predict` takes them.

        Returns:
            Predictions by table name.
        """
        submissions = []
        empty = {}
        # Held throughout, so a table dropped meanwhile cannot leave a stale id.
        with self._lock:
            for name, rows in queries.items():
                table = self.table(name)
                rows, _ = table._encode(rows, assign=False)
                if len(rows) == 0:
                    empty[name] = table._empty()
                    continue
                cutoffs = _times((before or {}).get(name), len(rows), table.times)
                submit = functools.partial(self._node.submit, table._id, rows, cutoffs)
                submissions.append((name, submit))
            answered = self._answer(submissions)
        return {
            name: answered[name] if name in answered else empty[name]
            for name in queries
        }

    def _next_step(self) -> Step | None:
        """The next step the node plans, once the tables it holds for their fits' rows are measured."""
        while True:
            step = self._node.schedule()
            waiting = self._node.take_unmeasured()
            for table in waiting:
                model, features = self._fitting[table]
                self._node.set_fit_limits(
                    table, *self._worker.fit_rows(model, features)
                )
            if step is not None or not waiting:
                return step

    def _answer(
        self, submissions: Sequence[tuple[str, Callable[[], int]]]
    ) -> dict[str, np.ndarray]:
        """Submits requests, runs the steps they plan into, and answers them by name.

        Called with the lock held. Nothing it queues outlives the call.

        Raises:
            KeyError: If the node refuses a request naming what does not exist.
            ValueError: If the node refuses a request otherwise, naming its table.
        """
        requests: dict[int, str] = {}
        predictions: dict[int, np.ndarray] = {}
        answers: dict[int, Answer] = {}
        try:
            for name, submit in submissions:
                try:
                    requests[submit()] = name
                except ValueError as error:
                    raise ValueError(f"table {name!r}: {error}") from None
            while (step := self._next_step()) is not None:
                spent = sum(self._worker.stats.seconds.values())
                try:
                    predictions.update(self._worker.run(step))
                except BaseException:
                    # The worker may hold only part of what the step encoded.
                    for movement in self._node.fail(step):
                        self._worker.apply(movement)
                    if self._worker.lost:
                        # Its device pool went with a failed write.
                        self._node.forget_cache()
                        self._worker.forget_cache()
                    raise
                for movement in self._node.complete(step):
                    self._worker.apply(movement)
                # A request's share of the step: by the queries' units read,
                # and by the forecasts made.
                spent = sum(self._worker.stats.seconds.values()) - spent
                read = sum(len(batch.units) for batch in step.queries)
                read = (read + len(step.forecasts)) or 1
                for batch in step.queries:
                    answers[batch.request] = Answer(
                        batch.epoch, spent * len(batch.units) / read
                    )
                made: dict[int, tuple[int, int]] = {}
                for forecast in step.forecasts:
                    epoch, count = made.get(forecast.request, (forecast.epoch, 0))
                    made[forecast.request] = (epoch, count + 1)
                for request, (epoch, count) in made.items():
                    answers[request] = Answer(epoch, spent * count / read)
        finally:
            # Nothing this call queued outlives it, to run or fail in the next.
            refused = self._node.take_refused()
            past_cap = self._node.take_past_cap()
            for request in requests:
                self._node.cancel(request)
        for table in self._tables.values():
            if isinstance(table, LocalTable) and table._id in past_cap:
                warnings.warn(
                    f"table {table.name!r} read a context of more than "
                    f"{table.definition.context_rows:,} rows, the most its "
                    "model was trained on; past_cap='clamp' cuts contexts "
                    "to that length",
                    stacklevel=4,
                )
        for request, error, missing in refused:
            raised = KeyError if missing else ValueError
            raise raised(f"table {requests[request]!r}: {error}")
        for request, name in requests.items():
            if request in answers:
                self._tables[name].last_answer = answers[request]
        return {name: predictions[request] for request, name in requests.items()}


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


def _code_labels(
    labels: list[str | None], known: Mapping[str, int], classes: int, table: str
) -> tuple[np.ndarray, list[str]]:
    """Labels' codes: new labels take the next up to `classes`, and None takes -1.

    Returns:
        The codes, and the new labels in code order.

    Raises:
        ValueError: If more labels than `classes` would be known.
    """
    codes = np.empty(len(labels), dtype=np.int32)
    new: dict[str, int] = {}
    for index, label in enumerate(labels):
        if label is None:
            codes[index] = -1
            continue
        code = known.get(label, new.get(label))
        if code is None:
            code = new[label] = len(known) + len(new)
            if code >= classes:
                held = [*known, *new][:classes]
                raise ValueError(
                    f"table {table!r} has {classes} classes, labelled {held}; "
                    f"{label!r} would be another"
                )
        codes[index] = code
    return codes, list(new)


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


def _check_task(name: str, properties: TabularProperties, classes: int | None) -> int:
    """The class count a table of the model takes, 0 for a regressor's.

    Raises:
        ValueError: If a regressor is given classes, a classifier none, or a
            count the model cannot serve.
    """
    if properties.regression:
        if classes is not None:
            raise ValueError(
                f"{name} predicts values; create the table without classes"
            )
        return 0
    if classes is None:
        raise ValueError(f"{name} predicts classes; give the table its classes")
    _check_classes(classes)
    if classes > properties.class_slots:
        raise ValueError(
            f"{name} serves at most {properties.class_slots} classes, not {classes}"
        )
    return classes


def _check_classes(classes: int | None) -> None:
    """Refuses a class count no classifier's table can have."""
    if classes is not None and classes < 2:
        # sklearn's checks look for "1 class" in what a one-class fit raises.
        noun = "class" if classes == 1 else "classes"
        raise ValueError(
            f"a classifier tells at least 2 classes apart, not {classes} {noun}; "
            "a regressor's table takes none"
        )


def local(**options: Any) -> LocalClient:
    """A client with a node and worker in this process; see `LocalClient`."""
    return LocalClient(**options)
