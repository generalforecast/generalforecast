# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Python Node and Worker table client, kept to test the worker's scheduling white-box."""

import datetime
import functools
import shutil
import sys
import tempfile
import warnings
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

from generalforecast._core import DEFAULT_FIT_ROWS, GroupStrategy, Node, Strategy
from generalforecast.columns import (
    Column,
    categorical,
    columns_of,
    encode,
    offsets,
    prepare,
    retrieval,
    width,
)
from generalforecast.keys import Key, keys_of
from generalforecast.models import registry
from generalforecast.models.contract import TabularFunctions, TabularProperties
from generalforecast.models.series import SeriesFunctions
from generalforecast.native.handles import (  # noqa: F401
    _REBUILD_WAIT,
    _SERIES,
    Definition,
    Described,
    PastCap,
    Rebuild,
    RebuildWait,
    RefitCutoffs,
    RetrievalCategoricals,
    SeriesDefinition,
    SeriesTable,
    Summary,
    Table,
    Times,
    WriteKind,
    _at_levels,
    _bucket,
    _check_classes,
    _check_times,
    _duration,
    _filled,
    _given,
    _labels_of,
    _levels,
    _rebuild_of,
    _rebuild_spec,
    _series_definition_of,
    _strategy_of,
    _times,
    _vector,
    _wait_of,
    implied_levels,
)
from generalforecast.runtime.budget import MemoryBudget
from generalforecast.runtime.device import DeviceSettings, default_settings
from generalforecast.worker import Worker, WorkerStats, check_summary
from tests.utils.driver import Driver

# The most a test client's device tier takes by default, and its host tier
# as much again: a runner's test workers each make many clients.
TEST_TIER = 256 << 20

TabularModel = tuple[TabularFunctions, Any]


SeriesModel = tuple[SeriesFunctions, Any]


def _rebuild_arguments(rebuild: Rebuild) -> dict[str, Any]:
    """A rebuild policy as the node takes it: a kind, and its share or period."""
    if isinstance(rebuild, datetime.timedelta):
        return {"rebuild": "every", "rebuild_at": rebuild.total_seconds()}
    if isinstance(rebuild, int | float):
        return {"rebuild": "churn", "rebuild_at": float(rebuild)}
    return {"rebuild": rebuild}


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

    def insert(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._change("insert", keys, features, targets, times)

    def update(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._change("update", keys, features, targets, times)

    def upsert(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._change("upsert", keys, features, targets, times)

    def label(self, keys: Iterable[Key], targets: Any) -> None:
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

    def delete(self, keys: Iterable[Key], missing_ok: bool = False) -> None:
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

    def truncate(self) -> None:
        with self._client._lock:
            self._live()
            self._client._node.truncate(self._id)
            self._client._purge()

    def predict(
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

    def predict_given(
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

    def contexts(self, queries: Any, before: Any = None) -> list[np.ndarray]:
        with self._client._lock:
            self._live()
            rows, _ = self._encode(queries, assign=False)
            if len(rows) == 0:
                return []
            cutoffs = _times(before, len(rows), self.times)
            return self._client._node.contexts(self._id, rows, cutoffs)

    def region_churn(self, queries: Any) -> np.ndarray:
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
    def labels(self) -> list[str] | None:
        with self._client._lock:
            return list(self._labels) or None

    def describe(self) -> Described:
        with self._client._lock:
            self._live()
            return Described(
                **{
                    name: getattr(self.definition, name)
                    for name in Definition.__dataclass_fields__
                },
                rows=self._client._node.len(self._id),
                resolved_strategy=self._client._node.strategy(self._id),
                effective_fit_rows=None
                if self.definition.summary == "context"
                else self._client._node.effective_fit_rows(self._id),
                vocabularies=self.vocabularies(),
                labels=self.labels,
                region_stats=self._client._node.region_stats(self._id),
            )

    def set_strategy(self, strategy: Strategy) -> None:
        with self._client._lock:
            self._live()
            self._client._node.set_strategy(self._id, strategy)
            self.definition = replace(self.definition, strategy=strategy)

    def vocabularies(self) -> dict[str, list[str]]:
        with self._client._lock:
            self._live()
            return {
                column.name: list(vocabulary)
                for column, vocabulary in zip(
                    self.columns, self._vocabularies, strict=True
                )
                if column.kind == "categorical"
            }

    def refit(self) -> None:
        with self._client._lock:
            self._live()
            for movement in self._client._node.refit(self._id):
                self._client._worker.apply(movement)

    def compact(self) -> None:
        """Does nothing: local tables keep no log."""

    def __len__(self) -> int:
        with self._client._lock:
            self._live()
            return self._client._node.len(self._id)


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
        stamps, keys, _ = self._placed(rows)
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

    def insert(self, rows: Any) -> None:
        self._change("insert", rows)

    def upsert(self, rows: Any) -> None:
        self._change("upsert", rows)

    def delete(self, rows: Any, missing_ok: bool = False) -> None:
        _, keys, _ = self._placed(rows)
        if not len(keys):
            return
        with self._client._lock, self._naming():
            self._live()
            if missing_ok:
                keys = self._client._node.present(self._id, keys)
            self._client._node.delete(self._id, keys)
            self._client._purge()

    def truncate(self) -> None:
        with self._client._lock:
            self._live()
            self._client._node.truncate(self._id)
            self._client._purge()

    def forecast(
        self,
        series: Iterable[Key],
        horizon: int | None = None,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
        future: Any = None,
    ) -> np.ndarray:
        return self._forecast(series, None, horizon, before, quantiles, future)

    def forecast_given(
        self,
        series: Iterable[Key],
        groups: Sequence[Iterable[Key]],
        horizon: int | None = None,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
        future: Any = None,
    ) -> np.ndarray:
        return self._forecast(series, groups, horizon, before, quantiles, future)

    def _forecast(
        self,
        series: Iterable[Key],
        groups: Sequence[Iterable[Key]] | None,
        horizon: int | None,
        before: Any,
        quantiles: Sequence[float] | None,
        future: Any = None,
    ) -> np.ndarray:
        """Forecasts series, in the groups given them if any."""
        if future is not None:
            raise ValueError(
                "the in-process node takes a known future through gf-server"
            )
        horizon = self._horizon(horizon)
        levels = _levels(self.name, self.quantile_levels, quantiles)
        digests = self._keys(series)
        if not len(digests):
            return self._empty(horizon, len(levels or self.quantile_levels))
        cutoffs = _times(before, len(digests), self.times)
        members = None if groups is None else self._groups(groups, len(digests))
        self._live()
        submit = functools.partial(
            self._client._node.submit_forecast,
            self._id,
            digests,
            horizon,
            cutoffs,
            members,
        )
        with self._naming():
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

    def vocabularies(self) -> dict[str, list[str]]:
        with self._client._lock:
            self._live()
            return {
                column.name: list(vocabulary)
                for column, vocabulary in zip(
                    self.columns, self._vocabularies, strict=True
                )
                if column.kind == "categorical"
            }

    def __len__(self) -> int:
        with self._client._lock:
            self._live()
            return self._client._node.len(self._id)


class LocalClient:
    """A node and a worker in this process.

    Safe to share between threads: a `Driver` thread runs the node's steps,
    and a prediction made while another runs joins the next step with every
    other caller's, as a server batches its callers' queries into shared
    steps; one `predict` call with many tables shares a step too.
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
                device's by default, as a local client takes, at most
                `TEST_TIER`.
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
            # A test process makes many clients, whose tables fill a few
            # megabytes, so its tiers are smaller than a local client's.
            device_bytes = min(budget.device_tier, TEST_TIER)
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
        self._driver = Driver(self._node, self._worker)
        self._minimum = bucket_minimum
        self._tables: dict[str, LocalTable | LocalSeriesTable] = {}
        self._models: dict[int, tuple[int, Any]] = {}
        # Models loaded by spec, so tables naming one share its weights and calls.
        self._loaded: dict[str, registry.Model] = {}
        self._lock = self._driver.lock

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickles as the settings to make a new, empty client with."""
        return functools.partial(LocalClient, **self._options), ()

    def __repr__(self) -> str:
        return f"<LocalClient: {len(self._tables)} tables>"

    def close(self) -> None:
        """Drops every table, frees the device pool and removes any temporary disk tier."""
        self._driver.close()
        with self._lock:
            for name in self.tables():
                self.drop_table(name)
            self._worker.close()
            self._cleanup()
            self._temporary = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
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
                predictor does, and embeds that context's queries against it.
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
            self._driver.fitting[table_id] = (model_id, features)
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
                quantile_levels=None
                if count
                else implied_levels(properties.num_outputs),
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
            self._loaded[model] = registry.load_engine_model(model)
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
            self._driver.fitting.pop(table._id, None)

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
        # A table dropped before the driver queues its request is refused by name.
        answered = self._answer(submissions)
        return {
            name: answered[name] if name in answered else empty[name]
            for name in queries
        }

    def _answer(
        self, submissions: Sequence[tuple[str, Callable[[], int]]]
    ) -> dict[str, np.ndarray]:
        """Answers requests through the driver, in steps shared with other callers, by name.

        Called without the lock, which the driver takes to queue the requests.
        Nothing it queues outlives the call.

        Raises:
            KeyError: If the node refuses a request naming what does not exist.
            ValueError: If the node refuses a request otherwise, naming its table.
        """
        answered = self._driver.answer(submissions)
        named = {name for name, _ in submissions}
        for name in sorted(named):
            table = self._tables.get(name)
            if isinstance(table, LocalTable) and table._id in answered.past_cap:
                warnings.warn(
                    f"table {table.name!r} read a context of more than "
                    f"{table.definition.context_rows:,} rows, the most its "
                    "model was trained on; past_cap='clamp' cuts contexts "
                    "to that length",
                    stacklevel=4,
                )
        for name, answer in answered.answers.items():
            if name in self._tables:
                self._tables[name].last_answer = answer
        return answered.predictions


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


def local(**options: Any) -> LocalClient:
    """A client with a node and worker in this process; see `LocalClient`."""
    return LocalClient(**options)
