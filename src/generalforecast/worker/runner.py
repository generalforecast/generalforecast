# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The worker: runs the steps a node schedules, on one accelerator.

The node decides what to compute and where state lives. The worker holds
everything with data in it: each table's fitted state, and pages of row
representations and encoded contexts packed into the slots the node assigns,
in one pool of device memory and one of host memory, each as large as the
node's tier. Work from many tables that shares a model, task and padded shapes
runs as one vectorized call.
"""

import dataclasses
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast import protocol
from generalforecast._core import CacheKey
from generalforecast.constants import SCATTER_THREADS
from generalforecast.models.contract import TabularFunctions, TabularProperties
from generalforecast.models.series import SeriesFunctions, SeriesProperties
from generalforecast.runtime import shipped
from generalforecast.runtime.budget import Chunks, MemoryBudget
from generalforecast.runtime.decoding import PROGRAMS, Context
from generalforecast.runtime.device import (
    DeviceSettings,
    cache_root,
    configure,
    portable_cpu_programs,
    report_fallback,
    store_cache,
    use_compile_cache,
    without_command_buffers,
)
from generalforecast.runtime.memory import (
    EncodeProfile,
    FitProfile,
    MemoryProfile,
    WorkingProfile,
)
from generalforecast.runtime.paging import (
    CacheLayout,
    DevicePool,
    DiskPool,
    HostPool,
    PoolLost,
    pack,
    pack_state,
    state_layout,
    unpack_states,
)
from generalforecast.runtime.programs import Programs
from generalforecast.runtime.series import SeriesBytes
from generalforecast.system import usable_cpus
from generalforecast.worker.admission import Admission
from generalforecast.worker.ahead import Ahead, _manifests
from generalforecast.worker.calls import Calls, _item, _only_item, _stack_columns
from generalforecast.worker.decodes import Decodes
from generalforecast.worker.encoding import Encoding
from generalforecast.worker.pages import Pages, _Page
from generalforecast.worker.records import (
    Memo,
    Step,
    _Model,
    _SeriesModel,
    _Table,
    _TabularModel,
)
from generalforecast.worker.series import Series
from generalforecast.worker.slots import Slots
from generalforecast.worker.stats import WorkerStats


def check_summary(properties: TabularProperties, summary: str) -> None:
    """Refuses a column-summary scope a model cannot serve.

    Raises:
        ValueError: If `summary` is neither `table` nor `context`, or is
            `context` for a model whose state pools over no context rows.
    """
    if summary not in ("table", "context"):
        raise ValueError(f"summary is table or context, not {summary!r}")
    if summary == "context" and not properties.column_summary:
        raise ValueError(
            "the model has no column summary to fit per context; use summary='table'"
        )


def _message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _tables(step: protocol.Step) -> list[int]:
    """The tables a step does anything for, in order of first appearance."""
    tables = [fit.table for fit in step.fits]
    tables += [page.table for page in step.pages]
    tables += [unit.table for unit in step.units]
    tables += [batch.table for batch in step.queries]
    tables += [unit.table for unit in step.series_units]
    return list(dict.fromkeys(tables))


def _part(step: protocol.Step, table: int) -> protocol.Step:
    """The part of a step for one table, with its indices renumbered and no copies."""
    pages = [index for index, page in enumerate(step.pages) if page.table == table]
    units = [index for index, unit in enumerate(step.units) if unit.table == table]
    series = [
        index for index, unit in enumerate(step.series_units) if unit.table == table
    ]
    page_at = {old: new for new, old in enumerate(pages)}
    unit_at = {old: new for new, old in enumerate(units)}
    series_at = {old: new for new, old in enumerate(series)}
    return protocol.Step(
        movements=[],
        fits=[
            dataclasses.replace(fit, pages=[page_at[index] for index in fit.pages])
            for fit in step.fits
            if fit.table == table
        ],
        pages=[step.pages[index] for index in pages],
        units=[step.units[index] for index in units],
        queries=[
            dataclasses.replace(batch, units=[unit_at[unit] for unit in batch.units])
            for batch in step.queries
            if batch.table == table
        ],
        series_units=[step.series_units[index] for index in series],
        forecasts=[
            dataclasses.replace(forecast, unit=series_at[forecast.unit])
            for forecast in step.forecasts
            if forecast.table == table
        ],
    )


class Worker(Calls, Admission, Ahead, Slots, Pages, Encoding, Decodes, Series):
    """Runs steps for the tables a node assigns it."""

    def __init__(
        self,
        settings: DeviceSettings | None = None,
        bucket_minimum: int = 8,
        slot_bytes: int = 1 << 20,
        disk_path: Path | None = None,
        compile_cache: bool = True,
        budget: MemoryBudget | None = None,
        compile_ahead: bool = True,
        timers: bool = False,
    ) -> None:
        """Sets up an empty worker.

        Args:
            settings: Device settings for precision and attention, if any.
            bucket_minimum: Smallest padded size; must match the node's tables.
            slot_bytes: Bytes in a cache slot; must match the node's.
            disk_path: A file for the disk tier, if the node has one.
            compile_cache: Keep compiled programs on disk for later processes,
                through JAX's process-wide cache; see `use_compile_cache`.
            budget: How the worker divides its memory; this machine's if None.
            compile_ahead: Compile sizes a table may need next in quiet gaps
                (see `Programs`); without it, each compiles when first called.
            timers: Time each part of a step; see `WorkerStats.parts`.
        """
        without_command_buffers()
        portable_cpu_programs()
        report_fallback()
        self._settings, self._minimum = settings, bucket_minimum
        self._slot_bytes = slot_bytes
        # On a CPU a stacked fused call takes as long per unit as one alone,
        # and pads to its bucket and compiles for it, so units run one a
        # call, side by side on the worker's cores: two to four at once
        # served 20 to 40% more than one at a time on four cores.
        self._stacks = jax.default_backend() != "cpu"
        self._side_by_side = 1 if self._stacks else usable_cpus()
        self._alongside: ThreadPoolExecutor | None = None
        # NumPy copies a fancy index's rows without the GIL, so answers
        # scatter on several cores.
        self._scatter_threads = min(SCATTER_THREADS, usable_cpus())
        self._scatterers: ThreadPoolExecutor | None = None
        # Takes the keys of programs readied beside steps for deferred
        # tables; without it, no table's part waits for a program.
        self.announce: Callable[[list[str]], None] | None = None
        # The tables the running step defers, by the program they wait for.
        self.deferred: dict[int, str] = {}
        # Programs being readied beside steps, and those that failed to be,
        # which the steps that need them compile and report; the step thread
        # and the thread readying them share them under the lock.
        self._preparing: set[str] = set()
        self._unpreparable: set[str] = set()
        self._readying = threading.Lock()
        # On an accelerator a program compiles for tens of seconds, so a
        # stacked count not yet compiled compiles on the side; see `Chunks.run`.
        self._on_side = self._stacks and compile_ahead
        if compile_cache:
            use_compile_cache()
        # Where measured fits are kept for later processes, beside compiled programs.
        self._fit_cache = cache_root() / "fit" if compile_cache else None
        # How the worker divides its memory; calls are sized from it as they run.
        self.budget = budget or MemoryBudget.measure()
        self._chunks = Chunks(
            self._count, self._compiled, self._part, self._seen, self._noted
        )
        self._manifest_dir = (
            _manifests(bucket_minimum, slot_bytes) if compile_cache else None
        )
        # The worker's compiled programs, and those it compiles ahead.
        self.programs = Programs(
            settings,
            lambda: self.budget.short_of_memory(),
            (
                _item,
                _only_item,
                _stack_columns,
                store_cache,
                pack,
                pack_state,
                unpack_states,
                *PROGRAMS,
            ),
            ahead=compile_ahead,
            manifests=self._manifest_dir,
            dropped=self._chunks.forget_counts,
        )
        self._models: dict[int, _Model] = {}
        # The spec each model was loaded from, under which its kinds of call are recorded.
        self._specs: dict[int, str] = {}
        self._tables: dict[int, _Table] = {}
        # What each model measures by tracing or compiling, once per shape.
        self._layouts: Memo[tuple[int, int], CacheLayout] = Memo()
        self._profiles: Memo[tuple[int, int], MemoryProfile] = Memo()
        self._decodes: Memo[int, WorkingProfile] = Memo()
        self._encodes: Memo[tuple[int, bool, tuple[int, int]], EncodeProfile | None] = (
            Memo()
        )
        self._encode_bytes: Memo[tuple[int, bool, int, int], int | None] = Memo()
        self._fit_profiles: Memo[tuple[int, int], FitProfile] = Memo()
        self._checked: Memo[tuple[int, int, int], bool] = Memo()
        self._disk_path = disk_path
        self._slots = (0, 0, 0)
        self._pools(0, 0, 0)
        self._allocated = False
        self._contexts: dict[CacheKey, Context] = {}
        # The leaves a series cache keeps beside its patches, by entry.
        self._fixed: dict[CacheKey, list[Any]] = {}
        self._series_layouts: Memo[tuple[int, int], CacheLayout] = Memo()
        self._series_profiles: Memo[int, SeriesBytes] = Memo()
        self._pages: dict[CacheKey, _Page] = {}
        self.stats = WorkerStats()
        self.timers = timers

    @property
    def timers(self) -> bool:
        """Whether each part of a step is timed; see `WorkerStats.parts`."""
        return self.stats.timers

    @timers.setter
    def timers(self: "Worker", timers: bool) -> None:
        self.stats.timers = timers
        self._chunks.timed = self._part if timers else None

    def allocate(self, device_slots: int, host_slots: int, disk_slots: int) -> None:
        """Sizes the pools as the node's tiers; call it before the first step.

        Args:
            device_slots: Slots in the node's device tier.
            host_slots: Slots in its host tier.
            disk_slots: Slots in its disk tier, if the worker has a disk path.
        """
        self._pools(device_slots, host_slots, disk_slots)
        self._allocated = True

    def _pools(self, device_slots: int, host_slots: int, disk_slots: int) -> None:
        """Makes the pools at these sizes, freeing the old device pool first."""
        self._slots = (device_slots, host_slots, disk_slots)
        self._pool_bytes = device_slots * self._slot_bytes
        # What calls that ran out of memory beside the old pool taught no longer holds.
        self._chunks.relax()
        # The old pool goes first, so the device never holds two at once.
        if hasattr(self, "_device"):
            self._device.release()
        self._device = DevicePool(self._slot_bytes, device_slots)
        self._host = HostPool(self._slot_bytes, host_slots)
        self._disk = (
            None
            if self._disk_path is None
            else DiskPool(self._slot_bytes, disk_slots, self._disk_path)
        )

    @property
    def allocated(self) -> bool:
        """Whether `allocate` sized the pools, which steps and `warm_start` read.

        Until then the pools are empty, so a warm-up would compile paged
        decodes for a pool of no slots.
        """
        return self._allocated

    @property
    def _call_bytes(self) -> int:
        """Bytes one compiled call may take beside the device pool."""
        return self.budget.call_bytes(self._pool_bytes)

    @property
    def _window_bytes(self) -> int:
        """Bytes of contexts a decode gathers before its call."""
        return self.budget.window_bytes(self._pool_bytes)

    @property
    def lost(self) -> bool:
        """Whether the device pool lost its contexts; see `forget_cache`."""
        return self._device.lost

    def close(self) -> None:
        """Frees the device pool and forgets every cached context and page."""
        if self._alongside is not None:
            self._alongside.shutdown()
            self._alongside = None
        if self._scatterers is not None:
            self._scatterers.shutdown()
            self._scatterers = None
        self._contexts.clear()
        self._fixed.clear()
        self._pages.clear()
        self._device.release()

    def forget_cache(self) -> None:
        """Drops every cached context and page, with fresh pools.

        The node's `forget_cache` does the same on its side.
        """
        self._contexts.clear()
        self._fixed.clear()
        self._pages.clear()
        self._pools(*self._slots)

    def add_model(
        self,
        model: int,
        functions: TabularFunctions | SeriesFunctions,
        parameters: Any,
        spec: str | None = None,
    ) -> None:
        """Loads a model's parameters under the node's model id.

        With `spec`, what the model was loaded from, the kinds of call it
        compiles for are recorded under it, and `warm_start` compiles them.
        """
        if self._settings is not None:
            parameters = configure(parameters, self._settings)
        self._models[model] = _Model[Any](functions, parameters)
        if spec is not None:
            self._specs[model] = spec
            manifest = self.programs.manifest(spec)
            if manifest is not None and not manifest.path.exists():
                # A machine that never ran the model starts from what its
                # release compiled on this platform, if it shipped any.
                assert self._manifest_dir is not None
                shipped.install(cache_root(), self._manifest_dir.name, spec)

    def forecaster(self, model: int) -> bool:
        """Whether a loaded model is a forecaster."""
        return isinstance(self._models[model].functions, SeriesFunctions)

    def properties(self, model: int) -> TabularProperties:
        """A loaded tabular model's properties."""
        entry = self._tabular(model)
        return entry.functions.properties(entry.parameters)

    def series_extendable(self, model: int) -> bool:
        """Whether a forecaster grows an encoding by whole patches of new steps (`extend`)."""
        return self._series(model).functions.extend is not None

    def series_properties(self, model: int) -> SeriesProperties:
        """A loaded forecaster's properties."""
        entry = self._series(model)
        return entry.functions.properties(entry.parameters)

    def _tabular(self, model: int) -> _TabularModel:
        """A loaded tabular model."""
        entry = self._models[model]
        if isinstance(entry.functions, SeriesFunctions):
            raise ValueError(  # noqa: TRY004
                f"model {model} is a forecaster, which only a series table uses"
            )
        return cast(_TabularModel, entry)

    def _series(self, model: int) -> _SeriesModel:
        """A loaded forecaster."""
        entry = self._models[model]
        if not isinstance(entry.functions, SeriesFunctions):
            raise ValueError(  # noqa: TRY004
                f"model {model} is not a forecaster, which a series table needs"
            )
        return cast(_SeriesModel, entry)

    def add_table(
        self,
        table: int,
        model: int,
        num_classes: int,
        features: int,
        categorical: Sequence[int] = (),
        page_size: int = 1,
        summary: str = "table",
        series: bool = False,
    ) -> None:
        """Serves a table with a loaded model.

        Args:
            table: The node's id for the table.
            model: A loaded model's id.
            num_classes: Number of classes, or 0 for regression.
            features: Number of feature columns.
            categorical: Indices of the columns that hold category codes.
            page_size: The table's page size, which page embeds pad to.
            summary: `table`, or `context` to fit each unit's state on its rows.
            series: Whether the table holds series, forecast by a forecaster.

        Raises:
            ValueError: If a column is out of range, or the model has no
                column summary to fit per context.
        """
        if series:
            self._series(model)
            self._tables[table] = _Table(model, 0, features, series=True)
            return
        self._tabular(model)
        if any(not 0 <= column < features for column in categorical):
            raise ValueError(f"categorical columns must be in 0 .. {features - 1}")
        check_summary(self.properties(model), summary)
        self.programs.add_shape((model, num_classes, features))
        self._remember(model, ("width", features))
        self.layout(model, features)
        mask = None
        if categorical:
            mask = np.zeros(features, dtype=bool)
            mask[list(categorical)] = True
        own = None
        if summary == "context":
            size = self._size(1)
            regression = num_classes == 0
            state, _ = jax.eval_shape(
                self._stage("fit", model, num_classes),
                self._models[model].parameters,
                jax.ShapeDtypeStruct((size, features), jnp.float32),
                jax.ShapeDtypeStruct((size,), jnp.float32 if regression else jnp.int32),
                1,
                mask,
            )
            own = state_layout(state, self._slot_bytes)
        self._tables[table] = _Table(
            model, num_classes, features, mask, page_size=max(1, page_size), own=own
        )

    def remove_table(self, table: int) -> None:
        """Forgets a table's state; the node drops its cache entries separately."""
        del self._tables[table]

    def run(self, step: Step) -> dict[int, np.ndarray]:
        """Runs a step: copies, fits, page embeds, encodes and decodes, in that order.

        Args:
            step: The step from the node.

        Returns:
            Predictions per request id.
        """
        with self.stats.timed("move"):
            for movement in step.movements:
                self.apply(movement)
        return self._counted(step)

    def run_apart(
        self, step: protocol.Step
    ) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        """Runs a step so that one table's error fails only that table.

        The step runs whole first. If that raises, each table's part runs
        alone, so the tables that work still answer.

        Args:
            step: The step from the server.

        Returns:
            Predictions per request id, and an error message per failed table.
            A table `deferred` names has neither: it answers once its program
            is ready, and a failed table is never deferred.
        """
        self.deferred = {}
        with self.stats.timed("move"):
            for movement in step.movements:
                self.apply(movement)
        try:
            return self._counted(step), {}
        except PoolLost:
            # Every table reads the pool, so no part can run alone.
            raise
        except Exception as error:
            tables = _tables(step)
            if not tables:
                raise
            if len(tables) == 1:
                self.deferred.pop(tables[0], None)
                return {}, {tables[0]: _message(error)}
        # Each part defers its table again if its program is still being
        # readied, and not if it was readied meanwhile and the part answered.
        self.deferred = {}
        predictions: dict[int, np.ndarray] = {}
        failed: dict[int, str] = {}
        for table in tables:
            try:
                predictions.update(self._counted(_part(step, table)))
            except Exception as error:  # noqa: BLE001
                failed[table] = _message(error)
                self.deferred.pop(table, None)
        return predictions, failed

    def _counted(self, step: Step) -> dict[int, np.ndarray]:
        """Computes a step, keeping its counts in `stats` only if it succeeds.

        The step's time in no other part is counted as the part `other`. A
        step that fails keeps none of its counts, and its whole time is
        counted as `other`, so the parts still add up to the time it took.
        """
        before = self.stats.snapshot()
        started = time.perf_counter() if self.timers else 0.0
        try:
            with self.programs.step():
                predictions = self._compute(step)
        except BaseException:
            self.stats = before
            if self.timers:
                self.stats.part("other", time.perf_counter() - started)
            raise
        if self.timers:
            taken = time.perf_counter() - started
            in_parts = sum(
                seconds for _, seconds in self.stats.parts_since(before).values()
            )
            self.stats.part("other", taken - in_parts)
        return predictions

    def _compute(self, step: Step) -> dict[int, np.ndarray]:
        """Fits, page embeds, encodes, decodes and forecasts a step whose copies are applied."""
        pages = self._fit(step.fits, step.pages)
        pages.update(self._embed_pages(step.pages, pages))
        transient = self._encode(step.units, step.pages, pages)
        self.stats.steps += 1
        self.stats.queries += sum(len(batch.units) for batch in step.queries)
        self.stats.queries += len(step.forecasts)
        fused = {
            index for index, unit in enumerate(step.units) if unit.action == "fuse"
        }
        reads = [batch for batch in step.queries if fused.isdisjoint(batch.units)]
        alone = [batch for batch in step.queries if not fused.isdisjoint(batch.units)]
        predictions = self._decode(reads, step.units, transient)
        predictions.update(self._decode_fused(alone, step.units))
        predictions.update(self._forecast(step.series_units, step.forecasts))
        return predictions
