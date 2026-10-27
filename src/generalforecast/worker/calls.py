# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Running one stage for many tables: vectorized calls in as few as the call budget allows.

Items of one model, task and argument shapes stack into one compiled call,
split so no call's inputs and outputs exceed the budget; on a CPU, items run
one a call, side by side on the worker's cores.
"""

import functools
import logging
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as finished
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast._core import bucket_size
from generalforecast.constants import FUSED_STACKS, STACK_CHUNK
from generalforecast.runtime.budget import Call
from generalforecast.runtime.decoding import paged_decode
from generalforecast.runtime.device import (
    load_cache,
    to_cache_dtype,
    with_summary_chunk,
)
from generalforecast.runtime.memory import tree_bytes
from generalforecast.runtime.paging import CacheLayout
from generalforecast.runtime.programs import Kind
from generalforecast.runtime.programs import signature as kind_signature
from generalforecast.runtime.shapes import padded as padded_items
from generalforecast.worker.stats import _COUNTING

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


def _signature(arguments: tuple) -> tuple:
    """What a call compiles to: the arguments' structure, shapes and dtypes.

    Read from the leaves' metadata, so no device array is copied to the host.
    """
    leaves, structure = jax.tree.flatten(arguments)
    return structure, tuple(
        (np.shape(leaf), np.dtype(getattr(leaf, "dtype", type(leaf))).str)
        for leaf in leaves
    )


def _stacked_shape(value: Any, count: int, stacked: bool = False) -> Any:
    """The shapes and types of `count` copies of `value` stacked, as a call stacks items.

    With `stacked`, `value` is a tree of leaves, each stacked alone.
    """
    if stacked:
        return jax.tree.map(lambda leaf: _stacked_shape(leaf, count), value)
    return jax.ShapeDtypeStruct(
        (count, *np.shape(value)),
        jnp.result_type(value),
        weak_type=bool(getattr(value, "weak_type", False)),
    )


def _block(value: Any) -> None:
    """Waits for the device to finish computing `value`."""
    jax.block_until_ready(value)


def _host_padded(values: Any, size: int) -> np.ndarray:
    """Rows padded with zeros to `size` on the host, so no shape compiles a pad."""
    values = np.asarray(values)
    padded = np.zeros((size, *values.shape[1:]), dtype=values.dtype)
    padded[: len(values)] = values
    return padded


def _stacked(items: Sequence[Any]) -> Any:
    """Items of one tree structure stacked leaf by leaf on a new first axis.

    Host arrays stack on the host, so they reach the device in one copy with
    their call rather than one dispatch each. The other leaves stack on the
    device in one program for them all, compiled once per number of items,
    where a stack of each leaf alone would compile once per leaf.
    """
    flat, structures = zip(*(jax.tree.flatten(item) for item in items), strict=True)
    columns = list(zip(*flat, strict=True))
    on_host = [
        all(
            isinstance(value, np.ndarray)
            and jax.dtypes.canonicalize_dtype(value.dtype) == value.dtype
            for value in column
        )
        for column in columns
    ]
    leaves: list[Any] = [
        np.stack(column) if host else None
        for column, host in zip(columns, on_host, strict=True)
    ]
    on_device = [index for index, host in enumerate(on_host) if not host]
    if on_device:
        stacked = _stacked_on_device([columns[index] for index in on_device])
        for index, values in zip(on_device, stacked, strict=True):
            leaves[index] = values
    return jax.tree.unflatten(structures[0], leaves)


def _stacked_on_device(columns: list[tuple]) -> list[jax.Array]:
    """Each column's device values stacked on a new first axis.

    Past `STACK_CHUNK` items the columns stack a chunk at a time and one
    program joins the chunks, so no count compiles a stack of thousands.
    """
    count = len(columns[0])
    if count <= STACK_CHUNK:
        return _stack_columns(columns)
    chunks = [
        _stack_columns([column[start : start + STACK_CHUNK] for column in columns])
        for start in range(0, count, STACK_CHUNK)
    ]
    return _join_columns([list(parts) for parts in zip(*chunks, strict=True)])


@jax.jit
def _stack_columns(columns: list[tuple]) -> list[jax.Array]:
    """Each column's values stacked on a new first axis, in one dispatch."""
    return [jnp.stack(column) for column in columns]


@jax.jit
def _join_columns(columns: list[list[jax.Array]]) -> list[jax.Array]:
    """Each column's stacked chunks joined on their first axis, in one dispatch."""
    return [jnp.concatenate(parts) for parts in columns]


@jax.jit
def _item(outputs: Any, at: jax.Array) -> Any:
    """One item of a call's stacked outputs, in one dispatch rather than one per leaf."""
    return jax.tree.map(lambda value: value[at], outputs)


@functools.partial(jax.jit, donate_argnums=0)
def _only_item(outputs: Any) -> Any:
    """The item of a call's outputs stacked over one, taking over their buffers."""
    return jax.tree.map(lambda value: value[0], outputs)


class Calls:
    """How a worker builds, sizes and runs its compiled calls."""

    def _size(self: "Worker", rows: int) -> int:
        return bucket_size(rows, self._minimum)

    def _settled(self: "Worker", value: Any) -> None:
        """Waits for the device to finish `value` if parts are timed, so its time counts in the part that asked for it."""
        if self.timers:
            _block(value)

    def _call(
        self: "Worker",
        stage: str,
        work: Sequence[tuple[int, tuple]],
        consume: Callable[[int, Any], None] | None = None,
    ) -> list[Any]:
        """Runs one stage for many tables, vectorized over items that share a signature.

        Items that share a model, task and argument shapes run as one call, split
        so that no call's inputs and outputs exceed the worker's call budget.

        Args:
            stage: `fit`, `embed`, `encode` or `decode`.
            work: Per item, its table and its arguments after the model
                parameters; host arrays move to the device only with their call.
            consume: Takes each item's outputs as its call finishes, with the
                item's position, instead of holding them all until the end.

        Returns:
            Each item's outputs, in order, or None for each with `consume`.
        """
        groups: dict[tuple, list[int]] = defaultdict(list)
        for index, (table, arguments) in enumerate(work):
            spec = self._tables[table]
            groups[(spec.model, spec.num_classes, _signature(arguments))].append(index)
        results: list[Any] = [None] * len(work)

        def keep(index: int, item: Any) -> None:
            if consume is None:
                results[index] = item
            else:
                consume(index, item)

        for (model, num_classes, signature), indices in groups.items():
            key = (stage, model, num_classes, signature)
            items = [work[index][1] for index in indices]
            self._call_alike(key, items, indices, keep)
        return results

    def _call_alike(
        self: "Worker",
        key: tuple,
        items: list[tuple],
        indices: list[int],
        consume: Callable[[int, Any], None],
    ) -> None:
        """Runs items of one stage, model, task and signature in as few calls as fit.

        `consume` takes each item's outputs with its index from `indices`.
        """
        stage, model, num_classes, _ = key
        parameters = self._models[model].parameters
        compiled = self._compiled_stage(stage, model, num_classes, len(items[0]))
        estimate = self._estimate(stage, model, num_classes, items[0])
        self._run_items(
            stage,
            key,
            compiled,
            (parameters,),
            items,
            estimate,
            indices,
            consume,
            program=(stage, model, num_classes, len(items[0])),
            buckets=self._buckets(stage),
        )

    def _buckets(self: "Worker", stage: str) -> tuple[int, ...] | None:
        """The only stacked counts a stage's calls take, or None for every power of two."""
        return FUSED_STACKS if self._stacks and stage == "fit_decode" else None

    def _run_items(
        self: "Worker",
        stage: str,
        key: tuple,
        compiled: Callable[..., Any],
        shared: tuple,
        items: list[tuple],
        estimate: Callable[[], int],
        indices: Sequence[Any],
        consume: Callable[[Any, Any], None],
        program: tuple | None = None,
        buckets: tuple[int, ...] | None = None,
    ) -> None:
        """Runs alike items through a compiled call in as few calls as fit the budget.

        `consume` takes each item's outputs with its entry in `indices`, and
        `program`, if given, is the `Programs` key `compiled` is kept under.
        Calls take only `buckets`' stacked counts if given, and on an
        accelerator a count not yet compiled compiles on the side.
        """

        def prepare(first: int, stop: int, padded: int) -> Call:
            with self.stats.timed("stack"):
                stacked = _stacked(padded_items(items[first:stop], padded))
            return Call(compiled, shared, stacked, padded, program)

        def taken(first: int, stop: int, outputs: Any) -> None:
            alone = all(leaf.shape[0] == 1 for leaf in jax.tree.leaves(outputs))
            for position in range(stop - first):
                with self.stats.timed("unstack"):
                    item = (
                        _only_item(outputs)
                        if alone
                        else _item(outputs, np.int32(position))
                    )
                consume(indices[first + position], item)

        self._chunks.run(
            stage,
            key,
            len(items),
            self._call_bytes,
            estimate,
            prepare,
            taken,
            self._compile_on_side if self._on_side else None,
            buckets,
        )

    def _compile_on_side(
        self: "Worker", call: Call, done: Callable[[bool], None]
    ) -> None:
        """Compiles a call's program for its shapes beside steps, then calls `done`."""
        self.programs.compile_soon(call.function, (*call.shared, *call.items), done)

    def _estimate(
        self: "Worker", stage: str, model: int, num_classes: int, arguments: tuple
    ) -> Callable[[], int]:
        """What estimates an item's bytes in a call: its inputs and outputs."""

        def measure() -> int:
            function = self._stage(stage, model, num_classes)
            outputs = jax.eval_shape(
                function, self._models[model].parameters, *arguments
            )
            inputs = sum(
                getattr(leaf, "nbytes", 0) for leaf in jax.tree.leaves(arguments)
            )
            return inputs + tree_bytes(outputs)

        def estimate() -> int:
            # Measured by tracing the model, so kept for later processes.
            key = ("estimate", stage, num_classes, repr(kind_signature(arguments)))
            return self._measured(model, key, measure)

        return estimate

    def _count(self: "Worker", stage: str, seconds: float, flops: float) -> None:
        """Counts a compiled call of a stage, the seconds it took and its FLOPs."""
        # A fit of the state alone counts as a fit.
        stage = "fit" if stage == "state" else stage
        with _COUNTING:
            self.stats.calls[stage] = self.stats.calls.get(stage, 0) + 1
            self.stats.seconds[stage] = self.stats.seconds.get(stage, 0.0) + seconds
            self.stats.flops[stage] = self.stats.flops.get(stage, 0.0) + flops

    def _compiled(self: "Worker", stage: str) -> None:
        """Counts a call of a stage whose shape its program had not seen."""
        with _COUNTING:
            self.stats.compiles[stage] = self.stats.compiles.get(stage, 0) + 1

    def _part(self: "Worker", part: str, seconds: float) -> None:
        """Counts one run of a part of a step; see `WorkerStats.parts`."""
        self.stats.part(part, seconds)

    def _seen(self: "Worker", call: Call) -> None:
        """Records a kind's call at its count, once compiled or first run here, if a later process can compile it again."""
        if call.program is None or call.program[1] not in self._specs:
            return
        # What a paged decode reads its slots from follows from its items and
        # the pool; see `_shared`.
        shared = () if call.program[0] == "paged" else call.shared[1:]
        kind = Kind(
            (call.program[0], None, *call.program[2:]),
            kind_signature(shared),
            kind_signature(call.items, stacked=True),
        )
        self.programs.remember(self._specs[call.program[1]], kind, call.padded)

    def _noted(
        self: "Worker", stage: str, key: Hashable, call: Call, seconds: float
    ) -> None:
        """Logs a call that compiled, with its shapes, while parts are timed."""
        if self.stats.timers:
            logger.warning(
                "compiled %s %s at %d items in %.2f s: items %s, kind %s",
                stage,
                call.program,
                call.padded,
                seconds,
                kind_signature(call.items, stacked=True),
                key,
            )

    def _stage(
        self: "Worker", stage: str, model: int, num_classes: int
    ) -> Callable[..., Any]:
        """One stage for one table, before vectorizing."""
        entry = self._tabular(model)
        functions = entry.functions
        # Caches as the runtime stores them, or back in float32 for a model
        # that reads them only as its `encode` returned them.
        stored = functions.properties(entry.parameters).stored_cache
        read = (lambda cache: cache) if stored else load_cache
        settings = self._settings

        def stored_dtype(cache: Any) -> Any:
            return (
                cache
                if settings is None
                else to_cache_dtype(cache, settings.cache_dtype)
            )

        def fit(parameters, features, targets, rows, categorical):
            return functions.fit(
                parameters,
                features,
                targets,
                num_classes,
                num_rows=rows,
                categorical=categorical,
            )

        # The state alone: XLA drops the rows `fit` embeds, and each page is
        # embedded once a plan reads it. A column summary over more cells
        # than the call budget takes at once reads a chunk of rows at a time;
        # a fit that embeds its rows takes few enough to read them at once.
        def state(parameters, features, targets, rows, categorical):
            chunk = self._summary_chunk(model, features.shape[-1])
            parameters = with_summary_chunk(parameters, chunk)
            return fit(parameters, features, targets, rows, categorical)[0]

        stages: dict[str, Callable[..., Any]] = {
            "fit": fit,
            "state": state,
            "embed": functions.embed,
            # Cast inside the call, so it holds the cache in one dtype only.
            "encode": lambda parameters, state, rows, targets, count: stored_dtype(
                functions.encode(parameters, state, rows, targets, num_rows=count)
            ),
            "decode": lambda parameters, state, cache, rows: functions.postprocess(
                parameters,
                state,
                functions.decode(parameters, state, read(cache), rows),
            ),
        }

        # A unit fit on its own rows: its state comes with its cache, and its
        # queries embed against it.
        def fit_encode(parameters, features, targets, rows, categorical):
            state, context_rows = fit(parameters, features, targets, rows, categorical)
            encode = stages["encode"]
            return state, encode(parameters, state, context_rows, targets, rows)

        def embed_decode(parameters, state, cache, features):
            rows = functions.embed(parameters, state, features)
            return stages["decode"](parameters, state, cache, rows)

        def fit_decode(parameters, features, targets, rows, categorical, queries):
            state, cache = fit_encode(parameters, features, targets, rows, categorical)
            return state, embed_decode(parameters, state, cache, queries)

        stages["fit_encode"], stages["embed_decode"] = fit_encode, embed_decode
        stages["fit_decode"] = fit_decode
        return stages[stage]

    def _compiled_stage(
        self: "Worker", stage: str, model: int, num_classes: int, arity: int
    ) -> Callable[..., Any]:
        """A stage vectorized over its arguments, compiled once per model and task."""

        def build() -> Callable[..., Any]:
            function = self._stage(stage, model, num_classes)
            return jax.vmap(function, in_axes=(None,) + (0,) * arity)

        return self.programs.compiled((stage, model, num_classes, arity), build)

    def _padded(
        self: "Worker", features: np.ndarray, targets: np.ndarray, at_least: int = 1
    ) -> tuple[np.ndarray, np.ndarray, int]:
        count = len(features)
        size = self._size(max(count, at_least))
        with self.stats.timed("pad"):
            return _host_padded(features, size), _host_padded(targets, size), count

    def _alone(
        self: "Worker", stage: str, work: Sequence[tuple[int, tuple]]
    ) -> list[Any]:
        """Runs each item in a call of its own, up to a core's worth side by side.

        Outputs come back in the work' order. Every call ends before this
        returns, and a failure raises the first failed item's error, as when
        they run in turn, so `run_apart` still fails only its table.
        """
        if len(work) < 2 or self._side_by_side < 2:
            return [self._call(stage, [item])[0] for item in work]
        if self._alongside is None:
            self._alongside = ThreadPoolExecutor(
                self._side_by_side, thread_name_prefix="gf-alone"
            )
        calls: list[Future[Any]] = [
            self._alongside.submit(lambda item=item: self._call(stage, [item])[0])
            for item in work
        ]
        finished(calls)
        return [call.result() for call in calls]

    def _staged(self: "Worker", work: Sequence[tuple[str, int, tuple]]) -> list[Any]:
        """Runs work of several stages, one `_call` per stage, with outputs in order."""
        by_stage: dict[str, list[int]] = defaultdict(list)
        for index, (stage, _, _) in enumerate(work):
            by_stage[stage].append(index)
        outputs: list[Any] = [None] * len(work)
        for stage, indices in by_stage.items():
            done = self._call(stage, [work[index][1:] for index in indices])
            for index, output in zip(indices, done, strict=True):
                outputs[index] = output
        return outputs

    @staticmethod
    def _paged_key(
        model: int,
        num_classes: int,
        layout: CacheLayout,
        masked: bool,
        in_order: bool,
        own: bool,
    ) -> tuple:
        """The `Programs` key of a model and task's paged decodes for a layout and way of reading."""
        return ("paged", model, num_classes, layout, masked, in_order, own)

    def _compiled_paged(
        self: "Worker",
        model: int,
        num_classes: int,
        layout: CacheLayout,
        masked: bool,
        in_order: bool,
        own: bool,
    ) -> Callable[..., Any]:
        """Paged decodes of a model and task, compiled once per layout and way of reading."""

        def build() -> Callable[..., Any]:
            entry = self._tabular(model)
            functions = entry.functions
            stored = functions.properties(entry.parameters).stored_cache
            return paged_decode(functions, stored, layout, masked, in_order, own)

        key = self._paged_key(model, num_classes, layout, masked, in_order, own)
        return self.programs.compiled(key, build)

    def _program(self: "Worker", key: tuple) -> Callable[..., Any]:
        """The compiled function a `Call.program` key names: a stage or a paged decode."""
        if key[0] == "paged":
            return self._compiled_paged(*key[1:])
        stage, model, num_classes, arity = key
        return self._compiled_stage(stage, model, num_classes, arity)
