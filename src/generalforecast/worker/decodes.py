# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decodes: queries against contexts read straight from their slots, put off until now, or fit in the same call."""

import dataclasses
import logging
import os
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast import protocol
from generalforecast._core import bucket_size
from generalforecast.constants import (
    GROWN_QUERIES,
    HELD_STATES,
    SCATTER_ROWS,
    SPLIT_QUERIES,
    SPLIT_WASTE,
)
from generalforecast.runtime.budget import Call, out_of_memory
from generalforecast.runtime.decoding import (
    Decoding,
    Group,
    alike,
    decoding_of,
    merged,
    packed,
    paged_inputs,
    paged_shape,
    query_rows,
    split,
)
from generalforecast.runtime.paging import (
    CacheLayout,
    StateLayout,
    unpack_states,
)
from generalforecast.runtime.shapes import padded as padded_items
from generalforecast.worker.calls import _host_padded, _item, _stacked
from generalforecast.worker.encoding import Transient
from generalforecast.worker.helper import HELPER_ENV, CompileHelper, compiles_aside
from generalforecast.worker.records import QueryBatch, Unit

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Dense:
    """A batch's queries that read a context there was no room to keep.

    Attributes:
        batch: The batch.
        rows: Its embedded query rows.
        unit: The unit the queries read.
        queries: The queries' positions in the batch.
    """

    batch: QueryBatch
    rows: Any
    unit: int
    queries: list[int]

    def split(self, limit: int | None) -> list["_Dense"]:
        """The queries in parts of at most `limit`, or whole without one."""
        if limit is None or len(self.queries) <= limit:
            return [self]
        return [
            dataclasses.replace(self, queries=self.queries[start : start + limit])
            for start in range(0, len(self.queries), limit)
        ]


def query_pieces(count: int, size: Callable[[int], int]) -> list[int]:
    """Query counts a batch of `count` queries decodes in, so padding wastes little.

    Each call pads its queries to `size`'s bucket, a power of two, so 9,769
    queries in one call compute 16,384 rows. Past `SPLIT_QUERIES` a batch
    instead takes the largest bucket that it fills while padding would waste
    more than `SPLIT_WASTE` of what is left, then decodes the rest the same
    way: 9,769 queries decode as 8,192 and 1,577 padded to 2,048, so 10,240
    rows. Every piece is a size the worker compiles anyway.
    """
    pieces = []
    left = count
    while left:
        padded = size(left)
        if padded <= SPLIT_QUERIES or padded - left <= left * SPLIT_WASTE:
            pieces.append(left)
            break
        pieces.append(padded // 2)
        left -= padded // 2
    return pieces


def split_queries(
    batches: Sequence[QueryBatch], size: Callable[[int], int]
) -> tuple[list[QueryBatch], list[tuple[int, int, int]]]:
    """Each batch as `query_pieces` splits it, and where each piece's predictions go.

    A piece's request is its position, as `merged` numbers its batches; its
    placement is its batch's request, its first query there and the batch's
    queries, so the pieces of a batch fill one array.
    """
    pieces: list[QueryBatch] = []
    placement = []
    for batch in batches:
        start = 0
        for count in query_pieces(len(batch.units), size):
            stop = start + count
            whole = start == 0 and stop == len(batch.units)
            pieces.append(
                protocol.QueryBatch(
                    len(pieces),
                    batch.table,
                    batch.epoch,
                    batch.features if whole else batch.features[start:stop],
                    batch.units if whole else batch.units[start:stop],
                )
            )
            placement.append((batch.request, start, len(batch.units)))
            start = stop
    return pieces, placement


def near_larger(queries: int) -> list[int]:
    """The padded query counts up to `GROWN_QUERIES` times `queries`, a bucket: steps merging more callers reach them."""
    return [queries << step for step in range(1, GROWN_QUERIES.bit_length())]


def _with_queries(call: Call, queries: int) -> tuple:
    """A paged decode call's arguments as shapes, its query rows and reads at `queries` an item."""

    def shape_of(value: Any, rows: int | None = None) -> jax.ShapeDtypeStruct:
        shape = list(np.shape(value))
        if rows is not None:
            shape[1] = rows
        return jax.ShapeDtypeStruct(
            tuple(shape),
            value.dtype,
            weak_type=bool(getattr(value, "weak_type", False)),
        )

    parameters, source = call.shared
    states, index, blocks, positions, counts, rows, *reads = call.items
    return (
        jax.tree.map(shape_of, parameters),
        shape_of(source),
        jax.tree.map(shape_of, states),
        shape_of(index),
        shape_of(blocks),
        shape_of(positions),
        shape_of(counts),
        shape_of(rows, queries),
        *[shape_of(read, queries) for read in reads],
    )


class Decodes:
    """How a worker decodes a step's queries, and scatters their predictions back."""

    def _decode(
        self: "Worker",
        batches: Sequence[QueryBatch],
        units: Sequence[Unit],
        transient: Transient,
    ) -> dict[int, np.ndarray]:
        """Decodes every query against the unit it reads.

        A step's requests to one table decode as one batch, so their queries
        embed in one call and queries reading the same context read it once.
        Cached contexts are read straight from their slots, every context of a
        model and task in one call. Contexts there was no room to keep are
        encoded and decoded as dense arrays.
        """
        joined, parts = merged(batches)
        pieces, placement = split_queries(joined, self._size)
        predictions = self._decode_batches(pieces, units, transient, placement)
        return {
            request: predictions[index][start:stop]
            for index, owned in enumerate(parts)
            if index in predictions
            for request, start, stop in owned
        }

    def _decode_fused(
        self: "Worker", batches: Sequence[QueryBatch], units: Sequence[Unit]
    ) -> dict[int, np.ndarray]:
        """Fits, encodes and decodes each fused unit and its queries in one call.

        A fused unit is a context one request brought: no page or slot keeps
        it, and its table keeps only the state, so a later step that reads the
        context again encodes it from pages. Units of one model, task and
        padded shapes share their calls.
        """
        if not batches:
            return {}
        joined, parts = merged(batches)
        work = []
        kept = []
        for batch, owned in zip(joined, parts, strict=True):
            unit = units[batch.units[0]]
            if any(units[index] is not unit for index in batch.units):
                raise RuntimeError(
                    f"queries of table {batch.table} read more than its fused unit"
                )
            with self.stats.timed("pad"):
                queries = _host_padded(batch.features, self._size(len(batch.units)))
            arguments = (*self._own_arguments(unit), queries)
            if self._defers(unit, arguments):
                continue
            self._fits_device(unit)
            work.append((batch.table, arguments))
            kept.append((batch, owned))
        predictions: dict[int, np.ndarray] = {}
        if self._stacks:
            outputs = self._call("fit_decode", work)
        else:
            outputs = self._alone("fit_decode", work)
        for (batch, owned), (state, values) in zip(kept, outputs, strict=True):
            unit = units[batch.units[0]]
            self._tables[batch.table].state = state
            self._warn_crowded(batch.table, state)
            self.stats.encoded += 1
            self.stats.transient += 1
            self.stats.rows_encoded += len(unit.rows)
            with self.stats.timed("fetch"):
                values = np.asarray(values)[: len(batch.units)]
            for request, start, stop in owned:
                predictions[request] = values[start:stop]
        return predictions

    def _decode_batches(
        self: "Worker",
        batches: Sequence[QueryBatch],
        units: Sequence[Unit],
        transient: Transient,
        placement: Sequence[tuple[int, int, int]] | None = None,
    ) -> dict[int, np.ndarray]:
        """Decodes batches whose requests are their positions; see `_decode`.

        A table that summarizes each context embeds its queries against their
        unit's own state, in the decode, so each decoding reads one unit.
        With `placement`, each batch's predictions go to the request, at the
        first query, of an array of the queries `split_queries` gives it.
        """
        fit_own = [self._tables[batch.table].own is not None for batch in batches]
        # A batch whose every query reads a cached context embeds in its
        # decode's program: its features go with the decode, so the step
        # makes no embed call and no device copy of its rows to gather.
        fused = [
            not mine and all(unit not in transient for unit in batch.units)
            for batch, mine in zip(batches, fit_own, strict=True)
        ]
        hosts, features = [], []
        for batch, inline in zip(batches, fused, strict=True):
            with self.stats.timed("pad"):
                hosts.append(_host_padded(batch.features, self._size(len(batch.units))))
            if inline:
                features.append(None)
                continue
            with self.stats.timed("dispatch"):
                features.append(jnp.asarray(hosts[-1]))
                self._settled(features[-1])
        embedded = iter(
            self._call(
                "embed",
                [
                    (batch.table, (self._tables[batch.table].state, rows))
                    for batch, rows, mine, inline in zip(
                        batches, features, fit_own, fused, strict=True
                    )
                    if not mine and not inline
                ],
            )
        )
        for batch in batches:
            self._warm_query_embeds(batch)
        dense: list[_Dense] = []
        paged: dict[tuple, list[Decoding]] = defaultdict(list)
        for batch, raw, host, mine, inline in zip(
            batches, features, hosts, fit_own, fused, strict=True
        ):
            rows = host if inline else raw if mine else next(embedded)
            groups: dict[int, list[int]] = defaultdict(list)
            for query, unit in enumerate(batch.units):
                groups[unit].append(query)
            table = self._tables[batch.table]
            layout = self._layouts[table.model, table.features]
            together: list[Group] = []
            for unit, queries in groups.items():
                if unit in transient:
                    dense.append(_Dense(batch, rows, unit, queries))
                else:
                    together.append((unit, queries))
            # Only states of one structure stack: a table with categorical
            # columns can keep more state than one without.
            structure = jax.tree.structure(table.state)
            key = (table.model, table.num_classes, layout, table.own, structure, inline)
            packs = (
                [[one] for one in together] if mine else self._packed(together, units)
            )
            for groups_of in packs:
                paged[key].append(self._decoding(batch, rows, groups_of, units))
        outputs = self._decode_dense(dense, units, transient)
        for (model, num_classes, layout, own, _, inline), decodings in paged.items():
            for same in alike(decodings, layout.slots_per_row, self._size):
                outputs += self._decode_paged(
                    model, num_classes, layout, same, own, embeds=inline
                )
        with self.stats.timed("fetch"):
            fetched = jax.device_get([values for _, values in outputs])
        predictions: dict[int, np.ndarray] = {}
        with self.stats.timed("scatter"):
            parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for ((batch, queries), _), values in zip(outputs, fetched, strict=True):
                values = np.asarray(values)
                request, offset, total = (
                    (batch.request, 0, len(batch.units))
                    if placement is None
                    else placement[batch.request]
                )
                if request not in predictions:
                    shape = (total, values.shape[-1])
                    predictions[request] = np.zeros(shape, dtype=values.dtype)
                at = np.asarray(queries, dtype=np.intp) + offset
                for start in range(0, len(at), SCATTER_ROWS):
                    stop = start + SCATTER_ROWS
                    parts.append(
                        (predictions[request], at[start:stop], values[start:stop])
                    )
            self._scatter(parts)
        return predictions

    def _scatter(
        self: "Worker", parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> None:
        """Copies each part's values to its rows of its output, side by side on the worker's cores."""

        def put(part: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
            output, rows, values = part
            output[rows] = values[: len(rows)]

        if len(parts) < 2 or self._scatter_threads < 2:
            for part in parts:
                put(part)
            return
        if self._scatterers is None:
            self._scatterers = ThreadPoolExecutor(
                self._scatter_threads, thread_name_prefix="gf-scatter"
            )
        for _ in self._scatterers.map(put, parts):
            pass

    def _decode_dense(
        self: "Worker", work: list[_Dense], units: Sequence[Unit], transient: Transient
    ) -> list[tuple[tuple[QueryBatch, list[int]], Any]]:
        """Decodes queries against units put off until now, a window of contexts at a time.

        Encoding each such unit makes a context, so encoding every one of a
        step first would hold them all at once.
        """
        outputs: list[tuple[tuple[QueryBatch, list[int]], Any]] = []
        window: list[_Dense] = []
        held = 0
        for item in work:
            window.append(item)
            # A put-off context holds its rows, gathered for its encode, beside
            # its cache: for a model whose row representations are wide, more.
            held += self._context_bytes(units[item.unit]) + self._input_bytes(
                units[item.unit]
            )
            if held >= self._window_bytes:
                outputs += self._decode_window(window, transient)
                window, held = [], 0
        return outputs + self._decode_window(window, transient)

    def _decode_window(
        self: "Worker", work: list[_Dense], transient: Transient
    ) -> list[tuple[tuple[QueryBatch, list[int]], Any]]:
        """Decodes some dense queries, encoding the units they read first."""
        encoded = self._encode_deferred([item.unit for item in work], transient)
        kinds: dict[tuple, list[_Dense]] = defaultdict(list)
        for item in work:
            table = self._tables[item.batch.table]
            kinds["dense", table.model, table.num_classes, table.features].append(item)
        outputs: list[tuple[tuple[QueryBatch, list[int]], Any]] = []
        for key, same in kinds.items():
            outputs += self._decode_parts(key, same, encoded)
        return outputs

    def _decode_parts(
        self: "Worker",
        key: tuple,
        work: list[_Dense],
        encoded: dict[int, tuple[Any, Any]],
    ) -> list[tuple[tuple[QueryBatch, list[int]], Any]]:
        """Decodes dense queries of one model, task and width against their contexts.

        A unit fit on its own rows comes with its state, which its queries'
        features embed against. A context too large to decode all its queries
        at once decodes them in parts, halved until they fit, down to the
        smallest chunk; later calls start at that size.
        """
        limit = self._chunks.parts(key)
        parts = [part for item in work for part in item.split(limit)]
        stages: dict[str, list[int]] = defaultdict(list)
        items: list[tuple[int, tuple]] = []
        for index, part in enumerate(parts):
            own, cache = encoded[part.unit]
            rows = query_rows(part.rows, part.queries, self._size(len(part.queries)))
            if own is None:
                stages["decode"].append(index)
                state = self._tables[part.batch.table].state
            else:
                stages["embed_decode"].append(index)
                state = own
            items.append((part.batch.table, (state, cache, rows)))
        done: dict[int, Any] = {}

        def keeper(indices: list[int]) -> Callable[[int, Any], None]:
            def keep(position: int, values: Any) -> None:
                done[indices[position]] = values

            return keep

        try:
            for stage, indices in stages.items():
                self._call(
                    stage, [items[index] for index in indices], consume=keeper(indices)
                )
        except jax.errors.JaxRuntimeError as error:
            rest = [part for index, part in enumerate(parts) if index not in done]
            most = max(len(part.queries) for part in rest)
            if not out_of_memory(error) or most <= self._size(1):
                raise
            # One context's queries alone ran out of memory: they go in halves,
            # from here on.
            self._chunks.split(key, max(self._size(1), most // 2))
            finished = [
                ((part.batch, part.queries), done[index])
                for index, part in enumerate(parts)
                if index in done
            ]
            return finished + self._decode_parts(key, rest, encoded)
        return [
            ((part.batch, part.queries), done[index])
            for index, part in enumerate(parts)
        ]

    def _packed(
        self: "Worker", groups: list[Group], units: Sequence[Unit]
    ) -> list[list[Group]]:
        """Query groups packed into decodings that read few slots each; see `packed`."""
        slots = {unit: len(self._contexts[units[unit].key].slots) for unit, _ in groups}
        return packed(groups, slots, self._window_bytes // self._slot_bytes)

    def _decoding(
        self: "Worker",
        batch: QueryBatch,
        rows: Any,
        groups: list[Group],
        units: Sequence[Unit],
    ) -> Decoding:
        """One decoding of a batch's queries against the units they read."""
        contexts = {unit: self._contexts[units[unit].key] for unit, _ in groups}
        padded = self._size(sum(len(members) for _, members in groups))
        return decoding_of(batch, rows, groups, contexts, padded)

    def _decode_paged(
        self: "Worker",
        model: int,
        num_classes: int,
        layout: CacheLayout,
        decodings: list[Decoding],
        own: StateLayout | None = None,
        embeds: bool = False,
        queries: int | None = None,
        wants: int | None = None,
    ) -> list[tuple[tuple[QueryBatch, list[int]], np.ndarray]]:
        """Decodes contexts read from their slots, as few calls as the budget allows.

        A context too large to decode all its queries at once decodes them in
        parts, halved until they fit, down to the smallest chunk; later calls
        start at that size. With `own`, how units keep their own states, each
        decoding reads one unit, whose state is read from its slots and embeds
        the queries' features. With `embeds`, the decodings carry their
        queries' features, which embed against their tables' states in the
        decode's program.

        A model the compile helper can rebuild never compiles a decode in a
        step once its kind has a program for any query count: a decode pads
        to a larger count with one, at most `GROWN_QUERIES` times its own, or
        else runs in parts of the largest smaller one, while the helper
        compiles the count it wanted, and, once a count first runs, those up
        to `GROWN_QUERIES` times larger (see `_compile_aside`). `queries`, if
        given, is the count the decodings pad to, and `wants` the count to
        compile for them.
        """
        key = ("paged", model, num_classes, layout, own, embeds)
        limit = self._chunks.parts(key)
        if limit is not None:
            decodings = [part for d in decodings for part in split(d, limit)]
        shape = paged_shape(decodings, layout, self._size)
        program = self._paged_key(
            model,
            num_classes,
            layout,
            shape.masked,
            shape.in_order,
            own is not None or embeds,
        )
        family = dataclasses.replace(shape, queries=0)

        def kind(padded: int) -> Hashable:
            return (program, family, padded)

        wanted = wants or shape.queries
        aside = self._aside(model)
        if queries is not None:
            shape = dataclasses.replace(shape, queries=queries)
        elif aside:
            ready = self.programs.sizes(kind(bucket_size(len(decodings), 1)))
            if ready and wanted not in ready:
                larger = [n for n in ready if wanted < n <= GROWN_QUERIES * wanted]
                smaller = [n for n in ready if n < wanted]
                if larger:
                    shape = dataclasses.replace(shape, queries=min(larger))
                elif smaller:
                    return self._decode_parts_of(
                        model,
                        num_classes,
                        layout,
                        decodings,
                        own,
                        embeds,
                        max(smaller),
                        wanted,
                    )
        signature = (key, shape)
        compiled = self._program(program)
        parameters = self._models[model].parameters
        results: list[tuple[tuple[QueryBatch, list[int]], np.ndarray]] = []
        # The latest call alone, so earlier chunks' arguments are freed.
        latest: list[Call] = []

        def prepare(first: int, stop: int, padded: int) -> Call:
            with self.stats.timed("pad"):
                tables, inputs = paged_inputs(
                    decodings[first:stop], layout, shape, padded
                )
            index, blocks, *rest = inputs
            if own is not None:
                states = self._own_states(decodings[first:stop], own, padded)
                index = np.arange(len(index), dtype=np.int32)
            else:
                states = self._stacked_states(tables, padded)
            # Rows come straight from the pool into the decode's cache, so it
            # holds neither the slots' bytes whole nor a copy of the pool.
            source, blocks = self._device.source(blocks)
            arguments = (states, index, blocks, *rest)
            function = compiled
            if aside:
                function = self.programs.loaded(kind(padded), shape.queries) or compiled
            latest[:] = [
                Call(function, (parameters, source), arguments, len(index), program)
            ]
            return latest[0]

        def taken(first: int, stop: int, values: Any) -> None:
            call = latest[0]
            if aside:
                self.programs.ran(kind(call.padded), shape.queries)
                if wanted != shape.queries:
                    self._compile_aside(model, kind(call.padded), wanted, call, 0)
                if self.programs.first(("grown", kind(call.padded), shape.queries)):
                    for more in near_larger(shape.queries):
                        self._compile_aside(model, kind(call.padded), more, call, 1)
            if self.programs.first(("decodes", signature)):
                self._warm_decodes(compiled, call.shared, call.items, shape.queries)
            for position, decoding in enumerate(decodings[first:stop]):
                with self.stats.timed("unstack"):
                    item = _item(values, np.int32(position))
                # Its copy to the host runs beside the step's later calls.
                for leaf in jax.tree.leaves(item):
                    leaf.copy_to_host_async()
                results.append(((decoding.batch, decoding.queries), item))

        # A context and the working copy beside it, and the queries' rows; and
        # a unit's own state, as its slots' bytes and unpacked.
        width = decodings[0].rows.shape[-1]
        estimate = 2 * shape.context_rows * layout.row_bytes + shape.queries * width * 4
        if own is not None:
            estimate += 2 * own.slots * self._slot_bytes
        try:
            self._chunks.run(
                "decode",
                signature,
                len(decodings),
                self._call_bytes,
                lambda: estimate,
                prepare,
                taken,
                self._compile_on_side if self._on_side else None,
            )
        except jax.errors.JaxRuntimeError as error:
            # One decoding alone ran out of memory: its queries go in halves,
            # from here on.
            most = len(decodings[len(results)].queries)
            if not out_of_memory(error) or most <= self._size(1):
                raise
            self._chunks.split(key, max(self._size(1), most // 2))
            rest = decodings[len(results) :]
            return results + self._decode_paged(
                model, num_classes, layout, rest, own, embeds
            )
        return results

    def _decode_parts_of(
        self: "Worker",
        model: int,
        num_classes: int,
        layout: CacheLayout,
        decodings: list[Decoding],
        own: StateLayout | None,
        embeds: bool,
        queries: int,
        wants: int,
    ) -> list[tuple[tuple[QueryBatch, list[int]], np.ndarray]]:
        """Decodes in parts of at most `queries` queries, padded to it, a part of each decoding a round.

        Each call then stacks as many decodings as the whole would have, at a
        count with a program; `wants`, the count the whole wanted, compiles
        aside.
        """
        parts = [split(decoding, queries) for decoding in decodings]
        results: list[tuple[tuple[QueryBatch, list[int]], np.ndarray]] = []
        for round_ in range(max(len(each) for each in parts)):
            chunk = [each[round_] for each in parts if round_ < len(each)]
            results += self._decode_paged(
                model, num_classes, layout, chunk, own, embeds, queries, wants
            )
        return results

    def _aside(self: "Worker", model: int) -> bool:
        """Whether a model's decodes compile in the compile helper: on an accelerator, compiling ahead, for a catalog model."""
        if not self._on_side or os.environ.get(HELPER_ENV) == "0":
            return False
        return compiles_aside(self._specs.get(model))

    def _compile_aside(
        self: "Worker", model: int, kind: Hashable, size: int, call: Call, priority: int
    ) -> None:
        """Asks the compile helper for a paged decode's program at `size` queries, once per kind and count.

        The program serves the kind's calls at that count once it arrives,
        unless programs dropped meanwhile. A count the helper could not
        compile compiles in the step that next needs it.
        """
        if self.programs.loaded(kind, size) is not None:
            return
        if not self.programs.first(("aside", kind, size)):
            return
        shapes = _with_queries(call, size)
        generation = self.programs.generation

        def done(executable: Any) -> None:
            if executable is not None:
                self.programs.offer(kind, size, executable, shapes, generation)

        request = {"spec": self._specs[model], "key": call.program, "shapes": shapes}
        self._compile_helper().submit((kind, size), priority, request, done)

    def _compile_helper(self: "Worker") -> CompileHelper:
        """The worker's compile helper, made when first asked for."""
        helper = self.__dict__.get("_helper")
        if helper is None:
            helper = CompileHelper(
                {
                    "settings": self._settings,
                    "bucket_minimum": self._minimum,
                    "slot_bytes": self._slot_bytes,
                    "compile_cache": self._compile_cache,
                }
            )
            self.__dict__["_helper"] = helper
        return helper

    def _stacked_states(self: "Worker", tables: list[int], padded: int) -> Any:
        """The tables' states stacked over a call's `padded` items.

        A call of one table reuses the stack it made last, while the table
        keeps that state, so a warm step makes no stacking dispatch. The
        worker holds at most `HELD_STATES` such stacks, of tables it still
        serves: each is a copy of a state.
        """
        states = [self._tables[table].state for table in tables]
        held = self.__dict__.setdefault("_held_states", OrderedDict())
        key = (tables[0], padded) if len(tables) == 1 else None
        if key is not None and key in held and held[key][0] is states[0]:
            held.move_to_end(key)
            return held[key][1]
        with self.stats.timed("stack"):
            stacked = _stacked(padded_items(states, padded))
        if key is not None:
            held[key] = (states[0], stacked)
            for gone in [k for k in held if k[0] not in self._tables]:
                del held[gone]
            while len(held) > HELD_STATES:
                held.popitem(last=False)
        return stacked

    def _own_states(
        self: "Worker", decodings: list[Decoding], own: StateLayout, count: int
    ) -> Any:
        """The own states of the units `decodings` read, one each, from their slots, padded to `count`."""
        contexts = padded_items([decoding.contexts[0] for decoding in decodings], count)
        slots = []
        for context in contexts:
            start = context.layout.slots(self._size(context.rows))
            slots += context.slots[start : start + own.slots]
        with self.stats.timed("fetch"):
            data = self._device.read(slots).reshape(len(contexts), own.slots, -1)
        return unpack_states(own, data)
