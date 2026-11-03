# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decodes: queries against contexts read straight from their slots, put off until now, or fit in the same call."""

import dataclasses
import logging
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast import protocol
from generalforecast.constants import (
    FUSED_QUERIES,
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
from generalforecast.worker.encoding import Transient, _Gather
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


def _query_index(queries: list[int], count: int) -> np.ndarray:
    """The positions of a context's queries in their batch, padded to `count` by repeating the last."""
    index = np.full(count, queries[-1], dtype=np.int32)
    index[: len(queries)] = queries
    return index


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
        """Decodes some dense queries, encoding the units they read first.

        Contexts that gather their rows from the step's rows on the device,
        each read by a few queries, encode and decode in one call for many of
        them, which never keeps their caches outside it; the rest encode
        first, then decode, a context's many queries in parts if they do not
        fit at once.
        """
        fused: dict[tuple, list[_Dense]] = defaultdict(list)
        rest: list[_Dense] = []
        for item in work:
            deferred = transient[item.unit]
            arguments = deferred.arguments
            if (
                deferred.stage == "encode"
                and isinstance(arguments[1], _Gather)
                and arguments[1].on_device
                and len(item.queries) <= FUSED_QUERIES
            ):
                # Alike calls: one table, one batch's query rows, one padded
                # size of context, and queries per context within one count.
                count = self._size(len(item.queries))
                size = len(arguments[1].index)
                fused[(item.batch.table, id(item.batch), size, count)].append(item)
            else:
                rest.append(item)
        outputs: list[tuple[tuple[QueryBatch, list[int]], Any]] = []
        for (_, _, _, count), same in fused.items():
            outputs += self._encode_decode(same, transient, count)
        if not rest:
            return outputs
        work = rest
        encoded = self._encode_deferred([item.unit for item in work], transient)
        kinds: dict[tuple, list[_Dense]] = defaultdict(list)
        for item in work:
            table = self._tables[item.batch.table]
            kinds["dense", table.model, table.num_classes, table.features].append(item)
        for key, same in kinds.items():
            outputs += self._decode_parts(key, same, encoded)
        return outputs

    def _encode_decode(
        self: "Worker", work: list[_Dense], transient: Transient, count: int
    ) -> list[tuple[tuple[QueryBatch, list[int]], Any]]:
        """Encodes put-off contexts of one table and decodes their queries, many to a call.

        Each context's rows are gathered from the step's rows on the device,
        encoded, and read by its queries in the same program, vectorized over
        the contexts of a call, so a step of many such contexts takes a few
        calls rather than an encode, an unstack and a decode for each. Calls
        are sized to the worker's call budget as any other's.
        """
        first = transient[work[0].unit]
        table_id = first.table
        table = self._tables[table_id]
        state, gather, _ = first.arguments
        size = len(gather.index)
        index = np.stack([transient[item.unit].arguments[1].index for item in work])
        lengths = np.asarray(
            [transient[item.unit].arguments[2] for item in work], np.int32
        )
        queries = np.stack([_query_index(item.queries, count) for item in work])
        rows = work[0].rows
        model, num_classes = table.model, table.num_classes
        compiled = self._compiled_encode_decode(model, num_classes)
        parameters = self._models[model].parameters
        shared = (parameters, state, gather.values, gather.targets, rows)
        key = ("encode_decode", model, num_classes, size, count, rows.shape[1:])
        # Before a call is measured: a context's cache and its gathered rows.
        profile = self.profile(model, table.features)
        per_context = size * (profile.cache_bytes_per_row + profile.row_bytes)
        results: dict[int, np.ndarray] = {}

        def prepare(start: int, stop: int, padded: int) -> Call:
            # Padding repeats the last context, as `padded_items` pads items.
            def taken(values: np.ndarray) -> np.ndarray:
                part = values[start:stop]
                extra = padded - len(part)
                return np.concatenate([part, np.repeat(part[-1:], extra, axis=0)])

            with self.stats.timed("stack"):
                stacked = (taken(index), taken(lengths), taken(queries))
            return Call(compiled, shared, stacked, padded, None)

        def consume(start: int, stop: int, outputs: Any) -> None:
            with self.stats.timed("fetch"):
                values = np.asarray(outputs)
            for position in range(start, stop):
                results[position] = values[position - start]

        self._chunks.run(
            "encode",
            key,
            len(work),
            self._call_bytes,
            lambda: per_context,
            prepare,
            consume,
            None,
            None,
        )
        for item in work:
            deferred = transient[item.unit]
            if not deferred.encoded:
                deferred.encoded = True
                self.stats.encoded += 1
                self.stats.transient += 1
                self.stats.rows_encoded += deferred.rows
        return [
            ((item.batch, item.queries), results[position][: len(item.queries)])
            for position, item in enumerate(work)
        ]

    def _compiled_encode_decode(
        self: "Worker", model: int, num_classes: int
    ) -> Callable[..., Any]:
        """A context's gather, encode and decode of its queries, vectorized over contexts."""

        def build() -> Callable[..., Any]:
            encode = self._stage("encode", model, num_classes)
            decode = self._stage("decode", model, num_classes)

            def one(parameters, state, values, targets, rows, index, length, queries):
                context = jnp.take(values, index, axis=0, mode="fill", fill_value=0)
                labels = jnp.take(targets, index, axis=0, mode="fill", fill_value=0)
                cache = encode(parameters, state, context, labels, length)
                return decode(parameters, state, cache, rows[queries])

            return jax.vmap(one, in_axes=(None,) * 5 + (0,) * 3)

        return self.programs.compiled(("encode_decode", model, num_classes), build)

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
    ) -> list[tuple[tuple[QueryBatch, list[int]], np.ndarray]]:
        """Decodes contexts read from their slots, as few calls as the budget allows.

        A context too large to decode all its queries at once decodes them in
        parts, halved until they fit, down to the smallest chunk; later calls
        start at that size. With `own`, how units keep their own states, each
        decoding reads one unit, whose state is read from its slots and embeds
        the queries' features. With `embeds`, the decodings carry their
        queries' features, which embed against their tables' states in the
        decode's program.
        """
        key = ("paged", model, num_classes, layout, own, embeds)
        limit = self._chunks.parts(key)
        if limit is not None:
            decodings = [part for d in decodings for part in split(d, limit)]
        shape = paged_shape(decodings, layout, self._size)
        signature = (key, shape)
        program = self._paged_key(
            model,
            num_classes,
            layout,
            shape.masked,
            shape.in_order,
            own is not None or embeds,
        )
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
            latest[:] = [
                Call(compiled, (parameters, source), arguments, len(index), program)
            ]
            return latest[0]

        def taken(first: int, stop: int, values: Any) -> None:
            if self.programs.first(("decodes", signature)):
                call = latest[0]
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
