# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pages and contexts: row representations and encoded contexts in the tiers' slots, and the decodes that read them.

The node decides where each entry lives; the worker packs pages into host
slots and contexts into device slots as the node assigned them, moves them
between tiers, and decodes queries straight from the slots, or encodes a
context there was no room to keep only when its decode needs it.
"""

import dataclasses
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast import protocol
from generalforecast._core import CacheKey
from generalforecast.constants import SCATTER_ROWS
from generalforecast.runtime.budget import Call, out_of_memory
from generalforecast.runtime.decoding import (
    Context,
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
    HostPool,
    StateLayout,
    pack_state,
    unpack_states,
)
from generalforecast.runtime.shapes import padded as padded_items
from generalforecast.worker.calls import _host_padded, _item, _stacked
from generalforecast.worker.records import Fit, Movement, Page, QueryBatch, Unit

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


@dataclass
class _Deferred:
    """A unit there is no room to keep, encoded only when its decode needs it.

    Attributes:
        table: The unit's table.
        arguments: Its encode's arguments after the model parameters.
        rows: Its real rows.
        stage: `encode`, or `fit_encode` for a unit fit on its own rows.
        encoded: Whether it was encoded once already, for the counts.
    """

    table: int
    arguments: tuple
    rows: int
    stage: str = "encode"
    encoded: bool = False


# Units without slots, put off until decoded.
Transient = dict[int, _Deferred]


# A page's row representations, on the device or the host, and its targets.
Representations = tuple[Any, np.ndarray]


@dataclass
class _Page:
    """Where a page's representations are in host slots, and its targets."""

    slots: list[int]
    shape: tuple[int, ...]
    dtype: np.dtype
    targets: np.ndarray


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


class Contexts:
    """How a worker fits, embeds, encodes, keeps and decodes tabular contexts."""

    def apply(self: "Worker", movement: Movement) -> None:
        """Applies one copy or drop from the node.

        A drop may name an entry a failed step never stored, and a copy one
        the node dropped since it planned the copy, when the step that was to
        encode it failed meanwhile; both are skipped.
        """
        # Pages are only ever dropped.
        if movement.key.kind == "rows":
            self._pages.pop(movement.key, None)
            return
        if movement.target is None:
            self._fixed.pop(movement.key, None)
            if self._contexts.pop(movement.key, None) is not None:
                self.stats.dropped += 1
            return
        context = self._contexts.get(movement.key)
        if context is None:
            return
        slots = list(movement.slots)
        if movement.source == "device":
            data = self._device.read_host(context.slots)
        else:
            data = self._below(movement.source).read(context.slots)
        if movement.target == "device":
            self._device.write(slots, data)
            self._settled(self._device.data)
            self.stats.promoted += 1
        else:
            self._below(movement.target).write(slots, data)
            if movement.target == "host":
                self.stats.demoted += 1
            else:
                self.stats.spilled += 1
        context.slots = slots

    def _below(self: "Worker", tier: protocol.Tier) -> HostPool:
        """The host pool, or the disk pool."""
        if tier == "host":
            return self._host
        if self._disk is None:
            raise RuntimeError(
                "the node moved a context to disk, but the worker has no disk tier"
            )
        return self._disk

    def _keep(self: "Worker", page: Page, representations: Representations) -> None:
        """Caches a computed page in the host slots the node kept for it, if any."""
        if page.action != "encode":
            return
        values, targets = representations
        # Targets can be views of the whole frame the step came in; a copy keeps
        # the cached page from holding that frame, and its shared memory, alive.
        targets = np.array(targets, copy=True)
        data = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
        capacity = len(page.slots) * self._slot_bytes
        if data.size > capacity:
            raise RuntimeError(
                f"a page takes {data.size} bytes, but the node kept {capacity}"
            )
        # Only the page's bytes: a page may fill a fraction of a large slot.
        with self.stats.timed("store"):
            self._host.write_bytes(list(page.slots), data)
        self._pages[page.key] = _Page(
            list(page.slots), values.shape, values.dtype, targets
        )

    def _cached(self: "Worker", key: CacheKey) -> Representations:
        """A cached page's representations and targets, read from its host slots."""
        page = self._pages[key]
        size = int(np.prod(page.shape)) * page.dtype.itemsize
        data = self._host.read_bytes(page.slots, size)
        return data.view(page.dtype).reshape(page.shape), page.targets

    def _fit(
        self: "Worker", fits: Sequence[Fit], pages: Sequence[Page]
    ) -> dict[int, Representations]:
        """Fits tables, and splits the rows of fits that fill pages into them."""
        work = [
            (
                fit.table,
                (
                    *self._padded(fit.features, fit.targets),
                    self._tables[fit.table].categorical,
                ),
            )
            for fit in fits
        ]
        # A fit filling pages embeds its rows in the call that fits its state;
        # one past the rows a call takes computes the state alone, and each
        # page is embedded once a plan reads it.
        fused = [index for index, fit in enumerate(fits) if fit.pages]
        alone = [index for index, fit in enumerate(fits) if not fit.pages]
        for index, state in zip(
            alone, self._call("state", [work[index] for index in alone]), strict=True
        ):
            self._tables[fits[index].table].state = state
            self._warn_crowded(fits[index].table, state)
            self._warm(fits[index])
        filled: dict[int, Representations] = {}
        for index, (state, representations) in zip(
            fused, self._call("fit", [work[index] for index in fused]), strict=True
        ):
            fit = fits[index]
            self._tables[fit.table].state = state
            self._warn_crowded(fit.table, state)
            self._warm(fit)
            # Split on the host: slicing on the device compiles once per page
            # length, and pages change length with every write.
            with self.stats.timed("fetch"):
                on_host = np.asarray(representations)
            offset = 0
            for index in fit.pages:
                count = len(pages[index].rows)
                targets = np.asarray(fit.targets[offset : offset + count])
                filled[index] = (on_host[offset : offset + count], targets)
                self._keep(pages[index], filled[index])
                offset += count
        return filled

    def _warn_crowded(self: "Worker", table: int, state: Any) -> None:
        """Logs once per table that categorical columns held more categories than kept.

        Args:
            table: The table a state was fit for, on the whole table or a unit.
            state: The fitted state.
        """
        entry = self._tables[table]
        count = self._models[entry.model].functions.crowded_categories
        if entry.crowded or entry.categorical is None or count is None:
            return
        crowded = count(state)
        if crowded:
            entry.crowded = True
            columns = "column holds" if crowded == 1 else "columns hold"
            logger.warning(
                "table %d: %d categorical %s more categories than the model "
                "keeps, so their ranks are interpolated",
                table,
                crowded,
                columns,
            )

    def _embed_pages(
        self: "Worker", pages: Sequence[Page], filled: dict[int, Representations]
    ) -> dict[int, Representations]:
        """Reads cached pages and embeds the rest, one vectorized call per signature.

        A page with sources copies their rows, representations and targets,
        and embeds only its own. Those pad to their own bucket, not the page
        size: a page written a row at a time then embeds a few rows, or none.
        """
        found: dict[int, Representations] = {}
        work: list[tuple[int, tuple]] = []
        embedding: list[tuple[int, list[int]]] = []
        partial: dict[int, Representations] = {}
        for index, page in enumerate(pages):
            if index in filled:
                continue
            if page.action == "reuse":
                found[index] = self._cached(page.key)
                continue
            if page.features is None or page.targets is None:
                raise RuntimeError(
                    f"a page of table {page.table} to embed came without its rows"
                )
            missing = list(range(len(page.rows)))
            if page.copies:
                partial[index] = self._copied(page)
                copied = {to for _, positions in page.copies for _, to in positions}
                missing = [at for at in missing if at not in copied]
            if not missing:
                continue
            at_least = 1 if page.copies else self._tables[page.table].page_size
            features, targets, _ = self._padded(page.features, page.targets, at_least)
            work.append(
                (page.table, (self._tables[page.table].state, features, targets))
            )
            embedding.append((index, missing))
        for (index, missing), representations in zip(
            embedding, self._call("embed", work), strict=True
        ):
            page = pages[index]
            with self.stats.timed("fetch"):
                embedded = np.asarray(representations)[: len(missing)]
            if index in partial:
                values, targets = partial[index]
                values[missing] = embedded
                targets[missing] = np.asarray(page.targets)
            else:
                partial[index] = (embedded, np.asarray(page.targets))
            self.stats.rows_embedded += len(missing)
        for index, computed in partial.items():
            self._keep(pages[index], computed)
            found[index] = computed
        return found

    def _copied(self: "Worker", page: Page) -> Representations:
        """A page's representations and targets with its sources' rows copied in."""
        values: np.ndarray | None = None
        targets: np.ndarray | None = None
        for source, positions in page.copies:
            held, labels = self._cached(source)
            if values is None or targets is None:
                values = np.empty((len(page.rows), *held.shape[1:]), held.dtype)
                targets = np.empty((len(page.rows), *labels.shape[1:]), labels.dtype)
            at, to = (list(part) for part in zip(*positions, strict=True))
            values[to] = held[at]
            targets[to] = labels[at]
            self.stats.rows_copied += len(positions)
        if values is None or targets is None:
            raise RuntimeError(f"a page of table {page.table} copies from no source")
        return values, targets

    @staticmethod
    def _tables_rows(
        pages: Sequence[Page], computed: dict[int, Representations]
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Each table's page rows sorted, with their representations and targets.

        Built once per step, so units gather from them without joining pages
        again. They join on the host: on the device, every new number of rows
        would compile its own join and gather, and writes change it each time.
        """
        by_table: dict[int, list[int]] = defaultdict(list)
        for index, page in enumerate(pages):
            by_table[page.table].append(index)
        joined = {}
        for table, indices in by_table.items():
            rows = np.sort(
                np.concatenate([np.asarray(pages[index].rows) for index in indices])
            )
            # Each page's rows go straight to their place, so the step holds
            # one copy of the representations beside the pages'.
            first_values, first_targets = (
                np.asarray(part) for part in computed[indices[0]]
            )
            values = np.empty((len(rows), *first_values.shape[1:]), first_values.dtype)
            targets = np.empty(
                (len(rows), *first_targets.shape[1:]), first_targets.dtype
            )
            for index in indices:
                at = np.searchsorted(rows, np.asarray(pages[index].rows))
                values[at], targets[at] = computed[index]
            joined[table] = (rows, values, targets)
        return joined

    def _encode(
        self: "Worker",
        units: Sequence[Unit],
        pages: Sequence[Page],
        computed: dict[int, Representations],
    ) -> Transient:
        """Encodes the units that are not cached; returns those there was no room to keep.

        A unit with no room to keep is encoded only by the decode that reads
        it, so a step never holds every such context at once.
        """
        transient: Transient = {}
        waiting = [
            index
            for index, unit in enumerate(units)
            if unit.action not in ("reuse", "fuse")
        ]
        self.stats.reused += sum(unit.action == "reuse" for unit in units)
        self.stats.rows_reused += sum(
            len(unit.rows) for unit in units if unit.action == "reuse"
        )
        joined = self._tables_rows(pages, computed) if waiting else {}
        work: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
        group: dict[str, list[int]] = defaultdict(list)
        for index in waiting:
            unit = units[index]
            self._fits_device(unit)
            if self._tables[unit.table].own is not None:
                stage, arguments = "fit_encode", self._own_arguments(unit)
            else:
                stage, arguments = "encode", self._gathered(unit, joined)
            if unit.action == "transient":
                transient[index] = _Deferred(
                    unit.table, arguments, len(unit.rows), stage
                )
                continue
            work[stage].append((unit.table, arguments))
            group[stage].append(index)

        # Each context goes to its slots as its call finishes, so a step never
        # holds all it encodes beside the pool.
        def keeper(stage: str) -> Callable[[int, Any], None]:
            def keep(position: int, outputs: Any) -> None:
                unit = units[group[stage][position]]
                state, cache = outputs if stage == "fit_encode" else (None, outputs)
                if state is not None:
                    self._warn_crowded(unit.table, state)
                self.stats.encoded += 1
                self.stats.rows_encoded += len(unit.rows)
                table = self._tables[unit.table]
                layout = self._layouts[table.model, table.features]
                self._store(
                    unit.key,
                    unit.slots,
                    layout,
                    len(unit.rows),
                    cache,
                    table.own,
                    state,
                )

            return keep

        for stage, items in work.items():
            self._call(stage, items, consume=keeper(stage))
        return transient

    def _gathered(
        self: "Worker",
        unit: Unit,
        joined: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple:
        """A unit's encode arguments, its rows' representations gathered from pages."""
        rows, values, targets = joined[unit.table]
        wanted = np.asarray(unit.rows)
        positions = np.searchsorted(rows, wanted)
        if np.any(positions >= len(rows)) or np.any(rows[positions] != wanted):
            raise RuntimeError(
                f"a unit of table {unit.table} reads rows that are in none "
                "of the step's pages"
            )
        size = self._size(len(unit.rows))
        with self.stats.timed("pad"):
            return (
                self._tables[unit.table].state,
                _host_padded(values[positions], size),
                _host_padded(targets[positions], size),
                len(unit.rows),
            )

    def _own_arguments(self: "Worker", unit: Unit) -> tuple:
        """A unit's fit arguments, from the rows it came with."""
        if unit.features is None or unit.targets is None:
            raise RuntimeError(
                f"a unit of table {unit.table} to fit came without its rows"
            )
        size = self._size(len(unit.rows))
        with self.stats.timed("pad"):
            return (
                _host_padded(unit.features, size),
                _host_padded(unit.targets, size),
                len(unit.rows),
                self._tables[unit.table].categorical,
            )

    def _store(
        self: "Worker",
        key: CacheKey,
        slots: Sequence[int],
        layout: CacheLayout,
        rows: int,
        cache: Any,
        own: StateLayout | None = None,
        state: Any = None,
    ) -> None:
        """Packs an encoded context of `rows` layout rows, and any own state after it, into the device slots the node assigned it."""
        slots = list(slots)
        packed = layout.packed_slots(cache)
        own_data = None if own is None else pack_state(own, state)
        used = packed + (0 if own_data is None else own_data.shape[0])
        # The node keeps room for the table's state, with or without
        # categorical columns; the state may take fewer whole slots.
        written = slots if own is None else slots[:used]
        if used != len(written):
            raise RuntimeError(
                f"a context packs into {used} slots, but the node gave {len(slots)}"
            )
        with self.stats.timed("store"):
            self._device.write_cache(slots[:packed], layout, cache)
            if own_data is not None:
                self._device.write(slots[packed:used], own_data)
            self._settled(self._device.data)
        self._contexts[key] = Context(layout, rows, slots)

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
        predictions = self._decode_batches(joined, units, transient)
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
    ) -> dict[int, np.ndarray]:
        """Decodes batches whose requests are their positions; see `_decode`.

        A table that summarizes each context embeds its queries against their
        unit's own state, in the decode, so each decoding reads one unit.
        """
        features = []
        for batch in batches:
            with self.stats.timed("pad"):
                padded = _host_padded(batch.features, self._size(len(batch.units)))
            with self.stats.timed("dispatch"):
                features.append(jnp.asarray(padded))
                self._settled(features[-1])
        fit_own = [self._tables[batch.table].own is not None for batch in batches]
        embedded = iter(
            self._call(
                "embed",
                [
                    (batch.table, (self._tables[batch.table].state, rows))
                    for batch, rows, mine in zip(
                        batches, features, fit_own, strict=True
                    )
                    if not mine
                ],
            )
        )
        for batch in batches:
            self._warm_query_embeds(batch)
        dense: list[_Dense] = []
        paged: dict[tuple, list[Decoding]] = defaultdict(list)
        for batch, raw, mine in zip(batches, features, fit_own, strict=True):
            rows = raw if mine else next(embedded)
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
            key = (table.model, table.num_classes, layout, table.own, structure)
            packs = (
                [[one] for one in together] if mine else self._packed(together, units)
            )
            for groups_of in packs:
                paged[key].append(self._decoding(batch, rows, groups_of, units))
        outputs = self._decode_dense(dense, units, transient)
        for (model, num_classes, layout, own, _), decodings in paged.items():
            for same in alike(decodings, layout.slots_per_row, self._size):
                outputs += self._decode_paged(model, num_classes, layout, same, own)
        with self.stats.timed("fetch"):
            fetched = jax.device_get([values for _, values in outputs])
        predictions: dict[int, np.ndarray] = {}
        with self.stats.timed("scatter"):
            parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for ((batch, queries), _), values in zip(outputs, fetched, strict=True):
                values = np.asarray(values)
                if batch.request not in predictions:
                    shape = (len(batch.units), values.shape[-1])
                    predictions[batch.request] = np.zeros(shape, dtype=values.dtype)
                at = np.asarray(queries, dtype=np.intp)
                for start in range(0, len(at), SCATTER_ROWS):
                    stop = start + SCATTER_ROWS
                    parts.append(
                        (predictions[batch.request], at[start:stop], values[start:stop])
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
            held += self._context_bytes(units[item.unit])
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

    def _encode_deferred(
        self: "Worker", indices: list[int], transient: Transient
    ) -> dict[int, tuple[Any, Any]]:
        """Encodes the put-off units among `indices`, once each, in shared calls.

        Returns:
            Each unit's own state, or None for one that gathered its rows from
            pages, and its cache.
        """
        wanted = list(dict.fromkeys(indices))
        deferred = [transient[unit] for unit in wanted]
        outputs = self._staged([(d.stage, d.table, d.arguments) for d in deferred])
        encoded: dict[int, tuple[Any, Any]] = {}
        for unit, put_off, output in zip(wanted, deferred, outputs, strict=True):
            state, cache = output if put_off.stage == "fit_encode" else (None, output)
            if state is not None:
                self._warn_crowded(put_off.table, state)
            if not put_off.encoded:
                put_off.encoded = True
                self.stats.encoded += 1
                self.stats.transient += 1
                self.stats.rows_encoded += put_off.rows
            encoded[unit] = (state, cache)
        return encoded

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
    ) -> list[tuple[tuple[QueryBatch, list[int]], np.ndarray]]:
        """Decodes contexts read from their slots, as few calls as the budget allows.

        A context too large to decode all its queries at once decodes them in
        parts, halved until they fit, down to the smallest chunk; later calls
        start at that size. With `own`, how units keep their own states, each
        decoding reads one unit, whose state is read from its slots and embeds
        the queries' features.
        """
        key = ("paged", model, num_classes, layout, own)
        limit = self._chunks.parts(key)
        if limit is not None:
            decodings = [part for d in decodings for part in split(d, limit)]
        shape = paged_shape(decodings, layout, self._size)
        signature = (key, shape)
        program = self._paged_key(
            model, num_classes, layout, shape.masked, shape.in_order, own is not None
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
                with self.stats.timed("stack"):
                    states = _stacked(
                        padded_items(
                            [self._tables[table].state for table in tables], padded
                        )
                    )
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
            return results + self._decode_paged(model, num_classes, layout, rest, own)
        return results

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
