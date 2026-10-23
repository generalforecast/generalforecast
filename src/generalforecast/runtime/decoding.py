# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast import _core, protocol
from generalforecast._core import bucket_size
from generalforecast.models.contract import TabularFunctions
from generalforecast.runtime.device import load_cache
from generalforecast.runtime.paging import CacheLayout, read_context
from generalforecast.runtime.shapes import pad_rows, padded

QueryBatch = _core.QueryBatch | protocol.QueryBatch
# The unit a group of queries reads, and those queries.
Group = tuple[int, list[int]]


@dataclass
class Context:
    """Where an encoded context's slots are, and its real rows.

    A unit fit on its own rows keeps its state in the slots after its rows'.
    """

    layout: CacheLayout
    rows: int
    slots: list[int]


@dataclass
class Decoding:
    """Queries of one batch, and the contexts they read, each read once.

    Attributes:
        reads: For each query, the index of the context it reads, or None
            when the decoding reads one context.
    """

    batch: Any
    queries: list[int]
    contexts: list[Context]
    rows: Any
    reads: list[int] | None = None


@dataclass(frozen=True)
class PagedShape:
    """What a paged decode compiles for, padded so that alike decodings share it.

    Attributes:
        context_rows: Context rows, padded to a bucket.
        queries: Queries, padded to a bucket.
        blocks: Blocks of rows each decoding reads.
        masked: Whether a decoding reads several contexts, one per query.
        in_order: Whether every decoding reads one context, a block at a time.
    """

    context_rows: int
    queries: int
    blocks: int
    masked: bool
    in_order: bool


def merged(
    batches: Sequence[QueryBatch],
) -> tuple[list[protocol.QueryBatch], list[list[tuple[int, int, int]]]]:
    """Requests to one table and epoch joined into one batch each.

    Args:
        batches: A step's query batches.

    Returns:
        The joined batches, each with its position as its request, and for
        each, the requests it holds with their first and last query, exclusive.
    """
    joined: dict[tuple, list[QueryBatch]] = defaultdict(list)
    for batch in batches:
        joined[batch.table, batch.epoch].append(batch)
    merged, parts = [], []
    for index, ((table, epoch), members) in enumerate(joined.items()):
        features = (
            np.asarray(members[0].features)
            if len(members) == 1
            else np.concatenate([np.asarray(member.features) for member in members])
        )
        reads = [unit for member in members for unit in member.units]
        merged.append(protocol.QueryBatch(index, table, epoch, features, reads))
        owned, start = [], 0
        for member in members:
            owned.append((member.request, start, start + len(member.units)))
            start += len(member.units)
        parts.append(owned)
    return merged, parts


def packed(
    groups: list[Group], slots: Mapping[int, int], window: int
) -> list[list[Group]]:
    """Query groups packed into decodings that read few slots each.

    Groups whose units differ, as each query's own context does, would make
    one decoding read them all, so a decoding takes groups only while the
    units it reads hold at most twice the slots of its largest group's, and
    no more than `window`.

    Args:
        groups: Query groups, by the unit they read.
        slots: Slots each unit's context holds.
        window: Slots of contexts a decode may gather.

    Returns:
        The groups of each decoding.
    """
    largest = max((slots[unit] for unit, _ in groups), default=0)
    most = max(largest, min(2 * largest, window))
    packs: list[list[Group]] = []
    held = 0
    for unit, members in groups:
        if packs and held + slots[unit] <= most:
            packs[-1].append((unit, members))
        else:
            packs.append([(unit, members)])
            held = 0
        held += slots[unit]
    return packs


def decoding_of(
    batch: QueryBatch,
    rows: Any,
    groups: list[Group],
    contexts: Mapping[int, Context],
    padded: int,
) -> Decoding:
    """One decoding of a batch's queries against every unit any of them reads.

    Queries that read different units still decode together, each reading
    only its own unit, so one call serves many small contexts.

    Args:
        batch: The queries' batch.
        rows: The batch's embedded query rows.
        groups: Query groups, by the unit they read.
        contexts: Each unit's context.
        padded: Queries the decoding pads to.

    Returns:
        The decoding.
    """
    queries = [query for _, members in groups for query in members]
    reads = None
    if len(groups) > 1:
        reads = [index for index, (_, members) in enumerate(groups) for _ in members]
    held = [contexts[unit] for unit, _ in groups]
    return Decoding(batch, queries, held, query_rows(rows, queries, padded), reads)


def alike(
    decodings: list[Decoding], slots_per_row: int, size: Callable[[int], int]
) -> list[list[Decoding]]:
    """Decodings grouped by their padded size, in slots and in queries.

    A call pads its decodings to its largest, so decodings of one padded size
    call together.

    Args:
        decodings: Decodings of one model, task and layout.
        slots_per_row: Slots a block of their contexts' rows takes.
        size: Pads a number of queries.

    Returns:
        The decodings of each size.
    """
    sizes: dict[tuple[int, int], list[Decoding]] = defaultdict(list)
    for decoding in decodings:
        held = sum(len(context.slots) for context in decoding.contexts)
        blocks = bucket_size(held // slots_per_row, 1)
        sizes[blocks, size(len(decoding.queries))].append(decoding)
    return list(sizes.values())


def split(decoding: Decoding, limit: int) -> list[Decoding]:
    """A decoding's queries in parts of at most `limit`, each with only its own contexts."""
    if len(decoding.queries) <= limit:
        return [decoding]
    parts = []
    for start in range(0, len(decoding.queries), limit):
        if decoding.reads is None:
            contexts, reads = decoding.contexts, None
        else:
            own = decoding.reads[start : start + limit]
            kept = sorted(set(own))
            position = {member: index for index, member in enumerate(kept)}
            contexts = [decoding.contexts[member] for member in kept]
            reads = [position[member] for member in own]
        parts.append(
            dataclasses.replace(
                decoding,
                queries=decoding.queries[start : start + limit],
                rows=decoding.rows[start : start + limit],
                contexts=contexts,
                reads=reads,
            )
        )
    return parts


def paged_shape(
    decodings: list[Decoding], layout: CacheLayout, size: Callable[[int], int]
) -> PagedShape:
    """The shape a paged decode of `decodings` compiles for.

    The cache pads to a bucket of its real rows, and the decode compiles once
    per bucket. A decoding of one context reads its blocks whole, in order;
    one of several reads only blocks holding real rows, at most twice as many
    as rows, so the block indices pad to that.

    Args:
        decodings: Decodings to call together.
        layout: Their contexts' layout.
        size: Pads a number of rows or queries.

    Returns:
        The shape.
    """
    context_rows = size(max(sum(c.rows for c in d.contexts) for d in decodings))
    in_order = all(len(d.contexts) == 1 for d in decodings)
    blocks = -(-context_rows // layout.rows_per_slot) if in_order else 2 * context_rows
    return PagedShape(
        context_rows,
        size(max(len(d.queries) for d in decodings)),
        blocks,
        any(d.reads is not None for d in decodings),
        in_order,
    )


def paged_inputs(
    chunk: list[Decoding],
    layout: CacheLayout,
    shape: PagedShape,
    count: int | None = None,
) -> tuple[list[int], tuple]:
    """What a paged decode of `chunk` reads, beside the pool and the tables' states.

    Args:
        chunk: Decodings of one call.
        layout: Their contexts' layout.
        shape: The shape the call compiles for.
        count: Items the call takes, the chunk padded; its bucket if None.

    Returns:
        The tables whose states the call stacks, in order, and its index into
        them, blocks, positions, counts and query rows, and with `masked`
        which rows each query reads, each with one item per decoding, padded
        to `count`.
    """
    count = bucket_size(len(chunk), 1) if count is None else count
    per_slot, per_row = layout.rows_per_slot, layout.slots_per_row
    tables = sorted({d.batch.table for d in chunk})
    index = np.zeros(count, dtype=np.int32)
    blocks = np.zeros((count, shape.blocks, per_row), dtype=np.int32)
    positions = np.zeros((count, shape.context_rows), dtype=np.int32)
    counts = np.zeros(count, dtype=np.int32)
    # The query rows stay on the device, where the embed left them.
    rows = jnp.stack(padded([_padded(d.rows, shape.queries) for d in chunk], count))
    # Which rows each query reads, when any decoding says; padding rows read
    # every row. Queries by rows, so only a masked call builds it.
    masked = (count, shape.queries, shape.context_rows) if shape.masked else (0, 0, 0)
    reads = np.ones(masked, dtype=bool)
    for position, decoding in enumerate(chunk):
        index[position] = tables.index(decoding.batch.table)
        block_at, row_at, starts = 0, 0, []
        for context in decoding.contexts:
            starts.append(row_at)
            held_blocks = -(-context.rows // per_slot)
            blocks[position, block_at : block_at + held_blocks] = np.reshape(
                context.slots[: held_blocks * per_row], (held_blocks, per_row)
            )
            positions[position, row_at : row_at + context.rows] = (
                block_at * per_slot + np.arange(context.rows)
            )
            block_at += held_blocks
            row_at += context.rows
        counts[position] = row_at
        # Padding reads the decoding's own first block, so a CPU's copy of the
        # slots read holds only the slots it reads.
        blocks[position, block_at:] = blocks[position, 0]
        if shape.masked and decoding.reads is not None:
            reads[position, : len(decoding.queries)] = False
            members = np.asarray(decoding.reads)
            # Queries come grouped by the context they read, so a few runs
            # of them cover every query.
            edges = np.flatnonzero(members[1:] != members[:-1]) + 1
            for first, stop in zip(
                np.r_[0, edges], np.r_[edges, len(members)], strict=True
            ):
                member = members[first]
                start = starts[member]
                end = start + decoding.contexts[member].rows
                reads[position, first:stop, start:end] = True
    # Padding decodings read the first one's blocks, for the same reason.
    blocks[len(chunk) :] = blocks[0]
    inputs = (index, blocks, positions, counts, rows)
    return tables, inputs + ((reads,) if shape.masked else ())


def paged_decode(
    functions: TabularFunctions,
    stored: bool,
    layout: CacheLayout,
    masked: bool,
    in_order: bool,
    embeds: bool = False,
) -> Callable[..., Any]:
    """Decode and postprocess over contexts read from their slots, vectorized over decodings.

    With `masked`, it also takes which rows each query reads; with
    `in_order`, every context is one entry, read a block at a time.

    Args:
        functions: The model's contract calls.
        stored: Whether the model reads caches as the runtime stores them.
        layout: The contexts' layout.
        masked: Whether a decoding reads several contexts, one per query.
        in_order: Whether every decoding reads one context.
        embeds: Whether the query rows are features to embed against the
            decoding's state first, as for a unit with its own.

    Returns:
        The function, which takes the parameters, the pool, the tables' states
        and `paged_inputs`.
    """
    indices = jax.tree.unflatten(layout.treedef, range(layout.treedef.num_leaves))

    def decode(
        parameters, source, states, index, blocks, positions, count, rows, reads=None
    ):
        def read(after: Any, part: Callable[[Any], Any]) -> Any:
            # Tied to `after`, so XLA gathers each part only once it is needed.
            _, slots = jax.lax.optimization_barrier((after, source))
            wanted = frozenset(jax.tree.leaves(part(indices)))
            cache = read_context(
                layout, slots, blocks, positions, count, reads, in_order, wanted
            )
            chosen = part(cache)
            return chosen if stored else load_cache(chosen)

        state = jax.tree.map(lambda values: values[index], states)
        if embeds:
            rows = functions.embed(parameters, state, rows)
        if functions.decode_reading is not None:
            outputs = functions.decode_reading(parameters, state, read, rows)
        else:
            cache = read_context(
                layout, source, blocks, positions, count, reads, in_order
            )
            if not stored:
                cache = load_cache(cache)
            outputs = functions.decode(parameters, state, cache, rows)
        return functions.postprocess(parameters, state, outputs)

    in_axes = (None, None, None, 0, 0, 0, 0, 0) + ((0,) if masked else ())
    return jax.vmap(decode, in_axes=in_axes)


def query_rows(rows: jax.Array, queries: list[int], size: int) -> jax.Array:
    """The rows of `queries`, padded to `size` by repeating the last.

    The index pads rather than the rows, so the gather's shapes are buckets
    and a new number of queries compiles nothing.

    Args:
        rows: A batch's embedded query rows.
        queries: The queries to take.
        size: Rows to pad to.

    Returns:
        The queries' rows.
    """
    index = np.full(size, queries[-1], dtype=np.int32)
    index[: len(queries)] = queries
    return _taken(rows, index)


@jax.jit
def _taken(rows: jax.Array, index: np.ndarray) -> jax.Array:
    """`rows[index]`, in one dispatch."""
    return rows[index]


# Rows padded in one dispatch, compiled once per pair of bucketed sizes.
_padded = jax.jit(pad_rows, static_argnums=(1, 2))

# The compiled helpers here, whose programs drop with a worker's own.
PROGRAMS = (_taken, _padded)
