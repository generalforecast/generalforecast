# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Context caches in fixed-size slots of bytes, so device and host memory are pools.

A layout says where each part of a model's context cache goes. Leaves with a
row axis (keys and values) pack into blocks of `rows_per_slot` rows: within a
block, each leaf's rows are one contiguous run, so one layer's keys for the
block's rows can be read in place. A block is one slot, or for a row wider
than a slot, `slots_per_row` slots in a row of the entry's slot list. Leaves
without a row axis (attention lengths) are a few bytes and travel beside the
slots.

Every model shares one pool per tier: a slot holds bytes, and each model's
layout decides how many of its rows fit in one.
"""

import gc
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.constants import (
    MOVE_SLOTS,
    PACK_BYTES,
    PAGE_TRACE_LARGE,
    PAGE_TRACE_SMALL,
)
from generalforecast.models.contract import TabularFunctions
from generalforecast.nn.attention import AttentionCache
from generalforecast.runtime.device import to_cache_dtype
from generalforecast.runtime.shapes import bucket_size


@dataclass(frozen=True)
class _RowLeaf:
    index: int
    axis: int
    dtype: np.dtype
    # The leaf's shape with its row axis removed.
    shape: tuple[int, ...]
    row_bytes: int


@dataclass(frozen=True)
class CacheLayout:
    """How one model's context cache packs into slots.

    Attributes:
        treedef: The cache's tree structure.
        row_leaves: Leaves with a row axis, in packing order.
        fixed_leaves: Indices of leaves without a row axis.
        fixed_shapes: Their shapes and dtypes.
        row_bytes: Bytes of one context row, across every leaf.
        rows_per_slot: Rows one block holds: one if a row is wider than a slot.
        slot_bytes: Bytes in a slot.
        slots_per_row: Slots in a block: more than one only for a row wider
            than a slot.
    """

    treedef: Any
    row_leaves: tuple[_RowLeaf, ...]
    fixed_leaves: tuple[int, ...]
    fixed_shapes: tuple[jax.ShapeDtypeStruct, ...]
    row_bytes: int
    rows_per_slot: int
    slot_bytes: int
    slots_per_row: int = 1

    @property
    def block_bytes(self) -> int:
        """Bytes in a block of slots."""
        return self.slots_per_row * self.slot_bytes

    def slots(self, rows: int) -> int:
        """Slots a cache of `rows` rows takes."""
        return -(-rows // self.rows_per_slot) * self.slots_per_row

    def packed_slots(self, cache: Any) -> int:
        """Slots `pack` packs `cache` into: one block for each of its encoded rows' slots."""
        first = self.row_leaves[0]
        leaves = self.treedef.flatten_up_to(cache)
        return self.slots(leaves[first.index].shape[first.axis])


def cache_layout(
    functions: TabularFunctions,
    model: Any,
    num_features: int,
    slot_bytes: int,
    cache_dtype: jnp.dtype | None = None,
) -> CacheLayout:
    """Traces a model's context cache, without running it, to lay it out in slots.

    Args:
        functions: The model's contract calls.
        model: Model parameters, real or abstract.
        num_features: Number of feature columns in the table.
        slot_bytes: Bytes in a slot.
        cache_dtype: Dtype caches are stored in, or None for the dtype `encode` returns.

    Returns:
        The layout.

    Raises:
        ValueError: If the cache is not laid out as `TabularFunctions` says:
            rows first, scalar lengths.
    """
    properties = functions.properties(model)
    classes = 0 if properties.regression else min(properties.class_slots, 2)
    target_dtype = jnp.float32 if properties.regression else jnp.int32

    def traced(rows: int) -> Any:
        features = jax.ShapeDtypeStruct((rows, num_features), jnp.float32)
        targets = jax.ShapeDtypeStruct((rows,), target_dtype)
        count = jax.ShapeDtypeStruct((), jnp.int32)

        def encoded(model: Any, features: Any, targets: Any, count: Any) -> Any:
            state, context_rows = functions.fit(
                model, features, targets, classes, num_rows=count
            )
            cache = functions.encode(
                model, state, context_rows, targets, num_rows=count
            )
            return cache if cache_dtype is None else to_cache_dtype(cache, cache_dtype)

        return jax.eval_shape(encoded, model, features, targets, count)

    return traced_layout(
        traced, slot_bytes, PAGE_TRACE_SMALL, PAGE_TRACE_LARGE, lengths_only=True
    )


def traced_layout(
    traced: Callable[[int], Any],
    slot_bytes: int,
    small: int,
    large: int,
    lengths_only: bool = False,
) -> CacheLayout:
    """Lays out a cache from its shapes at two row counts.

    The axis that grows between the two is the row axis; leaves that do not
    grow are fixed and travel beside the slots.

    Args:
        traced: The cache's shapes and dtypes for a row count.
        slot_bytes: Bytes in a slot.
        small: The smaller row count traced.
        large: The larger.
        lengths_only: Whether a fixed leaf must be a scalar, as a tabular
            cache's attention lengths are.

    Returns:
        The layout.

    Raises:
        ValueError: If a leaf grows on any axis but its first, or, with
            `lengths_only`, a fixed leaf has a shape.
    """
    shapes, treedef = jax.tree.flatten(traced(small))
    grown_shapes = jax.tree.leaves(traced(large))
    row_leaves, fixed_leaves = [], []
    for index, (short, long) in enumerate(zip(shapes, grown_shapes, strict=True)):
        grown = [
            axis
            for axis, (a, b) in enumerate(zip(short.shape, long.shape, strict=True))
            if a != b
        ]
        if not grown:
            if lengths_only and short.shape:
                raise ValueError(
                    f"cache leaf {index} has no row axis, so must be a length"
                )
            fixed_leaves.append(index)
            continue
        if grown != [0] or short.shape[0] != small:
            raise ValueError(
                f"cache leaf {index} does not have its rows on its first axis"
            )
        axis = grown[0]
        shape = short.shape[:axis] + short.shape[axis + 1 :]
        dtype = np.dtype(short.dtype)
        row_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        row_leaves.append(_RowLeaf(index, axis, dtype, shape, row_bytes))
    row_bytes = sum(leaf.row_bytes for leaf in row_leaves)
    # A row wider than a slot takes whole slots of its own, as the node counts.
    slots_per_row = -(-row_bytes // slot_bytes)
    return CacheLayout(
        treedef,
        tuple(row_leaves),
        tuple(fixed_leaves),
        tuple(
            jax.ShapeDtypeStruct(shapes[i].shape, shapes[i].dtype) for i in fixed_leaves
        ),
        row_bytes,
        max(1, slot_bytes // row_bytes),
        slot_bytes,
        slots_per_row,
    )


def _packed(layout: CacheLayout, cache: Any, start: jax.Array, count: int) -> jax.Array:
    """Blocks `start` to `start + count` of a cache's rows as slots; rows past its end pack as zeros."""
    leaves = layout.treedef.flatten_up_to(cache)
    rows = start * layout.rows_per_slot + jnp.arange(count * layout.rows_per_slot)
    blocks = []
    for leaf in layout.row_leaves:
        values = jnp.take(
            leaves[leaf.index], rows, axis=leaf.axis, mode="fill", fill_value=0
        )
        values = jnp.moveaxis(values, leaf.axis, 0)
        # Booleans have no bit pattern to reinterpret; a byte each is exact.
        if values.dtype == jnp.bool_:
            values = values.astype(jnp.uint8)
        data = jax.lax.bitcast_convert_type(values, jnp.uint8)
        blocks.append(data.reshape(count, layout.rows_per_slot * leaf.row_bytes))
    data = jnp.concatenate(blocks, axis=1)
    data = jnp.pad(data, ((0, 0), (0, layout.block_bytes - data.shape[1])))
    return data.reshape(count * layout.slots_per_row, layout.slot_bytes)


@partial(jax.jit, static_argnums=0)
def pack(layout: CacheLayout, cache: Any) -> jax.Array:
    """Packs a cache's rows into slots.

    Args:
        layout: The cache's layout.
        cache: The cache, with every row it was encoded with, padding included.

    Returns:
        The slots' bytes, `(slots, slot_bytes)` uint8.
    """
    count = layout.packed_slots(cache) // layout.slots_per_row
    return _packed(layout, cache, jnp.int32(0), count)


@partial(jax.jit, static_argnums=0, donate_argnums=1)
def _write_blocks(
    layout: CacheLayout, pool: jax.Array, cache: Any, start: jax.Array, slots: jax.Array
) -> jax.Array:
    """`pool` with the cache's blocks from `start` packed into `slots`; a slot past its end is padding, and dropped."""
    count = slots.shape[0] // layout.slots_per_row
    return pool.at[slots].set(_packed(layout, cache, start, count), mode="drop")


def pack_into(
    pool: jax.Array, slots: Sequence[int], layout: CacheLayout, cache: Any
) -> jax.Array:
    """`pool` with `cache` packed into `slots`, a part of `PACK_BYTES` at a time.

    The pool is donated to each part's write, so the device holds one part's
    bytes beside the cache and the pool. Each part but the last is the same
    size, and the last pads to a bucket, so a layout compiles a few programs.
    """
    most = max(1, PACK_BYTES // layout.block_bytes)
    most = 1 << (most.bit_length() - 1)
    total = len(slots) // layout.slots_per_row
    for start in range(0, total, most):
        count = min(most, total - start)
        index = np.full(
            bucket_size(count, 1) * layout.slots_per_row, pool.shape[0], np.int32
        )
        part = slots[
            start * layout.slots_per_row : (start + count) * layout.slots_per_row
        ]
        index[: len(part)] = part
        pool = _write_blocks(layout, pool, cache, jnp.int32(start), jnp.asarray(index))
    return pool


@dataclass(frozen=True)
class StateLayout:
    """How a unit's own state packs into whole slots, after its rows.

    Attributes:
        treedef: The state's tree structure, static fields included.
        leaves: Each leaf's shape and dtype.
        slot_bytes: Bytes in a slot.
    """

    treedef: Any
    leaves: tuple[jax.ShapeDtypeStruct, ...]
    slot_bytes: int

    @property
    def slots(self) -> int:
        """Slots the state takes."""
        total = sum(
            math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize
            for leaf in self.leaves
        )
        return -(-total // self.slot_bytes)


def state_layout(state: Any, slot_bytes: int) -> StateLayout:
    """Lays out a state, real or abstract, in slots of `slot_bytes`."""
    leaves, treedef = jax.tree.flatten(state)
    shapes = tuple(jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves)
    return StateLayout(treedef, shapes, slot_bytes)


@partial(jax.jit, static_argnums=0)
def pack_state(layout: StateLayout, state: Any) -> jax.Array:
    """Packs a state's leaves, one after another, into whole slots.

    Args:
        layout: The state's layout.
        state: The state, as `fit` returned it.

    Returns:
        The slots' bytes, `(slots, slot_bytes)` uint8.
    """
    data = [_leaf_bytes(leaf) for leaf in layout.treedef.flatten_up_to(state)]
    flat = jnp.concatenate(data) if data else jnp.zeros(0, jnp.uint8)
    size = layout.slots * layout.slot_bytes
    return jnp.pad(flat, (0, size - flat.shape[0])).reshape(-1, layout.slot_bytes)


@partial(jax.jit, static_argnums=0)
def unpack_states(layout: StateLayout, data: jax.Array) -> Any:
    """States from the slots `pack_state` filled, stacked.

    Args:
        layout: The states' layout.
        data: Each state's slots' bytes, `(states, slots, slot_bytes)`.

    Returns:
        The states, each leaf with the states on its first axis.
    """

    def one(slots: jax.Array) -> Any:
        flat, offset, leaves = slots.reshape(-1), 0, []
        for leaf in layout.leaves:
            dtype = np.dtype(leaf.dtype)
            size = math.prod(leaf.shape) * dtype.itemsize
            piece = flat[offset : offset + size]
            offset += size
            if dtype == np.bool_:
                value = piece.astype(jnp.bool_)
            elif dtype.itemsize == 1:
                value = jax.lax.bitcast_convert_type(piece, dtype)
            else:
                value = jax.lax.bitcast_convert_type(
                    piece.reshape(-1, dtype.itemsize), dtype
                )
            leaves.append(value.reshape(leaf.shape))
        return jax.tree.unflatten(layout.treedef, leaves)

    return jax.vmap(one)(data)


def _leaf_bytes(leaf: jax.Array) -> jax.Array:
    """A leaf's bytes, flat; booleans take one byte each."""
    leaf = jnp.asarray(leaf).reshape(-1)
    if leaf.dtype == jnp.bool_:
        return leaf.astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(leaf, jnp.uint8).reshape(-1)


def read_context(
    layout: CacheLayout,
    source: jax.Array,
    blocks: jax.Array,
    positions: jax.Array,
    count: jax.Array,
    reads: jax.Array | None = None,
    in_order: bool = False,
    parts: frozenset[int] | None = None,
) -> Any:
    """A context's cache, gathered row by row from the slots of its entries.

    Meant to run inside a compiled decode. Each row's bytes come straight from
    `source`, real rows first, so the decode holds the cache once: neither the
    slots' bytes whole nor a reordered copy of them. Rows past `count` are
    zeros, which are always numbers, so masked keys stay harmless to attention.

    Args:
        layout: The entries' layout.
        source: Slots' bytes, `(slots, slot_bytes)`: the pool, or a copy of the
            slots read.
        blocks: The slots of each block the context reads in `source`,
            `(blocks, slots_per_row)`.
        positions: Each cache row's place among those blocks' rows, `block *
            rows_per_slot + row`, real rows first.
        count: Real rows.
        reads: Which rows each query reads, `(queries, rows)`, or None when
            every query reads them all.
        in_order: Whether the context is one entry, its rows its blocks' rows
            in order, at least as many as `positions` has. Each leaf then comes
            a block at a time, far faster for long contexts than row by row.
        parts: The flat indices of the row-bearing leaves to read, or None
            for all; the others are None.

    Returns:
        The context cache, its attention lengths set to `count`, and its mask
        to the rows each query reads.
    """
    rows = positions.shape[0]
    block = positions // layout.rows_per_slot
    within = positions % layout.rows_per_slot
    real = jnp.arange(rows) < count
    leaves: list[Any] = [None] * layout.treedef.num_leaves
    offset = 0
    for leaf in layout.row_leaves:
        if parts is not None and leaf.index not in parts:
            offset += layout.rows_per_slot * leaf.row_bytes
            continue
        if in_order and layout.slots_per_row == 1:
            width = layout.rows_per_slot * leaf.row_bytes
            starts = jnp.full(blocks.shape[0], offset, jnp.int32)
            data = _slices(source, blocks[:, 0], starts, width)
            data = data.reshape(-1, leaf.row_bytes)[:rows]
        else:
            data = _leaf_rows(
                layout, source, blocks, block, within, offset, leaf.row_bytes
            )
        data = jnp.where(real[:, None], data, 0)
        if leaf.dtype == np.bool_:
            values = data.astype(jnp.bool_)
        else:
            if leaf.dtype.itemsize > 1:
                data = data.reshape(rows, -1, leaf.dtype.itemsize)
            values = jax.lax.bitcast_convert_type(data, leaf.dtype)
        values = values.reshape((rows, *leaf.shape))
        leaves[leaf.index] = jnp.moveaxis(values, 0, leaf.axis)
        offset += layout.rows_per_slot * leaf.row_bytes
    for index, shape in zip(layout.fixed_leaves, layout.fixed_shapes, strict=True):
        leaves[index] = jnp.zeros(shape.shape, shape.dtype)
    cache = jax.tree.unflatten(layout.treedef, leaves)

    def limited(layer: Any) -> Any:
        if not isinstance(layer, AttentionCache):
            return layer
        return layer._replace(length=count.astype(jnp.int32), mask=reads)

    return jax.tree.map(
        limited, cache, is_leaf=lambda node: isinstance(node, AttentionCache)
    )


def _leaf_rows(
    layout: CacheLayout,
    source: jax.Array,
    blocks: jax.Array,
    block: jax.Array,
    within: jax.Array,
    offset: int,
    row_bytes: int,
) -> jax.Array:
    """One leaf's bytes for each row, `(rows, row_bytes)`, gathered from `source`.

    A block's leaves lie one after another, each its rows in order, from
    `offset`. A row wider than a slot spans its block's slots, so its bytes
    come in pieces, one per slot they touch.
    """
    slot_bytes = layout.slot_bytes
    if layout.slots_per_row == 1:
        return _slices(source, blocks[block, 0], offset + within * row_bytes, row_bytes)
    # A row wider than a slot is the only row of its block.
    pieces = []
    for slot in range(offset // slot_bytes, (offset + row_bytes - 1) // slot_bytes + 1):
        low = max(offset, slot * slot_bytes)
        high = min(offset + row_bytes, (slot + 1) * slot_bytes)
        starts = jnp.full(block.shape, low - slot * slot_bytes, jnp.int32)
        pieces.append(_slices(source, blocks[block, slot], starts, high - low))
    return jnp.concatenate(pieces, axis=1)


def _slices(
    source: jax.Array, slots: jax.Array, starts: jax.Array, size: int
) -> jax.Array:
    """`size` bytes of each of `slots` in `source` from its start, `(slots, size)`."""

    def piece(slot: jax.Array, start: jax.Array) -> jax.Array:
        return jax.lax.dynamic_slice(source, (slot, start), (1, size))[0]

    return jax.vmap(piece)(slots, starts)


@jax.jit
def _read(data: jax.Array, slots: np.ndarray) -> jax.Array:
    """`data[slots]`, in one dispatch."""
    return data[slots]


@partial(jax.jit, donate_argnums=0)
def _write(pool: jax.Array, slots: jax.Array, data: jax.Array) -> jax.Array:
    """`pool` with `data` in `slots`; a slot past its end is padding, and dropped."""
    return pool.at[slots].set(data, mode="drop")


def _parts(slots: list[int], pad: int) -> list[tuple[int, int, np.ndarray]]:
    """`slots` in parts of `MOVE_SLOTS`, the last padded with `pad` to a bucket: each part's start, count and slots."""
    found = []
    for start in range(0, len(slots), MOVE_SLOTS):
        part = slots[start : start + MOVE_SLOTS]
        padded = np.full(bucket_size(len(part), 1), pad, np.int32)
        padded[: len(part)] = part
        found.append((start, len(part), padded))
    return found


class PoolLost(RuntimeError):
    """A write consumed the device pool and failed, so every context it held is gone.

    The pool refuses reads and writes from then on, since slots the node
    believes filled would read as zeros; the node and worker must forget their
    caches before the next step.
    """


def _check(slots: list[int], capacity: int) -> None:
    """Refuses slots past a pool's end, which the node never assigns."""
    if max(slots) >= capacity:
        raise ValueError(f"slot {max(slots)} is past the pool's {capacity} slots")


class DevicePool:
    """Slots of bytes in device memory, at most as many as the node's device tier has.

    On an accelerator the whole pool is allocated when it is made, before any
    step, so it never briefly holds two copies while growing and never has to
    find one block that large among a step's buffers. On a CPU the device is
    host memory, so the pool is one NumPy array whose pages the operating system
    commits as slots are first written, as the host pool's are.
    """

    def __init__(self, slot_bytes: int, slots: int) -> None:
        """Sets the pool's size, and allocates it on an accelerator.

        Raises:
            MemoryError: If the device cannot hold the pool, even once the
                pools of clients no longer referenced are freed.
        """
        self._shape = (slots, slot_bytes)
        self._on_host = jax.default_backend() == "cpu"
        self._data: jax.Array | None = None
        self._host: np.ndarray | None = None
        self._lost: str | None = None
        if self._on_host or not slots:
            return
        try:
            self._on_device().block_until_ready()
        except jax.errors.JaxRuntimeError:
            # Pools of clients no longer referenced may wait for the collector.
            self._data = None
            gc.collect()
            try:
                self._on_device().block_until_ready()
            except jax.errors.JaxRuntimeError as error:
                self._data = None
                stats = jax.devices()[0].memory_stats() or {}
                raise MemoryError(
                    f"the device tier's pool of {slots * slot_bytes / 2**30:.2f} GiB "
                    f"does not fit: JAX may use {stats.get('bytes_limit', 0) / 2**30:.2f}"
                    f" GiB and holds {stats.get('bytes_in_use', 0) / 2**30:.2f} GiB, and "
                    "other libraries may hold more. Close other clients, free other "
                    "GPU memory, or give a smaller device_bytes."
                ) from error

    def release(self) -> None:
        """Frees the pool's memory; its slots read as empty until the next write."""
        if self._data is not None and not self._data.is_deleted():
            self._data.delete()
        self._data = None
        self._host = None

    @property
    def lost(self) -> bool:
        """Whether a failed write took the pool's contents with it."""
        return self._lost is not None

    @property
    def data(self) -> jax.Array | np.ndarray:
        """Every slot's bytes."""
        if self._lost is not None:
            raise PoolLost(self._lost)
        if self._on_host:
            if self._host is None:
                self._host = np.zeros(self._shape, np.uint8)
            return self._host
        return self._on_device()

    def _on_device(self) -> jax.Array:
        """The accelerator's pool, allocated at its first use."""
        if self._data is None:
            self._data = jnp.zeros(self._shape, dtype=jnp.uint8)
        return self._data

    def write(self, slots: list[int], data: jax.Array | np.ndarray) -> None:
        """Writes `data` into `slots`, in place.

        Bytes from the host, as a context moved up from a lower tier, go in
        parts (`MOVE_SLOTS`), so moves of every size share a few programs;
        the last part's padding points past the pool, and is dropped.
        """
        _check(slots, self._shape[0])
        pool = self.data
        if isinstance(pool, np.ndarray):
            pool[slots] = np.asarray(data)
            return
        if isinstance(data, np.ndarray):
            calls = []
            for start, count, index in _parts(slots, self._shape[0]):
                part = np.zeros((len(index), self._shape[1]), np.uint8)
                part[:count] = data[start : start + count]
                calls.append((index, part))
        else:
            calls = [(jnp.asarray(slots, dtype=jnp.int32), data)]
        self._update(
            [
                lambda pool, index=index, part=part: _write(pool, index, part)
                for index, part in calls
            ]
        )

    def write_cache(self, slots: list[int], layout: CacheLayout, cache: Any) -> None:
        """Packs `cache` into `slots`, in place; see `pack_into`."""
        _check(slots, self._shape[0])
        pool = self.data
        if isinstance(pool, np.ndarray):
            pool[slots] = np.asarray(pack(layout, cache))
            return
        self._update([lambda pool: pack_into(pool, slots, layout, cache)])

    def _update(self, writes: list[Callable[[jax.Array], jax.Array]]) -> None:
        """Replaces the device pool by each of `writes` in turn, each donated the last's pool."""
        pool = self._on_device()
        try:
            for write in writes:
                pool = write(pool)
                self._data = pool
        except Exception as error:
            if not pool.is_deleted():
                raise
            # The write took the pool, as donation lets it, and then failed.
            self._data = None
            self._lost = (
                f"writing the device pool failed and lost its contexts: {error}"
            )
            raise PoolLost(self._lost) from error

    def source(self, slots: np.ndarray) -> tuple[jax.Array, np.ndarray]:
        """What a decode reads `slots` from, and their places in it.

        The pool itself on an accelerator. A CPU's pool is a NumPy array, so a
        copy of the slots read, padded to a bucket of the slots named, so
        a call padded to its chunk has the chunk's source whichever slots
        repeat.
        """
        pool = self.data
        if not isinstance(pool, np.ndarray):
            return pool, slots
        unique, places = np.unique(slots, return_inverse=True)
        copy = np.zeros((bucket_size(slots.size, 1), pool.shape[1]), np.uint8)
        copy[: len(unique)] = pool[unique]
        return jnp.asarray(copy), places.reshape(slots.shape).astype(np.int32)

    def read_host(self, slots: list[int]) -> np.ndarray:
        """The bytes of `slots`, in order, in host memory.

        On an accelerator the slots are read in parts, as `write` writes
        them, so moving a context down a tier compiles a few programs and
        copies at most a part to the host at a time.
        """
        pool = self.data
        if isinstance(pool, np.ndarray):
            return pool[np.asarray(slots, dtype=np.int64)]
        read = np.empty((len(slots), self._shape[1]), np.uint8)
        for start, count, index in _parts(slots, slots[-1] if slots else 0):
            read[start : start + count] = np.asarray(_read(pool, index))[:count]
        return read

    def read(self, slots: list[int] | np.ndarray) -> jax.Array:
        """The bytes of `slots`, in order."""
        pool = self.data
        if isinstance(pool, np.ndarray):
            return jnp.asarray(pool[np.asarray(slots, dtype=np.int64)])
        return _read(pool, np.asarray(slots, dtype=np.int32))


class HostPool:
    """Slots of bytes in host memory, as many as the node's host tier has.

    Allocated at its first write; the operating system commits pages as they
    are written.
    """

    def __init__(self, slot_bytes: int, slots: int) -> None:
        """Sets the pool's size; nothing is allocated until a write."""
        self._shape = (slots, slot_bytes)
        self._data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        """Every slot's bytes."""
        if self._data is None:
            self._data = self._allocate()
        return self._data

    def write(self, slots: list[int], data: np.ndarray) -> None:
        """Writes `data` into `slots`."""
        _check(slots, self._shape[0])
        self.data[slots] = data

    def read(self, slots: list[int]) -> np.ndarray:
        """The bytes of `slots`, in order."""
        return np.asarray(self.data[slots])

    def write_bytes(self, slots: list[int], data: np.ndarray) -> None:
        """Writes `data`'s bytes across `slots` in order, without padding the last."""
        _check(slots, self._shape[0])
        width = self._shape[1]
        if data.size > len(slots) * width:
            raise ValueError(f"{data.size} bytes do not fit {len(slots)} slots")
        for index, slot in enumerate(slots):
            part = data[index * width : (index + 1) * width]
            self.data[slot, : part.size] = part

    def read_bytes(self, slots: list[int], size: int) -> np.ndarray:
        """The first `size` bytes across `slots`, in order, copying no more."""
        width = self._shape[1]
        read = np.empty(size, np.uint8)
        for index, slot in enumerate(slots):
            start = index * width
            count = min(width, size - start)
            if count <= 0:
                break
            read[start : start + count] = self.data[slot, :count]
        return read

    def _allocate(self) -> np.ndarray:
        return np.zeros(self._shape, np.uint8)


class DiskPool(HostPool):
    """Slots of bytes in a file, as many as the node's disk tier has, mapped into memory."""

    def __init__(self, slot_bytes: int, slots: int, path: Path) -> None:
        """Starts with an empty file at `path`, replacing any left from before."""
        super().__init__(slot_bytes, slots)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        self._path = path

    def _allocate(self) -> np.ndarray:
        with open(self._path, "r+b") as file:
            file.truncate(self._shape[0] * self._shape[1])
        return np.memmap(self._path, dtype=np.uint8, mode="r+", shape=self._shape)
