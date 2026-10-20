"""Groups of series as the worker runs them: bucketed, laid out, and decoded from slots.

A step's series units arrive as dense grids. Here they pad to buckets, so
groups of about one size share a compiled call, and their caches lay out with
the patches first, so the slot pools, `pack` and `read_context` serve them as
they serve tabular contexts with rows first. The leaves a cache keeps per row
rather than per patch travel beside the slots.
"""

import math
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.models.series import (
    TARGET,
    SeriesFunctions,
    SeriesGroup,
    SeriesProperties,
)
from generalforecast.nn.attention import AttentionCache
from generalforecast.runtime.device import load_cache, to_cache_dtype
from generalforecast.runtime.memory import tree_bytes
from generalforecast.runtime.paging import CacheLayout, read_context, traced_layout
from generalforecast.runtime.shapes import bucket_size, pad_rows

# Two traced patch counts; the axis that grows between them is the patch axis.
_SMALL, _LARGE = 2, 4


def abstract_group(rows: int, steps: int) -> SeriesGroup:
    """A group's shapes and dtypes, for tracing a forecaster without running it."""

    def shaped(shape: tuple[int, ...], dtype: Any) -> Any:
        return cast(Any, jax.ShapeDtypeStruct(shape, dtype))

    return SeriesGroup(
        shaped((rows, steps), jnp.float32),
        shaped((rows, steps), jnp.bool_),
        shaped((rows,), jnp.int32),
        None,
        None,
        shaped((), jnp.int32),
    )


def series_layout(
    functions: SeriesFunctions,
    parameters: Any,
    rows: int,
    slot_bytes: int,
    cache_dtype: jnp.dtype | None = None,
) -> CacheLayout:
    """Traces a forecaster's cache for groups of `rows` rows, to lay it out in slots.

    Args:
        functions: The forecaster's contract calls, with `encode`.
        parameters: Its parameters, real or abstract.
        rows: Rows in a group, padded.
        slot_bytes: Bytes in a slot.
        cache_dtype: Dtype caches are stored in, or None for `encode`'s.

    Returns:
        The layout, its "rows" being patches.

    Raises:
        ValueError: If a cache leaf grows on any axis but its first.
    """
    assert functions.encode is not None
    encode = functions.encode
    patch = functions.properties(parameters).patch

    def traced(patches: int) -> Any:
        def encoded(parameters: Any, group: SeriesGroup) -> Any:
            cache = encode(parameters, group)
            return cache if cache_dtype is None else to_cache_dtype(cache, cache_dtype)

        return jax.eval_shape(
            encoded, parameters, abstract_group(rows, patches * patch)
        )

    return traced_layout(traced, slot_bytes, _SMALL, _LARGE)


class SeriesBytes(NamedTuple):
    """What a forecaster's cache takes for a group of padded rows.

    A patch takes `per_row` bytes for each row and `per_patch` beside them;
    the leaves without a patch axis take `fixed_per_row` for each row and
    `fixed` beside them. Every cache leaf's size is linear in the rows, so
    these hold for any group.
    """

    per_row: int
    per_patch: int
    fixed_per_row: int
    fixed: int


def series_bytes(
    functions: SeriesFunctions,
    parameters: Any,
    cache_dtype: jnp.dtype | None = None,
) -> SeriesBytes:
    """Bytes a forecaster's cache takes per patch and beside the patches.

    Zeros for a model that keeps no cache.

    Raises:
        ValueError: If a cache leaf's size is not linear in the group's rows.
    """
    if functions.encode is None:
        return SeriesBytes(0, 0, 0, 0)

    def measured(rows: int) -> tuple[int, int]:
        layout = series_layout(functions, parameters, rows, 1 << 30, cache_dtype)
        fixed = sum(
            math.prod(shape.shape) * jnp.dtype(shape.dtype).itemsize
            for shape in layout.fixed_shapes
        )
        return layout.row_bytes, fixed

    (small, small_fixed), (large, large_fixed) = measured(_SMALL), measured(_LARGE)
    grown = _LARGE - _SMALL
    if (large - small) % grown or (large_fixed - small_fixed) % grown:
        raise ValueError("a forecaster's cache must grow linearly with a group's rows")
    per_row, fixed_per_row = (
        (large - small) // grown,
        (large_fixed - small_fixed) // grown,
    )
    return SeriesBytes(
        per_row,
        small - _SMALL * per_row,
        fixed_per_row,
        small_fixed - _SMALL * fixed_per_row,
    )


def buckets(
    rows: int, steps: int, properties: SeriesProperties, minimum: int
) -> tuple[int, int]:
    """The rows and steps a group pads to, as the node sizes its cache.

    Rows pad to a power of two of at least `minimum`, and so do patches,
    within the model's context limit.
    """
    patch = properties.patch
    limit = max(properties.context_limit // patch, 1)
    patches = min(bucket_size(steps // patch, minimum), limit)
    return bucket_size(rows, minimum), patches * patch


def horizon_bucket(horizon: int, properties: SeriesProperties) -> int:
    """The steps a horizon pads to: whole output patches, as upstream runs them.

    Not a power of two of them: a model whose context reads the horizon's
    tokens, as Chronos-2's does, forecasts differently with more of them.
    """
    return -(-horizon // properties.output_patch) * properties.output_patch


def laid_group(
    values: np.ndarray,
    observed: np.ndarray,
    roles: np.ndarray,
    rows: int,
    steps: int,
    future: np.ndarray | None = None,
    future_observed: np.ndarray | None = None,
    horizon: int | None = None,
) -> SeriesGroup:
    """A unit's grid padded to a bucket, as the model reads it.

    Steps pad on the left, unobserved; rows pad at the end, as padding the
    group's `num_rows` leaves out; the future pads on the right, unknown.

    Args:
        values: The grid's values, `(real rows, real steps)`.
        observed: Which are observed.
        roles: Each real row's role.
        rows: Rows after padding.
        steps: Steps after padding.
        future: Known future values, `(real rows, real horizon)`, or None.
        future_observed: Which are known, or None.
        horizon: Steps the future pads to, with `future`.

    Returns:
        The group, on the host.
    """
    real_rows, real_steps = values.shape
    padded = np.zeros((rows, steps), np.float32)
    padded[:real_rows, steps - real_steps :] = values
    seen = np.zeros((rows, steps), bool)
    seen[:real_rows, steps - real_steps :] = observed
    kinds = np.full(rows, TARGET, np.int32)
    kinds[:real_rows] = roles
    ahead = ahead_seen = None
    if future is not None and future_observed is not None and horizon is not None:
        ahead = np.zeros((rows, horizon), np.float32)
        ahead[:real_rows, : future.shape[1]] = future
        ahead_seen = np.zeros((rows, horizon), bool)
        ahead_seen[:real_rows, : future.shape[1]] = future_observed
    host = cast(Any, padded), cast(Any, seen), cast(Any, kinds)
    return SeriesGroup(
        *host, cast(Any, ahead), cast(Any, ahead_seen), cast(Any, np.int32(real_rows))
    )


def fixed_leaves(layout: CacheLayout, cache: Any) -> list[Any]:
    """A cache's leaves without a patch axis, in the layout's order."""
    leaves = layout.treedef.flatten_up_to(cache)
    return [leaves[index] for index in layout.fixed_leaves]


def series_decode(
    functions: SeriesFunctions,
    stored: bool,
    layout: CacheLayout,
    horizon: int,
) -> Callable[..., Any]:
    """A decode of groups whose caches sit in slots, vectorized over the groups.

    Each item reads one entry whole from `source`, in order, puts its fixed
    leaves back, and decodes `horizon` steps.

    Args:
        functions: The forecaster's contract calls, with `decode`.
        stored: Whether `decode` reads caches as the runtime stores them.
        layout: The caches' layout.
        horizon: Steps to decode, static.

    Returns:
        A function of the parameters, the slots' source, and per item its
        blocks, positions, patch count and fixed leaves.
    """
    assert functions.decode is not None
    decode = functions.decode

    def one(
        parameters: Any,
        source: jax.Array,
        blocks: jax.Array,
        positions: jax.Array,
        count: jax.Array,
        fixed: list[Any],
    ) -> jax.Array:
        cache = _read(layout, source, blocks, positions, count, fixed)
        return decode(parameters, cache if stored else load_cache(cache), horizon)

    axes: Any = (None, None, 0, 0, 0, 0)
    return jax.vmap(one, in_axes=axes)


def series_extend(
    functions: SeriesFunctions,
    stored: bool,
    layout: CacheLayout,
    patches: int,
) -> Callable[..., Any]:
    """An extension of groups whose caches sit in slots, vectorized over the groups.

    Each item reads one entry's bucket of patches from `source`, in order,
    with its real count on the attention caches' lengths, puts its fixed
    leaves back, grows it by the item's new steps, whole patches, and pads
    the grown cache to `patches`, the bucket it is stored at.

    Args:
        functions: The forecaster's contract calls, with `extend`.
        stored: Whether `extend` reads caches as the runtime stores them.
        layout: The caches' layout.
        patches: Patches the grown cache is stored with, its bucket.

    Returns:
        A function of the parameters, the slots' source, and per item its
        blocks, positions, patch count, fixed leaves and new steps.
    """
    assert functions.extend is not None
    extend = functions.extend

    def one(
        parameters: Any,
        source: jax.Array,
        blocks: jax.Array,
        positions: jax.Array,
        count: jax.Array,
        fixed: list[Any],
        group: SeriesGroup,
    ) -> Any:
        cache = _read(layout, source, blocks, positions, count, fixed)
        grown = extend(parameters, cache if stored else load_cache(cache), group)
        return resized(layout, _plain(grown), patches)

    axes: Any = (None, None, 0, 0, 0, 0, 0)
    return jax.vmap(one, in_axes=axes)


def _plain(cache: Any) -> Any:
    """The cache without the lengths and masks on its attention caches, as its layout was traced."""
    return jax.tree.map(
        lambda layer: (
            layer._replace(length=None, mask=None)
            if isinstance(layer, AttentionCache)
            else layer
        ),
        cache,
        is_leaf=lambda node: isinstance(node, AttentionCache),
    )


def _read(
    layout: CacheLayout,
    source: jax.Array,
    blocks: jax.Array,
    positions: jax.Array,
    count: jax.Array,
    fixed: list[Any],
) -> Any:
    """One entry's cache read from its slots, its fixed leaves put back and its real patches on its attention caches' lengths.

    The patches past `count` are zeros the model skips by the lengths; the
    masks a paged tabular decode sets are not set.
    """
    cache = _plain(
        read_context(layout, source, blocks, positions, count, in_order=True)
    )
    leaves = layout.treedef.flatten_up_to(cache)
    for index, value in zip(layout.fixed_leaves, fixed, strict=True):
        leaves[index] = value
    cache = jax.tree.unflatten(layout.treedef, leaves)
    return jax.tree.map(
        lambda layer: (
            layer._replace(length=count.astype(jnp.int32))
            if isinstance(layer, AttentionCache)
            else layer
        ),
        cache,
        is_leaf=lambda node: isinstance(node, AttentionCache),
    )


def resized(layout: CacheLayout, cache: Any, rows: int) -> Any:
    """The cache with each row leaf padded with zeros, or cut, to `rows` on its row axis.

    An extension is stored at the bucket of its patches, as an encoding of
    as many is; the rows past its real patches are never read.
    """
    leaves = layout.treedef.flatten_up_to(cache)
    for leaf in layout.row_leaves:
        value = jnp.moveaxis(leaves[leaf.index], leaf.axis, 0)
        value = pad_rows(value, rows) if value.shape[0] < rows else value[:rows]
        leaves[leaf.index] = jnp.moveaxis(value, 0, leaf.axis)
    return jax.tree.unflatten(layout.treedef, leaves)


def patch_bucket(patches: int, properties: SeriesProperties, minimum: int) -> int:
    """The patches an entry of `patches` real ones is stored at: a power of two of at least `minimum`, within the model's context."""
    limit = max(properties.context_limit // properties.patch, 1)
    return min(bucket_size(patches, minimum), limit)


def group_bytes(group: SeriesGroup) -> int:
    """Bytes a group's arrays take."""
    return tree_bytes(group)
