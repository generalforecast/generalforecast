from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from generalforecast.models.prepare import (
    Columns,
    constant_columns,
    kept_first,
    valid_rows,
)

# Standardized values are clipped here, as in TabPFN's own preprocessing.
CLIP = 100.0
# Missing-value flags: NaN, +inf and -inf each have their own value.
NAN_FLAG, POSITIVE_FLAG, NEGATIVE_FLAG = -2.0, 2.0, 4.0


class RankBuckets(NamedTuple):
    """Each column's bucket edges, with how many context values lie below and at most."""

    edges: Float[Array, " features buckets"]
    below: Float[Array, " features buckets"]
    at_most: Float[Array, " features buckets"]


class TabPFNSnapshot(NamedTuple):
    """Context means and scales, the target scale for regression, and rank buckets.

    `columns` puts first the columns TabPFN's preprocessing keeps: it drops a
    column whose context values all equal the first, NaN equal to nothing, or
    are all NaN.
    """

    center: Float[Array, " features"]
    scale: Float[Array, " features"]
    target_center: Float[Array, ""] | None
    target_scale: Float[Array, ""] | None
    columns: Columns
    ranks: RankBuckets | None = None


def _sum(values: Array, mask: Array) -> Array:
    return jnp.sum(jnp.where(mask, values, 0), axis=0)


def fit_snapshot(
    features: Float[Array, " rows features"],
    targets: Float[Array, " rows"] | None = None,
    num_rows: Int[Array, ""] | int | None = None,
    rank_buckets: int | None = None,
) -> TabPFNSnapshot:
    """Fits the means and scales TabPFN standardizes with, on the context rows.

    Args:
        features: Context rows, with NaN or infinities for missing values.
        targets: Regression targets, or None for classification.
        num_rows: Real rows at the start; the rest are padding. None means all.
        rank_buckets: Most edges each column is ranked against, or None for no
            ranks.

    Returns:
        The snapshot.
    """
    valid = valid_rows(features.shape[0], num_rows)
    count = jnp.sum(valid).astype(features.dtype)
    finite = valid[:, None] & jnp.isfinite(features)
    finite_count = jnp.sum(finite, axis=0).astype(features.dtype)
    finite_mean = _sum(features, finite) / jnp.maximum(finite_count, 1)
    filled = jnp.where(jnp.isfinite(features), features, finite_mean)
    center = _sum(filled, valid[:, None]) / jnp.maximum(count, 1)
    variance = _sum((filled - center) ** 2, valid[:, None]) / jnp.maximum(count - 1, 1)
    deviation = jnp.sqrt(variance)
    scale = jnp.where((deviation == 0) | (count == 1), 1, deviation)
    target_center = target_scale = None
    if targets is not None:
        target_center = _sum(targets, valid) / jnp.maximum(count, 1)
        spread = _sum((targets - target_center) ** 2, valid) / jnp.maximum(count, 1)
        target_scale = jnp.sqrt(spread) + 1e-20
    ranks = None
    if rank_buckets is not None:
        filled = jnp.where(jnp.isfinite(features), features, center)
        ranks = jax.vmap(_buckets, in_axes=(1, None, None, None))(
            filled, valid, jnp.sum(valid), rank_buckets
        )
    # Upstream raises when every column is constant; one stays here instead.
    columns = kept_first(constant_columns(features, valid))
    return TabPFNSnapshot(center, scale, target_center, target_scale, columns, ranks)


def _buckets(
    column: Float[Array, " rows"], valid: Array, count: Array, limit: int
) -> RankBuckets:
    """One column's rank buckets, padded to `limit` edges by repeating the last."""
    rows = column.shape[0]
    size = limit
    ordered = jnp.sort(jnp.where(valid, column.astype(jnp.float32), jnp.inf))
    position = jnp.arange(rows)
    new = jnp.concatenate([jnp.ones(1, bool), ordered[1:] != ordered[:-1]])
    distinct = jnp.cumsum(new) - 1
    # Padding sorts last; numbering it past every real value keeps `distinct`
    # ascending and out of every search below.
    distinct = jnp.where(position < count, distinct, rows + position)
    num_distinct = distinct[jnp.maximum(count - 1, 0)] + 1
    k = jnp.minimum(limit, count)
    steps = jnp.minimum(jnp.arange(size), k - 1).astype(jnp.float32)
    spread = jnp.maximum(k - 1, 1).astype(jnp.float32)
    by_value = jnp.round(steps * ((num_distinct - 1).astype(jnp.float32) / spread))
    by_row = jnp.round(steps * ((count - 1).astype(jnp.float32) / spread))
    targets = jnp.where(
        num_distinct <= k,
        by_value.astype(distinct.dtype),
        distinct[by_row.astype(jnp.int32)],
    )
    targets = jnp.where(k > 1, targets, 0)
    below = jnp.searchsorted(distinct, targets, side="left")
    at_most = jnp.searchsorted(distinct, targets, side="right")
    return RankBuckets(
        edges=ordered[below],
        below=below.astype(jnp.float32),
        at_most=at_most.astype(jnp.float32),
    )


def ranks(
    buckets: RankBuckets, features: Float[Array, " rows features"]
) -> Float[Array, " rows features"]:
    """Midrank of each filled cell among its column's context values, in [0, 1].

    Args:
        buckets: Rank buckets from `fit_snapshot`.
        features: Rows with non-finite cells filled.

    Returns:
        Ranks as fractions of the context rows.
    """

    def column(edges: Array, below: Array, at_most: Array, values: Array) -> Array:
        k = edges.shape[0]
        values = values.astype(jnp.float32)
        left = jnp.searchsorted(edges, values, side="left")
        right = jnp.searchsorted(edges, values, side="right")
        low = jnp.maximum(left - 1, 0)
        high = jnp.minimum(left, k - 1)
        width = edges[high] - edges[low]
        inside = width > 0
        weight = jnp.where(
            inside, (values - edges[low]) / jnp.where(inside, width, 1.0), 0.0
        )
        counts = at_most[low] + weight * (below[high] - at_most[low])
        counts = jnp.where(right > left, 0.5 * (below[high] + at_most[high]), counts)
        counts = jnp.where((left == 0) & (right == left), 0.0, counts)
        lowest = 0.5 * (below[0] + at_most[0])
        highest = 0.5 * (below[-1] + at_most[-1])
        return jnp.clip(counts, lowest, highest) / jnp.maximum(at_most[-1], 1)

    return jax.vmap(column, in_axes=(0, 0, 0, 1), out_axes=1)(
        buckets.edges, buckets.below, buckets.at_most, features
    )


def missing_flags(features: Float[Array, " rows features"]) -> Array:
    """-2 for NaN, 2 for +inf, 4 for -inf, and 0 for finite cells."""
    return (
        jnp.isnan(features) * NAN_FLAG
        + jnp.isposinf(features) * POSITIVE_FLAG
        + jnp.isneginf(features) * NEGATIVE_FLAG
    ).astype(features.dtype)


def fill(
    snapshot: TabPFNSnapshot, features: Float[Array, " rows features"]
) -> Float[Array, " rows features"]:
    """Replaces non-finite cells with the column means."""
    return jnp.where(jnp.isfinite(features), features, snapshot.center)


def prepare_features(
    snapshot: TabPFNSnapshot, features: Float[Array, " rows features"]
) -> Float[Array, " rows features channels"]:
    """Standardized, clipped values with their missing-value flags and ranks.

    Ranks are of the filled values, before clipping, which would tie outliers.

    Args:
        snapshot: Snapshot from `fit_snapshot`.
        features: Rows to prepare.

    Returns:
        Values and flags, then ranks if the snapshot has rank buckets, as
        channels per cell, the kept columns first in `snapshot.columns` order.
    """
    filled = fill(snapshot, features)
    eps = jnp.finfo(snapshot.scale.dtype).eps
    values = jnp.clip((filled - snapshot.center) / (snapshot.scale + eps), -CLIP, CLIP)
    channels = [values, missing_flags(features)]
    if snapshot.ranks is not None:
        channels.append(ranks(snapshot.ranks, filled).astype(values.dtype))
    return jnp.stack(channels, axis=-1)[:, snapshot.columns.order]
