from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Int

from generalforecast.models.prepare import (
    Columns,
    constant_columns,
    kept_first,
    masked_mean,
    masked_moments,
    valid_rows,
)
from generalforecast.models.targets import scale_targets

# Each value is ranked against the context's quantiles at levels k / 1000.
LEVELS = 999

# The scaled targets' range, 0 to 1 over the context, that a regressor's
# buckets split evenly, with half of it again on each side.
BUCKET_RANGE = (-0.5, 1.5)


class MitraSnapshot(NamedTuple):
    """Context fills and quantiles, rank moments, varying columns and target range.

    Attributes:
        fill: Each column's mean over its finite context values.
        quantiles: Each column's context quantiles at levels `1 .. 999` / 1000.
        count: Real context rows, which ranks are divided by.
        center: Each column's mean rank over the context.
        scale: Each column's rank deviation over the context, 0 if constant.
        columns: Columns that vary in the context after filling first, the
            rest after, as AutoGluon drops them.
        target_center: The targets' smallest value, or None for classification.
        target_scale: The targets' range, 1 if they are constant, or None for
            classification.
    """

    fill: Float[Array, " features"]
    quantiles: Float[Array, " levels features"]
    count: Float[Array, ""]
    center: Float[Array, " features"]
    scale: Float[Array, " features"]
    columns: Columns
    target_center: Float[Array, ""] | None
    target_scale: Float[Array, ""] | None


def _quantiles(
    filled: Float[Array, " rows features"], count: Int[Array, ""]
) -> Float[Array, " levels features"]:
    """Quantiles of the first `count` sorted rows, interpolated as `torch.quantile`."""
    levels = jnp.arange(1, LEVELS + 1, dtype=jnp.float32) / 1000
    rank = levels * (count - 1).astype(jnp.float32)
    below = jnp.floor(rank).astype(jnp.int32)
    above = jnp.ceil(rank).astype(jnp.int32)
    weight = (rank - below)[:, None].astype(filled.dtype)
    low, high = filled[below], filled[above]
    # torch.lerp's two forms, which round alike from either end.
    return jnp.where(
        weight < 0.5,
        low + weight * (high - low),
        high - (high - low) * (1 - weight),
    )


def _ranks(
    quantiles: Float[Array, " levels features"],
    count: Float[Array, ""],
    filled: Float[Array, " rows features"],
) -> Float[Array, " rows features"]:
    """Quantiles below each value, as `torch.bucketize`, over the context rows."""
    below = jax.vmap(jnp.searchsorted, in_axes=(1, 1), out_axes=1)(quantiles, filled)
    return below.astype(jnp.float32) / count


def fit_snapshot(
    features: Float[Array, " rows features"],
    targets: Float[Array, " rows"] | None = None,
    num_rows: Int[Array, ""] | int | None = None,
) -> MitraSnapshot:
    """Fits Mitra's preprocessing on the context rows.

    Args:
        features: Context rows, with NaN or infinities for missing values.
        targets: Regression targets, or None for classification.
        num_rows: Real rows at the start; the rest are padding. None means all.

    Returns:
        The snapshot.
    """
    valid = valid_rows(features.shape[0], num_rows)[:, None]
    fill = masked_mean(features, valid & jnp.isfinite(features))
    filled = jnp.where(jnp.isfinite(features), features, fill).astype(jnp.float32)
    rows = jnp.sum(valid)
    # Padding sorts last, past every real row the quantiles read.
    ordered = jnp.sort(jnp.where(valid, filled, jnp.inf), axis=0)
    quantiles = _quantiles(ordered, rows)
    count = rows.astype(jnp.float32)
    center, scale = masked_moments(_ranks(quantiles, count, filled), valid)
    # Without a column that varies, none is kept.
    columns = kept_first(constant_columns(filled, valid[:, 0]), keep_one=False)
    target_center = target_scale = None
    if targets is not None:
        # Targets not known yet take the mask, and no part in the range.
        labeled = valid[:, 0] & jnp.isfinite(targets)
        # With no labeled target, a range of 0 to 1, so nothing is infinite.
        known = jnp.any(labeled)
        low = jnp.min(jnp.where(labeled, targets, jnp.inf))
        high = jnp.max(jnp.where(labeled, targets, -jnp.inf))
        target_center = jnp.where(known, low, 0.0)
        span = jnp.where(known, high - low, 1.0)
        target_scale = jnp.where(span > 0, span, 1.0)
    return MitraSnapshot(
        fill,
        quantiles,
        count,
        center,
        scale,
        columns,
        target_center,
        target_scale,
    )


def bucket_borders(buckets: int) -> Float[Array, " borders"]:
    """Borders splitting `BUCKET_RANGE` evenly, as `torch.linspace` computes them.

    Each border rounds once from its nearer end, so context targets at the ends
    of their range fall in the same buckets as in AutoGluon's.

    Args:
        buckets: How many buckets the regressor predicts over.

    Returns:
        The `buckets + 1` borders in the scaled target space, ascending.
    """
    start, end = BUCKET_RANGE
    index = np.arange(buckets + 1)
    step = np.float64(np.float32(end - start) / np.float32(buckets))
    borders = np.where(
        index < (buckets + 1) // 2,
        start + step * index,
        end - step * (buckets - index),
    )
    return jnp.asarray(borders, jnp.float32)


def target_buckets(
    snapshot: MitraSnapshot, targets: Float[Array, " rows"], buckets: int
) -> Int[Array, " rows"]:
    """The bucket of each regression target, scaled to 0 to 1 over the context.

    Args:
        snapshot: Snapshot from `fit_snapshot`, fit with targets.
        targets: Regression targets on the original scale.
        buckets: How many buckets the regressor predicts over.

    Returns:
        Bucket ids `0 .. buckets - 1`.
    """
    scaled = scale_targets(snapshot, targets).astype(jnp.float32)
    below = jnp.searchsorted(bucket_borders(buckets), scaled, side="left")
    return jnp.clip(below - 1, 0, buckets - 1).astype(jnp.int32)


def prepare_features(
    snapshot: MitraSnapshot, features: Float[Array, " rows features"]
) -> Float[Array, " rows features"]:
    """Standardized ranks against the context, columns that vary first.

    Args:
        snapshot: Snapshot from `fit_snapshot`.
        features: Rows to prepare.

    Returns:
        Each value's standardized rank, 0 in constant columns.
    """
    filled = jnp.where(jnp.isfinite(features), features, snapshot.fill)
    ranks = _ranks(snapshot.quantiles, snapshot.count, filled.astype(jnp.float32))
    centered = ranks - snapshot.center
    safe = jnp.where(snapshot.scale > 0, snapshot.scale, 1)
    standardized = jnp.where(snapshot.scale > 0, centered / safe, 0)
    return standardized[:, snapshot.columns.order]
