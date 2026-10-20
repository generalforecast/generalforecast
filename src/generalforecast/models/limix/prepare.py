from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.prepare import (
    Columns,
    constant_columns,
    kept_first,
    masked_mean,
    masked_moments,
    valid_rows,
)

# Standardized values are clipped here, as in LimiX's own preprocessing.
CLIP = 100.0


class LimiXSnapshot(NamedTuple):
    """Context means for filling, standardization, group scales and the target scale.

    LimiX's predictor drops columns whose values all equal the first, NaN equal
    to nothing, and columns all NaN on the train or the test rows. It decides
    on the train and test rows together, and raises only when every column is
    constant over both. Here only the context rows decide, so a column
    constant in the context but not in the queries is left out too. When that
    rule cannot stand in for upstream's, with one context row or with every
    column constant in the context, every column is kept, since upstream then
    keeps whichever vary in the queries. Statistics are of the columns in
    `columns` order, kept ones first; groups are formed from them, the rest
    read as zero padding.

    Attributes:
        fill: Each column's mean over its finite context values.
        center: Each column's mean after filling.
        scale: Each column's sample deviation after filling, or 1 for one row.
        group_scale: `sqrt(group size / columns that vary)` per feature group.
        target_center: The targets' mean, or None for classification.
        target_scale: The targets' sample deviation, or None for classification.
        columns: The columns kept, first, then the constant ones.
    """

    fill: Float[Array, " features"]
    center: Float[Array, " features"]
    scale: Float[Array, " features"]
    group_scale: Float[Array, " groups"]
    target_center: Float[Array, ""] | None
    target_scale: Float[Array, ""] | None
    columns: Columns


def kept_groups(snapshot: LimiXSnapshot) -> Int[Array, ""]:
    """How many feature groups hold kept columns, which come first."""
    size = snapshot.fill.shape[0] // snapshot.group_scale.shape[0]
    return -(-snapshot.columns.count // size)


def pad_columns(
    features: Float[Array, " rows features"], group_size: int
) -> Float[Array, " rows padded"]:
    """Appends zero columns up to a whole number of feature groups."""
    missing = -features.shape[-1] % group_size
    return jnp.pad(features, ((0, 0), (0, missing)))


def _filled(snapshot_fill: Array, features: Array) -> Array:
    return jnp.where(jnp.isfinite(features), features, snapshot_fill)


def _kept(columns: Columns, features: int) -> Bool[Array, " features"]:
    """Which of `features` columns, in `columns` order, are kept."""
    return jnp.arange(features) < columns.count


def _ordered(columns: Columns, features: Array, group_size: int) -> Array:
    """Columns in `columns` order, padded to whole groups."""
    return pad_columns(features[:, columns.order], group_size)


def fit_snapshot(
    features: Float[Array, " rows features"],
    group_size: int,
    targets: Float[Array, " rows"] | None = None,
    num_rows: Int[Array, ""] | int | None = None,
) -> LimiXSnapshot:
    """Fits LimiX's preprocessing on the context rows.

    Args:
        features: Context rows, with NaN or infinities for missing values.
        group_size: Features per group.
        targets: Regression targets, or None for classification.
        num_rows: Real rows at the start; the rest are padding. None means all.

    Returns:
        The snapshot.
    """
    rows = valid_rows(features.shape[0], num_rows)
    constant = constant_columns(features, rows)
    constant = constant & (jnp.sum(rows) > 1) & ~jnp.all(constant)
    columns = kept_first(constant)
    padded = _ordered(columns, features, group_size)
    valid = rows[:, None]
    fill = masked_mean(padded, valid & jnp.isfinite(padded))
    filled = _filled(fill, padded)
    center, deviation = masked_moments(filled, valid, ddof=1)
    count = jnp.sum(valid)
    scale = jnp.where(count == 1, 1.0, deviation + 1e-20).astype(padded.dtype)
    standardized = jnp.clip((filled - center) / scale, -CLIP, CLIP)
    standardized = jnp.where(_kept(columns, padded.shape[1]), standardized, 0)
    first = standardized[0]
    varies = jnp.any(valid & (standardized != first), axis=0)
    varying = jnp.sum(varies.reshape(-1, group_size), axis=-1)
    group_scale = jnp.sqrt(group_size / jnp.maximum(varying, 1)).astype(padded.dtype)
    target_center = target_scale = None
    if targets is not None:
        # Targets not known yet take the mask, and no part in the moments.
        labeled = valid[:, 0] & jnp.isfinite(targets)
        target_center, target_scale = masked_moments(targets, labeled, ddof=1)
        target_scale = jnp.where(target_scale == 0, 1.0, target_scale)
    return LimiXSnapshot(
        fill, center, scale, group_scale, target_center, target_scale, columns
    )


def prepare_features(
    snapshot: LimiXSnapshot, features: Float[Array, " rows features"]
) -> Float[Array, " rows groups group_size"]:
    """Filled, standardized, clipped and group-scaled values, with NaN kept as NaN.

    Infinities are filled like missing values but, unlike NaN, are not masked.
    Left-out columns read as zero, as the padding after the kept ones does.

    Args:
        snapshot: Snapshot from `fit_snapshot`.
        features: Rows to prepare.

    Returns:
        Values per feature group, NaN where a kept cell was NaN.
    """
    groups = snapshot.group_scale.shape[0]
    padded = _ordered(snapshot.columns, features, snapshot.fill.shape[0] // groups)
    standardized = jnp.clip(
        (_filled(snapshot.fill, padded) - snapshot.center) / snapshot.scale, -CLIP, CLIP
    )
    kept = _kept(snapshot.columns, padded.shape[1])
    grouped = jnp.where(kept, standardized, 0).reshape(padded.shape[0], groups, -1)
    scaled = grouped * snapshot.group_scale[:, None]
    missing = (jnp.isnan(padded) & kept).reshape(grouped.shape)
    return jnp.where(missing, jnp.nan, scaled)
