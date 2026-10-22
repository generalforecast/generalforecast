# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TabICL's preprocessing, as its own inference pipeline runs it without a normalizer.

Upstream imputes missing values with column means, standardizes with a small
epsilon on the deviation and clips to 100 deviations, then fits an outlier
remover: values past four deviations are set aside, the moments are refit
without them, and every value is soft-clipped to the refit bounds with a
logarithm. Upstream first drops columns with one value on the training rows
after imputation, keeping the first column if all have one; here they move
last and the network leaves them out, so the table keeps its width.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from generalforecast.models.prepare import (
    Columns,
    constant_columns,
    kept_first,
    masked_stats,
    precise_mean,
    precise_moments,
    valid_rows,
)

# Standardized values are clipped here, as upstream's scaler clips them.
CLIP = 100.0
# Values this many deviations from the mean are outliers.
OUTLIER = 4.0
# Added to deviations, so a constant column scales rather than divides by zero.
EPSILON = 1e-6


class NumericSnapshot(NamedTuple):
    """What a table's preprocessing was fit to, plus the target scale for regression.

    Queries reuse the snapshot, and appends do not refit it.

    Attributes:
        center: Each column's mean, which also fills its missing cells.
        center_low: What rounding the mean to float32 left out, so values near
            it center as upstream's float64 mean centers them.
        scale: Each column's deviation plus `EPSILON`.
        lower: Each standardized column's lower soft bound.
        upper: Each standardized column's upper soft bound.
        target_center: The targets' mean, or None for classification.
        target_scale: The targets' deviation, or 1 where they barely vary.
        columns: The columns upstream keeps, first, then the constant ones.
    """

    center: Float[Array, " features"]
    center_low: Float[Array, " features"]
    scale: Float[Array, " features"]
    lower: Float[Array, " features"]
    upper: Float[Array, " features"]
    target_center: Float[Array, ""] | None
    target_scale: Float[Array, ""] | None
    columns: Columns


def _finite(values: Float[Array, " rows features"]) -> Float[Array, " rows features"]:
    """Replaces infinities with NaN so they are imputed like missing values."""
    return jnp.where(jnp.isfinite(values), values, jnp.nan)


def _target_scale(deviation: Array) -> Array:
    """The deviation, or 1 where values barely vary, as scikit-learn scales targets."""
    return jnp.where(deviation > 10 * jnp.finfo(deviation.dtype).eps, deviation, 1)


def _standardize(
    snapshot: NumericSnapshot, features: Float[Array, " rows features"]
) -> Float[Array, " rows features"]:
    """Imputes, standardizes and clips rows, before the outlier bounds."""
    cleaned = _finite(features)
    centered = (cleaned - snapshot.center) - snapshot.center_low
    # A cell imputed with the mean centers to 0.
    centered = jnp.where(jnp.isnan(cleaned), 0, centered)
    return jnp.clip(centered / snapshot.scale, -CLIP, CLIP)


def kept_columns(
    features: Float[Array, " rows features"],
    num_rows: Int[Array, ""] | int | None = None,
) -> Columns:
    """The columns TabICL keeps: those with two values after mean imputation.

    Every column stays when the context has one row, and the first when none
    would, as upstream does. Imputing a column of NaN and one value gives that
    value exactly here, so the column goes. Upstream imputes in the input's
    dtype, and in float32 its mean of a non-integer value is often off by a
    rounding step, which leaves two values and keeps the column; given
    float64, it drops it as here.

    Args:
        features: Context rows, with NaN for missing values.
        num_rows: Real rows at the start; the rest are padding. None means all.

    Returns:
        The kept columns first, and how many there are.
    """
    valid = valid_rows(features.shape[0], num_rows)
    cleaned = _finite(features)
    means = precise_mean(cleaned, valid[:, None] & ~jnp.isnan(cleaned))
    imputed = jnp.where(jnp.isnan(cleaned), means, cleaned)
    return kept_first(constant_columns(imputed, valid) & (jnp.sum(valid) > 1))


def fit_numeric_snapshot(
    features: Float[Array, " rows features"],
    targets: Float[Array, " rows"] | None = None,
    num_rows: Int[Array, ""] | int | None = None,
) -> NumericSnapshot:
    """Fits imputation, standardization, the outlier bounds and the target scale.

    Args:
        features: Context rows, with NaN for missing values.
        targets: Regression targets, or None for classification.
        num_rows: Real rows at the start; the rest are padding. None means all.

    Returns:
        The snapshot.
    """
    valid = valid_rows(features.shape[0], num_rows)
    cleaned = _finite(features)
    present = valid[:, None] & ~jnp.isnan(cleaned)
    # Imputing with the mean keeps the mean, so it centers the column too, and
    # the imputed cells count as rows without adding to the squares.
    stats = masked_stats(cleaned, present)
    observed = jnp.sum(present, axis=0) / jnp.maximum(jnp.sum(valid), 1)
    scale = stats.deviation * jnp.sqrt(observed) + EPSILON
    centered = jnp.where(present, (cleaned - stats.center) - stats.center_low, 0)
    standardized = jnp.clip(centered / scale, -CLIP, CLIP)
    # Two passes: moments without the values the first pass puts past the bounds.
    mean, deviation = precise_moments(standardized, valid[:, None], ddof=1)
    deviation = jnp.maximum(deviation, EPSILON)
    inside = (
        valid[:, None]
        & (standardized >= mean - OUTLIER * deviation)
        & (standardized <= mean + OUTLIER * deviation)
    )
    mean, deviation = precise_moments(standardized, inside, ddof=1)
    deviation = jnp.maximum(deviation, EPSILON)
    target_center = target_scale = None
    if targets is not None:
        target_center, target_deviation = precise_moments(targets, valid)
        target_scale = _target_scale(target_deviation)
    return NumericSnapshot(
        center=stats.center,
        center_low=stats.center_low,
        scale=scale,
        lower=mean - OUTLIER * deviation,
        upper=mean + OUTLIER * deviation,
        target_center=target_center,
        target_scale=target_scale,
        columns=kept_columns(features, num_rows),
    )


def apply_numeric_snapshot(
    snapshot: NumericSnapshot,
    features: Float[Array, " rows features"],
) -> Float[Array, " rows features"]:
    """Imputes, standardizes and clips rows, then soft-clips them to the outlier bounds.

    Args:
        snapshot: Snapshot from `fit_numeric_snapshot`.
        features: Rows to prepare.

    Returns:
        Prepared rows, the kept columns first in `snapshot.columns` order.
    """
    standardized = _standardize(snapshot, features)
    magnitude = jnp.log1p(jnp.abs(standardized))
    clipped = jnp.maximum(snapshot.lower - magnitude, standardized)
    prepared = jnp.minimum(snapshot.upper + magnitude, clipped)
    return prepared[:, snapshot.columns.order]
