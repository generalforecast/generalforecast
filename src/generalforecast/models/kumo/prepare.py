# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.prepare import (
    Columns,
    Standardizer,
    constant_columns,
    fit_standardizer,
    kept_first,
    precise_moments,
    soft_clip,
    standardize,
)

# The most distinct context values a categorical column keeps exactly: 64 KiB
# of float32 per column of a table with categorical columns.
CATEGORIES = 16384


class KumoCategories(NamedTuple):
    """Each column's distinct context values in ascending order, and how many there are.

    Past `CATEGORIES` distinct values, a column keeps evenly spaced ones, and a
    value between two of them takes an interpolated rank, so it no longer
    matches NVIDIA's exactly; `crowded` marks those columns. Unused slots hold
    +inf.
    """

    values: Float[Array, " features categories"]
    count: Int[Array, " features"]

    @property
    def crowded(self) -> Bool[Array, " features"]:
        """Which columns have more distinct values than they keep."""
        return self.count > CATEGORIES


def _slot_ranks(count: Int[Array, ""]) -> Int[Array, " categories"]:
    """The rank of the value each slot keeps, for `count` distinct values."""
    slots = jnp.arange(CATEGORIES)
    # slots * (count - 1) / (CATEGORIES - 1), rounded, in integers that stay
    # within int32 for any count.
    whole, part = jnp.divmod(count - 1, CATEGORIES - 1)
    spread = slots * whole + (slots * part + (CATEGORIES - 1) // 2) // (CATEGORIES - 1)
    return jnp.where(count <= CATEGORIES, slots, spread)


class KumoSnapshot(NamedTuple):
    """Feature standardization and outlier bounds, plus the target scale for regression.

    Missing values stay NaN: the network imputes them itself. Categorical
    columns first become their codes' ranks among the context's, as NVIDIA's
    AlignCategories does, so they also hold the context's categories. `columns`
    puts first the columns Kumo Tabular's DropConstantColumns keeps, numerical
    ones before categorical ones.
    """

    features: Standardizer
    target_center: Float[Array, ""] | None
    target_scale: Float[Array, ""] | None
    columns: Columns
    categories: KumoCategories | None = None


def kumo_columns(
    features: Float[Array, " rows features"],
    valid: Bool[Array, " rows"],
    categorical: Bool[Array, " features"] | None = None,
) -> Columns:
    """The columns Kumo Tabular keeps: those with two values, NaN one of them.

    Every column stays when the context has one row. When none would,
    upstream predicts from no columns at all; here the first stays, so a
    table with every column constant predicts from that one.

    Kept numerical columns come first and kept categorical ones after them, as
    upstream's StypeDispatch leaves them. Each cell's group reads the columns
    after it, so the order changes the outputs.

    Args:
        features: Context rows, with NaN for missing values.
        valid: Which rows are real context rows rather than padding.
        categorical: Which columns are categorical, or None if none are.

    Returns:
        The kept columns first, and how many there are.
    """
    constant = constant_columns(features, valid) & (jnp.sum(valid) > 1)
    return kept_first(constant, later=categorical)


def _fit_categories(
    column: Float[Array, " rows"], valid: Bool[Array, " rows"]
) -> KumoCategories:
    """One column's distinct finite values among the valid rows, and their count."""
    ordered = jnp.sort(jnp.where(valid & jnp.isfinite(column), column, jnp.inf))
    first = jnp.concatenate([jnp.ones(1, bool), ordered[1:] != ordered[:-1]])
    new = first & jnp.isfinite(ordered)
    count = jnp.sum(new)
    distinct = jnp.sort(jnp.where(new, ordered, jnp.inf))
    distinct = jnp.pad(
        distinct, (0, max(CATEGORIES - len(distinct), 0)), constant_values=jnp.inf
    )
    return KumoCategories(distinct[_slot_ranks(count)], count)


def _rank(
    values: Float[Array, " categories"],
    count: Int[Array, ""],
    column: Float[Array, " rows"],
) -> Float[Array, " rows"]:
    """Each value's rank among a column's categories, or -1 if missing or unseen."""
    ranks = _slot_ranks(count)
    left = jnp.searchsorted(values, column)
    high = jnp.minimum(left, len(values) - 1)
    low = jnp.maximum(left - 1, 0)
    finite = jnp.isfinite(column)
    hit = finite & (values[high] == column)
    # Only kept values with skipped ones between them can have a value between.
    between = (
        finite
        & (left > 0)
        & (left < len(values))
        & jnp.isfinite(values[high])
        & (ranks[high] - ranks[low] > 1)
    )
    width = jnp.where(between, values[high] - values[low], 1)
    guess = ranks[low] + (column - values[low]) / width * (ranks[high] - ranks[low])
    guess = jnp.clip(jnp.round(guess), ranks[low] + 1, ranks[high] - 1)
    rank = jnp.where(hit, ranks[high], jnp.where(between, guess, -1))
    return rank.astype(column.dtype)


def align_categories(
    categories: KumoCategories,
    features: Float[Array, " rows features"],
    categorical: Bool[Array, " features"],
) -> Float[Array, " rows features"]:
    """Replaces categorical codes with their ranks; missing and unseen codes become -1.

    Ranks follow the codes' order, as NVIDIA's AlignCategories(sort_by="value")
    does for numeric codes. A table's string columns code words in the order it
    first sees them, while upstream sorts the strings themselves, so for raw
    string inputs ranks follow first-seen order rather than string order, and
    predictions differ from upstream's.

    Args:
        categories: Categories from `fit_snapshot`.
        features: Rows, with NaN for missing values.
        categorical: Which columns are categorical.

    Returns:
        The rows, with categorical columns as ranks and no NaN in them.
    """
    ranked = jax.vmap(_rank, in_axes=(0, 0, 1), out_axes=1)(
        categories.values, categories.count, features
    )
    return jnp.where(categorical, ranked, features)


def fit_snapshot(
    features: Float[Array, " rows features"],
    targets: Float[Array, " rows"] | None,
    valid: Bool[Array, " rows"],
    categorical: Bool[Array, " features"] | None = None,
) -> KumoSnapshot:
    """Fits per-column standardization and outlier bounds on finite context values.

    The bounds are Kumo Tabular's ClipSigma: the standardized values' mean and
    deviation, fit again without the values more than `SIGMA` deviations out.
    Categorical columns are fit as their ranks, with -1 for missing values.

    Args:
        features: Context rows, with NaN for missing values.
        targets: Regression targets, or None for classification.
        valid: Which rows are real context rows rather than padding.
        categorical: Which columns are categorical, or None if none are.

    Returns:
        The snapshot.
    """
    categories = None
    if categorical is not None:
        categories = jax.vmap(_fit_categories, in_axes=(1, None))(features, valid)
        features = align_categories(categories, features, categorical)
    finite = valid[:, None] & jnp.isfinite(features)
    standardizer = fit_standardizer(features, finite, finite)
    target_center = target_scale = None
    if targets is not None:
        target_center, target_deviation = precise_moments(targets, valid)
        target_scale = jnp.where(target_deviation > 0, target_deviation, 1)
    return KumoSnapshot(
        standardizer,
        target_center,
        target_scale,
        kumo_columns(features, valid, categorical),
        categories,
    )


def prepare_features(
    snapshot: KumoSnapshot,
    features: Float[Array, " rows features"],
    categorical: Bool[Array, " features"] | None = None,
) -> Float[Array, " rows features"]:
    """Standardizes and clips features, then soft-clips them to the outlier bounds.

    Past a bound, a value keeps only the logarithm of its size beyond it, as
    Kumo Tabular's ClipSigma does. NaN stays NaN and infinities clip.
    Categorical columns become ranks first.

    Args:
        snapshot: Snapshot from `fit_snapshot`.
        features: Rows to prepare.
        categorical: The mask the snapshot was fit with.

    Returns:
        Prepared rows, the kept columns first in `snapshot.columns` order.
    """
    if snapshot.categories is not None and categorical is not None:
        features = align_categories(snapshot.categories, features, categorical)
    standardized = standardize(snapshot.features, features)
    return soft_clip(snapshot.features, standardized)[:, snapshot.columns.order]
