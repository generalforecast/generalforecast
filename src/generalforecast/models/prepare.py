# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row masks, masked statistics and column orders every model's preprocessing shares."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int


def valid_rows(
    rows: int, num_rows: Int[Array, ""] | int | None
) -> Bool[Array, " rows"]:
    """Which of `rows` rows are real: the first `num_rows`, or all if None."""
    if num_rows is None:
        return jnp.ones(rows, dtype=bool)
    return jnp.arange(rows) < num_rows


def batch_length(num_rows: Int[Array, ""] | int | None) -> Int[Array, " 1"] | None:
    """A row count as the length of a batch of one table, or None for all rows."""
    return None if num_rows is None else jnp.reshape(num_rows, (1,))


def _sum(values: Array) -> Array:
    """Sum over the leading axis, adding halves in turn, so rounding grows as its log."""
    if values.shape[0] == 0:
        return jnp.zeros(values.shape[1:], values.dtype)
    while values.shape[0] > 1:
        half = values.shape[0] // 2
        halves = values[:half] + values[half : 2 * half]
        if values.shape[0] % 2:
            halves = halves.at[0].add(values[-1])
        values = halves
    return values[0]


class Moments(NamedTuple):
    """Mean and standard deviation over the leading axis.

    Attributes:
        center: The mean, rounded to the values' dtype.
        center_low: What that rounding left out, so `(values - center) -
            center_low` centers values as a float64 mean does.
        deviation: The standard deviation.
    """

    center: Array
    center_low: Array
    deviation: Array


def masked_stats(values: Array, present: Bool[Array, "..."], ddof: int = 0) -> Moments:
    """Mean and standard deviation over the leading axis of the values `present` marks.

    Upstream packages compute them in float64. Without it on the device, two
    passes of pairwise sums match those to float32's precision: one of the
    differences from a present value, then one of the residuals from that
    first mean and of their squares.

    Args:
        values: Values, with any shape.
        present: Which values count, broadcastable to `values`.
        ddof: Delta degrees of freedom of the deviation; 0 for the population's.

    Returns:
        The moments, 0 where no value is present or too few are.
    """
    # Differences from one present value are exact for values near it, so a
    # column far from zero keeps as many digits as one near it.
    first = jnp.broadcast_to(jnp.argmax(present, axis=0), values.shape[1:])
    shift = jnp.take_along_axis(values, first[None], axis=0)[0]
    present = jnp.broadcast_to(present, values.shape)
    count = jnp.sum(present, axis=0)
    shift = jnp.where(count > 0, shift, 0)
    rows = jnp.maximum(count, 1).astype(values.dtype)
    shifted = jnp.where(present, values - shift, 0)
    offset = _sum(shifted) / rows
    residuals = jnp.where(present, shifted - offset, 0)
    correction = _sum(residuals) / rows
    # The squares about the mean, which is `correction` from the first one.
    squares = _sum(residuals**2) - count * correction**2
    variance = jnp.maximum(squares, 0) / jnp.maximum(count - ddof, 1)
    offset = offset + correction
    # Two-sum: the rounded mean and its exact remainder.
    center = shift + offset
    moved = center - shift
    low = (shift - (center - moved)) + (offset - moved)
    return Moments(center, low, jnp.sqrt(variance).astype(values.dtype))


def precise_mean(values: Array, present: Bool[Array, "..."]) -> Array:
    """`masked_mean` to float64's accuracy, for upstreams that compute in float64."""
    return masked_stats(values, present).center


def precise_moments(
    values: Array, present: Bool[Array, "..."], ddof: int = 0
) -> tuple[Array, Array]:
    """`masked_moments` to float64's accuracy, for upstreams that compute in float64."""
    stats = masked_stats(values, present, ddof)
    return stats.center, stats.deviation


def masked_mean(values: Array, present: Bool[Array, "..."]) -> Array:
    """Mean over the leading axis of the values `present` marks; 0 where none are.

    Args:
        values: Values, with any shape.
        present: Which values count, broadcastable to `values`.

    Returns:
        The mean over the leading axis, in the values' dtype.
    """
    present = jnp.broadcast_to(present, values.shape)
    count = jnp.sum(present, axis=0)
    # Summing differences from one present value keeps the sum small, so a
    # column far from zero keeps as many digits as one near it.
    first = jnp.argmax(present, axis=0)
    shift = jnp.take_along_axis(values, first[None], axis=0)[0]
    shift = jnp.where(count > 0, shift, 0)
    total = jnp.sum(jnp.where(present, values - shift, 0), axis=0)
    return (shift + total / jnp.maximum(count, 1)).astype(values.dtype)


def masked_moments(
    values: Array, present: Bool[Array, "..."], ddof: int = 0
) -> tuple[Array, Array]:
    """Mean and standard deviation over the leading axis of the values `present` marks.

    Args:
        values: Values, with any shape.
        present: Which values count, broadcastable to `values`.
        ddof: Delta degrees of freedom of the deviation; 0 for the population's.

    Returns:
        The mean and deviation, 0 where no value is present or too few are.
    """
    present = jnp.broadcast_to(present, values.shape)
    mean = masked_mean(values, present)
    count = jnp.sum(present, axis=0)
    squares = jnp.sum(jnp.where(present, (values - mean) ** 2, 0), axis=0)
    variance = squares / jnp.maximum(count - ddof, 1)
    return mean, jnp.sqrt(variance).astype(values.dtype)


class Columns(NamedTuple):
    """The columns a model reads, first, then the ones it leaves out.

    Upstream packages drop columns that are constant in the context. Here the
    table keeps its width: those columns move to the end, and every step whose
    result depends on the column count reads only the first `count`.

    Attributes:
        order: The kept columns in the table's order, or the order a model's
            upstream package puts them in, then the left-out ones.
        count: How many columns are kept.
    """

    order: Int[Array, " features"]
    count: Int[Array, ""]


def constant_columns(
    values: Float[Array, " rows features"], valid: Bool[Array, " rows"]
) -> Bool[Array, " features"]:
    """Columns whose valid rows all equal the first, or are all NaN.

    NaN equals nothing, so a column of NaN and one value is not constant.

    Args:
        values: Rows, with NaN for missing values.
        valid: Which rows count, a leading run of them.

    Returns:
        Whether each column is constant.
    """
    differs = jnp.any(valid[:, None] & (values != values[0]), axis=0)
    present = jnp.any(valid[:, None] & ~jnp.isnan(values), axis=0)
    return ~differs | ~present


def kept_first(
    dropped: Bool[Array, " features"],
    keep_one: bool = True,
    later: Bool[Array, " features"] | None = None,
) -> Columns:
    """Orders the kept columns first, as dropping the others would leave them.

    Args:
        dropped: Which columns to leave out.
        keep_one: Keep the first column when every column would be left out,
            as TabICL does, rather than none.
        later: Kept columns to put after the other kept ones, or None to keep
            the table's order.

    Returns:
        The column order and how many are kept.
    """
    if keep_one:
        dropped = dropped.at[0].set(dropped[0] & ~jnp.all(dropped))
    rank = 2 * dropped.astype(jnp.int32)
    if later is not None:
        rank = rank + later.astype(jnp.int32)
    order = jnp.argsort(rank, stable=True).astype(jnp.int32)
    return Columns(order, jnp.sum(~dropped, dtype=jnp.int32))
