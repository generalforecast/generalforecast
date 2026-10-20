"""Quantiles of a distribution over buckets, as bucket regression heads predict."""

import math
from statistics import NormalDist

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erfinv
from jaxtyping import Float

# Regressors return quantiles at levels (k + 1) / 1000, as the other models do.
QUANTILE_LEVELS = 999

# A half-normal's median is this many scales from its origin.
_HALF_NORMAL_MEDIAN = NormalDist().inv_cdf(0.75)


def _half_normal_quantile(width: Array, level: Array) -> Array:
    """The `level` quantile of the half-normal with half its mass within `width`."""
    return width / _HALF_NORMAL_MEDIAN * math.sqrt(2) * erfinv(level)


def bucket_quantiles(
    logits: Float[Array, "*batch buckets"],
    borders: Float[Array, " borders"],
    levels: Float[Array, " levels"] | None = None,
) -> Float[Array, "*batch levels"]:
    """Quantiles of the distribution the logits put over the buckets.

    Mass is uniform within a bucket, except in the two outer buckets, which
    become half-normal tails reaching past the outer borders.

    Args:
        logits: Logits over the buckets between consecutive borders.
        borders: Bucket borders, ascending; one more than the buckets.
        levels: Quantile levels in (0, 1), or None for the `QUANTILE_LEVELS`
            levels regressors return.

    Returns:
        One value per level.
    """
    if levels is None:
        levels = jnp.arange(1, QUANTILE_LEVELS + 1, dtype=jnp.float32) / (
            QUANTILE_LEVELS + 1
        )
    probabilities = jnn.softmax(logits.astype(jnp.float32), axis=-1)
    cumulative = jnp.cumsum(probabilities, axis=-1)
    # Mass at or right of each bucket, summed from the small end, so upper levels
    # do not subtract numbers close to 1.
    after = jnp.flip(jnp.cumsum(jnp.flip(probabilities, -1), axis=-1), -1)
    widths = borders[1:] - borders[:-1]
    buckets = logits.shape[-1]
    upper = levels > 0.5

    def one(cumulative: Array, after: Array, probabilities: Array) -> Array:
        index = jnp.where(
            upper,
            # The last bucket with more than `1 - level` at or right of it.
            buckets - 1 - jnp.searchsorted(jnp.flip(after), 1 - levels, side="right"),
            jnp.searchsorted(cumulative, levels),
        )
        index = jnp.clip(index, 0, buckets - 1)
        mass = probabilities[index]
        below = levels - (cumulative[index] - mass)
        above = (1 - levels) - (after[index] - mass)
        within = jnp.where(upper, 1 - above / mass, below / mass)
        within = jnp.clip(within, 0, 1)
        inside = borders[index] + widths[index] * within
        left = borders[1] - _half_normal_quantile(widths[0], 1 - within)
        right = borders[-2] + _half_normal_quantile(widths[-1], within)
        values = jnp.where(index == 0, left, inside)
        return jnp.where(index == buckets - 1, right, values)

    flat = (
        cumulative.reshape(-1, buckets),
        after.reshape(-1, buckets),
        probabilities.reshape(-1, buckets),
    )
    return jax.vmap(one)(*flat).reshape(*logits.shape[:-1], levels.shape[0])
