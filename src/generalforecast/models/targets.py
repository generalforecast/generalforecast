"""Regression targets standardized for the network, and quantiles mapped back.

Every model's table state keeps a target center and scale, None for
classification, and shares these steps.
"""

from typing import Protocol

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


class TargetScale(Protocol):
    """A snapshot's target center and scale; both None for classification."""

    @property
    def target_center(self) -> Array | None:
        """The targets' mean."""
        ...

    @property
    def target_scale(self) -> Array | None:
        """The targets' standard deviation, or 1 where they barely vary."""
        ...


def scale_targets(snapshot: TargetScale, targets: Array) -> Array:
    """Standardizes regression targets. Class labels pass through unchanged.

    Args:
        snapshot: A fit snapshot.
        targets: Regression targets on the original scale, or class labels.

    Returns:
        Scaled targets, or the class labels.
    """
    if snapshot.target_center is None or snapshot.target_scale is None:
        return targets
    return (targets - snapshot.target_center) / snapshot.target_scale


def unscale_quantiles(
    snapshot: TargetScale, quantiles: Float[Array, " queries quantiles"]
) -> Float[Array, " queries quantiles"]:
    """Sorts quantiles and maps them back to the original target scale.

    Args:
        snapshot: A snapshot fit with targets.
        quantiles: Quantiles in the scaled target space.

    Returns:
        Sorted quantiles on the original scale.
    """
    if snapshot.target_center is None or snapshot.target_scale is None:
        raise ValueError("this snapshot has no target scale")
    ordered = jnp.sort(quantiles, axis=-1)
    return ordered * snapshot.target_scale + snapshot.target_center
