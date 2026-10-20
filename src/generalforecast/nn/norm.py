from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


class LayerNormParameters(NamedTuple):
    """LayerNorm scale and optional bias."""

    weight: Float[Array, " model"]
    bias: Float[Array, " model"] | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RMSNormParameters:
    """RMSNorm with an optional scale.

    `epsilon` is static. None means float32's machine epsilon, which is PyTorch's
    default for the float32 checkpoints.
    """

    weight: Float[Array, " model"] | None
    epsilon: float | None = field(default=None, metadata={"static": True})


Norm = LayerNormParameters | RMSNormParameters


def layer_norm(
    parameters: LayerNormParameters,
    values: Float[Array, "... model"],
) -> Float[Array, "... model"]:
    """LayerNorm over the last axis, with PyTorch's epsilon of 1e-5.

    Statistics are computed in float32 and the result cast back to the input dtype.

    Args:
        parameters: Scale and optional bias.
        values: Inputs to normalize.

    Returns:
        Normalized values.
    """
    standardized = jnn.standardize(values.astype(jnp.float32), axis=-1, epsilon=1e-5)
    scaled = standardized.astype(values.dtype) * parameters.weight
    if parameters.bias is None:
        return scaled
    return scaled + parameters.bias


def rms_norm(
    parameters: RMSNormParameters,
    values: Float[Array, "... model"],
) -> Float[Array, "... model"]:
    """RMSNorm over the last axis, computed in float32.

    The checkpoints were trained in float32, so lower-precision inputs keep the
    epsilon they were trained with. The result is cast back to the input dtype.

    Args:
        parameters: Scale.
        values: Inputs to normalize.

    Returns:
        Normalized values.
    """
    full = values.astype(jnp.float32)
    epsilon = parameters.epsilon
    if epsilon is None:
        epsilon = float(jnp.finfo(jnp.float32).eps)
    mean_square = jnp.mean(jnp.square(full), axis=-1, keepdims=True)
    normalized = (full * jax.lax.rsqrt(mean_square + epsilon)).astype(values.dtype)
    if parameters.weight is None:
        return normalized
    return normalized * parameters.weight


def normalize(
    parameters: Norm, values: Float[Array, "... model"]
) -> Float[Array, "... model"]:
    """Applies LayerNorm or RMSNorm, depending on the parameter type.

    Args:
        parameters: LayerNorm or RMSNorm parameters.
        values: Inputs to normalize.

    Returns:
        Normalized values.
    """
    if isinstance(parameters, RMSNormParameters):
        return rms_norm(parameters, values)
    return layer_norm(parameters, values)
