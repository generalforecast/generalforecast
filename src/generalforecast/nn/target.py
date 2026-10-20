from typing import NamedTuple

import jax.nn as jnn
from jax import Array
from jaxtyping import Float, Int

from generalforecast.nn.linear import LinearParameters, linear
from generalforecast.nn.norm import LayerNormParameters, layer_norm


class ClassEncoder(NamedTuple):
    """Embeds class labels through a one-hot linear layer, one input per class slot.

    `norm`, if set, normalizes the embeddings.
    """

    linear: LinearParameters
    norm: LayerNormParameters | None = None


class ValueEncoder(NamedTuple):
    """Embeds scalar targets through a linear layer with one input.

    `norm`, if set, normalizes the embeddings.
    """

    linear: LinearParameters
    norm: LayerNormParameters | None = None


TargetEncoder = ClassEncoder | ValueEncoder


def embed_targets(
    encoder: TargetEncoder,
    targets: Int[Array, "*batch"] | Float[Array, "*batch"],
) -> Float[Array, "*batch model"]:
    """Embeds class labels or scalar targets, depending on the encoder type.

    Args:
        encoder: Class or value encoder.
        targets: Integer class labels, or scaled regression targets.

    Returns:
        Target embeddings, in the encoder's dtype.
    """
    if isinstance(encoder, ClassEncoder):
        num_classes = encoder.linear.weight.shape[1]
        one_hot = jnn.one_hot(targets, num_classes, dtype=encoder.linear.weight.dtype)
        embedded = linear(encoder.linear, one_hot)
    else:
        values = targets[..., None].astype(encoder.linear.weight.dtype)
        embedded = linear(encoder.linear, values)
    if encoder.norm is None:
        return embedded
    return layer_norm(encoder.norm, embedded)
