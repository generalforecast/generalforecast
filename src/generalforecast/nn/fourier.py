# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from generalforecast.nn.linear import LinearParameters, linear
from generalforecast.nn.norm import LayerNormParameters, layer_norm


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FourierCellParameters:
    """Fourier features of grouped values, plus a projection of their other channels.

    `rank_frequencies` is static.
    """

    frequencies: Float[Array, " group frequencies"]
    fourier: LinearParameters
    channels: LinearParameters
    norm: LayerNormParameters
    rank_frequencies: int = field(metadata={"static": True})


def fourier_features(
    values: Float[Array, "*batch"],
    frequencies: Float[Array, "*members frequencies"],
    dtype: jnp.dtype,
) -> Float[Array, "*batch features"]:
    """Sines, then cosines, of each value times each of its frequencies.

    Phases are computed in float32, which the features then leave for `dtype`.

    Args:
        values: Cell values.
        frequencies: Frequencies, broadcast against the values' trailing axes.
        dtype: Dtype of the features.

    Returns:
        Twice as many features as frequencies per value.
    """
    phase = values.astype(jnp.float32)[..., None] * frequencies.astype(jnp.float32)
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1).astype(dtype)


def rank_features(
    ranks: Float[Array, "*batch"], num_frequencies: int
) -> Float[Array, "*batch features"]:
    """Sines and cosines of `pi * k * rank` for k = 1 .. num_frequencies."""
    k = jnp.arange(1, num_frequencies + 1, dtype=ranks.dtype)
    phase = ranks[..., None] * (math.pi * k)
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def fourier_cells(
    parameters: FourierCellParameters, grouped: Float[Array, "... width"]
) -> Float[Array, "... model"]:
    """Embeds grouped cells, whose last axis is the channels of each group member.

    Phases and the channel projection are computed in float32.

    Args:
        parameters: Cell embedding parameters.
        grouped: Values, then other channels, then ranks, each `group` wide.

    Returns:
        One embedding per cell.
    """
    group = parameters.frequencies.shape[0]
    values, ranks = grouped[..., :group], grouped[..., -group:]
    features = fourier_features(values, parameters.frequencies, grouped.dtype)
    fourier = linear(parameters.fourier, features.sum(axis=-2))
    lifted = rank_features(ranks, parameters.rank_frequencies)
    channels = jnp.concatenate(
        [grouped[..., :-group], lifted.reshape(*ranks.shape[:-1], -1)], axis=-1
    )
    weight = parameters.channels.weight.astype(jnp.float32)
    projected = channels.astype(jnp.float32) @ weight.T
    if parameters.channels.bias is not None:
        projected = projected + parameters.channels.bias.astype(jnp.float32)
    return layer_norm(
        parameters.norm, (fourier.astype(jnp.float32) + projected)
    ).astype(grouped.dtype)
