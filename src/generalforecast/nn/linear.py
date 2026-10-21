# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import jax
import jax.nn as jnn
from jax import Array
from jaxtyping import Float

from generalforecast.nn.norm import Norm, normalize


class LinearParameters(NamedTuple):
    """Weight in PyTorch's (output, input) layout and an optional bias."""

    weight: Float[Array, " output input"]
    bias: Float[Array, " output"] | None


class MLPParameters(NamedTuple):
    """Two linear layers with an exact GELU between them."""

    input: LinearParameters
    output: LinearParameters


def linear(
    parameters: LinearParameters,
    values: Float[Array, "*batch input"],
) -> Float[Array, "*batch output"]:
    """Applies `values @ weight.T + bias`.

    Args:
        parameters: Weight and optional bias.
        values: Inputs with features on the last axis.

    Returns:
        Outputs with `output` features on the last axis.
    """
    if parameters.bias is None:
        return values @ parameters.weight.T
    return values @ parameters.weight.T + parameters.bias


def mlp(
    parameters: MLPParameters, values: Float[Array, "*batch input"]
) -> Float[Array, "*batch output"]:
    """Applies linear, exact GELU, then linear.

    Args:
        parameters: The two linear layers.
        values: Inputs with features on the last axis.

    Returns:
        Outputs of the second linear layer.
    """
    hidden = linear(parameters.input, values)
    activated = jnn.gelu(hidden, approximate=False)
    return linear(parameters.output, activated)


class GatedMLPParameters(NamedTuple):
    """A SwiGLU feed-forward layer: gate, up and down projections."""

    gate: LinearParameters
    up: LinearParameters
    down: LinearParameters


def gated_mlp(
    parameters: GatedMLPParameters, values: Float[Array, "*batch input"]
) -> Float[Array, "*batch output"]:
    """Applies `down(silu(gate(values)) * up(values))`.

    Args:
        parameters: The three projections.
        values: Inputs with features on the last axis.

    Returns:
        Outputs of the down projection.
    """
    gated = jnn.silu(linear(parameters.gate, values)) * linear(parameters.up, values)
    return linear(parameters.down, gated)


class ResidualMLPParameters(NamedTuple):
    """A pre-norm MLP whose output is added back to its input."""

    norm: Norm
    mlp: MLPParameters


def residual_mlp(
    parameters: ResidualMLPParameters, values: Float[Array, "*batch model"]
) -> Float[Array, "*batch model"]:
    """Applies `values + mlp(norm(values))`.

    Args:
        parameters: The norm and the MLP.
        values: Inputs with features on the last axis.

    Returns:
        The updated values.
    """
    return values + mlp(parameters.mlp, normalize(parameters.norm, values))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResidualBlockParameters:
    """A hidden layer with an activation, an output layer and a linear skip, summed.

    `activation` is static: ReLU or SiLU, as the model was trained.
    """

    hidden: LinearParameters
    output: LinearParameters
    residual: LinearParameters
    activation: Literal["relu", "silu"] = field(metadata={"static": True})


def residual_block(
    parameters: ResidualBlockParameters, values: Float[Array, "*batch input"]
) -> Float[Array, "*batch output"]:
    """Applies `output(activation(hidden(values))) + residual(values)`.

    Args:
        parameters: The three linear layers and the activation.
        values: Inputs with features on the last axis.

    Returns:
        Outputs with `output` features on the last axis.
    """
    activate = jnn.relu if parameters.activation == "relu" else jnn.silu
    hidden = activate(linear(parameters.hidden, values))
    return linear(parameters.output, hidden) + linear(parameters.residual, values)
