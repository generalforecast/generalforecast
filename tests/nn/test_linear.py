from __future__ import annotations

import jax
import numpy as np

from generalforecast.nn.linear import (
    GatedMLPParameters,
    LinearParameters,
    gated_mlp,
    linear,
)


def test_linear_without_bias_is_a_matmul() -> None:
    weight, values = jax.random.normal(jax.random.key(0), (2, 3, 3))
    bias_free = LinearParameters(weight=weight, bias=None)

    np.testing.assert_array_equal(
        np.asarray(linear(bias_free, values)), np.asarray(values @ weight.T)
    )


def test_gated_mlp_is_swiglu() -> None:
    keys = jax.random.split(jax.random.key(1), 4)
    parameters = GatedMLPParameters(
        gate=LinearParameters(jax.random.normal(keys[0], (6, 4)), None),
        up=LinearParameters(jax.random.normal(keys[1], (6, 4)), None),
        down=LinearParameters(jax.random.normal(keys[2], (4, 6)), None),
    )
    values = jax.random.normal(keys[3], (3, 4))

    gate = values @ parameters.gate.weight.T
    expected = (gate / (1 + np.exp(-gate)) * (values @ parameters.up.weight.T)) @ (
        parameters.down.weight.T
    )
    np.testing.assert_allclose(
        np.asarray(gated_mlp(parameters, values)), np.asarray(expected), rtol=1e-5
    )
