from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.nn.linear import LinearParameters
from generalforecast.nn.target import ClassEncoder, ValueEncoder, embed_targets


def _linear(inputs: int) -> LinearParameters:
    weight, bias = jax.random.normal(jax.random.key(0), (2, 8, inputs))
    return LinearParameters(weight=weight, bias=bias[:, 0])


def test_class_encoder_picks_the_weight_column_of_each_label() -> None:
    layer = _linear(4)
    labels = jnp.asarray([[0, 3, 1]])

    actual = embed_targets(ClassEncoder(linear=layer), labels)

    assert layer.bias is not None
    expected = layer.weight.T[labels] + layer.bias
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)


def test_value_encoder_scales_the_weight_by_each_target() -> None:
    layer = _linear(1)
    targets = jnp.asarray([[0.5, -2.0, 3.0]])

    actual = embed_targets(ValueEncoder(linear=layer), targets)

    assert layer.bias is not None
    expected = targets[..., None] * layer.weight[:, 0] + layer.bias
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)
