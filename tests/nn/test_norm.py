# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.nn.norm import (
    LayerNormParameters,
    RMSNormParameters,
    layer_norm,
    normalize,
    rms_norm,
)


def test_rms_norm_matches_the_pytorch_formula() -> None:
    values = jax.random.normal(jax.random.key(0), (3, 5, 8), dtype=jnp.float32)
    weight = jax.random.normal(jax.random.key(1), (8,), dtype=jnp.float32)

    actual = rms_norm(RMSNormParameters(weight=weight), values)

    # torch.nn.RMSNorm with eps=None: x / sqrt(mean(x^2) + finfo.eps) * weight
    array = np.asarray(values, dtype=np.float64)
    eps = np.finfo(np.float32).eps
    expected = array / np.sqrt(np.mean(array**2, axis=-1, keepdims=True) + eps)
    expected = expected * np.asarray(weight, dtype=np.float64)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_normalize_dispatches_on_the_parameter_type() -> None:
    values = jax.random.normal(jax.random.key(0), (4, 8))
    weight = jnp.linspace(0.5, 1.5, 8)
    rms = RMSNormParameters(weight=weight)
    layer = LayerNormParameters(weight=weight, bias=jnp.zeros(8))

    np.testing.assert_array_equal(
        np.asarray(normalize(rms, values)), np.asarray(rms_norm(rms, values))
    )
    np.testing.assert_array_equal(
        np.asarray(normalize(layer, values)), np.asarray(layer_norm(layer, values))
    )


def test_normalize_dispatches_under_jit() -> None:
    values = jax.random.normal(jax.random.key(0), (4, 8))
    rms = RMSNormParameters(weight=jnp.ones(8))

    np.testing.assert_allclose(
        np.asarray(jax.jit(normalize)(rms, values)),
        np.asarray(rms_norm(rms, values)),
        rtol=1e-6,
    )
