# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.nn.norm import (
    RMSNormParameters,
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
