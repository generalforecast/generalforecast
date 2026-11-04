# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bucket quantiles: their gradients, which fine-tuning a bucket regressor follows."""

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.models.buckets import bucket_quantiles

BORDERS = jnp.asarray([-1.0, 0.0, 1.0, 2.0, 3.0])


def test_levels_on_bucket_edges_have_finite_gradients():
    # Even mass: each level falls exactly at the top of an inner bucket,
    # where the upper tail's own quantile would be infinite.
    levels = jnp.asarray([0.25, 0.5, 0.75], jnp.float32)

    def total(logits):
        return bucket_quantiles(logits, BORDERS, levels).sum()

    np.testing.assert_allclose(
        bucket_quantiles(jnp.zeros(4), BORDERS, levels), [0.0, 1.0, 2.0]
    )
    for gradients in (
        jax.grad(total)(jnp.zeros(4)),
        jax.jit(jax.grad(total))(jnp.zeros(4)),
    ):
        assert bool(jnp.all(jnp.isfinite(gradients)))
