# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gradients with respect to a table's values, as a trainer's input adapter takes them.

A plane's trainer may differentiate a model's predictions with respect to
the features it reads (an input adapter in front of the base). Missing
cells and constant columns must not turn those gradients to NaN: a NaN
one fails every step of the training it reaches.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.kumo.model import kumo_functions
from generalforecast.models.limix.model import limix_functions
from generalforecast.models.mitra.model import mitra_functions
from generalforecast.models.prepare import (
    CLIP,
    Standardizer,
    masked_moments,
    masked_stats,
    standardize,
)
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.kumo.tiny import tiny_kumo
from tests.models.limix.tiny import tiny_limix
from tests.models.mitra.tiny import tiny_mitra
from tests.models.tabicl.tiny import tiny_tabicl

MODELS = {
    "tabicl": (tabicl_functions, lambda: tiny_tabicl(regression=False)),
    "kumo": (kumo_functions, lambda: tiny_kumo(regression=False)),
    "mitra": (mitra_functions, lambda: tiny_mitra(regression=False)),
    "limix": (limix_functions, lambda: tiny_limix(regression=False)),
}


def _table() -> tuple[np.ndarray, np.ndarray]:
    """GiveMeSomeCredit's shape of trouble: missing cells, a constant column, heavy tails."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(48, 4)).astype(np.float32)
    features[::5, 0] = np.nan
    features[:, 1] = 0.0
    features[3, 2] = 3e6
    targets = (rng.random(48) < 0.5).astype(np.int32)
    return features, targets


@pytest.mark.parametrize("name", sorted(MODELS))
def test_a_model_s_gradient_with_respect_to_its_values_is_finite(name: str) -> None:
    functions, build = MODELS[name]
    model = build()
    features, labels = _table()
    targets = jnp.asarray(labels)
    context = 32

    def loss(values: jax.Array) -> jax.Array:
        state, rows = functions.fit(model, values[:context], targets[:context], 2)
        cache = functions.encode(model, state, rows, targets[:context])
        outputs = functions.decode(
            model, state, cache, functions.embed(model, state, values[context:])
        )
        probabilities = functions.postprocess(model, state, outputs)
        chosen = jnp.take_along_axis(probabilities, targets[context:, None], axis=1)
        return -jnp.mean(jnp.log(chosen))

    value, gradient = jax.value_and_grad(loss)(jnp.asarray(features))
    assert np.isfinite(float(value))
    present = ~np.isnan(features)
    assert np.isfinite(np.asarray(gradient)[present]).all()


def test_the_moments_of_a_constant_column_have_a_finite_gradient() -> None:
    values = jnp.asarray([[1.0, 2.0], [1.0, 3.0], [1.0, np.nan]])
    present = ~jnp.isnan(values)

    def deviations(values: jax.Array) -> jax.Array:
        return (
            masked_stats(values, present).deviation.sum()
            + masked_moments(values, present)[1].sum()
        )

    gradient = jax.grad(deviations)(values)
    assert np.isfinite(np.asarray(gradient)[np.asarray(present)]).all()
    stats = masked_stats(values, present)
    np.testing.assert_array_equal(stats.deviation[0], 0.0)


# Soft bounds `standardize` does not read.
BOUNDS = jnp.zeros(1)


def test_standardizing_keeps_missing_cells_missing_and_infinite_ones_clipped() -> None:
    standardizer = Standardizer(
        jnp.asarray([1.0]), jnp.asarray([0.0]), jnp.asarray([2.0]), BOUNDS, BOUNDS
    )
    values = jnp.asarray([[3.0], [np.nan], [np.inf], [-np.inf], [1e9]])
    np.testing.assert_array_equal(
        standardize(standardizer, values)[:, 0], [1.0, np.nan, CLIP, -CLIP, CLIP]
    )

    def total(scale: jax.Array) -> jax.Array:
        changed = Standardizer(
            standardizer.center, standardizer.center_low, scale, BOUNDS, BOUNDS
        )
        standardized = standardize(changed, values)
        return jnp.where(jnp.isnan(standardized), 0, standardized).sum()

    # The scale's gradient reads the present, finite values alone.
    assert np.isfinite(float(jax.grad(total)(jnp.asarray([2.0]))[0]))
