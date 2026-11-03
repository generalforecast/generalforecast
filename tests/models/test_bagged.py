# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bags: weight sets run together, their predictions averaged."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.bagged import Bagged, bagged_functions
from generalforecast.models.contract import TabularFunctions
from generalforecast.models.ensemble import Ensemble, ensemble_functions
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.conformance.cases import table
from tests.models.tabicl.tiny import tiny_tabicl
from tests.utils.predict import predict

BAGGED = bagged_functions(tabicl_functions)


def _run(functions, model, data) -> np.ndarray:
    return np.asarray(
        predict(functions, model, data.features, data.targets, data.queries)
    )


def _stacked(*models):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)


@pytest.mark.parametrize("regression", [False, True])
def test_one_bag_predicts_as_its_weights_do(regression: bool) -> None:
    model, data = tiny_tabicl(regression=regression), table(regression=regression)
    np.testing.assert_allclose(
        _run(BAGGED, Bagged(None, _stacked(model), 1), data),
        _run(tabicl_functions, model, data),
        atol=1e-5,
    )
    assert isinstance(BAGGED, TabularFunctions)


def test_bags_average_probabilities_over_the_tasks_classes() -> None:
    data = table(regression=False)
    models = [tiny_tabicl(regression=False, seed=seed) for seed in (0, 1, 2)]
    each = np.stack([_run(tabicl_functions, model, data) for model in models])
    expected = each.mean(axis=0)
    expected = expected / expected.sum(axis=-1, keepdims=True)
    bagged = _run(BAGGED, Bagged(None, _stacked(*models), 3), data)
    np.testing.assert_allclose(bagged, expected, atol=1e-5)
    np.testing.assert_allclose(bagged.sum(axis=-1), 1.0, atol=1e-5)


def test_bags_average_quantiles_and_keep_them_sorted() -> None:
    data = table(regression=True)
    models = [tiny_tabicl(regression=True, seed=seed) for seed in (0, 1)]
    each = np.stack([_run(tabicl_functions, model, data) for model in models])
    bagged = _run(BAGGED, Bagged(None, _stacked(*models), 2), data)
    np.testing.assert_allclose(bagged, np.sort(each.mean(axis=0), axis=-1), atol=1e-4)
    assert np.all(np.diff(bagged, axis=-1) >= 0)


def test_bags_share_one_copy_of_what_they_have_in_common() -> None:
    data = table(regression=False)
    base, other = tiny_tabicl(regression=False), tiny_tabicl(regression=False, seed=1)
    # Each bag's own part is a difference from the base; adding it gives its weights.
    zero = jax.tree.map(jnp.zeros_like, base)
    delta = jax.tree.map(jnp.subtract, other, base)

    def plus(shared, own):
        return jax.tree.map(jnp.add, shared, own)

    bagged = Bagged(base, _stacked(zero, delta), 2, plus)
    np.testing.assert_allclose(
        _run(BAGGED, bagged, data),
        _run(BAGGED, Bagged(None, _stacked(base, other), 2), data),
        atol=1e-5,
    )
    # The base is held once, not once per bag.
    assert jax.tree.leaves(bagged.shared)[0].shape == jax.tree.leaves(base)[0].shape


def test_an_ensemble_of_bags_runs_each_estimator_on_every_bag() -> None:
    data = table(regression=False)
    models = [tiny_tabicl(regression=False, seed=seed) for seed in (0, 1)]
    functions = ensemble_functions(BAGGED)
    probabilities = _run(
        functions, Ensemble(Bagged(None, _stacked(*models), 2), 2), data
    )
    assert probabilities.shape == _run(tabicl_functions, models[0], data).shape
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5)
