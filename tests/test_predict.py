from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.tabicl.tiny import QUANTILES, tiny_tabicl
from tests.utils.predict import predict


def _table() -> tuple[jnp.ndarray, jnp.ndarray]:
    generator = np.random.default_rng(0)
    features = jnp.asarray(generator.normal(size=(24, 5)), dtype=jnp.float32)
    queries = jnp.asarray(generator.normal(size=(4, 5)), dtype=jnp.float32)
    return features, queries


def test_predict_takes_the_class_count_from_the_largest_label() -> None:
    features, queries = _table()
    targets = jnp.arange(24, dtype=jnp.int32) % 3

    probabilities = predict(
        tabicl_functions, tiny_tabicl(regression=False), features, targets, queries
    )

    assert probabilities.shape == (4, 3)
    np.testing.assert_allclose(np.asarray(probabilities).sum(axis=-1), 1.0, atol=1e-5)


def test_predict_returns_sorted_quantiles_for_regressors() -> None:
    features, queries = _table()

    quantiles = predict(
        tabicl_functions,
        tiny_tabicl(regression=True),
        features,
        features.sum(axis=1),
        queries,
    )

    assert quantiles.shape == (4, QUANTILES)
    assert np.all(np.diff(np.asarray(quantiles), axis=-1) >= 0)


@pytest.mark.parametrize("label", [-1, 10], ids=["negative", "beyond-slots"])
def test_predict_rejects_labels_the_model_cannot_serve(label: int) -> None:
    features, queries = _table()
    targets = jnp.zeros(24, dtype=jnp.int32).at[0].set(label)

    with pytest.raises(ValueError, match="class labels"):
        predict(
            tabicl_functions, tiny_tabicl(regression=False), features, targets, queries
        )
