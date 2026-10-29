# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LoRA finetuning: adapters that start as the base, learn, merge and round-trip."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import finetune
from generalforecast.models.conformance import predict
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.conformance.cases import table
from tests.models.tabicl.tiny import tiny_tabicl

pytestmark = pytest.mark.slow


def _dataset(regression: bool, rows: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose target follows their first two features, learnably."""
    generator = np.random.default_rng(1)
    features = generator.normal(size=(rows, 4)).astype(np.float32)
    signal = features[:, 0] - 2 * features[:, 1]
    if regression:
        return features, (3 * signal + 10).astype(np.float32)
    return features, np.where(signal > 0, "yes", "no")


def _model(regression: bool) -> finetune.Model:
    return tabicl_functions, tiny_tabicl(regression=regression)


def _close(a, b) -> None:
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-6)


def test_a_fresh_adapter_leaves_the_model_as_it_was() -> None:
    functions, parameters = _model(False)
    shapes = finetune.chosen_layers(parameters, rank=4)
    adapter = finetune.Adapter(finetune.initial_adapter(shapes, 4, seed=0), 4, 8.0)
    data = table(regression=False)
    _close(
        predict(functions, adapter.merge(parameters), data),
        predict(functions, parameters, data),
    )


def test_layers_narrower_than_the_rank_are_left_out() -> None:
    _, parameters = _model(False)
    every = finetune.linear_layers(parameters)
    chosen = finetune.chosen_layers(parameters, rank=4)
    assert set(chosen) < set(every)
    assert all(min(shape) > 4 for shape in chosen.values())
    icl = finetune.chosen_layers(parameters, rank=4, layers=[".icl."])
    assert icl and all(".icl." in name for name in icl)
    with pytest.raises(ValueError, match="no linear layer is wider than rank 4"):
        finetune.chosen_layers(parameters, rank=4, layers=["nowhere"])


@pytest.mark.parametrize("regression", [False, True])
def test_finetuning_lowers_the_training_loss(regression: bool) -> None:
    features, targets = _dataset(regression)
    tuned = finetune.finetune(
        _model(regression),
        features,
        targets,
        steps=40,
        rank=4,
        learning_rate=1e-2,
        context_rows=64,
        query_rows=32,
    )
    losses = np.asarray(tuned.losses)
    assert losses.shape == (40,) and np.isfinite(losses).all()
    assert losses[-10:].mean() < 0.9 * losses[:5].mean()
    # The merged model is a model like any other: same calls, new weights.
    functions, parameters = tuned.model
    assert functions is tabicl_functions
    data = table(regression=regression)
    assert np.isfinite(np.asarray(predict(functions, parameters, data))).all()


def test_an_adapter_round_trips_through_its_directory(tmp_path) -> None:
    features, targets = _dataset(False)
    base = _model(False)
    tuned = finetune.finetune(
        base, features, targets, steps=3, rank=4, context_rows=48, query_rows=16
    )
    tuned.save(tmp_path / "adapter")
    config = json.loads((tmp_path / "adapter" / "adapter.json").read_text())
    assert config["base"] is None and config["rank"] == 4
    assert config["layers"] == sorted(tuned.adapter.layers)
    with pytest.raises(ValueError, match="names no catalog model"):
        finetune.load(tmp_path / "adapter")
    loaded = finetune.load(tmp_path / "adapter", base)
    _close(loaded.model[1], tuned.model[1])


def test_an_adapter_refuses_a_model_it_does_not_fit() -> None:
    _, parameters = _model(False)
    name = next(iter(finetune.chosen_layers(parameters, rank=4)))
    wrong = finetune.LoRA(jnp.zeros((4, 3)), jnp.zeros((5, 4)))
    with pytest.raises(ValueError, match=r"is \(5, 3\), the model's"):
        finetune.Adapter({name: wrong}, 4, 8.0).merge(parameters)
    missing = finetune.LoRA(jnp.zeros((4, 3)), jnp.zeros((5, 4)))
    with pytest.raises(ValueError, match="has no layer .nowhere"):
        finetune.Adapter({".nowhere": missing}, 4, 8.0).merge(parameters)


def test_finetuning_refuses_what_it_cannot_train() -> None:
    features, targets = _dataset(False, rows=8)
    with pytest.raises(ValueError, match="needs more than the 8 rows"):
        finetune.finetune(_model(False), features, targets, context_rows=8)
    with pytest.raises(ValueError, match="at least two classes"):
        finetune.finetune(_model(False), features, np.zeros(8))
    with pytest.raises(ValueError, match="no codes or estimators"):
        finetune.finetune("general-forecast/tabicl-v2-classifier*4", features, targets)
