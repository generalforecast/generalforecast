# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Finetuning: full, LoRA and mixed groups learn, merge, round-trip and stop safely."""

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


def _close(a, b, tolerance: float = 1e-6) -> None:
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(
            np.asarray(x), np.asarray(y), rtol=tolerance, atol=tolerance
        )


def _train(model, regression, **options):
    features, targets = _dataset(regression)
    defaults = {
        "steps": 40,
        "learning_rate": 1e-2,
        "context_rows": 64,
        "query_rows": 32,
        "steps_per_call": 10,
    }
    return finetune.finetune(model, features, targets, **{**defaults, **options})


def test_groups_choose_what_trains() -> None:
    _, parameters = _model(False)
    everything = finetune.plan(parameters, finetune.full())
    assert everything.full and not everything.lora
    adapted = finetune.plan(parameters, finetune.lora(rank=4))
    assert adapted.lora and not adapted.full
    mixed = finetune.plan(
        parameters, finetune.lora(rank=4, layers=[".icl."], head=".icl.decoder.")
    )
    assert mixed.full and all(name.startswith(".icl.decoder.") for name in mixed.full)
    assert mixed.lora and all(
        name.startswith(".icl.") and not name.startswith(".icl.decoder.")
        for name in mixed.lora
    )
    with pytest.raises(ValueError, match="no parameter trains"):
        finetune.plan(parameters, (finetune.Group("", "frozen"),))
    with pytest.raises(ValueError, match="adapts no linear layer"):
        finetune.plan(parameters, (finetune.Group(".nowhere", "lora", rank=4),))


def test_a_fresh_adapter_leaves_the_model_as_it_was() -> None:
    functions, parameters = _model(False)
    chosen = finetune.plan(parameters, finetune.lora(rank=4))
    shapes = {
        jax.tree_util.keystr(path): node.weight.shape
        for path, node in jax.tree_util.tree_flatten_with_path(
            parameters, is_leaf=lambda n: isinstance(n, finetune.LinearParameters)
        )[0]
        if isinstance(node, finetune.LinearParameters)
    }
    delta = finetune.Delta(
        {},
        {
            name: (jnp.ones((4, shapes[name][1])), jnp.zeros((shapes[name][0], 4)))
            for name in chosen.lora
        },
        finetune.lora(rank=4),
    )
    data = table(regression=False)
    _close(
        predict(functions, delta.merge(parameters), data),
        predict(functions, parameters, data),
    )


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("kind", ["full", "lora"])
def test_finetuning_lowers_the_training_loss(regression: bool, kind: str) -> None:
    trainable = finetune.full() if kind == "full" else finetune.lora(rank=4)
    tuned = _train(_model(regression), regression, trainable=trainable)
    losses = np.asarray(tuned.losses)
    assert tuned.stopped is None
    assert losses.shape == (40,) and np.isfinite(losses).all()
    assert losses[-10:].mean() < 0.9 * losses[:5].mean()
    functions, parameters = tuned.model
    assert functions is tabicl_functions
    assert np.isfinite(
        np.asarray(predict(functions, parameters, table(regression=regression)))
    ).all()


def test_frozen_parameters_stay_as_the_base_has_them() -> None:
    base = _model(False)
    tuned = _train(
        base,
        False,
        steps=10,
        trainable=(finetune.Group(".icl.decoder."), finetune.Group("", "frozen")),
    )
    before = dict(jax.tree_util.tree_flatten_with_path(base[1])[0])
    after = dict(jax.tree_util.tree_flatten_with_path(tuned.model[1])[0])
    changed = {
        jax.tree_util.keystr(path)
        for path in before
        if not np.array_equal(np.asarray(before[path]), np.asarray(after[path]))
    }
    assert changed and all(name.startswith(".icl.decoder.") for name in changed)


def test_a_delta_round_trips_through_its_directory(tmp_path) -> None:
    base = _model(False)
    tuned = _train(
        base,
        False,
        steps=5,
        trainable=finetune.lora(rank=4, layers=[".icl."], head=".icl.decoder."),
    )
    tuned.save(tmp_path / "delta")
    config = json.loads((tmp_path / "delta" / "finetune.json").read_text())
    assert config["base"] is None and config["lora"] == sorted(tuned.delta.lora)
    assert config["full"] == sorted(tuned.delta.full)
    with pytest.raises(ValueError, match="names no catalog model"):
        finetune.load(tmp_path / "delta")
    loaded = finetune.load(tmp_path / "delta", base)
    _close(loaded.model[1], tuned.model[1])


def test_a_delta_refuses_a_model_it_does_not_fit() -> None:
    _, parameters = _model(False)
    name = next(iter(finetune.plan(parameters, finetune.lora(rank=4)).lora))
    wrong = finetune.Delta(
        {}, {name: (jnp.zeros((4, 3)), jnp.zeros((5, 4)))}, finetune.lora(rank=4)
    )
    with pytest.raises(ValueError, match=r"adapter is \(5, 3\), the model's"):
        wrong.merge(parameters)
    missing = finetune.Delta({".nowhere": jnp.zeros(3)}, {}, finetune.full())
    with pytest.raises(ValueError, match=r"has no \.nowhere"):
        missing.merge(parameters)


def test_a_non_finite_loss_stops_at_the_last_good_weights() -> None:
    tuned = _train(_model(True), True, learning_rate=1e8, steps=30, warmup=1)
    if tuned.stopped is None:
        pytest.skip("this learning rate stayed finite on this platform")
    assert tuned.stopped.startswith("non-finite loss at step")
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree.leaves(tuned.model[1])
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    )


def test_finetuning_refuses_what_it_cannot_train() -> None:
    features, targets = _dataset(False, rows=8)
    with pytest.raises(ValueError, match="needs more than the 8 rows"):
        finetune.finetune(_model(False), features, targets, context_rows=8)
    with pytest.raises(ValueError, match="at least two classes"):
        finetune.finetune(_model(False), features, np.zeros(8))
    with pytest.raises(ValueError, match="no codes or estimators"):
        finetune.finetune("general-forecast/tabicl-v2-classifier*4", features, targets)
