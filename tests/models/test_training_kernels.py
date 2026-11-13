# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A model's parameters as a trainer differentiates them (`training.for_training`).

Rematerialized layers and attention without row chunking change a step's
memory, not what the model computes: predictions and gradients match the
parameters as served.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast import training
from generalforecast.models.kumo.model import kumo_functions
from generalforecast.models.tabicl.model import tabicl_functions
from generalforecast.nn.attention import AttentionParameters
from tests.models.kumo.tiny import tiny_kumo
from tests.models.tabicl.tiny import tiny_tabicl

MODELS = {
    "tabicl": (tabicl_functions, lambda: tiny_tabicl(regression=False)),
    "kumo": (kumo_functions, lambda: tiny_kumo(regression=False)),
}
CONTEXT = 40


def _kernels(parameters: Any) -> list[Any]:
    found = jax.tree.leaves(
        parameters, is_leaf=lambda node: isinstance(node, AttentionParameters)
    )
    return [node.kernel for node in found if isinstance(node, AttentionParameters)]


def _chunked(parameters: Any, chunk: int) -> Any:
    """The parameters with a row chunk small enough that the tiny model's attention steps through rows."""

    def kernel(node: Any) -> Any:
        if not isinstance(node, AttentionParameters):
            return node
        return dataclasses.replace(
            node, kernel=dataclasses.replace(node.kernel, block_chunk=chunk)
        )

    return jax.tree.map(
        kernel, parameters, is_leaf=lambda node: isinstance(node, AttentionParameters)
    )


def _loss(functions: Any, structure: Any, features: Any, targets: Any):
    def loss(leaves: list[jax.Array]) -> jax.Array:
        model = jax.tree.unflatten(structure, leaves)
        state, rows = functions.fit(model, features[:CONTEXT], targets[:CONTEXT], 2)
        cache = functions.encode(model, state, rows, targets[:CONTEXT])
        outputs = functions.decode(
            model, state, cache, functions.embed(model, state, features[CONTEXT:])
        )
        probabilities = functions.postprocess(model, state, outputs)
        chosen = jnp.take_along_axis(probabilities, targets[CONTEXT:, None], axis=1)
        return -jnp.mean(jnp.log(chosen))

    return loss


def _table() -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(56, 4)).astype(np.float32)
    targets = (features[:, 0] + rng.normal(size=56) > 0).astype(np.int32)
    return jnp.asarray(features), jnp.asarray(targets)


def test_for_training_sets_every_kernel_and_keeps_the_arrays() -> None:
    model = _chunked(tiny_kumo(regression=False), 64)
    trained = training.for_training(model)
    assert all(k.remat and k.block_chunk is None for k in _kernels(trained))
    assert all(not k.remat and k.block_chunk == 64 for k in _kernels(model))
    for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(trained), strict=True):
        assert a is b
    kept = training.for_training(model, remat=False, chunked=True)
    assert all(not k.remat and k.block_chunk == 64 for k in _kernels(kept))


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize(
    "settings", [{"remat": True, "chunked": True}, {"remat": True}, {"remat": False}]
)
def test_a_trained_model_s_loss_and_gradients_are_the_served_ones(
    name: str, settings: dict[str, bool]
) -> None:
    functions, build = MODELS[name]
    # Rows step a few at a time as served, so turning chunking off is seen.
    served = _chunked(build(), 64)
    trained = training.for_training(served, **settings)
    features, targets = _table()
    leaves, structure = jax.tree.flatten(served)
    _, trained_structure = jax.tree.flatten(trained)
    expected = jax.value_and_grad(_loss(functions, structure, features, targets))
    found = jax.value_and_grad(_loss(functions, trained_structure, features, targets))
    (value, gradient), (trained_value, trained_gradient) = (
        jax.jit(expected)(leaves),
        jax.jit(found)(leaves),
    )
    np.testing.assert_allclose(float(trained_value), float(value), rtol=1e-6)
    for a, b in zip(gradient, trained_gradient, strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-4, atol=1e-6)
    # Each rematerialized layer is a `jax.checkpoint` in the step's program.
    traced = str(jax.make_jaxpr(found)(leaves))
    assert ("remat" in traced) == settings["remat"]
