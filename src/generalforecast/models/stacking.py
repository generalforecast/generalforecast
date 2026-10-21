# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Several estimators of one model, stored as one model's rows and caches.

Row representations and outputs put estimators side by side. Caches fold the
estimator axis into the heads of their attention caches, so a cache's rows stay
its first axis and page into slots like any other; see `TabularFunctions` for the
layout caches have.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from generalforecast.models.contract import CrowdedCategories, TabularFunctions
from generalforecast.nn.attention import AttentionCache


def stacked_crowded(functions: TabularFunctions) -> CrowdedCategories | None:
    """Counts crowded columns of a state that stacks the inner model's in `states`."""
    inner = functions.crowded_categories
    if inner is None:
        return None
    return lambda state: inner(state.states)


def side_by_side(rows: Float[Array, " estimators rows dim"]) -> Array:
    """Puts each estimator's rows side by side, as one row each."""
    return jnp.swapaxes(rows, 0, 1).reshape(rows.shape[1], -1)


def apart(rows: Float[Array, " rows dim"], estimators: int) -> Array:
    """Undoes `side_by_side`."""
    return jnp.swapaxes(rows.reshape(rows.shape[0], estimators, -1), 0, 1)


def _layers(
    cache: Any, rows: Callable[[Array], Array], shared: Callable[[Array], Array]
) -> Any:
    """Changes each attention cache in a context cache.

    `rows` changes its row arrays, and `shared` its length and mask, which every
    estimator shares.
    """

    def one(layer: Any) -> AttentionCache:
        if not isinstance(layer, AttentionCache):
            raise TypeError(f"a context cache holds attention caches, not {layer!r}")
        return AttentionCache(
            key=rows(layer.key),
            value=rows(layer.value),
            length=None if layer.length is None else shared(layer.length),
            key_scale=None if layer.key_scale is None else rows(layer.key_scale),
            value_scale=None if layer.value_scale is None else rows(layer.value_scale),
            mask=None if layer.mask is None else shared(layer.mask),
        )

    return jax.tree.map(
        one, cache, is_leaf=lambda node: isinstance(node, AttentionCache)
    )


def fold(cache: Any) -> Any:
    """Moves the estimator axis into the heads; the estimators' lengths and masks agree."""

    def rows(values: Array) -> Array:
        moved = jnp.swapaxes(values, 0, 1)
        return moved.reshape(moved.shape[0], -1, *moved.shape[3:])

    return _layers(cache, rows, lambda values: values[0])


def unfold(cache: Any, estimators: int) -> Any:
    """Undoes `fold`."""

    def rows(values: Array) -> Array:
        split = values.reshape(values.shape[0], estimators, -1, *values.shape[2:])
        return jnp.swapaxes(split, 0, 1)

    return _layers(
        cache,
        rows,
        lambda values: jnp.broadcast_to(values, (estimators, *values.shape)),
    )
