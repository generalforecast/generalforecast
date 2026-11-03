# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bags: several weight sets for one model, run together and averaged.

A bag is one set of weights for a model's contract calls. `Bagged` keeps the
parts every bag shares once, in `shared`, and each bag's own part stacked
along a leading axis in `own`; `combine(shared, own)` gives one bag's
parameters. Bags whose weights differ only in small parts, such as low-rank
updates over one base, so hold one copy of the base however many there are.
With `shared` None, `own` is each bag's whole parameters.

Every bag sees the same table. Their predictions are averaged: a
classifier's probabilities, renormalized over the task's classes, and a
regressor's quantiles, sorted again so they stay monotone.

The engine sees one model. Row representations put bags side by side, and
caches fold bags into their head axis, as ensembles' estimators do.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.contract import TabularFunctions, TabularProperties
from generalforecast.models.stacking import (
    apart,
    fold,
    side_by_side,
    stacked_crowded,
    unfold,
)


def _whole(shared: Any, own: Any) -> Any:
    """A bag's parameters when each bag holds all of them."""
    return own


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Bagged:
    """Weight sets for one model, run as `bags` bags.

    Attributes:
        shared: What every bag shares, or None.
        own: Each bag's own part, along a leading axis of length `bags`.
        bags: How many bags there are.
        combine: Gives one bag's parameters from `shared` and its own part.
    """

    shared: Any
    own: Any
    bags: int = field(metadata={"static": True})
    combine: Callable[[Any, Any], Any] = field(
        default=_whole, metadata={"static": True}
    )

    def bag(self, index: int) -> Any:
        """One bag's parameters."""
        return self.combine(self.shared, jax.tree.map(lambda x: x[index], self.own))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BaggedState:
    """Each bag's table state, stacked, and the table's class count."""

    states: Any
    num_classes: int = field(metadata={"static": True})


def bagged_functions(functions: TabularFunctions) -> TabularFunctions:
    """The contract calls of a `Bagged` model with these calls.

    Args:
        functions: The model's contract calls.

    Returns:
        Contract calls that take a `Bagged` set of the model's parameters.
    """

    def each(bagged: Bagged, call: Callable[..., Any], *stacked: Any) -> Any:
        """Runs `call(parameters, *one)` for every bag, over stacked arguments."""
        return jax.vmap(
            lambda own, *one: call(bagged.combine(bagged.shared, own), *one)
        )(bagged.own, *stacked)

    def fit(
        bagged: Bagged,
        features: Float[Array, " rows features"],
        targets: Array,
        num_classes: int,
        num_rows: Int[Array, ""] | int | None = None,
        categorical: Bool[Array, " features"] | None = None,
    ) -> tuple[BaggedState, Array]:
        states, rows = each(
            bagged,
            lambda model: functions.fit(
                model,
                features,
                targets,
                num_classes,
                num_rows=num_rows,
                categorical=categorical,
            ),
        )
        return BaggedState(states, num_classes), side_by_side(rows)

    def embed(
        bagged: Bagged,
        state: BaggedState,
        features: Float[Array, " rows features"],
        targets: Array | None = None,
    ) -> Array:
        rows = each(
            bagged,
            lambda model, one: functions.embed(model, one, features, targets),
            state.states,
        )
        return side_by_side(rows)

    def encode(
        bagged: Bagged,
        state: BaggedState,
        rows: Array,
        targets: Array,
        num_rows: Int[Array, ""] | int | None = None,
    ) -> Any:
        caches = each(
            bagged,
            lambda model, one, view: functions.encode(
                model, one, view, targets, num_rows=num_rows
            ),
            state.states,
            apart(rows, bagged.bags),
        )
        return fold(caches)

    def decode(bagged: Bagged, state: BaggedState, cache: Any, rows: Array) -> Array:
        outputs = each(
            bagged,
            lambda model, one, unfolded, view: functions.decode(
                model, one, unfolded, view
            ),
            state.states,
            unfold(cache, bagged.bags),
            apart(rows, bagged.bags),
        )
        return side_by_side(outputs)

    def postprocess(bagged: Bagged, state: BaggedState, outputs: Array) -> Array:
        predictions = each(
            bagged,
            lambda model, one, values: functions.postprocess(model, one, values),
            state.states,
            apart(outputs, bagged.bags),
        )
        mean = predictions.astype(jnp.float32).mean(axis=0)
        if state.num_classes == 0:
            return jnp.sort(mean, axis=-1)
        classes = jnp.arange(mean.shape[-1]) < state.num_classes
        mean = jnp.where(classes, mean, 0.0)
        return mean / jnp.maximum(mean.sum(axis=-1, keepdims=True), 1e-30)

    def properties(bagged: Bagged) -> TabularProperties:
        return functions.properties(bagged.bag(0))

    return TabularFunctions(
        fit=fit,
        embed=embed,
        encode=encode,
        decode=decode,
        postprocess=postprocess,
        properties=properties,
        crowded_categories=stacked_crowded(functions),
    )
