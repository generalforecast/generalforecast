# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ensembles of any model: estimators that each see the table another way.

Estimator 0 sees the table as it is. Each later one sees its columns in a fixed
random order and, in classification, its labels shifted by its index, as
TabICL's own ensembles permute features and shift classes. Estimators combine
as the model's own predictor combines them, which its `averaging` property says:
by decoder outputs postprocessed once, by predictions, or by mixing their
distributions over regression buckets. Class columns are shifted back first.

The engine sees one model. Row representations put estimators side by side, and
caches fold estimators into their head axis, so a cache's rows stay its first
axis and page into slots like any other.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.contract import (
    Averaging,
    TabularFunctions,
    TabularProperties,
)
from generalforecast.models.stacking import (
    apart,
    fold,
    side_by_side,
    stacked_crowded,
    unfold,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Ensemble:
    """A model's parameters, run as `estimators` estimators."""

    model: Any
    estimators: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EnsembleState:
    """Each estimator's table state, stacked, and the table's own facts.

    Attributes:
        states: Each estimator's state, along a leading axis.
        categorical: Which columns are categorical, in the table's order.
        num_classes: Number of classes, or 0 for regression.
    """

    states: Any
    categorical: Bool[Array, " features"] | None
    num_classes: int = field(metadata={"static": True})


def _orders(estimators: int, features: int) -> np.ndarray:
    """Each estimator's column order: the identity, then fixed random orders."""
    return np.stack(
        [np.arange(features)]
        + [
            np.random.default_rng(estimator).permutation(features)
            for estimator in range(1, estimators)
        ]
    )


def _views(
    estimators: int,
    features: Float[Array, " rows features"],
    categorical: Bool[Array, " features"] | None,
) -> tuple[Array, Array | None]:
    """Each estimator's columns, in its order."""
    orders = jnp.asarray(_orders(estimators, features.shape[-1]))
    views = jnp.take_along_axis(
        jnp.broadcast_to(features, (estimators, *features.shape)),
        orders[:, None, :],
        axis=2,
    )
    masks = None if categorical is None else categorical[orders]
    return views, masks


def _labels(estimators: int, targets: Array, num_classes: int) -> Array:
    """Each estimator's targets: labels shifted by its index, values unchanged."""
    if num_classes == 0:
        return jnp.broadcast_to(targets, (estimators, *targets.shape))
    shifts = jnp.arange(estimators)[:, None]
    shifted = (targets[None, :] + shifts) % num_classes
    return jnp.where(targets[None, :] < 0, targets[None, :], shifted).astype(
        targets.dtype
    )


def _shifted_back(each: Array, num_classes: int) -> Array:
    """Each estimator's class columns moved back to the labels the table has.

    An estimator that saw label c as c + e put its column for it at c + e; the
    columns past the task's classes stay as they are.
    """
    count, width = each.shape[0], each.shape[-1]
    columns = jnp.arange(width)[None, :] + jnp.arange(count)[:, None]
    columns = jnp.where(
        jnp.arange(width)[None, :] < num_classes, columns % num_classes, columns
    )
    columns = jnp.broadcast_to(columns[:, None, :], each.shape)
    return jnp.take_along_axis(each, columns, axis=-1)


def _combined(each: Array, averaging: Averaging) -> Array:
    """The mean over estimators, without the trimmed share at each end."""
    count = each.shape[0]
    cut = int(averaging.trim * count)
    if cut:
        each = jnp.sort(each, axis=0)[cut : count - cut]
    return each.mean(axis=0)


def ensemble_functions(functions: TabularFunctions) -> TabularFunctions:
    """The contract calls of an `Ensemble` of a model with these calls.

    Args:
        functions: The model's contract calls.

    Returns:
        Contract calls that take an `Ensemble` of the model's parameters.
    """

    def fit(
        ensemble: Ensemble,
        features: Float[Array, " rows features"],
        targets: Array,
        num_classes: int,
        num_rows: Int[Array, ""] | int | None = None,
        categorical: Bool[Array, " features"] | None = None,
    ) -> tuple[EnsembleState, Array]:
        count = ensemble.estimators
        views, masks = _views(count, features, categorical)

        def one(view: Array, labels: Array, mask: Array | None) -> Any:
            return functions.fit(
                ensemble.model,
                view,
                labels,
                num_classes,
                num_rows=num_rows,
                categorical=mask,
            )

        states, rows = jax.vmap(one)(views, _labels(count, targets, num_classes), masks)
        return EnsembleState(states, categorical, num_classes), side_by_side(rows)

    def embed(
        ensemble: Ensemble,
        state: EnsembleState,
        features: Float[Array, " rows features"],
        targets: Array | None = None,
    ) -> Array:
        count = ensemble.estimators
        views, _ = _views(count, features, state.categorical)
        if targets is None:
            rows = jax.vmap(
                lambda one, view: functions.embed(ensemble.model, one, view)
            )(state.states, views)
        else:
            labels = _labels(count, targets, state.num_classes)
            rows = jax.vmap(
                lambda one, view, label: functions.embed(
                    ensemble.model, one, view, label
                )
            )(state.states, views, labels)
        return side_by_side(rows)

    def encode(
        ensemble: Ensemble,
        state: EnsembleState,
        rows: Array,
        targets: Array,
        num_rows: Int[Array, ""] | int | None = None,
    ) -> Any:
        count = ensemble.estimators
        labels = _labels(count, targets, state.num_classes)

        def one(single: Any, view: Array, label: Array) -> Any:
            return functions.encode(
                ensemble.model, single, view, label, num_rows=num_rows
            )

        caches = jax.vmap(one)(state.states, apart(rows, count), labels)
        return fold(caches)

    def decode(
        ensemble: Ensemble, state: EnsembleState, cache: Any, rows: Array
    ) -> Array:
        count = ensemble.estimators
        outputs = jax.vmap(
            lambda single, unfolded, view: functions.decode(
                ensemble.model, single, unfolded, view
            )
        )(state.states, unfold(cache, count), apart(rows, count))
        return side_by_side(outputs)

    def postprocess(ensemble: Ensemble, state: EnsembleState, outputs: Array) -> Array:
        count = ensemble.estimators
        each = apart(outputs, count)
        averaging = functions.properties(ensemble.model).averaging
        first = jax.tree.map(lambda values: values[0], state.states)
        if averaging.over == "buckets":
            # The mixture's logits, at the temperature postprocess divides out.
            logits = each.astype(jnp.float32) / averaging.temperature
            mixture = jax.nn.logsumexp(
                jax.nn.log_softmax(logits, axis=-1), axis=0
            ) - jnp.log(count)
            return functions.postprocess(
                ensemble.model, first, mixture * averaging.temperature
            )
        if averaging.over == "outputs":
            if state.num_classes > 0:
                each = _shifted_back(each, state.num_classes)
            return functions.postprocess(ensemble.model, first, each.mean(axis=0))
        predictions = jax.vmap(
            lambda single, values: functions.postprocess(ensemble.model, single, values)
        )(state.states, each)
        if state.num_classes > 0:
            predictions = _shifted_back(predictions, state.num_classes)
        return _combined(predictions, averaging)

    def properties(ensemble: Ensemble) -> TabularProperties:
        return functions.properties(ensemble.model)

    return TabularFunctions(
        fit=fit,
        embed=embed,
        encode=encode,
        decode=decode,
        postprocess=postprocess,
        properties=properties,
        crowded_categories=stacked_crowded(functions),
    )
