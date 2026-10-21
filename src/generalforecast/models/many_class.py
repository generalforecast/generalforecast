# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""More classes than a classifier has slots, with error-correcting output codes.

Each of `codes` estimators sees the classes merged into at most as many groups
as the classifier has slots: a fixed random code gives every class a group, and
the estimator learns to predict groups. Codes differ between estimators, so any
two classes share a group in few of them. A class's probability is the
geometric mean, over estimators, of its group's probability shared among the
group's classes, normalized over the classes.

To the engine it is one model, stored like an ensemble's estimators.
"""

import functools
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models import catalog
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

# The most classes a table may have.
MAX_CLASSES = catalog.MAX_CLASSES
# Rounds of trades `codebook` makes before giving up on separating the classes.
_REPAIRS = 1000


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ManyClass:
    """A classifier's parameters, run as `codes` estimators over class groups."""

    model: Any
    codes: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ManyClassState:
    """Each estimator's table state, stacked, and the task's class count."""

    states: Any
    num_classes: int = field(metadata={"static": True})


@functools.cache
def codebook(codes: int, num_classes: int, slots: int) -> np.ndarray:
    """Each estimator's group for each class, as evenly sized as they can be.

    Codes start as random permutations. Wherever two classes share a code, one
    of them trades groups with a random class in a random estimator, which
    keeps every estimator's group sizes, until every class has its own code.
    That is only possible when `min(slots, num_classes) ** codes` reaches the
    class count, and `fit` refuses a book where it failed.

    Args:
        codes: Number of estimators.
        num_classes: Number of classes.
        slots: Most groups an estimator can tell apart.

    Returns:
        Groups, `(codes, num_classes)`, each in `0 .. min(slots, num_classes) - 1`,
        read-only.
    """
    groups = min(slots, num_classes)
    book = np.stack(
        [
            np.random.default_rng(code).permutation(num_classes) % groups
            for code in range(codes)
        ]
    )
    generator = np.random.default_rng(codes)
    for _ in range(_REPAIRS if groups**codes >= num_classes else 0):
        _, first = np.unique(book.T, axis=0, return_index=True)
        clashing = np.setdiff1d(np.arange(num_classes), first)
        if len(clashing) == 0:
            break
        for clash in clashing:
            code, other = generator.integers(codes), generator.integers(num_classes)
            book[code, [clash, other]] = book[code, [other, clash]]
    book.flags.writeable = False
    return book


def many_class_functions(functions: TabularFunctions) -> TabularFunctions:
    """The contract calls of a `ManyClass` of a classifier with these calls.

    Args:
        functions: The classifier's contract calls.

    Returns:
        Contract calls that take a `ManyClass` of the classifier's parameters.
    """

    def groups(many: ManyClass, num_classes: int) -> tuple[Array, int]:
        slots = functions.properties(many.model).class_slots
        book = codebook(many.codes, num_classes, slots)
        return jnp.asarray(book), min(slots, num_classes)

    def labels(many: ManyClass, targets: Array, num_classes: int) -> Array:
        book, _ = groups(many, num_classes)
        # A class not known yet stays unknown in every group.
        grouped = book[:, jnp.maximum(targets, 0)].astype(targets.dtype)
        return jnp.where(targets >= 0, grouped, -1)

    def fit(
        many: ManyClass,
        features: Float[Array, " rows features"],
        targets: Array,
        num_classes: int,
        num_rows: Int[Array, ""] | int | None = None,
        categorical: Bool[Array, " features"] | None = None,
    ) -> tuple[ManyClassState, Array]:
        if not 1 <= num_classes <= MAX_CLASSES:
            raise ValueError(
                f"num_classes must be in 1 .. {MAX_CLASSES}, got {num_classes}"
            )
        slots = functions.properties(many.model).class_slots
        book = codebook(many.codes, num_classes, slots)
        if len({tuple(column) for column in book.T}) < num_classes:
            raise ValueError(
                f"{many.codes} codes over {slots} class slots cannot tell "
                f"{num_classes} classes apart; use more codes"
            )
        _, count = groups(many, num_classes)
        states, rows = jax.vmap(
            lambda grouped: functions.fit(
                many.model,
                features,
                grouped,
                count,
                num_rows=num_rows,
                categorical=categorical,
            )
        )(labels(many, targets, num_classes))
        return ManyClassState(states, num_classes), side_by_side(rows)

    def embed(
        many: ManyClass,
        state: ManyClassState,
        features: Float[Array, " rows features"],
        targets: Array | None = None,
    ) -> Array:
        if targets is None:
            rows = jax.vmap(lambda one: functions.embed(many.model, one, features))(
                state.states
            )
        else:
            rows = jax.vmap(
                lambda one, grouped: functions.embed(many.model, one, features, grouped)
            )(state.states, labels(many, targets, state.num_classes))
        return side_by_side(rows)

    def encode(
        many: ManyClass,
        state: ManyClassState,
        rows: Array,
        targets: Array,
        num_rows: Int[Array, ""] | int | None = None,
    ) -> Any:
        def one(single: Any, view: Array, grouped: Array) -> Any:
            return functions.encode(
                many.model, single, view, grouped, num_rows=num_rows
            )

        caches = jax.vmap(one)(
            state.states,
            apart(rows, many.codes),
            labels(many, targets, state.num_classes),
        )
        return fold(caches)

    def decode(
        many: ManyClass, state: ManyClassState, cache: Any, rows: Array
    ) -> Array:
        outputs = jax.vmap(
            lambda single, unfolded, view: functions.decode(
                many.model, single, unfolded, view
            )
        )(state.states, unfold(cache, many.codes), apart(rows, many.codes))
        return side_by_side(outputs)

    def postprocess(
        many: ManyClass, state: ManyClassState, outputs: Array
    ) -> Float[Array, " queries classes"]:
        book, count = groups(many, state.num_classes)
        probabilities = jax.vmap(
            lambda single, apart_outputs: functions.postprocess(
                many.model, single, apart_outputs
            )
        )(state.states, apart(outputs, many.codes))
        # Each class gets an equal share of its group's probability.
        sizes = jax.vmap(lambda row: jnp.bincount(row, length=count))(book)
        index = jnp.broadcast_to(
            book[:, None, :], (*probabilities.shape[:2], state.num_classes)
        )
        shares = (
            jnp.take_along_axis(probabilities, index, axis=-1)
            / (jnp.take_along_axis(sizes, book, axis=-1)[:, None, :])
        )
        scores = jnp.mean(jnp.log(jnp.maximum(shares, 1e-30)), axis=0)
        return jax.nn.softmax(scores, axis=-1)

    def properties(many: ManyClass) -> TabularProperties:
        inner = functions.properties(many.model)
        if inner.regression:
            raise ValueError("error-correcting codes need a classifier")
        return TabularProperties(
            regression=False,
            class_slots=MAX_CLASSES,
            num_outputs=MAX_CLASSES,
            # Its outputs are scores of class groups; only predictions are classes.
            averaging=Averaging("predictions"),
            stored_cache=inner.stored_cache,
            context_rows=inner.context_rows,
            context_features=inner.context_features,
            column_summary=inner.column_summary,
            unlabeled_context=inner.unlabeled_context,
        )

    return TabularFunctions(
        fit=fit,
        embed=embed,
        encode=encode,
        decode=decode,
        postprocess=postprocess,
        properties=properties,
        crowded_categories=stacked_crowded(functions),
    )
