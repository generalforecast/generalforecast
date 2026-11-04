# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Input-adapted models: a change to the features in front of a model's own calls.

A derived model of kind `adapter` changes what its base reads, not its
weights. `Adapted` holds a model's parameters and the adapter's own part;
`adapted_functions` runs the model's contract calls on features the
adapter changed. `fit` computes the adapter's statistics from the
context's real rows and keeps them in the state, and every later `embed`,
of context pages and queries alike, changes its rows with those same
statistics, so the engine's paging and query batching see one change.
An adapter reads the columns it was made for, so an adapted model never
has its table's width padded.

The adapter's functions are static: models whose adapters differ only in
their values share compiled programs.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.contract import TabularFunctions, TabularProperties
from generalforecast.models.derived import Inputs


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Adapted:
    """A model's parameters and the input adapter in front of it.

    Attributes:
        model: The model's own parameters.
        own: The adapter's part, as `Inputs` reads it.
    """

    model: Any
    own: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AdaptedState:
    """The model's table state and the adapter's statistics of the context."""

    state: Any
    statistics: Any


@dataclass(frozen=True)
class WithInputs:
    """A bag's parameters for an adapted model: `combine`'s, and the bag's own part.

    Attributes:
        combine: Gives the model's parameters for a bag, as `Bagged` takes it.
    """

    combine: Callable[[Any, Any], Any]

    def __call__(self, shared: Any, own: Any) -> Adapted:
        """One bag's `Adapted` parameters."""
        return Adapted(self.combine(shared, own), own)


@lru_cache(maxsize=64)
def adapted_functions(functions: TabularFunctions, inputs: Inputs) -> TabularFunctions:
    """The contract calls of an `Adapted` model with these calls.

    The same calls and inputs give the same functions, so derived models of
    one base share them and their compiled programs.

    Args:
        functions: The model's contract calls.
        inputs: How the adapter's part changes the features the model reads.

    Returns:
        Contract calls that take `Adapted` parameters.
    """

    def fit(
        adapted: Adapted,
        features: Float[Array, " rows features"],
        targets: Array,
        num_classes: int,
        num_rows: Int[Array, ""] | int | None = None,
        categorical: Bool[Array, " features"] | None = None,
    ) -> tuple[AdaptedState, Array]:
        statistics = inputs.statistics(adapted.own, features, num_rows)
        state, rows = functions.fit(
            adapted.model,
            inputs.transform(adapted.own, statistics, features),
            targets,
            num_classes,
            num_rows=num_rows,
            categorical=categorical,
        )
        return AdaptedState(state, statistics), rows

    def embed(
        adapted: Adapted,
        state: AdaptedState,
        features: Float[Array, " rows features"],
        targets: Array | None = None,
    ) -> Array:
        changed = inputs.transform(adapted.own, state.statistics, features)
        return functions.embed(adapted.model, state.state, changed, targets)

    def encode(
        adapted: Adapted,
        state: AdaptedState,
        rows: Array,
        targets: Array,
        num_rows: Int[Array, ""] | int | None = None,
    ) -> Any:
        return functions.encode(
            adapted.model, state.state, rows, targets, num_rows=num_rows
        )

    def decode(adapted: Adapted, state: AdaptedState, cache: Any, rows: Array) -> Array:
        return functions.decode(adapted.model, state.state, cache, rows)

    def postprocess(adapted: Adapted, state: AdaptedState, outputs: Array) -> Array:
        return functions.postprocess(adapted.model, state.state, outputs)

    def properties(adapted: Adapted) -> TabularProperties:
        # An adapter reads the columns it was made for: the engine must not
        # pad a table's width for it.
        return dataclasses.replace(
            functions.properties(adapted.model), padded_width=False
        )

    crowded = functions.crowded_categories
    reading = functions.decode_reading
    return TabularFunctions(
        fit=fit,
        embed=embed,
        encode=encode,
        decode=decode,
        postprocess=postprocess,
        properties=properties,
        crowded_categories=None
        if crowded is None
        else (lambda state: crowded(state.state)),
        decode_reading=None
        if reading is None
        else (
            lambda adapted, state, read, rows: reading(
                adapted.model, state.state, read, rows
            )
        ),
    )
