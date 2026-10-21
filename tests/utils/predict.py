# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from generalforecast.models.contract import TabularFunctions


def predict(
    functions: TabularFunctions,
    model: Any,
    features: Float[Array, " rows features"],
    targets: Array,
    queries: Float[Array, " queries features"],
) -> Array:
    """Scores queries against one labeled context with any model's contract calls.

    Args:
        functions: The model's contract calls, such as `tabicl_functions`.
        model: Model parameters.
        features: Context rows, with NaN for missing values.
        targets: Class labels `0 .. C-1`, or regression targets.
        queries: Rows to score.

    Returns:
        Classification: probabilities over classes `0 .. C-1`, where C is one past
        the largest label. Regression: the model's quantiles on the original
        target scale.
    """
    properties = functions.properties(model)
    num_classes = 0
    if not properties.regression:
        num_classes = int(jnp.max(targets)) + 1
        if int(jnp.min(targets)) < 0 or num_classes > properties.class_slots:
            raise ValueError(
                f"class labels must be integers in 0 .. {properties.class_slots - 1}"
            )

    state, context_rows = functions.fit(model, features, targets, num_classes)
    cache = functions.encode(model, state, context_rows, targets)
    outputs = functions.decode(
        model, state, cache, functions.embed(model, state, queries)
    )
    return functions.postprocess(model, state, outputs)
