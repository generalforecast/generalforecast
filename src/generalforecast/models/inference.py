# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request-scoped native runs of any model's contract calls, shared by servers.

Packages and the Models API fit a request's or a revision's context once, then
score queries against it, for whatever `generalforecast.models.registry` loaded: a built-in
classifier or regressor, its `@K` codes or its `*N` ensemble, or a forecaster.
Each set of contract calls compiles once per process.
"""

from functools import lru_cache
from typing import Any

import jax
import numpy as np

from generalforecast.models.contract import TabularFunctions
from generalforecast.models.series import SeriesFunctions, SeriesGroup


@lru_cache(maxsize=16)
def _tabular(functions: TabularFunctions) -> tuple[Any, Any]:
    @jax.jit(static_argnums=3)
    def fit(model: Any, x: Any, y: Any, classes: int) -> Any:
        state, rows = functions.fit(model, x, y, classes)
        return state, functions.encode(model, state, rows, y)

    @jax.jit
    def predict(model: Any, state: Any, cache: Any, x: Any) -> Any:
        rows = functions.embed(model, state, x)
        return functions.postprocess(
            model, state, functions.decode(model, state, cache, rows)
        )

    return fit, predict


def fit(model: tuple[TabularFunctions, Any], x: Any, y: Any, classes: int) -> Any:
    """A context's fitted state and cache, ready before it returns.

    Args:
        model: Contract calls and parameters.
        x: Context features, float32 `(rows, features)`.
        y: Class indices, or regression targets.
        classes: The class count, or 0 for a regressor.
    """
    functions, parameters = model
    return jax.block_until_ready(_tabular(functions)[0](parameters, x, y, classes))


def predict(
    model: tuple[TabularFunctions, Any], fitted: Any, x: Any, classes: int
) -> np.ndarray:
    """Checked outputs for queries: class probabilities or increasing quantiles.

    Raises:
        ValueError: If the model returned outputs of another shape, or values
            no classifier or regressor returns.
    """
    functions, parameters = model
    state, cache = fitted
    result = np.asarray(_tabular(functions)[1](parameters, state, cache, x))
    width = classes or functions.properties(parameters).num_outputs
    valid = result.shape == (x.shape[0], width) and np.isfinite(result).all()
    if valid and classes:
        valid = bool(
            (result >= 0).all() and np.allclose(result.sum(axis=1), 1, atol=1e-5)
        )
    elif valid:
        valid = bool((np.diff(result, axis=1) >= 0).all())
    if not valid:
        raise ValueError("the model returned invalid outputs")
    return result


@lru_cache(maxsize=8)
def _forecast(functions: SeriesFunctions) -> Any:
    return jax.jit(functions.forecast, static_argnums=2)


def forecast(
    model: tuple[SeriesFunctions, Any], group: SeriesGroup, horizon: int
) -> np.ndarray:
    """Checked quantile forecasts for every row of a group, read-only.

    Returns:
        `(rows, horizon, quantiles)`, over the model's own quantile levels.

    Raises:
        ValueError: If the forecasts have another shape or are not finite.
    """
    functions, parameters = model
    levels = functions.properties(parameters).quantiles
    result = np.asarray(_forecast(functions)(parameters, group, horizon))
    if (
        result.shape != (group.values.shape[0], horizon, len(levels))
        or not np.isfinite(result).all()
    ):
        raise ValueError("the forecaster returned invalid forecasts")
    result.setflags(write=False)
    return result
