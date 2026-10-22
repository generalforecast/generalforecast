# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from generalforecast.nn.linear import MLPParameters, mlp


class QASSMaxParameters(NamedTuple):
    """Query-aware scalable softmax, which keeps attention sharp as the context grows.

    Attributes:
        base: MLP from log(n) to a scale per head and dimension.
        query: MLP from each query to its modulation.
    """

    base: MLPParameters
    query: MLPParameters


def key_counts(n_keys: int | Int[Array, "..."]) -> Float[Array, "batch rows_or_1"]:
    """Key counts as float32, `(batch, 1)` for one per batch row, or `(batch, rows)`."""
    counts = jnp.asarray(n_keys, dtype=jnp.float32)
    return counts.reshape(-1, 1) if counts.ndim < 2 else counts


def qassmax(
    parameters: QASSMaxParameters,
    query: Float[Array, " batch rows heads head_dim"],
    n_keys: int | Int[Array, "..."],
) -> Float[Array, " batch rows heads head_dim"]:
    """Scales queries as `query * base(log n) * (1 + tanh(query_mlp(query)))`.

    Args:
        parameters: The base and query MLPs.
        query: Per-head queries.
        n_keys: Number of keys the queries attend over: for all batch rows, one
            per batch row, or one per query, `(batch, rows)`. May be traced.

    Returns:
        Scaled queries.
    """
    num_heads = query.shape[-2]
    head_dim = query.shape[-1]

    counts = key_counts(n_keys)
    log_n = jnp.log(jnp.maximum(counts, 1)).astype(query.dtype)[..., None]

    base_scale = mlp(parameters.base, log_n)
    base_scale = base_scale.reshape(*counts.shape, num_heads, head_dim)

    query_adjustment = mlp(parameters.query, query)
    modulation = 1.0 + jnp.tanh(query_adjustment)

    return query * base_scale * modulation
