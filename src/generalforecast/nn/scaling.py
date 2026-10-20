import math
from typing import NamedTuple

import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from generalforecast.nn.linear import MLPParameters, mlp
from generalforecast.nn.qassmax import QASSMaxParameters, key_counts, qassmax


class LogScaleParameters(NamedTuple):
    """Scales each head's queries by `offset + head_scale * log(n)`.

    No offset means 0.
    """

    head_scale: Float[Array, " heads"]
    offset: Float[Array, " heads"] | None = None


class GatedLogScaleParameters(NamedTuple):
    """Log scaling, gated per query by `1 + tanh(gate(query))`.

    The gate MLP is shared across heads and reads each head's query vector.
    """

    head_scale: Float[Array, " heads"]
    gate: MLPParameters


class PerDimScaleParameters(NamedTuple):
    """Scales each query dimension by a learned softplus, in place of `1 / sqrt(d)`.

    The scale is `1.442695041 * softplus(scale)`, which starts at 1 where the
    parameter is 0, and includes the `sqrt(d)` attention divides by again.
    """

    scale: Float[Array, " head_dim"]


QueryScale = (
    QASSMaxParameters
    | LogScaleParameters
    | GatedLogScaleParameters
    | PerDimScaleParameters
)


def _log_keys(
    n_keys: int | Int[Array, "..."], dtype: jnp.dtype
) -> Float[Array, " batch rows 1 1"]:
    """`log(max(n, 1))` per batch row, or per query, computed in float32."""
    counts = key_counts(n_keys)[..., None, None]
    return jnp.log(jnp.maximum(counts, 1)).astype(dtype)


def log_scale(
    parameters: LogScaleParameters,
    query: Float[Array, " batch rows heads head_dim"],
    n_keys: int | Int[Array, "..."],
) -> Float[Array, " batch rows heads head_dim"]:
    """Scales queries as `query * (offset + log(n) * head_scale)`.

    Args:
        parameters: Per-head factors and offsets.
        query: Per-head queries.
        n_keys: Number of keys: for all batch rows, one per batch row, or one
            per query, `(batch, rows)`.

    Returns:
        Scaled queries.
    """
    head_scale = parameters.head_scale.astype(query.dtype)[:, None]
    if parameters.offset is None:
        return query * _log_keys(n_keys, query.dtype) * head_scale
    offset = parameters.offset.astype(query.dtype)[:, None]
    return query * (offset + _log_keys(n_keys, query.dtype) * head_scale)


def gated_log_scale(
    parameters: GatedLogScaleParameters,
    query: Float[Array, " batch rows heads head_dim"],
    n_keys: int | Int[Array, "..."],
) -> Float[Array, " batch rows heads head_dim"]:
    """Scales queries as `log_scale(query) * (1 + tanh(gate(query)))`.

    Args:
        parameters: Per-head factors and the gate MLP.
        query: Per-head queries.
        n_keys: Number of keys: for all batch rows, one per batch row, or one
            per query, `(batch, rows)`.

    Returns:
        Scaled queries.
    """
    scaled = log_scale(LogScaleParameters(parameters.head_scale), query, n_keys)
    return scaled * (1 + jnp.tanh(mlp(parameters.gate, query)))


def per_dim_scale(
    parameters: PerDimScaleParameters, query: Float[Array, " batch rows heads head_dim"]
) -> Float[Array, " batch rows heads head_dim"]:
    """Scales queries as `query * 1.442695041 * softplus(scale) * sqrt(head_dim)`.

    Args:
        parameters: The per-dimension parameter.
        query: Per-head queries.

    Returns:
        Scaled queries.
    """
    head_dim = query.shape[-1]
    factor = 1.442695041 * math.sqrt(head_dim) * jnn.softplus(parameters.scale)
    return query * factor.astype(query.dtype)


def scale_queries(
    parameters: QueryScale,
    query: Float[Array, " batch rows heads head_dim"],
    n_keys: int | Int[Array, "..."],
) -> Float[Array, " batch rows heads head_dim"]:
    """Applies QASSMax, log scaling, gated log scaling or per-dimension scaling.

    Args:
        parameters: Scaling parameters.
        query: Per-head queries.
        n_keys: Number of keys: for all batch rows, one per batch row, or one
            per query, `(batch, rows)`.

    Returns:
        Scaled queries.
    """
    if isinstance(parameters, QASSMaxParameters):
        return qassmax(parameters, query, n_keys)
    if isinstance(parameters, PerDimScaleParameters):
        return per_dim_scale(parameters, query)
    if isinstance(parameters, GatedLogScaleParameters):
        return gated_log_scale(parameters, query, n_keys)
    return log_scale(parameters, query, n_keys)
