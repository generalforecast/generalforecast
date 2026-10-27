# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from generalforecast.nn.attention import AttentionCache, keys_read
from generalforecast.nn.linear import LinearParameters, linear
from generalforecast.nn.qassmax import QASSMaxParameters, qassmax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RetrievalParameters:
    """A decoder that predicts classes from the labels of the context rows a query attends to.

    Each head's attention weights over the context rows vote for those rows'
    labels; the votes, averaged over heads, are the class probabilities.
    """

    query: LinearParameters
    key: LinearParameters
    num_heads: int = field(metadata={"static": True})
    scale: QASSMaxParameters | None = None


def _split_heads(values: Array, num_heads: int) -> Array:
    return values.reshape(*values.shape[:-1], num_heads, -1)


def retrieval_cache(
    parameters: RetrievalParameters,
    context_rows: Float[Array, " batch context model"],
    labels: Int[Array, " batch context"],
    num_classes: int,
    length: Int[Array, " batch"] | None = None,
) -> AttentionCache:
    """Projects context rows to keys, with their one-hot labels as the values.

    Args:
        parameters: Decoder parameters.
        context_rows: Normalized context rows.
        labels: Class ids of the context rows.
        num_classes: Width of the one-hot labels.
        length: Real rows at the start of each batch row, or None for all rows.

    Returns:
        Keys `(batch, context, heads, head_dim)` and one-hot labels
        `(batch, context, 1, num_classes)`.
    """
    key = _split_heads(linear(parameters.key, context_rows), parameters.num_heads)
    value = jnn.one_hot(labels, num_classes, dtype=key.dtype)[..., None, :]
    if length is not None:
        real = (jnp.arange(key.shape[-3]) < length[..., None])[..., None, None]
        key, value = jnp.where(real, key, 0), jnp.where(real, value, 0)
    return AttentionCache(key=key, value=value, length=length)


def retrieve(
    parameters: RetrievalParameters,
    query_rows: Float[Array, " batch rows model"],
    cache: AttentionCache,
) -> Float[Array, " batch rows classes"]:
    """Class log-probabilities from the labels the queries attend to.

    Keys past `cache.length`, and keys `cache.mask` leaves out of a query, get no
    votes. Probabilities are clipped below at 1e-5 and offset by 3e-5 before the log.

    Args:
        parameters: Decoder parameters.
        query_rows: Normalized query rows.
        cache: Keys and labels from `retrieval_cache`, possibly stored as int8.

    Returns:
        Log-probabilities over the label width.
    """
    q = _split_heads(linear(parameters.query, query_rows), parameters.num_heads)
    key, value = cache.key.astype(q.dtype), cache.value.astype(q.dtype)
    if cache.key_scale is not None and cache.value_scale is not None:
        key = key * cache.key_scale[..., None].astype(q.dtype)
        value = value * cache.value_scale[..., None].astype(q.dtype)
    if parameters.scale is not None:
        q = qassmax(parameters.scale, q, keys_read(cache))
    scores = jnp.einsum(
        "brhd,bchd->bhrc", q, key, preferred_element_type=jnp.float32
    ) * (q.shape[-1] ** -0.5)
    lowest = jnp.finfo(jnp.float32).min
    if cache.length is not None:
        real = jnp.arange(key.shape[-3]) < cache.length[:, None]
        scores = jnp.where(real[:, None, None], scores, lowest)
    if cache.mask is not None:
        scores = jnp.where(cache.mask[:, None], scores, lowest)
    weights = jnn.softmax(scores, axis=-1).mean(axis=1)
    probabilities = jnp.einsum(
        "brc,bck->brk", weights, value[..., 0, :].astype(jnp.float32)
    )
    return jnp.log(jnp.maximum(probabilities, 1e-5) + 3e-5).astype(q.dtype)
