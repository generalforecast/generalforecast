# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from generalforecast.nn.attention import (
    AttentionBlockParameters,
    AttentionCache,
    attention_block,
    attention_block_streamed,
    block_kv,
    in_chunks,
)


class InducedAttentionParameters(NamedTuple):
    """Inducing points with the blocks that compress into and expand from them."""

    inducing: Float[Array, " inducing model"]
    compress: AttentionBlockParameters
    expand: AttentionBlockParameters


def compress(
    parameters: InducedAttentionParameters,
    context: Float[Array, " batch context model"],
    length: Int[Array, " batch"] | None = None,
) -> Float[Array, " batch inducing model"]:
    """Summarizes context rows into the inducing points.

    Args:
        parameters: Inducing points and attention blocks.
        context: Context rows.
        length: Real rows at the start of each batch row, or None for all rows.

    Returns:
        The inducing points after attending to the context.
    """

    def summarize(context: Array, length: Array | None) -> Array:
        inducing = jnp.broadcast_to(
            parameters.inducing, (context.shape[0], *parameters.inducing.shape)
        )
        cache = block_kv(parameters.compress, context, length)
        return attention_block(parameters.compress, inducing, cache)

    # Each batch row reads all of its context rows, so batch rows go a few at a time.
    chunk = parameters.compress.attention.kernel.block_chunk
    per_step = None if chunk is None else max(1, chunk // context.shape[1])
    return in_chunks(summarize, (context, length), per_step, axis=0)


def compress_streamed(
    parameters: InducedAttentionParameters,
    context: Callable[[Array, int], Float[Array, " batch size model"]],
    rows: int,
    chunk: int,
    length: Int[Array, " batch"] | None = None,
) -> Float[Array, " batch inducing model"]:
    """`compress` to context rows made at most `chunk` at a time, so memory does not grow with rows.

    Args:
        parameters: Inducing points and attention blocks.
        context: Makes `size` context rows from row `start`, as `context(start, size)`.
        rows: Context rows in all.
        chunk: Most rows a step makes.
        length: Real rows at the start of each batch row, or None for all rows.

    Returns:
        The inducing points after attending to the context.
    """
    steps = -(-rows // chunk)
    # Equal steps, so the last overlaps the one before rather than padding.
    size = -(-rows // steps)
    batch = jax.eval_shape(lambda start: context(start, size), jnp.int32(0)).shape[0]

    def keys(index: Array) -> tuple[AttentionCache, Array]:
        start = jnp.minimum(index * size, rows - size)
        position = start + jnp.arange(size)
        # Rows the last step shares with the one before count only there.
        counts = jnp.broadcast_to(position >= index * size, (batch, size))
        if length is not None:
            counts = counts & (position < length[:, None])
        return block_kv(parameters.compress, context(start, size)), counts

    inducing = jnp.broadcast_to(
        parameters.inducing, (batch, *parameters.inducing.shape)
    )
    n_keys = rows if length is None else length
    return attention_block_streamed(parameters.compress, inducing, keys, steps, n_keys)


def expand(
    parameters: InducedAttentionParameters,
    rows: Float[Array, " batch rows model"],
    state: Float[Array, " batch inducing model"],
) -> Float[Array, " batch rows model"]:
    """Updates rows by attending to compressed inducing points.

    Args:
        parameters: Inducing points and attention blocks.
        rows: Rows to update, context or query.
        state: Inducing points from `compress`.

    Returns:
        Updated rows.
    """
    cache = block_kv(parameters.expand, state)
    chunk = parameters.expand.attention.kernel.block_chunk
    per_step = None if chunk is None else max(1, chunk // rows.shape[0])
    return in_chunks(
        lambda rows: attention_block(parameters.expand, rows, cache),
        (rows,),
        per_step,
        axis=1,
    )
