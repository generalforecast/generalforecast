# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention, QASSMax and induced attention over keys padded past a true length."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.nn.attention import attention_block, block_kv
from generalforecast.nn.induced import compress
from generalforecast.nn.qassmax import QASSMaxParameters, qassmax
from tests.models.tabicl.tiny import EMBED, HEADS, ICL, tiny_tabicl

LENGTHS = (7, 12)


def _padded_rows(width: int, seed: int = 0) -> jax.Array:
    """Two batches of 12 rows. Rows past each length hold large garbage values."""
    generator = np.random.default_rng(seed)
    rows = generator.normal(size=(len(LENGTHS), max(LENGTHS), width))
    for batch, length in enumerate(LENGTHS):
        rows[batch, length:] = 1e3 * generator.normal(size=rows[batch, length:].shape)
    return jnp.asarray(rows, dtype=jnp.float32)


def _close(actual: jax.Array, expected: jax.Array) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_attention_ignores_keys_past_the_cache_length() -> None:
    block = tiny_tabicl(regression=False).icl.blocks[0]
    # A NaN in padding must not reach attention either.
    context = _padded_rows(ICL).at[0, LENGTHS[0]].set(jnp.nan)
    queries = jnp.asarray(
        np.random.default_rng(1).normal(size=(2, 5, ICL)), jnp.float32
    )
    lengths = jnp.asarray(LENGTHS, dtype=jnp.int32)

    actual = attention_block(block, queries, block_kv(block, context, lengths))

    for batch, length in enumerate(LENGTHS):
        truncated = block_kv(block, context[batch : batch + 1, :length])
        expected = attention_block(block, queries[batch : batch + 1], truncated)
        _close(actual[batch : batch + 1], expected)


def test_attention_without_a_length_reads_every_key() -> None:
    block = tiny_tabicl(regression=False).icl.blocks[0]
    context = _padded_rows(ICL)[1:]
    full = jnp.asarray([context.shape[1]], dtype=jnp.int32)

    _close(
        attention_block(block, context, block_kv(block, context)),
        attention_block(block, context, block_kv(block, context, full)),
    )


def test_qassmax_takes_one_key_count_per_batch() -> None:
    parameters = tiny_tabicl(regression=False).icl.blocks[0].attention.scale
    assert isinstance(parameters, QASSMaxParameters)
    query = jnp.asarray(
        np.random.default_rng(2).normal(size=(3, 4, HEADS, ICL // HEADS)), jnp.float32
    )
    counts = (1, 5, 40)

    actual = qassmax(parameters, query, jnp.asarray(counts, dtype=jnp.int32))

    for batch, count in enumerate(counts):
        _close(
            actual[batch : batch + 1],
            qassmax(parameters, query[batch : batch + 1], count),
        )


def test_qassmax_key_count_can_be_traced() -> None:
    parameters = tiny_tabicl(regression=False).icl.blocks[0].attention.scale
    assert isinstance(parameters, QASSMaxParameters)
    query = jnp.asarray(
        np.random.default_rng(3).normal(size=(2, 4, HEADS, ICL // HEADS)), jnp.float32
    )

    traced = jax.jit(qassmax)(parameters, query, jnp.asarray([9, 9], dtype=jnp.int32))

    _close(traced, qassmax(parameters, query, 9))


def test_compress_ignores_context_rows_past_the_length() -> None:
    induced = tiny_tabicl(regression=False).columns.blocks[0]
    context = _padded_rows(EMBED)

    actual = compress(induced, context, jnp.asarray(LENGTHS, dtype=jnp.int32))

    for batch, length in enumerate(LENGTHS):
        expected = compress(induced, context[batch : batch + 1, :length])
        _close(actual[batch : batch + 1], expected)


def test_queries_masked_to_their_own_keys_attend_as_if_alone() -> None:
    block = tiny_tabicl(regression=False).icl.blocks[0]
    context = _padded_rows(ICL)[1:]
    queries = jnp.asarray(
        np.random.default_rng(2).normal(size=(1, 2, ICL)), jnp.float32
    )
    # The first query reads rows 0 to 6, the second rows 4 to 11.
    reads = [np.arange(0, 7), np.arange(4, 12)]
    mask = np.zeros((1, 2, context.shape[1]), dtype=bool)
    for query, rows in enumerate(reads):
        mask[0, query, rows] = True

    together = attention_block(
        block, queries, block_kv(block, context)._replace(mask=jnp.asarray(mask))
    )

    for query, rows in enumerate(reads):
        alone = attention_block(
            block, queries[:, query : query + 1], block_kv(block, context[:, rows])
        )
        _close(together[:, query : query + 1], alone)
