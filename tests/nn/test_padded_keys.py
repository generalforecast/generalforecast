# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention, QASSMax and induced attention over keys padded past a true length."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.nn.attention import attention_block, block_kv, keys_read
from generalforecast.nn.induced import compress
from generalforecast.nn.qassmax import QASSMaxParameters, qassmax
from generalforecast.nn.retrieval import RetrievalParameters, retrieval_cache, retrieve
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


def _reads_every_row(length: jax.Array, queries: int, rows: int) -> jax.Array:
    """The mask a decode of several units together gives a unit that reads one context.

    Every query reads every row of the call's padded context, past the
    unit's own length too; only the length bounds what it reads.
    """
    return jnp.ones((length.shape[0], queries, rows), dtype=bool)


def test_keys_read_counts_only_keys_within_the_length_that_the_mask_allows() -> None:
    cache = block_kv(
        tiny_tabicl(regression=False).icl.blocks[0],
        _padded_rows(ICL),
        jnp.asarray(LENGTHS, dtype=jnp.int32),
    )
    mask = _reads_every_row(cache.length, 3, max(LENGTHS)).at[1, 0, 2].set(False)

    counts = keys_read(cache._replace(mask=mask))

    np.testing.assert_array_equal(counts, [[7, 7, 7], [11, 12, 12]])
    np.testing.assert_array_equal(keys_read(cache), LENGTHS)
    assert keys_read(cache._replace(length=None)) == max(LENGTHS)


def test_a_mask_past_the_length_scales_queries_as_the_length_alone() -> None:
    block = tiny_tabicl(regression=False).icl.blocks[0]
    assert block.attention.scale is not None
    context = _padded_rows(ICL)
    queries = jnp.asarray(
        np.random.default_rng(4).normal(size=(2, 3, ICL)), jnp.float32
    )
    cache = block_kv(block, context, jnp.asarray(LENGTHS, dtype=jnp.int32))
    mask = _reads_every_row(cache.length, 3, context.shape[1])

    _close(
        attention_block(block, queries, cache._replace(mask=mask)),
        attention_block(block, queries, cache),
    )


def test_retrieval_scales_queries_by_the_keys_within_the_length() -> None:
    from tests.models.conformance.cases import _retrieval

    decoder = _retrieval(0).icl.decoder
    assert isinstance(decoder, RetrievalParameters) and decoder.scale is not None
    context = _padded_rows(ICL)
    labels = jnp.asarray(
        np.random.default_rng(5).integers(0, 3, context.shape[:2]), jnp.int32
    )
    queries = jnp.asarray(
        np.random.default_rng(6).normal(size=(2, 3, ICL)), jnp.float32
    )
    cache = retrieval_cache(
        decoder, context, labels, 3, jnp.asarray(LENGTHS, dtype=jnp.int32)
    )
    mask = _reads_every_row(cache.length, 3, context.shape[1])

    _close(
        retrieve(decoder, queries, cache._replace(mask=mask)),
        retrieve(decoder, queries, cache),
    )
