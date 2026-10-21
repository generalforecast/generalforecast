# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The retrieval decoder against a direct computation, with padding and int8 caches."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from generalforecast.nn.linear import LinearParameters
from generalforecast.nn.retrieval import RetrievalParameters, retrieval_cache, retrieve
from generalforecast.runtime.device import to_cache_dtype

WIDTH, HEADS, HEAD_DIM, CLASSES = 12, 3, 4, 7


def _decoder(seed: int = 0) -> RetrievalParameters:
    generator = np.random.default_rng(seed)

    def linear() -> LinearParameters:
        weight = generator.normal(size=(HEADS * HEAD_DIM, WIDTH))
        bias = generator.normal(size=HEADS * HEAD_DIM)
        return LinearParameters(
            jnp.asarray(weight, jnp.float32), jnp.asarray(bias, jnp.float32)
        )

    return RetrievalParameters(query=linear(), key=linear(), num_heads=HEADS)


def _expected(
    decoder: RetrievalParameters,
    context: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Each head's softmax over context rows, voting for their labels, averaged."""

    def project(linear: LinearParameters, rows: np.ndarray) -> np.ndarray:
        projected = rows @ np.asarray(linear.weight).T + np.asarray(linear.bias)
        return projected.reshape(len(rows), HEADS, HEAD_DIM)

    q, k = project(decoder.query, queries), project(decoder.key, context)
    scores = np.einsum("rhd,chd->hrc", q, k) / np.sqrt(HEAD_DIM)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    probabilities = weights.mean(axis=0) @ np.eye(CLASSES)[labels]
    return np.log(np.maximum(probabilities, 1e-5) + 3e-5)


def test_retrieve_matches_head_averaged_votes() -> None:
    decoder = _decoder()
    generator = np.random.default_rng(1)
    context = generator.normal(size=(9, WIDTH)).astype(np.float32)
    labels = generator.integers(0, CLASSES, size=9)
    queries = generator.normal(size=(5, WIDTH)).astype(np.float32)

    cache = retrieval_cache(
        decoder, jnp.asarray(context[None]), jnp.asarray(labels[None]), CLASSES
    )
    actual = retrieve(decoder, jnp.asarray(queries[None]), cache)[0]

    np.testing.assert_allclose(
        actual, _expected(decoder, context, labels, queries), rtol=1e-5, atol=1e-5
    )


def test_retrieve_ignores_rows_past_the_length() -> None:
    decoder = _decoder()
    generator = np.random.default_rng(2)
    context = generator.normal(size=(1, 12, WIDTH)).astype(np.float32)
    context[0, 7:] = np.nan
    labels = generator.integers(0, CLASSES, size=(1, 12))
    queries = jnp.asarray(generator.normal(size=(1, 5, WIDTH)), jnp.float32)

    padded = retrieval_cache(
        decoder, jnp.asarray(context), jnp.asarray(labels), CLASSES, jnp.asarray([7])
    )
    truncated = retrieval_cache(
        decoder, jnp.asarray(context[:, :7]), jnp.asarray(labels[:, :7]), CLASSES
    )

    np.testing.assert_allclose(
        retrieve(decoder, queries, padded),
        retrieve(decoder, queries, truncated),
        rtol=1e-6,
        atol=1e-6,
    )


def test_int8_caches_keep_labels_exact() -> None:
    decoder = _decoder()
    generator = np.random.default_rng(3)
    context = jnp.asarray(generator.normal(size=(1, 9, WIDTH)), jnp.float32)
    labels = jnp.asarray(generator.integers(0, CLASSES, size=(1, 9)))

    stored = to_cache_dtype(
        retrieval_cache(decoder, context, labels, CLASSES), jnp.int8
    )

    assert stored.value_scale is not None
    np.testing.assert_array_equal(
        stored.value * stored.value_scale[..., None],
        np.eye(CLASSES)[labels][:, :, None],
    )
