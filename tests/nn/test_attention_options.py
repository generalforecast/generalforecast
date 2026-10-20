"""Grouped-query attention, QK-norm, query scaling, token axes and a context norm."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.nn.attention import (
    AttentionBlockParameters,
    AttentionCache,
    AttentionParameters,
    attend,
    attention_block,
    block_kv,
    project_kv,
)
from generalforecast.nn.linear import LinearParameters, MLPParameters, linear, mlp
from generalforecast.nn.norm import RMSNormParameters, normalize, rms_norm
from generalforecast.nn.qassmax import QASSMaxParameters, qassmax
from generalforecast.nn.rope import RoPE, apply_rope
from generalforecast.nn.scaling import LogScaleParameters, log_scale
from generalforecast.runtime.device import to_cache_dtype

MODEL = 16
HEADS = 4
HEAD_DIM = MODEL // HEADS


def _linear(key: jax.Array, outputs: int, inputs: int) -> LinearParameters:
    return LinearParameters(
        weight=jax.random.normal(key, (outputs, inputs)) / np.sqrt(inputs), bias=None
    )


def _qassmax(key: jax.Array) -> QASSMaxParameters:
    keys = jax.random.split(key, 4)
    return QASSMaxParameters(
        base=MLPParameters(
            input=_linear(keys[0], 8, 1), output=_linear(keys[1], MODEL, 8)
        ),
        query=MLPParameters(
            input=_linear(keys[2], 8, HEAD_DIM), output=_linear(keys[3], HEAD_DIM, 8)
        ),
    )


def _attention(
    key: jax.Array,
    *,
    kv_heads: int = HEADS,
    qk_norm: bool = False,
    scaled: bool = False,
    rope: RoPE | None = None,
) -> AttentionParameters:
    keys = jax.random.split(key, 7)
    query_norm = key_norm = None
    if qk_norm:
        query_norm = RMSNormParameters(
            weight=jax.random.uniform(keys[4], (HEAD_DIM,), minval=0.5, maxval=2)
        )
        key_norm = RMSNormParameters(
            weight=jax.random.uniform(keys[5], (HEAD_DIM,), minval=0.5, maxval=2)
        )
    return AttentionParameters(
        query=_linear(keys[0], MODEL, MODEL),
        key=_linear(keys[1], kv_heads * HEAD_DIM, MODEL),
        value=_linear(keys[2], kv_heads * HEAD_DIM, MODEL),
        output=_linear(keys[3], MODEL, MODEL),
        num_heads=HEADS,
        scale=_qassmax(keys[6]) if scaled else None,
        query_norm=query_norm,
        key_norm=key_norm,
        rope=rope,
    )


def _heads(values: jax.Array) -> jax.Array:
    return values.reshape(*values.shape[:-1], HEADS, HEAD_DIM)


def _rows(seed: int, rows: int) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), (2, rows, MODEL))


def _close(actual: jax.Array, expected: jax.Array) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6
    )


def _naive(
    parameters: AttentionParameters,
    query: jax.Array,
    cache: AttentionCache,
    query_heads_for_kv_head: int,
) -> jax.Array:
    """Attention written out one query head at a time."""
    projected = _heads(linear(parameters.query, query))
    outputs = []
    for head in range(HEADS):
        kv = head // query_heads_for_kv_head
        scores = projected[:, :, head] @ jnp.swapaxes(cache.key[:, :, kv], -1, -2)
        weights = jax.nn.softmax(scores / np.sqrt(HEAD_DIM), axis=-1)
        outputs.append(weights @ cache.value[:, :, kv])
    merged = jnp.concatenate(outputs, axis=-1)
    assert parameters.output is not None
    return linear(parameters.output, merged)


def test_narrow_key_projection_gives_fewer_cached_heads() -> None:
    parameters = _attention(jax.random.key(0), kv_heads=2)
    cache = project_kv(parameters, _rows(1, 7))

    assert cache.key.shape == cache.value.shape == (2, 7, 2, HEAD_DIM)


def test_query_heads_share_kv_heads_in_order() -> None:
    # Matches torch repeat_interleave: query heads 0-1 use kv head 0, 2-3 use 1.
    parameters = _attention(jax.random.key(0), kv_heads=2)
    context, query = _rows(1, 7), _rows(2, 3)
    cache = project_kv(parameters, context)

    _close(
        attend(parameters, query, cache),
        _naive(parameters, query, cache, query_heads_for_kv_head=2),
    )


def test_an_int8_cache_attends_as_its_dequantized_values_do() -> None:
    parameters = _attention(jax.random.key(0), kv_heads=2)
    context, query = _rows(1, 7), _rows(2, 3)
    stored = to_cache_dtype(project_kv(parameters, context), jnp.dtype("int8"))
    assert stored.key_scale is not None and stored.value_scale is not None
    dequantized = stored._replace(
        key=stored.key * stored.key_scale[..., None],
        value=stored.value * stored.value_scale[..., None],
        key_scale=None,
        value_scale=None,
    )

    _close(attend(parameters, query, stored), attend(parameters, query, dequantized))


def test_every_query_head_shares_a_single_kv_head() -> None:
    parameters = _attention(jax.random.key(0), kv_heads=1)
    context, query = _rows(1, 7), _rows(2, 3)
    cache = project_kv(parameters, context)

    _close(
        attend(parameters, query, cache),
        _naive(parameters, query, cache, query_heads_for_kv_head=HEADS),
    )


def test_qk_norm_applies_after_rope() -> None:
    rope = RoPE(theta=100.0, interleaved=True)
    parameters = _attention(jax.random.key(0), qk_norm=True, rope=rope)
    context = _rows(1, 7)
    assert parameters.key_norm is not None

    cache = project_kv(parameters, context)

    rotated = apply_rope(_heads(linear(parameters.key, context)), rope)
    _close(cache.key, rms_norm(parameters.key_norm, rotated))


def test_qassmax_applies_after_the_query_norm() -> None:
    parameters = _attention(jax.random.key(0), qk_norm=True, scaled=True)
    context, query = _rows(1, 7), _rows(2, 3)
    assert parameters.query_norm is not None
    assert isinstance(parameters.scale, QASSMaxParameters)
    cache = project_kv(parameters, context)

    q = rms_norm(parameters.query_norm, _heads(linear(parameters.query, query)))
    q = qassmax(parameters.scale, q, n_keys=7)
    scores = jnp.einsum("bthd,bshd->bhts", q, cache.key) / np.sqrt(HEAD_DIM)
    attended = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(scores, -1), cache.value)
    assert parameters.output is not None
    expected = linear(parameters.output, attended.reshape(*attended.shape[:2], -1))
    _close(attend(parameters, query, cache), expected)


def test_qk_norm_changes_attention() -> None:
    plain = _attention(jax.random.key(0))
    normed = _attention(jax.random.key(0), qk_norm=True)
    context, query = _rows(1, 7), _rows(2, 3)

    with_norm = attend(normed, query, project_kv(normed, context))
    without = attend(plain, query, project_kv(plain, context))
    assert not np.allclose(np.asarray(with_norm), np.asarray(without))


def test_context_norm_normalizes_only_the_context() -> None:
    keys = jax.random.split(jax.random.key(0), 4)
    query_norm = RMSNormParameters(weight=jnp.full(MODEL, 2.0))
    context_norm = RMSNormParameters(weight=jnp.full(MODEL, 0.5))
    feed_forward = MLPParameters(
        input=_linear(keys[1], 2 * MODEL, MODEL),
        output=_linear(keys[2], MODEL, 2 * MODEL),
    )
    block = AttentionBlockParameters(
        attention_norm=query_norm,
        feed_forward_norm=query_norm,
        attention=_attention(keys[0]),
        feed_forward=feed_forward,
        context_norm=context_norm,
    )
    context, query = _rows(1, 7), _rows(2, 3)

    cache = project_kv(block.attention, normalize(context_norm, context))
    hidden = query + attend(block.attention, normalize(query_norm, query), cache)
    expected = hidden + mlp(feed_forward, normalize(query_norm, hidden))
    _close(attention_block(block, query, block_kv(block, context)), expected)


def test_log_scale_offset_adds_to_the_log_factor() -> None:
    head_scale = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    offset = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    query = jax.random.normal(jax.random.key(3), (2, 3, HEADS, HEAD_DIM))

    scaled = log_scale(LogScaleParameters(head_scale, offset), query, 7)

    factor = offset + np.log(7) * head_scale
    _close(scaled, query * factor[:, None])


def _token_attention(key: jax.Array) -> AttentionParameters:
    parameters = _attention(key, qk_norm=True)
    return dataclasses.replace(
        parameters,
        scale=LogScaleParameters(
            jax.random.uniform(key, (HEADS,)), jnp.arange(1.0, HEADS + 1)
        ),
    )


def test_token_axes_attend_token_by_token() -> None:
    parameters = _token_attention(jax.random.key(4))
    tokens = 3
    context = jax.random.normal(jax.random.key(5), (2, 7, tokens, MODEL))
    query = jax.random.normal(jax.random.key(6), (2, 4, tokens, MODEL))
    length = jnp.asarray([7, 5])

    attended = attend(parameters, query, project_kv(parameters, context, length))

    for token in range(tokens):
        cache = project_kv(parameters, context[:, :, token], length)
        _close(attended[:, :, token], attend(parameters, query[:, :, token], cache))


def test_attention_without_an_output_projection_returns_the_heads() -> None:
    parameters = _attention(jax.random.key(7))
    bare = dataclasses.replace(parameters, output=None)
    context, query = _rows(8, 5), _rows(9, 3)
    cache = project_kv(parameters, context)

    heads = attend(bare, query, cache)

    assert parameters.output is not None
    _close(linear(parameters.output, heads), attend(parameters, query, cache))
