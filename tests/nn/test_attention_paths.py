# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention blocks, static settings and the PyTorch QKV layout, without PyTorch fixtures."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.nn.attention import (
    AttentionBlockParameters,
    AttentionParameters,
    attend,
    attention_block,
    block_kv,
    project_kv,
    unpack_qkv,
)
from generalforecast.nn.linear import LinearParameters, MLPParameters
from generalforecast.nn.norm import LayerNormParameters
from generalforecast.nn.qassmax import QASSMaxParameters
from generalforecast.nn.rope import RoPE

MODEL = 16
HEADS = 4
HEAD_DIM = MODEL // HEADS


def _linear(key: jax.Array, outputs: int, inputs: int) -> LinearParameters:
    weight, bias = jax.random.normal(key, (2, outputs, inputs))
    return LinearParameters(weight=weight / np.sqrt(inputs), bias=bias[:, 0])


def _mlp(key: jax.Array, inputs: int, hidden: int, outputs: int) -> MLPParameters:
    first, second = jax.random.split(key)
    return MLPParameters(
        input=_linear(first, hidden, inputs), output=_linear(second, outputs, hidden)
    )


def _attention(
    key: jax.Array, *, scaled: bool = False, rope: RoPE | None = None
) -> AttentionParameters:
    query, key_, value, output, base, modulation = jax.random.split(key, 6)
    scale = None
    if scaled:
        scale = QASSMaxParameters(
            base=_mlp(base, 1, 8, MODEL),
            query=_mlp(modulation, HEAD_DIM, 8, HEAD_DIM),
        )
    return AttentionParameters(
        query=_linear(query, MODEL, MODEL),
        key=_linear(key_, MODEL, MODEL),
        value=_linear(value, MODEL, MODEL),
        output=_linear(output, MODEL, MODEL),
        num_heads=HEADS,
        scale=scale,
        rope=rope,
    )


def _block(key: jax.Array, *, scaled: bool) -> AttentionBlockParameters:
    attention_key, feed_forward_key = jax.random.split(key)
    norm = LayerNormParameters(weight=jnp.ones(MODEL), bias=jnp.zeros(MODEL))
    return AttentionBlockParameters(
        attention_norm=norm,
        feed_forward_norm=norm,
        attention=_attention(attention_key, scaled=scaled),
        feed_forward=_mlp(feed_forward_key, MODEL, 2 * MODEL, MODEL),
    )


def _rows(seed: int, rows: int) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), (2, rows, MODEL))


def _close(actual: jax.Array, expected: jax.Array) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("scaled", [False, True])
def test_query_rows_attend_only_to_the_context(scaled: bool) -> None:
    parameters = _block(jax.random.key(0), scaled=scaled)
    context, query = _rows(1, 7), _rows(2, 3)
    cache = block_kv(parameters, context)

    alone = attention_block(parameters, query[:, :1], cache)
    together = attention_block(parameters, query, cache)

    _close(together[:, :1], alone)


def test_static_settings_are_not_array_leaves() -> None:
    rope = RoPE(theta=100.0, interleaved=True)
    parameters = _attention(jax.random.key(0), rope=rope)

    leaves = jax.tree.leaves(parameters)

    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert len(leaves) == 8
    other = dataclasses.replace(parameters, rope=None)
    assert jax.tree.structure(parameters) != jax.tree.structure(other)


def test_attention_is_jittable_with_static_settings() -> None:
    parameters = _attention(
        jax.random.key(0), rope=RoPE(theta=100.0, interleaved=False)
    )
    context, query = _rows(1, 7), _rows(2, 3)

    def run(parameters: AttentionParameters, query: jax.Array, context: jax.Array):
        return attend(parameters, query, project_kv(parameters, context))

    _close(jax.jit(run)(parameters, query, context), run(parameters, query, context))


def test_rope_layouts_differ() -> None:
    context = _rows(1, 7)
    outputs = []
    for interleaved in (False, True):
        parameters = _attention(
            jax.random.key(0), rope=RoPE(theta=100.0, interleaved=interleaved)
        )
        outputs.append(attend(parameters, context, project_kv(parameters, context)))

    assert not np.allclose(np.asarray(outputs[0]), np.asarray(outputs[1]))


def test_rope_with_qassmax_is_rejected() -> None:
    parameters = _attention(
        jax.random.key(0), scaled=True, rope=RoPE(theta=100.0, interleaved=True)
    )
    context = _rows(1, 7)

    with pytest.raises(ValueError, match="RoPE and QASSMax"):
        attend(parameters, context, project_kv(parameters, context))


def test_unpack_qkv_splits_query_key_value_rows() -> None:
    unpacked = _attention(jax.random.key(0), scaled=True)
    biases = (unpacked.query.bias, unpacked.key.bias, unpacked.value.bias)
    assert all(bias is not None for bias in biases)
    assert unpacked.output is not None
    packed = unpack_qkv(
        jnp.concatenate(
            [unpacked.query.weight, unpacked.key.weight, unpacked.value.weight]
        ),
        jnp.concatenate([bias for bias in biases if bias is not None]),
        unpacked.output,
        num_heads=HEADS,
        scale=unpacked.scale,
    )
    context, query = _rows(1, 7), _rows(2, 3)

    _close(
        attend(packed, query, project_kv(packed, context)),
        attend(unpacked, query, project_kv(unpacked, context)),
    )
