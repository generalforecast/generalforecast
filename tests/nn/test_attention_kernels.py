# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention kernels: query and block chunking, the attention dtype, and cuDNN on GPU."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.nn.attention import (
    AttentionCache,
    AttentionKernel,
    AttentionParameters,
    attend,
    in_chunks,
    project_kv,
)
from generalforecast.nn.induced import compress
from generalforecast.runtime.device import DeviceSettings, configure
from tests.models.tabicl.tiny import EMBED, HEADS, ICL, INDUCING, tiny_tabicl


def _attention(
    kernel: AttentionKernel, dtype: Any = jnp.float32
) -> AttentionParameters:
    """The tiny model's first ICL attention (with QASSMax), with `kernel`."""
    attention = tiny_tabicl(regression=False).icl.blocks[0].attention
    attention = jax.tree.map(lambda leaf: leaf.astype(dtype), attention)
    return dataclasses.replace(attention, kernel=kernel)


def _rows(shape: tuple[int, ...], seed: int, dtype: Any = jnp.float32) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), dtype=dtype)


def _keys(dtype: Any = jnp.float32) -> AttentionCache:
    """Keys and values for two batch rows with 7 and 12 real rows."""
    cache = project_kv(
        _attention(AttentionKernel(), dtype), _rows((2, 12, ICL), 0, dtype)
    )
    return cache._replace(length=jnp.asarray([7, 12], dtype=jnp.int32))


def _equations(jaxpr: Any) -> list[Any]:
    """Every equation in a jaxpr, including those in nested jaxprs."""
    found = []
    for equation in jaxpr.eqns:
        found.append(equation)
        for value in equation.params.values():
            for inner in value if isinstance(value, tuple | list) else (value,):
                inner = getattr(inner, "jaxpr", inner)
                if hasattr(inner, "eqns"):
                    found.extend(_equations(inner))
    return found


def _largest_intermediate(jaxpr: Any) -> int:
    """Most elements in any value a jaxpr computes."""
    return max(
        math.prod(getattr(variable.aval, "shape", ()))
        for equation in _equations(jaxpr)
        for variable in equation.outvars
    )


def _close(actual: jax.Array, expected: jax.Array, tolerance: float) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=tolerance,
        atol=tolerance,
    )


def test_kernel_is_static() -> None:
    attention = _attention(AttentionKernel(query_chunk=8, dtype=jnp.dtype("float16")))

    assert hash(attention.kernel) == hash(
        AttentionKernel(query_chunk=8, dtype=jnp.dtype("float16"))
    )
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(attention))


@pytest.mark.parametrize("queries", [10, 16], ids=["ragged", "even"])
def test_query_chunks_match_full_attention(queries: int) -> None:
    query_rows = _rows((2, queries, ICL), seed=1)

    chunked = attend(_attention(AttentionKernel(query_chunk=4)), query_rows, _keys())
    full = attend(_attention(AttentionKernel()), query_rows, _keys())

    _close(chunked, full, 1e-6)


def test_query_chunks_never_hold_every_query_key_score() -> None:
    rows = _rows((1, 64, ICL), seed=2)
    scores = HEADS * 64 * 64

    def largest(kernel: AttentionKernel) -> int:
        attention = _attention(kernel)
        jaxpr = jax.make_jaxpr(
            lambda values: attend(attention, values, project_kv(attention, values))
        )(rows)
        return _largest_intermediate(jaxpr.jaxpr)

    assert largest(AttentionKernel()) >= scores
    assert largest(AttentionKernel(query_chunk=8)) <= scores // 4
    # cuDNN falls back to XLA here, as on GPUs it cannot run on, and chunks.
    assert largest(AttentionKernel("cudnn", query_chunk=8)) <= scores // 4


@pytest.mark.parametrize("length", [10, 16], ids=["ragged", "even"])
def test_chunks_match_one_step(length: int) -> None:
    values, weights = _rows((3, length, 5), seed=5), _rows((5, 2), seed=6)

    def function(values: jax.Array, missing: None) -> jax.Array:
        return jnp.tanh(values @ weights)

    chunked = in_chunks(function, (values, None), 4, axis=1)

    _close(chunked, function(values, None), 1e-6)


def test_block_chunks_never_hold_every_inducing_score() -> None:
    block = tiny_tabicl(regression=False).columns.blocks[0]
    context = _rows((4, 64, EMBED), seed=7)
    scores = 4 * HEADS * INDUCING * 64

    def largest(chunk: int | None) -> int:
        settings = DeviceSettings(
            jnp.dtype("float32"),
            jnp.dtype("float32"),
            AttentionKernel(block_chunk=chunk),
        )
        chunked = configure(block, settings)
        return _largest_intermediate(
            jax.make_jaxpr(lambda rows: compress(chunked, rows))(context).jaxpr
        )

    assert largest(None) >= scores
    assert largest(64) <= scores // 4


def test_kernel_dtype_sets_only_the_attention_precision() -> None:
    kernel = AttentionKernel(dtype=jnp.dtype("float16"))
    attention = _attention(kernel)
    query_rows = _rows((2, 5, ICL), seed=3)

    jaxpr = jax.make_jaxpr(lambda rows: attend(attention, rows, _keys()))(query_rows)
    reduced = attend(attention, query_rows, _keys())

    dots = [
        equation
        for equation in _equations(jaxpr.jaxpr)
        if equation.primitive.name == "dot_general"
    ]
    assert any(
        all(operand.aval.dtype == jnp.float16 for operand in equation.invars)
        for equation in dots
    )
    assert reduced.dtype == jnp.float32
    _close(reduced, attend(_attention(AttentionKernel()), query_rows, _keys()), 1e-2)


def test_attention_reads_a_cache_stored_in_another_dtype() -> None:
    query_rows = _rows((2, 5, ICL), seed=4)
    keys = _keys()
    stored = keys._replace(
        key=keys.key.astype(jnp.float16), value=keys.value.astype(jnp.float16)
    )

    from_stored = attend(_attention(AttentionKernel()), query_rows, stored)

    assert from_stored.dtype == jnp.float32
    _close(from_stored, attend(_attention(AttentionKernel()), query_rows, keys), 1e-3)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="cuDNN needs a GPU")
def test_cudnn_matches_xla_in_float16() -> None:
    half = jnp.dtype("float16")
    query_rows = _rows((2, 64, ICL), seed=5)

    cudnn = attend(
        _attention(AttentionKernel("cudnn", dtype=half)), query_rows, _keys()
    )
    xla = attend(_attention(AttentionKernel(dtype=half)), query_rows, _keys())

    _close(cudnn, xla, 1e-2)


@pytest.mark.parametrize(
    ("generation", "head_dim", "expected"),
    [
        (None, 64, "xla"),  # no NVIDIA GPU
        ((7, 5), 64, "xla"),  # T4: cuDNN flash attention needs Ampere
        ((8, 0), 64, "cudnn"),
        ((8, 9), 192, "xla"),  # heads past 128 need Hopper
        ((9, 0), 192, "cudnn"),
    ],
)
def test_cudnn_only_where_it_runs(
    monkeypatch: pytest.MonkeyPatch,
    generation: tuple[int, int] | None,
    head_dim: int,
    expected: str,
) -> None:
    from generalforecast.nn import attention

    monkeypatch.setattr(attention, "_gpu_generation", lambda: generation)
    q = jnp.zeros((1, 4, 2, head_dim), jnp.float16)

    assert attention._implementation(AttentionKernel("cudnn"), q) == expected
