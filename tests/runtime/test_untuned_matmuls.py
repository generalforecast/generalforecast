# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Programs compiled without autotuning send their matrix multiplies to cuBLAS, and predict as tuned ones do."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.mitra.model import mitra_functions
from generalforecast.runtime import budget, memory
from generalforecast.runtime.device import (
    UNTUNED_OPTIONS,
    device_settings,
    to_cache_dtype,
    untuned_matmuls,
    with_precision,
)
from tests.models.mitra.tiny import tiny_mitra


def test_untuned_options_turn_off_autotuning_and_tritons_matmuls() -> None:
    assert UNTUNED_OPTIONS == {
        "xla_gpu_autotune_level": 0,
        "xla_gpu_enable_triton_gemm": False,
    }
    # Calls compiled untuned, and measuring compiles, take the same options.
    assert budget.UNTUNED_OPTIONS is UNTUNED_OPTIONS
    assert memory.MEASURE_OPTIONS is UNTUNED_OPTIONS


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ("", ""),
        ("--xla_dump_to=/tmp/x", "--xla_dump_to=/tmp/x"),
        (
            "--xla_gpu_autotune_level=0",
            "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false",
        ),
        ("--xla_gpu_autotune_level=4", "--xla_gpu_autotune_level=4"),
        # The last level given is the one XLA takes.
        (
            "--xla_gpu_autotune_level=4 --xla_gpu_autotune_level=0",
            (
                "--xla_gpu_autotune_level=4 --xla_gpu_autotune_level=0"
                " --xla_gpu_enable_triton_gemm=false"
            ),
        ),
        (
            "--xla_gpu_autotune_level=0 --xla_gpu_autotune_level=1",
            "--xla_gpu_autotune_level=0 --xla_gpu_autotune_level=1",
        ),
        # A run that says where its matrix multiplies go keeps its say.
        (
            "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=true",
            "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=true",
        ),
    ],
)
def test_a_process_without_autotuning_sends_matmuls_to_cublas(
    monkeypatch: pytest.MonkeyPatch, flags: str, expected: str
) -> None:
    monkeypatch.setenv("XLA_FLAGS", flags)
    untuned_matmuls()
    assert os.environ["XLA_FLAGS"] == expected


def test_a_fused_retrieved_context_predicts_alike_untuned() -> None:
    """Mitra's encode and decode of per-query contexts, as `_encode_decode` runs them, compiled both ways."""
    settings = device_settings(jax.default_backend(), exact=True)
    model = tiny_mitra(regression=True)
    generator = np.random.default_rng(0)
    rows, contexts, size, queries = 96, 3, 64, 8
    features = jnp.asarray(generator.normal(size=(rows, 5)), jnp.float32)
    targets = jnp.asarray(generator.normal(size=rows), jnp.float32)
    state, values = mitra_functions.fit(model, features, targets, 0)
    query_rows = mitra_functions.embed(model, state, features[:queries])
    # A shared core, then each context's own neighbors, as a retrieved selection's.
    core = np.arange(16)
    index = np.stack(
        [
            np.concatenate(
                [core, generator.choice(np.arange(16, rows), size - 16, replace=False)]
            )
            for _ in range(contexts)
        ]
    ).astype(np.int32)
    lengths = np.full(contexts, size, np.int32)
    reads = np.stack([np.arange(queries)] * contexts).astype(np.int32)

    def one(parameters, state, values, targets, rows, index, length, queries):
        context = jnp.take(values, index, axis=0, mode="fill", fill_value=0)
        labels = jnp.take(targets, index, axis=0, mode="fill", fill_value=0)
        cache = mitra_functions.encode(
            parameters, state, context, labels, num_rows=length
        )
        cache = to_cache_dtype(cache, settings.cache_dtype)
        outputs = mitra_functions.decode(parameters, state, cache, rows[queries])
        return mitra_functions.postprocess(parameters, state, outputs)

    program = jax.jit(
        with_precision(
            jax.vmap(one, in_axes=(None,) * 5 + (0,) * 3), settings.matmul_precision
        )
    )
    arguments = (model, state, values, targets, query_rows, index, lengths, reads)
    lowered = program.lower(*arguments)
    tuned = np.asarray(lowered.compile()(*arguments))
    untuned = np.asarray(lowered.compile(compiler_options=UNTUNED_OPTIONS)(*arguments))
    assert tuned.shape == (contexts, queries, 999)
    assert np.isfinite(tuned).all()
    spread = np.abs(tuned).max()
    np.testing.assert_allclose(untuned, tuned, rtol=1e-4, atol=1e-4 * spread)
