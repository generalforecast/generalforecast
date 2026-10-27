# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN attention programs that load from JAX's compile cache beside running steps, on a GPU.

Loading a program with a cuDNN graph deserializes the graph, and cuDNN's
frontend warms it up inside a CUDA graph capture. Before jaxlib 0.12 that
capture ran on the device's compute stream, so a step another thread ran on it
meanwhile broke the capture (`cudaErrorStreamCaptureInvalidated`).
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU"),
]


def _attention(
    q: jax.Array, k: jax.Array, v: jax.Array, lengths: jax.Array
) -> jax.Array:
    return jax.nn.dot_product_attention(
        q, k, v, key_value_seq_lengths=lengths, implementation="cudnn"
    )


def _arguments(rows: int) -> tuple[jax.Array, ...]:
    values = jnp.ones((4, rows, 4, 64), jnp.float16)
    return values, values, values, jnp.full((4,), rows, jnp.int32)


def stress(programs: int = 48, compilers: int = 2, runners: int = 2) -> list[Exception]:
    """Compiles and first runs `programs` distinct shapes beside steps; returns errors."""
    errors: list[Exception] = []
    done = threading.Event()
    steady = jax.jit(_attention)
    arguments = _arguments(512)
    np.asarray(steady(*arguments))

    def run() -> None:
        while not done.is_set():
            try:
                np.asarray(steady(*arguments))
            except Exception as error:  # noqa: BLE001
                errors.append(error)
                return

    def compile_(offset: int) -> None:
        for rows in range(16 * (offset + 1), 16 * (programs + 1), 16 * compilers):
            try:
                np.asarray(jax.jit(_attention)(*_arguments(rows)))
            except Exception as error:  # noqa: BLE001
                errors.append(error)
                return

    threads = [threading.Thread(target=run) for _ in range(runners)]
    for thread in threads:
        thread.start()
    compiling = [threading.Thread(target=compile_, args=(i,)) for i in range(compilers)]
    for thread in compiling:
        thread.start()
    for thread in compiling:
        thread.join()
    done.set()
    for thread in threads:
        thread.join()
    return errors


def test_cached_programs_load_beside_steps(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "JAX_COMPILATION_CACHE_DIR": str(tmp_path),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    # The first process fills the cache; the others load every program from it.
    for _ in range(4):
        process = subprocess.run(
            [sys.executable, __file__],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr[-2000:]


if __name__ == "__main__":
    for error in (errors := stress()):
        print(error)
    sys.exit(1 if errors else 0)
