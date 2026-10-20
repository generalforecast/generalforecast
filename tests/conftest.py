from __future__ import annotations

import os
from collections.abc import Iterator

import jax
import pytest

from generalforecast.runtime.device import use_compile_cache

# The tests check exact invariants (padding, batching, jit), which TF32 and
# float16 rounding would blur on GPUs: run models in float32, with float32
# matrix multiplies, everywhere, including in servers the tests start.
os.environ.setdefault("GF_PRECISION", "exact")
jax.config.update("jax_default_matmul_precision", "highest")
# Compiled programs stay on disk, as workers keep them, so a second run skips
# most compiles.
use_compile_cache()

# Tests that compile models or start servers; CI leaves them out, since its
# 2-core runner takes most of an hour on them, and a plain pytest runs them.
SLOW = (
    "tests/models/",
    "tests/test_estimators.py",
    "tests/test_local.py",
    "tests/test_remote.py",
    "tests/test_router.py",
    "tests/test_server.py",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Marks the tests under `SLOW` as slow."""
    for item in items:
        if item.nodeid.startswith(SLOW):
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def float64() -> Iterator[None]:
    """Enable float64 for tests of code that computes in double precision."""
    with jax.enable_x64(True):
        yield
