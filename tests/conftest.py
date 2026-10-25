# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

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
os.environ.setdefault("GF_WORKER_MODULES", "tests.models.implementations")

import tests.models.implementations  # noqa: F401 - registers the suite's small models

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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Marks the tests under `SLOW` as slow, and deselects `conformance` tests
    unless GF_CONFORMANCE_URL names a server to replay against; never skips them.
    """
    for item in items:
        if item.nodeid.startswith(SLOW):
            item.add_marker(pytest.mark.slow)
    if os.environ.get("GF_CONFORMANCE_URL"):
        return
    marked = [item for item in items if item.get_closest_marker("conformance")]
    if not marked:
        return
    expression = config.getoption("markexpr") or ""
    if "conformance" in expression and "not conformance" not in expression:
        raise pytest.UsageError(
            "-m conformance needs GF_CONFORMANCE_URL to point at a gf serve"
        )
    for item in marked:
        items.remove(item)
    config.hook.pytest_deselected(items=marked)


@pytest.fixture(scope="session")
def gf_server() -> str:
    """Builds gf-server and puts it on PATH, where a runtime finds its front."""
    from tests.utils.server import build

    directory = str(build().parent)
    os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
    return directory


@pytest.fixture
def float64() -> Iterator[None]:
    """Enable float64 for tests of code that computes in double precision."""
    with jax.enable_x64(True):
        yield
