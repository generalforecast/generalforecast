# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Iterator
from pathlib import Path

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

# Tests that compile models or start servers; CI runs them in jobs of their
# own, split over runners, and a plain pytest runs them.
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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="K/N",
        help="run the K-th of N parts of the selected tests, as CI's runners split them",
    )
    parser.addoption(
        "--record-durations",
        metavar="PATH",
        help="write each test's seconds to PATH as JSON, for scripts/durations.py",
    )


# Seconds each test took in CI, which scripts/durations.py writes from the
# parts' recorded durations; `--shard` balances the parts by them.
DURATIONS = Path(__file__).with_name("durations.json")


def _durations() -> dict[str, float]:
    try:
        return json.loads(DURATIONS.read_text())
    except FileNotFoundError:
        return {}


def _unit(item: pytest.Item) -> str:
    """The tests that go to one part: a module's that share one of its fixtures, else one."""
    info = getattr(item, "_fixtureinfo", None)
    shared = info is not None and any(
        definitions[-1].scope == "module"
        for definitions in info.name2fixturedefs.values()
    )
    return item.nodeid.split("::")[0] if shared else item.nodeid


class _Shard:
    """Keeps the K-th of N parts of the selected tests under `--shard K/N`."""

    def __init__(self, part: int, count: int) -> None:
        self.part, self.count = part, count

    # Last, so the parts split what `-m` and `-k` selected. Tests sharing a
    # module's fixture stay together, so one part sets it up; the longest
    # units go first, each to the part with the least time yet. A test
    # without a recorded duration counts as the mean one.
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        recorded = _durations()
        typical = statistics.fmean(recorded.values()) if recorded else 1.0
        units: dict[str, list[pytest.Item]] = {}
        for item in items:
            units.setdefault(_unit(item), []).append(item)

        def seconds(unit: list[pytest.Item]) -> float:
            return sum(recorded.get(item.nodeid, typical) for item in unit)

        parts: list[list[pytest.Item]] = [[] for _ in range(self.count)]
        totals = [0.0] * self.count
        for unit in sorted(units.values(), key=lambda u: (-seconds(u), u[0].nodeid)):
            emptiest = min(range(self.count), key=lambda index: (totals[index], index))
            parts[emptiest].extend(unit)
            totals[emptiest] += seconds(unit)
        kept = {id(item) for item in parts[self.part - 1]}
        left = [item for item in items if id(item) not in kept]
        items[:] = [item for item in items if id(item) in kept]
        config.hook.pytest_deselected(items=left)


class _Recorder:
    """Adds up each test's setup, call and teardown seconds, and writes them at the end."""

    def __init__(self, path: Path) -> None:
        self.path, self.seconds = path, {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.seconds[report.nodeid] = (
            self.seconds.get(report.nodeid, 0.0) + report.duration
        )

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if hasattr(session.config, "workerinput"):
            return
        self.path.write_text(json.dumps(self.seconds, indent=0, sort_keys=True) + "\n")


def pytest_configure(config: pytest.Config) -> None:
    if (record := config.getoption("record_durations")) is not None:
        config.pluginmanager.register(_Recorder(Path(record)), "gf-durations")
    shard = config.getoption("shard")
    if shard is None:
        return
    part, _, count = shard.partition("/")
    if not (part.isdigit() and count.isdigit() and 1 <= int(part) <= int(count)):
        raise pytest.UsageError(f"--shard takes K/N with 1 <= K <= N, not {shard!r}")
    config.pluginmanager.register(_Shard(int(part), int(count)), "gf-shard")


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


@pytest.fixture(autouse=True)
def _servers_need_their_marker(request):
    """Lets the test start servers only when it is marked `server` or `router`."""
    from tests.utils import processes

    processes.MARKED = any(
        request.node.get_closest_marker(name) for name in ("server", "router")
    )
    yield
    processes.MARKED = None
