# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast import protocol
from generalforecast._core import CacheKey
from generalforecast.constants import STACK_CHUNK
from generalforecast.models.conformance import table as conformance_table
from generalforecast.models.tabicl.model import tabicl_functions
from generalforecast.worker import Worker, _stacked
from tests.models.conformance.cases import FEATURES
from tests.models.tabicl.tiny import tiny_tabicl


def _item(index: int) -> tuple:
    return (
        jnp.full(3, index, dtype=jnp.float32),
        jnp.int32(index),
        np.full(2, index, dtype=np.float32),
    )


def test_a_stack_past_a_chunk_matches_a_stack_of_every_item() -> None:
    items = [_item(index) for index in range(2 * STACK_CHUNK + 5)]
    stacked = _stacked(items)
    for leaf, column in zip(stacked, zip(*items, strict=True), strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.stack(column))
        assert leaf.shape == (len(items), *np.shape(column[0]))


def test_a_new_count_past_a_chunk_compiles_only_its_join() -> None:
    _stacked([_item(index) for index in range(STACK_CHUNK)])
    compiled: list[str] = []

    def compile_event(event: str, duration_secs: float, **details: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(details.get("fun_name", ""))

    jax.config.update("jax_enable_compilation_cache", False)
    jax.monitoring.register_event_duration_secs_listener(compile_event)
    try:
        _stacked([_item(index) for index in range(16 * STACK_CHUNK)])
    finally:
        jax.monitoring.unregister_event_duration_listener(compile_event)
        jax.config.update("jax_enable_compilation_cache", True)
    assert compiled == ["jit(_join_columns)"]


def _fused(worker: Worker, tables: int) -> tuple[Any, Any]:
    """Tables of a tiny classifier, and a step fusing each named one with its queries."""
    data = conformance_table(False, rows=48, queries=6)
    worker.add_model(0, tabicl_functions, tiny_tabicl(regression=False))
    worker.allocate(256, 0, 0)
    for index in range(tables):
        worker.add_table(index, 0, data.num_classes, FEATURES)

    def step(indices: list[int]) -> protocol.Step:
        units, batches = [], []
        for at, index in enumerate(indices):
            key = CacheKey(index, 0, 0, "context", 1)
            rows = np.arange(48, dtype=np.uint32)
            features = np.asarray(data.features) + np.float32(index)
            targets = np.asarray(data.targets)
            units.append(protocol.Unit(index, key, rows, "fuse", [], features, targets))
            queries = np.asarray(data.queries)
            batches.append(
                protocol.QueryBatch(index, index, index, queries, [at] * len(queries))
            )
        return protocol.Step([], [], [], units, batches)

    return data, step


def test_a_worker_is_allocated_only_once_told_its_tiers() -> None:
    worker = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    assert not worker.allocated
    worker.forget_cache()
    assert not worker.allocated

    worker.allocate(4, 0, 0)

    assert worker.allocated
    worker.forget_cache()
    assert worker.allocated


def test_a_table_its_part_answers_after_the_whole_step_failed_is_not_deferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = Worker(slot_bytes=4096, compile_ahead=False)
    _, step = _fused(worker, 2)
    # Table 0's program is being readied as the whole step runs, and is ready
    # by the time its part runs alone; table 1 fails either way.
    readying = [True]

    def defers(unit: Any, arguments: tuple) -> bool:
        if unit.table == 0 and readying[0]:
            readying[0] = False
            worker.deferred[0] = "readying"
            return True
        return False

    called = worker._call

    def failing(stage: str, work: Any, consume: Any = None) -> list[Any]:
        if stage == "fit_decode" and work[0][0] == 1:
            raise RuntimeError("table 1 failed")
        return called(stage, work, consume)

    monkeypatch.setattr(worker, "_defers", defers)
    monkeypatch.setattr(worker, "_call", failing)

    predictions, failed = worker.run_apart(step([0, 1]))

    assert list(predictions) == [0]
    assert failed == {1: "RuntimeError: table 1 failed"}
    assert worker.deferred == {}


def test_a_failed_table_is_never_also_deferred(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = Worker(slot_bytes=4096, compile_ahead=False)
    _, step = _fused(worker, 2)

    def defers(unit: Any, arguments: tuple) -> bool:
        worker.deferred[unit.table] = "readying"
        if unit.table == 1:
            raise RuntimeError("table 1 failed")
        return True

    monkeypatch.setattr(worker, "_defers", defers)

    predictions, failed = worker.run_apart(step([0, 1]))

    assert predictions == {}
    assert failed == {1: "RuntimeError: table 1 failed"}
    assert worker.deferred == {0: "readying"}


def test_dropping_every_program_forgets_the_counts_each_kind_compiled_for() -> None:
    worker = Worker(slot_bytes=4096, compile_ahead=False)
    _, step = _fused(worker, 1)
    worker.run(step([0]))
    assert any(worker._chunks._counts.values())

    worker.programs.drop()

    assert not any(worker._chunks._counts.values())
    assert list(worker.run(step([0]))) == [0]


def test_a_drop_ends_the_page_embeds_a_table_has_not_compiled_yet() -> None:
    import time

    worker = Worker(slot_bytes=4096)
    data, step = _fused(worker, 1)
    worker.run(step([0]))
    table = worker._tables[0]
    features = np.asarray(data.features)[:16]
    targets = np.asarray(data.targets)[:16]
    compiled: list[str] = []

    def compile_event(event: str, duration_secs: float, **details: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(details.get("fun_name", ""))

    # No quiet gap comes, so every queued compile waits.
    worker.programs.quiet = 3600.0
    jax.config.update("jax_enable_compilation_cache", False)
    jax.monitoring.register_event_duration_secs_listener(compile_event)
    try:
        worker._compile_embeds(0, table.num_classes, (table.state, features, targets))
        assert compiled == []
        started = time.perf_counter()
        worker.programs.drop()
        assert time.perf_counter() - started < 5
    finally:
        jax.monitoring.unregister_event_duration_listener(compile_event)
        jax.config.update("jax_enable_compilation_cache", True)
    assert compiled == []
