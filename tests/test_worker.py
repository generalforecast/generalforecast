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
from tests.models.conformance.cases import CASES, FEATURES
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


def test_each_call_a_step_finishes_tells_of_its_progress() -> None:
    # The server counts a worker silent for long on a step as hung, so a step
    # of many calls compiled long before says it is moving as each finishes.
    worker = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    _, step = _fused(worker, 2)
    worker.run(step([0, 1]))
    calls = sum(worker.stats.calls.values())
    told: list[None] = []
    worker.progressed = lambda: told.append(None)

    worker.run(step([0, 1]))

    assert calls > 0
    assert len(told) == sum(worker.stats.calls.values()) - calls


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


def test_a_measure_threads_share_runs_once_and_holds_up_no_other() -> None:
    import threading

    from generalforecast.worker.records import Memo

    memo: Memo[str, int] = Memo()
    started, release = threading.Event(), threading.Event()
    measured: list[str] = []

    def slow() -> int:
        measured.append("slow")
        started.set()
        assert release.wait(10)
        return 1

    threads = [threading.Thread(target=memo.get, args=("slow", slow)) for _ in range(2)]
    threads[0].start()
    assert started.wait(10)
    threads[1].start()
    # Another key measures while the first is still measuring.
    assert memo.get("fast", lambda: 2) == 2
    release.set()
    for thread in threads:
        thread.join()

    assert measured == ["slow"]
    assert memo["slow"] == 1


def test_a_measure_that_raises_is_taken_again() -> None:
    from generalforecast.worker.records import Memo

    memo: Memo[str, int] = Memo()
    attempts: list[int] = []

    def flaky() -> int:
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("no memory analysis yet")
        return 3

    with pytest.raises(ValueError):
        memo.get("key", flaky)
    assert memo.get("key", flaky) == 3
    assert len(attempts) == 2


def test_a_start_compiles_the_most_recently_used_programs_and_the_rest_ahead(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generalforecast.worker import ahead

    # Two programs at start, so the test compiles few.
    started = 2
    monkeypatch.setattr(ahead, "STARTED_PROGRAMS", started)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    first = Worker(slot_bytes=4096, compile_ahead=False)
    _, step = _fused(first, 1)
    # As `worker_process` loads models: by spec, so their calls are recorded.
    first._specs[0] = "tiny-tabicl"
    first.run(step([0]))
    manifest = first.programs.manifest("tiny-tabicl")
    assert manifest is not None
    # Each kind as if it took 8 items too: four programs each to start with.
    for kind in list(manifest.read().kinds):
        manifest.add(kind, 8)
    recorded = manifest.read()
    total = sum(recorded.programs(kind) for kind in recorded.kinds)
    assert total > started

    later = Worker(slot_bytes=4096)
    later.add_model(0, tabicl_functions, tiny_tabicl(regression=False))
    later._specs[0] = "tiny-tabicl"
    later.allocate(256, 0, 0)
    later.programs.quiet = 3600.0
    reported: list[tuple[int, int]] = []

    later.warm_start(lambda done, of: reported.append((done, of)))

    assert later.programs.warmed == started
    assert reported[-1] == (started, started)
    # The rest wait to compile ahead while the worker serves.
    assert len(later.programs._later) == total - started
    later.programs.drop()


def test_a_recorded_program_short_of_memory_at_start_is_skipped_by_later_starts(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generalforecast.worker import ahead

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    first = Worker(slot_bytes=4096, compile_ahead=False)
    _, step = _fused(first, 1)
    first._specs[0] = "tiny-tabicl"
    expected = first.run(step([0]))
    manifest = first.programs.manifest("tiny-tabicl")
    assert manifest is not None and manifest.read().kinds

    def started() -> tuple[Worker, Any]:
        later = Worker(slot_bytes=4096)
        _, run = _fused(later, 1)
        later._specs[0] = "tiny-tabicl"
        later.programs.quiet = 3600.0
        later.warm_start()
        return later, run

    # The device has no room for any recorded program at this start.
    real = ahead.note_compiled

    def short(*arguments: Any) -> None:
        raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory")

    monkeypatch.setattr(ahead, "note_compiled", short)
    crowded, _ = started()
    assert crowded.programs.warmed == 0
    crowded.programs.drop()
    recorded = manifest.read()
    assert set(recorded.short) == set(recorded.kinds)

    # A later start on this device compiles none of them again.
    monkeypatch.setattr(ahead, "note_compiled", real)
    later, run = started()
    assert later.programs.warmed == 0 and not later.programs._later
    # A step that needs one still compiles it, and answers as before.
    np.testing.assert_array_equal(later.run(run([0]))[0], expected[0])
    later.programs.drop()


def test_a_start_past_its_seconds_leaves_the_rest_to_compile_ahead(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generalforecast.worker import ahead

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    first = Worker(slot_bytes=4096, compile_ahead=False)
    _, step = _fused(first, 1)
    first._specs[0] = "tiny-tabicl"
    first.run(step([0]))
    manifest = first.programs.manifest("tiny-tabicl")
    assert manifest is not None
    recorded = manifest.read()
    total = sum(recorded.programs(kind) for kind in recorded.kinds)
    ran: list[Any] = []
    monkeypatch.setattr(ahead, "zeros", lambda tree: ran.append(tree))

    def started(seconds: float | None) -> Worker:
        later = Worker(slot_bytes=4096)
        later.add_model(0, tabicl_functions, tiny_tabicl(regression=False))
        later._specs[0] = "tiny-tabicl"
        later.allocate(256, 0, 0)
        later.programs.quiet = 3600.0
        later.warm_start(seconds=seconds)
        return later

    # Out of time at once: nothing compiles before steps, all of it ahead.
    late = started(0.0)
    assert late.programs.warmed == 0
    assert len(late.programs._later) == total
    late.programs.drop()
    # Zeros past `WARM_RUN_BYTES` are never made: those programs only compile.
    monkeypatch.setattr(ahead, "WARM_RUN_BYTES", 0)
    timely = started(None)
    assert timely.programs.warmed == min(total, ahead.STARTED_PROGRAMS)
    assert ran == []
    timely.programs.drop()


def test_a_warm_start_runs_on_zeros_only_programs_whose_temporaries_fit(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generalforecast.worker import ahead

    # Zeros of a few KiB, outputs of 16 MiB: what XLA counts is held too.
    outer = jax.jit(lambda x: jnp.outer(x, x))
    shapes = jax.ShapeDtypeStruct((2048,), jnp.float32)
    compiled = outer.lower(shapes).compile()
    held = ahead._run_bytes(compiled, shapes)
    assert held >= 2048 * 2048 * 4 > 64 * ahead._zeros_bytes(shapes)

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    first = Worker(slot_bytes=4096, compile_ahead=False)
    _, step = _fused(first, 1)
    first._specs[0] = "tiny-tabicl"
    first.run(step([0]))
    ran: list[Any] = []
    monkeypatch.setattr(ahead, "zeros", lambda tree: ran.append(tree))
    # Every recorded program's run would take more than a warm may: each
    # still compiles, and none runs.
    monkeypatch.setattr(
        ahead, "_run_bytes", lambda compiled, tree: ahead.WARM_RUN_BYTES + 1
    )
    later = Worker(slot_bytes=4096)
    later.add_model(0, tabicl_functions, tiny_tabicl(regression=False))
    later._specs[0] = "tiny-tabicl"
    later.allocate(256, 0, 0)
    later.programs.quiet = 3600.0
    later.warm_start()
    assert later.programs.warmed > 0
    assert ran == []
    later.programs.drop()


def test_a_worker_loads_a_model_beside_its_steps_and_reports_it_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import queue
    import threading

    from generalforecast import worker_process
    from generalforecast.models import registry

    release = threading.Event()

    def load(spec: str) -> tuple[Any, Any]:
        if spec == "broken":
            raise OSError("no weights")
        assert release.wait(10)
        return tabicl_functions, tiny_tabicl(regression=False)

    monkeypatch.setattr(registry, "load_engine_model", load)
    worker = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    worker.add_model(0, tabicl_functions, tiny_tabicl(regression=False))
    worker.allocate(16, 0, 0)
    names = {"served": 0}
    sent: queue.Queue[dict[str, Any]] = queue.Queue()
    loads = worker_process._Loads(worker, names, sent.put, lambda *_: None)

    def answer(header: dict[str, Any]) -> dict[str, Any]:
        replied, _ = worker_process._answer(
            worker, names, lambda *_: None, header, [], loads=loads
        )()
        loads.start()
        return replied

    loading = answer({"kind": "add_model", "name": "slow", "spec": "tiny"})
    assert loading == {"kind": "loading"}
    # The served model's requests are answered while the other loads.
    profiled = answer({"kind": "profile", "model": "served", "features": 5})
    assert profiled["kind"] == "profile"
    assert sent.empty() and "slow" not in names

    release.set()
    assert sent.get(timeout=10) == {"kind": "loaded", "name": "slow"}
    assert names["slow"] == 1
    assert answer({"kind": "add_model", "name": "broken", "spec": "broken"}) == {
        "kind": "loading"
    }
    assert sent.get(timeout=10) == {
        "kind": "loaded",
        "name": "broken",
        "failure": "OSError: no weights",
    }
    assert "broken" not in names


def _kept_state(case: Any, fused: bool) -> Any:
    """The state a worker keeps for a table of 300 rows, fit with its queries or alone."""
    worker = Worker(slot_bytes=4096, compile_ahead=False, compile_cache=False)
    worker.add_model(0, case.functions(), case.build(0))
    worker.allocate(256, 0, 0)
    data = conformance_table(case.regression, rows=300, queries=6, features=FEATURES)
    worker.add_table(0, 0, data.num_classes, FEATURES)
    features, targets = np.asarray(data.features), np.asarray(data.targets)
    rows = np.arange(300, dtype=np.uint32)
    if fused:
        key = CacheKey(0, 0, 0, "context", 1)
        unit = protocol.Unit(0, key, rows, "fuse", [], features, targets)
        queries = np.asarray(data.queries)
        batch = protocol.QueryBatch(0, 0, 0, queries, [0] * len(queries))
        worker.run(protocol.Step([], [], [], [unit], [batch]))
    else:
        fit = protocol.Fit(0, 0, rows, features, targets, [])
        worker.run(protocol.Step([], [fit], [], [], []))
    return worker._tables[0].state


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_a_context_fit_with_its_queries_keeps_the_state_a_fit_alone_would(
    case: Any,
) -> None:
    fused, alone = _kept_state(case, True), _kept_state(case, False)

    # To the bit, so a context reused after its first answer reads the state
    # a fit of it alone would, whichever came first.
    for kept, fit in zip(jax.tree.leaves(fused), jax.tree.leaves(alone), strict=True):
        assert np.asarray(kept).tobytes() == np.asarray(fit).tobytes()


def test_a_worker_whose_server_is_gone_exits_quietly():
    """Ctrl-C can stop gf-server while a restarted worker is still loading."""
    import os
    import subprocess
    import sys
    import tempfile

    # A socket path short enough for AF_UNIX, as tests/serve/test_shutdown.py makes.
    with tempfile.TemporaryDirectory(prefix="gfsock") as short:
        gone = os.path.join(short, "gone.sock")
        finished = subprocess.run(
            [
                sys.executable,
                "-m",
                "generalforecast.worker_process",
                "--socket",
                gone,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    assert finished.returncode == 0, finished.stderr
    assert "Traceback" not in finished.stderr, finished.stderr


def test_a_fit_profile_is_kept_for_later_processes_unless_unmeasured(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generalforecast.runtime.memory import Cost, FitProfile
    from generalforecast.worker import admission

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    measured: list[int] = []
    # Widths XLA measures, by their costs; any other it cannot measure.
    costs = {FEATURES: (Cost(1.0, 2.0), Cost(3.0, 4.0))}

    def measure(functions: Any, model: Any, width: int, cache: Any) -> FitProfile:
        measured.append(width)
        return FitProfile(width, *costs.get(width, ()))

    def started() -> Worker:
        worker = Worker()
        worker.add_model(
            0, tabicl_functions, tiny_tabicl(regression=False), spec="tiny"
        )
        return worker

    monkeypatch.setattr(admission, "fit_profile", measure)
    expected = FitProfile(FEATURES, *costs[FEATURES])
    assert started().fit_profile(0, FEATURES) == expected
    # A later process reads it without measuring.
    assert started().fit_profile(0, FEATURES) == expected
    assert measured == [FEATURES]

    # One XLA could not measure is measured again by the next process.
    wider = FEATURES + 1
    assert started().fit_profile(0, wider) == FitProfile(wider)
    started().fit_profile(0, wider)
    assert measured == [FEATURES, wider, wider]


def test_a_fit_profile_kept_before_steps_were_measured_is_measured_again(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generalforecast.runtime.memory import Cost, FitProfile
    from generalforecast.worker import admission

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    worker = Worker()
    worker.add_model(0, tabicl_functions, tiny_tabicl(regression=False), spec="tiny")
    # As a process kept it before fit profiles measured a step: two costs.
    old = FitProfile(FEATURES, Cost(1.0, 2.0), Cost(3.0, 4.0))
    worker.programs.measured(
        "tiny", ("fit_profile", FEATURES, repr(worker._settings)), lambda: old
    )
    fresh = FitProfile(FEATURES, Cost(1.0, 2.0), Cost(3.0, 4.0), Cost(0.0, 5.0))
    monkeypatch.setattr(admission, "fit_profile", lambda *_: fresh)

    assert worker.fit_profile(0, FEATURES) == fresh


def test_queries_group_by_the_unit_they_read_in_order_of_first_read() -> None:
    from generalforecast.worker.decodes import _groups

    assert _groups([]) == {}
    assert _groups([4, 4, 4]) == {4: [0, 1, 2]}
    assert _groups([3, 1, 3, 2, 1]) == {3: [0, 2], 1: [1, 4], 2: [3]}
    assert list(_groups([3, 1, 3, 2, 1])) == [3, 1, 2]


def test_requests_asking_the_same_levels_fetch_the_columns_those_levels_read() -> None:
    from generalforecast.worker.decodes import _picks

    known = np.arange(1, 1000) / 1000
    levels = np.asarray([0.1, 0.5, 0.9])

    def batch(request: int, asked: np.ndarray | None) -> protocol.QueryBatch:
        return protocol.QueryBatch(
            request, 0, 0, np.zeros((2, 1), np.float32), [0, 0], known, asked
        )

    same = _picks([batch(1, levels), batch(2, levels)], [[(1, 0, 2), (2, 2, 4)]])
    columns, at, asked = same[0]
    np.testing.assert_array_equal(columns, protocol.level_columns(known, levels))
    np.testing.assert_array_equal(at, known[columns])
    np.testing.assert_array_equal(asked, levels)
    # Requests asking different levels, or none, keep every column.
    assert not _picks(
        [batch(1, levels), batch(2, levels[:2])], [[(1, 0, 2), (2, 2, 4)]]
    )
    assert not _picks([batch(1, None)], [[(1, 0, 2)]])


def test_a_layout_hashes_once_and_pickles_without_its_hash() -> None:
    import pickle

    from generalforecast.runtime.paging import StateLayout

    layout = StateLayout(None, (jax.ShapeDtypeStruct((2, 3), jnp.float32),), 1 << 20)
    same = StateLayout(None, (jax.ShapeDtypeStruct((2, 3), jnp.float32),), 1 << 20)

    assert hash(layout) == hash(same) and layout == same
    assert "_hash" in layout.__dict__
    copy = pickle.loads(pickle.dumps(layout))
    assert "_hash" not in copy.__dict__ and copy == layout
    assert hash(copy) == hash(layout)


def test_a_step_of_several_requests_that_runs_out_of_memory_runs_again_in_halves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = Worker(compile_cache=False, compile_ahead=False)
    computed: list[list[int]] = []

    def compute(step: protocol.Step) -> dict[int, np.ndarray]:
        requests = [batch.request for batch in step.queries]
        computed.append(requests)
        # The device holds the queries of two requests at once, but not more.
        if len(requests) > 2:
            raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory")
        return {request: np.full((1, 1), request, np.float32) for request in requests}

    monkeypatch.setattr(worker, "_compute", compute)
    batches = [
        protocol.QueryBatch(request, 0, 0, np.zeros((1, 1), np.float32), [0])
        for request in range(5)
    ]
    step = protocol.Step(movements=[], fits=[], pages=[], units=[], queries=batches)

    predictions = worker._answered(step)

    assert sorted(predictions) == [0, 1, 2, 3, 4]
    assert computed == [[0, 1, 2, 3, 4], [0, 1], [2, 3, 4], [2], [3, 4]]

    # A lone request that runs out still fails, as does an error of another kind.
    def refusing(step: protocol.Step) -> dict[int, np.ndarray]:
        raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory")

    monkeypatch.setattr(worker, "_compute", refusing)
    with pytest.raises(jax.errors.JaxRuntimeError):
        worker._answered(protocol.Step([], [], [], [], batches[:1]))

    def broken(step: protocol.Step) -> dict[int, np.ndarray]:
        computed.append([-1])
        raise jax.errors.JaxRuntimeError("INVALID_ARGUMENT: bad")

    computed.clear()
    monkeypatch.setattr(worker, "_compute", broken)
    with pytest.raises(jax.errors.JaxRuntimeError):
        worker._answered(step)
    assert computed == [[-1]]
