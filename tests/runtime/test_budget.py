# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.runtime import budget as budget_module
from generalforecast.runtime.budget import (
    UNTUNED_OPTIONS,
    Call,
    Chunks,
    MemoryBudget,
    estimated_flops,
    flops_of,
    note_compiled,
)


def test_local_clients_and_servers_divide_memory_their_own_ways() -> None:
    server = MemoryBudget(64 << 30, 256 << 30, cpu=False, workers=4)
    local = MemoryBudget(64 << 30, 256 << 30, cpu=False, local=True)
    laptop = MemoryBudget(4 << 30, 16 << 30, cpu=True, local=True)

    assert (server.device_tier, server.host_tier, server.row_memory) == (
        32 << 30,
        16 << 30,
        16 << 30,
    )
    # A local client shares its machine, so it keeps less, and at most 4 GiB.
    assert local.device_tier == 4 << 30
    assert laptop.device_tier == 1 << 30


def test_a_call_takes_half_what_the_pool_leaves_within_bounds() -> None:
    gpu = MemoryBudget(16 << 30, 64 << 30, cpu=False)
    cpu = MemoryBudget(16 << 30, 64 << 30, cpu=True)

    assert [gpu.call_bytes(pool << 30) for pool in (0, 12, 16)] == [
        4 << 30,
        2 << 30,
        1 << 30,
    ]
    assert cpu.call_bytes(0) == 2 << 30
    assert gpu.window_bytes(0) == 2 << 30


def test_memory_held_for_a_workers_life_leaves_the_rest_to_divide() -> None:
    # A 24 GB GPU's 16.5 GiB, 5 GiB of it the parameters of every catalog
    # model: the pool and a call take shares of what is left, so together
    # with the parameters they fit the device.
    unheld = MemoryBudget(16 << 30, 64 << 30, cpu=False)
    gpu = unheld.holding(5 << 30)
    cpu = MemoryBudget(16 << 30, 64 << 30, cpu=True)

    def taken(budget: MemoryBudget) -> int:
        pool = budget.device_tier
        return (5 << 30) + pool + 2 * budget.call_bytes(pool)

    assert gpu.device == 11 << 30
    assert taken(gpu) <= 16 << 30 < taken(unheld)
    assert cpu.holding(5 << 30) == cpu


@pytest.mark.parametrize(
    ("free", "short"), [(None, False), (8 << 30, True), (32 << 30, False)]
)
def test_memory_is_short_only_when_known_to_be(
    monkeypatch: pytest.MonkeyPatch, free: int | None, short: bool
) -> None:
    monkeypatch.setattr(budget_module, "available_memory", lambda: free)

    assert MemoryBudget(1 << 30, 64 << 30, cpu=True).short_of_memory() is short


def _run(chunks: Chunks, function: Any, items: np.ndarray, budget: int) -> list[int]:
    """Runs one item per row of `items` through `chunks`; returns each call's item count."""
    calls: list[int] = []

    def prepare(first: int, stop: int, padded: int) -> Call:
        rows = items[first:stop]
        stacked = np.concatenate([rows, np.repeat(rows[-1:], padded - len(rows), 0)])
        return Call(function, (), (stacked,), padded)

    def consume(first: int, stop: int, outputs: Any) -> None:
        calls.append(stop - first)

    chunks.run(
        "fit", "key", len(items), budget, lambda: items[0].nbytes, prepare, consume
    )
    return calls


def test_calls_are_sized_by_what_xla_says_they_hold() -> None:
    items = np.ones((6, 256), np.float32)
    # Each item's square matrix, held whole to sort, against inputs of a row
    # and outputs of one number.
    held = jax.jit(jax.vmap(lambda row: jnp.sort(jnp.outer(row, row).ravel())[0]))
    light = jax.jit(jax.vmap(lambda row: row.sum()))

    # Room for sixteen items by their inputs alone.
    budget = 16 * items[0].nbytes
    assert _run(Chunks(lambda *_: None), light, items, budget) == [6]
    assert _run(Chunks(lambda *_: None), held, items, budget) == [1] * 6


def test_a_call_that_runs_out_of_memory_runs_again_in_smaller_calls() -> None:
    sizes: list[int] = []

    def refusing(values: np.ndarray) -> np.ndarray:
        sizes.append(len(values))
        # The device holds a call of two items at once, but not more.
        if len(values) > 2:
            raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory")
        return values

    counted: list[str] = []
    chunks = Chunks(lambda stage, seconds, flops: counted.append(stage))
    items = np.ones((6, 4), np.float32)
    budget = 8 * items[0].nbytes

    assert _run(chunks, refusing, items, budget) == [2, 2, 2]
    # The call of all six failed, and halves until two at a time went through.
    assert sizes == [8, 4, 2, 2, 2] and counted == ["fit"] * 3
    sizes.clear()
    _run(chunks, refusing, items, budget)
    # Later calls start there, until the pool is sized again.
    assert sizes == [2, 2, 2]
    chunks.relax()
    sizes.clear()
    _run(chunks, refusing, items, budget)
    assert sizes == [8, 4, 2, 2, 2]


def test_an_item_that_alone_runs_out_of_memory_raises() -> None:
    def refusing(values: np.ndarray) -> np.ndarray:
        raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory")

    items = np.ones((3, 4), np.float32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="RESOURCE_EXHAUSTED"):
        _run(Chunks(lambda *_: None), refusing, items, 8 * items[0].nbytes)


def test_a_call_whose_autotuning_runs_out_of_memory_runs_compiled_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jitted = jax.jit(lambda values: values * 2)
    tuned: list[int] = []

    class Autotuned:
        """A program whose compile fails autotuning, as XLA's does when its buffers do not fit."""

        def __call__(self, values: Any) -> Any:
            tuned.append(len(values))
            raise jax.errors.JaxRuntimeError(
                "RESOURCE_EXHAUSTED: Failed to get configs for: 8 out of 24 "
                "instructions. Example failure: Out of memory while trying to "
                "allocate 24.00MiB"
            )

        def lower(self, *arguments: Any) -> Any:
            return jitted.lower(*arguments)

    compiled: list[dict] = []
    real = jax.stages.Lowered.compile

    def compile(lowered: Any, compiler_options: Any = None) -> Any:
        if compiler_options:
            compiled.append(compiler_options)
        return real(lowered, compiler_options=compiler_options)

    monkeypatch.setattr(jax.stages.Lowered, "compile", compile)
    items = np.arange(8, dtype=np.float32).reshape(2, 4)
    outputs: list[np.ndarray] = []
    chunks = Chunks(lambda *_: None)
    program = Autotuned()

    def prepare(first: int, stop: int, padded: int) -> Call:
        return Call(program, (), (items[first:stop],), padded)

    def consume(first: int, stop: int, values: Any) -> None:
        outputs.append(np.asarray(values))

    for _ in range(2):
        chunks.run("decode", "key", 2, 1 << 20, lambda: 32, prepare, consume)

    # The first call fails autotuning; the program compiles once without it
    # and serves both runs, whose answers are the program's.
    assert tuned == [2]
    assert compiled == [budget_module.UNTUNED_OPTIONS]
    for answer in outputs:
        np.testing.assert_array_equal(answer, items * 2)


def test_a_run_past_one_call_pads_its_last_call_to_the_chunk() -> None:
    # Four items fit a call: seven items run as a call of four and one of
    # three padded to four, so the program compiles for one shape, and each
    # item's output is what it gets alone.
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.arange(7, dtype=np.float32).reshape(7, 1)
    compiled: list[str] = []
    chunks = Chunks(lambda *_: None, compiled.append)
    outputs: list[np.ndarray] = []
    padded_counts: list[int] = []

    def prepare(first: int, stop: int, padded: int) -> Call:
        padded_counts.append(padded)
        rows = items[first:stop]
        stacked = np.concatenate([rows, np.repeat(rows[-1:], padded - len(rows), 0)])
        return Call(doubled, (), (stacked,), padded)

    def consume(first: int, stop: int, found: Any) -> None:
        outputs.extend(np.asarray(found)[: stop - first])

    # An estimate past what XLA measures, so the chunk stays four items.
    per_item, budget = 1 << 20, 4 << 20
    chunks.run("fit", "key", 7, budget, lambda: per_item, prepare, consume)
    assert padded_counts == [4, 4] and compiled == ["fit"]
    np.testing.assert_array_equal(np.stack(outputs), items * 2)
    # A run that fits one call pads to its bucket alone, as before.
    chunks.run("fit", "key", 3, budget, lambda: per_item, prepare, consume)
    assert padded_counts[-1] == 4 and compiled == ["fit"]
    chunks.run("fit", "key", 1, budget, lambda: per_item, prepare, consume)
    assert padded_counts[-1] == 1 and compiled == ["fit", "fit"]


def test_encodes_run_in_calls_their_items_fill() -> None:
    # Each encode item is a whole context, so nineteen items run as calls of
    # sixteen, two and one rather than one padded to thirty-two; other stages
    # still pad to their bucket. Each item's output is what it gets alone.
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.arange(40, dtype=np.float32).reshape(40, 1)
    chunks = Chunks(lambda *_: None)
    padded_counts: list[int] = []
    outputs: list[np.ndarray] = []

    def prepare(first: int, stop: int, padded: int) -> Call:
        padded_counts.append(padded)
        rows = items[first:stop]
        stacked = np.concatenate([rows, np.repeat(rows[-1:], padded - len(rows), 0)])
        return Call(doubled, (), (stacked,), padded)

    def consume(first: int, stop: int, found: Any) -> None:
        outputs.extend(np.asarray(found)[: stop - first])

    # An estimate past what XLA measures, so the chunk stays 32 items.
    per_item, budget = 1 << 20, 32 << 20

    def run(stage: str, count: int) -> list[int]:
        padded_counts.clear()
        outputs.clear()
        chunks.run(stage, stage, count, budget, lambda: per_item, prepare, consume)
        np.testing.assert_array_equal(np.stack(outputs), items[:count] * 2)
        return list(padded_counts)

    assert run("encode", 19) == [16, 2, 1]
    assert run("fit_encode", 19) == [16, 2, 1]
    assert run("decode", 19) == [32]
    # A run past the chunk fills its last calls too.
    assert run("encode", 40) == [32, 8]
    assert run("encode", 32) == [32]
    assert run("encode", 1) == [1]


def test_measuring_a_kind_counts_under_the_call_that_reuses_its_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measuring a kind compiles its program, which the call that follows
    # runs, so the time is the stage's, as a compile in a call is.
    def measured_slowly(call: Call, untuned: bool = False) -> int:
        time.sleep(0.05)
        return 0

    monkeypatch.setattr(budget_module, "measured_bytes", measured_slowly)
    counted: list[tuple[str, float]] = []
    chunks = Chunks(lambda stage, seconds, flops: counted.append((stage, seconds)))
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.ones((2, 4), np.float32)

    _run(chunks, doubled, items, 8 * items[0].nbytes)
    assert len(counted) == 1 and counted[0][0] == "fit" and counted[0][1] >= 0.05
    counted.clear()
    _run(chunks, doubled, items, 8 * items[0].nbytes)
    assert len(counted) == 1 and counted[0][1] < 0.05


def test_a_calls_handing_to_the_device_is_compile_for_a_new_shape_and_dispatch_after() -> (
    None
):
    timed: list[str] = []
    chunks = Chunks(lambda *_: None, timed=lambda part, _: timed.append(part))
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.ones((2, 4), np.float32)

    _run(chunks, doubled, items, 8 * items[0].nbytes)
    assert timed == ["compile", "wait"]
    _run(chunks, doubled, items, 8 * items[0].nbytes)
    assert timed == ["compile", "wait", "dispatch", "wait"]


def test_only_calls_that_compile_are_noted_with_their_stage() -> None:
    noted: list[str] = []
    chunks = Chunks(lambda *_: None, noted=lambda stage, *_: noted.append(stage))
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.ones((2, 4), np.float32)

    _run(chunks, doubled, items, 8 * items[0].nbytes)
    _run(chunks, doubled, items, 8 * items[0].nbytes)
    assert len(noted) == 1
    _run(chunks, doubled, np.ones((2, 8), np.float32), 8 * items[0].nbytes)
    assert len(noted) == 2


def test_a_count_not_yet_compiled_compiles_on_the_side_while_compiled_counts_run() -> (
    None
):
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.arange(7, dtype=np.float32).reshape(7, 1)
    chunks = Chunks(lambda *_: None)
    padded_counts: list[int] = []
    outputs: list[np.ndarray] = []
    aside: list[tuple[int, Any]] = []

    def prepare(first: int, stop: int, padded: int) -> Call:
        padded_counts.append(padded)
        rows = items[first:stop]
        stacked = np.concatenate([rows, np.repeat(rows[-1:], padded - len(rows), 0)])
        return Call(doubled, (), (stacked,), padded)

    def consume(first: int, stop: int, found: Any) -> None:
        outputs.extend(np.asarray(found)[: stop - first])

    def ahead(call: Call, done: Any) -> None:
        aside.append((call.padded, done))

    def run(count: int) -> list[int]:
        padded_counts.clear()
        outputs.clear()
        chunks.run(
            "fit", "key", count, 8 << 20, lambda: 1 << 20, prepare, consume, ahead
        )
        np.testing.assert_array_equal(np.stack(outputs), items[:count] * 2)
        return padded_counts

    # A kind's first call compiles for its own count, in the call.
    assert run(2) == [2] and aside == []
    # Five items would pad to eight: that compiles on the side, once, while
    # they run two at a time, the most a compiled count takes.
    assert run(5) == [8, 2, 2, 2]
    assert [padded for padded, _ in aside] == [8]
    assert run(5) == [2, 2, 2]
    assert len(aside) == 1
    # One item pads up to two rather than compile for one.
    assert run(1)[-1] == 2
    aside[0][1](True)
    # Once it compiled, eight takes them in one call, and fewer pad up to it.
    assert run(5) == [8]
    assert run(7) == [8]


def test_calls_with_buckets_pad_to_the_least_that_takes_them_or_split_across_the_most() -> (
    None
):
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.arange(80, dtype=np.float32).reshape(80, 1)
    padded_counts: list[int] = []
    outputs: list[np.ndarray] = []

    def prepare(first: int, stop: int, padded: int) -> Call:
        padded_counts.append(padded)
        rows = items[first:stop]
        stacked = np.concatenate([rows, np.repeat(rows[-1:], padded - len(rows), 0)])
        return Call(doubled, (), (stacked,), padded)

    def consume(first: int, stop: int, found: Any) -> None:
        outputs.extend(np.asarray(found)[: stop - first])

    def run(count: int, budget: int = 128 << 20) -> list[int]:
        padded_counts.clear()
        outputs.clear()
        chunks = Chunks(lambda *_: None)
        chunks.run(
            "fit",
            "key",
            count,
            budget,
            lambda: 1 << 20,
            prepare,
            consume,
            buckets=(1, 4, 16, 64),
        )
        np.testing.assert_array_equal(np.stack(outputs), items[:count] * 2)
        return padded_counts

    assert run(1) == [1]
    assert run(3) == [4]
    assert run(5) == [16]
    assert run(64) == [64]
    assert run(70) == [64, 16]
    # Within a chunk of eight items, the most bucket that fits is four.
    assert run(7, 8 << 20) == [4, 4]


def test_a_call_out_of_memory_at_a_larger_compiled_count_runs_at_its_own() -> None:
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.arange(64, dtype=np.float32).reshape(64, 1)
    chunks = Chunks(lambda *_: None)
    calls: list[int] = []
    aside: list[int] = []
    outputs: list[np.ndarray] = []
    # How many items a call may take before the device runs out of memory.
    room = [64]

    def crowded(values: Any) -> Any:
        calls.append(len(values))
        if len(values) > room[0]:
            raise jax.errors.JaxRuntimeError(
                "RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
                "1008.01MiB with allocator GPU_0_bfc on device 0."
            )
        return doubled(values)

    def prepare(first: int, stop: int, padded: int) -> Call:
        rows = items[first:stop]
        stacked = np.concatenate([rows, np.repeat(rows[-1:], padded - len(rows), 0)])
        return Call(crowded, (), (stacked,), padded)

    def consume(first: int, stop: int, found: Any) -> None:
        outputs.extend(np.asarray(found)[: stop - first])

    def run(count: int) -> None:
        calls.clear()
        outputs.clear()
        chunks.run(
            "decode",
            "key",
            count,
            64 << 20,
            lambda: 1 << 20,
            prepare,
            consume,
            lambda call, done: aside.append(call.padded),
        )
        np.testing.assert_array_equal(np.stack(outputs), items[:count] * 2)

    # A warm call compiled the kind for 64 items while the device had room.
    run(64)
    assert calls == [64]
    # Then the device fills up. One item would reuse the program for 64
    # while its own count compiles on the side; that runs out of memory, so
    # it runs at its own count, compiled in the call, rather than failing.
    room[0] = 1
    run(1)
    assert calls == [64, 1] and aside == [1]
    # From then on its own count is compiled, and it runs there.
    run(1)
    assert calls == [1]


def test_a_count_that_fails_to_compile_on_the_side_starts_again_later() -> None:
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.arange(5, dtype=np.float32).reshape(5, 1)
    chunks = Chunks(lambda *_: None)
    aside: list[tuple[int, Any]] = []

    def prepare(first: int, stop: int, padded: int) -> Call:
        rows = items[first:stop]
        stacked = np.concatenate([rows, np.repeat(rows[-1:], padded - len(rows), 0)])
        return Call(doubled, (), (stacked,), padded)

    def run(count: int) -> None:
        chunks.run(
            "fit",
            "key",
            count,
            8 << 20,
            lambda: 1 << 20,
            prepare,
            lambda *_: None,
            lambda call, done: aside.append((call.padded, done)),
        )

    run(2)
    run(5)
    assert [padded for padded, _ in aside] == [8]
    # The side holds the call's shapes, not its items.
    aside[0][1](False)
    run(5)
    assert [padded for padded, _ in aside] == [8, 8]


def test_a_kind_compiles_in_its_call_again_once_its_programs_are_dropped() -> None:
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.arange(5, dtype=np.float32).reshape(5, 1)
    chunks = Chunks(lambda *_: None)
    aside: list[int] = []
    padded_counts: list[int] = []

    def prepare(first: int, stop: int, padded: int) -> Call:
        padded_counts.append(padded)
        rows = items[first:stop]
        stacked = np.concatenate([rows, np.repeat(rows[-1:], padded - len(rows), 0)])
        return Call(doubled, (), (stacked,), padded)

    def run(count: int) -> list[int]:
        padded_counts.clear()
        chunks.run(
            "fit",
            "key",
            count,
            8 << 20,
            lambda: 1 << 20,
            prepare,
            lambda *_: None,
            lambda call, done: aside.append(call.padded),
        )
        return padded_counts

    run(2)
    assert chunks.ran("key")
    chunks.forget_counts()

    # Nothing compiled is left for the kind, so it neither waits on a count
    # it no longer has nor sends its own count to the side.
    assert not chunks.ran("key")
    assert run(5) == [8]
    assert aside == []


@pytest.mark.parametrize(
    ("analysis", "flops"),
    [
        ({"flops": 8.0, "bytes accessed": 64.0}, 8.0),
        ([{"flops": 8.0}], 8.0),
        ([{"flops": 8.0}, {"flops": 2.0}], 10.0),
        ([], 0.0),
        ({"bytes accessed": 64.0}, 0.0),
        ({"flops": -1.0}, 0.0),
        ({"flops": float("nan")}, 0.0),
        (None, 0.0),
    ],
)
def test_flops_come_from_xla_cost_analysis_in_either_of_its_shapes(
    analysis: Any, flops: float
) -> None:
    assert flops_of(analysis) == flops


def test_a_programs_flops_are_xlas_estimate_for_its_padded_shapes() -> None:
    multiplied = jax.jit(lambda left, right: left @ right)
    left, right = np.ones((4, 8, 16), np.float32), np.ones((4, 16, 2), np.float32)

    # Two flops, a multiply and an add, per product term.
    assert estimated_flops(Call(multiplied, (), (left, right), 4)) == pytest.approx(
        2 * 4 * 8 * 16 * 2, rel=0.1
    )
    assert estimated_flops(Call(lambda rows: rows, (), (left,), 4)) == 0.0


def test_each_call_counts_its_programs_flops_read_once_per_padded_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimated: list[int] = []

    def estimate(call: Call) -> float:
        estimated.append(call.padded)
        return 10.0 * call.padded

    monkeypatch.setattr(budget_module, "estimated_flops", estimate)
    counted: list[float] = []
    chunks = Chunks(lambda stage, seconds, flops: counted.append(flops))
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.ones((3, 4), np.float32)

    for _ in range(3):
        _run(chunks, doubled, items, 8 * items[0].nbytes)
    # Three items pad to a call of four; one item runs at one.
    _run(chunks, doubled, items[:1], 8 * items[0].nbytes)
    assert counted == [40.0, 40.0, 40.0, 10.0]
    assert estimated == [4, 1]


def test_a_program_compiled_before_its_first_call_counts_the_flops_its_compile_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def estimate(call: Call) -> float:
        raise AssertionError("a call read XLA's cost analysis")

    counted: list[float] = []
    chunks = Chunks(lambda stage, seconds, flops: counted.append(flops))
    summed = jax.jit(lambda rows: (rows @ rows.T).sum(-1))
    items = np.ones((2, 8), np.float32)
    # As a worker's start compiles what earlier processes recorded, and runs it.
    note_compiled(summed, (items,), summed.lower(items).compile())
    summed(items)
    monkeypatch.setattr(budget_module, "estimated_flops", estimate)

    _run(chunks, summed, items, 8 * items[0].nbytes)
    _run(chunks, summed, items, 8 * items[0].nbytes)
    assert counted[0] > 0 and counted == [counted[0]] * 2


def test_a_program_compiled_unnoted_before_its_first_call_still_counts_its_flops() -> (
    None
):
    counted: list[float] = []
    chunks = Chunks(lambda stage, seconds, flops: counted.append(flops))
    summed = jax.jit(lambda rows: (rows @ rows.T).sum(-1) + 1)
    items = np.ones((2, 8), np.float32)
    summed(items)

    _run(chunks, summed, items, 8 * items[0].nbytes)
    assert counted[0] > 0


def test_a_kinds_first_call_at_a_count_is_seen_though_compiled_before() -> None:
    doubled = jax.jit(lambda rows: rows * 2)
    items = np.ones((2, 4), np.float32)
    _run(Chunks(lambda *_: None), doubled, items, 8 * items[0].nbytes)
    seen: list[int] = []

    # A later process's worker, its program compiled at start: its first
    # call is seen once, so the manifest keeps the kind by its last use.
    chunks = Chunks(lambda *_: None, seen=lambda call: seen.append(call.padded))
    _run(chunks, doubled, items, 8 * items[0].nbytes)
    _run(chunks, doubled, items, 8 * items[0].nbytes)

    assert seen == [2]


def test_a_call_that_runs_out_of_memory_beside_a_compile_runs_again_once_it_is_done() -> (
    None
):
    compiling = [True]
    sizes: list[int] = []

    def crowded(values: np.ndarray) -> np.ndarray:
        sizes.append(len(values))
        # A compile beside the call holds the device's memory as it autotunes.
        if compiling[0]:
            raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory")
        return values

    def settle() -> bool:
        was = compiling[0]
        compiling[0] = False
        return was

    chunks = Chunks(lambda *_: None)
    chunks.settle = settle
    items = np.ones((3, 4), np.float32)

    # Even a lone item runs again rather than raising, at its size.
    assert _run(chunks, crowded, items[:1], 8 * items[0].nbytes) == [1]
    assert sizes == [1, 1]

    # With nothing beside it, a call that runs out halves as before.
    compiling[0] = True
    sizes.clear()
    chunks.settle = lambda: False
    with pytest.raises(jax.errors.JaxRuntimeError, match="RESOURCE_EXHAUSTED"):
        _run(chunks, crowded, items[:1], 8 * items[0].nbytes)
    assert sizes == [1]


def test_a_call_too_large_to_autotune_beside_the_device_s_holdings_compiles_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generalforecast.runtime import budget

    jitted = jax.jit(lambda values: values * 2)
    tuned: list[int] = []

    class Autotuned:
        """A program that would autotune if called as it is."""

        def __call__(self, values: Any) -> Any:
            tuned.append(len(values))
            return jitted(values)

        def lower(self, *arguments: Any) -> Any:
            return jitted.lower(*arguments)

    compiled: list[dict] = []
    real = jax.stages.Lowered.compile

    def compile(lowered: Any, compiler_options: Any = None) -> Any:
        compiled.append(compiler_options or {})
        return real(lowered, compiler_options=compiler_options)

    monkeypatch.setattr(jax.stages.Lowered, "compile", compile)
    items = np.arange(8, dtype=np.float32).reshape(2, 4)
    program = Autotuned()

    def prepare(first: int, stop: int, padded: int) -> Call:
        return Call(program, (), (items[first:stop],), padded)

    outputs: list[np.ndarray] = []

    def consume(first: int, stop: int, values: Any) -> None:
        outputs.append(np.asarray(values))

    # Room for the call's items twice over: it runs as compiled, autotuned.
    monkeypatch.setattr(budget, "free_device_bytes", lambda: 1 << 20)
    chunks = Chunks(lambda *_: None)
    chunks.run("decode", "roomy", 2, 1 << 20, lambda: 32, prepare, consume)
    assert tuned == [2]

    # Less than twice its items free: its program compiles without autotuning.
    monkeypatch.setattr(budget, "free_device_bytes", lambda: 100)
    tuned.clear()
    compiled.clear()
    chunks.run("decode", "crowded", 2, 1 << 20, lambda: 32, prepare, consume)
    assert tuned == []
    assert UNTUNED_OPTIONS in compiled
    np.testing.assert_array_equal(outputs[-1], items * 2)

    # Where the device does not say, as on a CPU, nothing changes: a program
    # of its own, as the untuned one serves the last one's shapes.
    monkeypatch.setattr(budget, "free_device_bytes", lambda: None)
    program = Autotuned()
    chunks.run("decode", "cpu", 2, 1 << 20, lambda: 32, prepare, consume)
    assert tuned == [2]


def test_a_program_compiled_without_autotuning_compiles_with_it_once_there_is_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generalforecast.runtime import budget

    jitted = jax.jit(lambda values: values * 2)

    class Program:
        def __call__(self, values: Any) -> Any:
            return jitted(values)

        def lower(self, *arguments: Any) -> Any:
            return jitted.lower(*arguments)

    compiled: list[dict] = []
    real = jax.stages.Lowered.compile

    def compile(lowered: Any, compiler_options: Any = None) -> Any:
        compiled.append(compiler_options or {})
        return real(lowered, compiler_options=compiler_options)

    monkeypatch.setattr(jax.stages.Lowered, "compile", compile)
    items = np.arange(8, dtype=np.float32).reshape(2, 4)
    program = Program()

    def prepare(first: int, stop: int, padded: int) -> Call:
        return Call(program, (), (items[first:stop],), padded)

    outputs: list[np.ndarray] = []
    chunks = Chunks(lambda *_: None)
    gaps: list[Any] = []
    chunks.later = gaps.append
    free = [100]
    monkeypatch.setattr(budget, "free_device_bytes", lambda: free[0])

    def run() -> None:
        chunks.run(
            "decode",
            "key",
            2,
            1 << 20,
            lambda: 32,
            prepare,
            lambda first, stop, values: outputs.append(np.asarray(values)),
        )

    run()
    # Untuned for want of room, with a tuned compile waiting for a quiet gap.
    assert compiled[-1] == UNTUNED_OPTIONS and len(gaps) == 1
    # No room in that gap either: it stays untuned, and the next call asks again.
    gaps.pop()()
    assert compiled[-1] == UNTUNED_OPTIONS
    run()
    assert len(gaps) == 1
    free[0] = 1 << 20
    gaps.pop()()
    assert compiled[-1] == {}
    run()
    # Tuned now, it asks for nothing more, and answers as before.
    assert not gaps
    np.testing.assert_array_equal(outputs[-1], items * 2)


def test_a_kinds_first_measure_compiles_untuned_past_a_quarter_of_free_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generalforecast.runtime import budget

    measures: list[bool] = []
    real = budget.measured_bytes

    def measured(call: Call, untuned: bool = False) -> int:
        measures.append(untuned)
        return real(call, untuned)

    monkeypatch.setattr(budget, "measured_bytes", measured)
    doubled = jax.jit(lambda rows: rows * 2)
    # Two items of 16 bytes by the estimate: a call of 32 bytes.
    items = np.ones((2, 4), np.float32)

    # Room for its call four times over: it measures as it compiles, tuned.
    monkeypatch.setattr(budget, "free_device_bytes", lambda: 128)
    assert _run(Chunks(lambda *_: None), doubled, items, 1 << 20) == [2]
    # Less than that, though twice its call is free: its autotuner would
    # time kernels the measure cannot yet size beside what the device holds,
    # so the measure compiles without autotuning.
    monkeypatch.setattr(budget, "free_device_bytes", lambda: 127)
    assert _run(Chunks(lambda *_: None), doubled, items, 1 << 20) == [2]
    # Where the device does not say, as on a CPU, it measures tuned.
    monkeypatch.setattr(budget, "free_device_bytes", lambda: None)
    assert _run(Chunks(lambda *_: None), doubled, items, 1 << 20) == [2]
    assert measures == [False, True, False]
