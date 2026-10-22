# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.runtime import budget as budget_module
from generalforecast.runtime.budget import Call, Chunks, MemoryBudget


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
    chunks = Chunks(lambda stage, seconds: counted.append(stage))
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


def test_measuring_a_kind_counts_under_the_call_that_reuses_its_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measuring a kind compiles its program, which the call that follows
    # runs, so the time is the stage's, as a compile in a call is.
    def measured_slowly(call: Call) -> int:
        time.sleep(0.05)
        return 0

    monkeypatch.setattr(budget_module, "measured_bytes", measured_slowly)
    counted: list[tuple[str, float]] = []
    chunks = Chunks(lambda stage, seconds: counted.append((stage, seconds)))
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
