# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp

from generalforecast.constants import RECORDED_KINDS
from generalforecast.runtime.programs import (
    Kind,
    Manifest,
    Programs,
    abstract,
    counts_up_to,
    ladder,
    most_rows,
    signature,
    zeros,
)


class _Program:
    """Stands in for a compiled function, noting when it compiles."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, *arguments: Any) -> None:
        pass

    def lower(self, *arguments: Any) -> _Program:
        self.events.append("warm")
        return self

    def compile(self) -> None:
        pass


def test_compiles_ahead_wait_for_a_quiet_gap_after_the_last_step() -> None:
    now = [0.0]
    programs = Programs(None, lambda: False, clock=lambda: now[0])
    events: list[str] = []

    for _ in range(3):
        with programs.step():
            if not events:
                programs.compile_later(_Program(events))
            events.append("step")
    # No time passes between the steps or after them, so the compile waits:
    # one started mid-step would compete with the step's own compiles, and
    # one between steps sent back to back would slow the next.
    assert events == ["step"] * 3
    now[0] = programs.quiet
    programs.wait()

    assert events == ["step", "step", "step", "warm"]


def test_programs_without_compiles_ahead_compile_only_when_called() -> None:
    programs = Programs(None, lambda: False, ahead=False)
    events: list[str] = []
    programs.compile_later(_Program(events))
    programs.wait()

    assert events == []


def test_concurrent_compiled_calls_store_one_winner() -> None:
    programs = Programs(None, lambda: False, ahead=False)
    started = threading.Barrier(2)
    calls = []

    def build():
        calls.append(object())
        started.wait()
        return lambda value: value

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(programs.compiled, "same", build)
        second = pool.submit(programs.compiled, "same", build)
        assert first.result() is second.result()

    assert len(calls) == 2
    assert len(programs._compiled) == 1


def test_programs_drop_past_their_shapes_only_when_memory_is_short() -> None:
    short = [False]
    mine = jax.jit(lambda values: values + 1)
    theirs = jax.jit(lambda values: values * 2)
    programs = Programs(None, lambda: short[0], helpers=[mine])
    programs.limit = 2
    for shape in range(3):
        programs.add_shape(shape)
    assert programs.drops == 0
    mine(jnp.ones(3))
    theirs(jnp.ones(3))
    assert programs.first("warmed") and not programs.first("warmed")

    short[0] = True
    programs.add_shape(3)

    # Only the worker's own programs go, and what was compiled ahead of them
    # is compiled ahead again.
    assert programs.drops == 1
    assert (mine._cache_size(), theirs._cache_size()) == (0, 1)
    assert programs.first("warmed")


def test_ladders_double_from_a_padded_size() -> None:
    assert ladder(1, 4) == [1, 2, 4, 8]
    assert ladder(16, 5) == [16, 32, 64, 128, 256]


def test_start_compiles_each_power_of_two_up_to_the_most_items_taken() -> None:
    assert counts_up_to([]) == []
    assert counts_up_to([1]) == [1]
    assert counts_up_to([8, 2]) == [1, 2, 4, 8]


def test_signatures_drop_the_stacked_axis_and_stack_again() -> None:
    tree = {"rows": jnp.zeros((4, 3, 2)), "count": jnp.zeros((4,), jnp.int32)}
    recorded = signature(tree, stacked=True)
    shapes = abstract(recorded, 16)

    assert shapes["rows"].shape == (16, 3, 2)
    assert shapes["count"].shape == (16,)
    assert shapes["count"].dtype == jnp.int32


def test_manifests_keep_kinds_and_widths_for_later_processes(tmp_path) -> None:
    first = Programs(None, lambda: False, ahead=False, manifests=tmp_path)
    kind = Kind(("embed", None, 2, 2), signature(()), signature((jnp.zeros((1, 8)),)))
    first.remember("model", kind, 4)
    first.remember("model", kind, 1)
    first.remember_measure("model", ("width", 8))
    # Another worker on the machine adds to the same file.
    other = Kind(("fit", None, 2, 4), signature(()), signature((jnp.zeros((1, 3)),)))
    Programs(None, lambda: False, ahead=False, manifests=tmp_path).remember(
        "model", other, 2
    )

    recorded = Programs(None, lambda: False, manifests=tmp_path).recorded("model")

    assert recorded.kinds == {kind: {1, 4}, other: {2}}
    assert list(recorded.measures) == [("width", 8)]
    assert (
        Programs(None, lambda: False, manifests=tmp_path).recorded("other").kinds == {}
    )


def test_manifests_keep_only_the_latest_kinds(tmp_path) -> None:
    manifest = Manifest(tmp_path / "model.pickle")
    kinds = [
        Kind(("embed", None, 2, index), signature(()), signature(()))
        for index in range(RECORDED_KINDS + 1)
    ]
    for kind in kinds:
        manifest.add(kind, 1)
    manifest.add(kinds[1], 2)

    recorded = manifest.read().kinds
    assert len(recorded) == RECORDED_KINDS
    assert kinds[0] not in recorded
    assert list(recorded)[-1] == kinds[1]


def test_an_unreadable_manifest_reads_as_empty(tmp_path) -> None:
    (tmp_path / "model.pickle").write_bytes(b"not a manifest")

    assert Manifest(tmp_path / "model.pickle").read().kinds == {}


def test_measures_are_kept_for_later_processes(tmp_path) -> None:
    taken: list[str] = []

    def measure(value: Any) -> Any:
        taken.append(value)
        return value

    first = Programs(None, lambda: False, ahead=False, manifests=tmp_path)
    assert first.measured("model", ("width", 8), lambda: measure(1)) == 1
    assert first.measured("model", ("width", 8), lambda: measure(2)) == 1
    assert first.measured("model", ("encode", 8), lambda: measure(None)) is None
    later = Programs(None, lambda: False, ahead=False, manifests=tmp_path)

    assert later.measured("model", ("width", 8), lambda: measure(3)) == 1
    assert later.measured("model", ("encode", 8), lambda: measure(4)) is None
    assert later.measured("other", ("width", 8), lambda: measure(5)) == 5
    assert taken == [1, None, 5]
    assert (
        Programs(None, lambda: False, ahead=False).measured(
            "model", ("width", 8), lambda: measure(6)
        )
        == 6
    )


def test_zeros_keep_each_leaf_s_shape_dtype_and_weak_type() -> None:
    tree = (
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.int32, weak_type=True),
    )

    made = zeros(tree)

    assert made is not None
    assert [(leaf.shape, leaf.dtype) for leaf in made] == [
        ((2, 3), jnp.float32),
        ((1,), jnp.int32),
    ]
    assert signature(made) == signature(tree)
    assert zeros((jnp.zeros(2),)) is None


def test_recorded_values_round_trip_through_json_naming_only_this_package_s_types() -> (
    None
):
    import json

    import numpy as np
    import pytest

    from generalforecast.models.prepare import Columns
    from generalforecast.runtime.programs import Kind, Recorded, decoded, encoded

    tree = (jax.ShapeDtypeStruct((2, 3), jnp.float32), None, {"a": np.int64(3)})
    kind = Kind(("fit", None, 2), signature(tree), signature([Columns]))
    recorded = Recorded({kind: {1, 4}}, {("width", 10): None})

    back = decoded(json.loads(json.dumps(encoded(recorded))))

    assert back == recorded
    assert hash(next(iter(back.kinds))) == hash(kind)
    assert decoded(encoded(np.dtype("float32"))) == np.dtype("float32")
    with pytest.raises(ValueError):
        decoded({"class": "os:system", "items": []})
    with pytest.raises(TypeError):
        encoded(object())


def test_programs_drop_at_a_new_shape_when_mappings_are_short() -> None:
    mappings = [False]
    mine = jax.jit(lambda values: values + 1)
    programs = Programs(
        None, lambda: False, [mine], short_of_mappings=lambda: mappings[0]
    )
    programs.add_shape(0)
    mine(jnp.ones(3))
    mappings[0] = True
    programs.add_shape(0)
    assert programs.drops == 0

    # Short of mappings, even one more shape drops them, whatever memory is free.
    programs.add_shape(1)
    assert programs.drops == 1
    assert mine._cache_size() == 0


def test_a_drop_leaves_mappings_alone_once_it_shows_programs_do_not_hold_them() -> None:
    mappings = [True]
    programs = Programs(None, lambda: False, short_of_mappings=lambda: mappings[0])
    programs.limit = 3

    programs.add_shape(0)
    for shape in (1, 2):
        programs.add_shape(shape)
    # The first drop left mappings short, so the next waits for `limit` shapes.
    assert programs.drops == 1
    programs.add_shape(3)
    assert programs.drops == 2
    programs.add_shape(4)
    assert programs.drops == 2

    # Once mappings free up, running short again drops at the next shape.
    mappings[0] = False
    programs.add_shape(5)
    mappings[0] = True
    programs.add_shape(6)
    assert programs.drops == 3


def test_a_drop_cancels_the_compiles_ahead_still_waiting() -> None:
    events: list[str] = []
    dropped: list[bool] = []
    # The clock never moves, so no quiet gap comes and the compile waits.
    programs = Programs(
        None, lambda: False, clock=lambda: 0.0, dropped=lambda: dropped.append(True)
    )
    with programs.step():
        programs.compile_later(_Program(events))
    done = threading.Event()

    def drop() -> None:
        programs.drop()
        done.set()

    threading.Thread(target=drop, daemon=True).start()

    assert done.wait(5), "the drop waited for a compile ahead it drops"
    assert events == []
    assert dropped == [True]
    programs.wait()
    assert events == []


def test_a_drop_waits_for_what_steps_wait_on() -> None:
    programs = Programs(None, lambda: False)
    release = threading.Event()
    finished: list[bool] = []

    def task() -> None:
        assert release.wait(5)
        finished.append(True)

    programs.soon(task)
    threading.Timer(0.1, release.set).start()
    programs.drop()

    assert finished == [True]


def test_most_rows_is_a_signatures_longest_first_axis() -> None:
    table = (jnp.zeros((16384, 31)), jnp.zeros(16384, jnp.int32), 3)
    assert most_rows(signature(table)) == 16384
    assert most_rows(signature((jnp.float32(1),))) == 0
