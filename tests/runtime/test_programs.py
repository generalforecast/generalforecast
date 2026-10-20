from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from generalforecast.runtime.programs import Programs, ladder


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
