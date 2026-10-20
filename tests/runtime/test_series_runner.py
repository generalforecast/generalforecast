"""Forecasts through a node and a worker: groups bucketed, cached and decoded."""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast import _core
from generalforecast.models.series import (
    KNOWN,
    PAST,
    TARGET,
    SeriesFunctions,
    SeriesGroup,
)
from generalforecast.models.timestral.model import timestral_functions
from generalforecast.worker import Worker
from tests.models.chronos2.tiny import tiny_chronos2
from tests.models.timesfm3.tiny import tiny_timesfm
from tests.models.timestral.tiny import tiny_timestral

STEP = 10
A, B = 1, 2


def _key(series: int, time: int) -> list[int]:
    packed = (series << 64) | (time & 0xFFFFFFFFFFFFFFFF)
    return [packed & 0xFFFFFFFFFFFFFFFF, packed >> 64]


def _rows(count: int) -> list[tuple[int, int]]:
    """A at `count` steps from 100, B at `count` steps from 120."""
    return [(A, 100 + STEP * step) for step in range(count)] + [
        (B, 120 + STEP * step) for step in range(count)
    ]


def _values(series: int, time: int) -> tuple[float, float]:
    """A series' value and its one covariate at a time.

    The covariate is not a line: an exactly linear row detrends to rounding
    noise, which TimesFM 3 and its upstream read as noise.
    """
    return float(np.sin(time / 7.0) * series + series), float(
        np.cos(time / 5.0) + time / 100.0
    )


def _setup(
    functions: SeriesFunctions,
    parameters: Any,
    known: list[int],
    steps: int = 10,
    group: _core.GroupStrategy | None = None,
    placed: list[tuple[int, int]] | None = None,
) -> tuple[_core.Node, Worker, int]:
    """A node and worker serving one series table of A and B with one covariate."""
    properties = functions.properties(parameters)
    worker = Worker(bucket_minimum=1, slot_bytes=4096, compile_cache=False)
    worker.add_model(0, functions, parameters)
    sizes = worker.series_profile(0)
    node = _core.Node(device_bytes=1 << 22, host_bytes=1 << 20, slot_bytes=4096)
    table = node.create_series_table(
        model=0,
        features=1,
        step=STEP,
        patch=properties.patch,
        context=properties.context_limit,
        group=group or _core.GroupStrategy.random(2, 5),
        bytes_per_row=sizes.per_row,
        patch_bytes=sizes.per_patch,
        known=known,
        reusable=properties.reusable,
        extendable=functions.extend is not None,
    )
    worker.add_table(table, 0, 0, 1, series=True)
    worker.allocate(1 << 10, 1 << 8, 0)
    placed = _rows(steps) if placed is None else placed
    node.insert(
        table,
        np.array([_key(s, t) for s, t in placed], dtype=np.uint64),
        np.array([[_values(s, t)[1]] for s, t in placed], dtype=np.float32),
        np.array([_values(s, t)[0] for s, t in placed], dtype=np.float32),
        np.array([t for _, t in placed], dtype=np.int64),
    )
    return node, worker, table


def _run(node: _core.Node, worker: Worker) -> tuple[dict[int, np.ndarray], _core.Step]:
    step = node.schedule()
    assert step is not None
    predictions = worker.run(step)
    node.complete(step)
    return predictions, step


def _fresh(
    functions: SeriesFunctions,
    parameters: Any,
    step: _core.Step,
    predictions: dict[int, np.ndarray],
) -> None:
    """Checks each forecast against the model run on its unit's own grid, unpadded."""
    for forecast in step.forecasts:
        unit = step.series_units[forecast.unit]
        grid, future = unit.grid, forecast.future
        known = np.asarray(future.observed, bool)
        group = SeriesGroup(
            jnp.asarray(np.asarray(grid.values, np.float32)),
            jnp.asarray(np.asarray(grid.observed, bool)),
            jnp.asarray(np.asarray(grid.roles, np.int32)),
            jnp.asarray(np.asarray(future.values, np.float32)) if known.any() else None,
            jnp.asarray(known) if known.any() else None,
        )
        outputs = np.asarray(functions.forecast(parameters, group, forecast.horizon))
        at = [f for f in step.forecasts if f.request == forecast.request].index(
            forecast
        )
        np.testing.assert_allclose(
            predictions[forecast.request][at],
            outputs[forecast.target * 2],
            rtol=1e-4,
            atol=1e-4,
        )


def _expected(
    functions: SeriesFunctions,
    parameters: Any,
    cutoff: int,
    horizon: int,
    known: list[int],
    steps: int = 10,
) -> np.ndarray:
    """The forecasts of A and B as the model gives them on the group laid by hand."""
    patch = functions.properties(parameters).patch
    grid = {time: [] for time in range(100, cutoff, STEP)}
    rows = []
    for series in (A, B):
        for column in (0, 1):
            values = np.zeros(len(grid), np.float32)
            observed = np.zeros(len(grid), bool)
            for at, time in enumerate(grid):
                if (series, time) in _rows(steps):
                    values[at] = _values(series, time)[column]
                    observed[at] = True
            rows.append(
                (
                    values,
                    observed,
                    [TARGET, KNOWN if column - 1 in known else PAST][column],
                )
            )
    values = np.stack([row[0] for row in rows])
    observed = np.stack([row[1] for row in rows])
    roles = np.array([row[2] for row in rows], np.int32)
    pad = -values.shape[1] % patch
    values = np.pad(values, ((0, 0), (pad, 0)))
    observed = np.pad(observed, ((0, 0), (pad, 0)))
    future = future_observed = None
    if known:
        future = np.zeros((4, horizon), np.float32)
        future_observed = np.zeros((4, horizon), bool)
        for at, series in enumerate((A, B)):
            for step in range(horizon):
                time = cutoff + STEP * step
                if (series, time) in _rows(steps):
                    future[2 * at + 1, step] = _values(series, time)[1]
                    future_observed[2 * at + 1, step] = True
    group = SeriesGroup(
        jnp.asarray(values),
        jnp.asarray(observed),
        jnp.asarray(roles),
        None if future is None else jnp.asarray(future),
        None if future_observed is None else jnp.asarray(future_observed),
    )
    outputs = np.asarray(functions.forecast(parameters, group, horizon))
    return outputs[[0, 2]]


CASES = [
    (
        "timestral",
        lambda: (
            __import__(
                "generalforecast.models.timestral.model", fromlist=["x"]
            ).timestral_functions
        ),
        tiny_timestral,
    ),
    (
        "chronos2",
        lambda: (
            __import__(
                "generalforecast.models.chronos2.model", fromlist=["x"]
            ).chronos2_functions
        ),
        tiny_chronos2,
    ),
    (
        "timesfm3",
        lambda: (
            __import__(
                "generalforecast.models.timesfm3.model", fromlist=["x"]
            ).timesfm_functions
        ),
        tiny_timesfm,
    ),
]


@pytest.mark.parametrize(
    ("name", "functions", "build"), CASES, ids=[case[0] for case in CASES]
)
def test_forecasts_match_the_model_on_the_group_and_reuse_the_cache(
    name: str, functions: Any, build: Any
) -> None:
    functions, parameters = functions(), build()
    node, worker, table = _setup(functions, parameters, known=[])
    cutoff, horizon = 180, 6

    request = node.submit_forecast(
        table, np.array([A, B], dtype=np.uint64), horizon, np.array([cutoff, cutoff])
    )
    predictions, step = _run(node, worker)
    expected = _expected(functions, parameters, cutoff, horizon, known=[])

    assert predictions[request].shape == expected.shape
    np.testing.assert_allclose(predictions[request], expected, rtol=1e-4, atol=1e-4)
    reusable = functions.properties(parameters).reusable
    assert step.series_units[0].action == ("encode" if reusable else "transient")

    longer = node.submit_forecast(
        table, np.array([B], dtype=np.uint64), 9, np.array([cutoff])
    )
    again, step = _run(node, worker)
    assert step.series_units[0].action == ("reuse" if reusable else "transient")
    np.testing.assert_allclose(
        again[longer][0],
        _expected(functions, parameters, cutoff, 9, known=[])[1],
        rtol=1e-4,
        atol=1e-4,
    )
    assert worker.stats.reused == (1 if reusable else 0)
    assert set(worker.stats.seconds) & {"forecast", "series_encode", "series_decode"}


def test_known_columns_are_forecast_whole_with_their_future() -> None:
    parameters = tiny_timestral()
    node, worker, table = _setup(timestral_functions, parameters, known=[0])
    cutoff, horizon = 160, 5

    request = node.submit_forecast(
        table, np.array([A, B], dtype=np.uint64), horizon, np.array([cutoff, cutoff])
    )
    predictions, step = _run(node, worker)

    assert step.series_units[0].action == "transient"
    expected = _expected(timestral_functions, parameters, cutoff, horizon, known=[0])
    np.testing.assert_allclose(predictions[request], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("name", "functions", "build"), CASES, ids=[case[0] for case in CASES]
)
def test_groups_pad_their_rows_and_patches_as_the_node_sizes_them(
    name: str, functions: Any, build: Any
) -> None:
    """Three series with a covariate, 6 rows padding to 8, over 3 patches padding to 4."""
    functions, parameters = functions(), build()
    placed = [
        (s, start + STEP * k)
        for s, start in ((1, 100), (2, 150), (3, 120))
        for k in range(20)
    ]
    node, worker, table = _setup(
        functions,
        parameters,
        known=[],
        group=_core.GroupStrategy.random(3, 5),
        placed=placed,
    )

    request = node.submit_forecast(
        table, np.array([1, 2, 3], dtype=np.uint64), 7, np.array([220, 220, 220])
    )
    predictions, step = _run(node, worker)

    assert np.asarray(step.series_units[0].grid.values).shape == (6, 12)
    assert predictions[request].shape[:2] == (3, 7)
    _fresh(functions, parameters, step, predictions)
    reusable = functions.properties(parameters).reusable
    # A model that keeps no cache forecasts the unit whole, once for the
    # three forecasts of one group, cutoff and horizon.
    assert worker.stats.encoded == (1 if reusable else 0)
    assert worker.stats.transient == (0 if reusable else 1)


def test_patch_buckets_stay_within_the_context_limit() -> None:
    import dataclasses

    from generalforecast.runtime.series import buckets

    parameters = dataclasses.replace(tiny_timestral(), context_limit=12)
    properties = timestral_functions.properties(parameters)
    # Three patches would bucket to four, past the three the model reads.
    assert buckets(2, 12, properties, 1) == (2, 12)
    assert buckets(3, 8, properties, 4) == (4, 12)
    node, worker, table = _setup(timestral_functions, parameters, known=[], steps=30)

    request = node.submit_forecast(
        table, np.array([A], dtype=np.uint64), 5, np.array([400])
    )
    predictions, step = _run(node, worker)

    assert np.asarray(step.series_units[0].grid.values).shape[1] == 12
    assert predictions[request].shape == (1, 5, 9)
    _fresh(timestral_functions, parameters, step, predictions)


@pytest.mark.parametrize(
    ("name", "functions", "build"), CASES, ids=[case[0] for case in CASES]
)
def test_members_without_observations_before_the_cutoff_leave_positions_alone(
    name: str, functions: Any, build: Any
) -> None:
    """Series starting after the cutoff join a group as rows observed nowhere."""
    functions, parameters = functions(), build()
    placed = [(s, 100 * s + STEP * k) for s in range(1, 7) for k in range(20)]
    node, worker, table = _setup(
        functions,
        parameters,
        known=[],
        group=_core.GroupStrategy.random(5, 3),
        placed=placed,
    )

    request = node.submit_forecast(
        table, np.array([1], dtype=np.uint64), 6, np.array([360])
    )
    predictions, step = _run(node, worker)

    grid = step.series_units[0].grid
    observed = np.asarray(grid.observed, bool)
    assert not observed[::2].all(axis=1).all(), "a member starts after the cutoff"
    assert predictions[request].shape[:2] == (1, 6)
    _fresh(functions, parameters, step, predictions)


@pytest.mark.parametrize(
    ("name", "functions", "build"), CASES, ids=[case[0] for case in CASES]
)
def test_series_without_history_are_forecast_cold_through_their_group(
    name: str, functions: Any, build: Any
) -> None:
    """A series the table never saw is forecast from its observed siblings, as the model would."""
    functions, parameters = functions(), build()
    node, worker, table = _setup(
        functions, parameters, known=[], group=_core.GroupStrategy.random(2, 5)
    )

    request = node.submit_forecast(
        table, np.array([3], dtype=np.uint64), 4, np.array([200])
    )
    predictions, step = _run(node, worker)

    unit = step.series_units[0]
    assert len(unit.members) == 2 and 3 in unit.members
    assert not np.asarray(unit.grid.observed, bool)[unit.members.index(3) * 2].any()
    assert np.isfinite(predictions[request]).all()
    _fresh(functions, parameters, step, predictions)

    # Alone, it has nothing to forecast from.
    node, worker, table = _setup(
        functions, parameters, known=[], group=_core.GroupStrategy.alone()
    )
    request = node.submit_forecast(
        table, np.array([3], dtype=np.uint64), 4, np.array([200])
    )
    assert node.schedule() is None
    [(refused, reason, missing)] = node.take_refused()
    assert refused == request and "no series 3" in reason and not missing


@pytest.mark.parametrize(
    ("name", "functions", "build"), CASES, ids=[case[0] for case in CASES]
)
def test_a_cutoff_advanced_by_a_patch_extends_the_kept_encoding(
    name: str, functions: Any, build: Any
) -> None:
    """A model with `extend` grows the kept encoding by the new patch; the forecast equals a fresh encode."""
    functions, parameters = functions(), build()
    node, worker, table = _setup(
        functions, parameters, known=[], group=_core.GroupStrategy.random(2, 5)
    )
    node.submit_forecast(
        table, np.array([A, B], dtype=np.uint64), 4, np.array([220, 220])
    )
    _run(node, worker)

    patch = functions.properties(parameters).patch
    advance = STEP * patch
    placed = [(s, t) for s in (A, B) for t in range(220, 220 + advance, STEP)]
    node.insert(
        table,
        np.array([_key(s, t) for s, t in placed], dtype=np.uint64),
        np.array([[_values(s, t)[1]] for s, t in placed], dtype=np.float32),
        np.array([_values(s, t)[0] for s, t in placed], dtype=np.float32),
        np.array([t for _, t in placed], dtype=np.int64),
    )
    before = worker.stats.extended
    request = node.submit_forecast(
        table, np.array([A, B], dtype=np.uint64), 4, np.array([220 + advance] * 2)
    )
    predictions, step = _run(node, worker)

    unit = step.series_units[0]
    properties = functions.properties(parameters)
    extendable = functions.extend is not None and properties.reusable
    expected = (
        "extend" if extendable else "encode" if properties.reusable else "transient"
    )
    assert unit.action == expected
    assert unit.grown == (patch if extendable else 0)
    assert worker.stats.extended - before == (1 if extendable else 0)
    assert predictions[request].shape[:2] == (2, 4)
    # The grown encoding forecasts as a fresh one of the whole grid does.
    _fresh(functions, parameters, step, predictions)


def _advance(node: _core.Node, table: int, start: int, steps: int) -> None:
    """Inserts A's and B's next `steps` observations from `start`."""
    placed = [(s, t) for s in (A, B) for t in range(start, start + STEP * steps, STEP)]
    node.insert(
        table,
        np.array([_key(s, t) for s, t in placed], dtype=np.uint64),
        np.array([[_values(s, t)[1]] for s, t in placed], dtype=np.float32),
        np.array([_values(s, t)[0] for s, t in placed], dtype=np.float32),
        np.array([t for _, t in placed], dtype=np.int64),
    )


def test_a_cutoff_advancing_patch_by_patch_extends_the_last_extension() -> None:
    """Streaming: each advance grows the entry the one before wrote, in slots sized for the patches it holds."""
    functions, parameters = timestral_functions, tiny_timestral()
    node, worker, table = _setup(functions, parameters, known=[])
    patch = functions.properties(parameters).patch
    cutoff = 220
    node.submit_forecast(
        table, np.array([A, B], dtype=np.uint64), 4, np.array([cutoff] * 2)
    )
    _run(node, worker)
    slots = 0
    for _ in range(6):
        _advance(node, table, cutoff, patch)
        cutoff += STEP * patch
        node.submit_forecast(
            table, np.array([A, B], dtype=np.uint64), 4, np.array([cutoff] * 2)
        )
        predictions, step = _run(node, worker)
        unit = step.series_units[0]
        assert (unit.action, unit.grown) == ("extend", patch)
        assert len(unit.slots) >= slots
        slots = len(unit.slots)
        _fresh(functions, parameters, step, predictions)
    assert worker.stats.extended == 6


def test_two_cutoffs_of_one_group_in_one_step_both_encode() -> None:
    """The later cutoff cannot grow an entry the same step is still encoding."""
    functions, parameters = timestral_functions, tiny_timestral()
    node, worker, table = _setup(functions, parameters, known=[])
    patch = functions.properties(parameters).patch
    _advance(node, table, 220, patch)
    for cutoff in (220, 220 + STEP * patch):
        node.submit_forecast(
            table, np.array([A, B], dtype=np.uint64), 4, np.array([cutoff] * 2)
        )
    predictions, step = _run(node, worker)
    assert [unit.action for unit in step.series_units] == ["encode", "encode"]
    _fresh(functions, parameters, step, predictions)


def test_a_failed_extension_is_planned_again_from_the_entry_that_stands() -> None:
    """A step whose extension the worker never wrote is forgotten; the next grows the kept entry."""
    functions, parameters = timestral_functions, tiny_timestral()
    node, worker, table = _setup(functions, parameters, known=[])
    patch = functions.properties(parameters).patch
    node.submit_forecast(
        table, np.array([A, B], dtype=np.uint64), 4, np.array([220] * 2)
    )
    _run(node, worker)
    _advance(node, table, 220, patch)
    cutoff = 220 + STEP * patch
    node.submit_forecast(
        table, np.array([A, B], dtype=np.uint64), 4, np.array([cutoff] * 2)
    )
    step = node.schedule()
    assert step is not None and step.series_units[0].action == "extend"
    node.fail(step)
    node.submit_forecast(
        table, np.array([A, B], dtype=np.uint64), 4, np.array([cutoff] * 2)
    )
    predictions, again = _run(node, worker)
    assert again.series_units[0].action == "extend"
    assert again.series_units[0].source == step.series_units[0].source
    _fresh(functions, parameters, again, predictions)


def test_a_long_stream_compiles_a_handful_of_programs() -> None:
    """A hundred one-patch advances extend from entries stored at their buckets, so decode and extend compile per bucket, not per count."""
    import dataclasses

    functions = timestral_functions
    parameters = dataclasses.replace(tiny_timestral(), context_limit=1024)
    node, worker, table = _setup(functions, parameters, known=[])
    patch = functions.properties(parameters).patch
    cutoff = 220
    keys = np.array([A, B], dtype=np.uint64)
    node.submit_forecast(table, keys, 4, np.array([cutoff] * 2))
    _run(node, worker)
    actions: dict[str, int] = {}
    for advance in range(100):
        _advance(node, table, cutoff, patch)
        cutoff += STEP * patch
        node.submit_forecast(table, keys, 4, np.array([cutoff] * 2))
        predictions, step = _run(node, worker)
        action = step.series_units[0].action
        actions[action] = actions.get(action, 0) + 1
        if advance % 25 == 24:
            _fresh(functions, parameters, step, predictions)
    assert actions == {"extend": 100}
    traces: dict[str, int] = {}
    for key, function in worker.programs._compiled.items():
        name = cast(tuple[str, ...], key)[0]
        if name in ("series_extend", "series_decode"):
            traces[name] = traces.get(name, 0) + cast(Any, function)._cache_size()
    # Buckets from 4 to 128 patches: a decode per bucket, and an extension
    # per bucket read and bucket stored.
    assert traces["series_decode"] <= 8 and traces["series_extend"] <= 16, traces
