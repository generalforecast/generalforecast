# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The frames a server and its workers exchange, read on the worker's side."""

from __future__ import annotations

import json
import socket
import struct

import numpy as np
import pytest

from generalforecast import protocol


def _read(body: bytes) -> tuple[dict, list[memoryview]]:
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(struct.pack("<Q", len(body)) + body)
        return protocol.read_frame(receiver)


def test_frames_round_trip() -> None:
    sender, receiver = socket.socketpair()
    with sender, receiver:
        protocol.write_frame(sender, {"kind": "ok"}, [b"abc", b""])
        header, buffers = protocol.read_frame(receiver)

    assert header == {"kind": "ok"}
    assert [bytes(buffer) for buffer in buffers] == [b"abc", b""]


def test_workers_refuse_frames_past_the_limit_or_their_own_end() -> None:
    header = json.dumps({"kind": "ok"}).encode()
    whole = struct.pack("<I", len(header)) + header + struct.pack("<IQ", 1, 3) + b"abc"

    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(struct.pack("<Q", protocol.MAX_FRAME_BYTES + 1))
        with pytest.raises(ValueError, match="a frame claims"):
            protocol.read_frame(receiver)
    with pytest.raises(ValueError, match="past its end"):
        _read(whole[:-1])
    with pytest.raises(ValueError, match="past its end"):
        _read(struct.pack("<I", 1000) + header)
    with pytest.raises(ValueError, match="after the frame"):
        _read(whole + b"x")


def test_workers_refuse_unknown_tiers_and_key_kinds() -> None:
    def apply(key: list[object], source: str) -> object:
        movement = {"key": key, "source": source, "target": None, "slots": []}
        return protocol.request({"kind": "apply", "movements": [movement]}, [])

    moved = apply([1, 2, 3, "context", "4"], "host")
    assert isinstance(moved, protocol.Apply)
    assert moved.movements[0].key == protocol.CacheKey(1, 2, 3, "context", 4)
    with pytest.raises(ValueError, match="unknown tier 'attic'"):
        apply([1, 2, 3, "context", "4"], "attic")
    with pytest.raises(ValueError, match="rows or context"):
        apply([1, 2, 3, "pages", "4"], "host")


def test_series_frames_read_and_write_as_the_server_does() -> None:
    import numpy as np

    values = np.arange(6, dtype="<f4").reshape(2, 3)
    observed = np.array([[1, 1, 1], [0, 1, 1]], dtype="u1")
    future = np.zeros((2, 2), dtype="<f4")
    buffers = [
        memoryview(values.tobytes()),
        memoryview(observed.tobytes()),
        memoryview(np.array([0, 1], dtype="<i4").tobytes()),
        memoryview(future.tobytes()),
        memoryview(np.ones((2, 2), dtype="u1").tobytes()),
    ]
    header = {
        "movements": [],
        "fits": [],
        "pages": [],
        "units": [],
        "queries": [],
        "series_units": [
            {
                "table": 1,
                "key": [1, 2, 3, "context", "4"],
                "members": ["7"],
                "cutoff": 12,
                "rows": 2,
                "steps": 3,
                "values": 0,
                "observed": 1,
                "roles": 2,
                "action": "encode",
                "slots": [0, 1],
            }
        ],
        "forecasts": [
            {
                "request": 9,
                "table": 1,
                "epoch": 2,
                "unit": 0,
                "target": 0,
                "rows": 2,
                "horizon": 2,
                "values": 3,
                "observed": 4,
            }
        ],
    }
    step = protocol.step(header, buffers)
    unit, forecast = step.series_units[0], step.forecasts[0]
    assert unit.members.tolist() == [7] and unit.cutoff == 12
    assert unit.observed.tolist() == [[True, True, True], [False, True, True]]
    assert unit.roles.tolist() == [0, 1]
    assert forecast.horizon == 2 and forecast.future_observed.all()

    header, _ = protocol.predictions({9: np.zeros((1, 2, 3), dtype="<f4")})
    assert header["results"][0]["horizon"] == 2
    assert header["results"][0]["columns"] == 3
    assert "compile_cache" not in header
    timing = protocol.timing(0.5, {"fit": (2, 0.25, 1e9)}, compile_cache=(3, 1))
    assert timing["stages"] == {"fit": {"calls": 2, "seconds": 0.25, "flops": 1e9}}
    assert timing["compile_cache"] == {"hits": 3, "misses": 1}

    tabular = protocol.Profiled(1, 2, True, 0, 1, None, 0, False)
    assert "series" not in protocol.reply(tabular)
    forecaster = protocol.Profiled(
        1, 2, True, 0, 9, None, 0, False, series=True, patch=16
    )
    assert protocol.reply(forecaster)["patch"] == 16


def test_controls_carry_tables_and_tiers_changes_in_one_frame() -> None:
    header = {
        "kind": "controls",
        "requests": [
            {
                "kind": "add_table",
                "table": 3,
                "model": "tiny",
                "classes": 2,
                "features": 4,
                "categorical": [1],
                "page_size": 64,
                "summary": "table",
                "series": False,
            },
            {"kind": "remove_table", "table": 2},
        ],
    }
    controls = protocol.request(header, [])
    assert isinstance(controls, protocol.Controls)
    added, removed = controls.requests
    assert isinstance(added, protocol.AddTable) and added.categorical == (1,)
    assert removed == protocol.RemoveTable(2)
    failed = protocol.Controlled([protocol.ControlFailure(1, "KeyError: 2")])
    assert protocol.reply(failed) == {
        "kind": "controlled",
        "failures": [{"index": 1, "message": "KeyError: 2"}],
    }
    with pytest.raises(ValueError, match="must be controls"):
        protocol.request({"kind": "controls", "requests": [{"kind": "record"}]}, [])


def test_progress_replies_carry_the_warm_up_count_only_when_given() -> None:
    assert protocol.reply(protocol.Progress()) == {"kind": "progress"}
    assert protocol.reply(protocol.Progress(3, 12)) == {
        "kind": "progress",
        "done": 3,
        "total": 12,
    }


def _server_levels(values, known, levels):
    """The server's `quantile_values`, level by level, as its Rust computes them."""
    rows, columns = values.shape
    picked = np.zeros((rows, len(levels)), np.float32)
    for index, level in enumerate(levels):
        upper = min(int(np.searchsorted(known, level, side="left")), columns - 1)
        lower = max(upper - 1, 0)
        span = known[upper] - known[lower]
        weight = min(max((level - known[lower]) / span, 0.0), 1.0) if span > 0 else 0.0
        for row in range(rows):
            low, high = float(values[row, lower]), float(values[row, upper])
            picked[row, index] = np.float32(low * (1.0 - weight) + high * weight)
    return picked


def test_workers_take_the_levels_a_caller_asks_as_the_server_would() -> None:
    generator = np.random.default_rng(0)
    known = np.arange(1, 1000) / 1000
    values = np.sort(generator.normal(size=(5, 999)).astype(np.float32), axis=1)
    # Unordered and repeated levels keep their order; levels outside the
    # known ones take the outermost; one exactly at a known level is its column.
    levels = np.asarray([0.5, 0.05, 0.95, 0.0005, 0.9995, 0.5, 0.123456])

    picked = protocol.at_levels(values, known, levels)

    assert picked.dtype == np.float32
    assert picked.shape == (5, len(levels))
    np.testing.assert_array_equal(picked, _server_levels(values, known, levels))
    np.testing.assert_array_equal(picked[:, 0], values[:, 499])


def test_query_batches_carry_the_levels_their_caller_asked() -> None:
    features = np.zeros((2, 3), "<f4")
    header = {
        "kind": "step",
        "movements": [],
        "fits": [],
        "pages": [],
        "units": [],
        "queries": [
            {
                "request": 1,
                "table": 0,
                "epoch": 0,
                "count": 2,
                "features": 0,
                "units": [0, 0],
            },
            {
                "request": 2,
                "table": 0,
                "epoch": 0,
                "count": 2,
                "features": 0,
                "units": [0, 0],
                "known": [0.25, 0.5, 0.75],
                "levels": [0.5, 0.1],
            },
        ],
        "series_units": [],
        "forecasts": [],
    }

    plain, picked = protocol.step(header, [memoryview(features.tobytes())]).queries

    assert plain.known is None and plain.levels is None
    np.testing.assert_array_equal(picked.known, [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(picked.levels, [0.5, 0.1])


def test_the_columns_a_level_reads_give_its_value_alone() -> None:
    generator = np.random.default_rng(1)
    known = np.arange(1, 1000) / 1000
    values = np.sort(generator.normal(size=(7, 999)).astype(np.float32), axis=1)
    for levels in (
        np.asarray([0.1, 0.5, 0.9]),
        np.asarray([0.5, 0.05, 0.95, 0.0005, 0.9995, 0.5, 0.123456]),
        np.asarray([0.0001]),
        np.asarray([0.99995, 0.001]),
    ):
        columns = protocol.level_columns(known, levels)

        assert len(columns) <= 2 * len(levels)
        np.testing.assert_array_equal(
            protocol.at_levels(values[:, columns], known[columns], levels),
            protocol.at_levels(values, known, levels),
        )
