# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tables placed on a server's workers by load, and moved between them."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

flight = pytest.importorskip("pyarrow.flight")

from tests.utils.server import COLUMNS, NEAR_A, Server, build

pytestmark = [pytest.mark.server, pytest.mark.slow]

LOAD = """
[placement]
assign = "load"
rebalance_seconds = 3600
"""


@pytest.fixture(scope="session")
def binary() -> Path:
    return build()


@pytest.fixture
def server(binary: Path, tmp_path: Path) -> Iterator[Server]:
    running = Server(binary, tmp_path, workers=2, placement=LOAD).start()
    yield running
    running.stop()


def _rows(rows: int = 32) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    centers = np.where(np.arange(rows) % 2 == 0, 0.0, 10.0)[:, None]
    features = (centers + generator.normal(size=(rows, 5))).astype(np.float32)
    return features, (np.arange(rows) % 3).astype(np.int32)


def _table(server: Server, name: str) -> None:
    server.action(
        "create_table",
        {
            "name": name,
            "model": "tiny",
            "columns": COLUMNS,
            "classes": 3,
            "strategy": {"kind": "exact"},
        },
    )
    features, targets = _rows()
    server.write("insert", name, np.arange(32), features, targets)


def _shards(server: Server) -> dict[str, int]:
    return server.action("placements")[0]["tables"]


def test_a_moved_table_answers_as_before_and_after_a_restart(server: Server) -> None:
    for name in ("first", "second"):
        _table(server, name)
    placed = _shards(server)
    assert sorted(placed.values()) == [0, 1]
    before = server.predict("first", NEAR_A)

    other = 1 - placed["first"]
    server.action("move_table", {"table": "first", "shard": other})

    assert _shards(server)["first"] == other
    np.testing.assert_allclose(server.predict("first", NEAR_A), before, atol=1e-6)
    features, targets = _rows()
    server.write("insert", "first", np.arange(32, 64), features, targets)
    # A restored table fits its state on its rows when first read, as after a
    # restart, so the refit is logged to compare like with like.
    server.action("refit", {"table": "first"})
    after_write = server.predict("first", NEAR_A)
    server.stop()
    server.start()
    np.testing.assert_allclose(server.predict("first", NEAR_A), after_write, atol=1e-6)


def test_calls_during_a_move_wait_and_succeed(server: Server) -> None:
    _table(server, "busy")
    expected = server.predict("busy", NEAR_A)
    start = _shards(server)["busy"]

    with ThreadPoolExecutor(8) as pool:
        answers = [pool.submit(server.predict, "busy", NEAR_A) for _ in range(24)]
        for shard in (1 - start, start, 1 - start):
            server.action("move_table", {"table": "busy", "shard": shard})
        results = [answer.result() for answer in answers]

    for result in results:
        np.testing.assert_allclose(result, expected, atol=1e-6)
    assert server.action("placements")[0]["moves"] == 3


def test_a_round_moves_load_off_the_busy_worker(server: Server) -> None:
    for name in ("a", "b", "c"):
        _table(server, name)
    placed = _shards(server)
    shared = [name for name, shard in placed.items() if shard == placed["a"]]
    assert len(shared) == 2, placed
    server.action("rebalance")  # starts the load count from here
    for _ in range(20):
        for name in shared:
            server.predict(name, NEAR_A + np.float32(_ % 3))

    round_ = server.action("rebalance")[0]

    assert round_["imbalance_before"] > 1.2
    assert len(round_["moved"]) == 1
    after = _shards(server)
    assert {after[name] for name in shared} == {0, 1}


def test_tables_move_only_when_placed_by_load(binary: Path, tmp_path: Path) -> None:
    server = Server(binary, tmp_path, workers=2).start()
    try:
        _table(server, "rows")
        with pytest.raises(Exception, match=r"placement\.assign"):
            server.action("move_table", {"table": "rows", "shard": 1})
    finally:
        server.stop()


def test_rounds_run_only_when_asked_without_an_interval(
    binary: Path, tmp_path: Path
) -> None:
    load = Server(
        binary, tmp_path, workers=2, placement='[placement]\nassign = "load"\n'
    )
    server = load.start()
    try:
        _table(server, "rows")
        round_ = server.action("rebalance")[0]
        assert round_["moved"] == []
        other = 1 - _shards(server)["rows"]
        server.action("move_table", {"table": "rows", "shard": other})
        assert _shards(server)["rows"] == other
    finally:
        server.stop()


def test_a_table_created_after_a_drop_is_placed(server: Server) -> None:
    _table(server, "first")
    server.action("drop_table", {"table": "first"})
    _table(server, "second")
    _table(server, "third")

    assert sorted(_shards(server).values()) == [0, 1]


def test_a_drop_during_a_move_leaves_the_name_free(server: Server) -> None:
    _table(server, "going")
    other = 1 - _shards(server)["going"]

    with ThreadPoolExecutor(2) as pool:
        moving = pool.submit(
            server.action, "move_table", {"table": "going", "shard": other}
        )
        dropping = pool.submit(server.action, "drop_table", {"table": "going"})
        dropping.result()
        # The move either finished first or found the table gone.
        try:
            moving.result()
        except (flight.FlightError, KeyError):
            pass

    assert "going" not in _shards(server)
    _table(server, "going")
    assert server.predict("going", NEAR_A).shape == (2, 3)


def test_a_slow_upload_holds_up_neither_a_move_nor_other_calls(server: Server) -> None:
    _table(server, "slow")
    expected = server.predict("slow", NEAR_A)
    other = 1 - _shards(server)["slow"]
    features, targets = _rows()
    columns = {"key": pa.array(np.arange(32, 64, dtype=np.uint64))}
    for index in range(features.shape[1]):
        columns[f"f{index}"] = pa.array(features[:, index])
    columns["target"] = pa.array(targets)
    batch = pa.RecordBatch.from_pydict(columns)
    descriptor = flight.FlightDescriptor.for_path("insert", "slow")
    writer, _ = server.client.do_put(descriptor, batch.schema, server.options())
    writer.write_batch(batch)  # the upload stays open while the table moves

    # Not a context manager: a call stuck behind the upload would otherwise
    # hold the test open with it, instead of failing it on the timeout.
    pool = ThreadPoolExecutor(2)
    moving = pool.submit(server.action, "move_table", {"table": "slow", "shard": other})
    moving.result(timeout=60)
    predicted = pool.submit(server.predict, "slow", NEAR_A).result(timeout=60)
    pool.shutdown(wait=False)
    np.testing.assert_allclose(predicted, expected, atol=1e-6)
    writer.close()

    assert _shards(server)["slow"] == other
    assert server.action("describe", {"table": "slow"})[0]["rows"] == 64


def test_tables_created_together_spread_over_workers(server: Server) -> None:
    _table(server, "hot")
    server.action("rebalance")
    for _ in range(10):
        server.predict("hot", NEAR_A)
    server.action("rebalance")  # the hot table's worker now has all the load

    for index in range(4):
        _table(server, f"new-{index}")

    placed = _shards(server)
    assert {placed[f"new-{index}"] for index in range(4)} == {0, 1}
