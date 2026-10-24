# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The remote client against a running server: the same tables as local mode."""

from __future__ import annotations

import datetime
import pickle
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from pyarrow import flight

from tests.utils import engine_client as generalforecast
from tests.utils.server import NEAR_A, Server, build

NEAR_B = np.full((2, 5), 10.0, dtype=np.float32)


pytestmark = pytest.mark.server


@pytest.fixture(scope="module")
def binary() -> Path:
    return build()


@pytest.fixture
def server(binary: Path, tmp_path: Path) -> Iterator[Server]:
    running = Server(binary, tmp_path).start()
    yield running
    running.stop()


def _connect(server: Server) -> generalforecast.RemoteClient:
    return generalforecast.connect(f"grpc://127.0.0.1:{server.port}", token="secret")


def _clusters(rows: int = 32) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    centers = np.where(np.arange(rows) % 2 == 0, 0.0, 10.0)[:, None]
    features = (centers + generator.normal(size=(rows, 5))).astype(np.float32)
    return features, (np.arange(rows) % 3).astype(np.int32)


def test_remote_tables_match_local_tables(server: Server) -> None:
    features, targets = _clusters()
    remote = _connect(server).create_table("rows", model="tiny", features=5, classes=3)
    local = generalforecast.local().create_table(
        "rows", model="tests.models.tabicl.tiny:classifier", features=5, classes=3
    )

    for table in (remote, local):
        table.insert(np.arange(32), features, targets)
        table.predict(NEAR_A)
        table.update([1], features[[1]] + 1.0, [2])
        table.delete([4])
        table.refit()

    np.testing.assert_allclose(remote.predict(NEAR_A), local.predict(NEAR_A), atol=1e-5)
    assert len(remote) == len(local) == 31


def test_string_and_uuid_keys_work_alike_locally_and_on_a_server(
    server: Server,
) -> None:
    features, targets = _clusters(4)
    keys = ["acct-1", "acct-2", uuid.UUID(int=7), -5]
    remote = _connect(server).create_table("rows", model="tiny", features=5, classes=3)
    local = generalforecast.local().create_table(
        "rows", model="tests.models.tabicl.tiny:classifier", features=5, classes=3
    )

    for table in (remote, local):
        table.insert(keys, features, targets)
        table.upsert(["acct-1", "acct-9"], features[:2], targets[:2])
        table.delete([-5, "missing"], missing_ok=True)
        assert len(table) == 4
        # Errors name the key as given, not its digest.
        with pytest.raises(KeyError, match="acct-404"):
            table.delete(["acct-404"])

    np.testing.assert_allclose(remote.predict(NEAR_A), local.predict(NEAR_A), atol=1e-5)


def test_remote_cutoffs_match_local_ones(server: Server) -> None:
    features, targets = _clusters()
    times = np.arange(32) * 10
    remote = _connect(server).create_table("rows", model="tiny", features=5, classes=3)
    local = generalforecast.local().create_table(
        "rows", model="tests.models.tabicl.tiny:classifier", features=5, classes=3
    )
    for table in (remote, local):
        table.insert(np.arange(32), features, targets, times=times)

    before = np.array([100, 250])
    np.testing.assert_allclose(
        remote.predict(NEAR_A, before=before),
        local.predict(NEAR_A, before=before),
        atol=1e-5,
    )


def test_dated_tables_reopen_dated_and_truncate_like_local_ones(server: Server) -> None:
    features, targets = _clusters()
    stamps = np.datetime64("2026-10-01") + np.arange(32).astype("timedelta64[h]")
    client = _connect(server)
    remote = client.create_table(
        "dated", model="tiny", features=5, classes=3, times="datetime"
    )
    local = generalforecast.local().create_table(
        "dated",
        model="tests.models.tabicl.tiny:classifier",
        features=5,
        classes=3,
        times="datetime",
    )
    for table in (remote, local):
        table.insert(np.arange(32), features, targets, times=stamps)

    reopened = _connect(server).table("dated")
    assert reopened.times == "datetime"
    cutoff = np.datetime64("2026-10-01T12:00")
    np.testing.assert_allclose(
        reopened.predict(NEAR_A, before=cutoff),
        local.predict(NEAR_A, before=cutoff),
        atol=1e-5,
    )
    for table in (remote, local):
        table.truncate()
        assert len(table) == 0


def test_string_labels_are_coded_on_the_server_and_survive_restarts(
    server: Server,
) -> None:
    features, targets = _clusters()
    names = np.array(["cat", "dog", "fish"])[targets]
    remote = _connect(server).create_table("pets", model="tiny", features=5, classes=3)
    local = generalforecast.local().create_table(
        "pets", model="tests.models.tabicl.tiny:classifier", features=5, classes=3
    )
    for table in (remote, local):
        table.insert(np.arange(32), features, names)
    expected = local.predict(NEAR_A)

    np.testing.assert_allclose(remote.predict(NEAR_A), expected, atol=1e-5)
    assert remote.labels == ["cat", "dog", "fish"]
    with pytest.raises(ValueError, match="labels"):
        remote.insert([99], features[:1], [0])
    with pytest.raises(ValueError, match="would be"):
        remote.insert([99], features[:1], ["eel"])
    # The log carries the labels, through a compaction and a restart.
    remote.compact()
    server.stop()
    server.start()
    reopened = _connect(server).table("pets")
    assert reopened.labels == ["cat", "dog", "fish"]
    np.testing.assert_allclose(reopened.predict(NEAR_A), expected, atol=1e-5)


def test_retrieval_columns_place_rows_as_a_local_tables_do(server: Server) -> None:
    features, targets = _clusters()
    columns = {**{f"f{index}": "numeric" for index in range(5)}, "r": "retrieval"}
    strategy = generalforecast.Strategy.neighbors(4)
    local = generalforecast.local().create_table(
        "placed",
        model="tests.models.tabicl.tiny:classifier",
        columns=columns,
        classes=3,
        strategy=strategy,
    )
    remote = _connect(server).create_table(
        "placed", model="tiny", columns=columns, classes=3, strategy=strategy
    )
    placed = np.where(np.arange(32) % 2 == 0, 10.0, 0.0).astype(np.float32)
    queries = np.column_stack([NEAR_A, [0, 10]])
    for table in (local, remote):
        table.insert(np.arange(32), np.column_stack([features, placed]), targets)

    # Through a restart, which restores the table's retrieval space.
    server.stop()
    server.start()
    reopened = _connect(server).table("placed")

    for here, there in zip(
        local.contexts(queries), reopened.contexts(queries), strict=True
    ):
        np.testing.assert_array_equal(here, there)
    np.testing.assert_allclose(
        reopened.predict(queries), local.predict(queries), atol=1e-5
    )
    assert reopened.columns[-1] == generalforecast.Column("r", "retrieval")


def test_target_encodings_survive_a_restart_as_a_local_tables_place_rows(
    server: Server,
) -> None:
    codes = np.arange(32) % 10
    targets = np.where(np.isin(codes, [0, 9]), 0, 1 + codes % 2).astype(np.int32)
    rows = np.column_stack([np.zeros(32), codes]).astype(np.float32)
    settings: dict[str, Any] = {
        "columns": {"f": "numeric", "c": "categorical"},
        "classes": 3,
        "strategy": generalforecast.Strategy.neighbors(7),
        "retrieval_categoricals": "target",
    }
    local = generalforecast.local().create_table(
        "encoded", model="tests.models.tabicl.tiny:classifier", **settings
    )
    remote = _connect(server).create_table("encoded", model="tiny", **settings)
    for table in (local, remote):
        table.insert(np.arange(32), rows, targets)

    server.stop()
    server.start()
    reopened = _connect(server).table("encoded")

    np.testing.assert_array_equal(
        reopened.contexts(rows[:1])[0], local.contexts(rows[:1])[0]
    )
    assert reopened.describe().retrieval_categoricals == "target"


def test_labels_reach_a_servers_rows_and_survive_a_restart(server: Server) -> None:
    features, targets = _clusters(40)
    unlabeled = np.concatenate([targets[:32], np.full(8, -1, np.int32)])
    strategy = generalforecast.Strategy.neighbors(24)
    local = generalforecast.local().create_table(
        "late",
        model="tests.models.tabicl.tiny:classifier",
        features=5,
        classes=3,
        strategy=strategy,
    )
    remote = _connect(server).create_table(
        "late", model="tiny", features=5, classes=3, strategy=strategy
    )
    for table in (local, remote):
        table.insert(np.arange(40), features, unlabeled)
    assert remote.contexts(NEAR_A)[0][:, 0].max() < 32

    for table in (local, remote):
        table.label(np.arange(32, 40), targets[32:])
    server.stop()
    server.start()
    reopened = _connect(server).table("late")

    np.testing.assert_array_equal(
        reopened.contexts(NEAR_A)[0], local.contexts(NEAR_A)[0]
    )
    assert reopened.contexts(NEAR_A)[0][:, 0].max() >= 32


def test_a_server_predicts_against_given_rows_as_a_local_table_does(
    server: Server,
) -> None:
    features, targets = _clusters()
    local = generalforecast.local().create_table(
        "chosen", model="tests.models.tabicl.tiny:classifier", features=5, classes=3
    )
    remote = _connect(server).create_table(
        "chosen", model="tiny", features=5, classes=3
    )
    chosen = [[1, 3, 5, 7], [0, 2, 4, 6, 8]]
    for table in (local, remote):
        table.insert(np.arange(32), features, targets)

    np.testing.assert_allclose(
        remote.predict_given(NEAR_A, chosen),
        local.predict_given(NEAR_A, chosen),
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="either its given rows"):
        remote._client._predict(remote, NEAR_A, 8, None, given=chosen)


def test_contexts_hold_the_keys_a_prediction_reads(server: Server) -> None:
    features, targets = _clusters()
    strategy = generalforecast.Strategy.retrieved(4, core=4, kind="stratified")
    tables = [
        client.create_table(
            "rows",
            model="tests.models.tabicl.tiny:classifier",
            features=5,
            classes=3,
            strategy=strategy,
        )
        if client is not None
        else _connect(server).create_table(
            "rows", model="tiny", features=5, classes=3, strategy=strategy
        )
        for client in (generalforecast.local(), None)
    ]
    for table in tables:
        table.insert(np.arange(32), features, targets, times=np.arange(32))

    local, remote = (table.contexts(NEAR_A, before=[16, 32]) for table in tables)

    for here, there in zip(local, remote, strict=True):
        np.testing.assert_array_equal(here, there)
    assert [len(keys) for keys in remote] == [8, 8]
    assert (remote[0][:, 0] < 16).all() and (remote[0][:, 1] == 0).all()
    assert tables[1].contexts(NEAR_A[:0]) == []


def test_a_table_that_summarizes_each_context_predicts_as_a_local_one(
    server: Server,
) -> None:
    features, _ = _clusters()
    strategy = generalforecast.Strategy.neighbors(8)
    remote = _connect(server).create_table(
        "contexts",
        model="tiny-regressor",
        features=5,
        strategy=strategy,
        page_size=8,
        summary="context",
    )
    local = generalforecast.local().create_table(
        "contexts",
        model="tests.models.tabicl.tiny:regressor",
        features=5,
        strategy=strategy,
        page_size=8,
        summary="context",
    )
    for table in (remote, local):
        table.insert(np.arange(32), features, features.sum(axis=1))

    assert remote.describe().summary == "context"
    np.testing.assert_allclose(remote.predict(NEAR_A), local.predict(NEAR_A), atol=1e-5)


def test_tables_describe_what_they_are_and_pick_quantiles_on_the_server(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    features, _ = _clusters()
    values = features.sum(axis=1)
    client = _connect(server)
    strategy = generalforecast.Strategy.neighbors(64)
    # Each fits on half the rows, the same half.
    remote = client.create_table(
        "values",
        model="tiny-regressor",
        features=5,
        strategy=strategy,
        page_size=8,
        fit_rows=16,
        past_cap="clamp",
        region_rows=16,
        rebuild=datetime.timedelta(seconds=30),
        rebuild_wait=generalforecast.RebuildWait(steps=3, seconds=None),
        refit_cutoffs="never",
        scale_rows=16,
    )
    local = generalforecast.local().create_table(
        "values",
        model="tests.models.tabicl.tiny:regressor",
        features=5,
        strategy=strategy,
        page_size=8,
        fit_rows=16,
        region_rows=16,
        rebuild=datetime.timedelta(seconds=30),
        rebuild_wait=generalforecast.RebuildWait(steps=3, seconds=None),
        refit_cutoffs="never",
        scale_rows=16,
    )
    for table in (remote, local):
        table.insert(np.arange(32), features, values)

    reopened = _connect(server).table("values")
    described = reopened.describe()
    assert (reopened.model, reopened.strategy) == ("tiny-regressor", strategy)
    assert (described.rows, described.page_size, described.labels) == (32, 8, None)
    assert described.fit_rows == 16
    # A model that is not built in says nothing of its trained contexts.
    assert (described.past_cap, described.context_rows) == ("clamp", None)
    assert described.region_rows == 16
    assert described.rebuild == datetime.timedelta(seconds=30)
    assert described.rebuild_wait == generalforecast.RebuildWait(steps=3, seconds=None)
    assert described.region_stats is not None
    assert (described.refit_cutoffs, described.scale_rows) == ("never", 16)
    assert (
        repr(reopened)
        == "<RemoteTable 'values': 5 columns, values, model 'tiny-regressor'>"
    )
    assert repr(client) == f"<RemoteClient grpc://127.0.0.1:{server.port}>"
    levels = [0.05, 0.5, 0.95]
    picked = reopened.predict(NEAR_A, quantiles=levels)
    assert picked.shape == (2, 3)
    np.testing.assert_allclose(
        picked, local.predict(NEAR_A, quantiles=levels), atol=1e-5
    )
    with pytest.raises(ValueError, match="quantile levels"):
        reopened.predict(NEAR_A, quantiles=[1.0])
    # A's region is built and unchanged since; B's is not built.
    queries = np.concatenate([NEAR_A, NEAR_B])
    churn = reopened.region_churn(queries)
    np.testing.assert_array_equal(churn, [0.0, 0.0, np.nan, np.nan])
    np.testing.assert_array_equal(churn, local.region_churn(queries))
    # Each names the state that answered, whose epoch a refit raises, and the
    # worker time it took.
    for table in (reopened, local):
        assert table.last_answer is not None
        epoch = table.last_answer.epoch
        assert epoch is not None and table.last_answer.gpu_seconds > 0
        table.refit()
        table.predict(NEAR_A)
        assert table.last_answer.epoch == epoch + 1
    # Queries the server refuses are not sent again when the table is unchanged.
    calls = []
    sent = client._predict
    monkeypatch.setattr(
        client, "_predict", lambda *args: calls.append(1) or sent(*args)
    )
    with pytest.raises(ValueError):
        client.predict({"values": features[:, :4]})
    assert len(calls) == 1
    with pytest.warns(UserWarning, match="token"):
        pickle.dumps(client)


def test_predictions_follow_a_table_dropped_and_made_again(server: Server) -> None:
    features, targets = _clusters()
    client = _connect(server)
    first = client.create_table("rows", model="tiny", features=5, classes=3)
    first.insert(np.arange(32), features, targets)
    client.predict({"rows": NEAR_A})

    # Another client makes the table again with other columns; the description
    # this client remembers is refused, so it describes the table again.
    other = _connect(server)
    other.drop_table("rows")
    wider = other.create_table(
        "rows", model="tiny", columns=[f"c{i}" for i in range(6)], classes=3
    )
    wider.insert(np.arange(32), np.column_stack([features, features[:, 0]]), targets)

    queries = np.zeros((2, 6), dtype=np.float32)
    np.testing.assert_allclose(
        client.predict({"rows": queries})["rows"], wider.predict(queries), atol=1e-5
    )


def test_tables_reopen_by_name(server: Server) -> None:
    features, targets = _clusters()
    created = _connect(server).create_table(
        "rows",
        model="tiny",
        features=5,
        classes=3,
        strategy=generalforecast.Strategy.neighbors(16),
    )
    created.insert(np.arange(32), features, targets)
    expected = created.predict(np.concatenate([NEAR_A, NEAR_B]))

    client = _connect(server)
    reopened = client.table("rows")

    assert client.tables() == ["rows"]
    assert len(reopened) == 32
    np.testing.assert_array_equal(
        reopened.predict(np.concatenate([NEAR_A, NEAR_B])), expected
    )


def test_server_errors_raise_python_exceptions(server: Server) -> None:
    client = _connect(server)
    rows = client.create_table("rows", model="tiny", features=5, classes=3)
    features, targets = _clusters()
    rows.insert(np.arange(32), features, targets)

    with pytest.raises(KeyError):
        client.table("missing")
    with pytest.raises(ValueError, match="exists"):
        client.create_table("rows", model="tiny", features=5, classes=3)
    with pytest.raises(ValueError, match="key 0"):
        rows.insert([0], features[:1], targets[:1])
    with pytest.raises(PermissionError):
        generalforecast.connect(
            f"grpc://127.0.0.1:{server.port}", token="wrong"
        ).tables()


def test_arrow_tables_are_accepted_as_rows(server: Server) -> None:
    features, targets = _clusters()
    columns = {f"f{index}": features[:, index] for index in range(5)}
    rows = _connect(server).create_table("rows", model="tiny", features=5, classes=3)
    rows.insert(
        pa.array(np.arange(32, dtype=np.uint64)), pa.table(columns), pa.array(targets)
    )

    from_arrow = rows.predict(
        pa.table({f"f{index}": NEAR_A[:, index] for index in range(5)})
    )

    np.testing.assert_array_equal(from_arrow, rows.predict(NEAR_A))


def test_deletes_models_and_closing_reach_the_server(server: Server) -> None:
    features, targets = _clusters()
    with _connect(server) as client:
        rows = client.create_table("rows", model="tiny", features=5, classes=3)
        rows.insert(np.arange(32), features, targets)
        rows.delete(key for key in range(4))
        rows.delete([2, 100], missing_ok=True)
        with pytest.raises(KeyError, match="key 2"):
            rows.delete([2])

        assert len(rows) == 28
        assert client.models() == ["tiny", "tiny-forecaster", "tiny-regressor"]
        assert client.table("rows").outputs == 3
        assert rows.predict(NEAR_A[:0]).shape == (0, 3)


def test_the_async_client_runs_the_same_calls(server: Server) -> None:
    import asyncio

    features, targets = _clusters()

    async def run() -> tuple[list[str], np.ndarray, int, dict[str, np.ndarray]]:
        url = f"grpc://127.0.0.1:{server.port}"
        async with generalforecast.connect_async(
            url, token="secret", concurrency=4
        ) as client:
            table = await client.create_table(
                "rows", model="tiny", features=5, classes=3
            )
            await table.insert(np.arange(32), features, targets)
            await table.update([1], features[[1]] + 1.0, [2])
            await table.delete([4])
            many = await client.predict({"rows": NEAR_A})
            return (
                await client.tables(),
                await table.predict(NEAR_A),
                await table.count(),
                many,
            )

    tables, predicted, count, many = asyncio.run(run())

    assert tables == ["rows"] and count == 31
    assert predicted.shape == (2, 3)
    np.testing.assert_array_equal(many["rows"], predicted)


def test_strings_booleans_and_times_are_coded_on_the_server(server: Server) -> None:
    client = _connect(server)
    kinds = {
        "store": "categorical",
        "amount": "numeric",
        "paid": "boolean",
        "when": "datetime",
        "count": "numeric",
    }
    remote = client.create_table("typed", model="tiny", columns=kinds, classes=3)
    local = generalforecast.local().create_table(
        "typed", model="tests.models.tabicl.tiny:classifier", columns=kinds, classes=3
    )
    features, targets = _clusters()
    frame = pa.table(
        {
            "store": np.array(["north", "south", "east"])[np.arange(32) % 3],
            "amount": features[:, 1],
            "paid": features[:, 2] > 0,
            # Before 1970, a leap day, and times seconds apart, which the
            # server must split into the same features as the client.
            "when": np.array(
                [
                    "1969-12-31T23:59:59.999",
                    "2024-02-29T13:45:30.250",
                    "2024-02-29T13:45:31.750",
                    "1900-03-01T00:00:00",
                ],
                dtype="datetime64[ms]",
            )[np.arange(32) % 4],
            "count": features[:, 4],
        }
    )
    for table in (remote, local):
        table.insert(np.arange(32), frame, targets)
    query = {
        "store": ["south", "west"],
        "amount": [0.0, 0.0],
        "paid": [True, None],
        "when": ["1970-01-02T12:00:00", None],
        "count": [0.0, 0.0],
    }

    expected = local.predict(query)
    np.testing.assert_allclose(remote.predict(query), expected, rtol=1e-5, atol=1e-5)
    assert remote.vocabularies() == {"store": ["north", "south", "east"]}
    assert client.table("typed").columns == remote.columns
    # A count of rows, or the labels, leaves the words on the server.
    assert len(remote) == 32
    bare = server.action("describe", {"table": "typed", "words": False})[0]
    assert (bare["rows"], bare["vocabularies"]) == (32, {})
    # Typed Arrow columns reach the server as they are, and decode the same.
    typed = pa.RecordBatch.from_pydict(
        {
            "store": pa.array(["south", "west"]),
            "amount": pa.array([0.0, 0.0], pa.float64()),
            "paid": pa.array([True, None]),
            "when": pa.array(
                np.array(["1970-01-02T12:00:00", "NaT"], dtype="datetime64[s]")
            ),
            "count": pa.array([0, 0], pa.int32()),
        }
    )
    descriptor = flight.FlightDescriptor.for_path("predict", "typed")
    writer, reader = server.client.do_exchange(descriptor, server.options())
    writer.begin(typed.schema)
    writer.write_batch(typed)
    writer.done_writing()
    raw = np.column_stack([column.to_numpy() for column in reader.read_all().columns])
    writer.close()
    np.testing.assert_allclose(raw, expected, rtol=1e-5, atol=1e-5)


def test_series_tables_forecast_on_a_server_as_locally(server: Server) -> None:

    settings: dict[str, Any] = {
        "series": "item",
        "time": "t",
        "target": "units",
        "step": 10,
        "columns": {"price": "numeric"},
        "known": ["price"],
        "horizon": 5,
    }
    client = _connect(server)
    remote = client.create_series_table("sales", model="tiny-forecaster", **settings)
    local = generalforecast.local().create_series_table(
        "sales", model="tests.models.timestral.tiny:forecaster", **settings
    )
    # A table of targets alone, as a local one may be.
    plain = client.create_series_table(
        "plain",
        model="tiny-forecaster",
        series="item",
        time="t",
        target="units",
        step=10,
    )
    assert len(plain) == 0
    generator = np.random.default_rng(0)
    times = list(range(100, 200, 10))
    frame = {
        "item": ["shoes"] * 10 + ["socks"] * 10,
        "t": times + times,
        "units": generator.normal(size=20).astype(np.float32),
        "price": np.linspace(1.0, 2.0, 20, dtype=np.float32),
    }
    # Prices stored ahead, with no units yet.
    ahead = {
        "item": ["shoes"] * 3,
        "t": [200, 210, 220],
        "units": [None, None, None],
        "price": [2.1, 2.2, 2.3],
    }
    for table in (remote, local):
        table.insert(frame)
        table.insert(ahead)
    assert len(remote) == len(local) == 23

    forecasts = remote.forecast(["shoes", "socks"], before=200)

    np.testing.assert_allclose(
        forecasts, local.forecast(["shoes", "socks"], before=200), rtol=1e-4, atol=1e-4
    )
    assert forecasts.shape == (2, 5, 9)
    assert remote.last_answer is not None and remote.last_answer.gpu_seconds > 0

    # A table of given groups takes each forecast's group over the wire.
    given: dict[str, Any] = {
        "group": generalforecast.GroupStrategy.given(2),
        **settings,
    }
    given_remote = client.create_series_table("given", model="tiny-forecaster", **given)
    given_local = generalforecast.local().create_series_table(
        "given", model="tests.models.timestral.tiny:forecaster", **given
    )
    for table in (given_remote, given_local):
        table.insert(frame)
        table.insert(ahead)
    assert given_remote.definition.group == generalforecast.GroupStrategy.given(2)
    np.testing.assert_allclose(
        given_remote.forecast_given(["shoes"], [["shoes", "socks"]], before=200),
        given_local.forecast_given(["shoes"], [["shoes", "socks"]], before=200),
        rtol=1e-4,
        atol=1e-4,
    )
    with pytest.raises(KeyError, match="does not exist"):
        given_remote.forecast_given(["shoes"], [["shoes", "nowhere"]], before=200)
    picked = remote.forecast(["socks"], horizon=3, quantiles=[0.5, 0.15])
    np.testing.assert_allclose(
        picked,
        local.forecast(["socks"], horizon=3, quantiles=[0.5, 0.15]),
        rtol=1e-4,
        atol=1e-4,
    )

    # A reconnecting client finds the table by its description.
    reopened = _connect(server).series_table("sales")
    assert (reopened.series, reopened.time, reopened.target) == ("item", "t", "units")
    assert (reopened.step, reopened.horizon, reopened.known) == (10, 5, ("price",))
    assert reopened.quantile_levels == local.quantile_levels
    with pytest.raises(TypeError, match="holds series"):
        client.table("sales")
    with pytest.raises(ValueError, match="series 'socks'"):
        remote.insert({"item": ["socks"], "t": [195], "units": [1.0], "price": [1.0]})
    remote.delete({"item": ["socks"], "t": [190]})
    assert len(remote) == 22
