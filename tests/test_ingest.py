"""Change data capture from Debezium events on Kafka."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

import generalforecast
from generalforecast.ingest import Columns, apply, follow
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.tabicl.tiny import tiny_tabicl

pytestmark = pytest.mark.slow

COLUMNS = Columns(key="id", features=("a", "b"), target="label")


class _Message:
    def __init__(self, value: dict | None, key: dict | None = None) -> None:
        self._value = None if value is None else json.dumps(value).encode()
        self._key = None if key is None else json.dumps(key).encode()

    def value(self) -> bytes | None:
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def error(self) -> None:
        return None


class _Consumer:
    """Hands out batches of messages, then nothing; counts commits."""

    def __init__(self, batches: list[list]) -> None:
        self.batches, self.commits = batches, 0

    def consume(self, count: int, timeout: float) -> list[_Message]:
        if not self.batches:
            return []
        return [
            value if isinstance(value, _Message) else _Message(value)
            for value in self.batches.pop(0)
        ]

    def commit(self, asynchronous: bool) -> None:
        self.commits += 1


def _row(key: int, a: float | None, label: int) -> dict:
    return {"id": key, "a": a, "b": float(key % 2), "label": label}


def _event(op: str, row: dict) -> dict:
    return {
        "payload": {
            "op": op,
            "before": row if op == "d" else None,
            "after": None if op == "d" else row,
        }
    }


def _table(client: generalforecast.LocalClient, name: str) -> generalforecast.Table:
    model = (tabicl_functions, tiny_tabicl(regression=False))
    return client.create_table(name, model=model, features=2, classes=3, page_size=8)


def test_followed_events_match_the_source_even_when_delivered_twice() -> None:
    snapshot: list[dict | None] = [
        _event("r", _row(key, float(key), key % 3)) for key in range(16)
    ]
    changes: list[dict | None] = [
        _event("u", _row(3, 30.0, 2)),
        _event("d", _row(5, 5.0, 2)),
        None,
        _event("c", _row(16, None, 1)),
        _event("u", _row(16, 16.0, 0)),
    ]
    client = generalforecast.local()
    table = _table(client, "followed")
    consumer = _Consumer([snapshot, changes, changes])

    follow(table, consumer, COLUMNS, stop=lambda: not consumer.batches)

    expected = {key: _row(key, float(key), key % 3) for key in range(17)}
    expected[3], expected[16] = _row(3, 30.0, 2), _row(16, 16.0, 0)
    del expected[5]
    source = _table(client, "source")
    source.insert(
        list(expected),
        [[row["a"], row["b"]] for row in expected.values()],
        [row["label"] for row in expected.values()],
    )
    queries = np.array([[2.0, 0.0], [15.0, 1.0]], dtype=np.float32)
    assert consumer.commits == 3
    assert len(table) == 16
    np.testing.assert_allclose(
        table.predict(queries), source.predict(queries), rtol=1e-6, atol=1e-6
    )


def test_decimals_keyless_deletes_and_bad_events_are_handled() -> None:
    # Debezium's default for numeric(10, 2): the unscaled value's bytes in base64.
    schema = {
        "fields": [
            {
                "field": "after",
                "fields": [
                    {
                        "field": "a",
                        "name": "org.apache.kafka.connect.data.Decimal",
                        "parameters": {"scale": "2"},
                    }
                ],
            }
        ]
    }
    amount = base64.b64encode((1234).to_bytes(2, "big")).decode()
    row = {"id": 1, "a": amount, "b": 1.0, "label": 1}
    decimal = {**_event("c", row), "schema": schema}
    keyless = {"payload": {"op": "d", "before": None, "after": None}}
    skipped = []
    consumer = _Consumer(
        [
            [
                _event("c", _row(0, 0.0, 0)),
                decimal,
                _event("c", {"id": 2}),
                _event("c", _row(2**130, 1.0, 1)),
            ],
            [_Message(keyless, key={"payload": {"id": 0}})],
        ]
    )
    client = generalforecast.local()
    table = _table(client, "followed")

    follow(
        table,
        consumer,
        COLUMNS,
        stop=lambda: not consumer.batches,
        skip=lambda message, error: skipped.append(error),
    )

    source = _table(client, "source")
    source.insert([1], [[12.34, 1.0]], [1])
    queries = np.array([[12.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    assert consumer.commits == 2
    assert [str(error) for error in skipped] == [
        "a 'c' event has no 'label'",
        f"integer key {2**130} does not fit 128 bits",
    ]
    assert len(table) == 1
    np.testing.assert_allclose(
        table.predict(queries), source.predict(queries), rtol=1e-6
    )


def test_a_dropped_table_stops_following_without_committing() -> None:
    client = generalforecast.local()
    table = _table(client, "followed")
    consumer = _Consumer([[_event("c", _row(0, 0.0, 0))]])
    client.drop_table("followed")
    skipped = []

    with pytest.raises(KeyError, match="does not exist"):
        follow(
            table,
            consumer,
            COLUMNS,
            stop=lambda: not consumer.batches,
            skip=lambda message, error: skipped.append(error),
        )

    assert consumer.commits == 0
    assert skipped == []


def test_events_feed_columns_by_their_kinds() -> None:
    client = generalforecast.local()
    model = (tabicl_functions, tiny_tabicl(regression=False))
    kinds = {"store": "categorical", "when": "datetime"}
    table = client.create_table("typed", model=model, columns=kinds, classes=3)
    schema = {
        "fields": [
            {
                "field": "after",
                "fields": [{"field": "when", "name": "io.debezium.time.Timestamp"}],
            }
        ]
    }
    events = [
        {
            **_event(
                "c",
                {
                    "id": key,
                    "store": ["north", "south"][key % 2],
                    "when": 86_400_000 * key,
                    "label": key % 3,
                },
            ),
            "schema": schema,
        }
        for key in range(8)
    ]
    columns = Columns(key="id", target="label")

    assert apply(table, events, columns) == 8

    assert table.vocabularies() == {"store": ["north", "south"]}
    # A datetime becomes days since 1970, hour, day of the week (1970-01-01
    # was a Thursday), day of the month and month.
    plain = client.create_table("plain", model=model, features=6, classes=3)
    rows = [[key % 2, key, 0, (key + 3) % 7, key + 1, 1] for key in range(8)]
    plain.insert(range(8), rows, [key % 3 for key in range(8)])
    queries = {"store": ["south", "west"], "when": ["1970-01-04", None]}
    codes = np.array([[1, 3, 0, 6, 4, 1], [np.nan] * 6], np.float32)
    np.testing.assert_allclose(
        table.predict(queries), plain.predict(codes), rtol=1e-6, atol=1e-6
    )
    unscheduled = _event("c", {"id": 9, "store": "x", "when": 5, "label": 0})
    with pytest.raises(ValueError, match="schemas"):
        apply(table, [unscheduled], columns)


def test_string_targets_become_a_classifiers_labels() -> None:
    table = _table(generalforecast.local(), "labelled")
    names = ["low", "mid", "high"]
    events = [
        _event("c", {**_row(key, float(key), 0), "label": names[key % 3]})
        for key in range(9)
    ]

    apply(table, events, COLUMNS)

    assert table.labels == ["low", "mid", "high"]
    assert len(table) == 9


def test_truncates_string_keys_and_dated_times_follow_the_source() -> None:
    model = (tabicl_functions, tiny_tabicl(regression=False))
    table = generalforecast.local().create_table(
        "dated",
        model=model,
        columns={"a": "numeric", "b": "boolean"},
        classes=3,
        times="datetime",
    )
    schema = {
        "fields": [
            {
                "field": "after",
                "fields": [{"field": "at", "name": "io.debezium.time.MicroTimestamp"}],
            }
        ]
    }

    def event(key: str, flag: str) -> dict:
        row = {"id": key, "a": 1.0, "b": flag, "label": 0, "at": 86_400_000_000}
        return {**_event("c", row), "schema": schema}

    events = [
        event("acct-1", "true"),
        event("acct-2", "false"),
        {"payload": {"op": "t"}},
        event("acct-3", "TRUE"),
    ]
    columns = Columns(key="id", target="label", features=("a", "b"), time="at")

    assert apply(table, events, columns) == 4
    # The truncate took the rows before it, not the one after.
    assert len(table) == 1
    # The row's time is a day after 1970 began, so only a later cutoff reads it.
    table.predict([[1.0, 1.0]], before=np.datetime64("1970-01-03"))
    with pytest.raises(ValueError, match="precedes every row"):
        table.predict([[1.0, 1.0]], before=np.datetime64("1970-01-01"))
