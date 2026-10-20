"""Change data capture: keeps a table in step with a source database.

Debezium turns a database's log (Postgres logical replication, the MySQL binlog
and others) into change events, usually on Kafka. `apply` writes a run of them
to a table, and `follow` reads them from a Kafka consumer, committing offsets
once their events are written. Writes are upserts and deletes that skip missing
keys, so events delivered twice, as Kafka does after a restart, change nothing.

Values go to the table's columns by their kinds. Decimals arrive as Debezium
encodes them by default, base64 with the scale in the event's schema, or as
numbers or strings under its other `decimal.handling.mode` settings. Dates and
times arrive as the counts Debezium's schema names, or as ISO 8601 strings;
strings in categorical columns take codes from the table's vocabulary. A
truncate event empties the table, after the events before it and before
those after it.
"""

import base64
import json
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from generalforecast.columns import Column, Unit
from generalforecast.native.client import Table

logger = logging.getLogger(__name__)

# Debezium's names for decimals: a fixed scale in the schema, or one per value.
_DECIMAL = "org.apache.kafka.connect.data.Decimal"
_VARIABLE_DECIMAL = "io.debezium.data.VariableScaleDecimal"
# Debezium's names for dates and times, and the unit each counts in.
_UNITS: dict[str, Unit] = {
    "io.debezium.time.Date": "D",
    "org.apache.kafka.connect.data.Date": "D",
    "io.debezium.time.Timestamp": "ms",
    "org.apache.kafka.connect.data.Timestamp": "ms",
    "io.debezium.time.MicroTimestamp": "us",
    "io.debezium.time.NanoTimestamp": "ns",
}


@dataclass(frozen=True)
class Columns:
    """Where a table's rows come from in the source's rows.

    Attributes:
        key: The primary key column: integers, strings or UUIDs.
        target: The target column: class labels, as strings or codes
            `0 .. classes - 1`, or values.
        features: Feature columns, in the table's order; the table's own column
            names by default. Nulls and columns an event lacks are missing.
        time: A column of times, if rows carry times: integers for a table of
            integer times, or dates and times for a datetime table.
    """

    key: str
    target: str
    features: tuple[str, ...] = ()
    time: str | None = None

    def names(self, table: Sequence[Column]) -> tuple[str, ...]:
        """The source columns feeding a table's columns, in order."""
        names = self.features or tuple(column.name for column in table)
        if len(names) != len(table):
            raise ValueError(
                f"the table has {len(table)} columns, but {len(names)} are named"
            )
        return names


@dataclass(frozen=True)
class _Change:
    """One event's change to one row: its values, or None for a delete."""

    key: int | str
    features: list[Any] | None = None
    target: float | str | None = None
    time: Any = None


@dataclass(frozen=True)
class _Truncate:
    """A truncate event: every row goes."""


def _fields(event: Mapping[str, Any], side: str) -> dict[str, Mapping[str, Any]]:
    """The schemas of a row's columns, by name, if the event carries its schema."""
    for field in event.get("schema", {}).get("fields", ()):
        if field.get("field") == side:
            return {column["field"]: column for column in field.get("fields", ())}
    return {}


def _number(name: str, value: Any, field: Mapping[str, Any] | None) -> float:
    """A column's value as a number, decoding Debezium's decimals."""
    if value is None:
        return math.nan
    if isinstance(value, Mapping) and "scale" in value and "value" in value:
        return _decimal(value["value"], int(value["scale"]))
    if isinstance(value, str):
        if field is not None and field.get("name") == _DECIMAL:
            return _decimal(value, int(field.get("parameters", {}).get("scale", 0)))
        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"column {name!r} holds {value!r}, which looks like a base64 decimal; "
                "send events with their schemas, or set decimal.handling.mode=double"
            ) from None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"column {name!r} holds {value!r}, not a number")
    return float(value)


def _decimal(encoded: str, scale: int) -> float:
    """A decimal from Debezium's base64 of its unscaled two's complement bytes."""
    unscaled = int.from_bytes(base64.b64decode(encoded), "big", signed=True)
    return unscaled / 10**scale


def _moment(name: str, value: Any, field: Mapping[str, Any] | None) -> Any:
    """A date or time as a datetime64 in the unit its schema names, or an ISO string."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"column {name!r} holds {value!r}, not a date or time")
    unit = _UNITS.get((field or {}).get("name", ""))
    if unit is None:
        raise ValueError(
            f"column {name!r} holds the number {value!r}; send events with their "
            "schemas, which say whether it counts days, milliseconds or microseconds"
        )
    return np.datetime64(int(value), unit)


def _cell(name: str, value: Any, field: Mapping[str, Any] | None, kind: str) -> Any:
    """A feature column's value as its kind takes it; None for a missing category."""
    if kind == "categorical":
        if value is None or isinstance(value, str):
            return value
        return _number(name, value, field)
    if kind == "datetime":
        return None if value is None else _moment(name, value, field)
    if kind == "boolean" and isinstance(value, bool):
        return float(value)
    if (
        kind == "boolean"
        and isinstance(value, str)
        and value.lower() in ("true", "false")
    ):
        return float(value.lower() == "true")
    return _number(name, value, field)


def _integer(name: str, value: Any) -> int:
    """A time column's value, which must be a whole number."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"column {name!r} holds {value!r}, not an integer")
    return value


def _key(name: str, value: Any) -> int | str:
    """A key column's value: an integer, or a string such as a UUID's."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(
            f"column {name!r} holds {value!r}, not an integer or string key"
        )
    return value


def _change(
    event: Mapping[str, Any] | None,
    columns: Columns,
    table: Sequence[Column],
    times: str = "integer",
    classes: int | None = None,
) -> _Change | _Truncate | None:
    """The change an event makes, checked; None for a tombstone."""
    if event is None:
        return None
    names = columns.names(table)
    payload = event.get("payload", event)
    op = payload.get("op")
    if op == "t":
        return _Truncate()
    if op == "d":
        before = payload.get("before")
        if before is None or columns.key not in before:
            raise ValueError(
                "a delete event without its row's key; follow reads it from the "
                "Kafka message key"
            )
        return _Change(_key(columns.key, before[columns.key]))
    if op not in ("c", "r", "u"):
        raise ValueError(f"cannot apply change events of kind {op!r}")
    row, fields = payload.get("after") or {}, _fields(event, "after")
    for name in (columns.key, columns.target, columns.time):
        if name is not None and row.get(name) is None:
            raise ValueError(f"a {op!r} event has no {name!r}")
    return _Change(
        key=_key(columns.key, row[columns.key]),
        features=[
            _cell(name, row.get(name), fields.get(name), column.kind)
            for name, column in zip(names, table, strict=True)
        ],
        target=_target(
            columns.target, row[columns.target], fields.get(columns.target), classes
        ),
        time=None
        if columns.time is None
        else _time(columns.time, row[columns.time], fields.get(columns.time), times),
    )


def _target(
    name: str, value: Any, field: Mapping[str, Any] | None, classes: int | None
) -> float | str:
    """A row's target: a classifier's label as a string, or a number."""
    if classes is not None and isinstance(value, str):
        return value
    return _number(name, value, field)


def _time(name: str, value: Any, field: Mapping[str, Any] | None, times: str) -> Any:
    """A row's time, as its table's times are: an integer, or a date or time."""
    if times == "datetime":
        return _moment(name, value, field)
    return _integer(name, value)


def _write(
    table: Table, changes: Iterable[_Change | _Truncate], columns: Columns
) -> None:
    """Writes changes in order, emptying the table at each truncate between them."""
    run: list[_Change] = []
    for change in changes:
        if isinstance(change, _Truncate):
            _write_rows(table, run, columns)
            run = []
            table.truncate()
        else:
            run.append(change)
    _write_rows(table, run, columns)


def _write_rows(table: Table, changes: Iterable[_Change], columns: Columns) -> None:
    """Writes changes as one delete and one upsert; each key's last change counts."""
    latest = {change.key: change for change in changes}
    deleted = [key for key, change in latest.items() if change.features is None]
    if deleted:
        table.delete(deleted, missing_ok=True)
    rows = [change for change in latest.values() if change.features is not None]
    if rows:
        table.upsert(
            [row.key for row in rows],
            np.array([row.features for row in rows], dtype=object),
            [row.target for row in rows],
            None if columns.time is None else [row.time for row in rows],
        )


def apply(
    table: Table, events: Iterable[Mapping[str, Any] | None], columns: Columns
) -> int:
    """Writes change events to a table.

    Only each key's last event counts, so the events become one delete and one
    upsert.

    Args:
        table: The table to write to.
        events: Debezium change events decoded from JSON, with or without their
            schemas; None, as Kafka's tombstones decode, is skipped.
        columns: Where the table's rows come from.

    Returns:
        Number of events applied.

    Raises:
        ValueError: If an event is not a create, read, update, delete or
            truncate with the key, target and time it needs, or `columns` does not name one
            source column per table column. Nothing is written then.
        TypeError: If an event holds a value its column cannot take, a key that
            is neither an integer nor a string, or a time that is not an integer.
            Nothing is written then.
    """
    changes = [
        change
        for event in events
        if (
            change := _change(event, columns, table.columns, table.times, table.classes)
        )
    ]
    _write(table, changes, columns)
    return len(changes)


def _event(message: Any) -> Mapping[str, Any] | None:
    """A Kafka message's change event; a delete without its row takes the key's."""
    if message.value() is None:
        return None
    event = json.loads(message.value())
    payload = event.get("payload", event)
    if payload.get("op") == "d" and payload.get("before") is None and message.key():
        key = json.loads(message.key())
        payload["before"] = key.get("payload", key)
    return event


def _skip(message: Any, error: Exception) -> None:
    """Logs a message that cannot be applied, so following goes on past it."""
    logger.error(
        "skipping a change event at %s[%s] offset %s: %s",
        message.topic(),
        message.partition(),
        message.offset(),
        error,
    )


def follow(
    table: Table,
    consumer: Any,
    columns: Columns,
    batch: int = 4096,
    timeout: float = 1.0,
    stop: Callable[[], bool] = lambda: False,
    skip: Callable[[Any, Exception], None] = _skip,
) -> None:
    """Applies change events from a Kafka consumer to a table until `stop` is true.

    A message that cannot be applied, being malformed or refused by the table,
    goes to `skip` and its offset is committed with the rest, so one bad message
    never holds up the topic. Failures to reach the table raise instead, and
    so does a table that no longer exists, without committing: its events
    wait in Kafka for a table to apply them to.

    Args:
        table: The table to write to.
        consumer: A `confluent_kafka.Consumer` subscribed to the source table's
            topic, with `enable.auto.commit` off: offsets are committed only once
            their events are written, so a restart resumes where writes stopped.
        columns: Where the table's rows come from.
        batch: Most events written at once.
        timeout: Seconds to wait for events before checking `stop`.
        stop: Called between batches; following ends when it returns True.
        skip: Called with each message that cannot be applied, and why; logs
            it by default.

    Raises:
        RuntimeError: If the consumer reports an error.
        KeyError: If the table no longer exists, as after it was dropped.
    """
    while not stop():
        messages = consumer.consume(batch, timeout)
        if not messages:
            continue
        changes = []
        for message in messages:
            if message.error() is not None:
                raise RuntimeError(f"Kafka: {message.error()}")
            try:
                change = _change(
                    _event(message), columns, table.columns, table.times, table.classes
                )
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                skip(message, error)
                continue
            if change is not None:
                changes.append((message, change))
        # A KeyError from a write is the table's own: it no longer exists.
        try:
            _write(table, (change for _, change in changes), columns)
        except (ValueError, TypeError):
            # The table refused something; writing one at a time finds what.
            for message, change in changes:
                try:
                    _write(table, [change], columns)
                except (ValueError, TypeError) as error:
                    skip(message, error)
        consumer.commit(asynchronous=False)
