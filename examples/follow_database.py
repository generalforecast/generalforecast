# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""An example keeping a table in step with a database through Debezium events.

We apply hand-written events here, so the example runs without Kafka; in
practice one would attach `follow` to a Kafka consumer, as sketched at the end.

Run with:
  uv run python examples/follow_database.py
"""

import numpy as np

import generalforecast
from generalforecast.ingest import Columns, apply

generator = np.random.default_rng(0)


def row(key: int) -> dict:
    """A source row: a payment with an amount, an age, and whether it was fraud."""
    amount, age = float(generator.gamma(2.0, 50.0)), int(generator.integers(18, 90))
    return {"id": key, "amount": amount, "age": age, "fraud": int(amount > 180)}


def event(op: str, before: dict | None, after: dict | None) -> dict:
    """A Debezium change event, as its JSON decodes."""
    return {"payload": {"op": op, "before": before, "after": after}}


client = generalforecast.local()
payments = client.create_table(
    "payments",
    model="general-forecast/tabicl-v2-classifier",
    columns=["amount", "age"],
    classes=2,
)
# Events feed the table's columns by name; `features` can name others.
columns = Columns(key="id", target="fraud")

# Debezium sends a table's existing rows as reads ("r"), then each change.
source = {key: row(key) for key in range(300)}
apply(payments, [event("r", None, value) for value in source.values()], columns)
print("rows after the snapshot:", len(payments))

changed = dict(source[3], fraud=1)
changes = [
    event("u", source[3], changed),  # an update
    event("d", source[5], None),  # a delete
    event("c", None, row(300)),  # an insert
    None,  # a Kafka tombstone, skipped
]
# Kafka may deliver events twice, which changes nothing.
apply(payments, changes, columns)
apply(payments, changes, columns)
print("rows after the changes, applied twice:", len(payments))

queries = np.array([[20.0, 30], [400.0, 60]], dtype=np.float32)
print("fraud probability:", payments.predict(queries)[:, 1].round(3))

# With Kafka, `follow` commits offsets only once their changes are written, so a
# restart resumes where the writes stopped.
#
#   from confluent_kafka import Consumer
#   from generalforecast.ingest import follow
#
#   consumer = Consumer(
#       {"bootstrap.servers": "kafka:9092", "group.id": "generalforecast", "enable.auto.commit": False}
#   )
#   consumer.subscribe(["pg.public.payments"])
#   follow(payments, consumer, columns)
