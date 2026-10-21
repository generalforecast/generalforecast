# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""An example serving fifty small tables, one per customer, from one client.

Each keeps its own rows and preprocessing, yet tables sharing a model run in the
same compiled calls, so a single `client.predict` answers all of them at once.

Run with:
  uv run python examples/many_tables.py
"""

import time

import numpy as np

import generalforecast

CUSTOMERS, ROWS = 50, 200
generator = np.random.default_rng(0)

client = generalforecast.local()
queries, answers = {}, {}
for customer in range(CUSTOMERS):
    weights = generator.normal(size=6)
    X = generator.normal(size=(ROWS + 20, 6)).astype(np.float32)
    y = (X @ weights > 0).astype(np.int32)
    table = client.create_table(
        f"customer-{customer}",
        model="general-forecast/tabicl-v2-classifier",
        features=6,
        classes=2,
    )
    table.insert(np.arange(ROWS), X[:ROWS], y[:ROWS])
    queries[table.name], answers[table.name] = X[ROWS:], y[ROWS:]

for attempt in ("first", "again"):
    started = time.perf_counter()
    predictions = client.predict(queries)
    seconds = time.perf_counter() - started
    accuracy = np.mean(
        [np.mean(predictions[name].argmax(1) == answers[name]) for name in queries]
    )
    print(
        f"{attempt:5}: {CUSTOMERS} tables in {seconds:.2f} s, accuracy {accuracy:.3f}"
    )

stats = client.stats
print(f"compiled calls per stage: {stats.calls}")
print(f"contexts encoded {stats.encoded}, reused {stats.reused}")
