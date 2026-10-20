"""A comparison of the strategies that decide which rows each query reads.

At the 4,000 rows here, `exact`, where every query reads every row, is both the
fastest and the most accurate. One would reach for a retrieved context, a
sample every query shares plus each query's nearest rows, once a table no
longer fits in one context.

Run with:
  uv run python examples/context_strategies.py

This example requires scikit-learn (uv sync --extra sklearn).
"""

import time

import numpy as np
from sklearn.datasets import make_classification

import generalforecast

X, y = make_classification(
    n_samples=4_200,
    n_features=10,
    n_informative=6,
    n_clusters_per_class=4,
    random_state=0,
)
X = X.astype(np.float32)
rows, queries, answers = X[:4_000], X[4_000:], y[4_000:]

strategies = {
    "exact": generalforecast.Strategy.exact(),
    # One 512-row sample, encoded once and shared by every query.
    "subsample": generalforecast.Strategy.subsample(512, seed=0),
    # Each query's 512 nearest rows, encoded as its own context.
    "neighbors": generalforecast.Strategy.neighbors(512),
    # A 128-row sample stratified by class, plus each query's 384 nearest rows.
    "retrieved": generalforecast.Strategy.retrieved(384, core=128, kind="stratified"),
}

client = generalforecast.local()
for name, strategy in strategies.items():
    table = client.create_table(
        name,
        model="general-forecast/tabicl-v2-classifier",
        features=10,
        classes=2,
        strategy=strategy,
        page_size=128,
    )
    table.insert(np.arange(len(rows)), rows, y[:4_000])
    # The first call compiles and encodes; the second only decodes.
    for attempt in ("first", "again"):
        started = time.perf_counter()
        predicted = table.predict(queries).argmax(axis=1)
        seconds = time.perf_counter() - started
        print(
            f"{name:24} {attempt:5}  accuracy {np.mean(predicted == answers):.3f}"
            f"  {seconds:6.2f} s"
        )
