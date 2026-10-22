# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Models beyond the default TabICL.

We fit Kumo Tabular with a column of store names, an ensemble of four estimators
(`*4`), and 25 classes on a model with 10 slots through error-correcting codes
(`@8`). To the engine, each is still one model.

Run with:
  uv run python examples/models.py

This example requires scikit-learn (uv sync --extra sklearn).
"""

from typing import Any

import numpy as np
from sklearn.datasets import make_classification

import generalforecast

generator = np.random.default_rng(0)
client = generalforecast.local()


def accuracy(table: generalforecast.Table, queries: Any, answers: np.ndarray) -> float:
    """The share of queries whose most probable class is right."""
    return float(np.mean(table.predict(queries).argmax(axis=1) == answers))


# The store column holds names, which the table codes and Kumo Tabular embeds
# as categories; any frame with these columns, such as a pandas one, does too.
names = np.array(["north", "south", "east", "west", "central"])
stores = names[generator.integers(0, 5, size=1_200)]
X = generator.normal(size=(1_200, 3)).astype(np.float32)
y = ((X[:, 0] > 0) ^ np.isin(stores, ["north", "east"])).astype(np.int32)
frame = {"store": stores, "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2]}
kumo = client.create_table(
    "stores",
    model="general-forecast/kumo-tabular-small-classifier",
    columns={"store": "categorical", "x1": "numeric", "x2": "numeric", "x3": "numeric"},
    classes=2,
)
kumo.insert(
    np.arange(1_000),
    {name: values[:1_000] for name, values in frame.items()},
    y[:1_000],
)
held_out = {name: values[1_000:] for name, values in frame.items()}
print("Kumo Tabular, store as a category:", accuracy(kumo, held_out, y[1_000:]))

X, y = make_classification(n_samples=1_200, n_features=8, random_state=0)
X = X.astype(np.float32)
for spec in (
    "general-forecast/tabicl-v2-classifier",
    "general-forecast/tabicl-v2-classifier*4",
):
    table = client.create_table(spec.split("/")[1], model=spec, features=8, classes=2)
    table.insert(np.arange(1_000), X[:1_000], y[:1_000])
    print(f"{spec}:", accuracy(table, X[1_000:], y[1_000:]))

X, y = make_classification(
    n_samples=3_000,
    n_features=12,
    n_informative=10,
    n_classes=25,
    n_clusters_per_class=1,
    random_state=0,
)
X = X.astype(np.float32)
many = client.create_table(
    "many", model="general-forecast/tabicl-v2-classifier@8", features=12, classes=25
)
many.insert(np.arange(2_500), X[:2_500], y[:2_500])
probabilities = many.predict(X[2_500:])
print(
    "25 classes, shape",
    probabilities.shape,
    "accuracy",
    accuracy(many, X[2_500:], y[2_500:]),
)
