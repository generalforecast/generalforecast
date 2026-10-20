"""An example of a local table whose rows change between predictions.

We insert, update and delete rows, predicting after each change. Only the cached
contexts that held a changed row are encoded again, so we never retrain
anything. A server takes the same calls, as shown in examples/server/client.py.

Run with:
  uv run python examples/local_tables.py

This example requires scikit-learn (uv sync --extra sklearn).
"""

import numpy as np
from sklearn.datasets import make_classification

import generalforecast

X, y = make_classification(
    n_samples=600, n_features=8, n_informative=5, n_classes=3, random_state=0
)
X = X.astype(np.float32)
keys = np.arange(len(X))

client = generalforecast.local()
table = client.create_table(
    "accounts", model="general-forecast/tabicl-v2-classifier", features=8, classes=3
)

# Rows may come as NumPy arrays, or pandas, polars or Arrow tables.
table.insert(keys[:500], X[:500], y[:500])
queries = X[500:]
print("rows:", len(table))
print("accuracy:", np.mean(table.predict(queries).argmax(axis=1) == y[500:]))

table.update(keys[:50], X[:50], (y[:50] + 1) % 3)
print("after relabeling 50 rows:", np.mean(table.predict(queries).argmax(1) == y[500:]))

# An upsert replaces known keys and inserts the rest, so one may safely repeat it.
table.upsert(keys[:100], X[:100], y[:100])
table.upsert(keys[:100], X[:100], y[:100])
print("rows after upserting twice:", len(table))

table.delete(keys[400:500])
table.delete(np.array([10_000]), missing_ok=True)
print("rows after deleting:", len(table))

gappy = queries.copy()
gappy[:, 0] = np.nan
print("with a missing column:", np.mean(table.predict(gappy).argmax(1) == y[500:]))

# The table keeps the preprocessing it was fit with until we refit it.
table.refit()
print("after refit:", np.mean(table.predict(queries).argmax(1) == y[500:]))

client.drop_table("accounts")
print("tables:", client.tables())
