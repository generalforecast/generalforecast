# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Classification and regression with generalforecast's scikit-learn estimators.

Since a tabular foundation model learns from the rows it is shown when it
predicts, fitting only stores them and nothing is trained. For data that changes
over time, one may prefer the tables in examples/local_tables.py.

Run with:
  uv run python examples/sklearn_quickstart.py

This example requires scikit-learn (uv sync --extra sklearn).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import cross_val_score, train_test_split

import generalforecast

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = generalforecast.Classifier("general-forecast/tabicl-v2-classifier").fit(
    X_train, y_train
)
print("accuracy:", classifier.score(X_test, y_test))
print("first probabilities:", classifier.predict_proba(X_test[:3]).round(3))

scores = cross_val_score(
    generalforecast.Classifier("general-forecast/tabicl-v2-classifier"), X, y, cv=3
)
print("cross-validated accuracy:", scores.round(3))

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

regressor = generalforecast.Regressor("general-forecast/tabicl-v2-regressor").fit(
    X_train, y_train
)
print("R^2:", regressor.score(X_test, y_test))

# A regressor predicts quantiles, so a 90% interval is the 5th and 95th.
low, high = regressor.predict_quantiles(X_test, [0.05, 0.95]).T
covered = np.mean((y_test >= low) & (y_test <= high))
print(f"90% intervals cover {covered:.0%} of the test rows")
for target, a, median, b in zip(
    y_test[:3], low[:3], regressor.predict(X_test[:3]), high[:3], strict=True
):
    print(
        f"  actual {target:6.1f}, median {median:6.1f}, interval [{a:6.1f}, {b:6.1f}]"
    )
