# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The README's example, which the release runs on each built wheel."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import generalforecast as gf

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = gf.Classifier(
    "general-forecast/tabicl-v2-classifier",
    download=True,
).fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print("accuracy:", accuracy)
assert accuracy > 0.9, accuracy
