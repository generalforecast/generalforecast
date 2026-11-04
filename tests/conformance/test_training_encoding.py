# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A trainer's rows, encoded as the served predictor's model reads them.

`hosting.training.encode` must hand a trainer exactly the numbers the
predictor runtime hands the engine for the same rows: the context it binds
and the queries it scores, the class codes and the classes' order.
"""

import math

import numpy as np
import pyarrow as pa
import pytest

from generalforecast import hosting
from generalforecast.engine import Bound, Labeled
from tests.utils.engine import FakeEngine

CLASSIFIER = "general-forecast/tabicl-v2-classifier"
REGRESSOR = "general-forecast/tabicl-v2-regressor"


def _table(target: list) -> pa.Table:
    """Words and labels first seen out of sorted order, a boolean, and gaps."""
    return pa.table(
        {
            "color": ["red", "blue", None, "green", "red", "blue", "green", "red"],
            "flag": [True, False, None, True, True, False, False, True],
            "x": [1.5, None, 3.0, 4.0, 5.5, 6.0, None, 8.0],
            "k": [3, 1, 2, 7, 4, 4, 1, 9],
            "when": pa.array(
                [f"2026-01-0{day}T0{day}:00:00" for day in range(1, 9)]
            ).cast(pa.timestamp("ms")),
            "label": target,
        }
    )


class Recording(FakeEngine):
    """A fake engine that keeps the labeled context and queries it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.bound: list[Labeled] = []
        self.queries: list[np.ndarray] = []

    def bind(self, ref, context, strategy=None) -> Bound:
        self.bound.append(context)
        return super().bind(ref, context, strategy)

    def predict(self, bound, queries):
        self.queries.append(np.asarray(queries).copy())
        classes = len(bound.context.classes)
        if classes:
            return np.full((len(queries), classes), 1 / classes)
        return super().predict(bound, queries)


def _served(monkeypatch, model: str, table: pa.Table) -> Recording:
    """Runs the rows as an ad hoc predictor's context and queries, as a plane serves it."""
    engine = Recording()
    engine.install(monkeypatch)
    lock = hosting.ad_hoc_lock({"model": model, "target": "label"}, table)
    rows = hosting.served_context(lock, table)
    plan = hosting.compile_lock(lock, None, manifest=None)
    prepared = hosting.PredictorExecutor().prepare(plan, None)
    prepared.predict(table.drop(["label"]), context=rows)
    return engine


def _same(left: np.ndarray, right: np.ndarray) -> None:
    assert left.dtype == right.dtype and left.shape == right.shape
    np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "target",
    [
        ["zebra", "apple", "mango", "zebra", "apple", None, "zebra", "apple"],
        [10, 2, 1, 10, 2, None, 10, 1],
        [True, False, True, None, False, True, False, True],
    ],
    ids=["words", "integers", "booleans"],
)
def test_a_classifiers_rows_encode_as_its_served_predictor_reads_them(
    monkeypatch, target
):
    table = _table(target)
    encoded = hosting.training.encode(CLASSIFIER, table, "label")
    served = _served(monkeypatch, CLASSIFIER, table)
    (context,) = served.bound
    _same(encoded.features, context.context)
    _same(encoded.targets, context.targets)
    assert encoded.classes == context.classes
    assert encoded.task == "classification" and encoded.categorical is None
    # The served queries are every row, the unlabeled one included: the
    # labeled ones read as the context's rows do.
    labeled = [index for index, label in enumerate(target) if label is not None]
    _same(encoded.features, served.queries[0][labeled])
    # Classes in sorted order, whatever order they first appear in.
    present = [label for label in target if label is not None]
    assert list(encoded.classes) == sorted(set(present))
    assert [encoded.classes[code] for code in encoded.targets] == present
    assert encoded.columns == ("color", "flag", "x", "k", "when")


def test_the_encoding_is_the_binders(monkeypatch):
    """Words by first appearance, booleans as 0 and 1, times as parts, gaps as NaN."""
    table = _table(["b", "a", "b", "a", "b", "a", "b", "a"])
    encoded = hosting.training.encode(CLASSIFIER, table, "label")
    color, flag, x, k = (encoded.features[:, index] for index in range(4))
    nan = math.nan
    np.testing.assert_array_equal(color, [0, 1, nan, 2, 0, 1, 2, 0])
    np.testing.assert_array_equal(flag, [1, 0, nan, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(x, [1.5, nan, 3, 4, 5.5, 6, nan, 8])
    np.testing.assert_array_equal(k, [3, 1, 2, 7, 4, 4, 1, 9])
    # A timestamp takes five features: days, hour, weekday, day and month.
    assert encoded.features.shape == (8, 4 + 5)
    # 2026-01-01 at 01:00 was a Thursday.
    np.testing.assert_array_equal(encoded.features[0, 5:], [1, 3, 1, 1])


def test_a_regressors_rows_encode_as_its_served_predictor_reads_them(monkeypatch):
    table = _table([0.5, 2.0, None, 1.25, 3.0, 4.5, 0.0, 9.0])
    encoded = hosting.training.encode(REGRESSOR, table, "label")
    (context,) = _served(monkeypatch, REGRESSOR, table).bound
    _same(encoded.features, context.context)
    _same(encoded.targets, context.targets)
    assert encoded.task == "regression" and encoded.classes == ()
    np.testing.assert_array_equal(encoded.targets, [0.5, 2, 1.25, 3, 4.5, 0, 9])


def test_more_rows_than_a_context_takes_still_encode():
    """A trainer samples its own contexts, so the model's row cap does not bound it."""
    rows = 120_000
    table = pa.table(
        {"x": np.arange(rows, dtype=np.float64), "label": ["a", "b"] * (rows // 2)}
    )
    encoded = hosting.training.encode(CLASSIFIER, table, "label")
    assert encoded.features.shape == (rows, 1)
