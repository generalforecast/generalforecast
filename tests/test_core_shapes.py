# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The extension reads a matrix's rows at its table's width, and refuses any other."""

import numpy as np
import pytest

from generalforecast import _core
from generalforecast.engine import Labeled, LocalEngine
from generalforecast.models.catalog import parse
from tests.utils.engine import tabular_request

FEATURES = 4


@pytest.fixture
def node() -> tuple[_core.Node, int]:
    node = _core.Node(1 << 24, 1 << 24)
    table = node.create_table(0, FEATURES, 2, _core.Strategy.exact(), 64)
    return node, table


def _keys(rows: int) -> np.ndarray:
    return np.stack([np.arange(rows), np.zeros(rows)], axis=1).astype(np.uint64)


def test_rows_of_another_width_are_refused_even_when_their_values_would_fit(node):
    node, table = node
    # Twelve values make three rows of four, but these are two rows of six.
    queries = np.zeros((2, 6), np.float32)
    for call in (
        lambda: node.submit(table, queries),
        lambda: node.submit_keys(table, queries, [_keys(1), _keys(1)]),
        lambda: node.contexts(table, queries),
        lambda: node.region_churn(table, queries),
    ):
        with pytest.raises(ValueError, match="rows of 6 values"):
            call()
    # Eight values for two keys are two rows of four, shaped as four of two.
    features = np.zeros((4, 2), np.float32)
    targets = np.zeros(2, np.int32)
    for write in (node.insert, node.update, node.upsert):
        with pytest.raises(ValueError, match="rows of 2 values"):
            write(table, _keys(2), features, targets)
    assert node.len(table) == 0

    node.insert(table, _keys(2), np.zeros((2, FEATURES), np.float32), targets)
    assert node.len(table) == 2
    assert node.submit(table, np.zeros((3, FEATURES), np.float32)) >= 0


def test_a_bound_table_refuses_queries_of_another_width():
    engine = LocalEngine(
        models={"general-forecast/tabicl-v2-classifier": "tests/tabicl-classifier"},
        cpu_threads=2,
    )
    try:
        generator = np.random.default_rng(0)
        context = generator.normal(size=(16, 5)).astype(np.float32)
        request = tabular_request(
            "general-forecast/tabicl-v2-classifier",
            context,
            [("a", "b")[index % 2] for index in range(16)],
            context[:2],
        )
        labeled = Labeled(
            request.task, request.context, request.targets, tuple(request.classes)
        )
        bound = engine.bind(parse("general-forecast/tabicl-v2-classifier"), labeled)
        shards = engine._shared.shards
        # Two rows of ten values would be read as four of five.
        with pytest.raises(ValueError, match="rows of 10 values"):
            shards.predict(bound.table, np.zeros((2, 10), np.float32))
        assert shards.predict(bound.table, context[:3])[0].shape[0] == 3
    finally:
        engine.close()
