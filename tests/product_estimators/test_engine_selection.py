# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The public tables' context selection, on the process's engine with a tiny
classifier that says it was trained on 16-row contexts."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from generalforecast import engine
from generalforecast._core import Strategy
from generalforecast.engine import LocalEngine
from generalforecast.native.tables import Retrieved, TimeOptions, default_selection

MODEL = "general-forecast/tabicl-v2-classifier"
CAP = 16


@pytest.fixture(scope="module")
def capped() -> Iterator[None]:
    tiny = LocalEngine(models={MODEL: "tests/tabicl-classifier-capped"})

    def local(surface: Any = "local", allow_downloads: bool = False) -> LocalEngine:
        return tiny.gated(surface, allow_downloads)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("generalforecast.engine.local", local)
        patch.setattr("generalforecast.native.estimators._shared", {})
        yield
    tiny.close()


def _rows(start: int, count: int) -> pd.DataFrame:
    generator = np.random.default_rng(start)
    ids = np.arange(start, start + count)
    return pd.DataFrame(
        {
            "k": ids,
            "f0": generator.normal(size=count) + ids % 2,
            "f1": generator.normal(size=count),
            "y": ids % 2,
        }
    )


def _table(name: str) -> engine.Table:
    return engine.Table(
        name, model=MODEL, schema=engine.Schema(target="y", id="k"), download=False
    )


def _labels(table: engine.Table) -> set[int]:
    return set(table.predict(_rows(1000, 3).drop(columns=["y"]))["label"])


def test_the_default_selection_follows_the_tables_size(capped) -> None:
    table = _table("growing")
    try:
        assert table.selection is None
        table.insert(_rows(0, 8))
        assert table.selection == Retrieved(core=None, neighbors=0)
        assert table.selection.describe() == "whole table"
        assert _labels(table) <= {0, 1}

        # Past the model's cap, the same table retrieves: a stratified core of
        # a quarter of the cap and the nearest rows that fill the rest.
        table.insert(_rows(8, 32))
        assert table.selection == default_selection(40, CAP)
        assert table.selection == Retrieved(core=4, neighbors=12, kind="stratified")
        assert (
            table.selection.describe()
            == "retrieved  core 4 stratified, 12 nearest per query"
        )
        assert _labels(table) <= {0, 1}

        # And back under it, every query reads the whole table again.
        table.delete(range(8, 40))
        assert table.selection == Retrieved(core=None, neighbors=0)
        assert _labels(table) <= {0, 1}
    finally:
        table.drop()


def test_a_set_selection_holds_until_none_follows_the_size_again(capped) -> None:
    table = _table("chosen")
    chosen = Retrieved(core=4, neighbors=4, kind="random", seed=3)
    try:
        table.set_selection(chosen)
        assert table.selection == chosen
        table.insert(_rows(0, 40))
        assert table.selection == chosen
        table.delete(range(20, 40))
        assert table.selection == chosen
        assert _labels(table) <= {0, 1}

        table.set_selection(Retrieved(core=6, neighbors=0))
        assert table.selection == Retrieved(core=6, neighbors=0, kind="stratified")
        assert _labels(table) <= {0, 1}

        table.set_selection(None)
        assert table.selection == default_selection(20, CAP)
        table.insert(_rows(40, 20))
        assert table.selection == default_selection(40, CAP)
        assert _labels(table) <= {0, 1}
    finally:
        table.drop()


def test_the_default_rule_is_the_engines() -> None:
    for cap in (2, 3, 7, 8, 10, 100, 1000, 100_000):
        for rows in (0, 1, cap - 1, cap, cap + 1, 2 * cap, 50 * cap):
            for timed in (False, True):
                core = round(0.25 * cap)
                neighbors = cap - core
                expected = (
                    Retrieved(core=None, neighbors=0)
                    if rows <= cap
                    else Retrieved(
                        core=core,
                        neighbors=neighbors,
                        kind="stratified",
                        recent=neighbors * 4 if timed else None,
                    )
                )
                selection = default_selection(rows, cap, timed=timed)
                assert selection == expected, (rows, cap, timed)
                assert selection.strategy(cap) == Strategy.adaptive().resolve(
                    rows, cap, timed
                ), (rows, cap, timed)


def test_contexts_past_the_cap_are_clamped_by_default() -> None:
    assert TimeOptions().past == "clamp"
    assert TimeOptions(past="warn").past == "warn"
