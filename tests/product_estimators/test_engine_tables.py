# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The public engine tables, on the process's engine with the tiny catalog models."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from generalforecast import engine
from generalforecast.engine import LocalEngine
from generalforecast.errors import InvalidInput

CATALOG = {
    "general-forecast/tabicl-v2-classifier": "tests.models.tabicl.tiny:classifier",
    "general-forecast/chronos-2": "tests.models.chronos2.tiny:catalog_forecaster",
}


@pytest.fixture(scope="module")
def catalog() -> Iterator[None]:
    tiny = LocalEngine(models=CATALOG)

    def local(surface: Any = "local", allow_downloads: bool = False) -> LocalEngine:
        return tiny.gated(surface, allow_downloads)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("generalforecast.engine.local", local)
        patch.setattr("generalforecast.native.estimators._shared", {})
        yield
    tiny.close()


def test_tables_take_unlabeled_classifier_rows_until_labeled(catalog) -> None:
    rows = pd.DataFrame(
        {
            "k": np.arange(32),
            **{f"f{i}": np.random.default_rng(i).normal(size=32) for i in range(3)},
            "y": np.arange(32) % 2,
        }
    )
    pending = rows.tail(4).assign(y=None)
    table = engine.Table(
        "pending",
        model="general-forecast/tabicl-v2-classifier",
        schema=engine.Schema(target="y", id="k"),
        download=False,
    )
    try:
        table.insert(rows.head(28))
        # Rows whose label is not known yet take None (or NaN) for a classifier,
        # as they take NaN for a regressor; they are predicted but not fit on.
        table.insert(pending)
        table.insert(pending.assign(y=np.nan, k=np.arange(100, 104)))
        assert len(table) == 36
        scored = table.predict(pending.drop(columns=["y"]))
        assert scored["k"].tolist() == [28, 29, 30, 31]
        assert set(scored["label"]) <= {0, 1}
        table.label([28, 29, 30, 31], [0, 1, 1, 0])
        with pytest.raises(InvalidInput) as caught:
            table.label([100], [7])
        assert (
            caught.value.message
            == "target value `7` is not one of the classes from context"
        )
        table.label([100, 101, 102, 103], [None, 1, np.nan, 0])
        assert len(table) == 36
    finally:
        table.drop()


def test_series_tables_forecast_every_series_by_default(catalog) -> None:
    def observations(item: str, start: int, seed: int) -> pd.DataFrame:
        times = list(range(start, 20))
        return pd.DataFrame(
            {
                "item": [item] * len(times),
                "t": times,
                "units": np.random.default_rng(seed).normal(size=len(times)),
            }
        )

    table = engine.SeriesTable(
        "every-series",
        model="general-forecast/chronos-2",
        schema=engine.SeriesSchema(
            target="units", time="t", series="item", frequency=1
        ),
        horizon=3,
        download=False,
    )
    try:
        table.insert(observations("shoes", 0, 1))
        table.insert(observations("socks", 4, 2))
        table.insert(observations("shoes", 19, 3))
        everything = table.forecast()
        assert everything["item"].tolist() == ["shoes"] * 3 + ["socks"] * 3
        assert everything["step"].tolist() == [1, 2, 3] * 2
        chosen = table.forecast(["socks"])
        np.testing.assert_allclose(
            chosen["value"].to_numpy(), everything["value"].to_numpy()[3:]
        )
    finally:
        table.drop()
