# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The public Forecaster: its frequency spellings, its future grid and known-future rows."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from generalforecast import Forecaster
from generalforecast.engine import LocalEngine
from generalforecast.errors import InvalidInput

CATALOG = {"general-forecast/chronos-2": "tests/chronos2-catalog-forecaster"}


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


def _history(times: list[Any], *, price: bool = False) -> pd.DataFrame:
    count = len(times)
    frame = pd.DataFrame(
        {
            "store": ["a"] * count + ["b"] * count,
            "time": [*times, *times],
            "sales": np.arange(2 * count, dtype=float),
        }
    )
    if price:
        frame["price"] = np.tile(np.arange(count, dtype=float), 2)
    return frame


def test_future_grid_is_a_frame_of_the_steps_after_each_series(catalog) -> None:
    hours = pd.date_range("2024-01-01", periods=8, freq="h").to_list()
    estimator = Forecaster(
        target="sales", time="time", series="store", horizon=2, frequency="H"
    ).fit(_history(hours))
    grid = estimator.future_grid()
    assert isinstance(grid, pd.DataFrame)
    assert list(grid.columns) == ["store", "time"]
    assert grid["store"].tolist() == ["a", "a", "b", "b"]
    after = pd.date_range("2024-01-01 08:00", periods=2, freq="h", tz="UTC")
    assert grid["time"].tolist() == [*after, *after]
    assert estimator.forecast().shape == (4, 3)
    assert estimator.forecast(horizon=3).shape == (6, 3)
    estimator.close()


@pytest.mark.parametrize(
    ("given", "canonical", "freq"),
    [
        ("H", "1h", "h"),
        ("15min", "15min", "15min"),
        ("W-MON", "W-MON", "W-MON"),
        ("MS", "MS", "MS"),
    ],
)
def test_pandas_offset_aliases_are_frequencies(catalog, given, canonical, freq) -> None:
    times = pd.date_range("2024-01-01", periods=8, freq=freq).to_list()
    estimator = Forecaster(
        target="sales", time="time", series="store", horizon=2, frequency=given
    ).fit(_history(times))
    assert estimator._step == canonical
    expected = pd.date_range(times[-1], periods=3, freq=freq, tz="UTC")[1:]
    assert estimator.future_grid()["time"].tolist()[:2] == list(expected)
    assert estimator.forecast().shape == (4, 3)
    estimator.close()


def test_a_misspelled_frequency_is_refused(catalog) -> None:
    with pytest.raises(InvalidInput) as caught:
        Forecaster(target="sales", time="time", frequency="fast").fit(
            _history(list(range(8)))
        )
    assert caught.value.message == "frequency `fast` is not a frequency"
    assert caught.value.param == "frequency"


def test_forecast_takes_the_known_future_rows(catalog) -> None:
    estimator = Forecaster(
        target="sales",
        time="time",
        series="store",
        horizon=2,
        future_inputs=("price",),
    ).fit(_history(list(range(8)), price=True))
    with pytest.raises(InvalidInput) as caught:
        estimator.forecast()
    assert (
        caught.value.message
        == "this forecaster has known-future inputs; pass forecast(future=...)"
    )
    assert caught.value.param == "future"
    assert "price" in (caught.value.hint or "")
    future = estimator.future_grid().assign(price=[8.0, 9.0, 8.0, 9.0])
    values = estimator.forecast(future=future)
    assert values.shape == (4, 3) and np.isfinite(values).all()
    changed = estimator.forecast(future=future.assign(price=[80.0, 90.0, 80.0, 90.0]))
    assert not np.allclose(values, changed)
    with pytest.raises(InvalidInput):
        estimator.forecast(future=future.drop(columns=["price"]))
    estimator.close()
