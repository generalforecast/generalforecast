# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Frequencies: the spellings taken and the one kept."""

import datetime

import pytest

from generalforecast.predictors import frequency
from generalforecast.predictors.spec import frequency_seconds, parse_text


def _at(year: int, month: int, day: int) -> datetime.datetime:
    return datetime.datetime(year, month, day)  # noqa: DTZ001


@pytest.mark.parametrize(
    ("given", "canonical", "seconds"),
    [
        ("1d", "1d", 86400.0),
        ("D", "1d", 86400.0),
        ("7d", "1w", 604800.0),
        ("W", "1w", 604800.0),
        ("1h", "1h", 3600.0),
        ("H", "1h", 3600.0),
        ("24h", "1d", 86400.0),
        ("15min", "15min", 900.0),
        ("15T", "15min", 900.0),
        ("90s", "90s", 90.0),
        ("500ms", "500ms", 0.5),
        ("W-MON", "W-MON", 604800.0),
        ("w-mon", "W-MON", 604800.0),
        ("2W-SUN", "2W-SUN", 2 * 604800.0),
        ("MS", "MS", 2629746.0),
        ("M", "ME", 2629746.0),
        ("QS", "QS", 7889238.0),
        ("YS", "YS", 31556952.0),
        ("A", "YE", 31556952.0),
        (7, 7, 7.0),
        (0.5, 0.5, 0.5),
        (datetime.timedelta(hours=36), "36h", 129600.0),
    ],
)
def test_every_spelling_normalizes_to_one(given, canonical, seconds) -> None:
    assert frequency.normalize(given) == canonical
    assert frequency.normalize(canonical) == canonical
    assert frequency.seconds(given) == seconds
    assert frequency_seconds(given) == seconds


@pytest.mark.parametrize("bad", ["fast", "0d", "1y", "W-XYZ", "", "1.5h", -1, 0, True])
def test_spellings_of_nothing_are_refused(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        frequency.normalize(bad)


def test_calendar_aliases_step_by_the_calendar() -> None:
    assert not frequency.is_calendar("1w") and frequency.is_calendar("W-MON")
    origin = datetime.date(2024, 1, 31)
    assert frequency.grid(origin, 3, "MS") == [
        datetime.date(2024, 2, 1),
        datetime.date(2024, 3, 1),
        datetime.date(2024, 4, 1),
    ]
    monday = _at(2024, 1, 1)
    weeks = frequency.grid(monday, 2, "W-MON")
    assert weeks == [_at(2024, 1, 8), _at(2024, 1, 15)]
    assert frequency.grid(monday, 2, "1w") == weeks
    assert frequency.grid(10, 2, "1d") == [86410, 172810]
    assert frequency.grid(10, 2, 5) == [15, 20]
    months = [_at(2024, m, 1) for m in (1, 2, 3)]
    assert frequency.regular(months, "MS")
    assert not frequency.regular(months, "1w")
    assert not frequency.regular(months, "ME")


def test_manifests_keep_the_canonical_spelling() -> None:
    def spec(value: object) -> object:
        text = (
            "model: general-forecast/chronos-2\ntarget: y\ntime: t\nhorizon: 3\n"
            f"frequency: {value}\n"
        )
        return parse_text(text, default_name="sales").frequency

    assert spec("W-MON") == "W-MON"
    assert spec("H") == "1h"
    assert spec("15min") == "15min"
    assert spec("MS") == "MS"
    assert spec("7d") == "1w"
    assert spec(5) == 5
