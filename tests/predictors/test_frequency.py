# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Frequencies: the spellings taken and the one kept."""

import datetime

import pytest

from generalforecast.predictors import frequency
from generalforecast.predictors.spec import parse_text


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
    frequency.check({"a": months}, "MS")
    for other in ("1w", "ME"):
        with pytest.raises(frequency.FrequencyError):
            frequency.check({"a": months}, other)


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


def test_calendar_cadences_are_inferred_as_calendar_aliases() -> None:
    import calendar

    months = [datetime.date(2023, m, 1) for m in range(1, 13)]
    assert frequency.infer({"a": months}) == "MS"
    ends = [
        datetime.date(2023, m, calendar.monthrange(2023, m)[1]) for m in range(1, 13)
    ]
    assert frequency.infer({"a": ends}) == "ME"
    quarters = [datetime.date(2020 + q // 4, 1 + 3 * (q % 4), 1) for q in range(9)]
    assert frequency.infer({"a": quarters}) == "QS"
    years = [datetime.date(2000 + i, 1, 1) for i in range(6)]
    assert frequency.infer({"a": years}) == "YS"
    every_other = [datetime.date(2023, m, 1) for m in range(1, 13, 2)]
    assert frequency.infer({"a": every_other}) == "2MS"
    assert frequency.grid(datetime.date(2023, 12, 1), 3, "MS") == [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 2, 1),
        datetime.date(2024, 3, 1),
    ]


def test_one_rule_refuses_gaps_duplicates_and_mixed_steps() -> None:
    day = datetime.date(2024, 1, 1)
    gappy = [day + datetime.timedelta(days=d) for d in (0, 1, 2, 3, 10)]
    with pytest.raises(frequency.FrequencyError) as caught:
        frequency.infer({"a": gappy})
    assert caught.value.code == "AmbiguousFrequency"
    assert caught.value.because == (
        "series a at 2024-01-11 breaks the step the times before it follow"
    )
    with pytest.raises(frequency.FrequencyError) as caught:
        frequency.infer({"a": [day, day, day + datetime.timedelta(days=1)]})
    assert caught.value.code == "DuplicateTime"
    with pytest.raises(frequency.FrequencyError) as caught:
        frequency.check({"a": gappy}, "1d")
    assert caught.value.code == "FrequencyMismatch"
    with pytest.raises(frequency.FrequencyError) as caught:
        frequency.check({"a": [1, 2, 3]}, "1d")
    assert caught.value.code == "FrequencyMismatch"
    # Each series follows the step on its own: two daily series half a day apart.
    north = [_at(2024, 1, d) for d in range(1, 9)]
    south = [stamp + datetime.timedelta(hours=12) for stamp in north]
    assert frequency.infer({"north": north, "south": south}) == "1d"
    frequency.check({"north": north[:1]}, "1d")


def test_times_are_read_in_utc_on_every_machine(monkeypatch) -> None:
    import os
    import time

    days = [_at(2024, 3, 1) + datetime.timedelta(days=d) for d in range(20)]
    months = [_at(2024, m, 1) for m in range(1, 8)]
    for zone in ("UTC", "America/Los_Angeles", "Asia/Tokyo"):
        monkeypatch.setenv("TZ", zone)
        if hasattr(time, "tzset"):
            time.tzset()
        assert frequency.infer({"a": days}) == "1d"
        assert frequency.infer({"a": months}) == "MS"
        assert frequency.instant(_at(2024, 1, 1)) == 1704067200.0
        assert os.environ["TZ"] == zone
    monkeypatch.delenv("TZ")
    if hasattr(time, "tzset"):
        time.tzset()


def test_calendar_times_step_through_period_ordinals() -> None:
    months = [datetime.date(2023, m, 1) for m in range(1, 13)]
    places = [frequency.coordinate(month, "MS") for month in months]
    first = int(places[0])
    assert places == list(range(first, first + 12))
    assert frequency.step("MS") == 1 and frequency.step("3MS") == 3
    assert frequency.step("1d") == 86400 and frequency.step(7) == 7
    assert frequency.calendar_unit("2QE") == "Q"
    assert frequency.calendar_unit("W-MON") == "W-MON"
    assert frequency.calendar_unit("1d") is None
    assert frequency.ordinal(datetime.date(1970, 1, 5), "W-MON") == 0
    aware = datetime.datetime(
        2024, 2, 1, 9, tzinfo=datetime.timezone(datetime.timedelta(hours=9))
    )
    assert frequency.ordinal(aware, "MS") == frequency.ordinal(
        datetime.date(2024, 2, 1), "MS"
    )
