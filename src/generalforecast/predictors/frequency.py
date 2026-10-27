# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A forecaster's frequency: its spellings, the one it keeps, and the one rule for times.

A frequency is a number of periods (or seconds) or a string. Strings are a
span, `1d`, `1h`, `15min`, `1w`, or one of pandas' offset aliases, `H`, `D`,
`W`, `15min`, `W-MON`, `MS`, `ME`, `QS`, `QE`, `YS`, `YE`. Every spelling
normalizes to one: a span in the largest whole unit (`H` is `1h`, `7d` is
`1w`), or the calendar alias as pandas spells it (`W-MON`, `MS`). The lock
and the CLI show the normalized spelling.

Times are read on UTC alone. A date is its midnight in UTC, a datetime
without a zone is read as UTC, and one with a zone is converted to UTC, so
the same rows give the same instants on every machine. Calendar frequencies
step through UTC calendar periods: a month start is the first day of a UTC
month.

This module holds the one rule every path applies to a forecaster's times,
`gf check`, the predictor runtime, `gf serve` (whose Rust binder mirrors
`coordinate` and `step`) and hosted serving alike: within each series times
are distinct and sit exactly one step apart, with no gaps. `infer` finds the
step, preferring a fixed span and else the calendar alias the times follow;
`check` holds times to a declared frequency.
"""

import datetime
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

# A normalized span: a count and a unit, `15min`.
SPAN = re.compile(r"[1-9][0-9]*(ms|s|min|h|d|w)")
# A normalized calendar alias: anchored weeks, month, quarter and year starts and ends.
CALENDAR = re.compile(
    r"([1-9][0-9]*)?(W-(?:MON|TUE|WED|THU|FRI|SAT|SUN)|MS|ME|QS|QE|YS|YE)"
)
_WRITTEN = re.compile(r"\s*([1-9][0-9]*)?\s*([A-Za-z][A-Za-z\-]*)\s*")

_UNITS = {"ms": 0.001, "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}
# Spellings of the span units, pandas' among them; `m` is a minute, `M` a month.
_SPAN_ALIASES = {
    "ms": "ms", "L": "ms", "l": "ms",
    "s": "s", "S": "s", "sec": "s",
    "min": "min", "T": "min", "t": "min", "m": "min",
    "h": "h", "H": "h", "hr": "h",
    "d": "d", "D": "d",
    "w": "w", "W": "w",
}  # fmt: skip
_CALENDAR_ALIASES = {
    "MS": "MS", "M": "ME", "ME": "ME",
    "QS": "QS", "Q": "QE", "QE": "QE",
    "YS": "YS", "AS": "YS", "Y": "YE", "A": "YE", "YE": "YE",
}  # fmt: skip
# The mean length of each calendar step, for capacity and display.
_CALENDAR_SECONDS = {"W": 604800.0, "M": 2629746.0, "Q": 7889238.0, "Y": 31556952.0}
WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_EPOCH = datetime.date(1970, 1, 1)
_DAY = 86400


class FrequencyError(ValueError):
    """Times that break the one rule, with a stable code and the reason.

    Attributes:
        code: `AmbiguousFrequency` for times that follow no single step,
            `DuplicateTime` for a time repeated within a series, `TooFewTimes`
            when no series has two times and no frequency is declared, and
            `FrequencyMismatch` for times that differ from a declared one.
        message: What is wrong.
        because: Where the times break the rule, if known.
    """

    def __init__(self, code: str, message: str, because: str | None = None):
        """Keeps the code apart from the message."""
        self.code = code
        self.message = message
        self.because = because
        super().__init__(message)


def normalize(value: Any) -> str | int | float:
    """The one spelling of a frequency.

    Args:
        value: A positive number of periods or seconds, a span such as `1d`
            or `15min`, or a pandas offset alias such as `H`, `W-MON` or `MS`.

    Returns:
        The number as given, or the normalized string.

    Raises:
        ValueError: If the value spells no frequency.
        TypeError: If the value is not a number or a string.
    """
    if isinstance(value, bool):
        raise TypeError("a frequency is not a boolean")
    if isinstance(value, int | float):
        if not (math.isfinite(value) and value > 0):
            raise ValueError("a frequency is a positive number")
        return value
    if isinstance(value, datetime.timedelta):
        return _span(value.total_seconds())
    if not isinstance(value, str):
        raise TypeError("a frequency is a number or a string")
    match = _WRITTEN.fullmatch(value)
    if match is None:
        raise ValueError(f"`{value}` is not a frequency")
    count, unit = int(match.group(1) or 1), match.group(2)
    if unit in _SPAN_ALIASES:
        return _span(count * _UNITS[_SPAN_ALIASES[unit]])
    anchored = unit.upper()
    if anchored.startswith("W-") and anchored[2:] in WEEKDAYS:
        alias = anchored
    elif unit in _CALENDAR_ALIASES:
        alias = _CALENDAR_ALIASES[unit]
    else:
        raise ValueError(f"`{value}` is not a frequency")
    return alias if count == 1 else f"{count}{alias}"


def _span(total: float) -> str:
    for unit in ("w", "d", "h", "min", "s", "ms"):
        size = _UNITS[unit]
        steps = total / size
        if steps >= 1 and abs(steps - round(steps)) < 1e-9:
            return f"{round(steps)}{unit}"
    raise ValueError(f"{total} seconds is not a whole number of milliseconds")


def seconds(value: Any) -> float | None:
    """A frequency's step in seconds, or periods for a number; None when none.

    A calendar alias gives the mean length of its step: 7 days for an
    anchored week, 30.436875 days for a month. Only capacity and display read
    that mean; times step by `step` and `coordinate`.
    """
    if value is None:
        return None
    spelled = normalize(value)
    if isinstance(spelled, int | float):
        return float(spelled)
    span = SPAN.fullmatch(spelled)
    if span is not None:
        return int(spelled[: -len(span.group(1))]) * _UNITS[span.group(1)]
    calendar = CALENDAR.fullmatch(spelled)
    assert calendar is not None
    return int(calendar.group(1) or 1) * _CALENDAR_SECONDS[calendar.group(2)[0]]


def is_calendar(value: Any) -> bool:
    """Whether the frequency steps by the calendar, so its steps differ in length."""
    spelled = normalize(value)
    return isinstance(spelled, str) and CALENDAR.fullmatch(spelled) is not None


def calendar_unit(value: Any) -> str | None:
    """The period a calendar frequency counts: `M`, `Q`, `Y` or `W-<DAY>`; None for a span.

    This is what `gf serve`'s binder reads to place times on periods; a
    month-start and a month-end alias both count months.
    """
    spelled = normalize(value)
    match = CALENDAR.fullmatch(spelled) if isinstance(spelled, str) else None
    if match is None:
        return None
    alias = match.group(2)
    return alias if alias.startswith("W-") else alias[0]


def instant(value: Any) -> int | float:
    """A time on the UTC axis: integer periods as given, dates and datetimes as UTC seconds.

    A datetime without a zone is read as UTC, never as the machine's local
    time, so every machine reads the same rows the same way.

    Raises:
        TypeError: For a value that is no number, date or datetime.
    """
    if isinstance(value, bool):
        raise TypeError("a time is not a boolean")
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.UTC)
        return value.timestamp()
    if isinstance(value, datetime.date):
        return float((value - _EPOCH).days * _DAY)
    if isinstance(value, int | float):
        return value
    raise TypeError("forecast time must be numeric, date or timestamp")


def _utc(value: Any) -> datetime.datetime:
    """A date or datetime as a naive UTC datetime."""
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.UTC).replace(tzinfo=None)
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day)  # noqa: DTZ001
    raise TypeError("a calendar frequency needs dates or timestamps")


def ordinal(value: Any, alias: Any) -> int:
    """Which calendar period `value` falls in, counted from 1970 in UTC.

    Consecutive periods of a calendar alias (`MS`, `W-MON`) are consecutive
    integers, so a model stepping by one sees evenly spaced times. Weeks
    anchored on a day count from that day's first occurrence on or after
    1970-01-01. `gf serve`'s binder computes the same numbers.
    """
    unit = calendar_unit(alias)
    if unit is None:
        raise ValueError(f"`{alias}` is not a calendar frequency")
    stamp = _utc(value)
    if unit == "M":
        return (stamp.year - 1970) * 12 + stamp.month - 1
    if unit == "Q":
        return (stamp.year - 1970) * 4 + (stamp.month - 1) // 3
    if unit == "Y":
        return stamp.year - 1970
    days = (stamp.date() - _EPOCH).days
    anchor = (WEEKDAYS.index(unit[2:]) - _EPOCH.weekday()) % 7
    return (days - anchor) // 7


def step(value: Any) -> int | float:
    """The distance between consecutive times on the axis `coordinate` places them on.

    A number of periods or seconds as given; a span's seconds; a calendar
    alias's count of periods (`3MS` is 3).
    """
    spelled = normalize(value)
    if isinstance(spelled, int | float):
        return spelled
    calendar = CALENDAR.fullmatch(spelled)
    if calendar is not None:
        return int(calendar.group(1) or 1)
    total = seconds(spelled)
    assert total is not None
    return int(total) if float(total).is_integer() else total


def coordinate(value: Any, frequency: Any) -> int | float:
    """A time on its frequency's axis: a calendar period's ordinal, else its UTC instant."""
    if frequency is not None and is_calendar(frequency):
        return ordinal(value, frequency)
    return instant(value)


def _anchored(stamp: datetime.datetime, alias: str) -> bool:
    """Whether a UTC time is on one of a calendar alias's anchors."""
    if alias.startswith("W-"):
        return WEEKDAYS[stamp.weekday()] == alias[2:]
    starts = stamp.day == 1
    ends = (stamp + datetime.timedelta(days=1)).day == 1
    if alias == "MS":
        return starts
    if alias == "ME":
        return ends
    if alias == "QS":
        return starts and stamp.month in (1, 4, 7, 10)
    if alias == "QE":
        return ends and stamp.month in (3, 6, 9, 12)
    if alias == "YS":
        return starts and stamp.month == 1
    return ends and stamp.month == 12


def _sorted(series: Mapping[Any, Sequence[Any]]) -> dict[Any, list[Any]]:
    """Each series' times in order, refusing a repeated time."""
    ordered: dict[Any, list[Any]] = {}
    for key, times in series.items():
        values = list(times)
        if any(value is None for value in values):
            raise FrequencyError(
                "MissingTime", "forecast time and series cannot be missing"
            )
        try:
            values.sort(key=instant)
        except TypeError:
            raise FrequencyError(
                "InvalidTime", "forecast time must be numeric, date or timestamp"
            ) from None
        for left, right in pairwise(values):
            if instant(left) == instant(right):
                raise FrequencyError(
                    "DuplicateTime",
                    "duplicate forecast time within a series",
                    _where(key, right),
                )
        ordered[key] = values
    return ordered


def _where(key: Any, value: Any) -> str:
    label = "the series" if key in ((), None) else f"series {_label(key)}"
    shown = value.isoformat() if hasattr(value, "isoformat") else value
    return f"{label} at {shown}"


def _label(key: Any) -> str:
    if isinstance(key, tuple):
        return ", ".join(str(part) for part in key) if key else "(all rows)"
    return str(key)


def _numeric(times: Iterable[Any]) -> bool:
    return all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in times
    )


def infer(series: Mapping[Any, Sequence[Any]]) -> str | int | float:
    """The one frequency every series follows, in its normalized spelling.

    Args:
        series: Each series' times, by series key: integer periods, dates or
            datetimes.

    Returns:
        A number of periods for integer times; else a span such as `1d` when
        every step has one length, or the calendar alias the times follow,
        such as `MS` for month starts or `2QE` for every other quarter end.

    Raises:
        FrequencyError: `TooFewTimes`, `DuplicateTime` or `AmbiguousFrequency`.
    """
    every_series = _sorted(series)
    # A series of one time follows whatever step the others do.
    ordered = {key: times for key, times in every_series.items() if len(times) > 1}
    if not ordered:
        raise FrequencyError(
            "TooFewTimes",
            "no forecast series has two times to infer its frequency from",
        )
    found = _inferred(ordered)
    check(every_series, found)
    return found


def _inferred(ordered: Mapping[Any, list[Any]]) -> str | int | float:
    """`infer` over series of two or more times each."""
    every = [value for times in ordered.values() for value in times]
    if _numeric(every):
        gaps = {
            right - left
            for times in ordered.values()
            for left, right in pairwise(times)
        }
        if len(gaps) == 1:
            gap = gaps.pop()
            return int(gap) if float(gap).is_integer() else gap
        raise _irregular(ordered)
    lengths = {
        instant(right) - instant(left)
        for times in ordered.values()
        for left, right in pairwise(times)
    }
    # Steps of four weeks or more may be calendar periods even when the
    # times seen happen to sit equally far apart (three Jan-1s, two month
    # starts), so the calendar is tried first; shorter steps are spans.
    longest = max(lengths)
    if longest >= 28 * _DAY:
        for alias, months in (
            ("YS", 12),
            ("YE", 12),
            ("QS", 3),
            ("QE", 3),
            ("MS", 1),
            ("ME", 1),
        ):
            count = _calendar_count(ordered, alias, months)
            if count is not None:
                return alias if count == 1 else f"{count}{alias}"
    if len(lengths) == 1:
        length = lengths.pop()
        try:
            return _span(length)
        except ValueError:
            return length
    raise _irregular(ordered)


def _calendar_count(
    ordered: Mapping[Any, list[Any]], alias: str, months: int
) -> int | None:
    """How many `alias` periods apart the times sit, if they all follow it."""
    counts = set()
    for times in ordered.values():
        stamps = [_utc(value) for value in times]
        if len({stamp.time() for stamp in stamps}) != 1:
            return None
        if not all(_anchored(stamp, alias) for stamp in stamps):
            return None
        index = [stamp.year * 12 + stamp.month - 1 for stamp in stamps]
        for left, right in pairwise(index):
            if (right - left) % months:
                return None
            counts.add((right - left) // months)
    return counts.pop() if len(counts) == 1 else None


def check(series: Mapping[Any, Sequence[Any]], declared: Any) -> None:
    """Holds every series to a declared frequency: one step apart, no gaps.

    A series of one time follows any frequency.

    Raises:
        FrequencyError: `DuplicateTime`, or `FrequencyMismatch` naming the
            first time off the declared step.
    """
    spelled = normalize(declared)
    ordered = _sorted(series)
    for key, times in ordered.items():
        if len(times) < 2:
            continue
        if _numeric(times) and isinstance(spelled, str):
            raise FrequencyError(
                "FrequencyMismatch",
                "integer time periods step by a number of periods",
                f"frequency `{spelled}` is a duration; write `frequency: 1`",
            )
        if is_calendar(spelled):
            alias = CALENDAR.fullmatch(str(spelled))
            assert alias is not None
            stamps = [_utc(value) for value in times]
            first = stamps[0].time()
            for stamp, value in zip(stamps, times, strict=True):
                if stamp.time() != first or not _anchored(stamp, alias.group(2)):
                    raise _mismatch(key, value, spelled)
        width = step(spelled)
        places = [coordinate(value, spelled) for value in times]
        for (left, right), value in zip(pairwise(places), times[1:], strict=True):
            if right - left != width:
                raise _mismatch(key, value, spelled)


def _mismatch(key: Any, value: Any, spelled: Any) -> FrequencyError:
    return FrequencyError(
        "FrequencyMismatch",
        "declared frequency differs from observed times",
        f"{_where(key, value)} is not one `{spelled}` step after the time before it",
    )


def _irregular(ordered: Mapping[Any, list[Any]]) -> FrequencyError:
    """The refusal for times that follow no single step, naming the first odd gap."""
    first: tuple[Any, Any] | None = None
    for key, times in ordered.items():
        gaps = [instant(right) - instant(left) for left, right in pairwise(times)]
        for index in range(1, len(gaps)):
            if gaps[index] != gaps[0]:
                first = (key, times[index + 1])
                break
        if first is not None:
            break
    return FrequencyError(
        "AmbiguousFrequency",
        "forecast frequency is irregular; resample explicitly",
        None
        if first is None
        else f"{_where(*first)} breaks the step the times before it follow",
    )


def grid(origin: Any, count: int, value: Any) -> list[Any]:
    """The `count` times after `origin`, one frequency step apart.

    Args:
        origin: An integer period, a date or a datetime.
        count: How many steps.
        value: The frequency, in any spelling.
    """
    spelled = normalize(value)
    if isinstance(origin, int | float) and not isinstance(origin, bool):
        width = step(spelled)
        whole = int(width) if float(width).is_integer() else width
        return [origin + whole * (index + 1) for index in range(count)]
    if is_calendar(spelled):
        import pandas as pd

        offset = pd.tseries.frequencies.to_offset(str(spelled))
        start = pd.Timestamp(_utc(origin))
        stamps = [start + offset * (index + 1) for index in range(count)]
        if isinstance(origin, datetime.datetime):
            # Calendar periods are UTC periods; an origin with a zone gets
            # UTC times back.
            zone = datetime.UTC if origin.tzinfo is not None else None
            return [stamp.to_pydatetime().replace(tzinfo=zone) for stamp in stamps]
        return [stamp.date() for stamp in stamps]
    total = seconds(spelled)
    assert total is not None
    if isinstance(origin, datetime.datetime) and origin.tzinfo is not None:
        # Spans step through UTC, never a zone's wall clock, which jumps an
        # hour at a daylight saving change.
        origin = origin.astimezone(datetime.UTC)
    return [
        origin + datetime.timedelta(seconds=total * (index + 1))
        for index in range(count)
    ]


__all__ = [
    "CALENDAR",
    "SPAN",
    "FrequencyError",
    "calendar_unit",
    "check",
    "coordinate",
    "grid",
    "infer",
    "instant",
    "is_calendar",
    "normalize",
    "ordinal",
    "seconds",
    "step",
]
