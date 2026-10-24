# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A forecaster's frequency: the spellings it takes, and the one it keeps.

A frequency is a number of periods (or seconds) or a string. Strings are a
span, `1d`, `1h`, `15min`, `1w`, or one of pandas' offset aliases, `H`, `D`,
`W`, `15min`, `W-MON`, `MS`, `ME`, `QS`, `QE`, `YS`, `YE`. Every spelling
normalizes to one: a span in the largest whole unit (`H` is `1h`, `7d` is
`1w`), or the calendar alias as pandas spells it (`W-MON`, `MS`). The lock
and the CLI show the normalized spelling.
"""

import datetime
import math
import re
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
    if anchored.startswith("W-") and anchored[2:] in (
        "MON",
        "TUE",
        "WED",
        "THU",
        "FRI",
        "SAT",
        "SUN",
    ):
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
    anchored week, 30.436875 days for a month.
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


def grid(origin: Any, count: int, value: Any) -> list[Any]:
    """The `count` times after `origin`, one frequency step apart.

    Args:
        origin: An integer period, a date or a datetime.
        count: How many steps.
        value: The frequency, in any spelling.
    """
    spelled = normalize(value)
    if isinstance(origin, int | float) and not isinstance(origin, bool):
        step = seconds(spelled)
        assert step is not None
        whole = int(step) if step.is_integer() else step
        return [origin + whole * (index + 1) for index in range(count)]
    if is_calendar(spelled):
        import pandas as pd

        offset = pd.tseries.frequencies.to_offset(str(spelled))
        stamps = [pd.Timestamp(origin) + offset * (index + 1) for index in range(count)]
        if isinstance(origin, datetime.datetime):
            return [stamp.to_pydatetime() for stamp in stamps]
        return [stamp.date() for stamp in stamps]
    step = seconds(spelled)
    assert step is not None
    return [
        origin + datetime.timedelta(seconds=step * (index + 1))
        for index in range(count)
    ]


def regular(times: list[Any], value: Any) -> bool:
    """Whether sorted `times` sit one frequency step apart, from the first on."""
    if len(times) < 2:
        return True
    return grid(times[0], len(times) - 1, value) == list(times[1:])


def ordinal(value: Any, alias: Any) -> int:
    """Which calendar period `value` falls in, counted from the epoch.

    Consecutive periods of a calendar alias (`MS`, `W-MON`) are consecutive
    integers, so a model stepping by one sees evenly spaced times.
    """
    import pandas as pd

    spelled = normalize(alias)
    match = CALENDAR.fullmatch(spelled) if isinstance(spelled, str) else None
    if match is None:
        raise ValueError(f"`{alias}` is not a calendar frequency")
    unit = match.group(2)
    period = unit if unit.startswith("W-") else unit[0]
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    counted = pd.Period(stamp, freq=period)
    assert isinstance(counted, pd.Period)
    return int(counted.ordinal)
