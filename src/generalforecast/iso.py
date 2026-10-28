# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISO 8601 times on the UTC axis: the one reader every Python path uses.

Two grammars, as gf-core's `civil` module reads them in Rust:

- `timestamps`, a context or query column of ISO text, read as Arrow casts
  it: a date, or a date and a time, naive or with an offset, to naive UTC
  microseconds. The fixture `crates/gf-server/tests/fixtures/iso_times.json`
  pins it for both languages.
- `instant`, an `as_of`: an RFC 3339 date-time, naive meaning UTC, to whole
  UTC microseconds, digits past the microsecond dropped and never rounded
  up. The fixture `crates/gf-server/tests/fixtures/as_of.json` pins it.

Text with an offset is converted to UTC; text without one is UTC.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

import pyarrow as pa
import pyarrow.compute as _compute

pc: Any = _compute  # ty cannot see Arrow's generated compute functions

# An RFC 3339 date-time: date, separator, time to the second, an optional
# fraction of one to nine digits, and an optional offset.
_RFC3339 = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?"
    r"(?:([Zz])|([+-])(\d{2}):(\d{2}))?"
)
_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)


def timestamps(values: Any) -> Any:
    """A column of ISO text as naive UTC timestamps; text with an offset is converted to UTC.

    Raises:
        pyarrow.ArrowInvalid: For text that is not all times, or that mixes
            naive times with zoned ones.
    """
    try:
        return pc.cast(values, pa.timestamp("us"))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        zoned = pc.cast(values, pa.timestamp("us", tz="UTC"))
        return pc.cast(zoned, pa.timestamp("us"))


def instant(text: str) -> int | None:
    """An RFC 3339 date-time as whole UTC microseconds since 1970; None for other text."""
    match = _RFC3339.fullmatch(text)
    if match is None:
        return None
    year, month, day, hour, minute, second = map(int, match.groups()[:6])
    fraction, _zulu, sign, offset_hours, offset_minutes = match.groups()[6:]
    try:
        moment = datetime.datetime(
            year, month, day, hour, minute, second, tzinfo=datetime.UTC
        )
    except ValueError:
        return None
    offset = 0
    if sign is not None:
        if int(offset_hours) >= 24 or int(offset_minutes) >= 60:
            return None
        offset = (int(offset_hours) * 60 + int(offset_minutes)) * 60_000_000
        offset = offset if sign == "+" else -offset
    whole = (moment - _EPOCH) // datetime.timedelta(microseconds=1)
    # Past the microsecond, digits are dropped: never later than the text says.
    return whole + int((fraction or "0")[:6].ljust(6, "0")) - offset


__all__ = ["instant", "timestamps"]
