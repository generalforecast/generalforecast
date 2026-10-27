# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`as_of` on the wire: the one parser and formatter every surface uses.

A request's `as_of` is an RFC 3339 time, read as UTC when it names no
offset, or a number of seconds since the epoch. A reply's `as_of` is always
an RFC 3339 UTC string such as `2026-10-27T08:00:00Z`, or null. Anything
else is a `ValidationError` naming the parameter `as_of`. gf serve, the
hosting API and the SDK parse and format through these two functions, and
gf-server follows the same rules.
"""

from __future__ import annotations

import datetime
import math
from typing import Any

from generalforecast.service.contracts import InputError

# The parameter an invalid time is reported against.
PARAM = "as_of"
# The latest instant accepted, in seconds: the last second of year 9999.
_LAST = 253402300799.0


def _refused(message: str) -> InputError:
    return InputError("ValidationError", message, 422, param=PARAM)


def parse(value: Any) -> float | None:
    """A request's `as_of` as UTC seconds since the epoch, or None when absent.

    Accepts an RFC 3339 string (no offset means UTC), a number of seconds,
    or a `datetime`, `date` or pandas `Timestamp`, as Python callers hold
    times.

    Raises:
        InputError: `ValidationError` with param `as_of` for anything else.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise _refused("as_of is an RFC 3339 time or epoch seconds, not a boolean")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds) or not 0 <= seconds <= _LAST:
            raise _refused("as_of seconds must be a finite time after the epoch")
        return seconds
    if isinstance(value, str):
        try:
            moment = datetime.datetime.fromisoformat(value.strip())
        except ValueError:
            raise _refused(
                "as_of is an RFC 3339 time such as 2026-10-27T08:00:00Z, "
                "or epoch seconds"
            ) from None
        return _seconds(moment)
    if isinstance(value, datetime.datetime):
        return _seconds(value)
    if isinstance(value, datetime.date):
        return _seconds(
            datetime.datetime(value.year, value.month, value.day, tzinfo=datetime.UTC)
        )
    to_datetime = getattr(value, "to_pydatetime", None)
    if callable(to_datetime):
        return _seconds(to_datetime())
    raise _refused(
        f"as_of is an RFC 3339 time or epoch seconds, not {type(value).__name__}"
    )


def _seconds(moment: datetime.datetime) -> float:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.UTC)
    seconds = moment.timestamp()
    if not 0 <= seconds <= _LAST:
        raise _refused("as_of must be a time after the epoch")
    return seconds


def format(seconds: float | None) -> str | None:
    """UTC seconds as a reply's RFC 3339 `as_of`, `2026-10-27T08:00:00Z`; None stays None.

    Fractions of a second are kept, to the microsecond.
    """
    checked = parse(seconds)
    if checked is None:
        return None
    moment = datetime.datetime.fromtimestamp(checked, datetime.UTC)
    text = moment.isoformat(
        timespec="microseconds" if moment.microsecond else "seconds"
    )
    return text.removesuffix("+00:00") + "Z"


def wire(value: Any) -> str | None:
    """Any accepted `as_of` as the RFC 3339 UTC string a request sends."""
    return format(parse(value))
