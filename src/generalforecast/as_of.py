# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`as_of` on the wire: the one parser and formatter every Python surface uses.

A request's `as_of` is an RFC 3339 date-time, `2026-10-27T08:00:00Z`, read
as UTC when it names no offset, or a number of seconds since the epoch, at
least 0. A date alone, an hour alone, an ISO week or a negative number is
refused as a `ValidationError` naming the parameter `as_of`. Python callers
may also pass a `datetime`, `date` or pandas `Timestamp`.

Instants are kept to the microsecond: digits past it are dropped, never
rounded up, so an `as_of` never reads a write after the time it names; a
number is read from its shortest decimal spelling, as JSON carries it. A
reply's `as_of` is RFC 3339 UTC, whole seconds or six fractional digits,
or null. gf-server's `as_of.rs` reads and writes by the same rules, and the
fixture `crates/gf-server/tests/fixtures/as_of.json` pins both.
"""

from __future__ import annotations

import datetime
import decimal
import math
from typing import Any

from generalforecast.service.contracts import InputError

# The parameter an invalid time is reported against.
PARAM = "as_of"
# The latest instant accepted, in microseconds: the end of year 9999.
_LAST = 253_402_300_799_999_999
_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)


def _refused(message: str) -> InputError:
    return InputError("ValidationError", message, 422, param=PARAM)


def micros(value: Any) -> int | None:
    """A request's `as_of` as whole UTC microseconds since the epoch, or None when absent.

    Raises:
        InputError: `ValidationError` with param `as_of` for anything the
            rules above refuse.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise _refused("as_of is an RFC 3339 time or epoch seconds, not a boolean")
    if isinstance(value, (int, float)):
        return _seconds_micros(value)
    if isinstance(value, str):
        from generalforecast.iso import instant

        found = instant(value)
        if found is None:
            raise _refused(
                "as_of is an RFC 3339 time such as 2026-10-27T08:00:00Z, "
                "or epoch seconds"
            )
        return _checked(found)
    if isinstance(value, datetime.datetime):
        return _checked(_moment_micros(value))
    if isinstance(value, datetime.date):
        return _checked(
            _moment_micros(
                datetime.datetime(
                    value.year, value.month, value.day, tzinfo=datetime.UTC
                )
            )
        )
    to_datetime = getattr(value, "to_pydatetime", None)
    if callable(to_datetime):
        return _checked(_moment_micros(to_datetime()))
    raise _refused(
        f"as_of is an RFC 3339 time or epoch seconds, not {type(value).__name__}"
    )


def _seconds_micros(seconds: float) -> int:
    if not math.isfinite(seconds):
        raise _refused("as_of seconds must be a finite time after the epoch")
    # The shortest decimal spelling, as JSON carries the number, cut to the
    # microsecond: never a later instant than the one given.
    exact = decimal.Decimal(repr(seconds)) * 1_000_000
    return _checked(int(exact.to_integral_value(rounding=decimal.ROUND_FLOOR)))


def _moment_micros(moment: datetime.datetime) -> int:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.UTC)
    return (moment - _EPOCH) // datetime.timedelta(microseconds=1)


def _checked(value: int) -> int:
    if not 0 <= value <= _LAST:
        raise _refused("as_of must be a time after the epoch")
    return value


def parse(value: Any) -> float | None:
    """A request's `as_of` as UTC seconds since the epoch, or None when absent.

    The seconds are `micros(value) / 1e6`, which `format` writes back to the
    same microsecond.

    Raises:
        InputError: `ValidationError` with param `as_of`, as `micros` does.
    """
    found = micros(value)
    return None if found is None else found / 1_000_000


def format(seconds: float | None) -> str | None:
    """UTC seconds as a reply's RFC 3339 `as_of`, `2026-10-27T08:00:00Z`; None stays None.

    Fractions of a second are written to the microsecond, six digits, cut
    as a request's seconds are read: never rounded up to a later instant.
    """
    if seconds is None:
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise _refused("as_of seconds must be a number")
    return _written(_seconds_micros(seconds))


def _written(whole: int) -> str:
    moment = _EPOCH + datetime.timedelta(microseconds=whole)
    text = moment.isoformat(
        timespec="microseconds" if moment.microsecond else "seconds"
    )
    return text.removesuffix("+00:00") + "Z"


def wire(value: Any) -> str | None:
    """Any accepted `as_of` as the RFC 3339 UTC string a request sends."""
    found = micros(value)
    return None if found is None else _written(found)
