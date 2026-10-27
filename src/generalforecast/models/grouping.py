# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Forecast times as `gf_core::request` takes them, shared by every forecast surface.

`gf_core::request` aligns a query's related series and covariates into its
group; predictors read their context's times as integer periods or UTC seconds
here first.
"""

import datetime
from typing import Any


def instant(value: Any) -> int | float:
    """A time as integer periods, or as floating UTC seconds for a date or datetime.

    A datetime without a zone is read as UTC, as
    `generalforecast.predictors.frequency.instant` reads it, never as the
    machine's local time.
    """
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.UTC)
        return float(value.timestamp())
    if isinstance(value, datetime.date):
        return float((value - datetime.date(1970, 1, 1)).days * 86400)
    return value
