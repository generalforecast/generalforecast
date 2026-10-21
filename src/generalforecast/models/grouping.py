# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Forecast times as `gf_core::request` takes them, shared by every forecast surface.

`gf_core::request` aligns a query's related series and covariates into its
group; packages read their context's times as integer periods or UTC seconds
here first.
"""

import datetime
from typing import Any


def instant(value: Any) -> int | float:
    """A time as integer periods, or as floating UTC seconds for a date or datetime."""
    if isinstance(value, datetime.datetime):
        return float(value.timestamp())
    if isinstance(value, datetime.date):
        return float((value - datetime.date(1970, 1, 1)).days * 86400)
    return value
