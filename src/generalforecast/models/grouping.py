"""Aligned related series as one `SeriesGroup`, shared by every forecast surface.

A forecast query names its series and the series forecast with it, a time
origin and a step. Each series' observations sit at whole steps before the
origin; named covariates are rows after the targets, past-only or known over
the horizon. The Models API and packages both build their groups here.
"""

import datetime
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from generalforecast.models.series import KNOWN, PAST, TARGET, SeriesGroup


class AlignmentError(ValueError):
    """Related series that do not align into one group."""

    code = "UnsupportedCovariates"


def instant(value: Any) -> int | float:
    """A time as integer periods, or as floating UTC seconds for a date or datetime."""
    if isinstance(value, datetime.datetime):
        return float(value.timestamp())
    if isinstance(value, datetime.date):
        return float((value - datetime.date(1970, 1, 1)).days * 86400)
    return value


def aligned_group(
    series: Mapping[str, Mapping[int | float, Mapping[str, Any]]],
    names: Sequence[str],
    known: Sequence[str],
    origin: float,
    step: float,
    horizon: int,
    future: Mapping[str, Sequence[float]],
    patch: int,
) -> SeriesGroup:
    """Related series and their covariates, aligned on one time axis.

    Args:
        series: Each series' rows by time, each with a `value`, which may be
            None, and `covariates`, a mapping of named values.
        names: The group's target series, in output row order.
        known: Covariates whose future values are known.
        origin: The forecast origin, the last context time.
        step: The time between steps.
        horizon: Steps to forecast.
        future: Each known covariate's values over the horizon.
        patch: The model's input patch; time pads to a multiple of it.

    Raises:
        AlignmentError: If grouped targets disagree on a covariate's value.
    """
    import jax.numpy as jnp

    covariates = sorted(
        set(known)
        | {
            key
            for name in names
            for row in series[name].values()
            for key in row["covariates"]
        }
    )
    offsets = {
        moment: round((origin - moment) / step)
        for name in names
        for moment in series[name]
    }
    steps = (max(offsets.values()) + patch) // patch * patch
    rows = len(names) + len(covariates)
    values = np.zeros((rows, steps), np.float32)
    observed = np.zeros((rows, steps), bool)
    ahead = np.zeros((rows, horizon), np.float32)
    ahead_observed = np.zeros((rows, horizon), bool)
    roles = np.full(rows, TARGET, np.int32)
    for index, name in enumerate(names):
        for moment, row in series[name].items():
            position = steps - 1 - offsets[moment]
            if row["value"] is not None:
                values[index, position] = row["value"]
                observed[index, position] = True
            for covariate, value in row["covariates"].items():
                if value is None:
                    continue
                at = len(names) + covariates.index(covariate)
                if observed[at, position] and float(values[at, position]) != float(
                    np.float32(value)
                ):
                    raise AlignmentError(
                        "grouped targets must share named covariate values at each time"
                    )
                values[at, position] = value
                observed[at, position] = True
    for index, name in enumerate(covariates, start=len(names)):
        roles[index] = KNOWN if name in known else PAST
        if name in known:
            ahead[index] = future[name]
            ahead_observed[index] = True
    return SeriesGroup(
        jnp.asarray(values),
        jnp.asarray(observed),
        jnp.asarray(roles),
        jnp.asarray(ahead),
        jnp.asarray(ahead_observed),
    )
