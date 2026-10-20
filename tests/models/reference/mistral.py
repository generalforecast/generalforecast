"""Timestral from Mistral's package, and generalforecast's port, on the same weights."""

from __future__ import annotations

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from generalforecast.models.series import KNOWN, PAST, TARGET, SeriesGroup
from generalforecast.models.timestral.checkpoint import (
    PUBLISHED,
    load_timestral,
    timestral_checkpoint,
)
from generalforecast.models.timestral.model import timestral_functions

# The smaller release, with the larger one's layers and settings.
NAME = "timestral-28m"
# Two targets, a past-only covariate and a known covariate over 700 steps,
# which is not whole patches; the same rows crossing zero, forecast without
# averaging over signs; or one target that starts after 40 unobserved steps.
CASES = {
    "covariates": ((TARGET, TARGET, PAST, KNOWN), True),
    "plain": ((TARGET, TARGET, PAST, KNOWN), False),
    "univariate": ((TARGET,), True),
}
STEPS = 700
# Past one block of 32 patches, so the second block reads fed-back medians.
LONG_HORIZON = 1100


@functools.cache
def _series(case: str) -> tuple[np.ndarray, np.ndarray]:
    """Context rows with a few missing values, and the last row's future values."""
    generator = np.random.default_rng(13)
    roles, symmetric = CASES[case]
    rows = len(roles)
    trend = np.linspace(0, 3, STEPS + LONG_HORIZON)
    season = np.sin(np.arange(STEPS + LONG_HORIZON) * 2 * np.pi / 24)
    noise = generator.normal(size=(rows, STEPS + LONG_HORIZON))
    whole = 10 * (1 + np.arange(rows))[:, None] * (1 + 0.2 * season + 0.1 * trend)
    whole = (whole + noise).astype(np.float32)
    if not symmetric:
        # Crossing zero, so no row is clipped at it and the sign matters.
        whole -= whole.mean(axis=1, keepdims=True)
    context = whole[:, :STEPS].copy()
    context[0, 100:103] = np.nan
    if case == "univariate":
        context[0, :40] = np.nan
    else:
        context[2, 5] = np.nan
    return context, whole[-1, STEPS:]


@functools.cache
def _upstream() -> Any:
    from timestral import TimestralForecaster

    published = PUBLISHED[NAME]
    return TimestralForecaster.from_pretrained(
        published.repository, revision=published.revision
    )


def upstream_forecast(case: str, horizon: int) -> np.ndarray:
    """Mistral's forecaster on a group, `(targets, horizon, quantiles)`."""
    roles, symmetric = CASES[case]
    context, future = _series(case)
    targets = roles.count(TARGET)
    covariates: dict[str, Any] = {}
    if PAST in roles:
        covariates["past_covariates"] = torch.from_numpy(context[2:3])[None]
        known = np.concatenate([context[3], future[:horizon]])
        covariates["known_covariates"] = torch.from_numpy(known)[None, None]
    forecast = _upstream().forecast(
        torch.from_numpy(context[:targets])[None],
        horizon,
        symmetric_averaging=symmetric,
        clip_nonnegative=symmetric,
        **covariates,
    )
    return forecast.quantiles[0].numpy()


def our_forecast(case: str, horizon: int) -> np.ndarray:
    """generalforecast's port on the same group, converted from the same files."""
    roles, symmetric = CASES[case]
    context, future = _series(case)
    model = load_timestral(timestral_checkpoint(NAME), symmetric)
    rows = len(roles)
    pad = -STEPS % model.patch
    values = np.concatenate([np.zeros((rows, pad), np.float32), context], axis=1)
    observed = np.isfinite(values)
    observed[:, :pad] = False
    known = None
    if KNOWN in roles:
        known = np.full((rows, horizon), np.nan, np.float32)
        known[-1] = future[:horizon]
    group = SeriesGroup(
        jnp.asarray(np.nan_to_num(values)),
        jnp.asarray(observed),
        jnp.asarray(roles, jnp.int32),
        None if known is None else jnp.asarray(known),
    )
    predictions = timestral_functions.forecast(model, group, horizon)
    return np.asarray(predictions[: roles.count(TARGET)])
