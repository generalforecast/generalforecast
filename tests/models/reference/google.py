"""TimesFM 3 from Google's package, and generalforecast's port, on the same weights."""

from __future__ import annotations

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from generalforecast.models.series import KNOWN, PAST, TARGET, SeriesGroup
from generalforecast.models.timesfm3.checkpoint import (
    PUBLISHED,
    load_timesfm,
    timesfm_checkpoint,
)
from generalforecast.models.timesfm3.model import timesfm_functions

NAME = "timesfm-3"
# Two targets, a past-only covariate and a known covariate over 700 steps,
# which is not whole patches; or one target that starts after 40 unobserved
# steps, so its first patch is never a key.
CASES = {
    "covariates": (TARGET, TARGET, PAST, KNOWN),
    "univariate": (TARGET,),
}
STEPS = 700
LONG_HORIZON = 1100


@functools.cache
def _series(case: str) -> tuple[np.ndarray, np.ndarray]:
    """Context rows with a few missing values, and the last row's future values."""
    generator = np.random.default_rng(17)
    rows = len(CASES[case])
    trend = np.linspace(0, 3, STEPS + LONG_HORIZON)
    season = np.sin(np.arange(STEPS + LONG_HORIZON) * 2 * np.pi / 24)
    noise = generator.normal(size=(rows, STEPS + LONG_HORIZON))
    whole = 10 * (1 + np.arange(rows))[:, None] * (1 + 0.2 * season + 0.1 * trend)
    whole = (whole + noise).astype(np.float32)
    context = whole[:, :STEPS].copy()
    context[0, 100:103] = np.nan
    if case == "univariate":
        context[0, :40] = np.nan
    else:
        context[2, 5] = np.nan
    return context, whole[-1, STEPS:]


@functools.cache
def _upstream() -> Any:
    from timesfm3.torch import TimesFM3Torch

    published = PUBLISHED[NAME]
    model = TimesFM3Torch.from_pretrained(
        published.repository, revision=published.revision
    )
    return model.eval()


def upstream_forecast(case: str, horizon: int) -> np.ndarray:
    """Google's network on a group, `(targets, horizon, quantiles)`, sorted.

    The network is run directly with masks for the missing values, where its
    forecaster fills them, and the quantiles are sorted as the forecaster
    sorts them by default. Steps before the targets' first observed one are
    dropped, as the forecaster drops them, so positions count from there.
    """
    roles = CASES[case]
    context, future = _series(case)
    targets = roles.count(TARGET)
    first = int(np.argmax(~np.isnan(context[:targets]).all(axis=0)))
    context = context[:, first:]

    def tensor(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.nan_to_num(values))[None], torch.from_numpy(
            np.isnan(values)
        )[None]

    target, target_mask = tensor(context[:targets])
    covariates: dict[str, Any] = {}
    if PAST in roles:
        covariates["past_only_covariates"], covariates["past_only_mask"] = tensor(
            context[2:3]
        )
        known = np.concatenate([context[3], future[:horizon]])[None]
        covariates["past_future_covariates"], covariates["past_future_mask"] = tensor(
            known
        )
    with torch.no_grad():
        outputs = _upstream().decode(
            target, horizon, target_mask=target_mask, **covariates
        )
    return np.sort(outputs[0, :targets].numpy(), axis=-1)


def our_forecast(case: str, horizon: int) -> np.ndarray:
    """generalforecast's port on the same group, converted from the same files."""
    roles = CASES[case]
    context, future = _series(case)
    model = load_timesfm(timesfm_checkpoint(NAME))
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
    predictions = timesfm_functions.forecast(model, group, horizon)
    return np.asarray(predictions[: roles.count(TARGET)])
