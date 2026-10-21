# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-2 from Amazon's package, and generalforecast's port, on the same weights."""

from __future__ import annotations

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from generalforecast.models.chronos2.checkpoint import (
    PUBLISHED,
    chronos2_checkpoint,
    load_chronos2,
)
from generalforecast.models.chronos2.model import chronos2_functions
from generalforecast.models.series import KNOWN, PAST, TARGET, SeriesGroup

# Two targets, a past-only covariate and a known covariate, over 700 steps,
# which is not whole patches, so both sides pad the start; or one target that
# starts after 40 unobserved steps, so its first patches are never keys.
GROUPS = {"covariates": (TARGET, TARGET, PAST, KNOWN), "univariate": (TARGET,)}
STEPS = 700
# Past one pass of 64 patches, so the pipeline continues along quantile paths.
LONG_HORIZON = 1100


@functools.cache
def _series(case: str) -> tuple[np.ndarray, np.ndarray]:
    """Context rows with a few missing values, and the last row's future values."""
    generator = np.random.default_rng(11)
    rows = len(GROUPS[case])
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
    from chronos import Chronos2Pipeline

    published = PUBLISHED["chronos-2"]
    return Chronos2Pipeline.from_pretrained(
        published.repository, revision=published.revision, device_map="cpu"
    )


def upstream_forecast(case: str, horizon: int) -> np.ndarray:
    """Amazon's pipeline on a group, `(targets, horizon, quantiles)`.

    One pass runs the network directly on the padded batch the pipeline would
    build; a longer horizon runs the pipeline, with its quantile paths.
    """
    roles = GROUPS[case]
    context, future = _series(case)
    pipeline = _upstream()
    known = roles.count(KNOWN)
    if horizon <= pipeline.model_prediction_length:
        rows = len(roles)
        covariates = torch.full((rows, horizon), torch.nan)
        if known:
            covariates[-1] = torch.from_numpy(future[:horizon])
        with torch.no_grad():
            outputs = pipeline.model(
                context=torch.from_numpy(context),
                group_ids=torch.zeros(rows, dtype=torch.long),
                future_covariates=covariates,
                num_output_patches=horizon // pipeline.model_output_patch_size,
            )
        predictions = outputs.quantile_preds[: roles.count(TARGET)]
    else:
        inputs: dict[str, Any] = {"target": context[: roles.count(TARGET)]}
        if known:
            inputs["past_covariates"] = {"past": context[2], "known": context[3]}
            inputs["future_covariates"] = {"known": future[:horizon]}
        predictions = pipeline.predict([inputs], prediction_length=horizon)[0]
    return predictions.numpy().transpose(0, 2, 1)


def our_forecast(case: str, horizon: int) -> np.ndarray:
    """generalforecast's port on the same group, converted from the same files."""
    roles = GROUPS[case]
    context, future = _series(case)
    model = load_chronos2(chronos2_checkpoint("chronos-2"))
    rows = len(roles)
    pad = -STEPS % model.patch
    values = np.concatenate([np.zeros((rows, pad), np.float32), context], axis=1)
    observed = np.isfinite(values)
    observed[:, :pad] = False
    known = None
    if roles.count(KNOWN):
        known = np.full((rows, horizon), np.nan, np.float32)
        known[-1] = future[:horizon]
    group = SeriesGroup(
        jnp.asarray(np.nan_to_num(values)),
        jnp.asarray(observed),
        jnp.asarray(roles, jnp.int32),
        None if known is None else jnp.asarray(known),
    )
    predictions = chronos2_functions.forecast(model, group, horizon)
    return np.asarray(predictions[: roles.count(TARGET)])
