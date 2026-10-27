# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM 3's loader checks and the choices its forecasts make."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.conformance import group, paged_series
from generalforecast.models.series import KNOWN, TARGET, SeriesGroup
from generalforecast.models.timesfm3.checkpoint import (
    check_supported,
    timesfm_from_state,
)
from generalforecast.models.timesfm3.model import (
    Moments,
    _refined,
    encode,
    timesfm_functions,
)
from tests.models.timesfm3.tiny import (
    OUTPUT_PATCH,
    PATCH,
    tiny_checkpoint,
    tiny_timesfm,
)


def test_unsupported_settings_raise() -> None:
    state, settings = tiny_checkpoint()
    stack = settings["transformer_config"]

    with pytest.raises(NotImplementedError, match="use_stitching=False"):
        check_supported({**settings, "use_stitching": False})
    with pytest.raises(NotImplementedError, match="transformer use_rope_var=True"):
        check_supported(
            {
                **settings,
                "transformer_config": {
                    **stack,
                    "transformer": {**stack["transformer"], "use_rope_var": True},
                },
            }
        )
    with pytest.raises(NotImplementedError, match="whole input patches"):
        check_supported({**settings, "output_patch_len": 6})
    with pytest.raises(NotImplementedError, match="num_kv_heads"):
        check_supported({**settings, "num_kv_heads": 1})
    with pytest.raises(ValueError, match="parameters the model does not read"):
        timesfm_from_state(
            {**state, "extra.weight": state["output_head.bias"]}, settings
        )


def test_quantiles_come_sorted() -> None:
    data = group(steps=4 * PATCH, horizon=PATCH, known=False)

    predictions = timesfm_functions.forecast(tiny_timesfm(), data, 3 * PATCH)

    assert bool((jnp.diff(predictions, axis=-1) >= 0).all())


def test_a_row_without_observations_is_read_as_zeros() -> None:
    data = group(steps=4 * PATCH, horizon=PATCH, known=False)
    empty = data._replace(observed=data.observed.at[1].set(False))

    predictions = timesfm_functions.forecast(tiny_timesfm(), empty, PATCH)

    assert bool(np.isfinite(predictions).all())


def test_encode_refuses_known_covariates() -> None:
    data = group(steps=4 * PATCH, horizon=PATCH)

    assert bool((data.roles == KNOWN).any())
    with pytest.raises(ValueError, match="known covariate"):
        encode(tiny_timesfm(), data)


def test_the_horizon_frames_follow_the_median_by_its_level() -> None:
    # Nine levels, as the tiny head emits, with the median off the middle.
    levels = (0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    state, settings = tiny_checkpoint()
    model = timesfm_from_state(state, {**settings, "quantiles": list(levels)})
    rows, steps = 2, OUTPUT_PATCH // PATCH
    # No values seen yet, in a frame of scale 1.
    zeros = jnp.zeros(rows, jnp.float32)
    start = Moments(zeros, zeros, zeros + 1)
    # The median predicts 3 at every step; every other level predicts 100.
    last = jnp.full((rows, OUTPUT_PATCH, len(levels)), 100.0)
    last = last.at[..., levels.index(0.5)].set(3.0).reshape(rows, -1)
    raw = jnp.broadcast_to(last[:, None], (rows, steps, last.shape[-1]))

    frames = _refined(model, start, last, raw)

    np.testing.assert_allclose(frames.mean, 3.0)
    np.testing.assert_allclose(frames.deviation, 0.0)


def test_a_grid_without_a_median_is_refused_at_load() -> None:
    state, settings = tiny_checkpoint()
    levels = [0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9]

    with pytest.raises(NotImplementedError, match="must hold the median"):
        timesfm_from_state(state, {**settings, "quantiles": levels})


def test_a_cache_read_back_padded_decodes_as_the_cache_encode_made() -> None:
    model = tiny_timesfm()
    steps = 10 * PATCH
    time = np.arange(steps)
    noise = np.random.default_rng(0).normal(size=(2, steps))
    # Trending rows, so the trend's time scale counts the real patches only.
    values = np.stack([5 + 0.5 * time, 2 - 0.2 * time]) + 0.2 * noise
    data = SeriesGroup(
        jnp.asarray(values, jnp.float32),
        jnp.ones((2, steps), bool),
        jnp.asarray([TARGET, TARGET], jnp.int32),
    )
    cache = encode(model, data)
    assert bool(cache.trend.applies.all())
    # The patches pad to a bucket of 16, as the worker stores them.
    grows = [leaf.ndim > 0 and leaf.shape[0] == 10 for leaf in jax.tree.leaves(cache)]
    paged = paged_series(cache, grows, 6)

    np.testing.assert_allclose(
        timesfm_functions.decode(model, paged, 3 * PATCH),
        timesfm_functions.decode(model, cache, 3 * PATCH),
        rtol=1e-6,
        atol=1e-6,
    )
