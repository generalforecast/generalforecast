# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM 3's loader checks and the choices its forecasts make."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.conformance import group
from generalforecast.models.series import KNOWN
from generalforecast.models.timesfm3.checkpoint import (
    check_supported,
    timesfm_from_state,
)
from generalforecast.models.timesfm3.model import encode, timesfm_functions
from tests.models.timesfm3.tiny import PATCH, tiny_checkpoint, tiny_timesfm


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
