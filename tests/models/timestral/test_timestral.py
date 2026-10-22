# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Timestral's loader checks and the choices its forecasts make."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.conformance import group
from generalforecast.models.series import KNOWN
from generalforecast.models.timestral.checkpoint import (
    check_supported,
    timestral_from_state,
)
from generalforecast.models.timestral.model import (
    decode,
    encode,
    extend,
    timestral_functions,
)
from tests.models.timestral.tiny import CONTEXT, PATCH, tiny_checkpoint, tiny_timestral


def test_unsupported_settings_raise() -> None:
    state, settings = tiny_checkpoint()

    with pytest.raises(NotImplementedError, match="attn_bias=True"):
        check_supported({**settings, "attn_bias": True})
    with pytest.raises(NotImplementedError, match="layer kinds"):
        check_supported({**settings, "block_schedule": ["time", "space"]})
    with pytest.raises(NotImplementedError, match="median"):
        check_supported({**settings, "quantile_knots": [0.1, 0.9]})
    with pytest.raises(NotImplementedError, match="num_kv_heads"):
        check_supported({**settings, "num_kv_heads": 1})
    with pytest.raises(ValueError, match="parameters the model does not read"):
        timestral_from_state(
            {**state, "extra.weight": state["representation.forecast_query"]}, settings
        )


def test_symmetric_averaging_is_a_model_setting() -> None:
    data = group(steps=4 * PATCH, horizon=PATCH, known=False)

    plain = timestral_functions.forecast(tiny_timestral(symmetric=False), data, PATCH)
    averaged = timestral_functions.forecast(tiny_timestral(), data, PATCH)
    negated = timestral_functions.forecast(
        tiny_timestral(symmetric=False), data._replace(values=-data.values), PATCH
    )

    assert float(jnp.abs(plain - averaged).max()) > 1e-4
    np.testing.assert_allclose(
        averaged, 0.5 * (plain - negated[..., ::-1]), rtol=1e-5, atol=1e-5
    )


def test_encode_refuses_known_covariates() -> None:
    data = group(steps=4 * PATCH, horizon=PATCH)

    assert bool((data.roles == KNOWN).any())
    with pytest.raises(ValueError, match="known covariate"):
        encode(tiny_timestral(), data)


def test_decode_refuses_known_futures_and_extend_the_context_limit() -> None:
    data = group(steps=4 * PATCH, horizon=PATCH, known=False)
    model = tiny_timestral()
    cache = encode(model, data)

    with pytest.raises(ValueError, match="known covariate"):
        decode(model, cache, PATCH, future=jnp.zeros((3, PATCH)))
    with pytest.raises(ValueError, match="limit"):
        extend(model, cache, group(steps=CONTEXT, horizon=PATCH, known=False))
