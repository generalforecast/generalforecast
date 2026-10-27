# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Timestral's loader checks and the choices its forecasts make."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.conformance import group, paged_series
from generalforecast.models.series import KNOWN
from generalforecast.models.timestral.checkpoint import (
    check_supported,
    timestral_from_state,
)
from generalforecast.models.timestral.model import (
    decode,
    decode_within,
    encode,
    extend,
    timestral_functions,
)
from generalforecast.runtime.device import to_cache_dtype
from tests.models.timestral.tiny import (
    BLOCK_SIZE,
    CONTEXT,
    PATCH,
    tiny_checkpoint,
    tiny_timestral,
)


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


def _grows(cache: Any) -> list[bool]:
    """Which of a cache's leaves have its patches on their first axis."""
    patches = cache.valid.shape[0]
    return [
        leaf.ndim > 0 and leaf.shape[0] == patches for leaf in jax.tree.leaves(cache)
    ]


@pytest.mark.parametrize("patches", [CONTEXT // PATCH, CONTEXT // PATCH - 3])
@pytest.mark.parametrize("paged", [False, True])
def test_decode_past_the_context_limit_forecasts_as_forecast_does(
    patches: int, paged: bool
) -> None:
    model = tiny_timestral()
    data = group(steps=patches * PATCH, horizon=PATCH, known=False)
    # Fed-back medians pass the limit within the horizon, so forecast drops
    # the oldest steps and runs its window whole.
    horizon = 4 * BLOCK_SIZE * PATCH + PATCH
    cache = encode(model, data)
    if paged:
        # Read back padded past its real patches, as the worker reads a bucket.
        cache = paged_series(cache, _grows(cache), CONTEXT // PATCH - patches + 1)

    np.testing.assert_allclose(
        decode(model, cache, horizon),
        timestral_functions.forecast(model, data, horizon),
        rtol=1e-5,
        atol=1e-5,
    )


def test_paged_caches_of_different_lengths_decode_together_past_the_limit() -> None:
    model = tiny_timestral()
    bucket = CONTEXT // PATCH
    horizon = 4 * BLOCK_SIZE * PATCH + PATCH
    groups = [
        group(steps=patches * PATCH, horizon=PATCH, known=False, seed=seed)
        for seed, patches in enumerate((bucket, bucket - 5))
    ]
    pages = []
    for data in groups:
        cache = encode(model, data)
        pages.append(paged_series(cache, _grows(cache), bucket - cache.valid.shape[0]))
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *pages)

    together = jax.vmap(lambda cache: decode(model, cache, horizon))(stacked)

    for index, data in enumerate(groups):
        np.testing.assert_allclose(
            together[index],
            timestral_functions.forecast(model, data, horizon),
            rtol=1e-5,
            atol=1e-5,
        )


def test_caches_at_the_limit_bucket_decode_within_it_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_timestral()
    bucket = CONTEXT // PATCH
    # Three blocks; the fed-back medians keep every context within the limit,
    # though each is stored at the limit's bucket, as the worker stores it.
    horizon = 3 * BLOCK_SIZE * PATCH
    lengths = (bucket - 6, bucket - 7, bucket - 6, bucket - 9)
    groups = [
        group(steps=patches * PATCH, horizon=PATCH, known=False, seed=seed)
        for seed, patches in enumerate(lengths)
    ]
    assert all(g.values.shape[1] + horizon <= CONTEXT for g in groups)
    pages = []
    for data in groups:
        cache = encode(model, data)
        pages.append(paged_series(cache, _grows(cache), bucket - cache.valid.shape[0]))
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *pages)

    within = jax.jit(jax.vmap(lambda cache: decode_within(model, cache, horizon)))
    general = jax.jit(jax.vmap(lambda cache: decode(model, cache, horizon)))

    np.testing.assert_array_equal(within(stacked), general(stacked))
    for index, data in enumerate(groups):
        np.testing.assert_allclose(
            within(stacked)[index],
            timestral_functions.forecast(model, data, horizon),
            rtol=1e-5,
            atol=1e-5,
        )
    # No block of the promised decode runs the window whole.
    from generalforecast.models.timestral import model as timestral

    passes: list[int] = []
    window = timestral._window_block

    def counted(*args: Any) -> Any:
        passes.append(1)
        return window(*args)

    monkeypatch.setattr(timestral, "_window_block", counted)
    jax.eval_shape(jax.vmap(lambda c: decode_within(model, c, horizon)), stacked)
    assert not passes
    jax.eval_shape(jax.vmap(lambda c: decode(model, c, horizon)), stacked)
    assert passes


def test_decode_within_refuses_a_cache_it_can_see_passes_the_limit() -> None:
    model = tiny_timestral()
    cache = encode(model, group(steps=CONTEXT - PATCH, horizon=PATCH, known=False))
    decode_within(model, cache, PATCH)
    with pytest.raises(ValueError, match="limit"):
        decode_within(model, cache, 2 * PATCH)


def test_a_reduced_precision_cache_keeps_the_context_values_exact() -> None:
    cache = encode(tiny_timestral(), group(steps=4 * PATCH, horizon=PATCH, known=False))

    stored = to_cache_dtype(cache, jnp.dtype("float16"))

    assert stored.values.dtype == jnp.int32
    np.testing.assert_array_equal(stored.values, cache.values)
