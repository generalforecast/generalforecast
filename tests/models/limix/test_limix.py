# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LimiX's preprocessing, loader checks and properties."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.buckets import QUANTILE_LEVELS
from generalforecast.models.contract import Averaging, TabularProperties
from generalforecast.models.limix.checkpoint import (
    check_supported,
    limix_from_state,
    one_task,
)
from generalforecast.models.limix.model import _by_rows, limix_functions
from generalforecast.models.limix.prepare import fit_snapshot, prepare_features
from tests.models.limix.tiny import CLASS_SLOTS, tiny_checkpoint, tiny_limix


def _table() -> np.ndarray:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(12, 5)) * [1, 10, 0.1, 3, 2]
    features[:, 3] = 5.0
    features[2, 0] = np.nan
    features[3, 1] = np.inf
    return features.astype(np.float32)


def test_prepare_standardizes_on_filled_context_values() -> None:
    features = _table()
    snapshot = fit_snapshot(jnp.asarray(features), group_size=2)
    prepared = np.asarray(prepare_features(snapshot, jnp.asarray(features)))

    finite = np.isfinite(features)
    means = np.nanmean(np.where(finite, features, np.nan), axis=0)
    filled = np.where(finite, features, means)
    expected = (filled - filled.mean(axis=0)) / (filled.std(axis=0, ddof=1) + 1e-20)
    # The constant column 3 moves last and reads as zero, as LimiX drops it;
    # group 2 holds it and the zero padding column.
    expected = expected[:, [0, 1, 2, 4]]
    expected = np.concatenate([expected, np.zeros((12, 2))], axis=1).reshape(12, 3, 2)
    expected *= np.sqrt([2 / 2, 2 / 2, 2 / 1])[:, None]
    expected[2, 0, 0] = np.nan

    np.testing.assert_allclose(prepared, expected, rtol=1e-5, atol=1e-5)


def test_prepare_masks_nan_but_fills_infinities() -> None:
    features = _table()
    snapshot = fit_snapshot(jnp.asarray(features), group_size=2)
    prepared = np.asarray(prepare_features(snapshot, jnp.asarray(features)))

    assert np.isnan(prepared[2, 0, 0])
    assert np.isfinite(prepared[3, 0, 1])
    assert np.isnan(prepared).sum() == 1


def test_prepare_ignores_padding_rows() -> None:
    features = _table()
    padded = np.concatenate([features, np.full((4, 5), 1e6, np.float32)])

    fitted = fit_snapshot(jnp.asarray(features), group_size=2)
    with_padding = fit_snapshot(jnp.asarray(padded), group_size=2, num_rows=12)

    for name in ("fill", "center", "scale", "group_scale"):
        np.testing.assert_allclose(
            np.asarray(getattr(with_padding, name)),
            np.asarray(getattr(fitted, name)),
            rtol=1e-6,
        )


def test_properties_come_from_the_parameters() -> None:
    classifier = limix_functions.properties(tiny_limix(regression=False))
    regressor = limix_functions.properties(tiny_limix(regression=True))

    assert classifier == TabularProperties(
        regression=False,
        class_slots=CLASS_SLOTS,
        num_outputs=CLASS_SLOTS,
        averaging=Averaging("predictions"),
        stored_cache=True,
        unlabeled_context=True,
    )
    assert regressor == TabularProperties(
        regression=True,
        class_slots=0,
        num_outputs=QUANTILE_LEVELS,
        averaging=Averaging("buckets", temperature=0.9),
        stored_cache=True,
        unlabeled_context=True,
    )


def test_unsupported_settings_raise() -> None:
    _, settings = tiny_checkpoint()

    with pytest.raises(NotImplementedError, match="layer_arch"):
        check_supported({**settings, "layer_arch": "sfm"})
    with pytest.raises(NotImplementedError, match="arch_version"):
        check_supported({**settings, "arch_version": "1.0"})
    with pytest.raises(NotImplementedError, match="encoder_config_x"):
        check_supported(
            {
                **settings,
                "encoder_config_x": {
                    **settings["encoder_config_x"],
                    "use_feature_mask_indicator": True,
                },
            }
        )


def test_states_must_keep_one_task() -> None:
    state, settings = tiny_checkpoint()

    with pytest.raises(ValueError, match="both tasks"):
        limix_from_state(state, settings)
    with pytest.raises(ValueError, match="parameters the model does not read"):
        limix_from_state(
            {**one_task(state, False), "extra.weight": state["_reg_borders"]}, settings
        )


def test_column_codes_are_fixed_per_table_width() -> None:
    model = tiny_limix(regression=False)
    features = jnp.asarray(_table())
    targets = jnp.asarray(np.arange(12) % 2, jnp.int32)

    first, _ = limix_functions.fit(model, features, targets, 2)
    second, _ = limix_functions.fit(model, features + 1, targets, 2)

    np.testing.assert_array_equal(np.asarray(first.codes), np.asarray(second.codes))
    assert first.codes.shape == (3, 16)


@pytest.mark.parametrize("regression", [False, True])
def test_an_unlabeled_context_row_embeds_as_a_query(regression: bool) -> None:
    model = tiny_limix(regression=regression)
    features = jnp.asarray(_table())
    if regression:
        known = np.arange(12, dtype=np.float32)
        targets, classes = jnp.asarray(known).at[3].set(jnp.nan), 0
    else:
        targets, classes = jnp.asarray(np.arange(12) % 3, jnp.int32).at[3].set(-1), 3

    state, rows = limix_functions.fit(model, features, targets, classes)

    # Its target takes the mask a query's does, and no part in the moments.
    query = limix_functions.embed(model, state, features[3:4])
    np.testing.assert_allclose(rows[3], query[0], atol=1e-6)
    if regression:
        others = np.delete(known, 3)
        assert state.snapshot.target_center == pytest.approx(others.mean())
        assert state.snapshot.target_scale == pytest.approx(others.std(ddof=1))


@pytest.mark.parametrize("tokens", [None, 1, 7, 40, 1000])
def test_rows_in_blocks_match_all_at_once(tokens: int | None) -> None:
    values = jnp.asarray(
        np.random.default_rng(0).normal(size=(1, 9, 4, 3)), jnp.float32
    )

    def function(block: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(block @ jnp.ones((3, 2)))

    np.testing.assert_array_equal(_by_rows(function, values, tokens), function(values))
