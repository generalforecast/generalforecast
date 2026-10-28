# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mitra's preprocessing, loader checks and properties."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.mitra.checkpoint import check_supported, mitra_from_state
from generalforecast.models.mitra.model import mitra_functions
from generalforecast.models.mitra.prepare import fit_snapshot, prepare_features
from tests.models.mitra.tiny import tiny_checkpoint, tiny_mitra


def _table() -> np.ndarray:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(12, 5)) * [1, 10, 0.1, 3, 2]
    features[:, 1] = 5.0
    features[2, 0] = np.nan
    features[3, 4] = np.round(features[3, 4])
    return features.astype(np.float32)


def _expected_ranks(features: np.ndarray) -> np.ndarray:
    """Mitra's encoding, with NumPy's linear quantiles and a left search."""
    filled = np.where(np.isnan(features), np.nanmean(features, axis=0), features)
    levels = np.arange(1, 1000) / 1000
    columns = []
    for column in filled.T:
        quantiles = np.quantile(column, levels)
        ranks = np.searchsorted(quantiles, column, side="left") / len(column)
        deviation = ranks.std()
        columns.append((ranks - ranks.mean()) / deviation if deviation else 0 * ranks)
    return np.stack(columns, axis=1)


def test_prepare_ranks_against_context_quantiles() -> None:
    features = _table()
    snapshot = fit_snapshot(jnp.asarray(features))
    prepared = np.asarray(prepare_features(snapshot, jnp.asarray(features)))

    # The constant column 1 moves last.
    expected = _expected_ranks(features)[:, [0, 2, 3, 4, 1]]
    np.testing.assert_allclose(prepared, expected, rtol=1e-5, atol=1e-5)
    assert int(snapshot.columns.count) == 4


def _probabilities(features: np.ndarray) -> np.ndarray:
    model = tiny_mitra()
    targets = jnp.asarray(np.arange(12) % 3, jnp.int32)
    state, rows = mitra_functions.fit(model, jnp.asarray(features), targets, 3)
    cache = mitra_functions.encode(model, state, rows, targets)
    queries = mitra_functions.embed(model, state, jnp.asarray(features[:4]))
    outputs = mitra_functions.decode(model, state, cache, queries)
    return np.asarray(mitra_functions.postprocess(model, state, outputs))


def test_constant_columns_are_left_out() -> None:
    features = _table()

    without = _probabilities(np.delete(features, 1, axis=1))
    with_constant = _probabilities(features)

    np.testing.assert_allclose(with_constant, without, rtol=1e-5, atol=1e-6)


def test_unsupported_settings_raise() -> None:
    state, settings = tiny_checkpoint()

    with pytest.raises(NotImplementedError, match="task='REGRESSION'"):
        check_supported({**settings, "task": "REGRESSION", "dim_output": 1})
    with pytest.raises(NotImplementedError, match="n_kv_heads"):
        check_supported({**settings, "n_kv_heads": 1})
    with pytest.raises(ValueError, match="parameters the model does not read"):
        mitra_from_state({**state, "extra.weight": state["final_layer.bias"]}, settings)


@pytest.mark.parametrize("regression", [False, True])
def test_an_unlabeled_context_row_embeds_as_a_query(regression: bool) -> None:
    model = tiny_mitra(regression=regression)
    features = jnp.asarray(_table())
    if regression:
        known = np.arange(12, dtype=np.float32)
        targets, classes = jnp.asarray(known).at[0].set(jnp.nan), 0
    else:
        targets, classes = jnp.asarray(np.arange(12) % 3, jnp.int32).at[0].set(-1), 3

    state, rows = mitra_functions.fit(model, features, targets, classes)

    # Its target takes the mask a query's does, and no part in the range,
    # which would start at 0 with it.
    query = mitra_functions.embed(model, state, features[:1])
    np.testing.assert_allclose(rows[0], query[0], atol=1e-6)
    if regression:
        assert state.snapshot.target_center == pytest.approx(1.0)
        assert state.snapshot.target_scale == pytest.approx(10.0)
        # With no target known, the range stays finite.
        unknown = jnp.full(12, jnp.nan, jnp.float32)
        blank, _ = mitra_functions.fit(model, features, unknown, classes)
        assert np.isfinite(blank.snapshot.target_center)
        assert np.isfinite(blank.snapshot.target_scale)
