# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What is specific to TabFM; the conformance suite covers the contract."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.conformance import predict, table
from generalforecast.models.tabfm.checkpoint import tabfm_from_state
from generalforecast.models.tabfm.model import tabfm_functions, tabfm_outputs
from tests.models.tabfm.tiny import tiny_state, tiny_tabfm


def test_a_regressor_checkpoint_is_refused() -> None:
    state, settings = tiny_state()
    with pytest.raises(ValueError, match="one value"):
        tabfm_from_state(state, {**settings, "is_classifier": False})


def test_rope_rotates_by_the_checkpoints_frequencies() -> None:
    state, settings = tiny_state()
    stored = np.asarray(state["row_interactor.tf_row.rope.freqs"]) * 1.001
    state["row_interactor.tf_row.rope.freqs"] = jnp.asarray(stored, jnp.float32)
    rope = tabfm_from_state(state, settings).rows[0].blocks[0].attention.rope
    assert rope is not None and rope.frequencies is not None
    np.testing.assert_allclose(rope.frequencies, stored, rtol=1e-6)


def test_categorical_columns_take_their_own_embedding() -> None:
    model = tiny_tabfm()
    data = table(False, rows=24, queries=6, features=4, seed=2)
    numeric = predict(tabfm_functions, model, data)
    coded = jnp.array([False, True, False, False])
    categorical = predict(tabfm_functions, model, data, categorical=coded)
    assert not np.allclose(numeric, categorical)


def test_the_contract_calls_predict_as_one_pass_over_the_prepared_table() -> None:
    from generalforecast.models.tabicl.model import class_probabilities
    from generalforecast.models.tabicl.prepare import (
        apply_numeric_snapshot,
        fit_numeric_snapshot,
    )

    model = tiny_tabfm()
    data = table(False, rows=24, queries=6, features=4, seed=3)
    snapshot = fit_numeric_snapshot(data.features)
    prepared = jnp.concatenate(
        [
            apply_numeric_snapshot(snapshot, data.features),
            apply_numeric_snapshot(snapshot, data.queries),
        ]
    )
    logits = tabfm_outputs(
        model, prepared[None], data.targets[None], data.features.shape[0]
    )[0]
    expected = class_probabilities(logits, data.num_classes, model.temperature)
    np.testing.assert_allclose(
        predict(tabfm_functions, model, data), expected, atol=1e-5
    )
