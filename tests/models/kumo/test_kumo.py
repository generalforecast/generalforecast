# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What is specific to Kumo Tabular; the conformance suite covers the contract."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.contract import TabularProperties
from generalforecast.models.kumo.checkpoint import load_kumo, save_kumo
from generalforecast.models.kumo.model import kumo_functions
from generalforecast.models.kumo.prepare import (
    CATEGORIES,
    align_categories,
    fit_snapshot,
)
from generalforecast.nn.attention import AttentionKernel
from generalforecast.runtime.device import FLOAT32, DeviceSettings, configure
from generalforecast.runtime.memory import memory_profile
from tests.models.conformance.cases import table
from tests.models.kumo.tiny import (
    CLASS_SLOTS,
    QUANTILES,
    SETTINGS,
    tiny_kumo,
    tiny_kumo_state,
)
from tests.utils import node_harness


def test_properties_come_from_the_parameters() -> None:
    classifier = kumo_functions.properties(tiny_kumo(regression=False))
    regressor = kumo_functions.properties(tiny_kumo(regression=True))

    assert classifier == TabularProperties(
        regression=False,
        class_slots=CLASS_SLOTS,
        num_outputs=CLASS_SLOTS,
        stored_cache=True,
        column_summary=True,
    )
    assert regressor.class_slots == 0
    assert regressor.num_outputs == QUANTILES


def test_queries_read_only_their_key_value_heads() -> None:
    profile = memory_profile(kumo_functions, tiny_kumo(regression=False), 5)

    head_dim = SETTINGS["icl_channels"] // SETTINGS["num_icl_heads"]
    kv_heads = SETTINGS["num_icl_key_value_heads_for_query"]
    keys_and_values = 2 * kv_heads * head_dim * 4
    assert profile.cache_bytes_per_row == SETTINGS["num_icl_layers"] * keys_and_values


def _probabilities(categorical: jax.Array | None) -> np.ndarray:
    model, data = tiny_kumo(regression=False), table(regression=False)
    state, rows = kumo_functions.fit(
        model, data.features, data.targets, data.num_classes, categorical=categorical
    )
    cache = kumo_functions.encode(model, state, rows, data.targets)
    queries = kumo_functions.embed(model, state, data.queries)
    outputs = kumo_functions.decode(model, state, cache, queries)
    return np.asarray(kumo_functions.postprocess(model, state, outputs))


def test_categorical_columns_embed_through_their_own_weights() -> None:
    numerical = _probabilities(None)
    columns = jnp.arange(5)

    # Every column numerical takes the categorical path to the same result.
    np.testing.assert_allclose(_probabilities(columns < 0), numerical, atol=1e-6)
    assert not np.allclose(_probabilities(columns % 2 == 0), numerical, atol=1e-3)


@pytest.mark.parametrize("summary", ["table", "context"])
def test_columns_with_more_categories_than_kept_are_flagged_once(
    summary: Literal["table", "context"], caplog: pytest.LogCaptureFixture
) -> None:
    rows = CATEGORIES + 8
    generator = np.random.default_rng(0)
    features = np.stack(
        [3 * generator.permutation(rows), generator.normal(size=rows)], axis=1
    ).astype(np.float32)
    mask = jnp.asarray([True, False])
    snapshot = fit_snapshot(jnp.asarray(features), None, jnp.ones(rows, bool), mask)
    assert snapshot.categories is not None
    ranks = align_categories(snapshot.categories, jnp.asarray(features), mask)

    # Kept and interpolated codes alike rank exactly when no code is skipped.
    np.testing.assert_array_equal(ranks[:, 0], features[:, 0] / 3)
    assert bool(snapshot.categories.crowded[0])

    # The worker logs in this process, where caplog sees it; the engine's
    # workers run in processes of their own, whose logs reach only stderr.
    client = node_harness.local(compile_cache=False)
    crowded = client.create_table(
        "crowded",
        model="tests/kumo-classifier",
        columns={"code": "categorical", "x": "numeric"},
        classes=2,
        summary=summary,
    )
    crowded.insert(np.arange(rows), features, np.arange(rows) % 2)
    with caplog.at_level(logging.WARNING, logger="generalforecast.worker"):
        crowded.predict(features[:4])
        crowded.refit()
        crowded.predict(features[:4])

    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "1 categorical column holds" in caplog.text


def test_checkpoints_round_trip_through_safetensors(tmp_path: Path) -> None:
    state = tiny_kumo_state(regression=False)
    save_kumo(
        tmp_path,
        {key: value.astype(np.float32) for key, value in state.items()},
        SETTINGS,
    )

    loaded = load_kumo(tmp_path)

    jax.tree.map(np.testing.assert_array_equal, loaded, tiny_kumo(regression=False))


def test_a_fit_in_block_chunks_holds_a_few_rows_at_once() -> None:
    def temporaries(chunk: int | None) -> int:
        settings = DeviceSettings(FLOAT32, FLOAT32, AttentionKernel(block_chunk=chunk))
        model = configure(tiny_kumo(regression=False), settings)
        features = jax.ShapeDtypeStruct((4096, 24), jnp.float32)
        targets = jax.ShapeDtypeStruct((4096,), jnp.int32)
        fit = jax.jit(lambda model, x, y: kumo_functions.fit(model, x, y, 3))
        analysis = fit.lower(model, features, targets).compile().memory_analysis()
        assert analysis is not None
        return analysis.temp_size_in_bytes

    # Wide tables of many rows fit their column states in chunks of rows,
    # rather than holding every row's cells at every step of a layer.
    assert temporaries(1024) < temporaries(None) / 3
