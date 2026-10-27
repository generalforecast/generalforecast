# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What is specific to TabICL; the conformance suite covers the contract."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import registry
from generalforecast.models.column_embedding import embed_cells, summarize_columns
from generalforecast.models.contract import TabularProperties
from generalforecast.models.icl import decode_queries, encode_context
from generalforecast.models.row_interaction import interact_rows
from generalforecast.models.tabicl.checkpoint import load_tabicl
from generalforecast.models.tabicl.model import TabICL, tabicl_functions
from generalforecast.models.tabicl.prepare import (
    apply_numeric_snapshot,
    fit_numeric_snapshot,
)
from generalforecast.models.targets import scale_targets, unscale_quantiles
from generalforecast.nn.attention import AttentionKernel
from generalforecast.runtime.device import (
    DeviceSettings,
    configure,
    device_settings,
    store_cache,
)
from generalforecast.runtime.memory import memory_profile
from generalforecast.runtime.shapes import pad_rows
from tests.models.conformance.cases import Table, predict, table
from tests.models.tabicl.tiny import (
    CLASS_SLOTS,
    ICL,
    QUANTILES,
    tiny_tabicl,
)
from tests.utils.synthetic_table import synthetic_table

CHECKPOINTS = Path(__file__).resolve().parents[3] / "checkpoints"
TASKS = pytest.mark.parametrize("regression", [False, True], ids=["classes", "values"])
FLOAT32 = jnp.dtype("float32")


def _checkpoint(name: str) -> TabICL:
    """A checkpoint from the repository's directory, else the Hugging Face cache."""
    path = CHECKPOINTS / f"tabicl-v2-{name}"
    if not path.exists():
        from huggingface_hub import snapshot_download

        try:
            path = Path(
                snapshot_download(
                    f"general-forecast/tabicl-v2-{name}",
                    revision=registry.REVISIONS[f"tabicl-v2-{name}"],
                    local_files_only=True,
                )
            )
        except Exception:  # noqa: BLE001
            pytest.skip("the checkpoint is neither converted nor cached")
    return load_tabicl(path)


def _close(actual: object, expected: object, tolerance: float = 1e-5) -> None:
    jax.tree.map(
        lambda a, e: np.testing.assert_allclose(
            np.asarray(a), np.asarray(e), rtol=tolerance, atol=tolerance
        ),
        actual,
        expected,
    )


def _stages(model: TabICL, data: Table) -> jax.Array:
    """The contract's result, composed by hand from the stage functions it wraps."""
    targets = data.targets
    snapshot = fit_numeric_snapshot(
        data.features, targets if model.regression else None
    )
    context = apply_numeric_snapshot(snapshot, data.features)[None]
    scaled = scale_targets(snapshot, targets)[None]
    summary = summarize_columns(model.columns, context, scaled)
    context_rows = interact_rows(
        model.rows, embed_cells(model.columns, summary, context, scaled)
    )
    queries = apply_numeric_snapshot(snapshot, data.queries)[None]
    query_rows = interact_rows(model.rows, embed_cells(model.columns, summary, queries))
    outputs = decode_queries(
        model.icl, encode_context(model.icl, context_rows, scaled), query_rows
    )[0]
    if model.regression:
        return unscale_quantiles(snapshot, outputs)
    return jnn.softmax(outputs[:, : data.num_classes] / 0.9, axis=-1)


@TASKS
def test_contract_matches_the_stage_functions(regression: bool) -> None:
    model = tiny_tabicl(regression=regression)
    data = table(regression)

    _close(predict(tabicl_functions, model, data), _stages(model, data))


@TASKS
def test_block_chunks_match_one_step(regression: bool) -> None:
    data = table(regression, rows=40, queries=12)

    def run(chunk: int | None) -> tuple:
        settings = DeviceSettings(FLOAT32, FLOAT32, AttentionKernel(block_chunk=chunk))
        model = configure(tiny_tabicl(regression=regression), settings)
        state, rows = tabicl_functions.fit(
            model, data.features, data.targets, data.num_classes
        )
        return state, rows, tabicl_functions.embed(model, state, data.queries)

    _close(run(64), run(None))


def test_properties_come_from_the_parameters() -> None:
    classifier = tabicl_functions.properties(tiny_tabicl(regression=False))
    regressor = tabicl_functions.properties(tiny_tabicl(regression=True))

    assert classifier == TabularProperties(
        regression=False,
        class_slots=CLASS_SLOTS,
        num_outputs=CLASS_SLOTS,
        stored_cache=True,
        column_summary=True,
    )
    assert regressor.class_slots == 0
    assert regressor.num_outputs == QUANTILES


def test_cache_holds_keys_and_values_per_icl_block() -> None:
    profile = memory_profile(tabicl_functions, tiny_tabicl(regression=False), 5)

    assert profile.cache_bytes_per_row == 2 * 2 * ICL * 4
    # Each of the two blocks' caches keeps its int32 length.
    assert profile.cache_fixed_bytes == 2 * 4
    assert profile.row_bytes == ICL * 4


def test_preprocessing_stays_float32() -> None:
    half = jnp.dtype("float16")
    model = configure(
        tiny_tabicl(regression=True), DeviceSettings(half, half, AttentionKernel())
    )
    data = table(regression=True)

    state, _ = tabicl_functions.fit(model, data.features, data.targets, 0)

    statistics = [
        leaf
        for leaf in jax.tree.leaves(state.snapshot)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    assert all(leaf.dtype == jnp.float32 for leaf in statistics)
    assert all(leaf.dtype == half for leaf in jax.tree.leaves(state.columns))


@TASKS
def test_snapshot_ignores_rows_past_num_rows(regression: bool) -> None:
    data = table(regression, rows=20)
    features = np.array(pad_rows(data.features, 32, fill=0.0))
    features[20:] = 1e3 * np.random.default_rng(1).normal(size=(12, 5))
    features[20, 0] = np.nan
    targets = data.targets if regression else None
    padded_targets = pad_rows(data.targets, 32, fill=1e4) if regression else None

    padded = fit_numeric_snapshot(jnp.asarray(features), padded_targets, num_rows=20)

    _close(padded, fit_numeric_snapshot(data.features, targets))


def test_constant_and_empty_columns_stay_bounded() -> None:
    features = np.random.default_rng(0).normal(size=(16, 3)).astype(np.float32)
    features[:, 0] = np.nan
    features[:, 1] = 5.0
    # Small but varying, as rates are: still standardized.
    features[:, 2] *= 1e-4

    snapshot = fit_numeric_snapshot(jnp.asarray(features))
    prepared = apply_numeric_snapshot(snapshot, jnp.asarray(features))
    query = apply_numeric_snapshot(snapshot, jnp.asarray([[1.0, 6.0, 0.0]]))

    # Constant columns move last, as upstream drops them, and are 0 in the
    # context; a query that differs is clipped softly rather than scaled by the
    # inverse of a vanishing deviation.
    assert int(snapshot.columns.count) == 1
    np.testing.assert_allclose(prepared[:, 1:], 0.0)
    assert 0 < float(query[0, 1]) < 5 and 0 < float(query[0, 2]) < 5
    np.testing.assert_allclose(prepared[:, 0].std(), 1.0, rtol=2e-2)


def test_outliers_are_clipped_softly_as_upstream_clips_them() -> None:
    features = np.random.default_rng(0).normal(size=(64, 1)).astype(np.float32)
    features[0, 0] = 50.0

    snapshot = fit_numeric_snapshot(jnp.asarray(features))
    prepared = apply_numeric_snapshot(snapshot, jnp.asarray(features))

    # The bounds come from the rows without the outlier, four deviations wide.
    standardizer = snapshot.features
    rest = (features[1:, 0] - standardizer.center[0]) / standardizer.scale[0]
    expected_upper = rest.mean() + 4 * rest.std(ddof=1)
    np.testing.assert_allclose(standardizer.upper[0], expected_upper, rtol=1e-4)
    outlier = (features[0, 0] - standardizer.center[0]) / standardizer.scale[0]
    np.testing.assert_allclose(
        prepared[0, 0], np.log1p(abs(outlier)) + standardizer.upper[0], rtol=1e-5
    )
    assert float(prepared[0, 0]) < outlier


def test_checkpoint_sizes() -> None:
    model = _checkpoint("classifier")

    profile = memory_profile(tabicl_functions, model, num_features=10)

    assert profile.cache_bytes_per_row == 12 * 2 * 512 * 4
    assert profile.row_bytes == 512 * 4


def test_checkpoint_gpu_settings_stay_close_to_float32() -> None:
    model = _checkpoint("classifier")
    source = synthetic_table(512, 64, 8, missing=0.05)
    data = Table(
        jnp.asarray(source.features),
        jnp.asarray(source.labels),
        jnp.asarray(source.queries),
        int(source.labels.max()) + 1,
    )

    def run(settings: DeviceSettings) -> jax.Array:
        configured = configure(model, settings)
        state, rows = tabicl_functions.fit(
            configured, data.features, data.targets, data.num_classes
        )
        cache = store_cache(
            tabicl_functions.encode(configured, state, rows, data.targets), settings
        )
        query_rows = tabicl_functions.embed(configured, state, data.queries)
        outputs = tabicl_functions.decode(configured, state, cache, query_rows)
        return tabicl_functions.postprocess(configured, state, outputs)

    reference = DeviceSettings(FLOAT32, FLOAT32, AttentionKernel())
    # A CPU compiles no float16 products that accumulate in float32, so query
    # chunks, which compile their steps, go; they change no value.
    gpu = device_settings("gpu")
    unchunked = dataclasses.replace(gpu.kernel, query_chunk=None)
    gpu = run(dataclasses.replace(gpu, kernel=unchunked))

    # An L4 measured 0.015 here, and up to 0.026 on real tables at 8,192 rows.
    assert float(jnp.max(jnp.abs(gpu - run(reference)))) < 3e-2


def test_checkpoint_int8_cache_stays_close_to_float32() -> None:
    model = _checkpoint("classifier")
    source = synthetic_table(1024, 32, 8, missing=0.05)
    data = Table(
        jnp.asarray(source.features),
        jnp.asarray(source.labels),
        jnp.asarray(source.queries),
        int(source.labels.max()) + 1,
    )

    def run(settings: DeviceSettings) -> jax.Array:
        configured = configure(model, settings)
        state, rows = tabicl_functions.fit(
            configured, data.features, data.targets, data.num_classes
        )
        cache = store_cache(
            tabicl_functions.encode(configured, state, rows, data.targets), settings
        )
        query_rows = tabicl_functions.embed(configured, state, data.queries)
        outputs = tabicl_functions.decode(configured, state, cache, query_rows)
        return tabicl_functions.postprocess(configured, state, outputs)

    reference = DeviceSettings(FLOAT32, FLOAT32, AttentionKernel())
    int8 = DeviceSettings(FLOAT32, jnp.dtype("int8"), AttentionKernel())

    # A CPU measured 0.027 in probability at 1,024 rows.
    assert float(jnp.max(jnp.abs(run(int8) - run(reference)))) < 5e-2
