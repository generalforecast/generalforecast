# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What is specific to TabICL; the conformance suite covers the contract."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import registry
from generalforecast.models.tabicl.checkpoint import load_tabicl
from generalforecast.models.tabicl.model import TabICL, tabicl_functions
from generalforecast.models.tabicl.prepare import (
    apply_numeric_snapshot,
    fit_numeric_snapshot,
)
from generalforecast.nn.attention import AttentionKernel
from generalforecast.runtime.device import (
    DeviceSettings,
    configure,
    device_settings,
    store_cache,
)
from tests.models.conformance.cases import Table, table
from tests.models.tabicl.tiny import tiny_tabicl
from tests.utils.synthetic_table import synthetic_table

CHECKPOINTS = Path(__file__).resolve().parents[3] / "checkpoints"
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


def _checkpoint_drift(rows: int, queries: int, settings: DeviceSettings) -> float:
    """How far `settings` move the classifier checkpoint's probabilities from float32."""
    model = _checkpoint("classifier")
    source = synthetic_table(rows, queries, 8, missing=0.05)
    data = Table(
        jnp.asarray(source.features),
        jnp.asarray(source.labels),
        jnp.asarray(source.queries),
        int(source.labels.max()) + 1,
    )

    def run(settings: DeviceSettings) -> jax.Array:
        configured = configure(model, settings)
        state, context = tabicl_functions.fit(
            configured, data.features, data.targets, data.num_classes
        )
        cache = store_cache(
            tabicl_functions.encode(configured, state, context, data.targets), settings
        )
        query_rows = tabicl_functions.embed(configured, state, data.queries)
        outputs = tabicl_functions.decode(configured, state, cache, query_rows)
        return tabicl_functions.postprocess(configured, state, outputs)

    reference = DeviceSettings(FLOAT32, FLOAT32, AttentionKernel())
    return float(jnp.max(jnp.abs(run(settings) - run(reference))))


def test_checkpoint_gpu_settings_stay_close_to_float32() -> None:
    # A CPU compiles no float16 products that accumulate in float32, so query
    # chunks, which compile their steps, go; they change no value.
    gpu = device_settings("gpu")
    unchunked = dataclasses.replace(gpu.kernel, query_chunk=None)

    # An L4 measured 0.015 here, and up to 0.026 on real tables at 8,192 rows.
    assert _checkpoint_drift(512, 64, dataclasses.replace(gpu, kernel=unchunked)) < 3e-2


def test_checkpoint_int8_cache_stays_close_to_float32() -> None:
    int8 = DeviceSettings(FLOAT32, jnp.dtype("int8"), AttentionKernel())

    # A CPU measured 0.027 in probability at 1,024 rows.
    assert _checkpoint_drift(1024, 32, int8) < 5e-2
