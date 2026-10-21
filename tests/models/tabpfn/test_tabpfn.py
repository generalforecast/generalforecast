# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TabPFN's preprocessing, bucket quantiles and loader checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models.buckets import bucket_quantiles
from generalforecast.models.tabpfn.checkpoint import check_supported, tabpfn_from_state
from generalforecast.models.tabpfn.prepare import fit_snapshot, prepare_features
from tests.models.tabpfn.tiny import tiny_checkpoint, tiny_tabpfn


def _table() -> np.ndarray:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(12, 4)) * [1, 10, 0.1, 3]
    features[:, 3] = 5.0
    features[2, 0] = np.nan
    features[3, 1] = np.inf
    features[4, 2] = -np.inf
    return features.astype(np.float32)


def test_prepare_matches_tabpfn_standardization() -> None:
    features = _table()
    snapshot = fit_snapshot(jnp.asarray(features))
    prepared = np.asarray(prepare_features(snapshot, jnp.asarray(features)))

    finite = np.isfinite(features)
    means = np.nanmean(np.where(finite, features, np.nan), axis=0)
    filled = np.where(finite, features, means)
    deviation = filled.std(axis=0, ddof=1)
    deviation[3] = 1.0
    expected = (filled - filled.mean(axis=0)) / (deviation + np.finfo(np.float32).eps)

    np.testing.assert_allclose(prepared[..., 0], expected, rtol=1e-5, atol=1e-5)
    assert prepared[2, 0, 1] == -2 and prepared[3, 1, 1] == 2 and prepared[4, 2, 1] == 4
    assert (prepared[..., 1][finite] == 0).all()


def test_prepare_ignores_padding_rows() -> None:
    features = _table()
    padded = np.concatenate([features, np.full((4, 4), 1e6, np.float32)])

    np.testing.assert_allclose(
        np.asarray(fit_snapshot(jnp.asarray(padded), num_rows=12).scale),
        np.asarray(fit_snapshot(jnp.asarray(features)).scale),
        rtol=1e-6,
    )


def test_a_fit_without_rows_stays_finite() -> None:
    padded = jnp.ones((8, 4))
    snapshot = fit_snapshot(padded, jnp.ones(8), num_rows=0, rank_buckets=16)

    assert np.isfinite([snapshot.target_center, snapshot.target_scale]).all()
    assert np.isfinite(np.asarray(prepare_features(snapshot, padded))).all()


def test_prepare_clips_far_values() -> None:
    features = np.zeros((10, 1), np.float32)
    features[0] = 1.0
    queries = np.asarray([[1e6], [-1e6]], np.float32)

    prepared = prepare_features(
        fit_snapshot(jnp.asarray(features)), jnp.asarray(queries)
    )

    np.testing.assert_array_equal(np.asarray(prepared[:, 0, 0]), [100, -100])


def test_bucket_quantiles_are_uniform_within_buckets() -> None:
    borders = jnp.asarray([-10.0, -1.0, 0.0, 1.0, 2.0, 10.0])
    # All mass on the bucket [0, 1).
    logits = jnp.asarray([-1e4, -1e4, 0.0, -1e4, -1e4])
    levels = jnp.asarray([0.25, 0.5, 0.75])

    np.testing.assert_allclose(
        bucket_quantiles(logits, borders, levels), [0.25, 0.5, 0.75], atol=1e-6
    )


def test_bucket_quantiles_have_half_normal_tails() -> None:
    borders = jnp.asarray([-10.0, -1.0, 0.0, 1.0, 2.0, 10.0])
    logits = jnp.asarray([0.0, -1e4, -1e4, -1e4, -1e4])

    median = bucket_quantiles(logits, borders, jnp.asarray([0.5]))

    # The tail puts half its mass within the outer bucket's width of -1.
    np.testing.assert_allclose(median, [-10.0], rtol=1e-5)


def test_unsupported_settings_raise() -> None:
    with pytest.raises(NotImplementedError, match="feature_group_size"):
        check_supported({"feature_group_size": 2})


def test_head_counts_must_fit_the_weights() -> None:
    state, settings, architecture = tiny_checkpoint(regression=False)
    settings = {**settings, "icl_num_heads": 3}

    with pytest.raises(ValueError, match=r"icl\.blocks\[0\]\.attention has 3 heads"):
        tabpfn_from_state(state, settings, architecture)


def test_classifier_reads_every_parameter() -> None:
    model = tiny_tabpfn(regression=False)

    assert model.num_buckets is None
    assert model.icl.query_kv_heads == 1


def test_a_cell_projection_bias_in_a_checkpoint_is_applied() -> None:
    from generalforecast.models.conformance import predict, table
    from generalforecast.models.tabpfn.model import tabpfn_functions

    state, settings, architecture = tiny_checkpoint(regression=False, release="3.5")
    width = state["x_embed.metadata_linear.weight"].shape[0]
    biased = dict(state)
    biased["x_embed.metadata_linear.bias"] = np.linspace(-1, 1, width, dtype=np.float32)
    data = table(regression=False)

    plain = predict(
        tabpfn_functions, tabpfn_from_state(state, settings, architecture), data
    )
    shifted = predict(
        tabpfn_functions, tabpfn_from_state(biased, settings, architecture), data
    )

    assert not np.allclose(plain, shifted)


def _midranks(context: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each value's midrank among the context, held at the extremes' midranks."""
    below = (context[None, :] < values[:, None]).sum(axis=1)
    at_most = (context[None, :] <= values[:, None]).sum(axis=1)
    lowest = ((context < context.min()).sum() + (context <= context.min()).sum()) / 2
    highest = ((context < context.max()).sum() + (context <= context.max()).sum()) / 2
    return np.clip((below + at_most) / 2, lowest, highest) / len(context)


def test_ranks_are_exact_midranks_with_enough_buckets() -> None:
    generator = np.random.default_rng(1)
    context = np.round(generator.normal(size=(30, 1)) * 2).astype(np.float32)
    queries = np.concatenate([context[:10], [[-50.0], [50.0], [0.25]]]).astype(
        np.float32
    )
    padded = np.concatenate([context, np.full((6, 1), 1e9, np.float32)])

    snapshot = fit_snapshot(jnp.asarray(padded), num_rows=30, rank_buckets=64)
    prepared = prepare_features(snapshot, jnp.asarray(queries))

    # Between edges, ranks interpolate; every query here but 0.25 is an edge or
    # outside the context.
    np.testing.assert_allclose(
        np.asarray(prepared[:12, 0, 2]),
        _midranks(context[:, 0], queries[:12, 0]),
        atol=1e-6,
    )


def test_ranks_are_exact_on_edges_and_rise_between_them() -> None:
    context = np.arange(100, dtype=np.float32)[:, None]
    snapshot = fit_snapshot(jnp.asarray(context), rank_buckets=10)

    ranks = np.asarray(prepare_features(snapshot, jnp.asarray(context))[:, 0, 2])

    edges = np.round(np.arange(10) * 99 / 9).astype(int)
    np.testing.assert_allclose(ranks[edges], (edges + 0.5) / 100, atol=1e-6)
    assert (np.diff(ranks) > 0).all()


def test_multitask_states_must_keep_one_task() -> None:
    state, settings, architecture = tiny_checkpoint(
        regression=True, release="3.5", both_tasks=True
    )

    with pytest.raises(ValueError, match="both tasks"):
        tabpfn_from_state(state, settings, architecture)


def test_regressors_check_their_bucket_borders() -> None:
    state, settings, architecture = tiny_checkpoint(regression=True)
    state["regression_borders"] = state["regression_borders"] * 2

    with pytest.raises(ValueError, match="borders"):
        tabpfn_from_state(state, settings, architecture)


def test_rank_buckets_do_not_grow_with_the_context() -> None:
    small = fit_snapshot(jnp.zeros((5, 2)), rank_buckets=64)
    large = fit_snapshot(jnp.zeros((500, 2)), rank_buckets=64)

    assert small.ranks is not None and large.ranks is not None
    assert small.ranks.edges.shape == large.ranks.edges.shape == (2, 64)


def test_regressors_temper_their_logits_as_tabpfn_does() -> None:
    from generalforecast.models.tabpfn.buckets import regression_borders
    from generalforecast.models.tabpfn.model import tabpfn_functions
    from generalforecast.models.targets import unscale_quantiles

    model = tiny_tabpfn(regression=True)
    buckets = model.num_buckets
    assert buckets is not None
    features = jnp.asarray(np.random.default_rng(0).normal(size=(16, 3)), jnp.float32)
    state, _ = tabpfn_functions.fit(model, features, features[:, 0], 0)
    outputs = jnp.asarray(
        np.random.default_rng(1).normal(size=(2, buckets)), jnp.float32
    )

    borders = jnp.asarray(regression_borders(buckets))
    expected = unscale_quantiles(
        state.snapshot, bucket_quantiles(outputs / model.temperature, borders)
    )
    np.testing.assert_allclose(
        tabpfn_functions.postprocess(model, state, outputs), expected, rtol=1e-6
    )
    assert tabpfn_functions.properties(model).averaging.temperature == model.temperature
