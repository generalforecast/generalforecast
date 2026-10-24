# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every model's outputs against its upstream PyTorch implementation, on real weights.

Both sides get the same prepared table, or the same group of series, and the
same checkpoint, and run one estimator without preprocessing, so differences
come from the port alone. The tables hold columns constant in the context,
which upstream drops with its own preprocessing step and generalforecast leaves out
without changing the table's width. TabICL's and Kumo Tabular's
preprocessing are compared on their own, on raw columns, and Kumo Tabular's
categorical entry runs each side's preprocessing and network together on raw
category codes. Needs the reference group:
`uv run --group reference pytest tests/models/reference`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from tests.utils.synthetic_table import synthetic_table


class Prepared(NamedTuple):
    """Standardized context rows then query rows, context targets, categorical
    columns, and context rows whose targets are not known.

    Categorical columns hold raw category codes.
    """

    table: np.ndarray
    targets: np.ndarray
    context_size: int
    categorical: np.ndarray
    masked: tuple[int, ...] = ()


class Reference(NamedTuple):
    """How to run one checkpoint upstream and in generalforecast on a prepared table."""

    name: str
    regression: bool
    upstream: Callable[[Prepared], np.ndarray]
    ours: Callable[[Prepared], np.ndarray]
    # Whether the network handles missing values itself, so the table keeps some.
    missing: bool = False
    # Columns the network embeds as categories, for models that can.
    categorical: tuple[int, ...] = ()
    # Largest difference allowed, near the network's own float32 rounding.
    tolerance: float = 1e-4


def _prepared(
    regression: bool, missing: bool, categorical: tuple[int, ...] = ()
) -> Prepared:
    data = synthetic_table(160, 32, 7, seed=3, missing=0.05 if missing else 0.0)
    table = np.concatenate([data.features, data.queries]).astype(np.float32)
    table = (table - np.nanmean(table, axis=0)) / np.nanstd(table, axis=0)
    # Columns constant in the context: zeros, and with missing values an
    # all-NaN column and one of NaN and a single value, which TabPFN, Kumo
    # and LimiX keep and TabICL and Mitra, imputing first, drop.
    table = np.insert(table, 1, 0.0, axis=1)
    if missing:
        table = np.insert(table, 4, np.nan, axis=1)
        mixed = np.where(np.arange(len(table)) % 3 == 0, np.nan, 0.5)
        table = np.insert(table, 6, mixed, axis=1)
    if regression:
        targets = (data.targets - data.targets.mean()) / data.targets.std()
    else:
        targets = data.labels
    mask = np.zeros(table.shape[1], dtype=bool)
    mask[list(categorical)] = True
    context_size = data.features.shape[0]
    if categorical:
        # Codes with gaps, then queries with codes the context lacks, and missing.
        generator = np.random.default_rng(4)
        codes = np.array([0, 2, 3, 7, 11, 12, 5, 20], dtype=np.float32)
        drawn = generator.integers(0, 6, (len(table), len(categorical)))
        unseen = generator.random(drawn.shape) < 0.2
        unseen[:context_size] = False
        drawn[unseen] = generator.integers(6, 8, unseen.sum())
        values = codes[drawn]
        values[generator.random(values.shape) < 0.1] = np.nan
        table[:, list(categorical)] = values
    return Prepared(table, targets.astype(np.float32), context_size, mask)


TABICL = ("jingang/TabICL", "4dcd344ece2c00be9e831fdd35bed57b5ad83e19")


def _tabicl_checkpoint(filename: str) -> dict[str, Any]:
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TABICL[0], filename, revision=TABICL[1])
    return torch.load(path, map_location="cpu", weights_only=False)


def _tabicl_upstream(filename: str) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        import torch
        from tabicl._model.inference_config import InferenceConfig
        from tabicl._model.tabicl import TabICL
        from tabicl._sklearn.preprocessing import UniqueFeatureFilter

        context = prepared.table[: prepared.context_size]
        table = UniqueFeatureFilter().fit(context).transform(prepared.table)
        checkpoint = _tabicl_checkpoint(filename)
        model = TabICL(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        config = InferenceConfig()
        for stage in (config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG):
            stage.update({"device": "cpu", "use_amp": False, "use_fa3": False})
        with torch.no_grad():
            outputs = model.eval()(
                torch.from_numpy(table.astype(np.float32))[None],
                torch.from_numpy(prepared.targets)[None],
                return_logits=True,
                inference_config=config,
            )
        return outputs[0].numpy()

    return run


def _tabicl_ours(filename: str) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from generalforecast.models.tabicl.checkpoint import tabicl_from_state
        from generalforecast.models.tabicl.model import tabicl_outputs

        checkpoint = _tabicl_checkpoint(filename)
        state = {
            key: jnp.asarray(value.float().numpy())
            for key, value in checkpoint["state_dict"].items()
        }
        model = tabicl_from_state(state, checkpoint["config"])
        targets = jnp.asarray(prepared.targets)
        if not model.regression:
            targets = targets.astype(jnp.int32)
        outputs = tabicl_outputs(
            model,
            jnp.asarray(prepared.table)[None],
            targets[None],
            prepared.context_size,
        )
        return np.asarray(outputs[0])

    return run


def _kumo_upstream(
    size: str, regression: bool, recipe: bool = False
) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from tests.models.reference import kumo

        outputs = kumo.upstream_recipe_outputs if recipe else kumo.upstream_outputs
        return outputs(size, regression, prepared)

    return run


def _kumo_ours(
    size: str, regression: bool, recipe: bool = False
) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from tests.models.reference import kumo

        outputs = kumo.our_recipe_outputs if recipe else kumo.our_outputs
        return outputs(size, regression, prepared)

    return run


def _tabpfn_upstream(name: str, regression: bool) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from tests.models.reference.prior_labs import upstream_outputs

        return upstream_outputs(name, regression, prepared)

    return run


def _tabpfn_ours(name: str, regression: bool) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from tests.models.reference.prior_labs import our_outputs

        return our_outputs(name, regression, prepared)

    return run


def _limix_upstream(regression: bool) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from tests.models.reference.limix import upstream_outputs

        return upstream_outputs(regression, prepared)

    return run


def _limix_ours(regression: bool) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from tests.models.reference.limix import our_outputs

        return our_outputs(regression, prepared)

    return run


def _mitra_upstream(name: str, regression: bool) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from tests.models.reference.mitra import upstream_outputs

        return upstream_outputs(name, regression, prepared)

    return run


def _mitra_ours(name: str, regression: bool) -> Callable[[Prepared], np.ndarray]:
    def run(prepared: Prepared) -> np.ndarray:
        from tests.models.reference.mitra import our_outputs

        return our_outputs(name, regression, prepared)

    return run


REFERENCES = [
    Reference(
        "general-forecast/tabicl-v2-classifier",
        False,
        _tabicl_upstream("tabicl-classifier-v2-20260212.ckpt"),
        _tabicl_ours("tabicl-classifier-v2-20260212.ckpt"),
    ),
    Reference(
        "tabicl-v2-regressor",
        True,
        _tabicl_upstream("tabicl-regressor-v2-20260212.ckpt"),
        _tabicl_ours("tabicl-regressor-v2-20260212.ckpt"),
    ),
    Reference(
        "kumo-small-classifier",
        False,
        _kumo_upstream("small", False),
        _kumo_ours("small", False),
        missing=True,
    ),
    Reference(
        "kumo-small-regressor",
        True,
        _kumo_upstream("small", True),
        _kumo_ours("small", True),
        missing=True,
    ),
    # Through each side's own preprocessing, on category codes with gaps,
    # unseen codes and missing values, behind a column constant in the context.
    # Categorical columns sit between numerical ones, and upstream moves them
    # after the numerical ones, which changes the outputs.
    Reference(
        "kumo-small-classifier-categorical",
        False,
        _kumo_upstream("small", False, recipe=True),
        _kumo_ours("small", False, recipe=True),
        missing=True,
        categorical=(0, 3, 5),
    ),
    # Medium reads 2 of 8 key/value heads for queries, which small does not.
    Reference(
        "kumo-medium-classifier",
        False,
        _kumo_upstream("medium", False),
        _kumo_ours("medium", False),
        missing=True,
    ),
    *(
        Reference(
            name,
            name.endswith("regressor"),
            _tabpfn_upstream(name, name.endswith("regressor")),
            _tabpfn_ours(name, name.endswith("regressor")),
            missing=True,
        )
        for name in (
            "general-forecast/tabpfn-3-classifier",
            "general-forecast/tabpfn-3-regressor",
            "general-forecast/tabpfn-3.5-classifier",
            "general-forecast/tabpfn-3.5-regressor",
        )
    ),
    # Through 24 layers, float32 rounding alone moves LimiX's logits by up to
    # 3e-3 from a float64 run, upstream's as much as generalforecast's.
    *(
        Reference(
            f"general-forecast/limix-2-{task}",
            task == "regressor",
            _limix_upstream(task == "regressor"),
            _limix_ours(task == "regressor"),
            missing=True,
            tolerance=1e-2,
        )
        for task in ("classifier", "regressor")
    ),
    # Float32 rounding alone moves Mitra-2's 1,000 regression logits by up to
    # 5e-4 from a float64 run, upstream's as much as generalforecast's.
    *(
        Reference(
            name,
            name.endswith("regressor"),
            _mitra_upstream(name, name.endswith("regressor")),
            _mitra_ours(name, name.endswith("regressor")),
            missing=True,
            tolerance=1e-3 if name.endswith("regressor") else 1e-4,
        )
        for name in (
            "general-forecast/mitra-classifier",
            "general-forecast/mitra-2-classifier",
            "general-forecast/mitra-2-regressor",
        )
    ),
]


class SeriesReference(NamedTuple):
    """How to forecast one group upstream and in generalforecast, on one checkpoint."""

    name: str
    horizon: int
    upstream: Callable[[int], np.ndarray]
    ours: Callable[[int], np.ndarray]
    tolerance: float = 1e-4


def _chronos2_upstream(case: str) -> Callable[[int], np.ndarray]:
    def run(horizon: int) -> np.ndarray:
        from tests.models.reference.chronos2 import upstream_forecast

        return upstream_forecast(case, horizon)

    return run


def _chronos2_ours(case: str) -> Callable[[int], np.ndarray]:
    def run(horizon: int) -> np.ndarray:
        from tests.models.reference.chronos2 import our_forecast

        return our_forecast(case, horizon)

    return run


def _timestral_upstream(case: str) -> Callable[[int], np.ndarray]:
    def run(horizon: int) -> np.ndarray:
        from tests.models.reference.mistral import upstream_forecast

        return upstream_forecast(case, horizon)

    return run


def _timestral_ours(case: str) -> Callable[[int], np.ndarray]:
    def run(horizon: int) -> np.ndarray:
        from tests.models.reference.mistral import our_forecast

        return our_forecast(case, horizon)

    return run


def _timesfm_upstream(case: str) -> Callable[[int], np.ndarray]:
    def run(horizon: int) -> np.ndarray:
        from tests.models.reference.google import upstream_forecast

        return upstream_forecast(case, horizon)

    return run


def _timesfm_ours(case: str) -> Callable[[int], np.ndarray]:
    def run(horizon: int) -> np.ndarray:
        from tests.models.reference.google import our_forecast

        return our_forecast(case, horizon)

    return run


SERIES_REFERENCES = [
    *(
        SeriesReference(
            f"chronos-2-{case}-{horizon}",
            horizon,
            _chronos2_upstream(case),
            _chronos2_ours(case),
        )
        for case in ("covariates", "univariate")
        for horizon in (96, 1100)
    ),
    *(
        SeriesReference(
            f"timestral-{case}-{horizon}",
            horizon,
            _timestral_upstream(case),
            _timestral_ours(case),
        )
        for case in ("covariates", "plain", "univariate")
        for horizon in (96, 1100)
    ),
    *(
        SeriesReference(
            f"timesfm-3-{case}-{horizon}",
            horizon,
            _timesfm_upstream(case),
            _timesfm_ours(case),
        )
        for case in ("covariates", "univariate")
        for horizon in (96, 1100)
    ),
]


@pytest.mark.parametrize("reference", REFERENCES, ids=[r.name for r in REFERENCES])
def test_outputs_match_upstream(reference: Reference) -> None:
    prepared = _prepared(reference.regression, reference.missing, reference.categorical)

    ours = reference.ours(prepared)
    upstream = reference.upstream(prepared)
    if not reference.regression:
        # Upstream may return only the classes present in the context.
        ours = ours[:, : upstream.shape[-1]]

    assert ours.shape == upstream.shape
    np.testing.assert_allclose(
        ours, upstream, rtol=reference.tolerance, atol=reference.tolerance
    )


@pytest.mark.parametrize(
    "reference", SERIES_REFERENCES, ids=[r.name for r in SERIES_REFERENCES]
)
def test_forecasts_match_upstream(reference: SeriesReference) -> None:
    ours = reference.ours(reference.horizon)
    upstream = reference.upstream(reference.horizon)

    assert ours.shape == upstream.shape
    np.testing.assert_allclose(
        ours, upstream, rtol=reference.tolerance, atol=reference.tolerance
    )


MASKED = [r for r in REFERENCES if r.name.startswith(("limix", "mitra"))]


@pytest.mark.parametrize("reference", MASKED, ids=[r.name for r in MASKED])
def test_unlabeled_context_rows_take_the_mask_as_upstream_puts_it(
    reference: Reference,
) -> None:
    """Upstream masks query rows alone; a context row is masked here as it would be."""
    prepared = _prepared(
        reference.regression, reference.missing, reference.categorical
    )._replace(masked=(3, 17))

    ours = reference.ours(prepared)
    upstream = reference.upstream(prepared)
    if not reference.regression:
        ours = ours[:, : upstream.shape[-1]]

    # Twice the absolute tolerance, for a near-zero logit's float32 rounding.
    np.testing.assert_allclose(
        ours, upstream, rtol=reference.tolerance, atol=2 * reference.tolerance
    )


def test_tabicl_preprocessing_matches_upstreams_on_raw_columns() -> None:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from tabicl._sklearn.preprocessing import (
        PreprocessingPipeline,
        UniqueFeatureFilter,
    )

    from generalforecast.models.tabicl.prepare import (
        apply_numeric_snapshot,
        fit_numeric_snapshot,
    )

    rows = _raw_columns()
    context = rows[:160]

    upstream = make_pipeline(
        SimpleImputer(strategy="mean", keep_empty_features=True),
        UniqueFeatureFilter(),
        PreprocessingPipeline(normalization_method="none"),
    ).fit(context)
    expected = upstream.transform(rows)
    snapshot = fit_numeric_snapshot(jnp.asarray(context, jnp.float32))
    ours = np.asarray(apply_numeric_snapshot(snapshot, jnp.asarray(rows, jnp.float32)))

    # Imputed, the columns of NaN and one value are constant too; the kept
    # columns come first.
    assert int(snapshot.columns.count) == expected.shape[1] == 5
    np.testing.assert_allclose(
        ours[:, : expected.shape[1]], expected, rtol=PREPARED, atol=PREPARED
    )


# Largest difference allowed between prepared inputs, float32's own rounding
# of values standardized in float64.
PREPARED = 1e-6


def _raw_columns() -> np.ndarray:
    """Heavy tails, a far outlier, missing values, near-constant and constant columns.

    The near-constant columns are 1e6 with a few float32 steps of noise, and
    1 with a few rows just below it, as in OpenML's nomao. One float32 step
    of their mean moves their standardized values by up to half a deviation,
    so they match upstream's float64 only from a mean that keeps what
    rounding it to float32 leaves out. The constant columns: 7 and 0.1 with
    missing values, zeros, and all NaN. The 0.1 column reaches upstream as
    float64, whose mean imputation gives 0.1 back exactly; in float32 it
    would not.
    """
    generator = np.random.default_rng(5)
    rows = np.column_stack(
        [
            generator.standard_cauchy(200),
            generator.lognormal(3.0, 2.0, 200),
            np.full(200, 7.0),
            generator.normal(1e3, 1.0, 200),
            generator.normal(1e6, 0.1, 200),
            np.where(generator.random(200) < 0.04, 0.999758, 1.0),
        ]
    )
    rows[3, 0] = 1e9
    rows[generator.random(rows.shape) < 0.05] = np.nan
    tenth = np.where(generator.random(200) < 0.3, np.nan, 0.1)
    rows = np.column_stack([rows, tenth, np.zeros(200), np.full(200, np.nan)])
    # Tables hold float32, so upstream gets the same values.
    return rows.astype(np.float32).astype(np.float64)


def test_kumo_preprocessing_matches_upstreams_on_raw_columns() -> None:
    import sdm.processing as sp
    import torch
    from sdm import Stype, TableTensor

    from generalforecast.models.kumo.prepare import fit_snapshot, prepare_features

    rows = _raw_columns()
    context = rows[:160]

    def tensor(values: np.ndarray) -> TableTensor:
        names = [f"column{index}" for index in range(values.shape[1])]
        return TableTensor(
            columns={Stype.numerical: names}, numerical=torch.from_numpy(values)
        )

    # The first estimator's steps of the default recipe, whose choice for it
    # is the identity.
    pipeline = sp.Sequential(
        sp.DropConstantColumns(),
        sp.Standardize(eps=1e-6),
        sp.Clip(-100.0, 100.0),
        sp.ClipSigma(threshold=4.0),
    )
    pipeline.fit(tensor(context))
    expected = pipeline.transform(tensor(rows)).numerical.numpy()
    valid = jnp.ones(len(context), dtype=bool)
    snapshot = fit_snapshot(jnp.asarray(context, jnp.float32), None, valid)
    ours = np.asarray(prepare_features(snapshot, jnp.asarray(rows, jnp.float32)))

    # NaN counts as a value, so the columns of NaN and one value stay; the
    # kept columns come first.
    assert int(snapshot.columns.count) == expected.shape[1] == 7
    np.testing.assert_allclose(
        ours[:, : expected.shape[1]], expected, rtol=PREPARED, atol=PREPARED
    )
