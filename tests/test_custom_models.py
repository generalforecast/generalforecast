# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Custom models, named `module:attribute`, on every local surface and refused elsewhere."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from generalforecast import engine
from generalforecast.engine import Labeled, LocalEngine
from generalforecast.errors import InvalidInput
from generalforecast.estimators import Classifier, Forecaster, Regressor
from generalforecast.models.catalog import ModelRef, ModelRefError, admit, parse
from generalforecast.predictors.models import CatalogResolver
from generalforecast.predictors.spec import parse_text
from generalforecast.predictors.validation import PredictorError

CUSTOM = "tests.models.custom:functions"
HINT = 'pass a custom model as "module:attribute"'
REFUSAL = (
    f"custom model `{CUSTOM}` runs locally only; predictors, gf push and serving "
    "take a catalog model id, such as general-forecast/tabicl-v2-classifier"
)


@pytest.fixture(scope="module")
def process() -> Iterator[LocalEngine]:
    """The process engine, with no catalog model stood in for."""
    tiny = LocalEngine(cpu_threads=2)

    def local(surface: Any = "local", allow_downloads: bool = False) -> LocalEngine:
        return tiny.gated(surface, allow_downloads)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("generalforecast.engine.local", local)
        patch.setattr("generalforecast.native.estimators._shared", {})
        yield tiny
    tiny.close()


def _clusters(rows: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Rows around 0 (even rows, class 0) and around 10 (odd rows, class 1)."""
    generator = np.random.default_rng(0)
    centers = np.where(np.arange(rows) % 2 == 0, 0.0, 10.0)[:, None]
    features = (centers + generator.normal(size=(rows, 5))).astype(np.float32)
    return features, (np.arange(rows) % 2).astype(np.int32)


def test_tables_run_a_custom_model(process) -> None:
    features, labels = _clusters()
    rows = pd.DataFrame(features, columns=[f"f{i}" for i in range(5)]).assign(y=labels)
    table = engine.Table("custom", model=CUSTOM, target="y", download=False)
    try:
        assert (table.model, table.task) == (CUSTOM, "classification")
        table.insert(rows)
        scored = table.predict(rows.drop(columns=["y"]).head(4))
        assert scored["label"].tolist() == [0, 1, 0, 1]
    finally:
        table.drop()


def test_classifiers_run_a_custom_model(process, tmp_path) -> None:
    features, labels = _clusters()
    estimator = Classifier(model=CUSTOM, download=False).fit(features, labels)
    try:
        probabilities = estimator.predict_proba(features[:4])
        assert probabilities.shape == (4, 2)
        np.testing.assert_array_equal(estimator.predict(features[:4]), labels[:4])
        with pytest.raises(InvalidInput) as refused:
            estimator.save(tmp_path / "custom", name="custom")
        assert refused.value.code == "UnsupportedModel"
        assert REFUSAL in refused.value.message
    finally:
        estimator.close()


def test_regressors_refuse_a_custom_classifier(process) -> None:
    features, labels = _clusters()
    with pytest.raises(InvalidInput, match="model does not support estimator task"):
        Regressor(model=CUSTOM, download=False).fit(features, labels.astype(np.float32))


def test_the_local_engine_runs_a_custom_model(process) -> None:
    features, labels = _clusters()
    local = engine.local()
    context = Labeled("classification", features, labels, (0, 1))
    bound = local.bind(parse(CUSTOM), context)
    outputs = local.predict(bound, features[:4])
    assert outputs.shape == (4, 2) and outputs.argmax(1).tolist() == [0, 1, 0, 1]


def test_series_tables_run_a_custom_forecaster_at_its_levels(process) -> None:
    rows = pd.DataFrame(
        {
            "item": ["shoes"] * 20,
            "t": list(range(20)),
            "units": np.random.default_rng(1).normal(size=20),
        }
    )
    table = engine.SeriesTable(
        "custom-series",
        model="tests.models.custom:forecaster",
        schema=engine.SeriesSchema(
            target="units", time="t", series="item", frequency=1
        ),
        horizon=3,
        download=False,
    )
    try:
        table.insert(rows)
        forecast = table.forecast()
        from tests.models.custom import LEVELS

        assert forecast.columns.tolist() == ["item", "step", "value"] + [
            f"q_{level:g}" for level in LEVELS
        ]
        np.testing.assert_array_equal(forecast["value"], forecast["q_0.5"])
    finally:
        table.drop()


def test_custom_models_run_locally_only(process) -> None:
    with pytest.raises(ModelRefError, match="runs locally only"):
        process.gated("remote", False).load(parse(CUSTOM))


def test_a_custom_reference_built_with_a_variant_is_refused() -> None:
    for ref in (ModelRef(CUSTOM, codes=2), ModelRef(CUSTOM, estimators=3)):
        with pytest.raises(ModelRefError) as caught:
            admit(ref, "local")
        assert caught.value.code == "UnsupportedCapability"


def test_a_tuple_names_no_model() -> None:
    with pytest.raises(InvalidInput) as caught:
        engine.Table("t", model=("tests.models.custom", "functions"), target="y")
    assert caught.value.message.startswith("a model spec is a string, not tuple")
    assert caught.value.hint == HINT


def test_an_estimator_refuses_a_tuple_as_it_fits(process) -> None:
    features, labels = _clusters()
    with pytest.raises(InvalidInput) as caught:
        Classifier(model=("tests.models.custom", "functions")).fit(features, labels)
    assert caught.value.message.startswith("a model spec is a string, not tuple")
    assert caught.value.hint == HINT


def test_a_bare_name_keeps_its_refusal() -> None:
    with pytest.raises(InvalidInput) as caught:
        engine.Table("t", model="tabicl-classifier", target="y")
    assert caught.value.message == (
        "`tabicl-classifier` is not a catalog model id; "
        "use general-forecast/tabicl-v2-classifier"
    )


@pytest.mark.parametrize(
    ("spec", "code"),
    [
        ("a-b:c", "InvalidModel"),
        ("x:y:z", "InvalidModel"),
        (f"{CUSTOM}@2", "UnsupportedCapability"),
        (f"{CUSTOM}*3", "UnsupportedCapability"),
    ],
)
def test_a_custom_spec_is_a_module_and_an_attribute(spec: str, code: str) -> None:
    with pytest.raises(ModelRefError) as caught:
        parse(spec)
    assert caught.value.code == code


def test_a_model_failing_the_contracts_checks_fails_to_load(process) -> None:
    features, labels = _clusters()
    with pytest.raises(InvalidInput, match="fails the contract's `masks` check"):
        Classifier(model="tests.models.custom:broken", download=False).fit(
            features, labels
        )


def test_an_attribute_that_is_no_model_fails_to_load(process) -> None:
    features, labels = _clusters()
    with pytest.raises(InvalidInput, match="is not a model"):
        Classifier(model="tests.models.custom:weights", download=False).fit(
            features, labels
        )


def test_predictors_refuse_a_custom_model() -> None:
    with pytest.raises(InvalidInput) as caught:
        parse_text(f"model: {CUSTOM}\ntarget: y\n", default_name="d")
    assert (caught.value.message, caught.value.param) == (REFUSAL, "model")
    with pytest.raises(PredictorError) as refused:
        CatalogResolver().resolve(CUSTOM)
    assert (refused.value.code, refused.value.message) == ("UnsupportedModel", REFUSAL)


def test_forecasters_take_catalog_forecasters() -> None:
    with pytest.raises(InvalidInput, match="not custom model") as caught:
        Forecaster(model=CUSTOM)
    assert caught.value.hint == "run a custom forecaster through gf.engine.SeriesTable"
