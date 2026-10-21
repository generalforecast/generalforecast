"""Ensembles: estimators on reordered, label-shifted tables, averaged as upstream."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import registry
from generalforecast.models.contract import TabularFunctions
from generalforecast.models.ensemble import Ensemble, ensemble_functions
from generalforecast.models.tabicl.model import class_probabilities, tabicl_functions
from generalforecast.models.tabpfn.model import tabpfn_functions
from tests.models.conformance.cases import table
from tests.models.tabicl.tiny import tiny_tabicl
from tests.models.tabpfn.tiny import tiny_tabpfn
from tests.utils.predict import predict


def test_estimators_see_their_own_view_and_average_back() -> None:
    model, data = tiny_tabicl(regression=False), table(regression=False)
    functions = ensemble_functions(tabicl_functions)

    def run(functions, model, features, targets, queries) -> np.ndarray:
        return np.asarray(predict(functions, model, features, targets, queries))

    single = run(tabicl_functions, model, data.features, data.targets, data.queries)
    np.testing.assert_allclose(
        run(functions, Ensemble(model, 1), data.features, data.targets, data.queries),
        single,
        atol=1e-6,
    )

    # The second estimator sees columns in its fixed order and every label
    # moved up one class; the two estimators' logits are averaged, the
    # second's columns shifted back, before the softmax.
    def logits(features: jnp.ndarray, targets: jnp.ndarray, queries: jnp.ndarray):
        state, rows = tabicl_functions.fit(model, features, targets, data.num_classes)
        cache = tabicl_functions.encode(model, state, rows, targets)
        embedded = tabicl_functions.embed(model, state, queries)
        return np.array(tabicl_functions.decode(model, state, cache, embedded))

    order = jnp.asarray(np.random.default_rng(1).permutation(data.features.shape[1]))
    plain = logits(data.features, data.targets, data.queries)
    shifted = logits(
        data.features[:, order],
        (data.targets + 1) % data.num_classes,
        data.queries[:, order],
    )
    classes = data.num_classes
    shifted[:, :classes] = np.roll(shifted[:, :classes], -1, axis=-1)
    expected = class_probabilities(jnp.asarray((plain + shifted) / 2), classes)
    np.testing.assert_allclose(
        run(functions, Ensemble(model, 2), data.features, data.targets, data.queries),
        np.asarray(expected),
        atol=1e-5,
    )
    assert isinstance(functions, TabularFunctions)


def test_unknown_classes_stay_unknown_in_every_estimator() -> None:
    from generalforecast.models.limix.model import limix_functions
    from tests.models.limix.tiny import tiny_limix

    functions = ensemble_functions(limix_functions)
    model = Ensemble(tiny_limix(regression=False), 6)
    features = jnp.asarray(np.random.default_rng(0).normal(size=(30, 5)), jnp.float32)
    targets = jnp.asarray(np.arange(30) % 3, jnp.int32).at[4].set(-1)

    state, rows = functions.fit(model, features, targets, 3)
    query = functions.embed(model, state, features[4:5])

    np.testing.assert_allclose(rows[4], query[0], atol=1e-6)


def test_probability_averaging_models_average_each_estimators_probabilities() -> None:
    model, data = tiny_tabpfn(regression=False), table(regression=False)
    order = jnp.asarray(np.random.default_rng(1).permutation(data.features.shape[1]))
    plain = predict(tabpfn_functions, model, data.features, data.targets, data.queries)
    shifted = predict(
        tabpfn_functions,
        model,
        data.features[:, order],
        (data.targets + 1) % data.num_classes,
        data.queries[:, order],
    )

    # TabPFN averages probabilities, each tempered apart, not logits.
    np.testing.assert_allclose(
        predict(
            ensemble_functions(tabpfn_functions),
            Ensemble(model, 2),
            data.features,
            data.targets,
            data.queries,
        ),
        (np.asarray(plain) + np.roll(np.asarray(shifted), -1, axis=-1)) / 2,
        atol=1e-5,
    )


def test_ensembles_report_the_width_of_their_predictions() -> None:
    functions = ensemble_functions(tabpfn_functions)
    model, data = Ensemble(tiny_tabpfn(regression=True), 3), table(regression=True)

    predictions = predict(functions, model, data.features, data.targets, data.queries)

    # Buckets in, quantiles out: a table's width is what postprocess returns.
    assert predictions.shape[-1] == functions.properties(model).num_outputs
    assert np.isfinite(np.asarray(predictions)).all()


def test_specs_ending_in_a_count_load_ensembles() -> None:
    _, model = registry.load("tests.models.tabicl.tiny:classifier*3")

    assert isinstance(model, Ensemble) and model.estimators == 3
    for spec in ("*0", "*2*2", "*2@4"):
        with pytest.raises(ValueError, match="spec"):
            registry.load(f"tests.models.tabicl.tiny:classifier{spec}")
