# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Derived models of kind `adapter`: bags of input adapters in front of one shared base."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import catalog, derived, registry
from generalforecast.models.adapted import Adapted, adapted_functions
from generalforecast.models.bagged import Bagged, bagged_functions
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.conformance.cases import table
from tests.models.tabicl.tiny import tiny_tabicl
from tests.utils.predict import predict

BASE = "general-forecast/tabicl-v2-classifier"
MODEL = "acme/credit-default@ft4"


def statistics(own, features, num_rows):
    """The context's column means, over its real rows and present cells."""
    real = jnp.arange(features.shape[0]) < (
        features.shape[0] if num_rows is None else num_rows
    )
    present = real[:, None] & ~jnp.isnan(features)
    total = jnp.where(present, features, 0.0).sum(axis=0)
    return total / jnp.maximum(present.sum(axis=0), 1)


def transform(own, mean, features):
    """Each column moved by its gate along a learned curve of its centered value.

    The curve is not affine: TabICL standardizes its columns, so an affine
    change of one would leave its predictions nearly alone.
    """
    centered = features - mean
    return features + own["gate"] * (own["scale"] * centered**2 + own["shift"])


INPUTS = derived.Inputs(statistics, transform)


def keep(shared, own):
    """An adapter changes no weight of the base."""
    return shared


def _adapters(width, gates, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "gate": jnp.full(width, gate, jnp.float32),
            "scale": jnp.asarray(rng.normal(size=width), jnp.float32),
            "shift": jnp.asarray(rng.normal(size=width), jnp.float32),
        }
        for gate in gates
    ]


def _stacked(parts):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *parts)


class Plane:
    """Keeps one derived model of `kind`, whose bags are `parts`."""

    def __init__(self, parts, kind="adapter", inputs=INPUTS):
        self.parts, self.kind, self.inputs = parts, kind, inputs

    def resolve(self, model):
        if model != MODEL:
            raise catalog.ModelRefError("UnknownModel", model)
        made = catalog.ENTRIES[BASE].artifact.digest
        return derived.Derived(
            model, BASE, made, "sha256:" + "3" * 64, len(self.parts), self.kind
        )

    def weights(self, made, base):
        return derived.Weights(_stacked(self.parts), keep, self.inputs)


@pytest.fixture
def base(monkeypatch):
    """The catalog base, as a tiny TabICL."""
    model = tiny_tabicl(regression=False)
    monkeypatch.setattr(
        registry, "_native", lambda entry, surface, allow: (tabicl_functions, model)
    )
    registry._served.cache_clear()
    registry._derived_variant.cache_clear()
    yield model
    registry._served.cache_clear()
    registry._derived_variant.cache_clear()
    derived.use(None)


def _run(functions, parameters, data):
    return np.asarray(
        predict(functions, parameters, data.features, data.targets, data.queries)
    )


def test_an_adapter_at_gate_zero_reproduces_the_base_exactly(base):
    data = table(regression=False)
    (part,) = _adapters(data.features.shape[1], (0.0,))
    functions = adapted_functions(tabicl_functions, INPUTS)
    # Step 0: the features the base reads are its own, to the bit, and so
    # are its predictions.
    mean = statistics(part, data.features, None)
    np.testing.assert_array_equal(transform(part, mean, data.features), data.features)
    np.testing.assert_array_equal(
        _run(functions, Adapted(base, part), data), _run(tabicl_functions, base, data)
    )


def test_bagged_adapters_at_gate_zero_serve_the_base(base):
    data = table(regression=False)
    derived.use(Plane(_adapters(data.features.shape[1], (0.0, 0.0))))
    functions, parameters = registry.load(MODEL)
    # Bags run batched, whose kernels may round apart from the base's own.
    np.testing.assert_allclose(
        _run(functions, parameters, data),
        _run(tabicl_functions, base, data),
        rtol=0,
        atol=1e-6,
    )
    assert parameters.shared is base


def test_a_trained_adapter_changes_predictions_as_its_features_say(base):
    data = table(regression=False)
    (part,) = _adapters(data.features.shape[1], (0.5,))
    derived.use(Plane([part]))
    functions, parameters = registry.load(MODEL)
    found = _run(functions, parameters, data)
    assert np.abs(found - _run(tabicl_functions, base, data)).max() > 1e-3
    # The base reads the changed features, the queries changed by the
    # context's statistics, never their own.
    mean = statistics(part, data.features, None)
    expected = predict(
        tabicl_functions,
        base,
        transform(part, mean, data.features),
        data.targets,
        transform(part, mean, data.queries),
    )
    np.testing.assert_allclose(found, np.asarray(expected), atol=1e-5)


def test_statistics_read_only_the_context_s_real_rows(base):
    data = table(regression=False)
    (part,) = _adapters(data.features.shape[1], (0.5,))
    functions = adapted_functions(tabicl_functions, INPUTS)
    model = Adapted(base, part)
    rows = data.features.shape[0]
    padded = jnp.concatenate([data.features, jnp.full_like(data.features, 1e6)])
    labels = jnp.concatenate([data.targets, jnp.zeros_like(data.targets)])
    state, _ = functions.fit(
        model, padded, labels, data.num_classes, num_rows=rows, categorical=None
    )
    np.testing.assert_allclose(
        state.statistics, statistics(part, data.features, None), rtol=1e-6
    )


def test_adapter_models_share_their_functions_and_programs(base):
    data = table(regression=False)
    width = data.features.shape[1]
    assert adapted_functions(tabicl_functions, INPUTS) is adapted_functions(
        tabicl_functions, INPUTS
    )
    from generalforecast.models.adapted import WithInputs

    functions = bagged_functions(adapted_functions(tabicl_functions, INPUTS))

    def lowered(gates, seed):
        parameters = Bagged(
            base, _stacked(_adapters(width, gates, seed)), 2, WithInputs(keep)
        )
        fit = jax.jit(lambda p, x, y: functions.fit(p, x, y, data.num_classes))
        return fit.lower(parameters, data.features, data.targets).as_text()

    assert lowered((0.0, 0.0), 0) == lowered((0.3, 0.9), 1)


def test_an_adapter_s_kind_and_weights_must_agree(base):
    data = table(regression=False)
    parts = _adapters(data.features.shape[1], (0.1,))
    derived.use(Plane(parts, kind="adapter", inputs=None))
    with pytest.raises(catalog.ModelRefError) as refused:
        registry.load(MODEL)
    assert refused.value.code == "ModelMismatch"
    registry._derived_variant.cache_clear()
    derived.use(Plane(parts, kind="lora"))
    with pytest.raises(catalog.ModelRefError) as refused:
        registry.load(MODEL)
    assert refused.value.code == "ModelMismatch"


def test_an_adapted_model_never_has_its_width_padded(base):
    model = Adapted(base, _adapters(5, (0.0,))[0])
    properties = adapted_functions(tabicl_functions, INPUTS).properties(model)
    assert properties.padded_width is False
    assert properties.regression is tabicl_functions.properties(base).regression


def test_context_pages_embed_as_the_fit_does(base):
    data = table(regression=False)
    (part,) = _adapters(data.features.shape[1], (0.5,))
    functions = adapted_functions(tabicl_functions, INPUTS)
    model = Adapted(base, part)
    state, rows = functions.fit(model, data.features, data.targets, data.num_classes)
    # The engine embeds a large context in pages against the fitted state:
    # each page is changed by the whole context's statistics.
    half = data.features.shape[0] // 2
    pages = jnp.concatenate(
        [
            functions.embed(model, state, data.features[:half], data.targets[:half]),
            functions.embed(model, state, data.features[half:], data.targets[half:]),
        ]
    )
    np.testing.assert_allclose(pages, rows, rtol=1e-5, atol=1e-6)
