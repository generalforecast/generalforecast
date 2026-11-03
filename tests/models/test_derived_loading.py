# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Running derived models: bags of weights over one shared copy of their base."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import catalog, derived, registry
from generalforecast.models.bagged import Bagged, bagged_functions
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.conformance.cases import table
from tests.models.tabicl.tiny import tiny_tabicl
from tests.utils.predict import predict

BASE = "general-forecast/tabicl-v2-classifier"
MODEL = "acme/credit-default@ft3"


def plus(shared, own):
    """A bag's parameters: the base's with its own difference added."""
    return jax.tree.map(jnp.add, shared, own)


class Plane:
    """Keeps one derived model: bags that each move the base by a difference."""

    def __init__(self, deltas, digest="sha256:" + "1" * 64):
        self.deltas, self.digest, self.asked = deltas, digest, []

    def resolve(self, model):
        if model != MODEL:
            raise catalog.ModelRefError("UnknownModel", model)
        made = catalog.ENTRIES[BASE].artifact.digest
        return derived.Derived(model, BASE, made, self.digest, len(self.deltas))

    def weights(self, made, base):
        self.asked.append(made)
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *self.deltas)
        return derived.Weights(stacked, plus)


@pytest.fixture
def base(monkeypatch):
    """The catalog base, as a tiny TabICL, loaded once per test."""
    model = tiny_tabicl(regression=False)
    loads = []

    def native(entry, surface, allow_downloads):
        loads.append(entry.id)
        return tabicl_functions, model

    monkeypatch.setattr(registry, "_native", native)
    registry._served.cache_clear()
    registry._derived_variant.cache_clear()
    yield model, loads
    registry._served.cache_clear()
    registry._derived_variant.cache_clear()
    derived.use(None)


def _deltas(model, scales):
    rng = np.random.default_rng(0)
    return [
        jax.tree.map(
            lambda leaf, s=s: jnp.asarray(rng.normal(size=leaf.shape) * s, leaf.dtype),
            model,
        )
        for s in scales
    ]


def _run(functions, parameters, data):
    return np.asarray(
        predict(functions, parameters, data.features, data.targets, data.queries)
    )


def test_a_derived_model_runs_its_bags_over_the_base(base):
    model, _ = base
    deltas = _deltas(model, (0.01, 0.02))
    derived.use(Plane(deltas))
    functions, parameters = registry.load(MODEL)
    data = table(regression=False)
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *deltas)
    expected = _run(
        bagged_functions(tabicl_functions), Bagged(model, stacked, 2, plus), data
    )
    np.testing.assert_allclose(_run(functions, parameters, data), expected, atol=1e-6)
    # The base is held once, not once per bag.
    assert parameters.shared is model


def test_derived_models_on_one_base_share_its_copy(base):
    model, loads = base
    plane = Plane(_deltas(model, (0.01,)))
    derived.use(plane)
    registry.load(MODEL, surface="predictor")
    plane.digest = "sha256:" + "2" * 64
    registry.load(MODEL, surface="predictor")
    # Two sets of weights, by digest, over one load of the base.
    assert loads == [BASE] and len(plane.asked) == 2


def test_weights_must_stack_one_part_per_bag(base):
    model, _ = base

    class Short(Plane):
        """Says three bags, but stacks two."""

        def resolve(self, model):
            return derived.Derived(
                model, BASE, catalog.ENTRIES[BASE].artifact.digest, self.digest, 3
            )

    derived.use(Short(_deltas(model, (0.01, 0.02))))
    with pytest.raises(catalog.ModelRefError) as refused:
        registry.load(MODEL)
    assert refused.value.code == "ModelMismatch"


def test_compiled_programs_do_not_depend_on_the_weights(base):
    model, _ = base
    data = table(regression=False)
    functions = bagged_functions(tabicl_functions)

    def lowered(deltas):
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *deltas)
        parameters = Bagged(model, stacked, len(deltas), plus)
        fit = jax.jit(lambda p, x, y: functions.fit(p, x, y, data.num_classes))
        return fit.lower(parameters, data.features, data.targets).as_text()

    # Two customers' weights of the same shapes compile to the same program,
    # so the compile cache holds no weights and is shared.
    assert lowered(_deltas(model, (0.01, 0.02))) == lowered(_deltas(model, (0.5, 0.7)))


def test_workers_find_the_resolver_by_name(monkeypatch):
    derived.use(None)
    monkeypatch.setenv(derived.RESOLVER, f"{__name__}:NAMED")
    try:
        assert derived.resolver() is NAMED
    finally:
        derived.use(None)
    monkeypatch.setenv(derived.RESOLVER, "no-colon")
    with pytest.raises(ValueError, match="module:attribute"):
        derived.resolver()


NAMED = Plane([])
