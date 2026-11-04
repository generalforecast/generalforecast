# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full fine-tuning: component maps, the `full/1` format and serving it as a derived model."""

import io
import json
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import catalog, derived, full, registry
from generalforecast.models.bagged import Bagged, bagged_functions
from generalforecast.models.tabicl.model import tabicl_functions
from tests.models.conformance.cases import table
from tests.models.kumo.tiny import tiny_kumo
from tests.models.mitra.tiny import tiny_mitra
from tests.models.tabicl.tiny import tiny_tabicl
from tests.models.tabpfn.tiny import tiny_tabpfn
from tests.utils.predict import predict

BASE = "general-forecast/tabicl-v2-classifier"
MODEL = "acme/credit-default@ft4"
FAMILIES = {
    "tabicl-classifier": lambda: tiny_tabicl(regression=False),
    "tabicl-regressor": lambda: tiny_tabicl(regression=True),
    "tabpfn-classifier": lambda: tiny_tabpfn(regression=False),
    "kumo-classifier": lambda: tiny_kumo(regression=False),
    "kumo-regressor": lambda: tiny_kumo(regression=True),
    "mitra-classifier": lambda: tiny_mitra(regression=False),
    "mitra-regressor": lambda: tiny_mitra(regression=True),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_weight_is_in_one_component(family):
    model = FAMILIES[family]()
    owners = full.component_of(model)
    # Every component holds weights; only the family's fixed values hold none.
    assert set(owners.values()) - {None} == set(full.COMPONENTS)
    fixed = sorted(path for path, owner in owners.items() if owner is None)
    assert fixed == sorted(full.FIXED.get(full.family(model), ()))


def test_kumo_trains_no_fourier_frequencies():
    model = tiny_kumo(regression=False)
    trained = full.trainable(model, full.COMPONENTS)
    assert ".rows.cells.frequencies" not in trained
    assert ".rows.cells.categorical_frequencies" not in trained
    assert ".rows.cells.linear.weight" in trained


def test_components_select_their_own_leaves():
    model = tiny_mitra(regression=False)
    head = full.trainable(model, ["prediction_head"])
    assert sorted(head) == [
        ".decoder.bias",
        ".decoder.weight",
        ".final_norm.bias",
        ".final_norm.weight",
    ]
    rows = full.trainable(model, ["row_encoder"])
    assert rows and all(".within_row." in path for path in rows)


def test_a_family_without_a_map_or_an_unknown_component_is_refused():
    with pytest.raises(ValueError, match="no component map"):
        full.trainable({"weight": jnp.ones(3)}, ["prediction_head"])
    with pytest.raises(ValueError, match="no component decoder"):
        full.trainable(tiny_tabicl(regression=False), ["decoder"])


def _moved(leaves, scale, seed=0):
    rng = np.random.default_rng(seed)
    return {
        path: leaf + jnp.asarray(rng.normal(size=leaf.shape) * scale, leaf.dtype)
        for path, leaf in leaves.items()
    }


def test_the_untrained_leaves_stay_the_bases_own():
    model = tiny_tabicl(regression=False)
    trained = _moved(full.trainable(model, ["prediction_head"]), 0.1)
    placed = full.combine(model, trained)
    assert placed.rows.cls_tokens is model.rows.cls_tokens
    np.testing.assert_array_equal(
        placed.icl.decoder.output.weight, trained[".icl.decoder.output.weight"]
    )


def test_packed_weights_hold_the_trained_leaves_as_deltas():
    model = tiny_tabicl(regression=False)
    components = ["in_context_transformer", "prediction_head"]
    start = full.trainable(model, components)
    body = full.pack(model, [start], components=components)
    stored = np.load(io.BytesIO(body), allow_pickle=False)
    meta = json.loads(bytes(stored["meta"]).decode())
    assert meta["format"] == "full/1" and meta["bags"] == 1
    assert meta["components"] == sorted(components)
    assert meta["paths"] == sorted(start)
    # The base's own leaves are zero deltas, which deflate to almost nothing.
    assert all(not np.any(stored[f"d{i}"]) for i in range(len(meta["paths"])))
    raw = sum(int(np.asarray(leaf).nbytes) for leaf in start.values())
    assert len(body) < raw / 4
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        assert {info.compress_type for info in archive.infolist()} == {
            zipfile.ZIP_DEFLATED
        }


def test_unpacking_restores_the_trained_values():
    model = tiny_tabicl(regression=True)
    trained = _moved(full.trainable(model, ["column_encoder"]), 0.01)
    weights = full.unpack(
        full.pack(model, [trained], components=["column_encoder"]), model, 1
    )
    assert weights.combine is full.combine
    for path, value in trained.items():
        np.testing.assert_allclose(weights.own[path][0], value, rtol=1e-6, atol=1e-7)


def test_pack_refuses_leaves_of_other_components():
    model = tiny_tabicl(regression=False)
    trained = full.trainable(model, ["prediction_head", "row_encoder"])
    with pytest.raises(ValueError, match="every bag"):
        full.pack(model, [trained], components=["prediction_head"])


def test_unpack_refuses_what_does_not_fit_the_base():
    model = tiny_tabicl(regression=False)
    body = full.pack(
        model,
        [full.trainable(model, ["prediction_head"])],
        components=["prediction_head"],
    )
    for weights, bags, message in (
        (b"not a zip", 1, "not a full/1"),
        (body, 2, "stack 1 bags"),
    ):
        with pytest.raises(catalog.ModelRefError, match=message) as refused:
            full.unpack(weights, model, bags)
        assert refused.value.code == "ModelMismatch"
    # Another family's base has none of these leaves.
    with pytest.raises(catalog.ModelRefError) as refused:
        full.unpack(body, tiny_mitra(regression=False), 1)
    assert refused.value.code == "ModelMismatch"


class Plane:
    """Keeps one full fine-tune of the base: trained leaves, by path."""

    def __init__(self, trained, components, digest="sha256:" + "4" * 64):
        self.trained, self.components, self.digest = trained, components, digest

    def resolve(self, model):
        if model != MODEL:
            raise catalog.ModelRefError("UnknownModel", model)
        made = catalog.ENTRIES[BASE].artifact.digest
        return derived.Derived(model, BASE, made, self.digest, 1, "full")

    def weights(self, made, base):
        body = full.pack(base, [self.trained], components=self.components)
        return full.unpack(body, base, made.bags)


@pytest.fixture
def base(monkeypatch):
    """The catalog base, as a tiny TabICL classifier."""
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


def test_a_fine_tune_of_the_bases_own_weights_predicts_as_the_base(base):
    data = table(regression=False)
    components = list(full.COMPONENTS)
    derived.use(Plane(full.trainable(base, components), components))
    functions, parameters = registry.load(MODEL)
    assert parameters.shared is base
    np.testing.assert_allclose(
        _run(functions, parameters, data),
        _run(tabicl_functions, base, data),
        atol=1e-6,
    )


def test_a_fine_tune_predicts_with_its_trained_components(base):
    data = table(regression=False)
    trained = _moved(full.trainable(base, ["prediction_head"]), 0.5)
    derived.use(Plane(trained, ["prediction_head"]))
    served = _run(*registry.load(MODEL), data)
    expected = _run(tabicl_functions, full.combine(base, trained), data)
    np.testing.assert_allclose(served, expected, atol=1e-5)
    assert not np.allclose(served, _run(tabicl_functions, base, data), atol=1e-3)


def test_full_fine_tunes_share_compiled_programs(base):
    data = table(regression=False)
    functions = bagged_functions(tabicl_functions)
    leaves = full.trainable(base, ["row_encoder"])

    def lowered(scale):
        own = jax.tree.map(lambda x: x[None], _moved(leaves, scale))
        parameters = Bagged(base, own, 1, full.combine)
        fit = jax.jit(lambda p, x, y: functions.fit(p, x, y, data.num_classes))
        return fit.lower(parameters, data.features, data.targets).as_text()

    assert lowered(0.01) == lowered(0.3)
