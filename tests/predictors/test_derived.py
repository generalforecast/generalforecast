# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Derived models: named by `@ft<version>`, resolved by the plane, pinned by lock 2."""

from pathlib import Path

import pytest
import yaml

from generalforecast import hosting
from generalforecast.contracts import parse_lock
from generalforecast.models import catalog, derived
from generalforecast.predictors import load, load_lock, resolve, write_lock
from generalforecast.predictors.lock import DERIVED_LOCK_VERSION, document
from generalforecast.predictors.models import CatalogResolver, ModelPin
from generalforecast.predictors.validation import PredictorError
from tests.predictors.conftest import MODEL, write_manifest

FINE_TUNE = "acme/credit-default@ft3"
WEIGHTS = "sha256:" + "1" * 64


class Plane:
    """A resolver that keeps one fine-tune of the catalog classifier."""

    def __init__(self, bags=8, base_digest=None):
        self.bags = bags
        self.base_digest = base_digest or catalog.ENTRIES[MODEL].artifact.digest

    def resolve(self, model):
        if model != FINE_TUNE:
            raise catalog.ModelRefError("UnknownModel", f"no model {model}")
        return derived.Derived(model, MODEL, self.base_digest, WEIGHTS, self.bags)


@pytest.fixture
def plane():
    installed = Plane()
    hosting.use_derived(installed)
    yield installed
    hosting.use_derived(None)


@pytest.fixture
def fine_tuned(churn: Path) -> Path:
    write_manifest(
        churn,
        f"model: {FINE_TUNE}\ntarget: churned\nid: customer_id\n"
        "inputs:\n  exclude: [note]\ncontext:\n  snapshot: context.csv\n",
    )
    return churn


def test_a_derived_id_is_its_owner_name_and_version():
    ref = catalog.parse("acme/credit-default@ft3@4*2")
    assert ref.derived and ref.version == 3 and ref.codes == 4 and ref.estimators == 2
    assert ref.model == FINE_TUNE and str(ref) == "acme/credit-default@ft3@4*2"
    assert catalog.parse(f"{FINE_TUNE}@{WEIGHTS}").digest == WEIGHTS
    # Without the `ft`, digits after `@` stay error-correcting codes.
    assert catalog.parse(f"{MODEL}@3").codes == 3
    for bad in (
        f"{MODEL}@ft3",
        "acme/credit-default@ft0",
        "acme@ft3",
        "Acme/credit-default@ft3",
    ):
        with pytest.raises(catalog.ModelRefError) as refused:
            catalog.parse(bad)
        assert refused.value.code == "InvalidModel"


def test_without_a_resolver_no_derived_model_resolves():
    with pytest.raises(catalog.ModelRefError) as refused:
        catalog.admit(catalog.parse(FINE_TUNE), "predictor")
    assert refused.value.code == "UnknownModel"
    with pytest.raises(PredictorError) as unpinned:
        CatalogResolver().resolve(FINE_TUNE)
    assert unpinned.value.code == "ModelResolutionRequired"


def test_each_bag_counts_as_one_estimator(plane):
    catalog.admit(catalog.parse(FINE_TUNE), "predictor")
    with pytest.raises(catalog.ModelRefError) as refused:
        catalog.admit(catalog.parse(f"{FINE_TUNE}*2"), "predictor")
    assert refused.value.code == "CapacityExceeded"
    plane.bags = 4
    catalog.admit(catalog.parse(f"{FINE_TUNE}*2"), "predictor")


def test_a_derived_model_takes_its_bases_task_and_bounds(plane):
    assert catalog.task_of(FINE_TUNE) == "classification"
    assert catalog.limits(FINE_TUNE) == catalog.limits(MODEL)
    # A pin to other weights than the plane keeps is refused.
    with pytest.raises(catalog.ModelRefError) as refused:
        catalog.admit(catalog.parse(f"{FINE_TUNE}@sha256:{'2' * 64}"), "predictor")
    assert refused.value.code == "ModelMismatch"


def test_weights_made_against_another_base_are_refused(plane):
    plane.base_digest = "sha256:" + "9" * 64
    with pytest.raises(catalog.ModelRefError) as refused:
        catalog.admit(catalog.parse(FINE_TUNE), "predictor")
    assert refused.value.code == "ModelMismatch"


def test_a_derived_pin_describes_its_base_and_pins_its_weights(plane):
    pin = CatalogResolver().resolve(FINE_TUNE)
    base = CatalogResolver().resolve(MODEL)
    assert pin.model == FINE_TUNE and pin.model_digest == WEIGHTS
    assert pin.digest == base.digest and pin.files == base.files
    assert pin.resolved == f"{FINE_TUNE}@{WEIGHTS}"
    assert ModelPin.from_dict(pin.to_dict()) == pin
    # A catalog pin keeps its form: no `derived` key.
    assert "derived" not in base.to_dict()
    forged = pin.to_dict()
    forged["derived"]["base_digest"] = "sha256:" + "9" * 64
    with pytest.raises(PredictorError) as refused:
        ModelPin.from_dict(forged)
    assert refused.value.code == "ModelMismatch"


def test_gf_check_writes_lock_version_2_for_a_derived_model(plane, fine_tuned):
    resolution = resolve(load(fine_tuned), fine_tuned)
    write_lock(resolution, fine_tuned)
    written = yaml.safe_load((fine_tuned / "gf.lock").read_text())
    assert written["lock_version"] == DERIVED_LOCK_VERSION
    predictor = written["predictor"]
    assert predictor["model"] == FINE_TUNE and predictor["model_digest"] == WEIGHTS
    assert predictor["derived"] == {
        "base": MODEL,
        "base_digest": catalog.ENTRIES[MODEL].artifact.digest,
        "bags": 8,
    }
    frozen = load_lock(fine_tuned)
    assert frozen.derived == predictor["derived"]
    assert parse_lock(written).digest == written["digest"]


def test_a_catalog_lock_stays_version_1(churn: Path):
    value = document(resolve(load(churn), churn))
    assert value["lock_version"] == 1 and "derived" not in value["predictor"]


def test_a_derived_block_and_lock_version_2_go_together(plane, fine_tuned, churn):
    value = document(resolve(load(fine_tuned), fine_tuned))
    for change in (
        lambda d: d.update(lock_version=1),
        lambda d: d["predictor"].pop("derived"),
        lambda d: d["predictor"]["derived"].update(bags=0),
        lambda d: d["predictor"].update(model=MODEL),
    ):
        forged = yaml.safe_load(yaml.safe_dump(value))
        change(forged)
        with pytest.raises(PredictorError):
            parse_lock(forged)


def test_gf_server_runs_a_derived_model_as_its_own_worker_model(plane):
    from generalforecast.predictors.models import describe
    from generalforecast.serve.models import resolved_id, worker_model

    ref = catalog.parse(f"{FINE_TUNE}*2")
    plane.bags = 4
    # Its own pinned id, owner kept, and the worker model it loads on demand.
    assert resolved_id(ref) == f"{FINE_TUNE}@{WEIGHTS}*2"
    assert worker_model(ref) == FINE_TUNE
    assert describe(FINE_TUNE).model_digest == WEIGHTS
    catalog_ref = catalog.parse(MODEL)
    assert worker_model(catalog_ref) == MODEL
    assert resolved_id(catalog_ref).startswith(f"{MODEL}@sha256:")
