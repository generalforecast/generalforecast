# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from generalforecast.contracts import LOCK_VERSION, parse_lock
from generalforecast.errors import InvalidInput
from generalforecast.predictors import load, load_lock, resolve, write_lock
from generalforecast.predictors.lock import (
    changes,
    document,
    lock_digest,
    validate,
)
from generalforecast.predictors.validation import PredictorError
from tests.predictors.conftest import MODEL, write_context, write_manifest


def test_lock_shape(churn: Path):
    resolution = resolve(load(churn), churn)
    path = write_lock(resolution, churn)
    value = yaml.safe_load(path.read_text())
    assert value["lock_version"] == LOCK_VERSION == 1
    assert list(value) == ["lock_version", "digest", "manifest", "predictor", "model"]
    assert value["digest"] == lock_digest(value)
    predictor = value["predictor"]
    assert list(predictor) == [
        "name",
        "model",
        "model_digest",
        "task",
        "target",
        "target_type",
        "classes",
        "id",
        "id_type",
        "outputs",
        "inputs",
        "context",
        "selection",
    ]
    assert predictor["model"] == MODEL
    assert predictor["model_digest"] == resolution.model_digest
    assert predictor["outputs"] == ["label", "probabilities"]
    assert predictor["inputs"][0] == {"name": "spend", "type": "float", "role": "input"}
    assert predictor["context"] == {
        "kind": "snapshot",
        "path": "context.csv",
        "digest": resolution.context.digest,
        "rows": 180,
        "query_rows": 20,
    }
    assert predictor["selection"] == {
        "strategy": "exact",
        "core": None,
        "neighbors": 0,
        "kind": "stratified",
        "recent": None,
        "seed": 0,
    }
    assert set(value["model"]["files"]) == {
        "LICENSE",
        "config.json",
        "model.safetensors",
    }
    assert parse_lock(value).digest == value["digest"]


def test_lock_changes_against_previous(churn: Path):
    first = resolve(load(churn), churn)
    write_lock(first, churn)
    write_manifest(
        churn,
        f"model: {MODEL}\ntarget: churned\nid: customer_id\n"
        "inputs: {exclude: [note, signups], types: {spend: int}}\n"
        "context: {snapshot: context.csv, holdout: 0.5}\n",
    )
    write_context(churn, empty_every=0)
    second = resolve(load(churn), churn)
    assert ("-", "input", "signups", None) in second.lock_changes
    assert ("~", "input", "spend float", "spend int") in second.lock_changes
    assert ("~", "context.rows", 180, 100) in second.lock_changes
    assert ("~", "context.query_rows", 20, 100) in second.lock_changes
    assert any(key == "context.digest" for _, key, _, _ in second.lock_changes)
    assert all(op in "+-~" for op, *_ in second.lock_changes)
    write_lock(second, churn)
    assert resolve(load(churn), churn).lock_changes == ()


def test_every_setting_that_changes_predictions_changes_the_lock(churn: Path):
    write_lock(resolve(load(churn), churn), churn)
    base = (
        f"model: {MODEL}\ntarget: churned\nid: customer_id\n"
        "inputs: {exclude: [note]}\ncontext: {snapshot: context.csv}\n"
    )
    for extra, key in (
        ("selection: {seed: 9}\n", "selection.seed"),
        ("selection: {kind: random}\n", "selection.kind"),
        ("resources: {accelerator: L40S}\n", "resources.accelerator"),
    ):
        write_manifest(churn, base + extra)
        found = resolve(load(churn), churn).lock_changes
        assert any(change[1] == key for change in found), (extra, found)
        with pytest.raises(InvalidInput) as caught:
            resolve(load(churn), churn, locked=True)
        assert caught.value.message == "gf.lock is out of date"


def test_changes_table():
    base = document(_resolution())
    assert changes(None, base) == ()
    assert changes(base, base) == ()
    other = yaml.safe_load(yaml.safe_dump(base))
    other["predictor"]["selection"] = {
        "strategy": "retrieved",
        "core": 1000,
        "neighbors": 3000,
        "kind": "stratified",
        "recent": None,
        "seed": 0,
    }
    other["predictor"]["inputs"].append(
        {"name": "signup_source", "type": "category", "role": "input"}
    )
    other["predictor"]["model_digest"] = "sha256:" + "0" * 64
    other["predictor"]["id"] = "customer_id"
    result = changes(base, other)
    assert (
        "~",
        "model_digest",
        base["predictor"]["model_digest"],
        "sha256:" + "0" * 64,
    ) in result
    assert ("+", "id", None, "customer_id") in result
    assert ("~", "selection.core", None, 1000) in result
    assert ("~", "selection.neighbors", 0, 3000) in result
    assert ("+", "input", None, "signup_source") in result
    assert changes(other, base).count(("-", "input", "signup_source", None)) == 1


def _resolution():
    from generalforecast.predictors.resolution import ContextInfo, Resolution
    from generalforecast.predictors.selection import SelectionInfo

    return Resolution(
        name="churn",
        model=MODEL,
        model_digest="sha256:" + "f" * 64,
        task="classification",
        target="churned",
        target_type="0/1",
        classes=(0, 1),
        id=None,
        time=None,
        series=(),
        inputs=(("spend", "float"), ("plan", "category")),
        input_roles=("input", "input"),
        inferred={},
        context=ContextInfo("snapshot", 1840, 212, "sha256:" + "a" * 64),
        selection=SelectionInfo("exact"),
        outputs=("label", "probabilities"),
        manifest="sha256:" + "b" * 64,
    )


def test_load_lock_is_frozen_and_notices_any_manifest_edit(churn: Path):
    with pytest.raises(InvalidInput) as caught:
        load_lock(churn)
    assert caught.value.hint == "run `gf check` to write it"
    resolution = resolve(load(churn), churn)
    write_lock(resolution, churn)
    (churn / "context.csv").unlink()
    frozen = load_lock(churn)
    assert frozen.inputs == resolution.inputs
    assert frozen.input_roles == resolution.input_roles
    assert frozen.context == resolution.context
    assert frozen.selection == resolution.selection
    assert frozen.pinned == resolution.pinned
    # An edit that keeps name, model and target still makes the lock stale.
    write_manifest(
        churn,
        f"model: {MODEL}\ntarget: churned\nid: customer_id\n"
        "inputs:\n  exclude: [note, plan]\ncontext:\n  snapshot: context.csv\n",
    )
    with pytest.raises(InvalidInput) as caught:
        load_lock(churn)
    assert caught.value.message == "gf.lock was written for another gf.yaml"
    assert caught.value.hint == "run `gf check` to update it"


def test_a_lock_of_another_format_asks_for_gf_check(churn: Path):
    (churn / "gf.lock").write_text("lock_version: 2\npredictor: {name: churn}\n")
    with pytest.raises(InvalidInput) as caught:
        load_lock(churn)
    assert caught.value.message == (
        "gf.lock is not in the current format; run `gf check` to write it again"
    )
    resolution = resolve(load(churn), churn)
    assert resolution.lock_changes == ()
    write_lock(resolution, churn)
    assert yaml.safe_load((churn / "gf.lock").read_text())["lock_version"] == 1
    with pytest.raises(PredictorError) as refused:
        parse_lock({"lock_version": 2, "predictor": {}, "model": {}})
    assert refused.value.code == "UnsupportedLockVersion"


def test_a_tampered_lock_is_refused_by_its_digest(churn: Path):
    value = document(resolve(load(churn), churn))
    value["predictor"]["target"] = "plan"
    with pytest.raises(PredictorError) as caught:
        validate(value)
    assert caught.value.code == "LockMismatch"


def test_a_catalog_model_change_needs_upgrade(churn: Path):
    resolution = resolve(load(churn), churn)
    write_lock(resolution, churn)
    value = yaml.safe_load((churn / "gf.lock").read_text())
    value["predictor"]["model_digest"] = "sha256:" + "1" * 64
    body = {key: item for key, item in value.items() if key != "digest"}
    value["digest"] = lock_digest(body)
    (churn / "gf.lock").write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(InvalidInput) as caught:
        resolve(load(churn), churn)
    assert caught.value.hint == "run `gf check --upgrade` to pin the new artifact"
    upgraded = resolve(load(churn), churn, upgrade=True)
    assert (
        "~",
        "model_digest",
        "sha256:" + "1" * 64,
        resolution.model_digest,
    ) in upgraded.lock_changes
    assert upgraded.files == resolution.files
