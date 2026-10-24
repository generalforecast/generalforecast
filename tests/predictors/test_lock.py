# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from generalforecast.errors import InvalidInput
from generalforecast.predictors import load, load_lock, resolve, write_lock
from generalforecast.predictors.lock import changes, document
from tests.predictors.conftest import MODEL, write_context, write_manifest


def test_lock_shape(churn: Path):
    resolution = resolve(load(churn), churn)
    path = write_lock(resolution, churn)
    value = yaml.safe_load(path.read_text())
    assert value["lock_version"] == 1
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
        "inputs",
        "context",
        "selection",
    ]
    assert predictor["model"] == MODEL
    assert predictor["model_digest"] == resolution.model_digest
    assert predictor["inputs"][0] == {"name": "spend", "type": "float"}
    assert predictor["context"] == {
        "kind": "snapshot",
        "path": "context.csv",
        "digest": resolution.context.digest,
        "rows": 180,
        "query_rows": 20,
    }
    assert predictor["selection"] == {
        "core": None,
        "neighbors": 0,
        "kind": "stratified",
    }
    assert set(value["model"]["files"]) == {
        "LICENSE",
        "config.json",
        "model.safetensors",
    }


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


def test_changes_table():
    base = document(_resolution())
    assert changes(None, base) == ()
    assert changes(base, base) == ()
    other = yaml.safe_load(yaml.safe_dump(base))
    other["predictor"]["selection"] = {
        "core": 1000,
        "neighbors": 3000,
        "kind": "stratified",
    }
    other["predictor"]["inputs"].append({"name": "signup_source", "type": "category"})
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
    from generalforecast.predictors.resolution import (
        ContextInfo,
        Resolution,
        SelectionInfo,
    )

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
        inferred={},
        context=ContextInfo("snapshot", 1840, 212, "sha256:" + "a" * 64),
        selection=SelectionInfo(None, 0, "stratified"),
    )


def test_load_lock_is_frozen(churn: Path):
    with pytest.raises(InvalidInput) as caught:
        load_lock(churn)
    assert caught.value.hint == "run `gf check` to write it"
    resolution = resolve(load(churn), churn)
    write_lock(resolution, churn)
    (churn / "context.csv").unlink()
    frozen = load_lock(churn)
    assert frozen.inputs == resolution.inputs
    assert frozen.context == resolution.context
    assert frozen.selection == resolution.selection
    assert frozen.pinned == resolution.pinned
    write_manifest(
        churn, f"model: {MODEL}\ntarget: plan\ncontext: {{snapshot: context.csv}}\n"
    )
    with pytest.raises(InvalidInput) as caught:
        load_lock(churn)
    assert (
        caught.value.message
        == "gf.lock was written for `target: churned`, and gf.yaml now says `target: plan`"
    )


def test_old_lock_is_replaced_without_diff(churn: Path):
    (churn / "gf.lock").write_text("lock_version: 1\npackage: abc\n")
    resolution = resolve(load(churn), churn)
    assert resolution.lock_changes == ()
    write_lock(resolution, churn)
    assert yaml.safe_load((churn / "gf.lock").read_text())["lock_version"] == 1


def test_upgrade_guards_the_model_digest(churn: Path):
    resolution = resolve(load(churn), churn)
    write_lock(resolution, churn)
    value = yaml.safe_load((churn / "gf.lock").read_text())
    value["predictor"]["model_digest"] = "sha256:" + "1" * 64
    (churn / "gf.lock").write_text(yaml.safe_dump(value, sort_keys=False))
    kept = resolve(load(churn), churn)
    assert kept.model_digest == "sha256:" + "1" * 64
    assert not any(change[1] == "model_digest" for change in kept.lock_changes)
    upgraded = resolve(load(churn), churn, upgrade=True)
    assert (
        "~",
        "model_digest",
        "sha256:" + "1" * 64,
        resolution.model_digest,
    ) in upgraded.lock_changes
