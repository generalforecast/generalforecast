# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A lock records the gf that wrote it, so a plane tells an old gf from a hand edit."""

import dataclasses
from pathlib import Path

import pytest

from generalforecast.models import catalog
from generalforecast.predictors import _registry, load, resolve, verified, write_lock
from generalforecast.predictors.lock import (
    LOCK_RULES,
    document,
    lock_digest,
    lock_document,
    rules_of,
    validate,
)
from generalforecast.predictors.resolution import pushed_admission
from generalforecast.predictors.selection import resolve as resolve_selection
from generalforecast.predictors.validation import PredictorError

UPGRADE = (
    "upgrade the gf SDK (`pip install -U generalforecast`) and run `gf check` again"
)


def _checked(root: Path, *, cap: int | None = None) -> tuple[dict, dict, object]:
    """The lock `gf check` writes for a directory, its manifest and its rows.

    `cap` stands in for an older gf whose catalog capped the model's context
    rows lower, as gf before #76 capped TabICL v2 at 60,000 rows.
    """
    if cap is None:
        resolution = resolve(load(root), root)
    else:
        limits = _registry.limits

        def lower(ref):
            return dataclasses.replace(limits(ref), context_rows=cap)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_registry, "limits", lower)
            resolution = resolve(load(root), root)
    assert resolution.spec is not None
    return (
        document(resolution),
        resolution.spec.to_dict(),
        resolution.context_table,
    )


def _as_written_before_rules(value: dict) -> dict:
    """The same lock as a gf from before `checked_with` wrote it."""
    older = {key: item for key, item in value.items() if key != "checked_with"}
    assert lock_digest(older) == value["digest"]
    return older


def test_a_current_lock_is_admitted(churn: Path):
    value, manifest, rows = _checked(churn)
    assert rules_of(value) == LOCK_RULES
    pushed_admission(validate(value), manifest, rows)


def test_a_lock_from_before_the_rules_is_admitted_when_it_agrees(churn: Path):
    value, manifest, rows = _checked(churn)
    older = _as_written_before_rules(value)
    assert rules_of(older) == 1
    pushed_admission(validate(older), manifest, rows)


def test_an_older_gfs_selection_asks_for_an_upgrade(churn: Path):
    value, manifest, rows = _checked(churn, cap=100)
    assert value["predictor"]["selection"]["strategy"] == "retrieved"
    older = _as_written_before_rules(value)
    with pytest.raises(PredictorError) as caught:
        pushed_admission(validate(older), manifest, rows)
    assert caught.value.code == "LockMismatch"
    message = caught.value.message
    assert message.startswith(
        "gf.lock's selection is not what this service's gf check writes for it: "
        "this predictor was checked with a gf from before locks recorded it "
        "(lock rules 1); this service runs gf "
    )
    assert f"(lock rules {LOCK_RULES})" in message
    assert message.endswith(UPGRADE)


def test_an_older_recorded_gf_is_named(churn: Path):
    value, manifest, rows = _checked(churn, cap=100)
    value["checked_with"] = {"gf": "0.1.12", "rules": LOCK_RULES - 1}
    with pytest.raises(PredictorError) as caught:
        pushed_admission(validate(value), manifest, rows)
    assert "checked with gf 0.1.12 (lock rules 1)" in caught.value.message
    assert caught.value.message.endswith(UPGRADE)


def test_a_newer_gfs_lock_says_so(churn: Path):
    value, manifest, rows = _checked(churn, cap=100)
    value["checked_with"] = {"gf": "9.0.0", "rules": LOCK_RULES + 1}
    with pytest.raises(PredictorError) as caught:
        pushed_admission(validate(value), manifest, rows)
    message = caught.value.message
    assert f"checked with gf 9.0.0 (lock rules {LOCK_RULES + 1}), newer than" in message
    assert UPGRADE not in message


def test_a_mismatch_under_the_same_rules_keeps_its_refusal(churn: Path):
    value, manifest, rows = _checked(churn, cap=100)
    assert rules_of(value) == LOCK_RULES
    with pytest.raises(PredictorError) as caught:
        pushed_admission(validate(value), manifest, rows)
    assert caught.value.code == "LockMismatch"
    assert caught.value.message == (
        "gf.lock's selection is not what gf check writes for it; run `gf check`"
    )


def test_a_snapshot_mismatch_is_not_put_down_to_the_gf(churn: Path):
    value, manifest, rows = _checked(churn)
    older = _as_written_before_rules(value)
    with pytest.raises(PredictorError) as caught:
        pushed_admission(validate(older), manifest, rows.slice(0, 10))
    assert caught.value.code == "SnapshotMismatch"
    assert "pip install" not in caught.value.message


def test_gf_check_adds_checked_with_to_an_older_lock_without_a_new_digest(
    churn: Path,
):
    import yaml

    resolution = resolve(load(churn), churn)
    path = write_lock(resolution, churn)
    current = yaml.safe_load(path.read_text())
    older = _as_written_before_rules(current)
    path.write_text(yaml.safe_dump(older, sort_keys=False))
    assert lock_document(churn) == older
    verified(load(churn), churn)
    rewritten = yaml.safe_load(
        write_lock(resolve(load(churn), churn), churn).read_text()
    )
    assert rewritten["digest"] == older["digest"]
    assert rewritten["checked_with"]["rules"] == LOCK_RULES


@pytest.mark.parametrize(
    "recorded",
    [{"gf": "0.1.13"}, {"gf": "0.1.13", "rules": 0}, {"gf": 1, "rules": 2}, "0.1.13"],
)
def test_a_malformed_checked_with_is_refused(churn: Path, recorded):
    value, _, _ = _checked(churn)
    value["checked_with"] = recorded
    with pytest.raises(PredictorError):
        validate(value)


# What `gf check` writes for a selection follows each model's context row
# cap and the engine's adaptive rule past it. These are both as of
# LOCK_RULES. A change to either makes `gf check` write another lock for the
# same gf.yaml and rows, so it bumps LOCK_RULES (and its history in
# predictors/lock.py) and updates these. A new model adds its cap without a
# bump: no older lock names it.
RULES = 2
CAPS = {
    "general-forecast/tabicl-v2-classifier": 100_000,
    "general-forecast/tabicl-v2-regressor": 100_000,
    "general-forecast/kumo-tabular-small-classifier": 60_000,
    "general-forecast/kumo-tabular-small-regressor": 60_000,
    "general-forecast/kumo-tabular-medium-classifier": 60_000,
    "general-forecast/kumo-tabular-medium-regressor": 60_000,
    "general-forecast/kumo-tabular-large-classifier": 60_000,
    "general-forecast/kumo-tabular-large-regressor": 60_000,
    "general-forecast/tabpfn-3-classifier": 100_000,
    "general-forecast/tabpfn-3-regressor": 100_000,
    "general-forecast/tabpfn-3.5-classifier": 100_000,
    "general-forecast/tabpfn-3.5-regressor": 100_000,
    "general-forecast/limix-2-classifier": 100_000,
    "general-forecast/limix-2-regressor": 100_000,
    "general-forecast/mitra-classifier": 512,
    "general-forecast/mitra-2-classifier": 5_120,
    "general-forecast/mitra-2-regressor": 5_120,
    "general-forecast/tabfm-1-classifier": 100_000,
    "general-forecast/chronos-2": 100_000,
    "general-forecast/timestral-209m": 100_000,
    "general-forecast/timestral-209m-unmirrored": 100_000,
    "general-forecast/timestral-28m": 100_000,
    "general-forecast/timestral-28m-unmirrored": 100_000,
    "general-forecast/timesfm-3": 100_000,
}
# (cap, rows, timed) -> (core, neighbors, recent) of the selection past the cap.
ADAPTIVE = {
    (512, 25_600, False): (128, 384, None),
    (60_000, 300_000, False): (15_000, 45_000, None),
    (100_000, 5_000_000, True): (25_000, 75_000, 300_000),
}


def test_the_rules_are_the_ones_these_inputs_were_recorded_under():
    assert LOCK_RULES == RULES, "update CAPS and ADAPTIVE to the new rules"


@pytest.mark.parametrize("model", sorted(CAPS))
def test_a_models_context_row_cap_changes_with_the_rules(model):
    assert catalog.limits(model).context_rows == CAPS[model], (
        f"{model}'s context row cap changes what gf check writes for its "
        "locks; bump LOCK_RULES in predictors/lock.py and update CAPS"
    )


@pytest.mark.parametrize("case", sorted(ADAPTIVE))
def test_the_engines_selection_past_the_cap_changes_with_the_rules(case):
    cap, rows, timed = case
    chosen, _ = resolve_selection(None, rows=rows, cap=cap, timed=timed, model="m")
    assert (chosen.core, chosen.neighbors, chosen.recent) == ADAPTIVE[case], (
        "the engine's adaptive selection changes what gf check writes for "
        "locks past a model's cap; bump LOCK_RULES in predictors/lock.py and "
        "update ADAPTIVE"
    )
