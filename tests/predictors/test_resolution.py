# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from generalforecast.errors import InvalidInput
from generalforecast.predictors import load, resolve, write_lock
from generalforecast.predictors.resolution import SelectionInfo, _selection
from generalforecast.predictors.spec import PredictorSpec, Selection
from tests.predictors.conftest import (
    FORECASTER,
    MODEL,
    REGRESSOR,
    write_context,
    write_manifest,
)

CAP = 60_000


def test_resolve_infers_everything(churn: Path):
    resolution = resolve(load(churn), churn)
    assert resolution.task == "classification"
    assert resolution.inputs == (
        ("spend", "float"),
        ("plan", "category"),
        ("signups", "int"),
        ("signed_up", "timestamp"),
    )
    assert resolution.target_type == "0/1" and resolution.classes == (0, 1)
    assert resolution.context.kind == "snapshot"
    assert (resolution.context.rows, resolution.context.query_rows) == (180, 20)
    assert (
        resolution.context.digest is not None
        and resolution.context.digest.startswith("sha256:")
    )
    assert resolution.model_digest is not None and resolution.model_digest.startswith(
        "sha256:"
    )
    assert resolution.pinned == f"{MODEL}@{resolution.model_digest}"
    assert resolution.selection == SelectionInfo("exact")
    assert resolution.inferred["name"] == "the directory name"
    assert resolution.inferred["inputs"] == (
        "every column of context.csv except churned (target), customer_id (id), minus note"
    )
    assert resolution.inferred["queries"] == "20 rows with no target"
    assert resolution.inferred["classes"] == "2 classes in context.csv"
    assert resolution.lock_changes == ()


def test_inputs_list_replaces_inference(churn: Path):
    write_manifest(
        churn,
        f"model: {MODEL}\ntarget: churned\ninputs: [spend, plan]\ncontext: {{snapshot: context.csv}}\n",
    )
    resolution = resolve(load(churn), churn)
    assert resolution.input_names == ("spend", "plan")
    assert "inputs" not in resolution.inferred


def test_types_override(churn: Path):
    write_manifest(
        churn,
        f"model: {MODEL}\ntarget: churned\ninputs: {{types: {{signups: category}}}}\ncontext: {{snapshot: context.csv}}\n",
    )
    resolution = resolve(load(churn), churn)
    assert dict(resolution.inputs)["signups"] == "category"
    assert (
        resolution.inferred["inputs.signups"]
        == "declared category; context.csv looks like int"
    )


FAILURES = [
    (
        f"model: {MODEL}\ntarget: churn\ncontext: {{snapshot: context.csv}}\n",
        "target",
        "target column `churn` is not in the context",
        "did you mean `churned`?",
    ),
    (
        f"model: {MODEL}\ntarget: churned\ninputs: [spend, plam]\ncontext: {{snapshot: context.csv}}\n",
        "inputs",
        "input `plam` is not in the context",
        "did you mean `plan`?",
    ),
    (
        f"model: {MODEL}\ntarget: churned\ninputs: {{exclude: [nothing]}}\ncontext: {{snapshot: context.csv}}\n",
        "inputs.exclude",
        "`inputs.exclude` names `nothing`, which is not in the context",
        "context.csv has customer_id, spend, plan, signups, signed_up, churned, note",
    ),
    (
        f"model: {MODEL}\ntarget: churned\nid: customer\ncontext: {{snapshot: context.csv}}\n",
        "id",
        "id column `customer` is not in the context",
        "did you mean `customer_id`?",
    ),
    (
        f"model: {MODEL}\ntarget: churned\ninputs: {{exclude: [customer_id, spend, plan, signups, signed_up, note]}}\ncontext: {{snapshot: context.csv}}\n",
        "inputs",
        "predictor churn has no input columns",
        "add columns to the context, or list `inputs`",
    ),
    (
        f"model: {MODEL}\ntarget: churned\ncontext: {{request: {{}}}}\n",
        "inputs",
        "`inputs` must be listed for a request context",
        "a request context has no rows to infer from; write `inputs: [spend, plan]`",
    ),
    (
        f"model: {MODEL}\ntarget: churned\ncontext: {{snapshot: missing.csv}}\n",
        "context.snapshot",
        "`missing.csv` does not exist",
        "check the path",
    ),
    (
        f"model: {MODEL}\ntarget: spend\ncontext: {{snapshot: context.csv}}\n",
        "target",
        "target `spend` has float values; a classifier needs classes",
        "pick a regression model with `gf model list --task regression`",
    ),
    (
        f"model: {REGRESSOR}\ntarget: plan\ncontext: {{snapshot: context.csv}}\n",
        "target",
        "target `plan` has category values; a regressor needs numbers",
        "pick a classifier with `gf model list --task classification`",
    ),
]


@pytest.mark.parametrize(("text", "param", "message", "hint"), FAILURES)
def test_resolution_failures(tmp_path: Path, text, param, message, hint):
    root = tmp_path / "churn"
    write_context(root, empty_every=0)
    write_manifest(root, text)
    with pytest.raises(InvalidInput) as caught:
        resolve(load(root), root)
    error = caught.value
    assert (error.message, error.param, error.hint) == (message, param, hint)


def test_target_pointer_has_line_and_column(churn: Path):
    write_manifest(
        churn, f"model: {MODEL}\ntarget: churn\ncontext: {{snapshot: context.csv}}\n"
    )
    with pytest.raises(InvalidInput) as caught:
        resolve(load(churn), churn)
    assert (caught.value.line, caught.value.column) == (2, 9)
    assert (
        caught.value.because
        == "context.csv has customer_id, spend, plan, signups, signed_up, churned, note"
    )


def test_request_context_resolves_declared_inputs(tmp_path: Path):
    root = tmp_path / "adhoc"
    write_manifest(
        root,
        f"model: {MODEL}\ntarget: y\ninputs: [a, b]\ncontext: {{request: {{max_rows: 10}}}}\n",
    )
    resolution = resolve(load(root), root)
    assert resolution.inputs == (("a", None), ("b", None))
    assert resolution.target_type is None and resolution.classes is None
    assert resolution.context.rows is None
    assert resolution.selection.core is None


def test_forecast_infers_frequency_and_quantiles(tmp_path: Path):
    root = tmp_path / "sales"
    root.mkdir()
    rows = [(f"2025-01-{day:02d}", "a", day * 1.0, day % 2) for day in range(1, 31)]
    pq.write_table(
        pa.table(
            {
                "date": [row[0] for row in rows],
                "sku": [row[1] for row in rows],
                "units": [row[2] for row in rows],
                "promo": [row[3] for row in rows],
            }
        ),
        root / "sales.parquet",
    )
    write_manifest(
        root,
        f"model: {FORECASTER}\ntarget: units\ntime: date\nseries: [sku]\nhorizon: 7\ninputs: {{past: [promo]}}\ncontext: {{snapshot: sales.parquet}}\n",
    )
    resolution = resolve(load(root), root)
    assert resolution.task == "forecast"
    assert resolution.frequency == "1d"
    assert resolution.inferred["frequency"] == "the times in date step by 1d"
    assert resolution.quantiles == (0.1, 0.5, 0.9)
    assert resolution.inferred["quantiles"] == "the default levels"
    assert resolution.input_names == ("promo",)
    assert resolution.horizon == 7


SELECTIONS = [
    (1_000, None, SelectionInfo("exact")),
    (CAP, None, SelectionInfo("exact")),
    (CAP + 1, None, SelectionInfo("retrieved", 15_000, 45_000, "stratified")),
    (
        CAP + 1,
        Selection(core=0.5, neighbors=0.25),
        SelectionInfo("retrieved", 30_000, 15_000, "stratified"),
    ),
    (
        CAP + 1,
        Selection(core=1_000, kind="random"),
        SelectionInfo("retrieved", 1_000, 59_000, "random"),
    ),
    (
        100,
        Selection(core=10, neighbors=20, seed=7),
        SelectionInfo("retrieved", 10, 20, "stratified", seed=7),
    ),
    (None, None, SelectionInfo("adaptive")),
]


@pytest.mark.parametrize(("rows", "declared", "expected"), SELECTIONS)
def test_selection_defaults(rows, declared, expected):
    spec = PredictorSpec(name="t", model=MODEL, target="y", selection=declared)
    assert _selection(spec, rows, CAP)[0] == expected


def test_selection_recent_with_time_role():
    spec = PredictorSpec(name="t", model=MODEL, target="y", time="t")
    selection, reason = _selection(spec, CAP + 1, CAP)
    assert selection == SelectionInfo(
        "retrieved", 15_000, 45_000, "stratified", 180_000
    )
    assert reason is not None and reason.startswith(
        "60,001 rows exceed the 60,000 row cap"
    )


def test_a_selection_past_the_row_cap_is_refused():
    spec = PredictorSpec(
        name="t",
        model=MODEL,
        target="y",
        selection=Selection(core=100_000, neighbors=50_000),
    )
    with pytest.raises(InvalidInput) as caught:
        _selection(spec, CAP + 1, CAP)
    assert caught.value.message == (
        f"selection reads 150,000 rows per query; {MODEL} reads at most 60,000"
    )
    assert (caught.value.limit, caught.value.actual) == (CAP, 150_000)


def test_locked_requires_an_up_to_date_lock(churn: Path):
    spec = load(churn)
    with pytest.raises(InvalidInput) as caught:
        resolve(spec, churn, locked=True)
    assert caught.value.message == "gf.lock does not exist"
    write_lock(resolve(spec, churn), churn)
    assert resolve(spec, churn, locked=True).lock_changes == ()
    write_manifest(
        churn,
        f"model: {MODEL}\ntarget: churned\ninputs: [spend]\ncontext: {{snapshot: context.csv}}\n",
    )
    with pytest.raises(InvalidInput) as caught:
        resolve(load(churn), churn, locked=True)
    assert caught.value.message == "gf.lock is out of date"
    assert caught.value.hint == "run `gf check` without '--locked' to write it"


def test_whole_number_float_targets_resolve_integer_classes(tmp_path: Path):
    root = tmp_path / "floaty"
    rows = "".join(f"{i},{float(i % 2) if i % 5 else ''}\n" for i in range(40))
    write_manifest(
        root, f"model: {MODEL}\ntarget: y\ncontext: {{snapshot: context.csv}}\n"
    )
    (root / "context.csv").write_text("x,y\n" + rows)
    resolution = resolve(load(root), root)
    assert resolution.classes == (0, 1)
    assert resolution.target_type == "0/1"
    assert resolution.context.rows == 32 and resolution.context.query_rows == 8


def test_the_id_column_type_is_resolved(churn: Path):
    assert resolve(load(churn), churn).id_type == "int"
