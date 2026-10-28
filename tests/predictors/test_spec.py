# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from generalforecast.errors import InvalidInput
from generalforecast.predictors import PredictorSpec, load, write_manifest
from generalforecast.predictors.spec import Context, Holdout, parse_text
from tests.predictors.conftest import FORECASTER, MODEL, write_context

BASE = f"model: {MODEL}\ntarget: churned\ncontext: {{snapshot: context.csv}}\n"


def test_parses_every_field():
    spec = parse_text(
        f"""
name: churn
model: {MODEL}
target: churned
id: customer_id
time: signed_up
inputs:
  exclude: [note]
  types: {{plan: category}}
context:
  snapshot: context.csv
  until: 2025-09-01
selection:
  core: 0.25
  neighbors: 0.75
  kind: stratified
resources:
  accelerator: L4
  replicas: {{min: 0, max: 2}}
  idle_seconds: 300
""",
        default_name="dir",
    )
    assert spec.name == "churn"
    assert spec.exclude == ("note",)
    assert spec.types == (("plan", "category"),)
    assert spec.context == Context("snapshot", "context.csv", until="2025-09-01")
    assert spec.selection is not None and spec.selection.core == 0.25
    assert spec.resources is not None and spec.resources.replicas_max == 2
    mark = spec.mark("target")
    assert mark is not None and (mark.line, mark.column) == (4, 9)


def test_inputs_list_and_defaults():
    spec = parse_text(f"model: {MODEL}\ntarget: y\ninputs: [a, b]\n", default_name="d")
    assert spec.inputs == ("a", "b")
    assert spec.name == "d"
    assert spec.context == Context("request")


def test_holdout_shorthand_and_mapping():
    spec = parse_text(BASE.replace("}", ", holdout: 0.2}"), default_name="d")
    assert spec.context.holdout == Holdout(0.2, 0)
    spec = parse_text(
        BASE.replace("}", ", holdout: {fraction: 0.3, seed: 7}}"), default_name="d"
    )
    assert spec.context.holdout == Holdout(0.3, 7)


def test_forecast_fields():
    spec = parse_text(
        f"model: {FORECASTER}\ntarget: units\ntime: date\nseries: [sku]\n"
        "frequency: 1d\nhorizon: 7\nquantiles: [0.1, 0.9]\n"
        "inputs: {past: [promo], future: [price]}\n",
        default_name="sales",
    )
    assert spec.time == "date" and spec.series == ("sku",)
    assert spec.frequency == "1d" and spec.horizon == 7
    assert spec.quantiles == (0.1, 0.9)
    assert spec.past == ("promo",) and spec.future == ("price",)


RETIRED = [
    (
        "features: [a]\n",
        "features",
        "`features` is no longer a manifest key",
        "use `inputs`: a list of columns, or `{exclude: [...]}`",
    ),
    (
        "task: classification\n",
        "task",
        "`task` is no longer a manifest key",
        "remove `task`; the model determines it",
    ),
    (
        "data: a.csv\n",
        "data",
        "`data` is no longer a manifest key",
        "use `context: {snapshot: FILE}`",
    ),
    (
        "strategy: exact\n",
        "strategy",
        "`strategy` is no longer a manifest key",
        "use `selection: {core, neighbors, kind}`",
    ),
    (
        "step: 1d\n",
        "step",
        "`step` is no longer a manifest key",
        "use `frequency`, such as `1d` or `1h`",
    ),
    (
        "classes: [a, b]\n",
        "classes",
        "`classes` is no longer a manifest key",
        "remove `classes`; `gf check` reads them from the context",
    ),
    (
        "target_type: int64\n",
        "target_type",
        "`target_type` is no longer a manifest key",
        "remove `target_type`; `gf check` infers it from the context",
    ),
    (
        "past_inputs: [a]\n",
        "past_inputs",
        "`past_inputs` is no longer a manifest key",
        "use `inputs: {past: [...]}`",
    ),
    (
        "future_inputs: [a]\n",
        "future_inputs",
        "`future_inputs` is no longer a manifest key",
        "use `inputs: {future: [...]}`",
    ),
    (
        "source: a.csv\n",
        "source",
        "`source` is no longer a manifest key",
        "use `context: {snapshot: FILE}`",
    ),
    (
        "mode: snapshot\n",
        "mode",
        "`mode` is no longer a manifest key",
        "use `context: {snapshot: FILE}`, `{request: {max_rows: N}}` or `{live: NAME}`",
    ),
    (
        "cpu: 2\n",
        "cpu",
        "`cpu` is no longer a manifest key",
        "use `resources: {accelerator: L4}`",
    ),
    (
        "memory: 8Gi\n",
        "memory",
        "`memory` is no longer a manifest key",
        "use `resources: {accelerator: L4}`",
    ),
    (
        "min_replicas: 1\n",
        "min_replicas",
        "`min_replicas` is no longer a manifest key",
        "use `resources: {replicas: {min: 0, max: 2}}`",
    ),
    (
        "context: {source: a.csv}\n",
        "context.source",
        "`source` is no longer a manifest key",
        "use `context: {snapshot: FILE}`",
    ),
    (
        "context: {mode: request}\n",
        "context.mode",
        "`mode` is no longer a manifest key",
        "use `context: {snapshot: FILE}`, `{request: {max_rows: N}}` or `{live: NAME}`",
    ),
]


@pytest.mark.parametrize(("line", "param", "message", "hint"), RETIRED)
def test_retired_keys(line, param, message, hint):
    text = f"model: {MODEL}\ntarget: y\n" + line
    with pytest.raises(InvalidInput) as caught:
        parse_text(text, default_name="d")
    error = caught.value
    assert (error.message, error.param, error.hint) == (message, param, hint)
    assert error.line == 3 and error.column is not None


INVALID = [
    (
        "target: y\n",
        "model",
        "gf.yaml has no `model`",
        f"add `model: {MODEL}`, or run `gf model list`",
    ),
    (
        "model: tabicl-classifier\ntarget: y\n",
        "model",
        f"`tabicl-classifier` is not a catalog model id; use {MODEL}",
        None,
    ),
    (
        "model: tabicl-v2-classifier\ntarget: y\n",
        "model",
        f"`tabicl-v2-classifier` is not a catalog model id; use {MODEL}",
        None,
    ),
    (
        "model: general-forecast/nope\ntarget: y\n",
        "model",
        "model `general-forecast/nope` is not in the catalog",
        "run `gf model list` to see the catalog",
    ),
    (
        f"model: {MODEL}\n",
        "target",
        "gf.yaml has no `target`",
        "add `target: <column>`; `gf init` picks one from the context",
    ),
    (
        f"model: {MODEL}\ntarget: y\nname: Churn\n",
        "name",
        "name `Churn` is not a predictor name",
        "use lowercase letters, digits and hyphens, starting with a letter",
    ),
    (
        f"model: {MODEL}\ntarget: y\nfoo: 1\n",
        "foo",
        "unknown key `foo`",
        "gf.yaml takes name, model, target, id, time, series, related, frequency, horizon, quantiles, inputs, context, selection, resources",
    ),
    (
        f"model: {MODEL}\ntarget: y\ncontext: a.csv\n",
        "context",
        "`context` must say which kind it is",
        "write `context: {snapshot: a.csv}`",
    ),
    (
        f"model: {MODEL}\ntarget: y\ncontext: {{snapshot: a.csv, live: b}}\n",
        "context",
        "`context` must have exactly one of `snapshot`, `request` or `live`",
        "write `context: {snapshot: context.csv}`",
    ),
    (
        f"model: {MODEL}\ntarget: y\ncontext: {{snapshot: ../a.csv}}\n",
        "context.snapshot",
        "snapshot `../a.csv` is outside the predictor directory",
        "copy the file into the predictor directory and name it relative to gf.yaml",
    ),
    (
        f"model: {MODEL}\ntarget: y\ntime: t\ncontext: {{snapshot: a.csv, until: 2025-01-01, holdout: 0.2}}\n",
        "context.holdout",
        "`until` and `holdout` cannot both be set",
        "pick one way to split queries from the snapshot",
    ),
    (
        f"model: {MODEL}\ntarget: y\ncontext: {{request: {{}}, holdout: 0.2}}\n",
        "context.holdout",
        "`holdout` needs a snapshot context",
        "a request context has no rows to split; remove `holdout`",
    ),
    (
        f"model: {MODEL}\ntarget: y\ncontext: {{snapshot: a.csv, holdout: 1.5}}\n",
        "context.holdout.fraction",
        "`context.holdout.fraction` must be a number between 0 and 1",
        "write `holdout: {fraction: 0.2}`",
    ),
    (
        f"model: {MODEL}\ntarget: y\ncontext: {{snapshot: a.csv, until: 2025-01-01}}\n",
        "context.until",
        "`until` needs a `time` column to compare against",
        "add `time: <column>`, or use `holdout` for a random split",
    ),
    (
        f"model: {MODEL}\ntarget: y\ninputs: [a, a]\n",
        "inputs",
        "`inputs` lists `a` twice",
        "name each column once",
    ),
    (
        f"model: {MODEL}\ntarget: y\ninputs: [y]\n",
        "inputs",
        "`y` is the target, not an input",
        "remove it from `inputs`; role columns are never inputs",
    ),
    (
        f"model: {MODEL}\ntarget: y\nid: k\ninputs: {{exclude: [k]}}\n",
        "inputs.exclude",
        "cannot exclude `k`; it is the id",
        "remove it from `inputs.exclude`",
    ),
    (
        f"model: {MODEL}\ntarget: y\ninputs: {{types: {{a: int64}}}}\n",
        "inputs.types.a",
        "`inputs.types.a` must be one of int, float, 0/1, category, text, timestamp, bool",
        None,
    ),
    (
        f"model: {MODEL}\ntarget: y\nhorizon: 7\n",
        "horizon",
        "`horizon` is for forecast models",
        "remove `horizon`, or pick a forecast model with `gf model list --task forecast`",
    ),
    (
        f"model: {FORECASTER}\ntarget: y\n",
        "time",
        "predictor d has no cadence",
        "add `time: date` and `frequency: 1d`, or pick a tabular model\nwith `gf model list --task classification`",
    ),
    (
        f"model: {FORECASTER}\ntarget: y\ntime: t\n",
        "horizon",
        "predictor d has no `horizon`",
        "add `horizon: 7` for seven steps ahead",
    ),
    (
        f"model: {MODEL}\ntarget: y\nselection: {{kind: nearest}}\n",
        "selection.kind",
        "`selection.kind` must be one of stratified, random",
        None,
    ),
    (
        f"model: {MODEL}\ntarget: y\nresources: {{accelerator: H100}}\n",
        "resources.accelerator",
        "`resources.accelerator` must be one of L4, A10G, L40S, cpu",
        None,
    ),
    (
        f"model: {MODEL}\ntarget: y\nresources: {{replicas: {{min: 3, max: 2}}}}\n",
        "resources.replicas.min",
        "`resources.replicas.min` must not exceed `max`",
        None,
    ),
    (
        f"model: {MODEL}\ntarget: y\nfrequency: fast\n",
        "frequency",
        (
            "`frequency` must be a span such as `1d`, `1h` or `7d`, a pandas offset "
            "alias such as `W-MON` or `MS`, or seconds as a number"
        ),
        None,
    ),
]


@pytest.mark.parametrize(("text", "param", "message", "hint"), INVALID)
def test_invalid_manifests(text, param, message, hint):
    with pytest.raises(InvalidInput) as caught:
        parse_text(text, default_name="d")
    error = caught.value
    assert (error.message, error.param, error.hint) == (message, param, hint)
    assert error.code == "ValidationError"


def test_forecast_error_has_because():
    with pytest.raises(InvalidInput) as caught:
        parse_text(f"model: {FORECASTER}\ntarget: y\n", default_name="sales")
    assert caught.value.because == (
        f"{FORECASTER} forecasts series, and gf.yaml\ndeclares no `time` column"
    )
    assert str(caught.value).startswith(
        "error[ValidationError]: predictor sales has no cadence\nnote:"
    )


def test_programmatic_inputs_and_exclude_conflict():
    from generalforecast.predictors.spec import validate

    spec = PredictorSpec(
        name="d", model=MODEL, target="y", inputs=("a",), exclude=("b",)
    )
    with pytest.raises(InvalidInput) as caught:
        validate(spec)
    assert caught.value.message == "`inputs` lists columns and also excludes some"
    assert caught.value.param == "inputs"
    assert (
        caught.value.hint
        == "list the inputs you want, or exclude from the inferred set, not both"
    )


def test_load_defaults_name_to_directory(churn: Path):
    spec = load(churn)
    assert spec.name == "churn"
    assert load(churn / "gf.yaml") == spec


def test_load_missing_manifest(tmp_path: Path):
    with pytest.raises(InvalidInput) as caught:
        load(tmp_path / "nothing")
    assert caught.value.message.startswith("no gf.yaml in ")
    assert caught.value.hint == "run `gf init` to write one"


def test_write_manifest_round_trips(tmp_path: Path):
    root = tmp_path / "churn"
    write_context(root)
    spec = PredictorSpec(
        name="churn",
        model=MODEL,
        target="churned",
        id="customer_id",
        exclude=("note",),
        context=Context("snapshot", "context.csv"),
    )
    path = write_manifest(spec, root)
    assert path == root / "gf.yaml"
    assert path.read_text() == (
        f"name: churn\nmodel: {MODEL}\ntarget: churned\nid: customer_id\n"
        "inputs:\n  exclude: [note]\ncontext: {snapshot: context.csv}\n"
    )
    assert load(root) == spec
    with pytest.raises(InvalidInput) as caught:
        write_manifest(spec, root)
    assert (
        caught.value.message.endswith(
            "gf.yaml already exists in "
            + str(root).split("/", 1)[0]
            + "/"
            + str(root).split("/", 1)[1]
        )
        or "already exists" in caught.value.message
    )
    assert (
        caught.value.hint
        == "pass --force to overwrite it, or run `gf check` to validate it"
    )
    write_manifest(spec, root, force=True)


def test_write_manifest_never_leaves_a_half_made_directory(tmp_path: Path):
    root = tmp_path / "new"
    spec = PredictorSpec(
        name="new", model=MODEL, target="y", context=Context("snapshot", "missing.csv")
    )
    with pytest.raises(InvalidInput) as caught:
        write_manifest(spec, root)
    assert caught.value.param == "context.snapshot"
    assert "is not a file in" in caught.value.message
    assert not root.exists()
    with pytest.raises(InvalidInput):
        write_manifest(PredictorSpec(name="Bad", model=MODEL, target="y"), root)
    assert not root.exists()
