# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`gf train` and `gf runs`: prompts and flags, the snapshot, and a run followed live."""

import io
import json
import re
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from generalforecast import cli, term
from generalforecast.errors import NotFound
from generalforecast.tabular import table_digest
from generalforecast.train_commands import (
    FINE_TUNABLE,
    Follower,
    default_name,
    name_problem,
    snapshot_name,
)
from tests.product_cli.test_cli_output import gf

TABICL = "general-forecast/tabicl-v2-classifier"
MITRA = "general-forecast/mitra-2-classifier"
KUMO = "general-forecast/kumo-tabular-medium-classifier"
REPORT = {
    "rows": 200,
    "base": 0.4521,
    "kept": 0.4188,
    "gain": 0.0333,
    "ci95": [0.0151, 0.0515],
    "metrics": {
        "base": {"log_loss": 0.4521, "accuracy": 0.8},
        "kept": {"log_loss": 0.4188, "accuracy": 0.82},
    },
}


def _published(model, *, kept_step=50):
    return {
        "model": model,
        "digest": "sha256:" + "9" * 64,
        "kept_step": kept_step,
        "validation_loss": 0.419,
        "base_validation_loss": 0.4521,
        "report": {"data": REPORT},
    }


def _events(model):
    return [
        {"at": 1.0, "kind": "created", "body": {}},
        {"at": 2.0, "kind": "state", "body": {"state": "queued"}},
        {"at": 3.0, "kind": "state", "body": {"state": "setting_up"}},
        {"at": 4.0, "kind": "state", "body": {"state": "training"}},
        {"at": 5.0, "kind": "metrics", "body": {"steps": 10, "loss": 0.52}},
        {
            "at": 6.0,
            "kind": "metrics",
            "body": {
                "steps": 25,
                "loss": 0.5,
                "validation_loss": 0.4402,
                "check_gain": 0.0102,
                "check_standard_error": 0.0061,
            },
        },
        {
            "at": 7.0,
            "kind": "metrics",
            "body": {
                "steps": 50,
                "loss": 0.48,
                "validation_loss": 0.419,
                "check_gain": 0.0314,
                "check_standard_error": 0.0072,
            },
        },
        {
            "at": 8.0,
            "kind": "metrics",
            "body": {
                "steps": 75,
                "loss": 0.47,
                "validation_loss": 0.4231,
                "check_gain": 0.0273,
                "check_standard_error": 0.007,
            },
        },
        {"at": 9.0, "kind": "published", "body": _published(model)},
        {"at": 10.0, "kind": "state", "body": {"state": "finished"}},
    ]


class _Snapshots:
    def __init__(self, existing=None):
        self.existing = existing or {}
        self.made = []

    def get(self, name):
        if name not in self.existing:
            raise NotFound(f"snapshot {name} does not exist")
        return self.existing[name]

    def create(self, name, data, *, target, task):
        self.made.append((name, data.num_rows, target, task))
        return {"name": name, "state": "ready", "rows": data.num_rows}


class _Runs:
    def __init__(self, *, ended="finished"):
        self.ended = ended
        self.made = []
        self.watched = []
        self.cancelled = []

    def _run(self, model, state, base=TABICL):
        run = {"model": model, "state": state, "base": base, "reason": None}
        run["method"] = {"type": "lora", "rank": 16}
        if state == "finished":
            run["published"] = _published(model)
        return run

    def create(self, name, **spec):
        self.made.append((name, spec))
        return self._run(f"acme/{name}@ft1", "queued", spec["base"])

    def watch(self, model):
        self.watched.append(model)
        yield from _events(model)

    def get(self, model):
        model = model if "@" in model else f"{model}@ft1"
        base = self.made[-1][1]["base"] if self.made else TABICL
        run = self._run(model, self.ended, base)
        if self.ended == "failed":
            run["reason"] = "calibration"
        return run

    def list(self, *, state=None):
        return [
            self._run("acme/churn@ft1", "finished"),
            self._run("acme/churn@ft2", "training"),
        ]

    def cancel(self, model, *, save=False):
        self.cancelled.append((model, save))
        return self._run(model, "cancelling")


class _Client:
    def __init__(self, snapshots=None, runs=None):
        self.snapshots = snapshots or _Snapshots()
        self.runs = runs or _Runs()


@pytest.fixture
def client(monkeypatch):
    made = _Client()
    monkeypatch.setattr("generalforecast.commands.client_for", lambda args: made)
    return made


def _churn(path, rows=300):
    lines = ["id,x,z,churn"] + [f"{i},{i / 10},{i % 5},{i % 2}" for i in range(rows)]
    path.write_text("\n".join(lines) + "\n")
    return path


def _prices(path, rows=300):
    lines = ["id,size,price"] + [
        f"{i},{i % 7},{i * 1.37 % 50:.2f}" for i in range(rows)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


class _Offered(term.Prompter):
    """A terminal prompter that takes each default and records what it offered."""

    offered: ClassVar[dict] = {}

    def __init__(self, out, **options):
        super().__init__(out, **options, interactive=True)

    def _raw(self, label, options, default, *, multi):
        self.offered[label] = [option.value for option in options]
        return [default]

    def _edit(self, label, default, aside, problem):
        self.offered[label] = default
        return default

    def confirm(self, question, *, flag="--yes"):
        self.offered["confirm"] = question


@pytest.fixture
def offered(monkeypatch):
    _Offered.offered = {}
    monkeypatch.setattr(term, "Prompter", _Offered)
    return _Offered.offered


def test_yes_takes_every_default_and_follows_the_run(monkeypatch, tmp_path, client):
    table = _churn(tmp_path / "churn.csv")
    monkeypatch.chdir(tmp_path)
    code, stdout, stderr = gf("train", "--yes", monkeypatch=monkeypatch)
    assert code == 0, stderr
    from generalforecast.commands import read_data

    snapshot = snapshot_name("churn", table_digest(read_data(str(table), tmp_path)))
    assert snapshot.startswith("churn-") and len(snapshot) == len("churn-") + 12
    # The upload's duration is the one measured value in the transcript.
    assert re.sub(r" in [0-9.]+s\n", " in 0.00s\n", stderr) == (
        "        Read churn.csv  300 rows, 4 columns\n"
        "\n"
        "Target       churn\n"
        "Task         classification  churn has 2 distinct 0/1 values\n"
        f"Base         {TABICL}  predictors and API\n"
        "Method       lora rank 16\n"
        "Name         churn\n"
        "\n"
        "  plan     400 steps at learning rate 3e-06, 8,192-row episodes, "
        "a check every 25 steps, patience 6\n"
        "  holdout  0.2 of the rows\n"
        "  budget   2 GPU hours\n"
        f"    Uploaded snapshot {snapshot}  300 rows in 0.00s\n"
        f"     Started acme/churn@ft1  {TABICL}, lora rank 16\n"
        "      Queued acme/churn@ft1\n"
        "  Setting up acme/churn@ft1\n"
        "    Training acme/churn@ft1\n"
        "     Checked step 25  select loss 0.4402, gain +0.0102 ± 0.0061\n"
        "     Checked step 50  select loss 0.4190, gain +0.0314 ± 0.0072, kept\n"
        "     Checked step 75  select loss 0.4231, gain +0.0273 ± 0.0070\n"
        "   Published acme/churn@ft1  kept step 50\n"
        "    Finished acme/churn@ft1\n"
    )
    assert stdout == (
        "version      acme/churn@ft1\n"
        "kept step    50\n"
        "report rows  200\n"
        "log loss     0.4521 base, 0.4188 kept\n"
        "gain         +7.37%, 95% CI +3.34% to +11.39%\n"
        "next: put `model: acme/churn@ft1` in gf.yaml, then run `gf push`\n"
    )
    assert client.snapshots.made == [(snapshot, 300, "churn", "classification")]
    [(name, spec)] = client.runs.made
    assert name == "churn"
    assert spec == {
        "base": TABICL,
        "data": [snapshot],
        "method": {"type": "lora", "rank": 16},
        "plan": {
            "steps": 400,
            "learning_rate": 3e-6,
            "episode_sampling": {"rows": 8192, "context_fraction": 0.9375},
            "eval_steps": 25,
            "early_stopping_patience": 6,
        },
        "evaluation": {"holdout": 0.2},
        "budget": {"gpu_hours": 2.0},
    }
    assert client.runs.watched == ["acme/churn@ft1"]


def test_every_flag_settles_its_prompt(monkeypatch, tmp_path, client, offered):
    table = _churn(tmp_path / "churn.csv")
    code, _, stderr = gf(
        "train",
        str(table),
        "--target",
        "churn",
        "--base",
        KUMO,
        "--method",
        "full",
        "--components",
        "row_encoder,prediction_head",
        "--name",
        "attrition",
        "--steps",
        "200",
        "--episode-rows",
        "4096",
        "--eval-steps",
        "20",
        "--patience",
        "4",
        "--holdout",
        "0.25",
        "--budget-gpu-hours",
        "1.5",
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    assert set(offered) == {"confirm"} and offered["confirm"] == "Train?"
    assert "Task         classification  from --base\n" in stderr
    assert f"Base         {KUMO}  API only\n" in stderr
    assert "Method       full row_encoder, prediction_head\n" in stderr
    assert (
        "  plan     200 steps at learning rate 1e-07, 4,096-row episodes, "
        "a check every 20 steps, patience 4\n"
        "  holdout  0.25 of the rows\n"
        "  budget   1.5 GPU hours\n"
    ) in stderr
    [(name, spec)] = client.runs.made
    assert name == "attrition"
    assert spec["method"] == {
        "type": "full",
        "components": ["row_encoder", "prediction_head"],
    }
    assert spec["plan"]["learning_rate"] == 1e-7
    assert spec["evaluation"] == {"holdout": 0.25}
    assert spec["budget"] == {"gpu_hours": 1.5}


def test_choices_are_filtered_by_the_task_and_the_base(
    monkeypatch, tmp_path, client, offered
):
    table = _prices(tmp_path / "prices.csv")
    code, stdout, stderr = gf("train", str(table), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert offered["Target"] == ["size", "price"]
    assert "Task         classification  size has 7 distinct int values" in stderr
    assert offered["Base"] == [
        ref for ref in FINE_TUNABLE if ref.endswith("-classifier")
    ]

    offered.clear()
    code, stdout, stderr = gf(
        "train", str(table), "--task", "regression", monkeypatch=monkeypatch
    )
    assert code == 0, stderr
    assert offered["Target"][0] == "price"
    assert "Task         regression  from --task" in stderr
    regressors = [ref for ref in FINE_TUNABLE if ref.endswith("-regressor")]
    assert offered["Base"] == regressors
    assert offered["Base"][0] == "general-forecast/tabicl-v2-regressor"
    assert offered["Method"] == ["lora", "input_adapter", "full"]
    assert offered["Name"] == "prices"
    assert "next: put `model: acme/prices@ft1` in gf.yaml" in stdout

    offered.clear()
    code, stdout, stderr = gf(
        "train",
        str(table),
        "--target",
        "size",
        "--task",
        "classification",
        "--base",
        MITRA,
        "--name",
        "sizes",
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    assert offered["Method"] == ["lora"]
    assert "Task         classification  from --base\n" in stderr
    assert "next: predict with `acme/sizes@ft1` through the Models API\n" in stdout


def test_bases_and_methods_the_service_refuses_are_refused_first(
    monkeypatch, tmp_path, client
):
    table = _churn(tmp_path / "churn.csv")
    cases = (
        (
            ("--base", "general-forecast/tabpfn-3-classifier"),
            (
                "error[UnsupportedCapability]: general-forecast/tabpfn-3-classifier cannot "
                "be fine-tuned\n"
            ),
        ),
        (
            ("--base", MITRA, "--method", "full"),
            (
                f"error[UnsupportedCapability]: {MITRA} does not take full fine-tuning\n"
                "help: pass '--method' with one of lora\n"
            ),
        ),
        (
            ("--base", MITRA, "--input-adapter"),
            f"error[UnsupportedCapability]: {MITRA} does not take an input adapter\n",
        ),
        (
            ("--method", "full", "--rank", "8"),
            "error[ValidationError]: --rank applies to lora fine-tuning, not full\n",
        ),
        (
            ("--method", "lora", "--components", "row_encoder"),
            (
                "error[ValidationError]: --components applies to full fine-tuning, "
                "not lora\n"
            ),
        ),
        (
            ("--method", "full", "--components", "decoder"),
            "error[ValidationError]: `decoder` is not a component\n",
        ),
        (
            ("--task", "regression", "--base", TABICL),
            f"error[ValidationError]: {TABICL} does classification, not regression\n",
        ),
        (
            ("--target", "churm"),
            "error[ValidationError]: target column `churm` is not in churn.csv\n",
        ),
        (
            ("--name", "Churn"),
            (
                "error[ValidationError]: name must be lowercase letters, digits and "
                "dashes, starting with a letter\n"
            ),
        ),
        (
            ("--holdout", "1.5"),
            "error[ValidationError]: --holdout must be between 0 and 1\n",
        ),
    )
    for flags, expected in cases:
        code, _, stderr = gf(
            "train", str(table), "--yes", *flags, monkeypatch=monkeypatch
        )
        assert code == 1, flags
        assert expected in stderr, (flags, stderr)
    assert client.runs.made == [] and client.snapshots.made == []


def test_no_input_names_the_flag_of_whatever_is_missing(monkeypatch, tmp_path, client):
    table = _churn(tmp_path / "churn.csv")
    code, stdout, stderr = gf(
        "--json", "train", str(table), "--no-input", monkeypatch=monkeypatch
    )
    assert code == 1 and not stdout
    assert json.loads(stderr.splitlines()[-1])["error"] == {
        "code": "ValidationError",
        "message": "target could not be inferred and no terminal is attached",
        "param": "target",
        "hint": "pass '--target', or run without --no-input in a terminal",
    }
    flags = ("--target", "churn", "--base", TABICL, "--method", "lora")
    code, _, stderr = gf(
        "train", str(table), "--no-input", *flags, monkeypatch=monkeypatch
    )
    assert code == 1 and "error[ValidationError]: name could not be inferred" in stderr
    code, _, stderr = gf(
        "train",
        str(table),
        "--no-input",
        *flags,
        "--name",
        "churn",
        monkeypatch=monkeypatch,
    )
    assert code == cli.ABORTED_EXIT
    assert stderr.endswith(
        "error[Aborted]: `Train?` needs an answer\n"
        "help: pass --yes to go ahead without asking\n"
    )
    assert "  plan     400 steps" in stderr
    assert client.runs.made == []


def test_typed_answers_choose_and_a_no_cancels(monkeypatch, tmp_path, client):
    table = _churn(tmp_path / "churn.csv")
    monkeypatch.setattr(sys, "stdin", io.StringIO("2\n5\n\nretention\nn\n"))
    code, _, stderr = gf("train", str(table), monkeypatch=monkeypatch)
    assert code == cli.ABORTED_EXIT
    assert "Target       z\n" in stderr
    assert f"Base         {MITRA}  API only\n" in stderr
    assert "Method       lora rank 16\n" in stderr
    assert "Name         retention\n" in stderr
    assert stderr.endswith("error[Aborted]: cancelled; nothing was changed\n")
    assert client.runs.made == []


def test_a_ready_snapshot_of_the_same_table_is_reused(monkeypatch, tmp_path, client):
    from generalforecast.commands import read_data

    table = _churn(tmp_path / "churn.csv")
    name = snapshot_name("churn", table_digest(read_data(str(table), tmp_path)))
    client.snapshots.existing[name] = {
        "name": name,
        "state": "ready",
        "rows": 300,
        "metadata": {"target": "churn", "task": "classification"},
    }
    code, _, stderr = gf("train", str(table), "--yes", monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert f"      Reused snapshot {name}  300 rows\n" in stderr
    assert client.snapshots.made == []
    assert client.runs.made[0][1]["data"] == [name]

    client.snapshots.existing[name]["metadata"] = {"target": "z", "task": "regression"}
    code, _, stderr = gf("train", str(table), "--yes", monkeypatch=monkeypatch)
    assert code == 1
    assert (
        f"error[Conflict]: snapshot {name} holds this table for target `z` (regression)\n"
        "help: pass '--name' to train another fine-tune on it\n"
    ) in stderr

    client.snapshots.existing[name] = {"name": name, "state": "deleted"}
    code, _, stderr = gf("train", str(table), "--yes", monkeypatch=monkeypatch)
    assert code == 1
    assert f"snapshot {name} was deleted, and its name is not reused" in stderr


def test_snapshot_names_follow_the_push_rule():
    digest = "sha256:" + "ab" * 32
    assert snapshot_name("churn", digest) == "churn-abababababab"
    long = "a" * 63
    assert snapshot_name(long, digest) == "a" * 50 + "-abababababab"
    assert default_name(Path("Credit Default 2026.csv")) == ("credit-default-2026")
    assert default_name(Path("2026.csv")) == "fine-tune"
    assert default_name(Path("tabicl-v2-classifier.csv")) == ("tabicl-v2-classifier-ft")
    assert name_problem("tabicl-v2-classifier") == "is a catalog model's name"
    assert name_problem("churn") is None


def test_detach_starts_the_run_and_returns(monkeypatch, tmp_path, client):
    table = _churn(tmp_path / "churn.csv")
    code, stdout, stderr = gf(
        "train", str(table), "--yes", "--detach", monkeypatch=monkeypatch
    )
    assert code == 0, stderr
    assert stderr.rstrip("\n").endswith(
        f"     Started acme/churn@ft1  {TABICL}, lora rank 16"
    )
    assert stdout == "next: follow it with `gf runs watch acme/churn@ft1`\n"
    assert client.runs.watched == []
    code, stdout, _ = gf(
        "--json", "train", str(table), "--yes", "--detach", monkeypatch=monkeypatch
    )
    assert code == 0 and json.loads(stdout)["state"] == "queued"


def _follow(events, threshold=2.0):
    err = io.StringIO()
    out = term.Output(err=err, environ={"NO_COLOR": "1"})
    follower = Follower(out, "acme/churn@ft1", threshold)
    follower.follow(events)
    return err.getvalue(), follower


def _check(steps, loss, gain, error):
    body = {"steps": steps, "validation_loss": loss}
    body |= {"check_gain": gain, "check_standard_error": error}
    return {"at": float(steps), "kind": "metrics", "body": body}


def test_a_step_is_kept_only_on_a_significant_gain_and_a_lower_loss():
    text, follower = _follow(
        [
            _check(25, 0.44, 0.010, 0.006),
            _check(50, 0.42, 0.030, 0.007),
            _check(75, 0.43, 0.040, 0.007),
            _check(100, 0.41, 0.045, 0.007),
            {"at": 1.0, "kind": "metrics", "body": {"steps": 110, "loss": 0.4}},
            {"at": 2.0, "kind": "state", "body": {"state": "preempted"}},
            {"at": 3.0, "kind": "resumed", "body": {}},
            {
                "at": 4.0,
                "kind": "state",
                "body": {"state": "cancelled", "reason": "budget"},
            },
        ]
    )
    assert text == (
        "     Checked step 25  select loss 0.4400, gain +0.0100 ± 0.0060\n"
        "     Checked step 50  select loss 0.4200, gain +0.0300 ± 0.0070, kept\n"
        "     Checked step 75  select loss 0.4300, gain +0.0400 ± 0.0070\n"
        "     Checked step 100  select loss 0.4100, gain +0.0450 ± 0.0070, kept\n"
        "   Preempted acme/churn@ft1\n"
        "     Resumed acme/churn@ft1\n"
        "   Cancelled acme/churn@ft1  budget\n"
    )
    assert follower.best == 0.41
    text, _ = _follow([_check(25, 0.44, 0.010, 0.006)], threshold=1.0)
    assert text.endswith(", kept\n")
    text, _ = _follow(
        [{"at": 1.0, "kind": "metrics", "body": {"steps": 25, "validation_loss": 0.5}}]
    )
    assert text == "     Checked step 25  select loss 0.5000\n"


def test_a_published_version_kept_at_step_zero_says_so():
    text, _ = _follow(
        [
            {
                "at": 1.0,
                "kind": "published",
                "body": _published("acme/churn@ft1", kept_step=0),
            }
        ]
    )
    assert text == (
        "   Published acme/churn@ft1  kept step 0, no check beat the base\n"
    )


def test_a_failed_run_ends_in_an_error_with_its_reason(monkeypatch, tmp_path):
    made = _Client(runs=_Runs(ended="failed"))
    monkeypatch.setattr("generalforecast.commands.client_for", lambda args: made)
    original = _events

    def failing(model):
        return [
            *original(model)[:4],
            {"at": 5.0, "kind": "failure", "body": {"message": "step 0 drifted"}},
            {"at": 6.0, "kind": "state", "body": {"state": "failed"}},
        ]

    monkeypatch.setattr(sys.modules[__name__], "_events", failing)
    code, stdout, stderr = gf(
        "runs", "watch", "acme/churn@ft1", monkeypatch=monkeypatch
    )
    assert code == 1 and not stdout
    assert stderr.endswith(
        "    Training acme/churn@ft1\n"
        "error[Internal]: run acme/churn@ft1 failed: calibration\n"
        "note: step 0 drifted\n"
    )


def test_runs_lists_follows_and_cancels(monkeypatch, client):
    code, stdout, stderr = gf("runs", "list", monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert stdout.splitlines() == [
        f"model           state     base{' ' * (len(TABICL) - 4)}  method        kept step",
        f"acme/churn@ft1  finished  {TABICL}  lora rank 16  50",
        f"acme/churn@ft2  training  {TABICL}  lora rank 16  -",
    ]

    code, stdout, stderr = gf("runs", "watch", "acme/churn", monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert client.runs.watched == ["acme/churn@ft1"]
    assert (
        "     Checked step 50  select loss 0.4190, gain +0.0314 ± 0.0072, kept\n"
        in (stderr)
    )
    assert stdout.startswith("version      acme/churn@ft1\n")

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code, _, stderr = gf("runs", "cancel", "acme/churn@ft2", monkeypatch=monkeypatch)
    assert code == cli.ABORTED_EXIT and "`Cancel run acme/churn@ft2?`" in stderr
    assert client.runs.cancelled == []
    code, _, stderr = gf(
        "runs", "cancel", "acme/churn@ft2", "--save", "--yes", monkeypatch=monkeypatch
    )
    assert code == 0, stderr
    assert stderr == "  Cancelling acme/churn@ft2\n"
    assert client.runs.cancelled == [("acme/churn@ft2", True)]

    code, _, stderr = gf("runs", monkeypatch=monkeypatch)
    assert code == 2 and "`gf runs` needs a command" in stderr


def test_train_finds_the_one_table_here(monkeypatch, tmp_path, client):
    monkeypatch.chdir(tmp_path)
    code, _, stderr = gf("train", "--yes", monkeypatch=monkeypatch)
    assert code == 1 and "error[NotFound]: no table here to train on" in stderr
    _churn(tmp_path / "a.csv")
    _churn(tmp_path / "b.csv")
    code, _, stderr = gf("train", "--yes", monkeypatch=monkeypatch)
    assert code == 1
    assert "this directory has 2 tables\nnote: training reads one: a.csv, b.csv\n" in (
        stderr
    )
