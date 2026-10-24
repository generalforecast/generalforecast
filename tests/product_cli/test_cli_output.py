# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The CLI's help, usage errors and output grammar, in process."""

import io
import json
import os
import subprocess
import sys

import pytest

from generalforecast import cli, term
from generalforecast.command_specs import CLOSING, COMMAND_HELP, COMMANDS, GROUPS


class Tty(io.StringIO):
    def isatty(self):
        return True


def gf(*argv, monkeypatch, env=None, tty=False):
    """Runs `gf argv` in process; returns (exit code, stdout, stderr)."""
    out, err = (Tty(), Tty()) if tty else (io.StringIO(), io.StringIO())
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("NO_COLOR", "1")
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    try:
        cli.main(list(argv))
    except SystemExit as error:
        code = error.code or 0
    else:
        code = 0
    return code, out.getvalue(), err.getvalue()


def test_root_help_lists_exactly_the_contract_groups(monkeypatch):
    code, stdout, stderr = gf("--help", monkeypatch=monkeypatch)
    assert code == 0 and not stderr
    expected = (
        "Predictors:\n"
        "  init       Write a predictor manifest here\n"
        "  check      Resolve the schema and write gf.lock\n"
        "  eval       Score the held-out rows\n"
        "  serve      Serve a predictor on this machine\n"
        "  predict    Score rows with a predictor\n"
        "\n"
        "Hosted:\n"
        "  login      Log in through your browser\n"
        "  logout     Remove this machine's credentials\n"
        "  whoami     Show the account and project in use\n"
        "  push       Deploy a predictor to an environment\n"
        "  status     Show a predictor's environments\n"
        "  promote    Point an environment at another's revision\n"
        "  rollback   Point an environment at an earlier revision\n"
        "\n"
        "Resources:\n"
        "  model      Catalog models: list, show, pull\n"
        "  context    Live contexts: create, append, label, show\n"
        "  admin      Keys, secrets, projects, nodes and routers\n"
    )
    assert expected in stdout
    assert stdout.rstrip("\n").endswith(CLOSING)
    assert len(COMMANDS) == 15 and [len(names) for _, names in GROUPS] == [5, 7, 3]
    assert all(len(text) < 50 for text in COMMAND_HELP.values())


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_help_ends_with_the_closing_line(monkeypatch, command):
    code, stdout, _ = gf(command, "--help", monkeypatch=monkeypatch)
    assert code == 0
    assert stdout.startswith(COMMAND_HELP[command] + "\n\nUsage: gf " + command)
    assert stdout.rstrip("\n").endswith(CLOSING)


def test_help_command_matches_dash_dash_help(monkeypatch):
    _, via_flag, _ = gf("eval", "--help", monkeypatch=monkeypatch)
    _, via_help, _ = gf("help", "eval", monkeypatch=monkeypatch)
    assert via_flag == via_help


def test_admin_lists_every_noun_and_nothing_is_hidden(monkeypatch):
    _, stdout, _ = gf("admin", "--help", monkeypatch=monkeypatch)
    for noun in (
        "key",
        "secret",
        "workspace",
        "project",
        "organization",
        "membership",
        "sso",
        "audit",
        "operator",
        "source",
        "node",
        "router",
        "logs",
        "diff",
        "delete",
    ):
        assert f"\n  {noun} " in stdout


def test_retired_flags_and_commands_are_gone(monkeypatch):
    for argv in (
        ("check", "--data", "x.csv"),
        ("predict", "--queries", "q.csv", "--features", "a"),
    ):
        code, _, stderr = gf(*argv, monkeypatch=monkeypatch)
        assert code == 2 and stderr.startswith("error: unexpected argument '--")
    for command in ("dataset", "datasets", "bench", "run"):
        code, stdout, stderr = gf("--json", command, monkeypatch=monkeypatch)
        assert code == 2 and not stdout
        assert json.loads(stderr) == {
            "error": {"code": "UsageError", "message": f"unknown command `{command}`"}
        }
    _, stdout, _ = gf("serve", "--help", monkeypatch=monkeypatch)
    for flag in ("--node", "--config", "--init", "--upstream", "--force"):
        assert flag not in stdout
    assert "--host" in stdout and "--port" in stdout and "--offline" in stdout


def test_typos_get_a_tip(monkeypatch):
    code, _, stderr = gf("chekc", monkeypatch=monkeypatch)
    assert code == 2
    assert (
        stderr
        == "error: unknown command `chekc`\n  tip: a similar command exists: 'check'\n"
    )
    code, _, stderr = gf("init", "--contxt", "x", monkeypatch=monkeypatch)
    assert stderr == (
        "error: unexpected argument '--contxt'\n  tip: a similar argument exists: '--context'\n"
    )


def test_required_arguments_align_under_the_label(monkeypatch):
    code, _, stderr = gf("predict", monkeypatch=monkeypatch)
    assert code == 2
    assert stderr == (
        "error: the following required arguments were not provided:\n"
        "       --queries <FILE>\n"
        "  hint: run 'gf predict --help' for usage\n"
    )


def test_exclusive_flags(monkeypatch):
    code, _, stderr = gf(
        "init", "--until", "2025-01-01", "--holdout", "0.2", monkeypatch=monkeypatch
    )
    assert code == 2
    assert stderr.startswith(
        "error: the argument '--holdout' cannot be used with '--until'"
    )


def test_json_errors_print_only_the_error_object(monkeypatch, tmp_path):
    (tmp_path / "gf.yaml").write_text(
        "name: x\nmodel: tabicl-classifier\ntarget: y\ncontext:\n  snapshot: c.csv\n"
    )
    (tmp_path / "c.csv").write_text("a,y\n1,0\n")
    code, stdout, stderr = gf("--json", "check", str(tmp_path), monkeypatch=monkeypatch)
    assert code == 1 and not stdout
    body = json.loads(stderr)
    assert body["error"]["code"] == "ValidationError"
    assert body["error"]["message"] == (
        "`tabicl-classifier` is not a catalog model id; "
        "use general-forecast/tabicl-v2-classifier"
    )
    assert "hint" not in body["error"]
    assert set(body["error"]) <= {
        "code",
        "message",
        "param",
        "hint",
        "limit",
        "actual",
        "request_id",
    }


def test_bare_model_names_are_rejected_with_the_canonical_hint(monkeypatch):
    code, _, stderr = gf(
        "model", "show", "tabicl-v2-classifier", monkeypatch=monkeypatch
    )
    assert code == 1
    assert stderr == (
        "error: `tabicl-v2-classifier` is not a catalog model id; "
        "use general-forecast/tabicl-v2-classifier\n"
    )


def test_internal_errors_never_show_a_traceback_without_verbose(monkeypatch):
    def boom(args, out):
        raise RuntimeError("secret internals")

    monkeypatch.setattr("generalforecast.commands.execute", boom)
    code, stdout, stderr = gf("model", "list", monkeypatch=monkeypatch)
    assert code == 1 and not stdout
    assert stderr == "error: internal error\n  hint: run again with -v\n"
    code, _, stderr = gf("-v", "model", "list", monkeypatch=monkeypatch)
    assert "RuntimeError: secret internals" in stderr
    assert stderr.endswith("error: internal error\n  hint: run again with -v\n")


def test_status_verbs_are_right_aligned_in_twelve_columns():
    err = io.StringIO()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.status("Initialized", "predictor churn in ./gf.yaml")
    out.status("Resolved", "churn  classification, 3 inputs", seconds=None)
    out.status("Locked", "gf.lock", seconds=0.21)
    out.status("Compiled", "12 programs", seconds=6.8)
    out.status("Deployed", "churn revision 7 to staging", seconds=41)
    assert err.getvalue() == (
        " Initialized predictor churn in ./gf.yaml\n"
        "    Resolved churn  classification, 3 inputs\n"
        "      Locked gf.lock in 0.21s\n"
        "    Compiled 12 programs in 6.8s\n"
        "    Deployed churn revision 7 to staging in 41s\n"
    )


def test_lock_change_lines():
    err = io.StringIO()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.changes(
        [
            ("~", "model", "general-forecast/tabicl-v2-classifier 5fb6a4c", "7a01c2e"),
            ("+", "input", None, "signup_source"),
            ("~", "selection", None, "core 1,000 → 1,500, neighbors 3,000 → 4,500"),
        ]
    )
    assert err.getvalue() == (
        " ~ model      general-forecast/tabicl-v2-classifier 5fb6a4c → 7a01c2e\n"
        " + input      signup_source\n"
        " ~ selection  core 1,000 → 1,500, neighbors 3,000 → 4,500\n"
    )


def test_error_shapes():
    err = io.StringIO()
    out = term.Output(out=io.StringIO(), err=err, environ={"NO_COLOR": "1"})
    out.error(
        "gf.yaml already exists in ./churn",
        hint="pass --force to overwrite it, or run `gf check` to validate it",
    )
    out.error(
        "target column `churned` is not in the context",
        pointer=term.Pointer(
            "gf.yaml",
            3,
            9,
            "target: churned",
            "context.csv has customer_id, spend, plan, churn",
            7,
        ),
        hint="did you mean `churn`?",
    )
    out.error(
        "queries exceed the limit for general-forecast/tabicl-v2-classifier",
        because="this model accepts 100 queries per call; the request sent 1,000",
        hint="split the queries, or push a predictor and call it in batches",
        request_id="r-3abb2401f119428db148927c6a0d358c",
    )
    assert err.getvalue() == (
        "error: gf.yaml already exists in ./churn\n"
        "  hint: pass --force to overwrite it, or run `gf check` to validate it\n"
        "error: target column `churned` is not in the context\n"
        "  --> gf.yaml:3:9\n"
        "   |\n"
        " 3 | target: churned\n"
        "   |         ^^^^^^^ context.csv has customer_id, spend, plan, churn\n"
        "  hint: did you mean `churn`?\n"
        "error: queries exceed the limit for general-forecast/tabicl-v2-classifier\n"
        "  because: this model accepts 100 queries per call; the request sent 1,000\n"
        "  hint: split the queries, or push a predictor and call it in batches\n"
        "  request r-3abb2401f119428db148927c6a0d358c\n"
    )


def test_colour_follows_the_terminal_and_no_color():
    assert not term.color_enabled(io.StringIO(), {})
    assert term.color_enabled(Tty(), {})
    assert not term.color_enabled(Tty(), {"NO_COLOR": "1"})
    assert term.color_enabled(io.StringIO(), {"FORCE_COLOR": "1"})
    err = Tty()
    out = term.Output(out=io.StringIO(), err=err, environ={})
    out.status("Locked", "gf.lock")
    out.error("x", hint="y", request_id="r-1")
    text = err.getvalue()
    assert "\x1b[1;32m      Locked\x1b[0m gf.lock" in text
    assert "\x1b[1;31merror:\x1b[0m x" in text
    assert "\x1b[1mhint:\x1b[0m y" in text
    assert "\x1b[2m  request r-1\x1b[0m" in text


def test_help_and_version_do_not_import_the_sdk():
    code = (
        "import sys; from generalforecast import cli\n"
        "try:\n    cli.main(sys.argv[1:])\nexcept SystemExit:\n    pass\n"
        "heavy = [m for m in ('pyarrow', 'numpy', 'pandas', 'generalforecast.client', "
        "'generalforecast.predictor_commands') if m in sys.modules]; "
        "sys.stdout.write('\\nHEAVY=' + ','.join(heavy))"
    )
    for argv in (["--help"], ["--version"], ["eval", "--help"]):
        result = subprocess.run(
            [sys.executable, "-c", code, *argv],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        assert result.stdout.endswith("HEAVY="), result.stdout[-200:]


def test_rust_launcher_runs_the_python_cli(tmp_path):
    binary = os.environ.get("GF_CLI")
    if not binary or not os.path.isfile(binary):
        pytest.skip("set GF_CLI to a built gf launcher for parity")
    ours = subprocess.run(
        [sys.executable, "-m", "generalforecast.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    theirs = subprocess.run(
        [binary, "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GF_PYTHON": sys.executable},
    )
    assert ours.stdout == theirs.stdout


def _write_rows(path, rows=60, *, timed=False, blanks=0):
    lines = ["id,x,t,y" if timed else "id,x,y"]
    for i in range(rows):
        y = "" if i < blanks else str(i % 3)
        when = f"2025-01-{1 + i % 28:02d}," if timed else ""
        lines.append(f"{i},{i / 10},{when}{y}")
    path.write_text("\n".join(lines) + "\n")


def _manifest(root):
    import yaml

    return yaml.safe_load((root / "gf.yaml").read_text())


def test_init_yes_picks_a_query_rule_whose_precondition_holds(monkeypatch, tmp_path):
    from generalforecast.models.catalog import DEFAULTS

    flat = tmp_path / "flat"
    flat.mkdir()
    _write_rows(flat / "rows.csv")
    code, _, stderr = gf("init", "--yes", str(flat), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert "Task         classification  y has 3 distinct int values" in stderr
    assert f"Model        {DEFAULTS['classification']}" in stderr
    assert "Queries      a random fraction" in stderr
    assert "Holdout      0.2  12 rows" in stderr
    assert _manifest(flat)["context"] == {
        "snapshot": "rows.csv",
        "holdout": {"fraction": 0.2, "seed": 0},
    }

    blank = tmp_path / "blank"
    blank.mkdir()
    _write_rows(blank / "rows.csv", blanks=6)
    code, _, stderr = gf("init", "--yes", str(blank), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert "Queries      6 rows with no target" in stderr

    timed = tmp_path / "timed"
    timed.mkdir()
    _write_rows(timed / "rows.csv", timed=True)
    code, _, stderr = gf("init", "--yes", str(timed), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert "Queries      rows after a date" in stderr
    manifest = _manifest(timed)
    assert manifest["time"] == "t" and manifest["context"]["until"] == "2025-01-28"
    code, _, stderr = gf("check", str(timed), monkeypatch=monkeypatch)
    assert code == 0, stderr


def test_init_until_without_a_time_column_names_the_flag(monkeypatch, tmp_path):
    _write_rows(tmp_path / "rows.csv")
    code, _, stderr = gf(
        "init", "--yes", "--until", "2025-01-01", str(tmp_path), monkeypatch=monkeypatch
    )
    assert code == 1
    assert stderr.endswith(
        "error: --until needs a time column\n"
        "  because: rows.csv has no timestamp column to compare dates against\n"
        "  hint: pass --holdout FRACTION for a random split\n"
    )


def test_init_task_and_model_override_the_inferred_task(monkeypatch, tmp_path):
    from generalforecast.models.catalog import DEFAULTS

    _write_rows(tmp_path / "rows.csv")
    code, _, stderr = gf(
        "init", "--yes", "--task", "regression", str(tmp_path), monkeypatch=monkeypatch
    )
    assert code == 0, stderr
    assert "Task         regression  from --task" in stderr
    assert f"Model        {DEFAULTS['regression']}" in stderr
    code, _, stderr = gf(
        "init",
        "--yes",
        "--force",
        "--model",
        "general-forecast/tabicl-v2-regressor",
        "--queries",
        "none",
        str(tmp_path),
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    assert "Task         regression  from --model" in stderr
    assert "Queries      none" in stderr
    assert _manifest(tmp_path)["model"] == "general-forecast/tabicl-v2-regressor"


def test_the_default_holdout_fits_one_call():
    from generalforecast.predictor_commands import _holdout_default, _holdout_note

    assert _holdout_default(178, 100) == "0.2"
    assert _holdout_note(0.2, 178, 100, "m") == "36 rows"
    capped = _holdout_default(2052, 100)
    assert round(float(capped) * 2052) == 100
    assert _holdout_note(float(capped), 2052, 100, "m") == (
        "100 rows, the most m scores in one call"
    )


def test_wine_init_check_eval_end_to_end(monkeypatch, tmp_path):
    import pandas as pd
    from sklearn.datasets import load_wine

    import generalforecast.predictor as predictor_module
    from generalforecast.models.catalog import DEFAULTS
    from generalforecast.predictors import _registry

    root = tmp_path / "wine"
    root.mkdir()
    frame = load_wine(as_frame=True).frame
    frame.to_csv(root / "wine.csv", index=False)
    batches = []

    def fake_run(request, *, download=True):
        batches.append(len(request.queries))
        truth = request.queries.column("target").to_pylist()
        columns = {"label": truth}
        for label in request.classes:
            columns[f"p_{label}"] = [1.0 if t == label else 0.0 for t in truth]
        return pd.DataFrame(columns)

    monkeypatch.setattr(predictor_module, "run_local", fake_run)

    code, _, stderr = gf("init", "--yes", str(root), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert "Task         classification  target has 3 distinct int values" in stderr
    assert f"Model        {DEFAULTS['classification']}" in stderr
    assert "Queries      a random fraction" in stderr
    assert "Holdout      0.2  36 rows" in stderr
    code, _, stderr = gf("check", str(root), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert "142 context rows, 36 query rows" in stderr
    code, stdout, stderr = gf("eval", str(root), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert batches == [36]
    assert "Predicted 36 rows" in stderr
    assert "accuracy   1.000" in stdout

    cap = _registry.limits(DEFAULTS["classification"]).queries
    held = frame.sample(cap + 25, random_state=0)
    held.to_csv(tmp_path / "held.csv", index=False)
    batches.clear()
    code, stdout, stderr = gf(
        "eval",
        str(root),
        "--queries",
        str(tmp_path / "held.csv"),
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    assert batches == [cap, 25]
    assert f"Predicted {cap + 25} rows" in stderr

    code, _, stderr = gf(
        "init",
        "--yes",
        "--force",
        "--queries",
        "none",
        str(root),
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    assert "Queries      none" in stderr
    code, _, stderr = gf("check", str(root), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert "178 context rows, 0 query rows" in stderr
    code, _, stderr = gf("eval", str(root), monkeypatch=monkeypatch)
    assert code == 1
    assert "\nerror: predictor wine has no query rows to score\n" in stderr
    assert "`gf eval --queries FILE`" in stderr
