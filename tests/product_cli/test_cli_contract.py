# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The CLI's output contract: who writes stdout, exit codes, errors and prompts."""

import argparse
import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import ClassVar

import pytest

from generalforecast import cli, term
from tests.product_cli.test_cli_output import _write_rows, gf

ROOT = Path(__file__).resolve().parents[2]
TABICL = "general-forecast/tabicl-v2-classifier"


def _returns(monkeypatch, result):
    monkeypatch.setattr("generalforecast.commands.execute", lambda args, out: result)


def _raises(monkeypatch, error):
    def execute(args, out):
        raise error

    monkeypatch.setattr("generalforecast.commands.execute", execute)


# -- flags ------------------------------------------------------------------


def test_flags_are_never_abbreviated(monkeypatch):
    _returns(monkeypatch, {"models": []})
    code, stdout, stderr = gf("model", "list", "--js", monkeypatch=monkeypatch)
    assert code == 2 and not stdout
    assert stderr == (
        "error: unexpected argument '--js'\n  tip: a similar argument exists: '--json'\n"
    )


def test_global_flags_come_from_the_parser_even_combined(monkeypatch):
    class Registry:
        def canonical(self, ref):
            return ref

        def is_cached(self, ref):
            return True

    monkeypatch.setattr("generalforecast.commands.model_registry", Registry)
    code, stdout, stderr = gf("model", "pull", TABICL, "-qv", monkeypatch=monkeypatch)
    assert code == 0 and stderr == ""
    code, stdout, stderr = gf("model", "pull", TABICL, monkeypatch=monkeypatch)
    assert stderr == f"      Cached {TABICL}\n"
    code, stdout, _ = gf(
        "model", "-q", "pull", "--json", TABICL, monkeypatch=monkeypatch
    )
    assert json.loads(stdout) == {"id": TABICL, "cached": True}


def test_model_pull_reports_the_download_through_the_status_lines(monkeypatch):
    class Registry:
        def canonical(self, ref):
            return ref

        def is_cached(self, ref):
            return False

        def pull(self, ref, *, progress):
            progress("Pulling", f"{ref} (2 files, 1.2 GB)")
            return Path("/cache") / ref

    monkeypatch.setattr("generalforecast.commands.model_registry", Registry)
    code, _, stderr = gf("model", "pull", TABICL, monkeypatch=monkeypatch)
    assert code == 0
    assert stderr.splitlines() == [
        f"     Pulling {TABICL} (2 files, 1.2 GB)",
        f"      Pulled {TABICL} to /cache/{TABICL}",
    ]
    _, _, stderr = gf("-q", "model", "pull", TABICL, monkeypatch=monkeypatch)
    assert stderr == ""


# -- results ----------------------------------------------------------------


def test_json_results_turn_values_json_lacks_into_null(monkeypatch):
    _returns(monkeypatch, {"auroc": float("nan"), "big": float("inf"), "n": 3})
    code, stdout, stderr = gf("--json", "model", "list", monkeypatch=monkeypatch)
    assert code == 0 and stderr == ""
    assert json.loads(stdout) == {"auroc": None, "big": None, "n": 3}


def test_a_result_that_cannot_be_written_is_an_error_not_a_traceback(monkeypatch):
    _returns(monkeypatch, {"value": object()})
    code, stdout, stderr = gf("--json", "model", "list", monkeypatch=monkeypatch)
    assert code == 1 and stdout == ""
    assert json.loads(stderr)["error"]["code"] == "Internal"


def _closed_stdout(code: str) -> subprocess.CompletedProcess:
    read, write = os.pipe()
    os.close(read)
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            stdout=write,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
            timeout=120,
            check=False,
        )
    finally:
        os.close(write)


def test_a_closed_pipe_ends_the_command_quietly():
    done = _closed_stdout("from generalforecast import cli; cli.main(['--help'])")
    assert done.returncode == cli.BROKEN_PIPE_EXIT and done.stderr == ""
    done = _closed_stdout(
        textwrap.dedent(
            """
            from generalforecast import cli, commands
            rows = [{"id": str(i), "task": "x"} for i in range(5000)]
            commands.execute = lambda args, out: {"models": rows}
            cli.main(["model", "list"])
            """
        )
    )
    assert done.returncode == cli.BROKEN_PIPE_EXIT and done.stderr == ""


def test_ctrl_c_exits_130_with_an_interrupted_error(monkeypatch):
    _raises(monkeypatch, KeyboardInterrupt())
    code, stdout, stderr = gf("--json", "model", "list", monkeypatch=monkeypatch)
    assert code == cli.INTERRUPTED_EXIT and not stdout
    assert json.loads(stderr)["error"]["reason"] == "interrupted"
    code, _, stderr = gf("model", "list", monkeypatch=monkeypatch)
    assert code == cli.INTERRUPTED_EXIT and stderr == "\nerror: interrupted\n"


# -- errors -----------------------------------------------------------------


def _folded(status: int, envelope: dict) -> BaseException:
    """The error the SDK raises for a service's folded error envelope."""
    from generalforecast.transport import response_value

    try:
        response_value(status, json.dumps({"error": envelope}).encode())
    except Exception as error:  # noqa: BLE001
        return error
    raise AssertionError("the envelope did not raise")


@pytest.mark.parametrize(
    ("status", "envelope", "hint"),
    [
        (
            401,
            {
                "code": "Unauthorized",
                "reason": "not_logged_in",
                "message": "not logged in",
            },
            "run `gf login`",
        ),
        (
            401,
            {
                "code": "Unauthorized",
                "reason": "expired_token",
                "message": "expired key",
            },
            "run `gf login`",
        ),
        (
            403,
            {"code": "Unauthorized", "reason": "forbidden", "message": "no access"},
            "run `gf login` again, or use a key that has this permission",
        ),
        (
            503,
            {
                "code": "ModelNotReady",
                "reason": "wait_timeout",
                "message": "warming up",
            },
            None,
        ),
    ],
)
def test_folded_service_errors_read_by_code_and_reason(
    monkeypatch, status, envelope, hint
):
    _raises(monkeypatch, _folded(status, envelope))
    code, stdout, stderr = gf("--json", "model", "list", monkeypatch=monkeypatch)
    assert code == 1 and not stdout
    body = json.loads(stderr)["error"]
    assert (body["code"], body["reason"], body["message"]) == (
        envelope["code"],
        envelope["reason"],
        envelope["message"],
    )
    _, _, stderr = gf("model", "list", monkeypatch=monkeypatch)
    assert stderr.startswith(f"error: {envelope['message']}\n")
    assert (f"  hint: {hint}\n" in stderr) == (hint is not None)


def test_not_logged_in_is_one_error_everywhere(monkeypatch, tmp_path):
    from generalforecast.credentials import CredentialStore

    monkeypatch.setattr(CredentialStore, "resolve", lambda self, server: None)
    for name in ("GF_API_KEY", "GF_PROFILE", "GF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GF_CONFIG_PATH", str(tmp_path / "profiles.json"))
    bodies = []
    for argv in (("whoami",), ("status", "churn"), ("admin", "key", "list")):
        code, _, stderr = gf("--json", *argv, monkeypatch=monkeypatch)
        assert code == 1
        bodies.append(json.loads(stderr))
    assert (
        bodies[0]
        == bodies[1]
        == bodies[2]
        == {
            "error": {
                "code": "Unauthorized",
                "reason": "not_logged_in",
                "message": "not logged in",
                "hint": "run `gf login`",
            }
        }
    )


def test_an_error_points_into_the_named_predictors_manifest(monkeypatch, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_rows(sub / "rows.csv")
    (sub / "gf.yaml").write_text(
        "name: sub\nmodel: general-forecast/nope-model\ntarget: y\n"
        "context:\n  snapshot: rows.csv\n"
    )
    (tmp_path / "gf.yaml").write_text(f"name: top\nmodel: {TABICL}\ntarget: y\n")
    monkeypatch.chdir(tmp_path)
    code, _, stderr = gf("check", "sub", monkeypatch=monkeypatch)
    assert code == 1
    assert " 2 | model: general-forecast/nope-model\n" in stderr
    assert TABICL not in stderr


def test_a_declined_or_unanswered_question_says_so(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code, stdout, stderr = gf(
        "--json", "admin", "delete", "churn", monkeypatch=monkeypatch
    )
    assert code == 1 and not stdout
    error = json.loads(stderr.splitlines()[-1])["error"]
    assert error == {
        "code": "ValidationError",
        "reason": "aborted",
        "message": "`Delete churn and its environments?` needs an answer",
        "hint": "pass --yes to go ahead without asking",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
    code, _, stderr = gf("admin", "delete", "churn", monkeypatch=monkeypatch)
    assert code == 1
    assert stderr == (
        "Delete churn and its environments? [y/N] "
        "error: cancelled; nothing was changed\n"
    )


# -- admin ------------------------------------------------------------------


class _Identity:
    def __init__(self):
        self.calls = []
        self.identity = self

    def manage(self, action, payload=None):
        self.calls.append((action, payload))
        return {"ok": True}


def test_admin_actions_bind_name_and_file_and_confirm_what_they_destroy(
    monkeypatch, tmp_path
):
    client = _Identity()
    monkeypatch.setattr("generalforecast.commands.client_for", lambda args: client)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code, _, stderr = gf("admin", "key", "revoke", "k-2f91", monkeypatch=monkeypatch)
    assert code == 1 and "needs an answer" in stderr and not client.calls
    code, _, _ = gf(
        "admin", "key", "revoke", "k-2f91", "--yes", monkeypatch=monkeypatch
    )
    assert code == 0 and client.calls.pop() == ("keys.revoke", {"id": "k-2f91"})
    code, _, _ = gf("admin", "secret", "get", "db", monkeypatch=monkeypatch)
    assert code == 0 and client.calls.pop() == ("secrets.get", {"name": "db"})
    code, _, _ = gf("admin", "workspace", "get", monkeypatch=monkeypatch)
    assert code == 0 and client.calls.pop() == ("workspace.get", {})
    body = tmp_path / "rotate.json"
    body.write_text('{"value": "s3cret", "expected_version": 2}')
    code, _, _ = gf(
        "admin", "secret", "rotate", "db", "--file", str(body), monkeypatch=monkeypatch
    )
    assert code == 0 and client.calls.pop() == (
        "secrets.rotate",
        {"value": "s3cret", "expected_version": 2, "name": "db"},
    )
    body.write_text('{"name": "other", "value": "x", "expected_version": 1}')
    code, _, stderr = gf(
        "admin", "secret", "rotate", "db", "--file", str(body), monkeypatch=monkeypatch
    )
    assert code == 1 and stderr.startswith("error: the file's `name` is not `db`")
    code, _, stderr = gf("admin", "key", "create", monkeypatch=monkeypatch)
    assert code == 2 and "--file <FILE>" in stderr
    code, _, _ = gf("admin", "audit", "list", "--limit", "5", monkeypatch=monkeypatch)
    assert code == 0 and client.calls.pop() == ("audit.list", {"after": 0, "limit": 5})


# -- predict and eval -------------------------------------------------------


class _Recorder:
    made: ClassVar[list] = []

    def __init__(self, reference, **options):
        self.made.append((reference, options))

    def predict(self, queries, context=None):
        import pandas as pd

        return pd.DataFrame({"id": [1, 2], "label": [0, 1], "p_1": [0.25, 0.75]})


@pytest.fixture
def recorder(monkeypatch):
    _Recorder.made = []
    monkeypatch.setattr(
        "generalforecast.predictor_commands.predictor_class", lambda: _Recorder
    )
    return _Recorder


def test_predict_on_a_catalog_model_takes_its_flags(monkeypatch, tmp_path, recorder):
    monkeypatch.chdir(tmp_path)
    _write_rows(tmp_path / "rows.csv")
    code, _, stderr = gf(
        "predict", TABICL, "--queries", "rows.csv", monkeypatch=monkeypatch
    )
    assert code == 1 and stderr.startswith("error: a catalog model needs a target\n")
    code, _, stderr = gf(
        "predict",
        TABICL,
        "--queries",
        "rows.csv",
        "--target",
        "y",
        monkeypatch=monkeypatch,
    )
    assert code == 1 and "pass '--context FILE'" in stderr
    code, stdout, stderr = gf(
        "predict",
        TABICL,
        "--queries",
        "rows.csv",
        "--target",
        "y",
        "--inputs",
        "x",
        "--context",
        "rows.csv",
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    assert recorder.made == [(TABICL, {"env": None, "target": "y", "inputs": ["x"]})]
    assert stdout == "id,label,p_1\n1,0,0.25\n2,1,0.75\n"


def test_predict_names_a_directory_and_a_hosted_name_apart(
    monkeypatch, tmp_path, recorder
):
    churn = tmp_path / "churn"
    churn.mkdir()
    (churn / "gf.yaml").write_text("name: churn\n")
    monkeypatch.chdir(tmp_path)
    code, _, stderr = gf(
        "predict", "churn", "--queries", "q.csv", monkeypatch=monkeypatch
    )
    assert code == 1
    assert stderr == (
        "error: `churn` is both a directory with gf.yaml and a hosted predictor name\n"
        "  hint: pass ./churn for the local predictor, "
        "or --env production for the hosted one\n"
    )
    code, _, _ = gf(
        "predict",
        "churn",
        "--queries",
        "q.csv",
        "--env",
        "staging",
        monkeypatch=monkeypatch,
    )
    assert code == 0 and recorder.made[-1][1]["env"] == "staging"
    code, _, _ = gf(
        "predict", "elsewhere", "--queries", "q.csv", monkeypatch=monkeypatch
    )
    assert code == 0 and recorder.made[-1] == (
        "elsewhere",
        {"env": "production", "target": None, "inputs": None},
    )


def test_predict_checks_its_output_before_scoring(monkeypatch, tmp_path, recorder):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "q.csv").write_text("id\n1\n")
    code, _, stderr = gf(
        "predict",
        "elsewhere",
        "--queries",
        "q.csv",
        "--output",
        "nodir/out.csv",
        monkeypatch=monkeypatch,
    )
    assert code == 1 and not recorder.made
    assert stderr == (
        "error: cannot write `nodir/out.csv`: directory `nodir` does not exist\n"
    )
    code, _, stderr = gf(
        "predict",
        "elsewhere",
        "--queries",
        "q.csv",
        "--output",
        "q.csv",
        monkeypatch=monkeypatch,
    )
    assert code == 1 and "is also an input" in stderr
    assert (tmp_path / "q.csv").read_text() == "id\n1\n"
    code, stdout, _ = gf(
        "predict",
        "elsewhere",
        "--queries",
        "q.csv",
        "--output",
        "out.parquet",
        monkeypatch=monkeypatch,
    )
    assert code == 0 and stdout == ""
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
    umask = os.umask(0)
    os.umask(umask)
    assert (tmp_path / "out.parquet").stat().st_mode & 0o777 == 0o666 & ~umask


def test_a_write_failure_names_the_file(monkeypatch):
    error = cli._oserror(OSError("Cannot save file into a non-existent directory"))
    assert error.message == "cannot save file into a non-existent directory"
    error = cli._oserror(PermissionError(13, "Permission denied", "out.csv"))
    assert error.message == "cannot open `out.csv`: permission denied"


def test_push_waits_for_readiness_by_wait_not_timeout():
    from generalforecast.errors import Error
    from generalforecast.predictor_commands import PredictorCommands

    class Starting:
        status, name, predictor = "starting", "production", "churn"

    commands = object.__new__(PredictorCommands)
    commands.args = argparse.Namespace(wait=0.0, timeout=40.0)
    with pytest.raises(Error) as raised:
        commands.wait(Starting())
    assert raised.value.message == "environment production is not ready yet"
    assert "'--wait'" in (raised.value.hint or "")


# -- prompts ----------------------------------------------------------------


def test_typed_answers_are_checked_before_they_are_used(monkeypatch, tmp_path):
    series = tmp_path / "sales"
    series.mkdir()
    (series / "sales.csv").write_text(
        "date,store,units\n"
        + "".join(f"2025-01-{d:02d},{s},{d}\n" for d in range(1, 29) for s in "ab")
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n\n\nseven\n"))
    code, _, stderr = gf(
        "init",
        "--target",
        "units",
        "--task",
        "forecast",
        str(series),
        monkeypatch=monkeypatch,
    )
    assert code == 1
    assert stderr.endswith(
        "error: horizon must be a whole number of steps\n  hint: pass '--horizon'\n"
    )


def test_the_until_prompt_reads_the_table_once(monkeypatch, tmp_path):
    from generalforecast import predictor_commands

    _write_rows(tmp_path / "rows.csv", timed=True)
    reads = []
    table = predictor_commands._table

    def counted(path):
        reads.append(path)
        return table(path)

    class Typing(term.Prompter):
        def __init__(self, out, **options):
            super().__init__(out, **options, interactive=True)

        def _raw(self, label, options, default, *, multi):
            if label == "Queries":
                return [next(i for i, o in enumerate(options) if o.value == "date")]
            return sorted(multi) if multi is not None else [default]

        def _edit(self, label, default, aside, problem):
            for typed in ("2025-01-1", "2025-01-15", "2025-01-2", "2025-01-20"):
                aside(typed)
            return "2025-01-20"

    monkeypatch.setattr(predictor_commands, "_table", counted)
    monkeypatch.setattr(term, "Prompter", Typing)
    code, _, stderr = gf(
        "init",
        "--target",
        "y",
        "--model",
        TABICL,
        str(tmp_path),
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    assert len(reads) == 1
    assert "Until        2025-01-20  18 rows" in stderr


def test_a_selector_with_nothing_to_offer_asks_nothing():
    err = io.StringIO()
    out = term.Output(err=err, environ={"NO_COLOR": "1"})
    prompter = term.Prompter(out, stdin=io.StringIO(""), interactive=True)
    assert prompter.choose_many("Inputs", [], flag="--inputs") == []
    assert err.getvalue() == "Inputs       none\n"


def test_raw_keys_read_whole_characters_and_whole_escape_sequences():
    read, write = os.pipe()
    os.write(write, "é".encode() + b"\x1b[3~" + b"\x1b[A" + b"\x1bOB" + "日".encode())
    with os.fdopen(read, "r") as stdin:
        prompter = term.Prompter(term.Output(), stdin=stdin, interactive=False)
        keys = [prompter._key() for _ in range(5)]
    os.close(write)
    assert keys == ["é", "other", "up", "down", "日"]


def test_frames_fit_the_terminal_whatever_the_characters():
    assert term.visible("日本") == 4 and term.visible("é") == 1
    assert term.visible("\x1b[2mdim\x1b[0m") == 3
    assert term.clip("abcdef", 4) == "abcd"
    assert term.clip("日本語", 5) == "日本"
    assert term.clip("\x1b[2mabcdef\x1b[0m", 3) == "\x1b[2mabc\x1b[0m"
    assert term.pad("日", 4) == "日  "


def test_output_never_overwrites_what_the_predictor_reads(
    monkeypatch, tmp_path, recorder
):
    root = tmp_path / "churn"
    root.mkdir()
    _write_rows(root / "rows.csv")
    (root / "gf.yaml").write_text(
        f"name: churn\nmodel: {TABICL}\ntarget: y\ncontext:\n  snapshot: rows.csv\n"
    )
    before = (root / "rows.csv").read_text()
    for argv in (
        ("eval", str(root), "--output", str(root / "rows.csv")),
        (
            "predict",
            str(root),
            "--queries",
            "q.csv",
            "--output",
            str(root / "rows.csv"),
        ),
        ("predict", str(root), "--queries", "q.csv", "--output", str(root / "gf.yaml")),
        ("predict", str(root), "--queries", "q.csv", "--output", str(tmp_path)),
    ):
        code, _, stderr = gf(*argv, monkeypatch=monkeypatch)
        assert code == 1, argv
        assert "is also an input" in stderr or "is a directory" in stderr, stderr
    assert (root / "rows.csv").read_text() == before
    assert not recorder.made


def test_json_errors_keep_why_and_the_near_miss(monkeypatch):
    from generalforecast.errors import InvalidInput

    _raises(monkeypatch, InvalidInput("gf.lock would change", because="2 changes"))
    _, _, stderr = gf("--json", "model", "list", monkeypatch=monkeypatch)
    assert json.loads(stderr)["error"]["because"] == "2 changes"
    code, _, stderr = gf(
        "--json", "admin", "key", "list", "--jso", monkeypatch=monkeypatch
    )
    assert code == 2
    assert json.loads(stderr)["error"]["tip"] == "a similar argument exists: '--json'"


def test_an_unknown_admin_noun_is_called_a_noun(monkeypatch):
    code, _, stderr = gf("admin", "keys", monkeypatch=monkeypatch)
    assert code == 2
    assert stderr == (
        "error: unknown noun `keys` for `gf admin`\n"
        "  tip: a similar noun exists: 'key'\n"
    )


def test_only_ascii_digits_make_a_horizon():
    from generalforecast.predictor_commands import _horizon_problem

    assert _horizon_problem("7") is None
    assert _horizon_problem("²") == _horizon_problem("0") == _horizon_problem("")
    assert _horizon_problem("²") == "must be a whole number of steps"


def test_audit_pages_are_checked_before_the_request(monkeypatch):
    client = _Identity()
    monkeypatch.setattr("generalforecast.commands.client_for", lambda args: client)
    code, _, stderr = gf(
        "admin", "audit", "list", "--limit", "500", monkeypatch=monkeypatch
    )
    assert code == 1 and stderr.startswith("error: --limit must be from 1 to 100")
    assert not client.calls


def test_node_models_are_named_before_gf_server_runs(monkeypatch, tmp_path):
    from generalforecast.serve import listener

    monkeypatch.setattr(listener, "binary", lambda name: tmp_path / "absent")
    code, _, stderr = gf(
        "admin",
        "node",
        "init",
        "--path",
        str(tmp_path / "n.toml"),
        "--model",
        "bad",
        monkeypatch=monkeypatch,
    )
    assert code == 1
    assert stderr.startswith("error: --model takes NAME=MODEL, not `bad`\n")
