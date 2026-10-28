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


def test_flags_are_never_abbreviated(monkeypatch):
    _returns(monkeypatch, {"models": []})
    code, stdout, stderr = gf("model", "list", "--js", monkeypatch=monkeypatch)
    assert code == 2 and not stdout
    assert stderr == (
        "error[UsageError]: unexpected argument '--js'\nhelp: a similar argument exists: '--json'\n"
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
    assert code == cli.INTERRUPTED_EXIT == 130 and not stdout
    assert json.loads(stderr) == {
        "error": {
            "code": "ValidationError",
            "reason": "interrupted",
            "message": "interrupted",
        }
    }
    code, _, stderr = gf("model", "list", monkeypatch=monkeypatch)
    assert (
        code == cli.INTERRUPTED_EXIT and stderr == "\nerror[Interrupted]: interrupted\n"
    )


def test_ctrl_c_at_a_question_is_an_interruption_not_a_declined_one(monkeypatch):
    class Interrupting(io.StringIO):
        def readline(self, *_args):
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdin", Interrupting())
    code, stdout, stderr = gf(
        "--json", "admin", "delete", "churn", monkeypatch=monkeypatch
    )
    assert code == cli.INTERRUPTED_EXIT and not stdout
    assert json.loads(stderr.splitlines()[-1])["error"]["reason"] == "interrupted"


def test_the_exit_codes_are_documented_where_failures_are_presented():
    doc = cli.__doc__ or ""
    for exit_code, reason in (
        (cli.USAGE_EXIT, "usage_error"),
        (cli.ABORTED_EXIT, "aborted"),
        (cli.INTERRUPTED_EXIT, "interrupted"),
    ):
        assert f"\n    {exit_code} " in doc and f"reason `{reason}`" in doc
    assert f"\n    {cli.BROKEN_PIPE_EXIT}  " in doc
    # The reasons the docstring names are the ones the errors carry.
    assert cli.error_object(term.Aborted("cancelled"))["error"]["reason"] == "aborted"


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
    assert stderr.startswith(f"error[{body['code']}]: {envelope['message']}\n")
    assert (f"help: {hint}\n" in stderr) == (hint is not None)


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
    assert code == cli.ABORTED_EXIT == 2 and not stdout
    error = json.loads(stderr.splitlines()[-1])["error"]
    assert error == {
        "code": "ValidationError",
        "reason": "aborted",
        "message": "`Delete churn and its environments?` needs an answer",
        "hint": "pass --yes to go ahead without asking",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
    code, _, stderr = gf("admin", "delete", "churn", monkeypatch=monkeypatch)
    assert code == cli.ABORTED_EXIT
    assert stderr == (
        "Delete churn and its environments? [y/N] "
        "error[Aborted]: cancelled; nothing was changed\n"
    )


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
    assert code == cli.ABORTED_EXIT and "needs an answer" in stderr
    assert not client.calls
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
    assert code == 1 and stderr.startswith(
        "error[ValidationError]: the file's `name` is not `db`"
    )
    code, _, stderr = gf("admin", "key", "create", monkeypatch=monkeypatch)
    assert code == 2 and "--file <FILE>" in stderr
    code, _, _ = gf("admin", "audit", "list", "--limit", "5", monkeypatch=monkeypatch)
    assert code == 0 and client.calls.pop() == ("audit.list", {"after": 0, "limit": 5})


class _Recorder:
    made: ClassVar[list] = []

    def __init__(self, reference, **options):
        self.made.append((reference, options))

    @classmethod
    def local(cls, path):
        cls.made.append(("local", Path(path).resolve()))
        return object.__new__(cls)

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
    assert code == 1 and stderr.startswith(
        "error[ValidationError]: a catalog model needs a target\n"
    )
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


def test_a_bare_name_is_hosted_and_a_path_is_local_everywhere(
    monkeypatch, tmp_path, recorder
):
    churn = tmp_path / "churn"
    churn.mkdir()
    (churn / "gf.yaml").write_text(f"name: churn\nmodel: {TABICL}\ntarget: y\n")
    monkeypatch.chdir(tmp_path)
    code, _, stderr = gf(
        "predict", "churn", "--queries", "q.csv", monkeypatch=monkeypatch
    )
    assert code == 0 and recorder.made[-1] == (
        "churn",
        {"env": "production", "target": None, "inputs": None},
    )
    assert stderr.startswith(
        "warning: `churn` is the hosted predictor; pass ./churn for the directory here\n"
    )
    code, _, _ = gf(
        "predict",
        "churn",
        "--queries",
        "q.csv",
        "--env",
        "staging",
        "-q",
        monkeypatch=monkeypatch,
    )
    assert code == 0 and recorder.made[-1][1]["env"] == "staging"
    for reference in ("./churn", str(churn)):
        code, _, stderr = gf(
            "predict", reference, "--queries", "q.csv", monkeypatch=monkeypatch
        )
        assert code == 0, stderr
        assert recorder.made[-1] == ("local", Path(reference).resolve())
    code, _, stderr = gf(
        "predict",
        "./churn",
        "--queries",
        "q.csv",
        "--env",
        "staging",
        monkeypatch=monkeypatch,
    )
    assert code == 1 and stderr.startswith(
        "error[ValidationError]: --env does not apply to a local predictor\n"
    )
    from generalforecast.predictor_commands import is_local

    assert [is_local(r) for r in ("churn", ".", "./churn", "~/churn", "a/b")] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert not is_local("http://127.0.0.1:8000/churn")


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
        "error[NotFound]: cannot write `nodir/out.csv`: directory `nodir` does not exist\n"
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
        "error[ValidationError]: horizon must be a whole number of steps\nhelp: pass '--horizon'\n"
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
        "error[UsageError]: unknown noun `keys` for `gf admin`\n"
        "help: a similar noun exists: 'key'\n"
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
    assert code == 1 and stderr.startswith(
        "error[ValidationError]: --limit must be from 1 to 100"
    )
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
    assert stderr.startswith(
        "error[ValidationError]: --model takes NAME=MODEL, not `bad`\n"
    )


def test_eval_reads_a_relative_directory_as_local(monkeypatch, tmp_path, recorder):
    (tmp_path / "cls2").mkdir()
    (tmp_path / "cls2" / "gf.yaml").write_text("name: cls2\n")
    monkeypatch.chdir(tmp_path)
    gf("eval", "./cls2", monkeypatch=monkeypatch)
    assert recorder.made[0] == ("local", (tmp_path / "cls2").resolve())


def test_predict_to_a_file_keeps_the_rows_off_stdout(monkeypatch, tmp_path, recorder):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "q.csv").write_text("id\n1\n")
    code, stdout, _ = gf(
        "--json",
        "predict",
        "elsewhere",
        "--queries",
        "q.csv",
        "--output",
        "o.csv",
        monkeypatch=monkeypatch,
    )
    assert code == 0 and json.loads(stdout) == {"rows": 2, "output": "o.csv"}
    code, stdout, _ = gf(
        "--json", "predict", "elsewhere", "--queries", "q.csv", monkeypatch=monkeypatch
    )
    assert json.loads(stdout)["predictions"][1] == {"id": 2, "label": 1, "p_1": 0.75}


def test_server_text_never_reaches_the_terminal_with_its_escapes(monkeypatch):
    out = term.Output(err=io.StringIO(), environ={"FORCE_COLOR": "1"})
    link = out.link("https://x.test/\x1b]8;;evil\x1b\\", "code\x1b[2J")
    # Only the link's own two OSC 8 sequences remain.
    assert "[2J" not in link and link.count("\x1b") == 4
    _returns(
        monkeypatch,
        {"name": "ctx\x1b[31mred", "lines": ["\x1b]0;title\x07x"]},
    )
    _, stdout, _ = gf("context", "show", "ctx", monkeypatch=monkeypatch)
    assert "\x1b" not in stdout and "ctx[31mred" not in stdout


def test_login_shows_the_servers_words_without_their_escapes(monkeypatch, tmp_path):
    from generalforecast import account, login

    class Flow:
        def __init__(self, server, *, timeout):
            pass

        def start(self, machine):
            return {
                "user_code": "AB\x1b[2JCD",
                "verification_uri": "https://x.test/\x1b]8;;https://evil\x1b\\",
                "expires_in": 600,
            }

        def wait(self, grant):
            raise KeyboardInterrupt

    monkeypatch.setattr(login, "DeviceLogin", Flow)
    monkeypatch.setattr(login, "browser_unavailable", lambda: "GF_NO_BROWSER is set")
    monkeypatch.delenv("GF_API_KEY", raising=False)
    monkeypatch.setattr(account.CredentialStore, "resolve", lambda self, s: None)
    code, _, stderr = gf("login", monkeypatch=monkeypatch)
    assert code == cli.INTERRUPTED_EXIT
    assert "ABCD" in stderr and "\x1b" not in stderr


def test_logout_forgets_the_key_even_when_the_server_cannot_revoke_it(monkeypatch):
    from generalforecast import account, client
    from generalforecast.credentials import Login
    from generalforecast.service.contracts import ServiceError

    removed = []

    class Store:
        def load(self, server):
            return Login(server, "k-1", None, None, None, "p", "file", key="gfk_x1234")

        def remove(self, server):
            removed.append(server)
            return True

        def location(self, saved):
            return "/tmp/credentials"

    class Unreachable:
        def __init__(self, **_):
            self.identity = self

        def manage(self, action, payload=None):
            raise ServiceError("TransportError", "request failed", 503)

    monkeypatch.setattr(account, "CredentialStore", Store)
    monkeypatch.setattr(client, "Client", Unreachable)
    code, stdout, stderr = gf("--json", "logout", monkeypatch=monkeypatch)
    assert code == 0 and removed
    result = json.loads(stdout)
    assert result["revoked"] is False and result["revoke_error"] == "request failed"
    _, _, stderr = gf("logout", monkeypatch=monkeypatch)
    assert "could not revoke key" in stderr and "stays valid there" in stderr


def test_a_quiet_init_prints_no_summary(monkeypatch, tmp_path):
    _write_rows(tmp_path / "rows.csv")
    code, stdout, stderr = gf(
        "-q", "init", "--yes", str(tmp_path), monkeypatch=monkeypatch
    )
    assert code == 0 and stdout == "" and stderr == ""
    (tmp_path / "gf.yaml").unlink()
    code, stdout, stderr = gf(
        "--json", "init", "--yes", str(tmp_path), monkeypatch=monkeypatch
    )
    assert code == 0 and stderr == "" and json.loads(stdout)["name"]


def test_a_port_out_of_range_is_a_usage_error(monkeypatch):
    code, _, stderr = gf("serve", "--port", "99999", monkeypatch=monkeypatch)
    assert code == 2
    assert stderr.startswith(
        "error[UsageError]: invalid value '99999' for '--port': expected a port from 0 to 65535\n"
    )


def test_serving_an_unknown_model_names_the_catalog_as_the_cli_does(monkeypatch):
    code, _, stderr = gf("serve", "nosuch-model", monkeypatch=monkeypatch)
    assert code == 1 and stderr == (
        "error[NotFound]: model `nosuch-model` is not in the catalog\n"
        "help: run `gf model list` to see the catalog\n"
    )


def test_verbose_shows_every_request(monkeypatch):
    monkeypatch.setenv("GF_API_KEY", "gfk_" + "a" * 40)
    monkeypatch.setenv("GF_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("GF_PROFILE", raising=False)
    monkeypatch.delenv("GF_TOKEN", raising=False)
    code, _, stderr = gf("-v", "whoami", "--timeout", "2", monkeypatch=monkeypatch)
    assert code == 1
    shown = [line for line in stderr.splitlines() if "/v1/auth/status" in line]
    assert shown and shown[0].startswith("GET http://127.0.0.1:9/v1/auth/status ")
    assert shown[0].split()[-1].startswith("r-")
    _, _, stderr = gf("whoami", "--timeout", "2", monkeypatch=monkeypatch)
    assert "/v1/auth/status" not in stderr


def test_a_quiet_locked_check_says_which_changes_it_would_make(monkeypatch, tmp_path):
    from generalforecast.predictors import _registry
    from tests.product_cli import render_suite

    render_suite.write_fixtures(tmp_path)
    _registry.use(render_suite.FakeRegistry())
    try:
        code, _, stderr = gf(
            "--json",
            "check",
            "--locked",
            str(tmp_path / "orders-stale"),
            monkeypatch=monkeypatch,
        )
    finally:
        _registry.use(None)
    assert code == 1
    because = json.loads(stderr)["error"]["because"]
    assert because.splitlines() == [
        "+ input      signup_source",
        "~ selection  core 1,000 → 1,500, neighbors 3,000 → 4,500",
    ]


def _sales(path: Path) -> None:
    rows = ["date,store,sales"]
    for store in ("s1", "s2", "s3"):
        for day in range(1, 31):
            rows.append(
                f"2024-01-{day:02d},{store},{10 + (day * 7 + len(store)) % 5}.5"
            )
    path.write_text("\n".join(rows) + "\n")


def test_init_defaults_follow_the_task(monkeypatch, tmp_path):
    from tests.product_cli.test_cli_output import _manifest

    _sales(tmp_path / "sales.csv")
    code, _, stderr = gf(
        "init", "--task", "forecast", "--yes", str(tmp_path), monkeypatch=monkeypatch
    )
    assert code == 0, stderr
    manifest = _manifest(tmp_path)
    assert (manifest["target"], manifest["time"], manifest["series"]) == (
        "sales",
        "date",
        ["store"],
    )
    # Seven days of each of the three stores are held out: the default horizon.
    assert manifest["context"]["until"] == "2024-01-24"
    assert "Until        2024-01-24  21 rows" in stderr
    regression = tmp_path / "regression"
    regression.mkdir()
    (regression / "data.csv").write_text(
        "row,x,y\n"
        + "".join(f"{i},{i * 0.37 % 3:.3f},{i * 1.9 % 7:.3f}\n" for i in range(60))
    )
    code, _, stderr = gf("init", "--yes", str(regression), monkeypatch=monkeypatch)
    assert code == 0, stderr
    assert _manifest(regression)["target"] == "y"


def test_a_json_output_file_is_the_text_json_stdout_carries(tmp_path):
    import datetime
    import json

    import pandas as pd

    from generalforecast.predictor_commands import _write_frame
    from generalforecast.render import dumps

    frame = pd.DataFrame(
        {
            "sku": ["a"],
            "date": [datetime.date(2026, 1, 5)],
            "value": [18.123456789012345],
            "q_0.1": [float("nan")],
        }
    )
    path = tmp_path / "out.json"
    _write_frame(frame, path)
    written = json.loads(path.read_text())
    assert written == json.loads(dumps(frame))
    assert written == [
        {"sku": "a", "date": "2026-01-05", "value": 18.123456789012345, "q_0.1": None}
    ]
