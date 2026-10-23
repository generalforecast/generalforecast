# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Human output, device-code login and error presentation of the gf command."""

import io
import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from generalforecast import cli, command_specs, commands, login, term
from generalforecast.credentials import CredentialStore
from generalforecast.deployments.contracts import ServiceError

SERVER = "https://gf.example.com"
TOKEN = "gf_live_" + "a" * 36 + "a91f"
GRANT = {
    "device_code": "gf_dc_" + "x" * 43,
    "user_code": "WDJB-MJHT",
    "verification_uri": f"{SERVER}/cli",
    "verification_uri_complete": f"{SERVER}/cli?code=WDJB-MJHT",
    "interval": 5,
    "expires_in": 600,
}
APPROVED = {
    "access_token": TOKEN,
    "token_type": "bearer",
    "key": {
        "id": "key_1",
        "name": "gf CLI on box",
        "project": "acme",
        "expires_at": 2e9,
    },
    "email": "alice@acme.com",
    "workspace": {"id": "ws_1", "display_name": "Acme"},
    "project": "acme",
}


class Tty(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def home(tmp_path, monkeypatch):
    for name in (
        "GF_API_KEY",
        "GF_PROFILE",
        "GF_PROJECT",
        "SSH_CONNECTION",
        "SSH_TTY",
        "NO_COLOR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GF_CONFIG_PATH", str(tmp_path / "gf" / "profiles.json"))
    monkeypatch.setenv("GF_KEYCHAIN", "0")
    monkeypatch.setenv("GF_URL", SERVER)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("FORCE_COLOR", "0")
    monkeypatch.setattr(login.time, "sleep", lambda _seconds: None)
    return tmp_path / "gf"


def gf(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["gf", *argv])
    code = 0
    try:
        cli.main()
    except SystemExit as exit:
        code = exit.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def fake_server(monkeypatch, *answers):
    calls, pending = [], list(answers)

    def post(server, path, payload, timeout):
        calls.append((server, path, payload))
        if path == login.START:
            return GRANT
        answer = pending.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(login, "post", post)
    return calls


@pytest.mark.parametrize(
    ("env", "tty", "expected"),
    [
        ({}, True, True),
        ({}, False, False),
        ({"NO_COLOR": "1"}, True, False),
        ({"NO_COLOR": "1", "FORCE_COLOR": "1"}, True, False),
        ({"FORCE_COLOR": "1"}, False, True),
        ({"CLICOLOR_FORCE": "1"}, False, True),
        ({"FORCE_COLOR": "0"}, True, False),
        ({"CLICOLOR": "0"}, True, False),
        ({"TERM": "dumb"}, True, False),
    ],
)
def test_color_follows_tty_and_environment(env, tty, expected):
    stream = Tty() if tty else io.StringIO()
    assert term.color_enabled(stream, env) is expected


def test_status_verbs_are_right_aligned_and_uncolored_when_piped():
    out, err = io.StringIO(), io.StringIO()
    writer = term.Output(out=out, err=err, environ={})
    writer.verb("Logged in", "as alice@acme.com")
    writer.verb("Stored", "key gf_...a91f", "in 2ms")
    writer.error("boom")
    writer.hint("try again")
    assert err.getvalue().splitlines() == [
        "   Logged in as alice@acme.com",
        "      Stored key gf_...a91f in 2ms",
        "error: boom",
        "  hint: try again",
    ]
    colored = io.StringIO()
    term.Output(err=colored, environ={"FORCE_COLOR": "1"}).error("boom")
    assert colored.getvalue() == "\x1b[31;1merror:\x1b[0m \x1b[1mboom\x1b[0m\n"


def test_typo_suggestions():
    assert term.suggest("jobs", ["job", "key", "model"]) == "job"
    assert term.suggest("lst", ["list", "get"]) == "list"
    assert term.suggest("zzzzzz", ["job", "key"]) is None


@pytest.mark.parametrize(
    ("group", "action", "expected"),
    [
        ("key", "create", "Create an API key from --file"),
        ("key", "revoke", "Revoke an API key"),
        ("audit", "get", "Show one audit event"),
        ("organization", "delete", "Delete an organization"),
        ("job", "retry", "Retry a failed job"),
        ("job", "results", "Show a job's results"),
    ],
)
def test_action_help_picks_the_article(group, action, expected):
    assert command_specs.action_help(group, action) == expected


def test_grouped_help_lists_public_commands_only(home, monkeypatch, capsys):
    code, out, err = gf(monkeypatch, capsys, "--help")
    assert code == 0 and not err
    assert "Account:\n  login      Log in through your browser" in out
    for hidden in (
        "--url",
        "--base-url",
        "--key-env",
        "--profile",
        "--project",
        "organization",
    ):
        assert hidden not in out
    code, out, _ = gf(monkeypatch, capsys, "push", "--help")
    assert code == 0 and out.startswith("Usage: gf push [OPTIONS] [<PATH>]")
    assert "--key-env" not in out and "Global options:" in out


@pytest.mark.parametrize(
    ("argv", "message", "hint"),
    [
        (["jobs", "list"], "unknown command `jobs`", "did you mean `job`?"),
        (["job", "lst"], "unknown action `lst` for `gf job`", "did you mean `list`?"),
        (["push", "--josn"], "unexpected argument `--josn`", "did you mean `--json`?"),
        (["init", "pkg"], "missing required argument `--model`", None),
    ],
)
def test_argument_errors_suggest_and_exit_2(
    home, monkeypatch, capsys, argv, message, hint
):
    code, out, err = gf(monkeypatch, capsys, *argv)
    assert code == 2 and not out
    assert err.startswith(f"error: {message}\n")
    if hint:
        assert f"  hint: {hint}\n" in err
    code, out, err = gf(monkeypatch, capsys, "--json", *argv)
    assert code == 2
    assert json.loads(err) == {"error": {"code": "UsageError", "message": message}}


def test_unknown_catalog_model_suggests_the_closest(
    home, monkeypatch, capsys, tmp_path
):
    code, _, err = gf(
        monkeypatch,
        capsys,
        "init",
        str(tmp_path / "p"),
        "--model",
        "tabicl-v2-clasifier",
    )
    assert code == 1
    assert "error: unknown model `tabicl-v2-clasifier`" in err
    assert "hint: did you mean `tabicl-v2-classifier`?" in err
    assert not (tmp_path / "p").exists()


def test_login_opens_browser_polls_and_stores_key(home, monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(
        "webbrowser.open", lambda url, new=0: opened.append(url) or True
    )
    pending = ServiceError("AuthorizationPending", "pending", 400)
    calls = fake_server(
        monkeypatch, pending, ServiceError("SlowDown", "slow", 400), APPROVED
    )
    code, out, err = gf(monkeypatch, capsys, "login")
    assert code == 0 and not out
    assert opened == [GRANT["verification_uri_complete"]]
    assert [path for _, path, _ in calls] == [login.START, *[login.TOKEN] * 3]
    assert err.splitlines()[:2] == [
        f"Opening {SERVER}/cli?code=WDJB-MJHT in your browser.",
        "Confirm the code shown there is WDJB-MJHT.",
    ]
    assert "   Logged in as alice@acme.com to workspace Acme" in err
    path = home / "credentials.toml"
    assert f"      Stored key gf_...a91f in {path} (0600)" in err.replace(
        str(path), str(path)
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert TOKEN in path.read_text()
    saved = CredentialStore().load(SERVER)
    assert saved is not None and saved.email == "alice@acme.com" and saved.key == TOKEN


def test_login_without_browser_prints_url_and_code(home, monkeypatch, capsys):
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    monkeypatch.setattr("webbrowser.open", lambda *_: pytest.fail("opened a browser"))
    fake_server(monkeypatch, APPROVED)
    code, _, err = gf(monkeypatch, capsys, "login")
    assert code == 0
    assert err.startswith(
        "warning: no browser available (SSH session)\n\n"
        "Open this URL on any device:\n\n"
        f"    {SERVER}/cli\n\n"
        "and enter the code WDJB-MJHT.\n"
    )


def test_login_json_is_stable_and_never_contains_the_key(home, monkeypatch, capsys):
    monkeypatch.setattr("webbrowser.open", lambda *_args, **_kw: True)
    fake_server(monkeypatch, APPROVED)
    code, out, _ = gf(monkeypatch, capsys, "login", "--json")
    assert code == 0
    assert json.loads(out) == {
        "authenticated": True,
        "server": SERVER,
        "email": "alice@acme.com",
        "workspace": {"id": "ws_1", "display_name": "Acme"},
        "project": "acme",
        "key": {"id": "key_1", "suffix": "gf_...a91f"},
        "store": "file",
        "location": str(home / "credentials.toml"),
    }
    assert TOKEN not in out


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            ServiceError("AccessDenied", "denied", 400),
            "the login was denied in the browser",
        ),
        (
            ServiceError("ExpiredToken", "expired", 400),
            "the login code expired before it was approved",
        ),
    ],
)
def test_login_denied_or_expired(home, monkeypatch, capsys, failure, message):
    monkeypatch.setattr("webbrowser.open", lambda *_args, **_kw: True)
    fake_server(monkeypatch, failure)
    code, _, err = gf(monkeypatch, capsys, "login")
    assert code == 1
    assert f"error: {message}\n  hint: run `gf login` again\n" in err
    assert CredentialStore().load(SERVER) is None


class FakeIdentity:
    def __init__(self, calls, failure=None):
        self.calls, self.failure = calls, failure

    def manage(self, name, *_args, **_kw):
        self.calls.append(name)
        if self.failure:
            raise self.failure
        return {
            "key": {
                "id": "key_1",
                "name": "gf CLI on box",
                "project": "acme",
                "expires_at": 2e9,
            },
            "email": "alice@acme.com",
            "workspace": {"id": "ws_1", "display_name": "Acme"},
        }


class FakeClient:
    calls: ClassVar[list] = []
    failure: ClassVar[Exception | None] = None

    def __init__(self, *_args, **_kw):
        self.identity = FakeIdentity(FakeClient.calls, FakeClient.failure)
        self.project = "acme"


def logged_in(monkeypatch, capsys, failure=None):
    monkeypatch.setattr("webbrowser.open", lambda *_args, **_kw: True)
    fake_server(monkeypatch, APPROVED)
    cli.run(["login"])
    FakeClient.calls, FakeClient.failure = [], failure
    monkeypatch.setattr(commands, "Client", FakeClient)
    monkeypatch.setattr("generalforecast.client.Client", FakeClient)

    def call(server, path, payload=None, *, timeout, token=None, project=None):
        assert (server, path, payload) == (SERVER, "/v1alpha1/auth/status", None)
        assert token == os.environ.get("GF_API_KEY", TOKEN)
        return {"result": FakeIdentity([], failure).manage("auth.status")}

    monkeypatch.setattr(login, "call", call)
    capsys.readouterr()


def test_whoami_uses_stored_login(home, monkeypatch, capsys):
    logged_in(monkeypatch, capsys)
    capsys.readouterr()
    code, out, _ = gf(monkeypatch, capsys, "whoami")
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "alice@acme.com"
    assert lines[1] == "  workspace Acme (ws_1)"
    assert lines[2] == "  server    gf.example.com"
    assert lines[3].startswith("  key       gf_...a91f (")
    code, out, _ = gf(monkeypatch, capsys, "whoami", "--json")
    result = json.loads(out)
    assert result["source"] != "GF_API_KEY" and result["key"]["suffix"] == "gf_...a91f"
    assert set(result) == {"email", "workspace", "project", "server", "key", "source"}


def test_env_key_overrides_login_and_rejection_names_its_source(
    home, monkeypatch, capsys
):
    logged_in(
        monkeypatch,
        capsys,
        ServiceError("Unauthorized", "invalid or expired credential", 401),
    )
    monkeypatch.setenv("GF_API_KEY", "gf_live_" + "b" * 40)
    monkeypatch.setenv("GF_PROJECT", "acme")
    code, out, err = gf(monkeypatch, capsys, "whoami")
    assert code == 1 and not out
    assert err.splitlines() == [
        "error: gf.example.com rejected the API key from GF_API_KEY",
        "  Caused by: 401 Unauthorized: invalid or expired credential",
        f"  hint: create a new key at {SERVER}/#/api-keys, or unset GF_API_KEY to use your login",
    ]


def test_not_logged_in(home, monkeypatch, capsys):
    code, _, err = gf(monkeypatch, capsys, "job", "list")
    assert code == 1
    assert (
        err == "error: not logged in\n  hint: run `gf login`, or set GF_API_KEY in CI\n"
    )
    code, _, err = gf(monkeypatch, capsys, "--json", "job", "list")
    assert json.loads(err) == {
        "error": {"code": "NotLoggedIn", "message": "not logged in"}
    }


def test_server_messages_are_never_masked(home, monkeypatch, capsys):
    logged_in(
        monkeypatch,
        capsys,
        ServiceError("QuotaExceeded", "workspace Acme used 100 of 100 jobs", 429),
    )
    code, _, err = gf(monkeypatch, capsys, "whoami")
    assert code == 1
    assert err == "error: workspace Acme used 100 of 100 jobs\n"


def test_capacity_errors_point_at_gf_serve(home, monkeypatch, capsys):
    logged_in(
        monkeypatch,
        capsys,
        ServiceError("CapacityExceeded", "no GPU capacity for tabpfn", 503),
    )
    _, _, err = gf(monkeypatch, capsys, "whoami")
    assert "hint: run it on this machine or your own GPUs: `gf serve <package>`" in err


def test_logout_revokes_then_forgets(home, monkeypatch, capsys):
    logged_in(monkeypatch, capsys)
    capsys.readouterr()
    code, _, err = gf(monkeypatch, capsys, "logout")
    assert code == 0
    assert FakeClient.calls == ["auth.logout"]
    assert err.splitlines() == [
        "     Revoked key gf_...a91f on gf.example.com",
        f"     Removed credentials from {home / 'credentials.toml'}",
    ]
    assert CredentialStore().load(SERVER) is None
    code, _, err = gf(monkeypatch, capsys, "logout")
    assert code == 0 and err == "warning: not logged in to gf.example.com\n"


def test_quiet_hides_status_but_not_errors(home, monkeypatch, capsys):
    logged_in(monkeypatch, capsys)
    capsys.readouterr()
    code, _, err = gf(monkeypatch, capsys, "logout", "-q")
    assert code == 0 and err == ""
    code, _, err = gf(monkeypatch, capsys, "-q", "job", "list")
    assert code == 1 and err.startswith("error: not logged in")


HEAVY = (
    "pyarrow",
    "numpy",
    "generalforecast.client",
    "generalforecast.contracts",
    "generalforecast.commands",
    "generalforecast.transport",
    "generalforecast.deployments.contracts",
)
PROBE = """
import sys
sys.argv = ["gf", sys.argv[1]]
from generalforecast.cli import main
main()
print(sorted(m for m in {heavy!r} if m in sys.modules), file=sys.stderr)
"""


@pytest.mark.parametrize("flag", ["--version", "-V", "--help", "-h"])
def test_version_and_help_import_nothing_heavy(flag):
    from importlib.metadata import version

    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(heavy=HEAVY), flag],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stderr == "[]\n"
    if flag in ("--version", "-V"):
        assert result.stdout == f"gf {version('generalforecast')}\n"
    else:
        assert result.stdout.startswith("General Forecast: ")
        assert "Account:\n" in result.stdout


def test_unknown_gf_yaml_field_points_at_the_line(tmp_path, monkeypatch):
    from generalforecast.packages.validation import PackageError

    (tmp_path / "gf.yaml").write_text(
        "name: churn\nmodel: tabicl-v2-classifier\ntraget: y\n"
    )
    monkeypatch.setattr(cli, "ARGS", SimpleNamespace(path=str(tmp_path)))
    problem = cli.describe(
        PackageError("InvalidPackage", "unknown fields are not supported")
    )
    assert problem.message == "unknown field `traget` in gf.yaml"
    assert problem.hints == ["did you mean `target`?"]
    assert problem.pointer is not None
    assert problem.pointer[1:] == (3, 1, "traget: y", "unknown field", 6)
