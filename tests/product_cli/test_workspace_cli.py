# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The CLI never offers human-session workspace commands and explains denials."""

import json

import pytest

from generalforecast import cli, commands
from generalforecast.deployments.contracts import ServiceError


@pytest.mark.parametrize(
    "code,message",
    [
        (
            "SessionRequired",
            "this operation needs a signed-in Google session in the console",
        ),
        ("ReauthenticationRequired", "sign in again in the console"),
    ],
)
def test_session_denials_get_safe_console_hints(monkeypatch, capsys, code, message):
    def run():
        raise ServiceError(code, "server detail gf_inv_" + "C" * 43, 403)

    monkeypatch.setattr(cli, "run", run)
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": {"code": code, "message": message}}


def test_invitation_tokens_are_redacted_from_any_output():
    token = "gf_inv_" + "D" * 43
    assert token not in json.dumps(cli.redact({"note": f"accept {token}"}))


def test_workspace_onboarding_and_admin_stay_out_of_the_cli():
    assert commands.OPERATIONS["workspace"] == {"get"}
    for group in ("workspaces", "invitation", "invitations", "domain", "onboarding"):
        assert group not in commands.OPERATIONS
        assert group not in commands.IDENTITY_GROUPS
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["invitation", "accept", "--token", "x"])
    for argv in (["workspace", "create"], ["workspace", "join"]):
        with pytest.raises((SystemExit, ServiceError)):
            cli.run(argv)
