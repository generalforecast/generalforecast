# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`gf login`, `gf logout` and `gf whoami`."""

import argparse
import datetime
import os
from typing import Any
from urllib.parse import urlsplit

from generalforecast.credentials import (
    ENV_KEY,
    CredentialStore,
    Login,
    key_suffix,
    server_url,
)
from generalforecast.errors import Unauthorized, printable
from generalforecast.service.contracts import (
    WIRE_ERROR_CODES,
    ServiceError,
    public_code,
    reason_of,
)
from generalforecast.term import Output


def not_logged_in() -> Unauthorized:
    """The error every command raises without a credential, as the service folds it."""
    return Unauthorized(
        "not logged in", reason=reason_of("NotLoggedIn"), hint="run `gf login`"
    )


def wire_code(error: BaseException) -> tuple[str, str | None]:
    """An error's contract code and reason, folding a service-internal code.

    A service puts one of the OpenAPI codes on the wire and moves the specific
    name into `reason`; an error raised in this process may still carry the
    specific name, so both read the same way here.
    """
    code = str(getattr(error, "code", None) or "Internal")
    reason = getattr(error, "reason", None)
    if code in WIRE_ERROR_CODES:
        return code, reason
    status = getattr(error, "status", None)
    return public_code(code, status if isinstance(status, int) else 400), (
        reason or reason_of(code)
    )


def host(server: str) -> str:
    """`app.generalforecast.com` for display."""
    return urlsplit(server).netloc or server


def login(args: argparse.Namespace, out: Output) -> dict[str, Any]:
    """Device-code login: open the approval page, wait, store the key."""
    if os.environ.get(ENV_KEY):
        # Like `gh auth login` with GH_TOKEN set: the env key wins, so verify it.
        identity = whoami(args)
        out.verb(
            "Verified",
            f"key {identity['key']['suffix']} from {ENV_KEY} as "
            f"{printable(str(identity['email']))}",
        )
        out.hint(f"unset {ENV_KEY} to log in through your browser instead")
        return {"authenticated": True, **identity}
    import webbrowser

    from generalforecast.login import DeviceLogin, browser_unavailable, machine_name

    server = server_url().rstrip("/")
    flow = DeviceLogin(server, timeout=args.timeout)
    grant = flow.start(machine_name())
    # The server's words reach the terminal, so none may carry a terminal escape.
    code, uri = printable(grant["user_code"]), printable(grant["verification_uri"])
    complete = printable(grant.get("verification_uri_complete") or uri)
    reason = browser_unavailable()
    opened = False
    if reason is None:
        try:
            opened = webbrowser.open(complete, new=2)
        except webbrowser.Error:
            opened = False
        reason = None if opened else "none could be started"
    bold = out.style(code, "bold", stream=out.err)
    if opened:
        out.say(f"Opening {out.link(complete)} in your browser.", force=True)
        out.say(f"Confirm the code shown there is {bold}.", force=True)
        out.say(force=True)
        out.say(
            out.style(
                "Waiting for approval... (Ctrl-C to cancel)", "dim", stream=out.err
            )
        )
    else:
        minutes = max(1, round(int(grant["expires_in"]) / 60))
        out.warning(f"no browser available ({reason})")
        for text in (
            "",
            "Open this URL on any device:",
            "",
            f"    {out.link(uri)}",
            "",
        ):
            out.say(text, force=True)
        out.say(f"and enter the code {bold}.", force=True)
        out.say(force=True)
        waiting = (
            f"Waiting for approval... (expires in {minutes} minutes, Ctrl-C to cancel)"
        )
        out.say(out.style(waiting, "dim", stream=out.err))
    token = flow.wait(grant)
    store = CredentialStore()
    workspace = token["workspace"]
    saved = store.save(
        Login(
            server=server,
            key_id=token["key"]["id"],
            email=token.get("email"),
            workspace=workspace["display_name"],
            workspace_id=workspace["id"],
            project=token["project"],
            store="file",
            key=token["access_token"],
        )
    )
    location = store.location(saved)
    out.verb(
        "Logged in",
        f"as {printable(str(token.get('email')))} to workspace "
        f"{printable(workspace['display_name'])}",
    )
    where = f"the {location}" if saved.store == "keychain" else f"{location} (0600)"
    out.verb("Stored", f"key {saved.suffix} in {where}")
    return {
        "authenticated": True,
        "server": server,
        "email": token.get("email"),
        "workspace": workspace,
        "project": token["project"],
        "key": {"id": saved.key_id, "suffix": saved.suffix},
        "store": saved.store,
        "location": location,
    }


def logout(args: argparse.Namespace, out: Output) -> dict[str, Any]:
    """Revokes the stored key on the server, then forgets it locally.

    The key is forgotten whatever the server says: a server that cannot be
    reached, or refuses, leaves the key valid there until it expires, which
    the warning and `revoke_error` say.
    """
    server = server_url().rstrip("/")
    store = CredentialStore()
    saved = store.load(server)
    if saved is None:
        if os.environ.get(ENV_KEY):
            raise ServiceError(
                "EnvironmentCredential",
                f"no login is stored for {host(server)}; the key in {ENV_KEY} "
                "comes from your environment",
                400,
            )
        out.warning(f"not logged in to {host(server)}")
        return {"logged_out": False, "server": server}
    revoked, failure = False, None
    if saved.key:
        from generalforecast.client import Client
        from generalforecast.transport import Profile

        client = Client(
            profile=Profile(server, saved.project, key_env=None),
            token=saved.key,
            timeout=args.timeout,
        )
        try:
            client.identity.manage("auth.logout")
            revoked = True
        except ServiceError as error:
            if wire_code(error)[0] in {"Unauthorized", "NotFound"}:
                out.warning(f"{host(server)} no longer accepts key {saved.suffix}")
            else:
                failure = printable(error.message)
                out.warning(
                    f"could not revoke key {saved.suffix} on {host(server)} "
                    f"({failure}); it stays valid there until it expires"
                )
    location = store.location(saved)
    store.remove(server)
    if revoked:
        out.verb("Revoked", f"key {saved.suffix} on {host(server)}")
    where = f"the {location}" if saved.store == "keychain" else location
    out.verb("Removed", f"credentials from {where}")
    return {
        "logged_out": True,
        "server": server,
        "revoked": revoked,
        **({} if failure is None else {"revoke_error": failure}),
        "key": {"id": saved.key_id, "suffix": saved.suffix},
        "store": saved.store,
    }


def whoami(args: argparse.Namespace) -> dict[str, Any]:
    """Who the current credential acts for, confirmed by the server."""
    from generalforecast.credentials import (
        CONNECTION,
        SEEN,
        CredentialStore,
        server_url,
        uses_profiles,
    )
    from generalforecast.login import call

    if uses_profiles(args):
        return _profile_whoami(args)
    server = server_url()
    found = CredentialStore().resolve(server)
    if found is None:
        raise not_logged_in()
    SEEN.add(found.key)
    CONNECTION.clear()
    CONNECTION.update(server=server, source=found.source)
    project = os.environ.get("GF_PROJECT") or (
        found.login.project if found.login else None
    )
    # Global routes only need some project header; the key decides the scope.
    status = call(
        server,
        "/v1/auth/status",
        timeout=args.timeout,
        token=found.key,
        project=project or "local",
    )
    status = status.get("result", status)
    key = status.get("key") or {}
    return {
        "email": status.get("email"),
        "workspace": status.get("workspace"),
        "project": project or key.get("project"),
        "server": server,
        "key": {
            "id": key.get("id"),
            "name": key.get("name"),
            "suffix": key_suffix(found.key),
            "expires_at": key.get("expires_at"),
        },
        "source": found.source,
    }


def _profile_whoami(args: argparse.Namespace) -> dict[str, Any]:
    """Whoami for `GF_PROFILE`, `GF_TOKEN` or a saved default profile, through the SDK."""
    from generalforecast.commands import CONNECTION, client_for, connection_for

    settings = connection_for(args)
    client = client_for(args, settings)
    status = client.identity.manage("auth.status")
    key = status.get("key") or {}
    return {
        "email": status.get("email"),
        "workspace": status.get("workspace"),
        "project": client.project,
        "server": CONNECTION.get("server", settings.profile.url),
        "key": {
            "id": key.get("id"),
            "name": key.get("name"),
            "suffix": key_suffix(settings.token),
            "expires_at": key.get("expires_at"),
        },
        "source": CONNECTION.get("source"),
    }


def date(value: Any) -> str | None:
    """A Unix time as YYYY-MM-DD."""
    if not isinstance(value, (int, float)):
        return None
    return datetime.datetime.fromtimestamp(value, datetime.UTC).date().isoformat()
