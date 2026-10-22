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
from generalforecast.deployments.contracts import ServiceError
from generalforecast.term import Output


def host(server: str) -> str:
    """`app.generalforecast.com` for display."""
    return urlsplit(server).netloc or server


def login(args: argparse.Namespace, out: Output) -> dict[str, Any]:
    """Device-code login: open the approval page, wait, store the key."""
    if getattr(args, "profile", None) is not None or getattr(args, "key_env", None):
        from generalforecast.commands import configured_login

        return configured_login(args)
    if os.environ.get(ENV_KEY):
        # Like `gh auth login` with GH_TOKEN set: the env key wins, so verify it.
        identity = whoami(args)
        out.verb(
            "Verified",
            f"key {identity['key']['suffix']} from {ENV_KEY} as {identity['email']}",
        )
        out.hint(f"unset {ENV_KEY} to log in through your browser instead")
        return {"authenticated": True, **identity}
    import webbrowser

    from generalforecast.login import DeviceLogin, browser_unavailable, machine_name

    server = (getattr(args, "url", None) or server_url()).rstrip("/")
    flow = DeviceLogin(server, timeout=getattr(args, "timeout", 30))
    out.debug(f"POST {server}/v1alpha1/cli-logins")
    grant = flow.start(machine_name())
    code, uri = grant["user_code"], grant["verification_uri"]
    complete = grant.get("verification_uri_complete") or uri
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
        "Logged in", f"as {token.get('email')} to workspace {workspace['display_name']}"
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
    """Revokes the stored key on the server, then forgets it locally."""
    server = (getattr(args, "url", None) or server_url()).rstrip("/")
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
    revoked = False
    if saved.key:
        from generalforecast.client import Client
        from generalforecast.transport import Profile

        client = Client(
            profile=Profile(server, saved.project, key_env=None),
            token=saved.key,
            timeout=getattr(args, "timeout", 40),
        )
        try:
            client.identity.manage("auth.logout")
            revoked = True
        except ServiceError as error:
            if error.code not in {"Unauthorized", "NotFound"}:
                raise
            out.warning(f"{host(server)} no longer accepts key {saved.suffix}")
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

    if uses_profiles(args) or getattr(args, "local", False):
        return _profile_whoami(args)
    server = server_url()
    found = CredentialStore().resolve(server)
    if found is None:
        raise ServiceError("NotLoggedIn", "not logged in", 401)
    SEEN.add(found.key)
    CONNECTION.clear()
    CONNECTION.update(server=server, source=found.source)
    project = os.environ.get("GF_PROJECT") or (
        found.login.project if found.login else None
    )
    # Global routes only need some project header; the key decides the scope.
    status = call(
        server,
        "/v1alpha1/auth/status",
        timeout=getattr(args, "timeout", None) or 40,
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
    """Whoami for saved profiles and explicit endpoints, through the SDK."""
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
