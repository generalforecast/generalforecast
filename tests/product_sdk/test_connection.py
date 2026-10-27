# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One connection resolution for the SDK and the CLI; keys stay with their origin."""

import argparse
import asyncio
import inspect
from typing import Any

import pytest

import generalforecast as gf
from generalforecast.async_client import AsyncClient
from generalforecast.client import Client
from generalforecast.credentials import CredentialStore, Login
from generalforecast.service.boundary import Request, Resource, Scope, success
from generalforecast.transport import DEFAULT_SERVER, ConnectionSettings, Profile

LOCAL = "http://127.0.0.1:8000"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A clean configuration: no profiles, no keys, no ambient GF_* settings."""
    for variable in (
        "GF_API_KEY",
        "GF_TOKEN",
        "GF_PROFILE",
        "GF_URL",
        "GF_PROJECT",
        "GF_ENV",
        "GF_CONFIG_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("GF_KEYCHAIN", "0")
    return tmp_path


def logged_in(server: str, project: str = "acme", key: str = "gf_loginkey123") -> None:
    CredentialStore().save(
        Login(
            server, "key-1", "ada@example.com", "Acme", "w-1", project, "file", key=key
        )
    )


def test_nothing_configured_is_not_logged_in(home):
    with pytest.raises(gf.Unauthorized) as error:
        Client()
    assert error.value.code == "NotLoggedIn" and "gf login" in (error.value.hint or "")


def test_after_gf_login_the_sdk_uses_the_hosted_service_and_the_login(home):
    logged_in(DEFAULT_SERVER)
    client = Client()
    assert (client.profile.url, client.project) == (DEFAULT_SERVER, "acme")
    assert client._transport._headers["Authorization"] == "Bearer gf_loginkey123"


def test_an_environment_key_goes_to_the_hosted_service_not_localhost(home, monkeypatch):
    monkeypatch.setenv("GF_API_KEY", "gf_envkey12345")
    monkeypatch.setenv("GF_PROJECT", "acme")
    settings = ConnectionSettings.resolve()
    assert settings.profile.url == DEFAULT_SERVER
    assert (settings.token, settings.source) == ("gf_envkey12345", "GF_API_KEY")
    # A local endpoint named by url never receives the ambient hosted key.
    local = Client(url=LOCAL)
    assert "Authorization" not in local._transport._headers
    # Pointing GF_URL at that endpoint makes the key its own.
    monkeypatch.setenv("GF_URL", LOCAL)
    assert Client(url=LOCAL)._transport._headers["Authorization"] == (
        "Bearer gf_envkey12345"
    )


def test_a_stored_login_is_used_only_for_its_server(home, monkeypatch):
    logged_in(DEFAULT_SERVER)
    monkeypatch.setenv("GF_URL", "https://staging.example")
    with pytest.raises(gf.Unauthorized):
        Client()
    logged_in("https://staging.example", project="beta", key="gf_stagingkey1")
    client = Client()
    assert (client.profile.url, client.project) == ("https://staging.example", "beta")
    assert client._transport._headers["Authorization"] == "Bearer gf_stagingkey1"


def test_a_saved_profile_still_wins_and_keeps_its_key_reference(home, monkeypatch):
    Profile(LOCAL, "lab", key_env="LAB_KEY").save("default")
    monkeypatch.setenv("LAB_KEY", "lab-key-12345")
    logged_in(DEFAULT_SERVER)
    client = Client()
    assert (client.profile.url, client.project) == (LOCAL, "lab")
    assert client._transport._headers["Authorization"] == "Bearer lab-key-12345"


def test_the_cli_resolves_exactly_as_the_sdk(home, monkeypatch):
    from generalforecast.commands import CONNECTION, connection_for

    logged_in(DEFAULT_SERVER)
    cli = connection_for(argparse.Namespace())
    assert cli == ConnectionSettings.resolve()
    assert (
        CONNECTION["server"] == DEFAULT_SERVER and CONNECTION["project_pending"] == ""
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_an_unscoped_key_asks_the_service_for_its_project_once(
    home, monkeypatch, asynchronous
):
    monkeypatch.setenv("GF_API_KEY", "gf_envkey12345")
    sent = []

    def reply(method, path, body, headers, deadline=None):
        sent.append((path, headers["X-GF-Project"]))
        request = Request(
            Scope(headers["X-GF-Project"], headers["X-GF-Request-ID"]),
            Resource("workspace", "auth")
            if path == "/v1/auth/status"
            else Resource("predictor", "predictors"),
            {},
        )
        if path == "/v1/auth/status":
            return dict(success(request, {"key": {"project": "acme"}}))
        return dict(success(request, {"predictors": []}))

    async def areply(*args, **kwargs):
        return reply(*args, **kwargs)

    client = (AsyncClient if asynchronous else Client)()
    assert client.project is None
    transport: Any = client._transport
    transport.send = areply if asynchronous else reply

    async def twice():
        for _ in range(2):
            listed = client.predictors.list()
            assert (await listed if inspect.isawaitable(listed) else listed) == []

    asyncio.run(twice())
    assert client.project == "acme"
    assert sent == [
        ("/v1/auth/status", "local"),
        ("/v1/projects/acme/predictors", "acme"),
        ("/v1/projects/acme/predictors", "acme"),
    ]
