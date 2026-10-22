# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where `gf login` keeps the API key it receives, per server.

The key goes to the macOS keychain when `security` is available, otherwise to
~/.config/gf/credentials.toml with mode 0600. Non-secret metadata (email,
workspace, project, key id) always lives in that file. GF_API_KEY overrides
everything for CI.
"""

import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_SERVER = "https://app.generalforecast.com"
KEYCHAIN_SERVICE = "generalforecast"
ENV_KEY = "GF_API_KEY"
# Credential values seen this process, redacted from every output.
SEEN: set[str] = set()
# The server and credential source the current command uses, for diagnostics.
CONNECTION: dict[str, str] = {}
PROFILE_FLAGS = ("url", "profile", "key_env", "base_url")


def server_url(environ: dict[str, str] | None = None) -> str:
    """GF_URL for staging or self-hosted servers, else the hosted service."""
    env = os.environ if environ is None else environ
    return (env.get("GF_URL") or DEFAULT_SERVER).rstrip("/")


def uses_profiles(args: object, environ: Mapping[str, str] | None = None) -> bool:
    """Explicit endpoint flags, GF_PROFILE/GF_TOKEN or a saved default profile."""
    env = os.environ if environ is None else environ
    if any(getattr(args, flag, None) is not None for flag in PROFILE_FLAGS):
        return True
    if env.get("GF_PROFILE") or env.get("GF_TOKEN"):
        return True
    path = Path(env.get("GF_CONFIG_PATH") or config_dir(env) / "profiles.json")
    try:
        profiles = json.loads(path.expanduser().read_text())
    except (OSError, ValueError):
        return False
    return isinstance(profiles, dict) and "default" in profiles


def config_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The gf configuration directory, next to profiles.json."""
    env = os.environ if environ is None else environ
    if env.get("GF_CONFIG_PATH"):
        return Path(env["GF_CONFIG_PATH"]).expanduser().parent
    home = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(home) / "gf"


@dataclass
class Login:
    """One server's stored login."""

    server: str
    key_id: str
    email: str | None
    workspace: str | None
    workspace_id: str | None
    project: str
    store: str
    key: str | None = field(default=None, repr=False)

    @property
    def suffix(self) -> str:
        """The last characters of the key, safe to show."""
        return key_suffix(self.key)


@dataclass
class Credential:
    """The key a command will use and where it came from."""

    key: str = field(repr=False)
    source: str
    login: Login | None = None


def key_suffix(key: str | None) -> str:
    """`gf_...a91f`: enough to recognize a key without revealing it."""
    return f"{key[:3]}...{key[-4:]}" if key and len(key) > 12 else "unknown"


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class CredentialStore:
    """Reads and writes logins in the keychain and credentials.toml."""

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        """Uses the real environment unless a mapping is given (tests)."""
        self.env = dict(os.environ if environ is None else environ)
        self.path = config_dir(self.env) / "credentials.toml"

    def keychain(self) -> bool:
        """Whether the macOS keychain is usable."""
        if self.env.get("GF_KEYCHAIN") == "0":
            return False
        return sys.platform == "darwin" and shutil.which("security") is not None

    def _read(self) -> dict[str, dict]:
        try:
            data = tomllib.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        servers = data.get("servers", {})
        return servers if isinstance(servers, dict) else {}

    def _write(self, servers: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lines = [
            "# Written by gf login. Keys here are secrets; keep this file private.",
            "",
        ]
        for server, values in sorted(servers.items()):
            lines.append(f"[servers.{_toml_string(server)}]")
            lines += [
                f"{k} = {_toml_string(str(v))}"
                for k, v in values.items()
                if v is not None
            ]
            lines.append("")
        temporary = self.path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write("\n".join(lines))
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def load(self, server: str) -> Login | None:
        """The stored login for a server, with its key when readable."""
        values = self._read().get(server)
        if not isinstance(values, dict) or "key_id" not in values:
            return None
        store = values.get("store", "file")
        key = values.get("key") if store == "file" else self._keychain_read(server)
        return Login(
            server=server,
            key_id=values["key_id"],
            email=values.get("email"),
            workspace=values.get("workspace"),
            workspace_id=values.get("workspace_id"),
            project=values.get("project", "local"),
            store=store,
            key=key,
        )

    def save(self, login: Login) -> Login:
        """Stores the key in the keychain when possible, else in the file."""
        assert login.key is not None
        login.store = (
            "keychain" if self.keychain() and self._keychain_write(login) else "file"
        )
        servers = self._read()
        values = {k: v for k, v in asdict(login).items() if k not in {"server", "key"}}
        if login.store == "file":
            values["key"] = login.key
        servers[login.server] = values
        self._write(servers)
        return login

    def remove(self, server: str) -> bool:
        """Forgets a server's login; True when something was removed."""
        servers = self._read()
        found = servers.pop(server, None)
        if found is not None:
            self._write(servers)
            if found.get("store") == "keychain":
                self._keychain_delete(server)
        return found is not None

    def location(self, login: Login) -> str:
        """Where the key is, as the CLI reports it."""
        return "macOS keychain" if login.store == "keychain" else _home(self.path)

    def resolve(self, server: str) -> Credential | None:
        """GF_API_KEY first, then the stored login for this server."""
        if self.env.get(ENV_KEY):
            return Credential(self.env[ENV_KEY], ENV_KEY, self.load(server))
        login = self.load(server)
        if login is None or not login.key:
            return None
        return Credential(login.key, self.location(login), login)

    def _security(
        self, *args: str, secret: str | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["security", *args],
            input=secret,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def _keychain_write(self, login: Login) -> bool:
        assert login.key is not None
        if any(c in value for value in (login.server, login.key) for c in '"\\\n '):
            return False
        # `security -i` reads the command from stdin, keeping the key out of argv.
        command = (
            f"add-generic-password -U -s {KEYCHAIN_SERVICE} "
            f'-a "{login.server}" -w "{login.key}"\n'
        )
        return self._security("-i", secret=command).returncode == 0

    def _keychain_read(self, server: str) -> str | None:
        if not self.keychain():
            return None
        result = self._security(
            "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", server, "-w"
        )
        return result.stdout.strip() or None if result.returncode == 0 else None

    def _keychain_delete(self, server: str) -> None:
        if self.keychain():
            self._security(
                "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", server
            )


def _home(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)
