# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HTTPS transport with explicit, independent credential layers."""

import asyncio
import ipaddress
import json
import os
import re
import ssl
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

import httpx

from generalforecast._files import private_lock, read_private, write_private
from generalforecast.deployments.boundary import MAX_BYTES, json_object
from generalforecast.deployments.contracts import (
    WIRE_ERROR_CODES,
    ServiceError,
    name,
    number,
    require,
)

DEFAULT_KEY_ENV = "GF_API_KEY"
PROFILE_FIELDS = ("url", "project", "environment", "runtime", "key_env")
_PROFILE_LIMIT = 65536
_KEY_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_ESCAPES = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])|\x9b[0-?]*[ -/]*[@-~]"
)


def printable(message: str) -> str:
    """Removes terminal escapes and non-printable characters from a message."""
    return "".join(
        " " if character in "\t\n\r" else character
        for character in _ESCAPES.sub("", message)
        if character in "\t\n\r" or character.isprintable()
    ).strip()


class RelayedError(ServiceError):
    """A service's own short error message, made printable."""


def _invalid_profile() -> ServiceError:
    return ServiceError("ValidationError", "profile unavailable or invalid", 422)


@dataclass(frozen=True)
class Profile:
    """Non-secret defaults; credentials are referenced by environment variable."""

    url: str = "http://127.0.0.1:8000"
    project: str = "local"
    environment: str = "production"
    runtime: str | None = None
    key_env: str | None = DEFAULT_KEY_ENV

    def __post_init__(self) -> None:
        """Validates defaults without network access."""
        endpoint(self.url)
        name(self.project)
        name(self.environment)
        if self.runtime is not None:
            name(self.runtime)
        require(
            self.key_env is None
            or (
                type(self.key_env) is str
                and _KEY_ENV.fullmatch(self.key_env) is not None
            ),
            "invalid key environment reference",
        )

    @staticmethod
    def config_path(
        path: str | Path | None = None, *, environ: Mapping[str, str] | None = None
    ) -> Path:
        """Selects explicit path, GF_CONFIG_PATH, then the XDG profiles file."""
        env = os.environ if environ is None else environ
        if path is not None:
            return Path(path)
        if "GF_CONFIG_PATH" in env:
            return Path(env["GF_CONFIG_PATH"])
        home = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(home) / "gf" / "profiles.json"

    @classmethod
    def load_all(
        cls,
        path: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        missing_ok: bool | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Returns validated stored profile values, without environment overrides.

        A missing implicit file is empty; an explicit path must exist unless
        missing_ok is set. Every entry is checked against one schema.
        """
        env = os.environ if environ is None else environ
        config = cls.config_path(path, environ=env)
        if missing_ok is None:
            missing_ok = path is None and "GF_CONFIG_PATH" not in env
        raw = read_private(config, limit=_PROFILE_LIMIT, missing_ok=missing_ok)
        if raw is None:
            return {}
        try:
            profiles = (
                tomllib.loads(raw.decode())["profiles"]
                if config.suffix == ".toml"
                else json.loads(raw)
            )
            require(type(profiles) is dict, "invalid profile config")
            for profile, value in profiles.items():
                name(profile)
                require(
                    type(value) is dict and not set(value) - set(PROFILE_FIELDS),
                    "invalid profile fields; use environment credential references",
                )
                cls(**value)
        except (ValueError, KeyError, TypeError, ServiceError):
            raise _invalid_profile() from None
        return profiles

    @classmethod
    def load(
        cls,
        profile: str | None = None,
        path: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        missing_ok: bool = False,
    ) -> "Profile":
        """Reads canonical CLI profiles; files contain refs, never credential values.

        GF_URL, GF_PROJECT, GF_ENV and GF_RUNTIME override stored values.
        """
        env = os.environ if environ is None else environ
        selected = env.get("GF_PROFILE", "default") if profile is None else profile
        name(selected)
        profiles = cls.load_all(path, environ=env)
        if selected not in profiles and not (missing_ok or selected == "default"):
            raise _invalid_profile()
        value = dict(profiles.get(selected, {}))
        for key, variable in (
            ("url", "GF_URL"),
            ("project", "GF_PROJECT"),
            ("environment", "GF_ENV"),
            ("runtime", "GF_RUNTIME"),
        ):
            if variable in env:
                value[key] = env[variable]
        return cls(**value)

    def stored(self) -> dict[str, Any]:
        """Returns the persisted reference-only form; defaults stay implicit."""
        value: dict[str, Any] = {
            "url": self.url,
            "project": self.project,
            "key_env": self.key_env,
        }
        if self.environment != "production":
            value["environment"] = self.environment
        if self.runtime is not None:
            value["runtime"] = self.runtime
        return value

    def save(
        self,
        profile: str,
        path: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> Path:
        """Atomically replaces one profile and preserves every other stored value."""
        name(profile)
        config = self.config_path(path, environ=environ)
        if config.suffix == ".toml":
            raise ServiceError(
                "UnsupportedOperation",
                "TOML profiles are read-only; save to a JSON profile file",
                501,
            )
        with private_lock(config):
            profiles = self.load_all(config, missing_ok=True)
            profiles[profile] = self.stored()
            write_private(config, json.dumps(profiles, indent=2).encode() + b"\n")
        return config


def credential_from_env(
    key_env: str = DEFAULT_KEY_ENV, environ: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """Returns (value, variable). Only the default reference may use GF_TOKEN.

    A present but empty variable is selected, so it fails credential validation
    instead of falling back to an ambient key.
    """
    env = os.environ if environ is None else environ
    if key_env in env:
        return env[key_env], key_env
    if key_env == DEFAULT_KEY_ENV and "GF_TOKEN" in env:
        return env["GF_TOKEN"], "GF_TOKEN"
    return "", key_env


@dataclass(frozen=True)
class ConnectionSettings:
    """One resolved endpoint, project, profile and optional GF credential.

    A profile whose key_env is None, or an explicit base_url, is keyless: no
    Authorization header is sent unless a key is passed for that endpoint.
    """

    profile_name: str | None
    profile: Profile
    credential_env: str | None
    token: str | None = field(repr=False)

    @classmethod
    def resolve(
        cls,
        url: str | None = None,
        project: str | None = None,
        token: str | None = None,
        api_key: str | None = None,
        profile: str | Profile | None = None,
        config_path: str | Path | None = None,
        *,
        key_env: str | None = None,
        new_profile: bool = False,
        environ: Mapping[str, str] | None = None,
        base_url: str | None = None,
    ) -> "ConnectionSettings":
        """Explicit arguments, then GF_* overrides, then the profile, then defaults.

        base_url names one endpoint, such as a local `gf serve`, and ignores
        profiles and ambient GF_* settings so hosted keys never reach it. Only
        api_key/token, or the variable named by key_env, authenticates it.
        """
        env = os.environ if environ is None else environ
        require(token is None or api_key is None, "use either token or api_key")
        explicit = api_key if api_key is not None else token
        if base_url is not None:
            require(
                url is None and profile is None and not new_profile,
                "base_url cannot be combined with url or profile",
            )
            variable = None
            if explicit is None and key_env is not None:
                explicit, variable = env.get(key_env, ""), key_env
            local = Profile(
                base_url, "local" if project is None else project, key_env=key_env
            )
            if explicit is None:
                return cls(None, local, None, None)
            return cls(None, local, variable, credential(explicit, "API key"))
        if isinstance(profile, Profile):
            selected, base = None, profile
        else:
            selected = env.get("GF_PROFILE", "default") if profile is None else profile
            base = Profile.load(
                selected, config_path, environ=env, missing_ok=new_profile
            )
        resolved = Profile(
            base.url if url is None else url,
            base.project if project is None else project,
            base.environment,
            base.runtime,
            base.key_env if key_env is None else key_env,
        )
        if explicit is not None:
            value, variable = explicit, None
        elif resolved.key_env is None:
            return cls(selected, resolved, None, None)
        else:
            value, variable = credential_from_env(resolved.key_env, env)
        return cls(selected, resolved, variable, credential(value, "GF API key"))


def connection(
    url, project, token, api_key, profile, config_path, base_url=None
) -> tuple[Profile, str | None]:
    """Resolves explicit arguments above environment and non-secret profile."""
    settings = ConnectionSettings.resolve(
        url, project, token, api_key, profile, config_path, base_url=base_url
    )
    return settings.profile, settings.token


PLATFORM_PROBE = "/v1/auth/status"
_PORTABLE_ROUTES = (
    ("GET", re.compile(r"/v1/projects/[^/]+/models(/[^/]+)?")),
    ("POST", re.compile(r"/v1/projects/[^/]+/models/[^/]+/infer")),
    ("POST", re.compile(r"/v1/projects/[^/]+/predictions")),
)


class HostedOnlyError(ServiceError):
    """A hosted platform operation sent to a standalone model server."""

    def __init__(self, method: str, path: str) -> None:
        """Names the route without query parameters or request content."""
        super().__init__(
            "UnsupportedOperation",
            f"{method} {path.split('?', 1)[0]} requires the hosted platform; "
            "this endpoint is a standalone model server that only serves "
            "model listing and inference",
            501,
        )
        self.method, self.path = method, path.split("?", 1)[0]


def hosted_only(method: str, path: str) -> bool:
    """True for versioned routes a standalone `gf serve` model server lacks."""
    route = path.split("?", 1)[0]
    return route.startswith("/v1/") and not any(
        method == verb and pattern.fullmatch(route)
        for verb, pattern in _PORTABLE_ROUTES
    )


def route_missing(error: ServiceError) -> bool:
    """A 404 for the route itself, with or without the standard error envelope."""
    return error.status == 404 and error.code in ("NotFound", "ProtocolError")


def platform_from_probe(error: ServiceError | None) -> bool:
    """Any probe answer other than a missing route identifies the hosted platform."""
    return error is None or not route_missing(error)


def retryable(method: str, body: Any) -> bool:
    """Only retries reads and writes with server-enforced stable identities."""
    payload = body.get("payload", {}) if isinstance(body, dict) else {}
    return method == "GET" or bool(
        payload.get("write_id") or payload.get("submission_id")
    )


_ERROR_KEYS = frozenset(
    {"code", "message", "param", "hint", "limit", "actual", "request_id"}
)


def response_value(status: int, raw: bytes) -> dict[str, Any]:
    """Decodes bounded JSON, refusing credential forwarding and unsafe errors."""
    if 300 <= status < 400:
        raise ServiceError("RedirectRefused", "credential forwarding is disabled", 502)
    try:
        require(len(raw) <= MAX_BYTES, "response too large")
        value = json_object(json.loads(raw))
    except (ValueError, UnicodeError, ServiceError):
        raise ServiceError("ProtocolError", "invalid service response", 502) from None
    if not 200 <= status < 300:
        detail = value.get("error")
        if (
            type(detail) is not dict
            or not {"code", "message"} <= set(detail)
            or not set(detail) <= _ERROR_KEYS
            or any(type(detail[key]) is not str for key in ("code", "message"))
            or any(
                type(detail.get(key, "")) is not str
                for key in ("param", "hint", "request_id")
            )
            or any(
                type(detail.get(key, 0)) not in (int, float)
                for key in ("limit", "actual")
            )
        ):
            raise ServiceError("ProtocolError", "invalid service error", status)
        message = printable(detail["message"]) if len(detail["message"]) <= 512 else ""
        fields = {
            key: (printable(value) if type(value) is str else value)
            for key, value in detail.items()
            if key not in ("code", "message") and value is not None
        }
        if message:
            error: ServiceError = RelayedError(
                _wire_code(detail["code"]), message, status
            )
        else:
            error = ServiceError(
                _wire_code(detail["code"]), "service rejected request", status
            )
        for key, value in fields.items():
            setattr(error, key, value)
        raise error
    return value


def _wire_code(code: str) -> str:
    """Classifies an untrusted remote code without echoing unknown values."""
    return code if code in WIRE_ERROR_CODES else "RemoteError"


def request_data(body: Any) -> bytes | None:
    """Serializes bounded JSON or the separately bounded Arrow upload."""
    if isinstance(body, bytes):
        require(len(body) <= 134217728, "snapshot too large")
        return body
    return (
        None
        if body is None
        else json.dumps(json_object(body), allow_nan=False, ensure_ascii=False).encode()
    )


def endpoint(url: str) -> str:
    """Requires TLS except on loopback and forbids credential-bearing URLs."""
    require(type(url) is str, "invalid endpoint")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ServiceError("ValidationError", "invalid endpoint", 422) from None
    try:
        local = (
            parsed.hostname == "localhost"
            or ipaddress.ip_address(parsed.hostname or "").is_loopback
        )
    except ValueError:
        local = False
    require(
        bool(parsed.hostname)
        and (parsed.scheme == "https" or (parsed.scheme == "http" and local))
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and parsed.path in ("", "/")
        and (port is None or 1 <= port <= 65535)
        and not any(c.isspace() or ord(c) < 32 for c in url),
        "endpoint requires HTTPS (or loopback HTTP), no credentials/path/query",
    )
    return url.rstrip("/")


def credential(value: str, label: str) -> str:
    """Validates headers without reflecting credentials in errors."""
    require(
        type(value) is str
        and 0 < len(value) <= 8192
        and all(33 <= ord(c) <= 126 for c in value),
        f"provide a valid {label}",
    )
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ServiceError("RedirectRefused", "credential forwarding is disabled", 502)


class Transport:
    """No cookies, automatic retries, redirects, or credential-bearing repr."""

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        auth_mode: str = "bearer",
        platform_token: str | None = None,
        modal_key: str | None = None,
        modal_secret: str | None = None,
        tls_context: ssl.SSLContext | None = None,
        timeout: float = 40,
        retries: int = 0,
        retry_delay: float = 0.1,
    ) -> None:
        """Binds GF keys separately from Azure Entra or Modal proxy credentials.

        For mTLS, pass a verified SSLContext with load_cert_chain configured.
        Secrets never belong in URLs, command arguments or provider headers
        substituted for canonical GF authorization.
        """
        self.url = endpoint(url)
        number(timeout, 0.01, 300, "timeout")
        self.timeout = timeout
        require(type(retries) is int and 0 <= retries <= 5, "invalid retries")
        number(retry_delay, 0, 5, "retry_delay")
        self.retries, self.retry_delay, self.closed = retries, retry_delay, False
        require(
            auth_mode in ("bearer", "azure-easyauth-and-api-key"),
            "unsupported auth_mode",
        )
        self.platform: bool | None = None
        if token is not None:
            token = credential(token, "GF token")
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        if auth_mode == "azure-easyauth-and-api-key":
            if token is None:
                raise ServiceError(
                    "ValidationError", "Azure auth_mode requires a GF key", 422
                )
            self._headers["Authorization"] = "Bearer " + credential(
                platform_token or "", "Azure platform token"
            )
            self._headers["X-GF-API-Key"] = token
        else:
            require(platform_token is None, "platform_token requires Azure auth_mode")
            if token is not None:
                self._headers["Authorization"] = "Bearer " + token
        require(
            (modal_key is None) == (modal_secret is None),
            "Modal requires both proxy credentials",
        )
        if modal_key is not None:
            require(
                auth_mode == "bearer", "provider credential modes cannot be combined"
            )
            self._headers["Modal-Key"] = credential(modal_key, "Modal key")
            self._headers["Modal-Secret"] = credential(
                modal_secret or "", "Modal secret"
            )
        if tls_context is not None:
            require(
                tls_context.check_hostname
                and tls_context.verify_mode == ssl.CERT_REQUIRED,
                "TLS context must verify certificates and hostnames",
            )
        self._opener = build_opener(
            ProxyHandler({}), _NoRedirect(), HTTPSHandler(context=tls_context)
        )

    def send(
        self,
        method: str,
        path: str,
        body: Any,
        headers: dict[str, str],
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Raises HostedOnlyError for platform routes on a standalone model server.

        The first missing hosted-only route triggers one probe of PLATFORM_PROBE;
        the answer is remembered for this transport.
        """
        platform_route = hosted_only(method, path)
        if platform_route and self.platform is False:
            raise HostedOnlyError(method, path)
        try:
            value = self._exchange(method, path, body, headers, deadline=deadline)
        except ServiceError as error:
            if platform_route and self.platform is None and route_missing(error):
                try:
                    self._exchange("GET", PLATFORM_PROBE, None, {}, deadline=deadline)
                    self.platform = True
                except ServiceError as probe:
                    self.platform = platform_from_probe(probe)
                if not self.platform:
                    raise HostedOnlyError(method, path) from None
            raise
        if platform_route:
            self.platform = True
        return value

    def _exchange(
        self,
        method: str,
        path: str,
        body: Any,
        headers: dict[str, str],
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Uses a single deadline; never retries an unguarded mutation."""
        require(not self.closed, "client is closed")
        end = (
            min(deadline, time.monotonic() + self.timeout)
            if deadline is not None
            else time.monotonic() + self.timeout
        )
        for attempt in range(self.retries + 1):
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise ServiceError("Timeout", "request deadline exceeded", 504)
            try:
                return self._send_once(method, path, body, headers, remaining, end)
            except ServiceError as error:
                if (
                    not retryable(method, body)
                    or attempt == self.retries
                    or (error.code != "TransportError" and error.status != 503)
                ):
                    raise
                time.sleep(
                    min(self.retry_delay * 2**attempt, max(0, end - time.monotonic()))
                )
        raise AssertionError("unreachable")

    def close(self) -> None:
        """Prevents further requests; urllib closes each response after reading."""
        self.closed = True

    def _send_once(self, method, path, body, headers, timeout, deadline):
        require(
            path.startswith("/")
            and not path.startswith("//")
            and "\\" not in path
            and "#" not in path,
            "invalid request path",
        )
        data = request_data(body)
        request = Request(
            self.url + path,
            data=data,
            method=method,
            headers=self._headers | headers,
        )
        try:
            response = self._opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        except TimeoutError:
            raise ServiceError("Timeout", "request deadline exceeded", 504) from None
        except (URLError, OSError, HTTPException) as error:
            if isinstance(getattr(error, "reason", None), TimeoutError):
                raise ServiceError(
                    "Timeout", "request deadline exceeded", 504
                ) from None
            raise ServiceError(
                "TransportError",
                "request failed; outcome unknown; preserve write_id or submission_id",
                503,
            ) from None
        with response:
            status = response.code
            if 300 <= status < 400:
                raise ServiceError(
                    "RedirectRefused", "credential forwarding is disabled", 502
                )
            try:
                raw = bytearray()
                while len(raw) <= MAX_BYTES:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    reader = response
                    for _ in range(3):
                        raw_socket = getattr(
                            getattr(reader, "raw", None), "_sock", None
                        )
                        if raw_socket is not None:
                            raw_socket.settimeout(remaining)
                            break
                        reader = getattr(reader, "fp", None)
                    read = getattr(response, "read1", response.read)
                    chunk = read(min(65536, MAX_BYTES + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                    if not hasattr(response, "read1"):
                        break
                require(len(raw) <= MAX_BYTES, "response too large")
                json_object(json.loads(raw))
            except TimeoutError:
                raise ServiceError(
                    "Timeout", "request deadline exceeded", 504
                ) from None
            except (OSError, HTTPException):
                raise ServiceError(
                    "TransportError",
                    "response interrupted; outcome unknown; preserve write_id or submission_id",
                    503,
                ) from None
            except (ValueError, UnicodeError, ServiceError):
                raise ServiceError(
                    "ProtocolError", "invalid service response", 502
                ) from None
        if time.monotonic() > deadline:
            raise ServiceError("Timeout", "request deadline exceeded", 504)
        return response_value(status, bytes(raw))


class AsyncTransport:
    """Native async HTTP with bounded streams, cancellation and connection cleanup."""

    def __init__(
        self,
        url: str,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
        **options: Any,
    ) -> None:
        """Shares credential validation, never sync network I/O or event-loop wrappers."""
        settings = Transport(url, **options)
        self.url, self.timeout = settings.url, settings.timeout
        self.retries, self.retry_delay = settings.retries, settings.retry_delay
        self._headers = settings._headers
        self.platform: bool | None = None
        self._http = httpx.AsyncClient(
            verify=options.get("tls_context") or True,
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
            transport=http_transport,
        )

    async def aclose(self) -> None:
        """Closes all pooled sockets."""
        await self._http.aclose()

    async def send(
        self,
        method: str,
        path: str,
        body: Any,
        headers: dict[str, str],
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Matches Transport.send hosted-only classification with one async probe."""
        platform_route = hosted_only(method, path)
        if platform_route and self.platform is False:
            raise HostedOnlyError(method, path)
        try:
            value = await self._exchange(method, path, body, headers, deadline=deadline)
        except ServiceError as error:
            if platform_route and self.platform is None and route_missing(error):
                try:
                    await self._exchange(
                        "GET", PLATFORM_PROBE, None, {}, deadline=deadline
                    )
                    self.platform = True
                except ServiceError as probe:
                    self.platform = platform_from_probe(probe)
                if not self.platform:
                    raise HostedOnlyError(method, path) from None
            raise
        if platform_route:
            self.platform = True
        return value

    async def _exchange(
        self,
        method: str,
        path: str,
        body: Any,
        headers: dict[str, str],
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Enforces an overall timeout including retry delays and streaming reads."""
        require(
            path.startswith("/")
            and not path.startswith("//")
            and "\\" not in path
            and "#" not in path,
            "invalid request path",
        )
        data = request_data(body)
        remaining = (
            self.timeout
            if deadline is None
            else min(self.timeout, deadline - time.monotonic())
        )
        if remaining <= 0:
            raise ServiceError("Timeout", "request deadline exceeded", 504)
        try:
            async with asyncio.timeout(remaining):
                for attempt in range(self.retries + 1):
                    try:
                        # Do not persist console cookies or ambient proxy credentials.
                        self._http.cookies.clear()
                        async with self._http.stream(
                            method,
                            self.url + path,
                            content=data,
                            headers=self._headers | headers,
                        ) as response:
                            if 300 <= response.status_code < 400:
                                raise ServiceError(
                                    "RedirectRefused",
                                    "credential forwarding is disabled",
                                    502,
                                )
                            if (
                                response.headers.get(
                                    "Content-Encoding", "identity"
                                ).lower()
                                != "identity"
                            ):
                                raise ServiceError(
                                    "ProtocolError",
                                    "compressed service responses are disabled",
                                    502,
                                )
                            raw = bytearray()
                            async for chunk in response.aiter_bytes(chunk_size=65536):
                                if len(raw) + len(chunk) > MAX_BYTES:
                                    raise ServiceError(
                                        "ProtocolError", "invalid service response", 502
                                    )
                                raw.extend(chunk)
                            value = response_value(response.status_code, bytes(raw))
                        return value
                    except httpx.TimeoutException:
                        raise ServiceError(
                            "Timeout", "request deadline exceeded", 504
                        ) from None
                    except httpx.TransportError:
                        error = ServiceError(
                            "TransportError",
                            "request failed; outcome unknown; preserve idempotency identity",
                            503,
                        )
                    except ServiceError as caught:
                        error = caught
                    except RuntimeError:
                        raise ServiceError(
                            "ClientClosed", "client is closed", 409
                        ) from None
                    if (
                        not retryable(method, body)
                        or attempt == self.retries
                        or (error.code != "TransportError" and error.status != 503)
                    ):
                        raise error
                    await asyncio.sleep(self.retry_delay * 2**attempt)
        except TimeoutError:
            raise ServiceError("Timeout", "request deadline exceeded", 504) from None
        raise AssertionError("unreachable")
