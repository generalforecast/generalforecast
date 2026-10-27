# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HTTPS transport with explicit, independent credential layers.

`Transport` and `AsyncTransport` are the SDK's HTTP stack: the clients,
`gf login` and remote table reads use them, and `send` and `download` are
public for any other caller. They send no cookies, follow no redirects, read
no proxy settings from the environment and require HTTPS except on loopback,
so a credential reaches only the origin it was given for.
"""

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
from generalforecast.errors import (
    printable,  # noqa: F401  (re-exported)
    wire_fields,
)
from generalforecast.service.boundary import MAX_BYTES, json_object
from generalforecast.service.contracts import (
    ServiceError,
    name,
    number,
    require,
)

DEFAULT_KEY_ENV = "GF_API_KEY"
# The environment a service scores when a request names none.
DEFAULT_ENVIRONMENT = "production"
# The bounds of one request's timeout, in seconds.
MIN_TIMEOUT, MAX_TIMEOUT = 0.01, 300
PROFILE_FIELDS = ("url", "project", "environment", "key_env")
# Fields earlier releases stored, read and ignored so their profiles still load.
_RETIRED_FIELDS = ("runtime",)
_PROFILE_LIMIT = 65536
_KEY_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


class RelayedError(ServiceError):
    """A service's own short error message, made printable."""


def _current(value: dict[str, Any]) -> dict[str, Any]:
    """A stored profile without the fields earlier releases kept."""
    return {key: item for key, item in value.items() if key not in _RETIRED_FIELDS}


def _invalid_profile() -> ServiceError:
    return ServiceError("ValidationError", "profile unavailable or invalid", 422)


@dataclass(frozen=True)
class Profile:
    """Non-secret defaults; credentials are referenced by environment variable."""

    url: str = "http://127.0.0.1:8000"
    project: str = "local"
    environment: str = DEFAULT_ENVIRONMENT
    key_env: str | None = DEFAULT_KEY_ENV

    def __post_init__(self) -> None:
        """Validates defaults without network access."""
        endpoint(self.url)
        name(self.project)
        name(self.environment)
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
                    type(value) is dict
                    and not set(value) - {*PROFILE_FIELDS, *_RETIRED_FIELDS},
                    "invalid profile fields; use environment credential references",
                )
                cls(**_current(value))
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

        GF_URL, GF_PROJECT and GF_ENV override stored values.
        """
        env = os.environ if environ is None else environ
        selected = env.get("GF_PROFILE", "default") if profile is None else profile
        name(selected)
        profiles = cls.load_all(path, environ=env)
        if selected not in profiles and not (missing_ok or selected == "default"):
            raise _invalid_profile()
        value = _current(profiles.get(selected, {}))
        for key, variable in (
            ("url", "GF_URL"),
            ("project", "GF_PROJECT"),
            ("environment", "GF_ENV"),
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
        if self.environment != DEFAULT_ENVIRONMENT:
            value["environment"] = self.environment
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
# What a standalone `gf serve` (gf-server's front) answers: model listing and
# predictions. Every other versioned route belongs to the hosted platform.
_PORTABLE_ROUTES = (
    ("GET", re.compile(r"/v1/projects/[^/]+/models(/[^/]+)?")),
    ("GET", re.compile(r"/v1/projects/[^/]+/predictors(/[^/]+)?")),
    ("POST", re.compile(r"/v1/projects/[^/]+/predictors/[^/]+/predict")),
    ("POST", re.compile(r"/v1/projects/[^/]+/predict")),
)


class HostedOnlyError(ServiceError):
    """A hosted platform operation sent to a standalone model server."""

    def __init__(self, method: str, path: str) -> None:
        """Names the route without query parameters or request content."""
        super().__init__(
            "UnsupportedOperation",
            f"{method} {path.split('?', 1)[0]} requires the hosted platform; "
            "this endpoint is a standalone model server that only serves "
            "models, predictors and predictions",
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


def platform_from_probe(error: ServiceError | None) -> bool | None:
    """What a probe's answer says the server is: hosted, a model server, or unknown.

    A missing route is a model server; any other HTTP answer is the hosted
    platform; a probe that got no answer, or no readable one, says nothing,
    so the next missing route probes again.
    """
    if error is None:
        return True
    if route_missing(error):
        return False
    if error.code in _NO_ANSWER:
        return None
    return True


# Probe failures that carry no answer from the server.
_NO_ANSWER = frozenset(
    {"TransportError", "Timeout", "ProtocolError", "RedirectRefused"}
)


def retryable(method: str, body: Any) -> bool:
    """Only retries reads and writes with server-enforced stable identities."""
    payload = body.get("payload", {}) if isinstance(body, dict) else {}
    return method == "GET" or bool(
        payload.get("write_id") or payload.get("submission_id")
    )


def response_value(status: int, raw: bytes) -> dict[str, Any]:
    """Decodes bounded JSON, refusing credential forwarding and unsafe errors."""
    if 300 <= status < 400:
        raise ServiceError("RedirectRefused", "credential forwarding is disabled", 502)
    try:
        require(len(raw) <= MAX_BYTES, "response too large")
        value = json_object(json.loads(raw))
    except (ValueError, UnicodeError, ServiceError):
        # A refusal keeps its status, so a bare 404 is still a missing route; a
        # success that is not JSON is the service's fault.
        failed = 502 if 200 <= status < 300 else status
        raise ServiceError(
            "ProtocolError", "invalid service response", failed
        ) from None
    if not 200 <= status < 300:
        raise _failure(status, value.get("error"))
    return value


def _failure(status: int, detail: Any) -> ServiceError:
    """The error a non-2xx reply's error object describes."""
    fields = wire_fields(detail, status=status)
    if fields is None:
        return ServiceError("ProtocolError", "invalid service error", status)
    code, message = fields.pop("code"), fields.pop("message")
    fields.pop("status")
    if message:
        return RelayedError(code, message, status, **fields)
    return ServiceError(code, "service rejected request", status, **fields)


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


def _origin(url: str, *, paths: bool) -> str:
    """Checks a URL: HTTPS except on loopback, no credentials or fragment.

    An endpoint (`paths` False) is an origin alone; a resource URL may carry
    a path and a query.
    """
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
        and not parsed.fragment
        and (paths or (not parsed.query and parsed.path in ("", "/")))
        and (port is None or 1 <= port <= 65535)
        and not any(c.isspace() or ord(c) < 32 for c in url),
        "endpoint requires HTTPS (or loopback HTTP), no credentials/path/query"
        if not paths
        else "a URL requires HTTPS (or loopback HTTP) and no credentials",
    )
    return f"{parsed.scheme}://{parsed.netloc}"


def endpoint(url: str) -> str:
    """Requires TLS except on loopback and forbids credential-bearing URLs."""
    _origin(url, paths=False)
    return url.rstrip("/")


def split_url(url: str) -> tuple[str, str]:
    """A resource URL as its checked origin and its path with any query."""
    origin = _origin(url, paths=True)
    rest = url[len(origin) :] or "/"
    return origin, rest


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


def _closed() -> ServiceError:
    return ServiceError("ClientClosed", "client is closed", 400)


def _timeout() -> ServiceError:
    return ServiceError("Timeout", "request deadline exceeded", 504)


def _transport_failure(what: str) -> ServiceError:
    return ServiceError(
        "TransportError",
        f"{what}; outcome unknown; preserve write_id or submission_id",
        503,
    )


def _compressed() -> ServiceError:
    return ServiceError(
        "ProtocolError", "compressed service responses are disabled", 502
    )


class _Oversized(Exception):
    """A response past the limit its caller set; each caller names its error."""


def _too_large() -> ServiceError:
    return ServiceError("ProtocolError", "invalid service response", 502)


def _download_too_large(limit: int) -> ServiceError:
    return ServiceError(
        "CapacityExceeded",
        f"the resource is larger than {limit:,} bytes",
        413,
        limit=limit,
    )


def _checked_path(path: str) -> str:
    require(
        path.startswith("/")
        and not path.startswith("//")
        and "\\" not in path
        and "#" not in path,
        "invalid request path",
    )
    return path


def _download_failure(status: int, raw: bytes) -> ServiceError:
    """The error for a non-2xx download: the service's own error object, if any."""
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        value = None
    detail = value.get("error") if isinstance(value, dict) else None
    if detail is not None:
        return _failure(status, detail)
    return ServiceError(
        "NotFound" if status == 404 else "RemoteError",
        f"the server answered HTTP {status}",
        status,
    )


class _Policy:
    """What both transports share: settings, headers, retries and platform detection.

    The platform is learned once, from one probe of `PLATFORM_PROBE` after the
    first missing hosted-only route. It only explains a later missing route
    as `HostedOnlyError`; it never stops a request from being sent, so a
    server that turns out to serve a route answers it.
    """

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
        number(timeout, MIN_TIMEOUT, MAX_TIMEOUT, "timeout")
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
        self._tls = tls_context

    def _end(self, deadline: float | None) -> float:
        """The moment a request must finish by: its deadline or the timeout."""
        if self.closed:
            raise _closed()
        end = time.monotonic() + self.timeout
        return end if deadline is None else min(deadline, end)

    def _retry(self, method: str, body: Any, attempt: int, error: ServiceError) -> bool:
        return (
            retryable(method, body)
            and attempt < self.retries
            and (error.code == "TransportError" or error.status == 503)
        )

    def _probe_needed(self, method: str, path: str, error: ServiceError) -> bool:
        return (
            self.platform is None and hosted_only(method, path) and route_missing(error)
        )

    def _learn(self, probe: ServiceError | None) -> None:
        self.platform = platform_from_probe(probe)

    def _explain(self, method: str, path: str, error: ServiceError) -> None:
        """Raises a missing hosted-only route on a model server as `HostedOnlyError`."""
        if (
            self.platform is False
            and hosted_only(method, path)
            and route_missing(error)
        ):
            raise HostedOnlyError(method, path) from None

    def _answered(self, method: str, path: str) -> None:
        if hosted_only(method, path):
            self.platform = True


class Transport(_Policy):
    """No cookies, automatic retries, redirects, or credential-bearing repr.

    `send` is the public API: it takes a path under the endpoint, a JSON
    body (or Arrow bytes) and request headers, and returns the reply's JSON
    object or raises its error as a `ServiceError`. `download` returns the
    raw bytes of a resource.
    """

    def __init__(self, url: str, **options: Any) -> None:
        """Validates the endpoint and credentials; see `_Policy`."""
        super().__init__(url, **options)
        self._opener = build_opener(
            ProxyHandler({}), _NoRedirect(), HTTPSHandler(context=self._tls)
        )

    def send(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Sends one request and returns the reply's JSON object.

        Args:
            method: `GET` or `POST`.
            path: The path under the endpoint, with any query.
            body: A JSON object, Arrow stream bytes, or None.
            headers: Headers to add to the transport's own.
            deadline: A `time.monotonic()` moment the request must finish by.

        Raises:
            ServiceError: The reply's error; `HostedOnlyError` for a hosted
                route a standalone model server does not serve.
        """
        try:
            status, raw = self._exchange(
                method, path, body, dict(headers or {}), deadline, MAX_BYTES
            )
            value = response_value(status, raw)
        except _Oversized:
            raise _too_large() from None
        except ServiceError as error:
            if self._probe_needed(method, path, error):
                try:
                    response_value(
                        *self._exchange(
                            "GET", PLATFORM_PROBE, None, {}, deadline, MAX_BYTES
                        )
                    )
                    probe = None
                except ServiceError as answer:
                    probe = answer
                self._learn(probe)
            self._explain(method, path, error)
            raise
        self._answered(method, path)
        return value

    def download(
        self, path: str, *, limit: int, deadline: float | None = None
    ) -> bytes:
        """GETs a resource's bytes; more than `limit` of them is `CapacityExceeded`."""
        try:
            status, raw = self._exchange(
                "GET", path, None, {"Accept": "*/*"}, deadline, limit
            )
        except _Oversized:
            raise _download_too_large(limit) from None
        if not 200 <= status < 300:
            raise _download_failure(status, raw)
        return raw

    def close(self) -> None:
        """Prevents further requests; urllib closes each response after reading."""
        self.closed = True

    def _exchange(
        self,
        method: str,
        path: str,
        body: Any,
        headers: dict[str, str],
        deadline: float | None,
        limit: int,
    ) -> tuple[int, bytes]:
        """Uses a single deadline; never retries an unguarded mutation."""
        end = self._end(deadline)
        _checked_path(path)
        data = request_data(body)
        for attempt in range(self.retries + 1):
            if end - time.monotonic() <= 0:
                raise _timeout()
            try:
                status, raw = self._send_once(method, path, data, headers, end, limit)
                if status == 503:
                    error = _failure(status, _error_object(raw))
                    if self._retry(method, body, attempt, error):
                        self._pause(attempt, end)
                        continue
                return status, raw
            except ServiceError as error:
                if not self._retry(method, body, attempt, error):
                    raise
                self._pause(attempt, end)
        raise AssertionError("unreachable")

    def _pause(self, attempt: int, end: float) -> None:
        time.sleep(min(self.retry_delay * 2**attempt, max(0, end - time.monotonic())))

    def _send_once(self, method, path, data, headers, end, limit):
        request = Request(
            self.url + path,
            data=data,
            method=method,
            headers=self._headers | headers,
        )
        try:
            response = self._opener.open(request, timeout=end - time.monotonic())
        except HTTPError as error:
            response = error
        except TimeoutError:
            raise _timeout() from None
        except (URLError, OSError, HTTPException) as error:
            if isinstance(getattr(error, "reason", None), TimeoutError):
                raise _timeout() from None
            raise _transport_failure("request failed") from None
        with response:
            status = response.code
            if 300 <= status < 400:
                raise ServiceError(
                    "RedirectRefused", "credential forwarding is disabled", 502
                )
            received = getattr(response, "headers", None)
            if (
                received is not None
                and (received.get("Content-Encoding") or "identity").lower()
                != "identity"
            ):
                raise _compressed()
            try:
                raw = bytearray()
                while len(raw) <= limit:
                    remaining = end - time.monotonic()
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
                    chunk = read(min(65536, limit + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                    if not hasattr(response, "read1"):
                        break
            except TimeoutError:
                raise _timeout() from None
            except (OSError, HTTPException):
                raise _transport_failure("response interrupted") from None
        if len(raw) > limit:
            raise _Oversized
        if time.monotonic() > end:
            raise _timeout()
        return status, bytes(raw)


def _error_object(raw: bytes) -> Any:
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        return None
    return value.get("error") if isinstance(value, dict) else None


class AsyncTransport(_Policy):
    """Native async HTTP with bounded streams, cancellation and connection cleanup.

    `send` and `download` match `Transport`'s, awaited.
    """

    def __init__(
        self,
        url: str,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
        **options: Any,
    ) -> None:
        """Shares credential validation, never sync network I/O or event-loop wrappers."""
        super().__init__(url, **options)
        self._http = httpx.AsyncClient(
            verify=self._tls or True,
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
            transport=http_transport,
        )

    async def aclose(self) -> None:
        """Closes all pooled sockets; later requests raise `ClientClosed`."""
        self.closed = True
        await self._http.aclose()

    async def send(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """Sends one request and returns the reply's JSON object; see `Transport.send`."""
        try:
            status, raw = await self._exchange(
                method, path, body, dict(headers or {}), deadline, MAX_BYTES
            )
            value = response_value(status, raw)
        except _Oversized:
            raise _too_large() from None
        except ServiceError as error:
            if self._probe_needed(method, path, error):
                try:
                    response_value(
                        *await self._exchange(
                            "GET", PLATFORM_PROBE, None, {}, deadline, MAX_BYTES
                        )
                    )
                    probe = None
                except ServiceError as answer:
                    probe = answer
                self._learn(probe)
            self._explain(method, path, error)
            raise
        self._answered(method, path)
        return value

    async def download(
        self, path: str, *, limit: int, deadline: float | None = None
    ) -> bytes:
        """GETs a resource's bytes; more than `limit` of them is `CapacityExceeded`."""
        try:
            status, raw = await self._exchange(
                "GET", path, None, {"Accept": "*/*"}, deadline, limit
            )
        except _Oversized:
            raise _download_too_large(limit) from None
        if not 200 <= status < 300:
            raise _download_failure(status, raw)
        return raw

    async def _exchange(
        self,
        method: str,
        path: str,
        body: Any,
        headers: dict[str, str],
        deadline: float | None,
        limit: int,
    ) -> tuple[int, bytes]:
        """Enforces an overall timeout including retry delays and streaming reads."""
        end = self._end(deadline)
        _checked_path(path)
        data = request_data(body)
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise _timeout()
        try:
            async with asyncio.timeout(remaining):
                for attempt in range(self.retries + 1):
                    try:
                        status, raw = await self._send_once(
                            method, path, data, headers, limit
                        )
                        if status == 503:
                            error = _failure(status, _error_object(raw))
                            if self._retry(method, body, attempt, error):
                                await asyncio.sleep(self.retry_delay * 2**attempt)
                                continue
                        return status, raw
                    except ServiceError as error:
                        if not self._retry(method, body, attempt, error):
                            raise
                    await asyncio.sleep(self.retry_delay * 2**attempt)
        except TimeoutError:
            raise _timeout() from None
        raise AssertionError("unreachable")

    async def _send_once(self, method, path, data, headers, limit):
        if self.closed:
            raise _closed()
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
                        "RedirectRefused", "credential forwarding is disabled", 502
                    )
                if (
                    response.headers.get("Content-Encoding", "identity").lower()
                    != "identity"
                ):
                    raise _compressed()
                raw = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    if len(raw) + len(chunk) > limit:
                        raise _Oversized
                    raw.extend(chunk)
                return response.status_code, bytes(raw)
        except httpx.TimeoutException:
            raise _timeout() from None
        except httpx.TransportError:
            raise _transport_failure("request failed") from None
