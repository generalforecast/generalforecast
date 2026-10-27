# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Browser login for the CLI: the OAuth 2.0 device authorization grant.

`gf login` asks the server for a device code, opens the approval page, and
polls until a signed-in browser approves or denies it. Requests go through
the SDK's hardened `Transport`, so a key is never forwarded by a redirect,
read through an ambient proxy or sent over plain HTTP off loopback.
"""

import os
import socket
import sys
import time
from collections.abc import Callable
from typing import Any

from generalforecast.service.contracts import ServiceError, public_code, reason_of

START = "/v1/cli-logins"
TOKEN = "/v1/cli-logins/token"

# How the service says a poll came early: RFC 8628's 400, the wire code it
# folds the refusal into, and its name as the error's `reason`. A poll keeps
# waiting only for exactly these.
_EARLY = 400
PENDING = (public_code("AuthorizationPending", _EARLY), _EARLY)
PENDING_REASON = reason_of("AuthorizationPending")
SLOW_DOWN = (public_code("SlowDown", _EARLY), _EARLY)
SLOW_DOWN_REASON = reason_of("SlowDown")
# Seconds a `slow_down` adds to the polling interval, as RFC 8628 asks.
SLOW_DOWN_SECONDS = 5
# The longest polling interval and code lifetime a server may set, in seconds.
MAX_INTERVAL = 60
MAX_EXPIRES_IN = 3600


def machine_name() -> str:
    """The short host name the approved key is named after."""
    return (socket.gethostname().split(".")[0] or "unknown")[:64]


def browser_unavailable(environ: dict[str, str] | None = None) -> str | None:
    """Why no browser can open here (SSH, no display, GF_NO_BROWSER), or None."""
    env = os.environ if environ is None else environ
    if env.get("GF_NO_BROWSER"):
        return "GF_NO_BROWSER is set"
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        return "SSH session"
    if sys.platform.startswith("linux") and not (
        env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")
    ):
        return "no display"
    return None


def call(
    server: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
    token: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """GETs, or POSTs a JSON payload, through the hardened transport.

    Raises:
        ServiceError: The server's error, with its wire code and `reason`;
            a refused redirect or a plain-HTTP server off loopback.
    """
    from generalforecast.transport import MAX_TIMEOUT, MIN_TIMEOUT, Transport

    headers = {"X-GF-Request-ID": "r-" + os.urandom(16).hex()}
    if project:
        headers["X-GF-Project"] = project
    # One request is bounded as every SDK request is, whatever a flag asks.
    bounded = min(max(timeout, MIN_TIMEOUT), MAX_TIMEOUT)
    transport = Transport(server, token=token, timeout=bounded)
    try:
        return transport.send(
            "GET" if payload is None else "POST", path, payload, headers
        )
    finally:
        transport.close()


class DeviceLogin:
    """One login attempt against one server."""

    def __init__(
        self,
        server: str,
        *,
        timeout: float = 30,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Injectable sleep and clock keep tests instant."""
        self.server, self.timeout, self.sleep, self.clock = (
            server,
            timeout,
            sleep,
            clock,
        )

    def start(self, machine: str) -> dict[str, Any]:
        """Requests a device code; returns the RFC 8628 fields, checked."""
        return checked_grant(
            call(self.server, START, {"machine": machine}, timeout=self.timeout)
        )

    def wait(self, grant: dict[str, Any]) -> dict[str, Any]:
        """Polls until approval, honoring the interval and every `slow_down`.

        Raises:
            ServiceError: `ExpiredToken` once the code expires, or the
                server's refusal: an expired or consumed code
                (`expired_token`), a denial in the browser (`access_denied`),
                or a rate limit.
        """
        grant = checked_grant(grant)
        interval = grant["interval"]
        deadline = self.clock() + grant["expires_in"]
        while True:
            left = deadline - self.clock()
            if left <= 0:
                raise _expired()
            self.sleep(min(interval, left))
            if self.clock() >= deadline:
                raise _expired()
            try:
                return call(
                    self.server,
                    TOKEN,
                    {"device_code": grant["device_code"]},
                    timeout=self.timeout,
                )
            except ServiceError as error:
                answer = (error.public, error.status)
                if answer == PENDING and error.reason == PENDING_REASON:
                    continue
                if answer == SLOW_DOWN and error.reason == SLOW_DOWN_REASON:
                    interval = min(interval + SLOW_DOWN_SECONDS, MAX_INTERVAL)
                    continue
                raise


def _expired() -> ServiceError:
    return ServiceError("ExpiredToken", "the login code expired", 400)


def _incomplete() -> ServiceError:
    return ServiceError("ProtocolError", "the login server's grant is invalid", 502)


def checked_grant(grant: Any) -> dict[str, Any]:
    """A device grant whose fields are what RFC 8628 says, within this client's bounds.

    Raises:
        ServiceError: `ProtocolError` for a missing or malformed field, or an
            interval or lifetime past `MAX_INTERVAL` or `MAX_EXPIRES_IN`.
    """
    if not isinstance(grant, dict):
        raise _incomplete()
    for key in ("device_code", "user_code", "verification_uri"):
        if type(grant.get(key)) is not str or not grant[key]:
            raise _incomplete()
    for key, bound in (("interval", MAX_INTERVAL), ("expires_in", MAX_EXPIRES_IN)):
        value = grant.get(key)
        if type(value) is not int or not 1 <= value <= bound:
            raise _incomplete()
    return grant
