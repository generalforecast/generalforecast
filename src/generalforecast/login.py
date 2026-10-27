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

from generalforecast.service.contracts import ServiceError, reason_of

START = "/v1/cli-logins"
TOKEN = "/v1/cli-logins/token"

# How the service says a poll came early. It folds its codes into the wire
# vocabulary and keeps each one's name as the error's `reason`.
PENDING = reason_of("AuthorizationPending")
SLOW_DOWN = reason_of("SlowDown")
# Seconds a `slow_down` adds to the polling interval, as RFC 8628 asks.
SLOW_DOWN_SECONDS = 5


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


def post(
    server: str, path: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """POSTs JSON; server errors become ServiceError with the server's message."""
    return call(server, path, payload, timeout=timeout)


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
    from generalforecast.transport import Transport

    headers = {"X-GF-Request-ID": "r-" + os.urandom(16).hex()}
    if project:
        headers["X-GF-Project"] = project
    transport = Transport(server, token=token, timeout=timeout)
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
        """Requests a device code; returns the RFC 8628 fields."""
        grant = post(self.server, START, {"machine": machine}, self.timeout)
        for key in (
            "device_code",
            "user_code",
            "verification_uri",
            "interval",
            "expires_in",
        ):
            if key not in grant:
                raise ServiceError(
                    "ProtocolError", "login server response is incomplete", 502
                )
        return grant

    def wait(self, grant: dict[str, Any]) -> dict[str, Any]:
        """Polls until approval, honoring the interval and every `slow_down`.

        Raises:
            ServiceError: `ExpiredToken` once the code expires, or the
                server's refusal: an expired or consumed code
                (`expired_token`), a denial in the browser (`access_denied`),
                or a rate limit.
        """
        interval = max(int(grant["interval"]), 1)
        deadline = self.clock() + int(grant["expires_in"])
        while True:
            self.sleep(interval)
            if self.clock() > deadline:
                raise ServiceError("ExpiredToken", "the login code expired", 400)
            try:
                return post(
                    self.server,
                    TOKEN,
                    {"device_code": grant["device_code"]},
                    self.timeout,
                )
            except ServiceError as error:
                if error.reason == PENDING:
                    continue
                if error.reason == SLOW_DOWN:
                    interval += SLOW_DOWN_SECONDS
                    continue
                raise
