# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Browser login for the CLI: the OAuth 2.0 device authorization grant.

`gf login` asks the server for a device code, opens the approval page, and
polls until a signed-in browser approves or denies it. Standard library only,
so login stays fast and works before heavier SDK modules load.
"""

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from generalforecast.deployments.contracts import ServiceError

START = "/v1/cli-logins"
TOKEN = "/v1/cli-logins/token"


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
    """GETs, or POSTs a JSON payload, without loading the SDK transport."""
    headers = {
        "Accept": "application/json",
        "X-GF-Request-ID": "r-" + os.urandom(16).hex(),
    }
    if project:
        headers["X-GF-Project"] = project
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        server + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read())
            problem = body["error"]
            code, message = str(problem["code"]), str(problem["message"])
        except (ValueError, KeyError, TypeError):
            code, message = "RequestFailed", f"server returned HTTP {error.code}"
        raise ServiceError(code, message, error.code) from None
    except TimeoutError:
        raise ServiceError(
            "Timeout", f"{server} did not respond in time", 504
        ) from None
    except (urllib.error.URLError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise ServiceError(
            "TransportError", f"cannot reach {server} ({reason})", 503
        ) from None
    except ValueError:
        raise ServiceError(
            "ProtocolError", f"{server} returned invalid JSON", 502
        ) from None


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
        """Polls until approval, honoring interval and slow_down."""
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
                if error.code == "AuthorizationPending":
                    continue
                if error.code == "SlowDown":
                    interval += 5
                    continue
                raise
