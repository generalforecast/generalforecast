# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`gf login` over HTTP: the device grant as hosted answers it, through the hardened transport.

The server here answers with envelopes the shared service contract builds
(`ServiceError(...).public_body()`), the same code hosted's device login
raises and serializes, so a change to the wire folding shows up here.
"""

import json
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import generalforecast as gf
from generalforecast.login import DeviceLogin, call
from generalforecast.service.contracts import ServiceError

GRANT = {
    "device_code": "d" * 43,
    "user_code": "ABCD-EFGH",
    "verification_uri": "https://app.example/cli",
    "verification_uri_complete": "https://app.example/cli?code=ABCD-EFGH",
    "expires_in": 600,
    "interval": 5,
}
APPROVED = {
    "access_token": "gf_" + "k" * 43,
    "token_type": "bearer",
    "key": {"id": "key-1", "name": "laptop", "project": "p", "expires_at": 1.0},
    "email": "ada@example.com",
    "workspace": {"id": "w-1", "display_name": "Acme"},
    "project": "p",
}


def refusal(code: str, message: str, status: int) -> tuple[int, dict[str, Any]]:
    """The reply hosted sends for a `ServiceError` its handler raised."""
    return status, ServiceError(code, message, status).public_body()


@contextmanager
def server(replies: deque, seen: list) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def do_GET(self):
            self.reply()

        def do_POST(self):
            self.reply()

        def reply(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            seen.append((self.command, self.path, dict(self.headers), body))
            status, value = replies.popleft()
            data = json.dumps(value).encode()
            self.send_response(status)
            if status in (301, 302, 307, 308):
                self.send_header("Location", value["location"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        worker.join()


def login(url: str) -> tuple[DeviceLogin, list[float]]:
    sleeps: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    return DeviceLogin(url, sleep=sleep, clock=lambda: now[0]), sleeps


def test_device_login_waits_through_pending_and_slow_down_to_approval():
    replies = deque(
        [
            (200, GRANT),
            refusal("AuthorizationPending", "waiting for browser approval", 400),
            refusal("SlowDown", "poll at most every 10s", 400),
            refusal("AuthorizationPending", "waiting for browser approval", 400),
            (200, APPROVED),
        ]
    )
    seen: list = []
    with server(replies, seen) as url:
        flow, sleeps = login(url)
        grant = flow.start("laptop")
        assert flow.wait(grant) == APPROVED
    assert sleeps == [5, 5, 10, 10]
    assert [path for _method, path, _headers, _body in seen] == [
        "/v1/cli-logins",
        *["/v1/cli-logins/token"] * 4,
    ]
    for _method, _path, headers, body in seen[1:]:
        assert json.loads(body) == {"device_code": GRANT["device_code"]}
        assert "Authorization" not in headers


@pytest.mark.parametrize(
    "code,message,status,reason,category",
    [
        (
            "ExpiredToken",
            "the login code expired",
            400,
            "expired_token",
            gf.Unauthorized,
        ),
        (
            "AccessDenied",
            "the login was denied in the browser",
            403,
            "access_denied",
            gf.Unauthorized,
        ),
        ("RateLimited", "too many requests", 429, "rate_limited", gf.CapacityExceeded),
    ],
)
def test_device_login_stops_on_a_final_refusal(code, message, status, reason, category):
    replies = deque([refusal(code, message, status)])
    with server(replies, []) as url:
        flow, _sleeps = login(url)
        with pytest.raises(ServiceError) as error:
            flow.wait(GRANT)
    assert (error.value.reason, error.value.message) == (reason, message)
    assert isinstance(error.value, category)


def test_a_redirect_never_carries_the_key_to_another_origin():
    other_seen: list = []
    with server(deque([(200, {"result": {}})]), other_seen) as other:
        replies = deque([(302, {"location": other + "/steal"})])
        with server(replies, []) as url, pytest.raises(ServiceError) as error:
            call(url, "/v1/auth/status", timeout=5, token="gf_secret_key_123456")
    assert error.value.code == "RedirectRefused"
    assert not other_seen
    assert "gf_secret_key_123456" not in str(error.value)


def test_ambient_proxies_are_ignored_and_plain_http_is_loopback_only(monkeypatch):
    proxy_seen: list = []
    with server(deque([(200, {"via": "proxy"})]), proxy_seen) as proxy:
        for variable in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(variable, proxy)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        with server(deque([(200, {"direct": True})]), []) as url:
            assert call(url, "/v1/auth/status", timeout=5, token="gf_k" * 4) == {
                "direct": True
            }
    assert not proxy_seen
    with pytest.raises(ServiceError) as error:
        call("http://login.example", "/v1/auth/status", timeout=5, token="gf_key")
    assert error.value.code == "ValidationError"
