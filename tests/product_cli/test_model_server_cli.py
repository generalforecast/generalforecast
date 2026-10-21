"""`gf` against a standalone model server: --base-url, --key-env, typed errors."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from generalforecast import Client
from generalforecast.cli import main, run
from generalforecast.deployments.boundary import Request, Resource, Scope, success
from generalforecast.deployments.contracts import ServiceError
from generalforecast.transport import PLATFORM_PROBE, HostedOnlyError

HOSTED_SECRETS = {
    "GF_API_KEY": "hosted-api-key",
    "GF_TOKEN": "hosted-token",
    "GF_PLATFORM_TOKEN": "hosted-platform-token",
    "GF_MODAL_KEY": "hosted-modal-key",
    "GF_MODAL_SECRET": "hosted-modal-secret",
}


@pytest.fixture
def model_server(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("GF_URL", "https://hosted.example")
    for variable, value in HOSTED_SECRETS.items():
        monkeypatch.setenv(variable, value)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def do_GET(self):
            seen.append((self.command, self.path, dict(self.headers)))
            if self.path == "/v1alpha1/projects/local/models":
                request = Request(
                    Scope("local", self.headers["X-GF-Request-ID"]),
                    Resource("model", "models"),
                    {},
                )
                status = 200
                value = dict(success(request, {"models": [{"name": "m"}]}))
            else:
                status = 404
                value = {"error": {"code": "NotFound", "message": "route not found"}}
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()
    server.server_close()
    worker.join()
    for _method, _path, headers in seen:
        assert "Cookie" not in headers
        assert not any(
            secret in value
            for value in headers.values()
            for secret in HOSTED_SECRETS.values()
        )


def test_base_url_without_key_lists_models_with_no_authorization(model_server):
    url, seen = model_server
    assert "m" in json.dumps(run(["model", "list", "--base-url", url]))
    assert [path for _m, path, _h in seen] == ["/v1alpha1/projects/local/models"]
    assert "Authorization" not in seen[0][2]


def test_base_url_key_env_sends_only_that_bearer(model_server, monkeypatch):
    url, seen = model_server
    monkeypatch.setenv("SERVE_KEY", "serve-key")
    run(["model", "list", "--base-url", url, "--key-env", "SERVE_KEY"])
    assert seen[0][2]["Authorization"] == "Bearer serve-key"
    monkeypatch.setenv("SERVE_KEY", "")
    with pytest.raises(Exception) as error:
        run(["model", "list", "--base-url", url, "--key-env", "SERVE_KEY"])
    assert getattr(error.value, "code", None) == "ValidationError"
    assert len(seen) == 1


def test_hosted_only_command_against_model_server_is_typed(model_server):
    url, seen = model_server
    with pytest.raises(HostedOnlyError) as error:
        run(["deploy", "list", "--base-url", url])
    assert error.value.code == "UnsupportedOperation"
    assert [path for _m, path, _h in seen] == [
        "/v1alpha1/projects/local/deployments",
        PLATFORM_PROBE,
    ]


def test_base_url_rejects_profile_mixing_and_old_local_predict(model_server):
    url, seen = model_server
    with pytest.raises(Exception) as error:
        run(["model", "list", "--base-url", url, "--url", url])
    assert getattr(error.value, "code", None) == "ValidationError"
    with pytest.raises(SystemExit):
        run(["predict", "risk", "--local", "--input", "x.json"])
    assert not seen


def test_client_and_cli_relay_sanitized_server_errors(monkeypatch, capsys):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def do_GET(self):
            body = json.dumps(
                {
                    "error": {
                        "code": "Forbidden",
                        "message": "key lacks deployments:write \x1b[2J",
                    }
                }
            ).encode()
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with (
            Client(url, project="local", token="test-only-api-key") as client,
            pytest.raises(ServiceError) as error,
        ):
            client.models.list()
        assert str(error.value) == "key lacks deployments:write"

        monkeypatch.setenv("GF_API_KEY", "test-only-api-key")
        monkeypatch.setattr(
            sys,
            "argv",
            ["gf", "model", "list", "--base-url", url],
        )
        with pytest.raises(SystemExit) as exit_error:
            main()
        assert exit_error.value.code == 1
        assert json.loads(capsys.readouterr().err) == {
            "error": {
                "code": "Forbidden",
                "message": "key lacks deployments:write",
            }
        }
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
