"""HTTP contract recorder, not a model or provider implementation."""

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pytest

from generalforecast.deployments.boundary import (
    Request,
    Resource,
    ResourceKind,
    Scope,
    success,
)
from generalforecast.packages.validation import bytes_digest
from generalforecast.tabular import table_digest
from generalforecast.transport import PLATFORM_PROBE


@pytest.fixture
def wire():
    expected = deque()
    state = SimpleNamespace(
        expected=expected, requests=[], errors=[], corrupt=False, probes=0
    )

    def expect(method, path, result=None, *, payload=None, status=200):
        expected.append(
            (method, "/v1alpha1/projects/test" + path, payload, result, status)
        )

    state.expect = expect

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def do_GET(self):
            self.reply()

        def do_POST(self):
            self.reply()

        def reply(self):
            if self.path == PLATFORM_PROBE:
                state.probes += 1
                data = json.dumps(
                    {"error": {"code": "Unauthenticated", "message": "fixture"}}
                ).encode()
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            try:
                method, path, payload, result, status = expected.popleft()
                assert self.command == method and self.path == path
                assert self.headers["Authorization"] == "Bearer fixture-key"
                assert self.headers["X-GF-Project"] == "test"
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if (
                    self.headers.get("Content-Type")
                    == "application/vnd.apache.arrow.stream"
                ):
                    request = Request(
                        Scope("test", self.headers["X-GF-Request-ID"]),
                        Resource("dataset", "snapshots"),
                        {},
                    )
                    table = pa.ipc.open_stream(raw).read_all()
                    result = {"digest": table_digest(table), "rows": table.num_rows}
                    if state.corrupt:
                        result["rows"] = (
                            True if state.corrupt == "boolean" else result["rows"] + 1
                        )
                    state.requests.append((self.path, table))
                elif (
                    self.headers.get("Content-Type")
                    == "application/vnd.generalforecast.python-source+tar"
                ):
                    deployment = self.path.split("/")[5]
                    request = Request(
                        Scope("test", self.headers["X-GF-Request-ID"]),
                        Resource("deployment", deployment),
                        {},
                    )
                    if payload is not None:
                        assert raw == payload
                    if result is None:
                        result = {
                            "receipt_version": 1,
                            "kind": "python-source",
                            "project": "test",
                            "deployment": deployment,
                            "source": bytes_digest(raw),
                            "size": len(raw),
                        }
                    state.requests.append((self.path, raw))
                elif self.command == "POST":
                    request = Request.parse(json.loads(raw))
                    assert request.payload == payload
                    assert request.scope.project == "test"
                    state.requests.append((self.path, request.payload))
                else:
                    assert not raw
                    parts = self.path.split("/")
                    kind = {
                        "deployments": "deployment",
                        "datasets": "dataset",
                        "jobs": "job",
                        "runtimes": "operator",
                    }[parts[4]]
                    resource = parts[5] if len(parts) > 5 else parts[4]
                    request = Request(
                        Scope("test", self.headers["X-GF-Request-ID"]),
                        Resource(cast(ResourceKind, kind), resource),
                        {},
                    )
                    state.requests.append((self.path, None))
                response = (
                    success(request, result)
                    if status == 200
                    else {
                        "error": {
                            "code": result,
                            "message": "fixture-key must not be reflected",
                        }
                    }
                )
            except (
                AssertionError,
                IndexError,
                KeyError,
                ValueError,
                TypeError,
            ) as error:
                state.errors.append(str(error))
                status, response = (
                    500,
                    {"error": {"code": "FixtureError", "message": "contract mismatch"}},
                )
            data = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()
    worker.join()
    assert not state.errors, state.errors
    assert not expected, expected
