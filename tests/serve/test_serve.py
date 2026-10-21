"""`gf serve`, the standalone Models API server. Injected kernels are not model evidence."""

import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import yaml

from generalforecast.client import Client
from generalforecast.deployments.boundary import MAX_BYTES, Scope
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.contracts import ServiceError
from generalforecast.packages import Package
from generalforecast.packages.validation import PackageError
from generalforecast.serve import Application, Server, load
from generalforecast.serve.listener import public_warning
from generalforecast.tabular import arrow_bytes

ROOT = Path(__file__).resolve().parents[2]
KEY = "gf-serve-test-key"
KERNELS = """
import os, time
from types import SimpleNamespace
import numpy as np
import generalforecast.deployments.package_runtime as package_runtime
import generalforecast.models.inference as inference
import generalforecast.models.registry as registry

package_runtime._load = lambda *args: (SimpleNamespace(), None)
registry.load_ref = lambda *args, **kwargs: SimpleNamespace()

def fit(model, x, y, count):
    return count, float(np.mean(np.asarray(y)))

def predict(model, fitted, x, classes):
    time.sleep(float(os.environ.get("GF_TEST_DELAY", "0")))
    count, cache = fitted
    if count:
        return np.tile([0.25, 0.75], (len(x), 1))
    return np.tile(np.arange(999, dtype=float) + cache, (len(x), 1))

inference.fit, inference.predict = fit, predict
"""
FORBIDDEN = (
    "generalforecast.server",
    "generalforecast.deployments.identity",
    "generalforecast.deployments.workspace_",
    "generalforecast.deployments.security",
    "generalforecast.deployments.store",
    "generalforecast.deployments.postgres",
    "generalforecast.deployments.routes",
    "generalforecast.deployments.revision_api",
    "generalforecast.deployments.jobs",
    "generalforecast.deployments.sources",
    "generalforecast.deployments.packages",
    "generalforecast.deployments.python_runtime",
    "generalforecast.deployments.settings",
    "generalforecast.deployments.providers",
    "generalforecast.deployments.web",
    "generalforecast.deployments.runtime",
    "generalforecast.deployments.handoff",
    "generalforecast.deployments.data",
    "generalforecast.deployments.blobs",
    "generalforecast.deployments.operators",
    "generalforecast.deployments.sql",
    "psycopg",
    "google.cloud",
)


@pytest.fixture
def kernels(monkeypatch):
    fits = []
    monkeypatch.setattr(
        "generalforecast.deployments.package_runtime._load",
        lambda *args: (SimpleNamespace(), None),
    )
    monkeypatch.setattr(
        "generalforecast.models.registry.load_ref",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    delay = {"seconds": 0.0}

    def fit(model, x, y, count):
        fits.append(np.asarray(x).copy())
        return count, float(np.mean(np.asarray(y)))

    def predict(model, fitted, x, classes):
        time.sleep(delay["seconds"])
        count, cache = fitted
        if count:
            return np.tile([0.25, 0.75], (len(x), 1))
        return np.tile(np.arange(999, dtype=float) + cache, (len(x), 1))

    monkeypatch.setattr("generalforecast.models.inference.fit", fit)
    monkeypatch.setattr("generalforecast.models.inference.predict", predict)
    return SimpleNamespace(fits=fits, delay=delay)


def history():
    return pa.table(
        {
            "id": ["a", "b", "c", "d"],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [False, False, True, True],
        }
    )


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "churn"
    root.mkdir()
    spec = {
        "name": "churn",
        "model": "tabicl-classifier",
        "context": "history.arrow",
        "target": "y",
        "id": "id",
    }
    (root / "gf.yaml").write_text(yaml.safe_dump(spec))
    (root / "history.arrow").write_bytes(arrow_bytes(history()))
    Package.load(root).check()
    return root


@pytest.fixture
def lone_lock(tmp_path):
    root = tmp_path / "request"
    root.mkdir()
    spec = {
        "name": "scorer",
        "model": "tabicl-classifier",
        "context": "request",
        "target": "y",
    }
    (root / "gf.yaml").write_text(yaml.safe_dump(spec))
    Package.load(root).check(data=history().drop(["id"]))
    lone = tmp_path / "elsewhere"
    lone.mkdir()
    shutil.copy(root / "gf.lock", lone / "gf.lock")
    return lone / "gf.lock"


@pytest.fixture
def serve(kernels):
    servers = []

    def start(target, **options):
        application = Application([load(str(target))], **options)
        server = Server(application, "127.0.0.1", 0).start()
        assert server.wait_ready(30)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


def envelope(resource, payload, *, project="local", request_id="r-test"):
    return {
        "api_version": "gf/v1alpha1",
        "project": project,
        "request_id": request_id,
        "resource": {"kind": "model", "name": resource},
        "payload": payload,
    }


def call(url, path, body=None, *, key=None, headers=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    sent = {"Content-Type": "application/json"} if body is not None else {}
    if key is not None:
        sent["Authorization"] = "Bearer " + key
    sent |= headers or {}
    request = urllib.request.Request(
        url + path,
        data=data,
        method=method or ("POST" if body is not None else "GET"),
        headers=sent,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def predictions_payload(**changes):
    return {
        "model": "churn",
        "task": "classification",
        "target": "y",
        "features": ["x"],
        "context": [],
        "queries": [{"id": "q", "x": 1.5}],
        "options": {},
    } | changes


def infer_payload(**changes):
    return {
        "model": "tabicl-classifier",
        "task": "classification",
        "target": "y",
        "features": ["x"],
        "context": [
            {"key": "c1", "features": {"x": 0.0}, "label": False},
            {"key": "c2", "features": {"x": 3.0}, "label": True},
        ],
        "queries": [{"key": "q", "features": {"x": 1.5}}],
        "options": {},
    } | changes


def test_health_and_package_routes_over_http(serve, package, kernels):
    server = serve(package)
    url = server.url
    assert call(url, "/livez") == (200, {"live": True})
    assert call(url, "/readyz") == (200, {"ready": True})
    status, body = call(url, "/v1alpha1/projects/local/models")
    assert status == 200
    assert body["api_version"] == "gf/v1alpha1"
    assert body["project"] == "local"
    assert body["resource"] == {"kind": "model", "name": "models"}
    [described] = body["result"]["models"]
    assert described["name"] == "churn" and described["kind"] == "package"
    assert described["features"] == ["x"] and described["context_rows"] == 4
    status, body = call(url, "/v1alpha1/projects/team-a/models/churn")
    assert status == 200 and body["project"] == "team-a"
    assert body["result"] == described
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope("predictions", predictions_payload()),
        headers={"X-GF-Request-ID": "r-test", "X-GF-Project": "local"},
    )
    assert status == 200, body
    assert body["request_id"] == "r-test"
    result = body["result"]
    assert result["predictions"] == [
        {"id": "q", "label": True, "probabilities": {"false": 0.25, "true": 0.75}}
    ]
    assert result["task"] == "classification"
    assert result["context"]["source"] == "snapshot"
    assert result["context"]["rows"] == 4
    assert len(kernels.fits) == 1
    status, body = call(
        url,
        "/v1alpha1/projects/local/models/churn/infer",
        envelope("churn", predictions_payload()),
    )
    assert status == 422 and body["error"]["code"] == "UnsupportedCapability"
    assert body["request_id"] == "r-test"


def test_catalog_model_matches_the_hosted_models_api(serve, kernels):
    server = serve("tabicl-classifier")
    url = server.url
    hosted = CatalogService(allow_downloads=True)
    scope = Scope("local", "r-test")
    status, body = call(url, "/v1alpha1/projects/local/models/tabicl-v2-classifier")
    assert status == 200
    assert body["result"] == hosted.get(scope, "tabicl-classifier")
    status, body = call(url, "/v1alpha1/projects/local/models")
    assert body["result"] == {"models": [hosted.get(scope, "tabicl-classifier")]}
    payload = infer_payload()
    status, body = call(
        url,
        "/v1alpha1/projects/local/models/tabicl-v2-classifier/infer",
        envelope("tabicl-v2-classifier", payload),
    )
    assert status == 200, body
    assert body["result"] == hosted.infer(scope, payload)
    assert [row["key"] for row in body["result"]["outputs"]] == ["q"]
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope(
            "predictions",
            predictions_payload(
                model="tabicl-classifier",
                context=[{"x": 0.0, "y": False}, {"x": 3.0, "y": True}],
                queries=[{"x": 1.5}],
            ),
        ),
    )
    assert status == 200, body
    assert body["result"]["predictions"][0]["label"] is True
    assert body["result"]["context"]["source"] == "request"


def test_unserved_models_and_unknown_routes_are_not_found(serve, package):
    url = serve(package).url
    for path in (
        "/v1alpha1/auth/status",
        "/v1alpha1/projects/local/deployments",
        "/v1alpha1/projects/local/keys",
        "/v1alpha1/workspaces",
        "/",
    ):
        status, body = call(url, path)
        assert status == 404, path
        assert body == {"error": {"code": "NotFound", "message": "route not found"}}
    status, body = call(url, "/v1alpha1/projects/local/models/other")
    assert status == 404 and body["error"]["code"] == "ModelNotFound"
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope("predictions", predictions_payload(model="other")),
    )
    assert status == 404 and body["error"]["code"] == "ModelNotFound"
    assert body["resource"] == {"kind": "model", "name": "predictions"}
    status, body = call(url, "/v1alpha1/projects/local/models", method="DELETE")
    assert status == 405 and body["error"]["code"] == "MethodNotAllowed"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"target": "z"}, "ValidationError"),
        ({"features": ["x", "w"]}, "ValidationError"),
        ({"task": "regression"}, "UnsupportedTask"),
        ({"options": {"quantiles": [0.5]}}, "ValidationError"),
        ({"context": [{"x": 1.0, "y": True}]}, "ValidationError"),
        ({"queries": []}, "ValidationError"),
    ],
)
def test_package_requests_must_match_the_lock(serve, package, change, code):
    url = serve(package).url
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope("predictions", predictions_payload(**change)),
    )
    assert status == 422 and body["error"]["code"] == code


def test_envelope_mismatches_are_rejected(serve, package):
    url = serve(package).url
    path = "/v1alpha1/projects/local/predictions"
    for body, headers in (
        (envelope("predictions", predictions_payload(), project="other"), {}),
        (envelope("churn", predictions_payload()), {}),
        (envelope("predictions", predictions_payload()), {"X-GF-Project": "x"}),
        (envelope("predictions", predictions_payload()), {"X-GF-Request-ID": "r-2"}),
        ({"payload": {}}, {}),
    ):
        status, error = call(url, path, body, headers=headers)
        assert status == 422 and error["error"]["code"] == "ValidationError"
    status, _ = call(
        url, path, envelope("predictions", {}), headers={"Content-Type": "text/plain"}
    )
    assert status == 422


def test_api_key_is_required_when_configured(serve, package):
    url = serve(package, api_key=KEY).url
    assert call(url, "/livez") == (200, {"live": True})
    assert call(url, "/readyz") == (200, {"ready": True})
    path = "/v1alpha1/projects/local/predictions"
    body = envelope("predictions", predictions_payload())
    for key, headers in (
        (None, {}),
        ("wrong-key", {}),
        (KEY[:-1], {}),
        (None, {"Authorization": KEY}),
        (None, {"Authorization": "Basic " + KEY}),
        (None, {"X-GF-API-Key": KEY}),
    ):
        status, error = call(url, path, body, key=key, headers=headers)
        assert status == 401, (key, headers)
        assert error["error"]["code"] == "Unauthorized"
        assert error["request_id"] == "r-test"
        status, error = call(
            url, "/v1alpha1/projects/local/models", key=key, headers=headers
        )
        assert status == 401 and error["error"]["code"] == "Unauthorized"
    status, result = call(url, path, body, key=KEY)
    assert status == 200 and result["result"]["predictions"][0]["label"] is True


def test_sdk_client_works_against_the_standalone_server(serve):
    url = serve("tabicl-classifier", api_key=KEY).url
    with Client(url, project="local", api_key=KEY) as client:
        [model] = client.models.list()
        assert model["name"] == "tabicl-v2-classifier"
        assert client.models.get("tabicl-v2-classifier")["task"] == "classification"
        result = client.models.infer(infer_payload())  # type: ignore[arg-type]
        assert result["outputs"][0]["key"] == "q"
        prediction = client.predict(
            model="tabicl-classifier",
            target="y",
            context=pa.table({"x": [0.0, 3.0], "y": [False, True]}),
            queries=pa.table({"x": [1.5]}),
        )
        assert prediction.table.column("label").to_pylist() == [True]


def test_lone_lock_file_serves_request_context(serve, lone_lock, kernels):
    url = serve(lone_lock).url
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope(
            "predictions",
            predictions_payload(
                model="scorer",
                context=[
                    {"x": 0.0, "y": False},
                    {"x": 1.0, "y": False},
                    {"x": 3.0, "y": True},
                ],
                queries=[{"x": 2.5}],
            ),
        ),
    )
    assert status == 200, body
    assert body["result"]["context"] == {
        **body["result"]["context"],
        "source": "request",
        "rows": 3,
    }
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope("predictions", predictions_payload(model="scorer")),
    )
    assert status == 422


def test_local_only_models_serve_on_the_operators_own_server(serve):
    url = serve("tabpfn-3-classifier").url
    own = CatalogService(allow_downloads=True, surface="self-hosted")
    scope = Scope("local", "r-test")
    with pytest.raises(ServiceError) as caught:
        CatalogService().get(scope, "tabpfn-3-classifier")
    assert caught.value.code == "UnsupportedModel"
    status, body = call(url, "/v1alpha1/projects/local/models/tabpfn-3-classifier")
    assert status == 200
    assert body["result"] == own.get(scope, "tabpfn-3-classifier")
    assert body["result"]["license"] == "LicenseRef-tabpfn-3-license-v1.0"
    payload = infer_payload(model="tabpfn-3-classifier")
    status, body = call(
        url,
        "/v1alpha1/projects/local/models/tabpfn-3-classifier/infer",
        envelope("tabpfn-3-classifier", payload),
    )
    assert status == 200, body
    assert body["result"] == own.infer(scope, payload)
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope("predictions", predictions_payload(model="tabpfn-3-classifier")),
    )
    assert status == 422 and body["error"]["code"] == "UnsupportedCapability"


def test_unservable_targets_fail_before_listening(tmp_path, package):
    with pytest.raises(PackageError) as caught:
        load(str(tmp_path / "missing-model"))
    assert caught.value.code == "ModelNotFound"
    assert load("timesfm-3").name == "timesfm-3"
    for model, code in (
        ("tabpfn-3-classifier@4", "UnsupportedCapability"),
        ("limix-2-classifier", "UnsupportedModel"),
    ):
        with pytest.raises(PackageError) as caught:
            load(model)
        assert caught.value.code == code, model
    (tmp_path / "unlocked").mkdir()
    (tmp_path / "unlocked" / "gf.yaml").write_text(
        yaml.safe_dump({"name": "u", "model": "tabicl-classifier", "target": "y"})
    )
    with pytest.raises(PackageError) as caught:
        load(str(tmp_path / "unlocked"))
    assert caught.value.code == "LockRequired"
    lone = tmp_path / "lone"
    lone.mkdir()
    shutil.copy(package / "gf.lock", lone / "gf.lock")
    with pytest.raises(PackageError) as caught:
        load(str(lone / "gf.lock"))
    assert caught.value.code == "ContextRequired"
    (package / "history.arrow").write_bytes(arrow_bytes(history().slice(0, 3)))
    with pytest.raises(PackageError):
        load(str(package))


def test_bounded_body_and_loopback_origin(serve, package):
    server = serve(package)
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    connection.putrequest("POST", "/v1alpha1/projects/local/predictions")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", str(MAX_BYTES + 1))
    connection.endheaders()
    response = connection.getresponse()
    assert response.status == 413
    assert json.loads(response.read())["error"]["code"] == "CapacityExceeded"
    connection.close()
    for headers in (
        {"Origin": "http://evil.example"},
        {"Host": "evil.example"},
    ):
        status, body = call(
            server.url, "/v1alpha1/projects/local/models", headers=headers
        )
        assert status == 422 and body["error"]["code"] == "ValidationError"


def test_inference_deadline_returns_timeout(serve, package, kernels):
    url = serve(package, timeout=0.2).url
    kernels.delay["seconds"] = 1.0
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope("predictions", predictions_payload()),
    )
    assert status == 504 and body["error"]["code"] == "RequestTimeout"


def test_concurrent_requests_share_one_loaded_package(serve, package, kernels):
    url = serve(package, concurrency=4).url
    kernels.delay["seconds"] = 0.05

    def one(index):
        return call(
            url,
            "/v1alpha1/projects/local/predictions",
            envelope(
                "predictions",
                predictions_payload(queries=[{"id": f"q{index}", "x": 1.5}]),
                request_id=f"r-{index}",
            ),
        )

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(one, range(32)))
    assert all(status == 200 for status, _ in results)
    assert [body["request_id"] for _, body in results] == [
        f"r-{index}" for index in range(32)
    ]
    assert [body["result"]["predictions"][0]["id"] for _, body in results] == [
        f"q{index}" for index in range(32)
    ]
    assert len(kernels.fits) == 1


class Gated:
    name = "gated"

    def __init__(self):
        self.release = threading.Event()

    def warm(self):
        assert self.release.wait(30)

    def describe(self, scope):
        return {"name": self.name}

    def infer(self, scope, payload):
        return {}

    def predict(self, scope, payload):
        return {}


def test_readiness_waits_for_loading_and_drains_on_stop():
    model = Gated()
    server = Server(Application([model]), "127.0.0.1", 0).start()
    try:
        assert call(server.url, "/livez") == (200, {"live": True})
        assert call(server.url, "/readyz") == (503, {"ready": False})
        status, body = call(server.url, "/v1alpha1/projects/local/models")
        assert status == 503 and body["error"]["code"] == "NotReady"
        model.release.set()
        assert server.wait_ready(10)
        assert call(server.url, "/readyz") == (200, {"ready": True})
        assert call(server.url, "/v1alpha1/projects/local/models")[0] == 200
    finally:
        server.stop()
    with pytest.raises(OSError):
        call(server.url, "/livez")


def _start(arguments, *, env=None, entry="serve"):
    launcher = (
        "from generalforecast.serve import main; main(sys.argv[1:])"
        if entry == "serve"
        else "from generalforecast.cli import main; sys.argv = ['gf', *sys.argv[1:]]; main()"
    )
    return subprocess.Popen(
        [sys.executable, "-c", "import sys\n" + KERNELS + "\n" + launcher, *arguments],
        cwd=ROOT,
        env={k: v for k, v in os.environ.items() if k != "GF_API_KEY"} | (env or {}),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _event(process, expected):
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        assert line, process.stderr.read() if process.stderr else ""
        value = json.loads(line)
        if value.get("event") == expected:
            return value


@pytest.mark.parametrize("entry", ["serve", "cli"])
def test_public_bind_warns_and_sigterm_drains_in_flight(package, entry):
    process = _start(
        ([] if entry == "serve" else ["serve"])
        + [str(package), "--host", "0.0.0.0", "--port", "0"],
        env={"GF_TEST_DELAY": "1"},
        entry=entry,
    )
    try:
        serving = _event(process, "serving")
        assert serving["auth"] == "none" and serving["models"] == ["churn"]
        port = int(serving["url"].rsplit(":", 1)[1])
        url = f"http://127.0.0.1:{port}"
        _event(process, "ready")
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(
                call,
                url,
                "/v1alpha1/projects/local/predictions",
                envelope("predictions", predictions_payload()),
            )
            time.sleep(0.4)
            process.send_signal(signal.SIGTERM)
            status, body = pending.result(timeout=30)
        assert status == 200, body
        assert process.wait(timeout=30) == 0
        assert _event(process, "stopped")["url"] == serving["url"]
        assert process.stderr is not None
        assert public_warning("0.0.0.0", port) in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()


def test_api_key_from_environment_silences_the_warning(package):
    process = _start(
        [str(package), "--host", "0.0.0.0", "--port", "0"], env={"GF_API_KEY": KEY}
    )
    try:
        serving = _event(process, "serving")
        assert serving["auth"] == "bearer"
        port = int(serving["url"].rsplit(":", 1)[1])
        _event(process, "ready")
        url = f"http://127.0.0.1:{port}"
        assert call(url, "/v1alpha1/projects/local/models")[0] == 401
        assert call(url, "/v1alpha1/projects/local/models", key=KEY)[0] == 200
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=30) == 0
        assert process.stderr is not None
        stderr = process.stderr.read()
        assert "warning" not in stderr and KEY not in stderr
    finally:
        if process.poll() is None:
            process.kill()


def test_server_imports_nothing_from_the_private_control_plane(package):
    code = (
        "import json, sys\n"
        + KERNELS
        + """
import generalforecast.serve
from generalforecast.serve import Application, Server, load
import generalforecast.serve.__main__
import generalforecast.deployments.catalog_forecast
application = Application([load(sys.argv[1]), load("tabicl-classifier")])
application.warm()
body = json.dumps({
    "api_version": "gf/v1alpha1", "project": "local", "request_id": "r-1",
    "resource": {"kind": "model", "name": "predictions"},
    "payload": {"model": "churn", "task": "classification", "target": "y",
                "features": ["x"], "context": [], "queries": [{"x": 1.5}],
                "options": {}},
}).encode()
response = application.handle(
    "POST", "/v1alpha1/projects/local/predictions",
    [("Host", "127.0.0.1"), ("Content-Type", "application/json")], body,
)
assert response.status == 200, response.body
application.close()
print(json.dumps(sorted(sys.modules)))
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(package)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    modules = json.loads(result.stdout.splitlines()[-1])
    assert "generalforecast.serve.app" in modules
    leaked = [m for m in modules if m.startswith(FORBIDDEN)]
    assert not leaked
