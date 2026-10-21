"""`gf serve`, which runs gf-server's Models API over a catalog model or a locked package."""

import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from generalforecast.client import Client
from generalforecast.deployments.boundary import Scope
from generalforecast.deployments.catalog import CatalogService
from generalforecast.packages import Package
from generalforecast.packages.validation import PackageError
from generalforecast.serve.models import served
from generalforecast.tabular import arrow_bytes

ROOT = Path(__file__).resolve().parents[2]
KEY = "gf-serve-test-key"
CLEAN = (
    "GF_MICROSOFT_CLIENT_SECRET",
    "GF_MICROSOFT_CLIENT_ID",
    "GF_GITHUB_CLIENT_ID",
    "GF_GITHUB_CLIENT_SECRET",
)


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def envelope(resource, payload, *, project="local", request_id="r-test"):
    return {
        "api_version": "gf/v1alpha1",
        "project": project,
        "request_id": request_id,
        "resource": {"kind": "model", "name": resource},
        "payload": payload,
    }


def call(url, path, body=None, *, key=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    sent = {"Content-Type": "application/json"} if body is not None else {}
    if key is not None:
        sent["Authorization"] = "Bearer " + key
    sent |= headers or {}
    request = urllib.request.Request(url + path, data=data, headers=sent)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


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


def _start(arguments, *, env=None):
    from tests.utils.server import build

    binary = build()
    environment = {k: v for k, v in os.environ.items() if k not in CLEAN}
    environment["PATH"] = f"{binary.parent}{os.pathsep}{environment['PATH']}"
    environment |= env or {}
    return subprocess.Popen(
        [sys.executable, "-m", "generalforecast.serve", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=ROOT,
    )


def _event(process, expected):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            raise AssertionError(process.stderr.read())
        event = json.loads(line)
        if event.get("event") == expected:
            return event
    raise AssertionError(f"no {expected} event")


@pytest.fixture(scope="module")
def server():
    port = _port()
    process = _start(
        ["tabicl-classifier", "--port", str(port)], env={"GF_API_KEY": KEY}
    )
    serving = _event(process, "serving")
    assert serving == {
        "event": "serving",
        "url": f"http://127.0.0.1:{port}",
        "models": ["tabicl-v2-classifier"],
        "auth": "bearer",
    }
    _event(process, "ready")
    yield serving["url"], process
    process.send_signal(signal.SIGTERM)
    assert _event(process, "stopped")["url"] == serving["url"]
    assert process.wait(60) == 0


@pytest.mark.slow
def test_serve_answers_the_hosted_models_api(server):
    url, _ = server
    hosted = CatalogService(allow_downloads=True)
    scope = Scope("local", "r-test")
    assert call(url, "/livez") == (200, {"live": True})
    assert call(url, "/readyz") == (200, {"ready": True})
    status, body = call(url, "/v1alpha1/projects/local/models", key=KEY)
    assert status == 200
    assert body["result"] == {"models": [hosted.get(scope, "tabicl-classifier")]}
    path = "/v1alpha1/projects/local/models/tabicl-v2-classifier/infer"
    status, body = call(
        url, path, envelope("tabicl-v2-classifier", infer_payload()), key=KEY
    )
    assert status == 200, body
    expected = hosted.infer(scope, infer_payload())
    got = body["result"]
    assert {k: v for k, v in got.items() if k != "outputs"} == {
        k: v for k, v in expected.items() if k != "outputs"
    }
    [output] = got["outputs"]
    [reference] = expected["outputs"]
    assert output["key"] == "q"
    for label, probability in reference["probabilities"].items():
        assert output["probabilities"][label] == pytest.approx(probability, abs=1e-3)


@pytest.mark.slow
def test_serve_requires_its_key_and_keeps_the_api_errors(server):
    url, _ = server
    status, body = call(url, "/v1alpha1/projects/local/models")
    assert (status, body["error"]["message"]) == (401, "application bearer required")
    status, body = call(url, "/v1alpha1/projects/local/models/other", key=KEY)
    assert (status, body["error"]["code"]) == (404, "ModelNotFound")
    path = "/v1alpha1/projects/local/models/tabicl-v2-classifier/infer"
    payload = infer_payload(task="regression")
    status, body = call(url, path, envelope("tabicl-v2-classifier", payload), key=KEY)
    assert (status, body["error"]["code"]) == (422, "UnsupportedTask")
    assert body["request_id"] == "r-test"


@pytest.mark.slow
def test_sdk_client_and_predictions_work_against_gf_serve(server):
    url, _ = server
    client = Client(base_url=url, api_key=KEY)
    [described] = client.models.list()
    assert described["name"] == "tabicl-v2-classifier"
    payload = {
        "model": "tabicl-classifier",
        "task": "classification",
        "target": "y",
        "features": ["x"],
        "context": [{"x": 0.0, "y": False}, {"x": 3.0, "y": True}],
        "queries": [{"id": "a", "x": 1.5}],
        "options": {"id": "id"},
    }
    status, body = call(
        url,
        "/v1alpha1/projects/local/predictions",
        envelope("predictions", payload),
        key=KEY,
    )
    assert status == 200, body
    [row] = body["result"]["predictions"]
    assert row["id"] == "a" and type(row["label"]) is bool
    assert set(row["probabilities"]) == {"false", "true"}
    assert body["result"]["context"]["rows"] == 2


@pytest.mark.slow
def test_public_bind_without_a_key_warns():
    port = _port()
    process = _start(["tabicl-classifier", "--host", "0.0.0.0", "--port", str(port)])
    try:
        _event(process, "serving")
        warning = process.stderr.readline()
        while "warning" not in warning:
            warning = process.stderr.readline()
        assert warning.strip().endswith(
            "anyone who can reach this address can run inference. "
            "Set GF_API_KEY to require a bearer key."
        )
        assert warning.startswith(
            f"gf serve: warning: listening on 0.0.0.0:{port} without an API key"
        )
    finally:
        process.kill()
        process.wait(30)


def test_unservable_targets_fail_before_listening(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "generalforecast.serve", str(tmp_path / "missing")],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={k: v for k, v in os.environ.items() if k not in CLEAN},
        timeout=300,
    )
    assert result.returncode == 1
    assert (
        json.loads(result.stderr.strip().splitlines()[-1])["error"]["code"]
        == "ModelNotFound"
    )


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
    (lone / "gf.lock").write_bytes((root / "gf.lock").read_bytes())
    return lone / "gf.lock"


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


def _serving(target):
    port = _port()
    process = _start([str(target), "--port", str(port)], env={"GF_API_KEY": KEY})
    url = _event(process, "serving")["url"]
    _event(process, "ready")
    return url, process


def _stop(process):
    process.send_signal(signal.SIGTERM)
    _event(process, "stopped")
    assert process.wait(60) == 0


@pytest.mark.slow
def test_snapshot_package_binds_once_and_answers_predictions(package):
    url, process = _serving(package)
    try:
        status, body = call(url, "/v1alpha1/projects/local/models", key=KEY)
        assert status == 200
        [described] = body["result"]["models"]
        assert described["name"] == "churn" and described["kind"] == "package"
        assert described["features"] == ["x"] and described["context_rows"] == 4
        assert described["context"] == "snapshot"
        status, body = call(url, "/v1alpha1/projects/team-a/models/churn", key=KEY)
        assert status == 200 and body["result"] == described
        status, body = call(
            url,
            "/v1alpha1/projects/local/predictions",
            envelope(
                "predictions",
                predictions_payload(
                    queries=[{"id": "q", "x": 2.9}, {"id": "r", "x": 0.1}]
                ),
            ),
            key=KEY,
        )
        assert status == 200, body
        result = body["result"]
        assert [row["id"] for row in result["predictions"]] == ["q", "r"]
        assert [row["label"] for row in result["predictions"]] == [True, False]
        assert set(result["predictions"][0]["probabilities"]) == {"false", "true"}
        assert result["task"] == "classification"
        assert result["context"]["source"] == "snapshot"
        assert result["context"]["rows"] == 4
        lock = Package.load(package).lock
        assert lock is not None
        assert result["context"]["version"] == lock.data_digest
        status, body = call(
            url,
            "/v1alpha1/projects/local/models/churn/infer",
            envelope("churn", predictions_payload()),
            key=KEY,
        )
        assert status == 422 and body["error"]["code"] == "UnsupportedCapability"
        for change, code in [
            ({"target": "z"}, "ValidationError"),
            ({"features": ["x", "w"]}, "ValidationError"),
            ({"task": "regression"}, "UnsupportedTask"),
            ({"options": {"quantiles": [0.5]}}, "ValidationError"),
            ({"context": [{"x": 1.0, "y": True}]}, "ValidationError"),
            ({"queries": []}, "ValidationError"),
            ({"queries": [{"id": "q", "x": "high"}]}, "ValidationError"),
        ]:
            status, body = call(
                url,
                "/v1alpha1/projects/local/predictions",
                envelope("predictions", predictions_payload(**change)),
                key=KEY,
            )
            assert (status, body["error"]["code"]) == (422, code), change
    finally:
        _stop(process)


@pytest.mark.slow
def test_lone_lock_file_serves_request_context(lone_lock):
    url, process = _serving(lone_lock)
    try:
        context = [
            {"x": 0.0, "y": False},
            {"x": 1.0, "y": False},
            {"x": 3.0, "y": True},
        ]
        status, body = call(
            url,
            "/v1alpha1/projects/local/predictions",
            envelope(
                "predictions",
                predictions_payload(
                    model="scorer", context=context, queries=[{"x": 2.9}, {"x": 0.1}]
                ),
            ),
            key=KEY,
        )
        assert status == 200, body
        assert body["result"]["context"]["source"] == "request"
        assert body["result"]["context"]["rows"] == 3
        labels = [row["label"] for row in body["result"]["predictions"]]
        assert labels == [True, False]
        status, body = call(
            url,
            "/v1alpha1/projects/local/predictions",
            envelope("predictions", predictions_payload(model="scorer")),
            key=KEY,
        )
        assert status == 422
    finally:
        _stop(process)


def test_package_targets_are_checked_before_serving(tmp_path, package, lone_lock):
    entry = served(str(package), tmp_path, allow_downloads=True)
    assert entry["name"] == "churn" and entry["worker"] == "tabicl-v2-classifier"
    assert entry["package"]["context"] == "snapshot"
    snapshot = Path(entry["package"]["snapshot"])
    assert snapshot.parent == tmp_path and snapshot.read_bytes()
    entry = served(str(lone_lock), tmp_path, allow_downloads=True)
    assert entry["package"]["context"] == "request"
    assert entry["package"]["snapshot"] is None
    unlocked = tmp_path / "unlocked"
    unlocked.mkdir()
    (unlocked / "gf.yaml").write_text((package / "gf.yaml").read_text())
    (unlocked / "history.arrow").write_bytes(arrow_bytes(history()))
    with pytest.raises(PackageError) as caught:
        served(str(unlocked), tmp_path)
    assert caught.value.code == "LockRequired"
    lone_snapshot = tmp_path / "lone"
    lone_snapshot.mkdir()
    (lone_snapshot / "gf.lock").write_bytes((package / "gf.lock").read_bytes())
    with pytest.raises(PackageError) as caught:
        served(str(lone_snapshot / "gf.lock"), tmp_path)
    assert caught.value.code == "ContextRequired"


def _context_load(url, concurrency, requests, seed):
    """Requests per second for `requests` calls, each with a new 64-row context."""

    def one(index):
        rng = random.Random(seed + index)
        payload = infer_payload(
            features=["a", "b"],
            context=[
                {
                    "key": f"c{row}",
                    "features": {"a": rng.random(), "b": rng.random()},
                    "label": row % 2 == 0,
                }
                for row in range(64)
            ],
            queries=[{"key": "q", "features": {"a": 0.5, "b": 0.5}}],
        )
        status, body = call(
            url,
            "/v1alpha1/projects/local/models/tabicl-v2-classifier/infer",
            envelope("tabicl-v2-classifier", payload),
            key=KEY,
        )
        assert status == 200, body

    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, range(requests)))
    return requests / (time.perf_counter() - start)


@pytest.mark.slow
def test_new_contexts_hold_throughput_past_saturation(server):
    """Past saturation, a request per new context keeps the throughput it had.

    Every call fits a table of its own; doubling the callers from 256 to 512
    queues more fits per step, so throughput plateaus rather than falls. The
    first pass at 512 compiles the step sizes the measured ones reach.
    """
    url, _ = server
    _context_load(url, 512, 1024, seed=0)
    at_256 = _context_load(url, 256, 1024, seed=1 << 20)
    at_512 = _context_load(url, 512, 2048, seed=2 << 20)
    assert at_512 >= 0.8 * at_256, (at_256, at_512)
