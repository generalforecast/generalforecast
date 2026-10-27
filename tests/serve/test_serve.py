# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`gf serve`, which runs gf-server's Models API over a catalog model or a locked predictor."""

import json
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow as pa
import pytest

from generalforecast.predictors.validation import PredictorError
from generalforecast.serve.models import served
from generalforecast.service.boundary import Scope
from generalforecast.service.catalog import CatalogService
from generalforecast.tabular import arrow_bytes
from tests.serve.processes import assert_exited as _assert_exited
from tests.serve.processes import children as _children
from tests.utils.locks import directory, lock_of

ROOT = Path(__file__).resolve().parents[2]
MODEL = "general-forecast/tabicl-v2-classifier"
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


def envelope(resource, payload, *, project="local", request_id="r-test", kind="model"):
    return {
        "api_version": "gf/v1",
        "project": project,
        "request_id": request_id,
        "resource": {"kind": kind, "name": resource},
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


def _start(arguments, *, env=None, own_group=False):
    from tests.utils.server import build

    binary = build()
    environment = {k: v for k, v in os.environ.items() if k not in CLEAN}
    environment["PATH"] = f"{binary.parent}{os.pathsep}{environment['PATH']}"
    environment |= env or {}
    # `own_group` puts the server and its workers in their own process
    # group, so a test can send them a terminal's Ctrl-C.
    return subprocess.Popen(
        [sys.executable, "-m", "generalforecast.serve", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=ROOT,
        start_new_session=own_group,
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
    process = _start([MODEL, "--port", str(port)], env={"GF_API_KEY": KEY})
    serving = _event(process, "serving")
    assert serving == {
        "event": "serving",
        "url": f"http://127.0.0.1:{port}",
        "models": ["general-forecast/tabicl-v2-classifier"],
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
    assert call(url, "/livez") == (200, {"status": "ok"})
    assert call(url, "/readyz") == (200, {"status": "ok"})
    assert call(url, "/healthz") == (200, {"status": "ok"})
    status, body = call(url, "/v1/projects/local/models", key=KEY)
    assert status == 200
    [described] = body["result"]["models"]
    extras = {"model", "limits", "warm_shapes"}
    assert {k: v for k, v in described.items() if k not in extras} == {
        k: v
        for k, v in hosted.get(scope, "general-forecast/tabicl-v2-classifier").items()
        if k not in extras
    }
    assert described["model"].startswith(f"{MODEL}@sha256:")
    assert set(described["limits"]) == {"context_rows", "queries", "inputs"}
    assert [shape["context_rows"] for shape in described["warm_shapes"]] == [
        100,
        1_000,
        10_000,
    ]


@pytest.mark.slow
def test_a_dotted_catalog_id_reaches_gf_serves_models_route(server):
    from generalforecast.client import Client
    from generalforecast.errors import NotFound

    url, _ = server
    # A dotted name is a model name, not a malformed one: this server does
    # not serve tabpfn-3.5, so it is NotFound, never "invalid name".
    status, body = call(url, "/v1/projects/local/models/tabpfn-3.5-classifier", key=KEY)
    assert status == 404 and body["error"]["code"] == "NotFound", body
    with Client(base_url=url, api_key=KEY) as client:
        with pytest.raises(NotFound):
            client.models.get("general-forecast/tabpfn-3.5-classifier")
        assert client.models.get(MODEL)["name"] == MODEL


@pytest.mark.slow
def test_serve_requires_its_key_and_keeps_the_api_errors(server):
    url, _ = server
    status, body = call(url, "/v1/projects/local/models")
    assert (status, body["error"]["message"]) == (401, "application bearer required")
    # An error keeps the envelope, with the request id the caller sent.
    status, body = call(
        url,
        "/v1/projects/local/models/other",
        key=KEY,
        headers={"X-GF-Request-Id": "r-test"},
    )
    assert (status, body["error"]["code"]) == (404, "NotFound")
    assert body["request_id"] == "r-test"


def adhoc_payload(**changes):
    return {
        "model": MODEL,
        "target": "y",
        "inputs": ["x"],
        "id": "id",
        "context": [{"x": 0.0, "y": False}, {"x": 3.0, "y": True}],
        "queries": [{"id": "a", "x": 1.5}],
    } | changes


def predict(url, payload):
    # The ad hoc route's resource is the predictor `ad-hoc`, as the SDK names it.
    return call(
        url,
        "/v1/projects/local/predict",
        envelope("ad-hoc", payload, kind="predictor"),
        key=KEY,
    )


@pytest.mark.slow
def test_the_ad_hoc_route_predicts_and_names_what_it_refuses(server):
    url, _ = server
    status, body = predict(url, adhoc_payload())
    assert status == 200, body
    [row] = body["result"]["predictions"]
    assert row["id"] == "a" and type(row["label"]) is bool
    assert set(row["probabilities"]) == {"false", "true"}
    assert body["result"]["context"]["rows"] == 2
    assert body["result"]["context"]["source"] == "request"
    assert body["result"]["model"].startswith(f"{MODEL}@sha256:")
    status, body = predict(url, adhoc_payload(model="tabicl-v2-classifier"))
    assert status == 422, body
    assert body["error"]["param"] == "model"
    assert body["error"]["hint"] == f"use {MODEL}"
    for field, value in [
        ("task", "classification"),
        ("options", {}),
        ("features", ["x"]),
        ("selection", {"core": 1, "neighbors": 1}),
    ]:
        status, body = predict(url, adhoc_payload(**{field: value}))
        assert (status, body["error"]["param"]) == (422, field), body
    status, body = predict(
        url, adhoc_payload(context=[{"x": "low", "y": False}, {"x": 3.0, "y": True}])
    )
    assert status == 422
    assert body["error"]["message"] == (
        "column x has text values; this route takes numeric inputs"
    )
    assert body["error"]["param"] == "inputs.x"
    status, body = predict(url, adhoc_payload(queries=[{"x": 1.5}] * 10_001))
    assert status == 422
    assert body["error"]["code"] == "CapacityExceeded"
    assert body["error"]["actual"] == 10_001 and body["error"]["limit"] <= 10_000
    assert "general-forecast/tabicl-v2-classifier" in body["error"]["message"]
    status, body = call(
        url,
        "/v1/projects/local/predictions",
        envelope("predictions", adhoc_payload()),
        key=KEY,
    )
    assert status == 404


@pytest.mark.slow
def test_ten_thousand_context_rows_are_answered(server):
    url, _ = server
    rng = random.Random(0)
    context = [
        {"a": rng.random(), "b": rng.random(), "y": rng.randint(0, 1)}
        for _ in range(10_000)
    ]
    payload = adhoc_payload(
        inputs=["a", "b"], id=None, context=context, queries=[{"a": 0.5, "b": 0.5}]
    )
    del payload["id"]
    started = time.perf_counter()
    status, body = predict(url, payload)
    assert status == 200, body
    assert time.perf_counter() - started < 120
    assert body["result"]["context"]["rows"] == 10_000
    [row] = body["result"]["predictions"]
    assert set(row["probabilities"]) == {"0", "1"}


@pytest.mark.slow
def test_ctrl_c_stops_the_server_and_its_workers_quietly():
    port = _port()
    process = _start(
        [MODEL, "--port", str(port)], env={"GF_API_KEY": KEY}, own_group=True
    )
    url = _event(process, "serving")["url"]
    _event(process, "ready")
    workers = _children(process.pid)
    assert workers
    os.killpg(process.pid, signal.SIGINT)
    assert _event(process, "stopped")["url"] == url
    assert process.wait(60) == 0
    assert process.stdout.read() == ""
    stderr = process.stderr.read()
    assert "Traceback" not in stderr and "KeyboardInterrupt" not in stderr, stderr
    _assert_exited(workers)
    # Each server removed its workers' socket directories as it stopped.
    left = [
        entry.name
        for entry in Path(tempfile.gettempdir()).iterdir()
        if any(entry.name.startswith(f"gf-{pid}-") for pid in workers)
    ]
    assert left == [], left


@pytest.mark.slow
def test_public_bind_without_a_key_warns():
    port = _port()
    process = _start([MODEL, "--host", "0.0.0.0", "--port", str(port)])
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
    # A server that is never started: the target fails first.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gf-server").write_text("#!/bin/sh\nexit 99\n")
    (bin_dir / "gf-server").chmod(0o755)
    environment = {k: v for k, v in os.environ.items() if k not in CLEAN}
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    result = subprocess.run(
        [sys.executable, "-m", "generalforecast.serve", str(tmp_path / "missing")],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
        timeout=300,
    )
    assert result.returncode == 1
    assert (
        json.loads(result.stderr.strip().splitlines()[-1])["error"]["code"]
        == "NotFound"
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
def predictor_dir(tmp_path):
    return directory(tmp_path / "churn", history(), name="churn", id="id")


@pytest.fixture
def lone_lock(tmp_path):
    lone = directory(
        tmp_path / "elsewhere", history(), context="request", name="scorer"
    )
    (lone / "gf.yaml").unlink()
    (lone / "history.arrow").unlink()
    return lone / "gf.lock"


def predictor_payload(**changes):
    return {"queries": [{"id": "q", "x": 1.5}]} | changes


def predictor_predict(url, name, payload, project="local"):
    return call(
        url,
        f"/v1/projects/{project}/predictors/{name}/predict",
        envelope(name, payload, project=project, kind="predictor"),
        key=KEY,
    )


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
def test_snapshot_predictor_binds_once_and_answers_predictions(predictor_dir):
    url, process = _serving(predictor_dir)
    try:
        status, body = call(url, "/v1/projects/local/models", key=KEY)
        assert status == 200 and body["result"] == {"models": []}
        status, body = call(url, "/v1/projects/local/predictors", key=KEY)
        assert status == 200
        [described] = body["result"]["predictors"]
        assert described["name"] == "churn"
        assert described["inputs"] == [{"name": "x", "type": "number", "role": "input"}]
        assert described["environments"] == []
        assert described["model"].startswith(f"{MODEL}@sha256:")
        digest = lock_of(predictor_dir).data_digest
        assert described["context"] == {
            "kind": "snapshot",
            "rows": 4,
            "digest": digest,
        }
        assert described["lock_digest"].startswith("sha256:")
        status, body = call(url, "/v1/projects/team-a/predictors/churn", key=KEY)
        assert status == 200 and body["result"] == described
        assert body["resource"] == {"kind": "predictor", "name": "churn"}
        status, body = predictor_predict(
            url,
            "churn",
            predictor_payload(queries=[{"id": "q", "x": 2.9}, {"id": "r", "x": 0.1}]),
        )
        assert status == 200, body
        result = body["result"]
        assert [row["id"] for row in result["predictions"]] == ["q", "r"]
        assert [row["label"] for row in result["predictions"]] == [True, False]
        assert set(result["predictions"][0]["probabilities"]) == {"false", "true"}
        assert result["task"] == "classification"
        assert result["context"] == {
            "source": "snapshot",
            "name": None,
            "version": None,
            "digest": digest,
            "rows": 4,
            "as_of": None,
        }
        assert result["model"].startswith(f"{MODEL}@sha256:")
        # The retired infer route is no route at all.
        status, body = call(
            url,
            "/v1/projects/local/models/churn/infer",
            envelope("churn", predictor_payload()),
            key=KEY,
        )
        assert (status, body["error"]["code"]) == (404, "NotFound")
        for change, param in [
            ({"target": "z"}, "target"),
            ({"inputs": ["x", "w"]}, "inputs"),
            ({"features": ["x", "w"]}, "features"),
            ({"task": "regression"}, "task"),
            ({"model": MODEL}, "model"),
            ({"options": {"quantiles": [0.5]}}, "options"),
            ({"context": [{"x": 1.0, "y": True}]}, "context"),
            ({"queries": []}, None),
            ({"queries": [{"id": "q", "x": "high"}]}, None),
        ]:
            status, body = predictor_predict(url, "churn", predictor_payload(**change))
            assert (status, body["error"]["code"]) == (422, "ValidationError"), change
            assert body["error"].get("param") == param, change
            if param in ("target", "inputs", "model", "task"):
                assert (
                    body["error"]["message"] == f"{param} is declared by the predictor"
                )
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
        status, body = predictor_predict(
            url,
            "scorer",
            predictor_payload(context=context, queries=[{"x": 2.9}, {"x": 0.1}]),
        )
        assert status == 200, body
        assert body["result"]["context"]["source"] == "request"
        assert body["result"]["context"]["rows"] == 3
        labels = [row["label"] for row in body["result"]["predictions"]]
        assert labels == [True, False]
        status, body = predictor_predict(url, "scorer", predictor_payload())
        assert status == 422
        assert body["error"]["param"] == "context"
        status, body = call(url, "/v1/projects/local/predictors/scorer", key=KEY)
        assert status == 200
        assert body["result"]["context"] == {"kind": "request"}
    finally:
        _stop(process)


def test_predictor_targets_are_checked_before_serving(
    tmp_path, predictor_dir, lone_lock
):
    entry = served(str(predictor_dir), tmp_path, allow_downloads=True)
    assert (
        entry["name"] == "churn"
        and entry["worker"] == "general-forecast/tabicl-v2-classifier"
    )
    assert entry["predictor"]["context"] == "snapshot"
    snapshot = Path(entry["predictor"]["snapshot"])
    assert snapshot.parent == tmp_path and snapshot.read_bytes()
    entry = served(str(lone_lock), tmp_path, allow_downloads=True)
    assert entry["predictor"]["context"] == "request"
    assert entry["predictor"]["snapshot"] is None
    unlocked = tmp_path / "unlocked"
    unlocked.mkdir()
    (unlocked / "gf.yaml").write_text((predictor_dir / "gf.yaml").read_text())
    (unlocked / "history.arrow").write_bytes(arrow_bytes(history()))
    with pytest.raises(PredictorError) as caught:
        served(str(unlocked), tmp_path)
    assert caught.value.code == "LockRequired"
    with pytest.raises(PredictorError) as caught:
        served("tabicl-v2-classifier", tmp_path)
    assert caught.value.code == "ValidationError"
    assert str(caught.value).endswith(f"use {MODEL}")
    entry = served(MODEL, tmp_path, allow_downloads=True)
    assert entry["resolved"].startswith(f"{MODEL}@sha256:")
    assert entry["description"]["model"] == entry["resolved"]
    lone_snapshot = tmp_path / "lone"
    lone_snapshot.mkdir()
    (lone_snapshot / "gf.lock").write_bytes((predictor_dir / "gf.lock").read_bytes())
    with pytest.raises(PredictorError) as caught:
        served(str(lone_snapshot / "gf.lock"), tmp_path)
    assert caught.value.code == "ContextRequired"


def _read_snapshot(entry):
    with pa.ipc.open_stream(
        Path(entry["predictor"]["snapshot"]).read_bytes()
    ) as reader:
        return reader.read_all()


def test_snapshot_context_follows_the_manifest_split(tmp_path):
    from generalforecast.predictors import load, split

    rows = pa.table(
        {
            "id": list(range(10)),
            "x": [float(i) for i in range(10)],
            "y": [i % 2 == 0 for i in range(10)],
        }
    )
    root = directory(
        tmp_path / "held",
        rows,
        name="held",
        id="id",
        split={"holdout": {"fraction": 0.2, "seed": 0}},
    )
    entry = served(str(root), tmp_path, allow_downloads=True)
    snapshot = _read_snapshot(entry)
    expected, _ = split(rows, load(root))
    assert snapshot.num_rows == 8
    assert snapshot["id"].to_pylist() == expected["id"].to_pylist()
    [id_column] = [
        c for c in entry["predictor"]["schema"]["columns"] if c["role"] == "id"
    ]
    assert id_column["type"] == "int64"


def test_blank_and_whole_number_targets_count_like_gf_check(tmp_path):
    rows = pa.table(
        {
            "id": ["a", "b", "c", "d"],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": ["no", "yes", "", ""],
        }
    )
    root = directory(tmp_path / "blank", rows, name="blank", id="id")
    entry = served(str(root), tmp_path, allow_downloads=True)
    assert _read_snapshot(entry)["y"].to_pylist() == ["no", "yes"]
    rows = pa.table(
        {
            "id": ["a", "b", "c", "d", "e"],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 1.0, None, 0.0, 1.0],
        }
    )
    floaty = directory(tmp_path / "floaty", rows, name="floaty", id="id")
    snapshot = _read_snapshot(served(str(floaty), tmp_path, allow_downloads=True))
    assert snapshot["y"].type == pa.int64() and snapshot["y"].to_pylist() == [
        0,
        1,
        0,
        1,
    ]


def test_a_stale_lock_or_changed_snapshot_is_not_served(tmp_path, predictor_dir):
    manifest = predictor_dir / "gf.yaml"
    manifest.write_text(manifest.read_text() + "selection: {seed: 4}\n")
    with pytest.raises(PredictorError) as caught:
        served(str(predictor_dir), tmp_path, allow_downloads=True)
    assert caught.value.code == "LockRequired"
    fresh = directory(tmp_path / "fresh", history(), name="fresh", id="id")
    (fresh / "history.arrow").write_bytes(
        arrow_bytes(history().set_column(1, "x", pa.array([9.0, 1.0, 2.0, 3.0])))
    )
    with pytest.raises(PredictorError) as caught:
        served(str(fresh), tmp_path, allow_downloads=True)
    assert caught.value.code == "SnapshotMismatch"


def test_verbose_asks_gf_server_for_its_progress_lines(monkeypatch):
    import argparse

    from generalforecast.serve.listener import _environment

    def flags(**given):
        return argparse.Namespace(
            **({"verbose": False, "quiet": False, "json": False} | given)
        )

    monkeypatch.delenv("GF_LOG", raising=False)
    monkeypatch.delenv("GF_EVENTS", raising=False)
    quiet = _environment(flags(), GF_LOG_KEY="k")
    assert "GF_LOG" not in quiet and "GF_EVENTS" not in quiet
    assert quiet["GF_LOG_KEY"] == "k"
    assert _environment(flags(verbose=True))["GF_LOG"] == "info"
    assert _environment(flags(quiet=True))["GF_LOG"] == "error"
    assert _environment(flags(json=True))["GF_EVENTS"] == "json"
    monkeypatch.setenv("GF_LOG", "debug")
    assert _environment(flags(verbose=True))["GF_LOG"] == "debug"
