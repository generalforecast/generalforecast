# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`gf.Predictor` runs local predictors on the lock and calls served and hosted ones correctly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import yaml

import generalforecast as gf
from generalforecast.predictor import resolve_reference
from tests.utils.engine import FakeEngine
from tests.utils.locks import directory, lock_of

MODEL = "general-forecast/tabicl-v2-classifier"


def rows() -> pa.Table:
    return pa.table(
        {
            "id": list(range(10)),
            "spend": [float(i) for i in range(10)],
            "zip": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "y": [i % 2 for i in range(10)],
        }
    )


class Recorder:
    """A client double that records the calls a predictor makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.contexts = self
        self.predictors = self

    def append(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append(("append", args, kwargs))
        return "appended"

    def label(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append(("label", args, kwargs))
        return "labeled"

    def get(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", args, kwargs))
        return {
            "name": "churn",
            "model": MODEL,
            "task": "classification",
            "id": "id",
            "target": "y",
        }

    def push(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("push", args, kwargs))


def test_live_context_writes_use_the_client_signatures(tmp_path):
    client = Recorder()
    predictor = gf.Predictor("churn", client=client)
    context = gf.predictor.Context("live", None, "events", None, predictor)
    assert (
        context.append([{"id": 1, "spend": 2.0}], idempotency_key="w-1") == "appended"
    )
    assert context.label([1, 2], [0, 1], idempotency_key="w-2") == "labeled"
    _, args, kwargs = client.calls[1]
    assert args[0] == "events" and kwargs == {"id": "id", "idempotency_key": "w-1"}
    _, args, kwargs = client.calls[2]
    assert (
        args == ("events", [{"id": 1, "y": 0}, {"id": 2, "y": 1}])
        or args[0] == "events"
    )
    assert kwargs["id"] == "id" and kwargs["idempotency_key"] == "w-2"


def test_hosted_facts_read_the_hosted_predictor_mapping():
    client = Recorder()
    predictor = gf.Predictor("churn", client=client)
    assert predictor._id_name() == "id"
    assert (predictor.model, predictor.task) == (MODEL, "classification")


def test_served_references_need_a_predictor_path():
    with pytest.raises(gf.InvalidInput) as caught:
        resolve_reference("http://localhost:8000")
    assert caught.value.hint == "use http://host:port/<predictor>"
    assert resolve_reference("http://localhost:8000/churn") == (
        "served",
        ("http://localhost:8000", "churn"),
    )
    with pytest.raises(gf.InvalidInput):
        resolve_reference("http://localhost:8000/a/b")


def test_served_calls_go_through_the_sdk_transport_with_project_and_key(monkeypatch):
    from generalforecast import transport

    sent: list[tuple[str, str, Any, dict[str, str], dict[str, str]]] = []

    def send(self, method, path, body, headers, **options):
        sent.append((method, path, body, headers, dict(self._headers)))
        payload = body["payload"] if body else {}
        result = (
            {
                "predictions": [
                    {"id": 7, "label": 1, "probabilities": {"0": 0.2, "1": 0.8}}
                ]
            }
            if payload
            else {"name": "churn", "id": "id"}
        )
        return {
            "api_version": "gf/v1",
            "project": headers["X-GF-Project"],
            "request_id": headers["X-GF-Request-ID"],
            "resource": {"kind": "predictor", "name": "churn"},
            "result": result,
        }

    monkeypatch.setattr(transport.Transport, "send", send)
    predictor = gf.Predictor(
        "http://127.0.0.1:9/churn", project="team-a", api_key="k-1"
    )
    frame = predictor.predict([{"id": 7, "spend": 1.0}])
    assert frame["label"].tolist() == [1]
    _, path, body, headers, defaults = next(item for item in sent if item[0] == "POST")
    assert path == "/v1/projects/team-a/predictors/churn/predict"
    assert body["project"] == "team-a" and headers["X-GF-Project"] == "team-a"
    assert defaults["Authorization"] == "Bearer k-1"


def test_push_names_the_environment_alone(tmp_path):
    root = directory(tmp_path / "churn", rows(), name="churn", id="id")
    client = Recorder()
    gf.Predictor(root, client=client).push("staging")
    [(_, _, kwargs)] = [call for call in client.calls if call[0] == "push"]
    assert kwargs == {"env": "staging"}


def test_save_keeps_the_whole_manifest_and_its_lock(tmp_path, monkeypatch):
    FakeEngine().install(monkeypatch)
    root = directory(
        tmp_path / "churn",
        rows(),
        name="churn",
        id="id",
        inputs={"types": {"zip": "category"}},
        selection={"seed": 4},
        resources={"accelerator": "L40S"},
        split={"holdout": {"fraction": 0.3, "seed": 1}},
    )
    original = gf.Predictor(root)
    saved = original.save(tmp_path / "copy")
    manifest = yaml.safe_load((saved / "gf.yaml").read_text())
    assert manifest["inputs"] == {"types": {"zip": "category"}}
    assert manifest["selection"] == {"seed": 4}
    assert manifest["resources"] == {"accelerator": "L40S"}
    assert manifest["context"] == {
        "snapshot": "context.parquet",
        "holdout": {"fraction": 0.3, "seed": 1},
    }
    copy = gf.Predictor(saved)
    assert lock_of(saved).data_digest == lock_of(root).data_digest
    assert copy.context.rows == original.context.rows
    queries = pa.table({"id": [100], "spend": [3.0], "zip": [2]})
    assert copy.predict(queries).equals(original.predict(queries))


def test_local_predictions_run_the_lock_through_the_hosted_executor(
    tmp_path, monkeypatch
):
    fake = FakeEngine().install(monkeypatch)
    root = directory(tmp_path / "churn", rows(), name="churn", id="id")
    predictor = gf.Predictor(root)
    frame = predictor.predict([{"id": 100, "spend": 3.0, "zip": 2, "extra": "x"}])
    assert list(frame.columns) == ["id", "label", "p_0", "p_1"]
    assert fake.surface == "local"
    with pytest.raises(gf.InvalidInput):
        predictor.predict([{"id": 1, "spend": 1.0, "zip": 1}], context=rows())


def test_ad_hoc_forecasts_take_future_covariates(monkeypatch):
    from generalforecast.models.catalog import CHRONOS2_QUANTILES

    FakeEngine(CHRONOS2_QUANTILES).install(monkeypatch)
    context = pa.table(
        {
            "t": list(range(8)),
            "units": [float(i) for i in range(8)],
            "promo": [float(i % 2) for i in range(8)],
        }
    )
    predictor = gf.Predictor(
        "general-forecast/chronos-2",
        target="units",
        time="t",
        horizon=2,
        future=["promo"],
        context=context,
    )
    frame = predictor.predict(pa.table({"t": [8, 9], "promo": [0.0, 1.0]}))
    assert list(frame.columns[:2]) == ["t", "value"] and len(frame) == 2
    lock = predictor.explain()
    assert dict(zip(lock.input_names, lock.input_roles, strict=True)) == {
        "promo": "future"
    }


def test_explain_of_a_local_predictor_is_its_lock_resolution(tmp_path):
    root = directory(tmp_path / "churn", rows(), name="churn", id="id")
    resolution = gf.Predictor(root).explain()
    assert resolution.input_roles == ("input", "input")
    assert resolution.selection.strategy == "exact"
    assert Path(root / "gf.lock").is_file()


def test_a_served_predictor_never_receives_an_ambient_hosted_key(monkeypatch):
    import http.server
    import threading

    seen: list[str | None] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.headers.get("Authorization"))
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"code": "InternalError", "message": "x"}}')

        do_GET = do_POST

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.delenv("GF_URL", raising=False)
    monkeypatch.delenv("GF_TOKEN", raising=False)
    monkeypatch.setenv("GF_API_KEY", "gf_hosted_secret_key_0123456789abcdef")

    def call(**options):
        with pytest.raises(gf.Error):
            gf.Predictor(f"{url}/churn", **options).predict([{"x": 1}])
        assert len(set(seen)) == 1, seen
        sent = seen[0]
        seen.clear()
        return sent

    try:
        # A hosted key in the environment belongs to the hosted service.
        assert call() is None
        monkeypatch.setenv("GF_TOKEN", "gf_token_secret_0123456789abcdefgh")
        assert call() is None
        # Given for this server, or GF_URL naming it, the key goes.
        assert call(api_key="gf_serve_key_0123456789abcdefgh") == (
            "Bearer gf_serve_key_0123456789abcdefgh"
        )
        monkeypatch.setenv("GF_URL", url)
        assert call() == "Bearer gf_hosted_secret_key_0123456789abcdef"
    finally:
        server.shutdown()
    # Plain HTTP goes to loopback alone.
    with pytest.raises(gf.Error):
        gf.Predictor("http://example.com/churn").predict([{"x": 1}])
