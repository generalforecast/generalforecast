# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One table of requests, answered alike by every surface that runs the engine.

Each case runs in process through `CatalogService` on a
`LocalEngine`, and over HTTP through the SDK `Client` against the real
gf-server binary that `gf serve` configures. Tabular predictors run through
`PredictorExecutor` and through `gf serve <lock>`. Every surface must give the
same result envelope, with floats equal within tolerance, or the same error
code, message and status. The models are the tiny random ones the model
tests use: gf-server's worker loads them by spec, and the in-process engine
loads them under the same catalog references.
"""

import json
import math
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import yaml

from generalforecast.client import Client
from generalforecast.engine import LocalEngine
from generalforecast.predictors import load, resolve, write_lock
from generalforecast.predictors.runtime import PredictorExecutor
from generalforecast.serve import listener
from generalforecast.serve import models as serve_models
from generalforecast.service.boundary import Scope
from generalforecast.service.contracts import ServiceError
from generalforecast.tabular import arrow_bytes
from tests.utils.server import build

ROOT = Path(__file__).resolve().parents[2]
SCOPE = Scope("local", "r-conformance")
# Each served catalog model, and the tiny model both engines load for it.
TINY = {
    "general-forecast/tabicl-v2-classifier": "tests/tabicl-classifier",
    "general-forecast/tabicl-v2-regressor": "tests/tabicl-regressor",
    "general-forecast/timestral-28m": "tests/timestral-forecaster",
    "general-forecast/chronos-2": "tests/chronos2-catalog-forecaster",
}
RELATIVE, ABSOLUTE = 1e-5, 1e-6

Outcome = tuple[str, Any]


@pytest.fixture(scope="module")
def engine() -> Iterator[None]:
    """The process engine every in-process surface reaches, on the tiny models.

    One `LocalEngine` loads each catalog model's tiny spec in its workers, as
    gf-server's workers load it, and every surface's gates share its shards.
    """
    tiny = LocalEngine(models=TINY)

    def local(surface: Any = "local", allow_downloads: bool = False) -> LocalEngine:
        return tiny.gated(surface, allow_downloads)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("generalforecast.engine.local", local)
        yield
    tiny.close()


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Served:
    """gf-server as `gf serve` configures it, for one target, on the tiny models."""

    def __init__(
        self, binary: Path, target: str, *, metrics: str | None = None
    ) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="gf-conformance-"))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(serve_models, "checkpoint", lambda *_, **__: None)
            model = serve_models.served(target, self.directory)
        self.url = f"http://127.0.0.1:{_port()}"
        (self.directory / "models.json").write_text(json.dumps([model]))
        config = self.directory / "generalforecast.toml"
        config.write_text(
            listener.configuration(
                self.directory,
                self.url.removeprefix("http://"),
                120.0,
                [model | {"reference": TINY[model["worker"]]}],
                metrics=metrics,
            )
        )
        config.chmod(0o600)
        environment = dict(os.environ, GF_LOG_KEY="0" * 64, PYTHONPATH=str(ROOT))
        environment.pop(listener.API_KEY_ENV, None)
        self.process = subprocess.Popen(
            [str(binary), "run", "--config", str(config)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._wait("ready")
        self.client = Client(base_url=self.url)

    def _wait(self, expected: str) -> None:
        assert self.process.stdout is not None
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                assert self.process.stderr is not None
                raise AssertionError(self.process.stderr.read())
            if json.loads(line).get("event") == expected:
                return
        raise AssertionError(f"gf-server never reported {expected}")

    def post(self, path: str, body: bytes) -> tuple[int, dict]:
        """A raw POST, for bodies the SDK never sends."""
        request = urllib.request.Request(
            self.url + path, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(60)


@pytest.fixture(scope="module")
def servers(engine) -> Iterator[dict[str, Served]]:
    """One gf-server per served catalog model, as `gf serve MODEL` runs it."""
    binary = build()
    started: dict[str, Served] = {}
    try:
        for name in TINY:
            started[name] = Served(binary, name)
        yield started
    finally:
        for server in started.values():
            server.stop()


def _outcome(call: Callable[[], Any]) -> Outcome:
    try:
        return "result", call()
    except ServiceError as error:
        return "error", (error.code, error.message, error.status)


def _same(left: Any, right: Any, at: str = "") -> None:
    """Equal JSON values, with floats equal within tolerance."""
    if isinstance(left, float) or isinstance(right, float):
        assert isinstance(left, int | float) and isinstance(right, int | float), at
        assert math.isclose(left, right, rel_tol=RELATIVE, abs_tol=ABSOLUTE), (
            at,
            left,
            right,
        )
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys(), (at, left.keys(), right.keys())
        for key in left:
            _same(left[key], right[key], f"{at}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), at
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _same(a, b, f"{at}[{index}]")
    else:
        assert type(left) is type(right) and left == right, (at, left, right)


def _predicted(server: Served, predictor: str, queries: pa.Table) -> list[dict]:
    """The rows a served predictor answers for `queries` on its predict route."""
    body = {
        "api_version": "gf/v1",
        "project": "local",
        "request_id": f"r-{predictor}",
        "resource": {"kind": "predictor", "name": predictor},
        "payload": {"queries": queries.to_pylist()},
    }
    status, envelope = server.post(
        f"/v1/projects/local/predictors/{predictor}/predict",
        json.dumps(body, default=lambda value: value.isoformat()).encode(),
    )
    assert status == 200, envelope
    return envelope["result"]["predictions"]


def _columns(row: int) -> dict[str, Any]:
    """Numeric, string, categorical and datetime cells, as a predictor lock binds."""
    return {
        "x": float(row) / 2,
        "plan": ("basic", "pro", "team")[row % 3],
        "region": ("north", "south")[row % 2],
        "joined": f"2024-0{1 + row % 9}-1{row % 10}T08:00:00",
    }


def history(task: str) -> pa.Table:
    """A snapshot context with every column kind a predictor lock binds."""
    rows = [_columns(row) for row in range(9)]
    return pa.table(
        {
            "id": [f"r{row}" for row in range(9)],
            "x": [row["x"] for row in rows],
            "plan": pa.array([row["plan"] for row in rows]).dictionary_encode(),
            "region": [row["region"] for row in rows],
            "joined": pa.array([row["joined"] for row in rows], pa.string()).cast(
                pa.timestamp("s")
            ),
            "y": (
                [row % 2 == 0 for row in range(9)]
                if task == "classification"
                else [float(row) - 3 for row in range(9)]
            ),
        }
    )


@pytest.mark.parametrize(
    ("model", "task"),
    [
        ("general-forecast/tabicl-v2-classifier", "classification"),
        ("general-forecast/tabicl-v2-regressor", "regression"),
    ],
)
def test_packages_answer_alike(tmp_path, engine, model, task):
    data = history(task)
    served_root = tmp_path / "served"
    served_root.mkdir()
    (served_root / "gf.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "scorer",
                "model": model,
                "context": {"snapshot": "history.arrow"},
                "target": "y",
                "id": "id",
            }
        )
    )
    (served_root / "history.arrow").write_bytes(arrow_bytes(data))
    write_lock(resolve(load(served_root), served_root), served_root)
    queries = data.drop(["y"]).slice(2, 4)
    expected = _hosted(served_root, data).predict(queries)
    server = Served(build(), str(served_root))
    try:
        predictions = _predicted(server, "scorer", queries)
    finally:
        server.stop()
    _same(predictions, expected.to_pylist(), "predictor")


def _hosted(root, data):
    """The predictor hosted serving prepares for the directory's lock and snapshot."""
    from generalforecast.predictors.runtime.compile import compile_lock
    from generalforecast.predictors.runtime.context import served_context
    from tests.utils.locks import lock_of

    lock = lock_of(root)
    rows = served_context(lock, data)
    return PredictorExecutor().prepare(compile_lock(lock, rows), rows)


def series_history() -> pa.Table:
    """A snapshot of two series with a past input and a known-future one."""
    return pa.table(
        {
            "store": [store for store in ("north", "south") for _ in range(12)],
            "day": [step for _ in range(2) for step in range(12)],
            "sales": [
                float(10 + step + 3 * index + (step % 3))
                for index in range(2)
                for step in range(12)
            ],
            "footfall": [float((step * 7) % 5) for _ in range(2) for step in range(12)],
            "promo": [float(step % 2) for _ in range(2) for step in range(12)],
        }
    )


def test_package_forecasts_answer_alike(tmp_path, engine):
    data = series_history()
    served_root = tmp_path / "served"
    served_root.mkdir()
    (served_root / "gf.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "planner",
                "model": "general-forecast/chronos-2",
                "context": {"snapshot": "history.arrow"},
                "target": "sales",
                "time": "day",
                "series": ["store"],
                "horizon": 2,
                "quantiles": [0.1, 0.5, 0.9],
                "inputs": {"past": ["footfall"], "future": ["promo"]},
            }
        )
    )
    (served_root / "history.arrow").write_bytes(arrow_bytes(data))
    write_lock(resolve(load(served_root), served_root), served_root)
    queries = pa.table(
        {
            "store": ["south", "north", "north", "south"],
            "day": [13, 12, 13, 12],
            "promo": [1.0, 0.0, 1.0, 0.0],
        }
    )
    expected = _hosted(served_root, data).predict(queries)
    server = Served(build(), str(served_root))
    try:
        predictions = _predicted(server, "planner", queries)
    finally:
        server.stop()
    _same(predictions, expected.to_pylist(), "predictor")
