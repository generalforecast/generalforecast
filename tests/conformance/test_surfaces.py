"""One table of requests, answered alike by every surface that runs the engine.

Each case runs in process through `CatalogService` and `LocalModels` on a
`LocalEngine`, and over HTTP through the SDK `Client` against the real
gf-server binary that `gf serve` configures. Tabular packages run through
`PackageExecutor` and through `gf serve <lock>`. Every surface must give the
same result envelope, with floats equal within tolerance, or the same error
code, message and status. The models are the tiny random ones the model
tests use: gf-server's worker loads them by spec, and the in-process engine
loads them under the same catalog references.
"""

import contextlib
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
from typing import Any, cast

import pyarrow as pa
import pytest
import yaml

from generalforecast import _core
from generalforecast.client import Client
from generalforecast.deployments.boundary import Scope
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.contracts import ServiceError
from generalforecast.deployments.interfaces import InferenceRequest
from generalforecast.deployments.package_runtime import PackageExecutor
from generalforecast.engine import LocalEngine
from generalforecast.local_models import LocalModels
from generalforecast.models.catalog import parse
from generalforecast.packages import Package
from generalforecast.serve import listener
from generalforecast.serve import models as serve_models
from generalforecast.tabular import arrow_bytes
from tests.utils.server import build

ROOT = Path(__file__).resolve().parents[2]
SCOPE = Scope("local", "r-conformance")
# Each served catalog model, and the tiny model both engines load for it.
TINY = {
    "tabicl-v2-classifier": "tests.models.tabicl.tiny:classifier",
    "tabicl-v2-regressor": "tests.models.tabicl.tiny:regressor",
    "timestral-28m": "tests.models.timestral.tiny:forecaster",
    "chronos-2": "tests.models.chronos2.tiny:catalog_forecaster",
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

    def __init__(self, binary: Path, target: str) -> None:
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


def _features(row: int) -> dict[str, Any]:
    """Numeric features, the only kind the Models API's JSON rows carry."""
    return {"x": float(row) / 2, "count": row % 4, "ratio": 1 / (row + 1)}


def _columns(row: int) -> dict[str, Any]:
    """Numeric, string, categorical and datetime cells, as a package lock binds."""
    return {
        "x": float(row) / 2,
        "plan": ("basic", "pro", "team")[row % 3],
        "region": ("north", "south")[row % 2],
        "joined": f"2024-0{1 + row % 9}-1{row % 10}T08:00:00",
    }


def tabular(model: str, task: str, **changes: Any) -> dict[str, Any]:
    """Labeled rows with numeric, string, categorical and datetime features."""
    labels = (
        [("churn", "stay", "pause")[row % 3] for row in range(9)]
        if task == "classification"
        else [float(row) * 1.5 - 2 for row in range(9)]
    )
    return {
        "model": model,
        "task": task,
        "target": "y",
        "features": ["x", "count", "ratio"],
        "context": [
            {"key": f"c{row}", "features": _features(row), "label": labels[row]}
            for row in range(9)
        ],
        "queries": [
            {"key": f"q{row}", "features": _features(row + 4)} for row in range(3)
        ],
        "options": {},
    } | changes


def forecast(**changes: Any) -> dict[str, Any]:
    """Two series of eight periods, with a past and a known-future covariate."""
    series = changes.pop("series", {})
    return {
        "model": "timestral-28m",
        "task": "forecast",
        "target": "sales",
        "features": [],
        "context": [
            {
                "key": f"{name}-{period}",
                "series": name,
                "time": period,
                "value": base + period * (1.5 if name == "a" else -0.5),
                "covariates": {"promo": period % 2, "past": period / 10},
            }
            for name, base in (("a", 10.0), ("b", 30.0))
            for period in range(8)
        ],
        "queries": [
            {
                "key": "q-a",
                "series": "a",
                "origin": 7,
                "group": ["a"],
                "future_covariates": {},
            },
        ],
        "options": {
            "series": {
                "step": 1,
                "horizon": 3,
                "known_covariates": [],
                "quantiles": [0.5],
            }
            | series
        },
    } | changes


GROUPED = [
    {
        "key": f"q-{name}",
        "series": name,
        "origin": 7,
        "group": ["a", "b"],
        "future_covariates": {"promo": [0, 1, 0]},
    }
    for name in ("a", "b")
]

# Each case's request, and the error code every surface refuses it with, if any.
EXPECTED = {
    "classification": "result",
    "classification-boolean": "result",
    "regression": "result",
    "regression-quantiles": "result",
    "forecast-single": "result",
    "forecast-grouped": "result",
    "forecast-quantiles": "result",
    "forecast-chronos-2": "result",
    "unknown-model": "ModelNotFound",
    "unknown-feature": "ValidationError",
    "too-few-classes": "UnsupportedClassCount",
    "quantile-bounds": "ValidationError",
    "horizon-bounds": "ValidationError",
    "variant-policy": "CapacityExceeded",
    "surface-gate": "UnsupportedModel",
}

CASES: dict[str, dict[str, Any]] = {
    "classification": tabular("tabicl-classifier", "classification"),
    "classification-boolean": tabular(
        "tabicl-classifier",
        "classification",
        context=[
            row | {"label": index % 2 == 0}
            for index, row in enumerate(
                tabular("tabicl-classifier", "classification")["context"]
            )
        ],
    ),
    "regression": tabular("tabicl-regressor", "regression"),
    "regression-quantiles": tabular(
        "tabicl-regressor", "regression", options={"quantiles": [0.1, 0.5, 0.9]}
    ),
    "forecast-single": forecast(),
    "forecast-grouped": forecast(
        queries=GROUPED, series={"known_covariates": ["promo"]}
    ),
    "forecast-quantiles": forecast(
        queries=GROUPED,
        series={"known_covariates": ["promo"], "quantiles": [0.1, 0.5, 0.9]},
    ),
    "forecast-chronos-2": forecast(
        model="chronos-2",
        queries=GROUPED,
        series={"known_covariates": ["promo"], "quantiles": [0.01, 0.5, 0.99]},
    ),
    "unknown-model": tabular("no-such-model", "classification"),
    "unknown-feature": tabular(
        "tabicl-classifier", "classification", features=["x", "absent"]
    ),
    "too-few-classes": tabular(
        "tabicl-classifier",
        "classification",
        context=[
            row | {"label": "stay"}
            for row in tabular("tabicl-classifier", "classification")["context"]
        ],
    ),
    "quantile-bounds": tabular(
        "tabicl-regressor", "regression", options={"quantiles": [1.5]}
    ),
    "horizon-bounds": forecast(series={"horizon": 100000}),
    "variant-policy": tabular("tabicl-classifier*64", "classification"),
    "surface-gate": tabular("limix-2-classifier", "classification"),
}


def _served(servers: dict[str, Served], model: str) -> Served:
    """The server for a request's model; a model none serves goes to the first."""
    with contextlib.suppress(Exception):
        return servers[parse(model).entry.name]
    return next(iter(servers.values()))


@pytest.mark.parametrize("case", CASES)
def test_every_surface_answers_alike(case, engine, servers):
    """Surfaces answer as the catalog does at their gates; successes agree across gates.

    The hosted Models API and the local client apply the remote gates; the
    operator's own `gf serve` applies the self-hosted ones, so a model gated
    by license may fail differently there, but never differently from the
    in-process catalog at the same gates.
    """
    payload = cast(InferenceRequest, CASES[case])
    remote = _outcome(lambda: CatalogService().infer(SCOPE, payload))
    hosted = _outcome(
        lambda: CatalogService(surface="self-hosted").infer(SCOPE, payload)
    )
    surfaces = {
        "local client": (remote, _outcome(lambda: LocalModels().infer(payload))),
        "gf serve": (
            hosted,
            _outcome(
                lambda: _served(servers, payload["model"]).client.models.infer(payload)
            ),
        ),
    }
    code = remote[1][0] if remote[0] == "error" else "result"
    assert code == EXPECTED[case], remote
    if remote[0] == "result" or hosted[0] == "result":
        surfaces["self-hosted catalog"] = (remote, hosted)
    for name, (expected, outcome) in surfaces.items():
        assert outcome[0] == expected[0], (name, outcome, expected)
        _same(outcome[1], expected[1], name)


RAW = {
    "truncated": b'{"model": "tabicl-classifier", "task":',
    "duplicate-key": b'{"model": "tabicl-classifier", "model": "tabicl-regressor"}',
    "not-an-object": b'["tabicl-classifier"]',
}


@pytest.mark.parametrize("case", RAW)
def test_malformed_payloads_fail_alike(case, servers):
    """Bytes no SDK sends: the shared validator and gf-server refuse them alike."""
    with pytest.raises(_core.RequestError) as caught:
        _core.parse_inference(RAW[case], "self-hosted")
    code, message, status = caught.value.args
    body = (
        b'{"api_version": "gf/v1alpha1", "project": "local", "request_id": "r-raw",'
        b' "resource": {"kind": "model", "name": "tabicl-v2-classifier"},'
        b' "payload": ' + RAW[case] + b"}"
    )
    server = servers["tabicl-v2-classifier"]
    answered, envelope = server.post(
        "/v1alpha1/projects/local/models/tabicl-v2-classifier/infer", body
    )
    assert (answered, envelope["error"]) == (
        status,
        {"code": code, "message": message},
    )


def history(task: str) -> pa.Table:
    """A snapshot context with every column kind a package lock binds."""
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
    [("tabicl-classifier", "classification"), ("tabicl-regressor", "regression")],
)
def test_packages_answer_alike(tmp_path, engine, model, task):
    data = history(task)
    root = tmp_path / "package"
    root.mkdir()
    spec = {
        "name": "scorer",
        "model": model,
        "context": "history.arrow",
        "target": "y",
        "id": "id",
    }
    (root / "gf.yaml").write_text(yaml.safe_dump(spec))
    (root / "history.arrow").write_bytes(arrow_bytes(data))
    package = Package.load(root).check()
    assert package.lock is not None
    queries = data.drop(["y"]).slice(2, 4)
    expected = PackageExecutor().prepare(package.lock, data).predict(queries)
    server = Served(build(), str(root))
    try:
        result = server.client.predict(
            model="scorer",
            task=task,
            target="y",
            features=["x", "plan", "region", "joined"],
            context=[],
            queries=queries,
        )
    finally:
        server.stop()
    _same(result.table.to_pylist(), expected.to_pylist(), "package")


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
    root = tmp_path / "package"
    root.mkdir()
    spec = {
        "name": "planner",
        "model": "chronos-2",
        "context": "history.arrow",
        "target": "sales",
        "time": "day",
        "series": ["store"],
        "horizon": 2,
        "past_inputs": ["footfall"],
        "future_inputs": ["promo"],
        "outputs": ["value", "quantiles"],
    }
    (root / "gf.yaml").write_text(yaml.safe_dump(spec))
    (root / "history.arrow").write_bytes(arrow_bytes(data))
    package = Package.load(root).check()
    assert package.lock is not None
    queries = pa.table(
        {
            "store": ["south", "north", "north", "south"],
            "day": [13, 12, 13, 12],
            "promo": [1.0, 0.0, 1.0, 0.0],
        }
    )
    expected = PackageExecutor().prepare(package.lock, data).predict(queries)
    server = Served(build(), str(root))
    try:
        result = server.client.predict(
            model="planner",
            task="forecast",
            target="sales",
            features=["store", "day", "promo"],
            context=[],
            queries=queries,
        )
    finally:
        server.stop()
    _same(result.table.to_pylist(), expected.to_pylist(), "package")
