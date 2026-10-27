# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A gf serve stack on the tiny test models, for `pytest -m conformance`.

It serves what api/openapi.yaml's examples and tests/conformance name: the
snapshot predictor `churn` (general-forecast/tabicl-v2-classifier, id
`customer_id`, inputs `spend` and `plan`), the forecast predictor `sales`
(general-forecast/chronos-2, time `date`, series `sku`, a daily frequency,
the known-future input `price`, contexts dated before 2026-01-31), and the
ad hoc route over the classifier, the regressor and the forecaster.

gf-server is configured exactly as `gf serve` configures it
(`listener.configuration` over `serve.models.served` entries), with one
difference: each catalog model's worker loads the tiny model registered in
tests/models/implementations.py under its id, so nothing is downloaded and
no weights are verified. The answers are the tiny models'; their shapes are
the server's.

Once /readyz answers 200 it prints one JSON line, `{"url", "token",
"project"}`, which `GF_CONFORMANCE_URL`, `GF_CONFORMANCE_TOKEN` and
`GF_CONFORMANCE_PROJECT` take, and runs until it is signalled. Everything it
wrote is in a temporary directory removed on exit, and the server's whole
process group is killed with it.

    python scripts/conformance_server.py [--binary target/debug/gf-server]
"""

import argparse
import datetime
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generalforecast.serve import listener
from generalforecast.serve import models as serve_models
from tests.utils.locks import directory

PROJECT = "local"
# Each catalog model the stack serves, and the tiny model its worker loads.
TINY = {
    "general-forecast/tabicl-v2-classifier": "tests/tabicl-classifier",
    "general-forecast/tabicl-v2-regressor": "tests/tabicl-regressor",
    "general-forecast/chronos-2": "tests/chronos2-catalog-forecaster",
}
# The first day a `sales` query may name: every context row is before it.
FORECAST_START = datetime.date(2026, 1, 31)
READY_SECONDS = 600.0


def churn_rows() -> pa.Table:
    """Customers with spend, plan and whether they churned."""
    count = 40
    return pa.table(
        {
            "customer_id": list(range(1, count + 1)),
            "spend": [float((index * 37) % 120) for index in range(count)],
            "plan": [("free", "pro", "team")[index % 3] for index in range(count)],
            "churned": [int(index % 4 == 0) for index in range(count)],
        }
    )


def sales_rows() -> pa.Table:
    """Thirty days of two SKUs' units sold, and the price each day."""
    days = [FORECAST_START - datetime.timedelta(days=30 - step) for step in range(30)]
    return pa.table(
        {
            "sku": [sku for sku in ("a", "b") for _ in days],
            "date": pa.array([day for _ in ("a", "b") for day in days], pa.date32()),
            "units": [
                float(10 + step % 7 + 5 * index)
                for index in range(2)
                for step in range(len(days))
            ],
            "price": [float(9 + step % 3) for _ in ("a", "b") for step in range(30)],
        }
    )


def predictors(root: Path) -> list[Path]:
    """The `churn` and `sales` predictor directories, locked as `gf check` locks them."""
    churn = directory(
        root / "churn",
        churn_rows(),
        model="general-forecast/tabicl-v2-classifier",
        target="churned",
        name="churn",
        id="customer_id",
    )
    sales = directory(
        root / "sales",
        sales_rows(),
        model="general-forecast/chronos-2",
        target="units",
        name="sales",
        time="date",
        series=["sku"],
        frequency="1d",
        horizon=2,
        inputs={"future": ["price"]},
    )
    return [churn, sales]


def _unverified(*_: Any, **__: Any) -> None:
    """Stands in for weight verification: the workers load the tiny models instead."""


def entries(root: Path) -> list[dict[str, Any]]:
    """gf-server's models file: both predictors, then the catalog models."""
    # The tiny models stand in for the weights, so none are checked.
    setattr(serve_models, "checkpoint", _unverified)  # noqa: B010
    served = []
    for target in [*map(str, predictors(root / "predictors")), *TINY]:
        # Each target writes its bound context rows to a directory of its own.
        place = root / "bound" / str(len(served))
        place.mkdir(parents=True)
        served.append(serve_models.served(target, place, allow_downloads=False))
    return served


def tiny(served: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One worker model per catalog model, its reference the tiny model's."""
    workers: dict[str, dict[str, Any]] = {}
    for entry in served:
        workers.setdefault(
            entry["worker"],
            entry | {"reference": TINY[entry["resolved"].split("@")[0]]},
        )
    return list(workers.values())


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def ready(url: str, process: subprocess.Popen[bytes]) -> None:
    """Waits until /readyz answers 200, or the server exits."""
    deadline = time.monotonic() + READY_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"gf-server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    raise SystemExit(f"gf-server was not ready within {READY_SECONDS:.0f} seconds")


def binary(given: str | None) -> Path:
    """The gf-server binary given, or one cargo builds."""
    if given is not None:
        return Path(given)
    subprocess.run(["cargo", "build", "-q", "-p", "gf-server"], cwd=ROOT, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    return target / "debug" / "gf-server"


def stop(process: subprocess.Popen[bytes]) -> None:
    """Ends the server and its workers, which share its process group."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    """Serves the stack until signalled, then removes everything it made."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", help="gf-server; built with cargo if left out")
    args = parser.parse_args()
    program = binary(args.binary)
    token = secrets.token_hex(16)
    # SIGTERM unwinds like Ctrl-C, so the cleanup below runs either way.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    with tempfile.TemporaryDirectory(prefix="gf-conformance-") as name:
        root = Path(name)
        served = entries(root)
        (root / "models.json").write_text(json.dumps(served))
        url = f"http://127.0.0.1:{_port()}"
        config = root / "generalforecast.toml"
        config.write_text(
            listener.configuration(
                root,
                url.removeprefix("http://"),
                listener.REQUEST_TIMEOUT_SECONDS,
                tiny(served),
            )
        )
        config.chmod(0o600)
        environment = dict(
            os.environ,
            GF_LOG_KEY=secrets.token_hex(32),
            GF_WORKER_MODULES="tests.models.implementations",
            PYTHONPATH=str(ROOT),
            **{listener.API_KEY_ENV: token},
        )
        process = subprocess.Popen(
            [str(program), "run", "--config", str(config)],
            env=environment,
            # Its own JSON events go to stderr: stdout carries only the line above.
            stdout=sys.stderr,
            start_new_session=True,
        )
        try:
            ready(url, process)
            print(
                json.dumps({"url": url, "token": token, "project": PROJECT}), flush=True
            )
            process.wait()
        except KeyboardInterrupt:
            pass
        finally:
            stop(process)


if __name__ == "__main__":
    main()
