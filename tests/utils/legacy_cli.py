"""The `generalforecast` command: a server on this machine, and a benchmark of one.

`generalforecast serve` starts a gf-server on this machine with the built-in models,
keeping its configuration and client token under its data directory and its
log key in a file under the user's configuration directory, so one command
gives a server that `generalforecast.connect` reaches. `generalforecast bench` writes
rows to a server, predicts from them, and reports the throughput and latency.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Where `generalforecast serve` listens unless told otherwise.
LISTEN = "127.0.0.1:8815"


def binary(name: str) -> Path:
    """The `gf-server` or `gf-router` a wheel installed, or on PATH, or built here."""
    beside = Path(sys.executable).resolve().parent / name
    if beside.is_file():
        return beside
    found = shutil.which(name)
    if found is not None:
        return Path(found)
    root = Path(__file__).resolve().parents[2]
    built = [root / "target" / profile / name for profile in ("release", "debug")]
    if newest := [path for path in built if path.is_file()]:
        return max(newest, key=lambda path: path.stat().st_mtime)
    raise SystemExit(
        f"{name} is not installed: install the generalforecast wheel, or build it with "
        f"cargo build --release -p {name}"
    )


def _config_directory() -> Path:
    """Where this user's generalforecast secrets live: `$XDG_CONFIG_HOME/generalforecast` or `~/.config/generalforecast`."""
    root = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(root) / "generalforecast"


def _secret(path: Path, size: int) -> str:
    """A secret in a file only this user reads, made on first use."""
    if path.is_file():
        return path.read_text().strip()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    secret = os.urandom(size).hex()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(secret)
    return secret


def serve(options: argparse.Namespace) -> None:
    """Starts a server, taking its place in this process."""
    server = binary("gf-server")
    if options.config is not None:
        os.execv(server, [str(server), "run", "--config", str(options.config)])
    data = Path(options.data).resolve()
    token = options.token or _secret(data / "token", 16)
    environment = dict(os.environ)
    key = None
    if "GF_LOG_KEY" not in environment:
        # The server reads the key from its file, so it stays out of the environment.
        key = _config_directory() / "key"
        _secret(key, 32)
        print(
            f"the log encryption key is in {key}; "
            "beyond development, keep it apart from the data, or set GF_LOG_KEY"
        )
    config = data / "generalforecast.toml"
    if config.is_file():
        # A configuration made before, perhaps edited since, is kept as it is.
        if options.listen != LISTEN or options.model or options.token:
            print(
                f"{config} exists, so --listen, --model and --token leave it as it "
                "is; edit it, or remove it to start over"
            )
        listen = tomllib.loads(config.read_text())["listen"]
    else:
        listen = options.listen
        command = [
            str(server),
            "init",
            "--path",
            str(config),
            "--listen",
            listen,
            "--data",
            str(data),
            "--python",
            sys.executable,
        ]
        for model in options.model:
            command += ["--model", model]
        if key is not None:
            command += ["--key-file", str(key)]
        # The token reaches init through the environment, out of the process list.
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            env={**environment, "GF_TOKEN": token},
        )
    if options.token:
        print(f"serving on grpc://{listen} with the given client token")
    elif sys.stdout.isatty():
        print(f"serving on grpc://{listen}; the client token is {token}")
    else:
        # Output that is not a terminal tends to end up in logs.
        print(f"serving on grpc://{listen}; the client token is in {data / 'token'}")
    os.execve(server, [str(server), "run", "--config", str(config)], environment)


def bench(options: argparse.Namespace) -> None:
    """Writes rows to a server, predicts from them, and reports."""
    import numpy as np

    from tests.utils import legacy_client as generalforecast

    client = generalforecast.connect(
        options.url, token=options.token, allow_plaintext=options.allow_plaintext
    )
    generator = np.random.default_rng(0)
    rows = generator.normal(size=(options.rows, options.features)).astype(np.float32)
    if options.regression:
        targets = rows.sum(axis=1)
    else:
        targets = generator.integers(0, options.classes, size=options.rows)
    queries = generator.normal(size=(options.queries, options.features))
    queries = queries.astype(np.float32)
    batches = [
        queries[start : start + options.batch]
        for start in range(0, options.queries, options.batch)
    ]
    strategy = (
        generalforecast.Strategy.neighbors(options.context) if options.context else None
    )
    name = f"bench-{uuid.uuid4().hex[:8]}"
    table = client.create_table(
        name,
        model=options.model,
        features=options.features,
        classes=None if options.regression else options.classes,
        strategy=strategy,
    )
    try:
        started = time.perf_counter()
        for start in range(0, options.rows, options.batch):
            stop = min(start + options.batch, options.rows)
            table.insert(np.arange(start, stop), rows[start:stop], targets[start:stop])
        inserting = time.perf_counter() - started
        started = time.perf_counter()
        table.predict(batches[0])
        first = time.perf_counter() - started

        def call(batch: np.ndarray) -> float:
            started = time.perf_counter()
            table.predict(batch)
            return time.perf_counter() - started

        started = time.perf_counter()
        with ThreadPoolExecutor(options.concurrency) as pool:
            latencies = sorted(pool.map(call, batches))
        predicting = time.perf_counter() - started
    finally:
        client.drop_table(name)
    print(
        f"inserted   {options.rows} rows in {inserting:.2f} s ({options.rows / inserting:.0f} rows/s)"
    )
    print(f"first call {first:.2f} s, fitting the table and compiling")
    print(
        f"predicted  {options.queries} queries in {predicting:.2f} s "
        f"({options.queries / predicting:.0f} queries/s; {options.batch} per call, "
        f"{options.concurrency} calls at once)"
    )
    middle, high = (
        latencies[len(latencies) // 2],
        latencies[(len(latencies) * 19) // 20],
    )
    print(f"latency    p50 {middle * 1000:.0f} ms, p95 {high * 1000:.0f} ms per call")


def main(arguments: Sequence[str] | None = None) -> None:
    """Runs the `generalforecast` command."""
    parser = argparse.ArgumentParser(prog="generalforecast", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serving = commands.add_parser(
        "serve", help="start a server on this machine with the built-in models"
    )
    serving.add_argument(
        "--config", type=Path, help="a gf-server configuration to run instead"
    )
    serving.add_argument("--listen", default=LISTEN)
    serving.add_argument(
        "--data",
        default="gf-data",
        help="where the table logs, configuration and token go",
    )
    serving.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=SPEC",
        help="a model to serve, repeated; the TabICL models if none",
    )
    serving.add_argument("--token", help="the client token; kept in data otherwise")
    serving.set_defaults(run=serve)

    benching = commands.add_parser("bench", help="measure a server")
    benching.add_argument("--url", required=True, help="such as grpc://127.0.0.1:8815")
    benching.add_argument("--token")
    benching.add_argument("--allow-plaintext", action="store_true")
    benching.add_argument(
        "--model",
        default="general-forecast/tabicl-v2-classifier",
        help="a model name the server serves",
    )
    benching.add_argument("--rows", type=int, default=10_000)
    benching.add_argument("--queries", type=int, default=1_000)
    benching.add_argument("--features", type=int, default=16)
    benching.add_argument(
        "--classes", type=int, default=2, help="classes of a classifier's table"
    )
    benching.add_argument(
        "--regression", action="store_true", help="the model is a regressor"
    )
    benching.add_argument("--batch", type=int, default=100, help="rows per call")
    benching.add_argument("--concurrency", type=int, default=8, help="calls at once")
    benching.add_argument(
        "--context", type=int, help="rows each query reads, its nearest; all by default"
    )
    benching.set_defaults(run=bench)

    options = parser.parse_args(arguments)
    options.run(options)


if __name__ == "__main__":
    main()
