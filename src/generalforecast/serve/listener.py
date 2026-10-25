# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `gf serve` command line: writes a server's configuration, then runs it.

gf-server answers the Models API itself, batching concurrent requests'
queries into shared steps on one worker per device; this process only
checks the target, writes the configuration and becomes the server. With
`--node`s it becomes gf-router instead, which places live tables on those
nodes, and `--config` runs either from a file.
"""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from generalforecast.predictors.validation import PredictorError
from generalforecast.service.contracts import ServiceError, require

if TYPE_CHECKING:
    from generalforecast.term import Output

API_KEY_ENV = "GF_API_KEY"
CLIENT_TOKEN_ENV = "GF_TOKEN"
REPLICATION_TOKEN_ENV = "GF_REPLICATION_TOKEN"
DEFAULT_PORT = 8000
# Longest a Models API request may run.
REQUEST_TIMEOUT_SECONDS = 300.0
TABLE_PORT = 8815
SERVER = "gf-server"
ROUTER = "gf-router"
# Release wheels carry the servers here, off PATH, so `gf` is the one command.
BIN = Path(__file__).resolve().parents[1] / "bin"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The `gf serve` flags, shared by `gf` and `python -m generalforecast.serve`."""
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        metavar="DIR|MODEL",
        help="a predictor directory, its gf.lock, or a model id such as "
        "general-forecast/tabicl-v2-classifier (default: .)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, help=f"port to listen on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use only cached model weights; never download",
    )


def binary(name: str = SERVER) -> Path:
    """One of the servers the wheel carries inside the package, else the first on PATH.

    Images and older wheels install them beside Python instead.
    """
    for directory in (BIN, Path(sysconfig.get_path("scripts"))):
        if (directory / name).is_file():
            return directory / name
    found = shutil.which(name)
    if found is None:
        raise ServiceError(
            "ConfigurationError",
            f"{name} is not installed with gf; reinstall generalforecast",
            503,
        )
    return Path(found)


def _become(
    name: str, *arguments: str, environment: Mapping[str, str] | None = None
) -> NoReturn:
    """Replaces this process with one of the servers, so signals reach it directly."""
    program = binary(name)
    sys.stdout.flush()
    sys.stderr.flush()
    environ = dict(os.environ if environment is None else environment)
    os.execve(program, [str(program), *arguments], environ)


def _routes(config: Path) -> bool:
    """Whether a configuration file is a router's, which lists its `[[nodes]]`."""
    import tomllib

    try:
        with config.open("rb") as file:
            return "nodes" in tomllib.load(file)
    except FileNotFoundError:
        raise ServiceError("NotFound", f"{config} does not exist", 404) from None
    except tomllib.TOMLDecodeError as error:
        raise ServiceError(
            "ConfigurationError", f"{config} is not valid TOML: {error}", 400
        ) from None


def _toml(value: object) -> str:
    """A TOML string or array of strings; JSON's escapes are TOML's."""
    return json.dumps(value, ensure_ascii=False)


def configuration(
    directory: Path,
    address: str,
    timeout: float,
    models: list[dict],
    *,
    keys: Mapping[str, str] | None = None,
    surface: str = "self-hosted",
    upstream: str | None = None,
    trusted_proxies: tuple[str, ...] = (),
    revisions: str | None = None,
) -> str:
    """gf-server's configuration for the models served on this machine.

    Flight listens on loopback under a random token nothing else holds.
    Request contexts live in ephemeral tables; the data directory holds only
    a snapshot predictor's table, bound once at startup. Keys are the bearer
    key in `GF_API_KEY` unless `keys` names another key store. With
    `upstream`, every route the front does not own streams to it, such as a
    control plane on loopback, and `X-Forwarded-*` headers are kept only from
    `trusted_proxies`. With `revisions`, deployed revisions are predicted from
    the control plane at that origin, under the token in `GF_INTROSPECTION_TOKEN`.
    """
    published = (
        []
        if revisions is None
        else [
            "",
            "[http.revisions]",
            f"url = {_toml(revisions)}",
            'token_env = "GF_INTROSPECTION_TOKEN"',
        ]
    )
    keys = {"kind": "static", "env": API_KEY_ENV} if keys is None else keys
    front_door = [] if upstream is None else [f"upstream = {_toml(upstream)}"]
    if trusted_proxies:
        front_door.append(
            f"trusted_proxies = [{', '.join(_toml(c) for c in trusted_proxies)}]"
        )
    command = [sys.executable, "-m", "generalforecast.worker_process"]
    return "\n".join(
        [
            'listen = "127.0.0.1:0"',
            f"data = {_toml(str(directory / 'data'))}",
            f"tokens = [{_toml(secrets.token_hex(32))}]",
            'encryption_key_env = "GF_LOG_KEY"',
            "",
            "[http]",
            f"listen = {_toml(address)}",
            f"models = {_toml(str(directory / 'models.json'))}",
            f"surface = {_toml(surface)}",
            f"request_timeout_seconds = {timeout!r}",
            *front_door,
            *published,
            "",
            "[http.keys]",
            *(f"{field} = {_toml(value)}" for field, value in keys.items()),
            "",
            "[[workers]]",
            f"command = [{', '.join(_toml(part) for part in command)}]",
            "per_gpu = true",
            "",
            "[workers.models]",
            *(f"{_toml(m['worker'])} = {_toml(m['reference'])}" for m in models),
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict:
    """Checks the target, then execs gf-server in this process.

    Nodes and routers are started through `gf admin node` and
    `gf admin router`, which call `configuration`, `route` and `become`.
    """
    require(1 <= len(args.host) <= 255, "invalid host")
    require(
        args.port is None or (type(args.port) is int and 0 <= args.port <= 65535),
        "invalid port",
    )
    return _serve(args)


def _address(host: str, port: int) -> str:
    return f"{f'[{host}]' if ':' in host else host}:{port}"


def _nodes(nodes: list[str]) -> list[str]:
    return [part for node in nodes for part in ("--node", node)]


def _output(args: argparse.Namespace) -> "Output":
    from generalforecast.term import Output

    return Output(quiet=getattr(args, "quiet", False) or getattr(args, "json", False))


def route(args: argparse.Namespace, nodes: list[str]) -> dict:
    """Writes a router configuration over `nodes`, then checks it or becomes the router.

    The nodes already list the replication token, so both tokens come from
    the environment rather than being made up here.
    """
    require(
        args.target == ".", "a router serves tables, not a target; drop it or --node"
    )
    check = bool(getattr(args, "check", False))
    missing = [
        v for v in (CLIENT_TOKEN_ENV, REPLICATION_TOKEN_ENV) if not os.environ.get(v)
    ]
    if missing:
        raise ServiceError(
            "ConfigurationError",
            f"set {' and '.join(missing)}: clients present {CLIENT_TOKEN_ENV}, and each "
            f"node lists {REPLICATION_TOKEN_ENV} in its replication_tokens",
            400,
        )
    directory = Path(tempfile.mkdtemp(prefix=f"gf-{os.getpid()}-"))
    config = directory / "router.toml"
    listen = _address(args.host, TABLE_PORT if args.port is None else args.port)
    written = subprocess.run(
        [
            str(binary(ROUTER)),
            "init",
            "--path",
            str(config),
            "--listen",
            listen,
            *_nodes(nodes),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if written.returncode != 0:
        raise ServiceError(
            "ConfigurationError", written.stderr.strip() or "gf-router init failed", 400
        )
    if not check:
        _output(args).verb(
            "Routing",
            f"tables on {listen}",
            f"across {len(nodes)} node{'s' * (len(nodes) != 1)}",
        )
    _become(ROUTER, "check" if check else "run", "--config", str(config))


def _serve(args: argparse.Namespace) -> dict:
    """Checks the target and its weights, then execs gf-server in this process.

    The configuration goes to a private directory named for this process,
    which a later server removes once this one has exited. gf-server prints
    the `Loading`, `Compiled` and `Serving` status lines as it starts.
    """
    from generalforecast.serve.models import served

    directory = Path(tempfile.mkdtemp(prefix=f"gf-{os.getpid()}-"))
    model = served(args.target, directory, allow_downloads=not args.offline)
    binary()
    (directory / "models.json").write_text(json.dumps([model]))
    address = _address(args.host, DEFAULT_PORT if args.port is None else args.port)
    config = directory / "generalforecast.toml"
    # The configuration holds the Flight token, so only this user reads it.
    descriptor = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(configuration(directory, address, REQUEST_TIMEOUT_SECONDS, [model]))
    environment = dict(os.environ, GF_LOG_KEY=secrets.token_hex(32))
    if args.offline:
        environment["HF_HUB_OFFLINE"] = "1"
    _become(SERVER, "run", "--config", str(config), environment=environment)


def main(argv: list[str] | None = None) -> None:
    """`python -m generalforecast.serve`."""
    parser = argparse.ArgumentParser(
        prog="gf serve",
        description="Serve a predictor or a catalog model on this machine.",
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ServiceError, PredictorError) as error:
        print(
            json.dumps({"error": {"code": error.code, "message": str(error)}}),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None
    print(json.dumps(result), flush=True)
