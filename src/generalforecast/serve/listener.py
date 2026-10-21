"""The `gf serve` command line: writes gf-server's configuration, then runs it.

gf-server answers the Models API itself, batching concurrent requests'
queries into shared steps on one worker per device; this process only
checks the target, writes the configuration and becomes the server.
"""

import argparse
import json
import os
import secrets
import shutil
import sys
import sysconfig
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from generalforecast.deployments.contracts import ServiceError, require
from generalforecast.packages.validation import PackageError

API_KEY_ENV = "GF_API_KEY"
DEFAULT_PORT = 8000
SERVER = "gf-server"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The `gf serve` flags, shared by `gf` and `python -m generalforecast.serve`."""
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Package directory, gf.lock file or catalog model reference",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only cached model weights; never download",
    )
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument(
        "--upstream",
        help="Origin, such as http://127.0.0.1:8001, that every route the "
        "Models API does not own streams to",
    )


def binary() -> Path:
    """The gf-server the wheel installs beside Python, or the first on PATH."""
    installed = Path(sysconfig.get_path("scripts")) / SERVER
    if installed.is_file():
        return installed
    found = shutil.which(SERVER)
    if found is None:
        raise ServiceError(
            "ConfigurationError", "gf-server is not installed beside gf", 503
        )
    return Path(found)


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
    a snapshot package's table, bound once at startup. Keys are the bearer
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


def run(args: argparse.Namespace) -> NoReturn:
    """Checks the target and its weights, then execs gf-server in this process.

    The configuration goes to a private directory named for this process,
    which a later server removes once this one has exited.
    """
    require(1 <= len(args.host) <= 255, "invalid host")
    require(type(args.port) is int and 0 <= args.port <= 65535, "invalid port")
    require(
        0 < args.request_timeout <= 3600, "request timeout must be within (0, 3600]"
    )
    from generalforecast.serve.models import served

    directory = Path(tempfile.mkdtemp(prefix=f"gf-{os.getpid()}-"))
    model = served(args.target, directory, allow_downloads=not args.offline)
    server = binary()
    (directory / "models.json").write_text(json.dumps([model]))
    host = f"[{args.host}]" if ":" in args.host else args.host
    config = directory / "generalforecast.toml"
    # The configuration holds the Flight token, so only this user reads it.
    descriptor = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(
            configuration(
                directory,
                f"{host}:{args.port}",
                args.request_timeout,
                [model],
                upstream=args.upstream,
            )
        )
    environment = dict(os.environ, GF_LOG_KEY=secrets.token_hex(32))
    if args.offline:
        environment["HF_HUB_OFFLINE"] = "1"
    sys.stdout.flush()
    os.execve(server, [str(server), "run", "--config", str(config)], environment)


def main(argv: list[str] | None = None) -> None:
    """`python -m generalforecast.serve`."""
    parser = argparse.ArgumentParser(
        prog="gf serve", description="Serve a catalog model over the Models API."
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ServiceError, PackageError) as error:
        print(
            json.dumps({"error": {"code": error.code, "message": str(error)}}),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None
