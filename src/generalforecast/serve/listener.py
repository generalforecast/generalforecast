# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runs gf-server and gf-router for `gf serve`, `gf admin node` and `gf admin router`.

gf-server answers the Models API itself, batching concurrent requests'
queries into shared steps on one worker per device. `gf serve` checks the
target, writes the server's configuration to a private directory and
becomes the server, so signals reach it directly. A node or a router runs
from a configuration file its `init` wrote, which `check` validates.

The servers need Linux or macOS: they replace this process, signal their
workers and talk to them over Unix sockets. Elsewhere these commands refuse
before doing anything.
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

from generalforecast.errors import Error, InvalidInput, NotFound

if TYPE_CHECKING:
    from generalforecast.term import Output

API_KEY_ENV = "GF_API_KEY"
# Longest a Models API request may run.
REQUEST_TIMEOUT_SECONDS = 300.0
SERVER = "gf-server"
ROUTER = "gf-router"
# Whether this platform runs the servers: Linux and macOS do, Windows does not.
POSIX = os.name == "posix"
# Release wheels carry the servers here, off PATH, so `gf` is the one command.
BIN = Path(__file__).resolve().parents[1] / "bin"
# Removes the directory named by argv[1] once every holder of the pipe whose
# read end is fd argv[2] has exited: the server, and every worker it started.
_REMOVER = (
    "import os, shutil, sys\n"
    "fd = int(sys.argv[2])\n"
    "while os.read(fd, 4096):\n"
    "    pass\n"
    "shutil.rmtree(sys.argv[1], ignore_errors=True)\n"
)


def supported() -> None:
    """Refuses on a platform the servers do not run on.

    Raises:
        Error: `UnsupportedOperation` anywhere but Linux and macOS.
    """
    if not POSIX:
        raise Error(
            "the servers run on Linux and macOS only",
            code="UnsupportedOperation",
            hint="run `gf serve` on a Linux or macOS machine, or in a container",
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
        raise Error(
            f"{name} is not installed with gf",
            code="ConfigurationError",
            hint="reinstall generalforecast",
        )
    return Path(found)


def _become(program: Path, *arguments: str, environment: Mapping[str, str]) -> NoReturn:
    """Replaces this process with one of the servers, so signals reach it directly."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(program, [str(program), *arguments], dict(environment))


def _environment(
    args: argparse.Namespace, *, offline: bool = False, **extra: str
) -> dict[str, str]:
    """The server's environment: this process's, `extra`, and what the flags ask for.

    gf-server logs warnings only on a terminal unless `GF_LOG` says otherwise,
    so the status lines stand alone; `-v` asks for its progress lines too and
    `-q` for errors alone. `--json` asks for NDJSON events even on a terminal.
    `--offline` keeps the workers from downloading weights.
    """
    environment = dict(os.environ, **extra)
    if args.verbose:
        environment.setdefault("GF_LOG", "info")
    elif args.quiet:
        environment.setdefault("GF_LOG", "error")
    if args.json:
        environment["GF_EVENTS"] = "json"
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["GF_OFFLINE"] = "1"
    return environment


def _toml(value: object) -> str:
    """A TOML string or array of strings; JSON's escapes are TOML's."""
    return json.dumps(value, ensure_ascii=False)


def configuration(
    directory: Path,
    address: str,
    timeout: float,
    models: list[dict],
    *,
    metrics: str | None = None,
) -> str:
    """gf-server's configuration for the models served on this machine.

    Flight listens on loopback under a random token nothing else holds.
    Request contexts live in ephemeral tables; the data directory holds only
    a snapshot predictor's table, bound once at startup. Keys are the bearer
    key in `GF_API_KEY`. With `metrics`, a `host:port`, gf-server also serves
    `/metrics`, `/healthz` and `/readyz` there once its workers start; they
    take no token, so it belongs on loopback or a private network.
    """
    command = [sys.executable, "-m", "generalforecast.worker_process"]
    return "\n".join(
        [
            'listen = "127.0.0.1:0"',
            *([] if metrics is None else [f"metrics_listen = {_toml(metrics)}"]),
            f"data = {_toml(str(directory / 'data'))}",
            f"tokens = [{_toml(secrets.token_hex(32))}]",
            'encryption_key_env = "GF_LOG_KEY"',
            "",
            "[http]",
            f"listen = {_toml(address)}",
            f"models = {_toml(str(directory / 'models.json'))}",
            'surface = "self-hosted"',
            f"request_timeout_seconds = {timeout!r}",
            "",
            "[http.keys]",
            'kind = "static"',
            f"env = {_toml(API_KEY_ENV)}",
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


def address(host: str, port: int) -> str:
    """`host:port`, bracketing an IPv6 host."""
    if not 1 <= len(host) <= 255:
        raise InvalidInput("the host must be 1 to 255 characters", param="host")
    if not 0 <= port <= 65535:
        raise InvalidInput(f"port {port} is not between 0 and 65535", param="port")
    return f"{f'[{host}]' if ':' in host else host}:{port}"


def private_directory() -> tuple[Path, int]:
    """A new 0700 directory, removed once this process and what it becomes exit.

    A child process holds the read end of a pipe whose write end survives
    `exec`; when the last holder exits, however it exits, the read sees the
    end of the pipe and the child removes the directory. The directory is
    named `gf-<pid>-...`, so a server's own sweep of stale directories finds
    it too, should the child itself be killed.

    Returns:
        The directory, and the pipe's write end, which stays open until exit.
    """
    directory = Path(tempfile.mkdtemp(prefix=f"gf-{os.getpid()}-"))
    read, write = os.pipe()
    try:
        subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _REMOVER, str(directory), str(read)],
            pass_fds=(read,),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except BaseException:
        os.close(write)
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        os.close(read)
    os.set_inheritable(write, True)
    return directory, write


def serve(args: argparse.Namespace) -> NoReturn:
    """`gf serve`: checks the target and its weights, then becomes gf-server.

    The server binary is found before any weights are checked or downloaded.
    The configuration holds the Flight token, so it goes to a private
    directory that is removed when the server exits, or at once if anything
    here fails first.
    """
    from generalforecast.serve.models import served

    supported()
    listen = address(args.host, args.port)
    program = binary(SERVER)
    directory, holder = private_directory()
    try:
        model = served(args.target, directory, allow_downloads=not args.offline)
        (directory / "models.json").write_text(json.dumps([model]))
        config = directory / "generalforecast.toml"
        descriptor = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            file.write(
                configuration(directory, listen, REQUEST_TIMEOUT_SECONDS, [model])
            )
        environment = _environment(
            args, offline=args.offline, GF_LOG_KEY=secrets.token_hex(32)
        )
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        os.close(holder)
        raise
    _become(program, "run", "--config", str(config), environment=environment)


def _run(program: Path, *arguments: str) -> list[str]:
    """Runs a server's `init` or `check`; returns its report lines.

    Raises:
        Error: With the server's own words when it fails.
    """
    done = subprocess.run(
        [str(program), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if done.returncode != 0:
        failed = [line for line in lines if line.startswith("failed")]
        raise InvalidInput(
            f"{program.name} {arguments[0]} failed",
            because="\n".join(failed) or done.stderr.strip() or None,
        )
    return lines


def _config(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_file():
        raise NotFound(
            f"no configuration at `{path}`",
            hint="write one with `init`, or pass --config FILE",
            param="config",
        )
    return path


def admin(noun: str, args: argparse.Namespace, out: "Output") -> dict:
    """`gf admin node|router init|check|serve`, over gf-server and gf-router."""
    supported()
    program = binary(SERVER if noun == "node" else ROUTER)
    if args.action == "serve":
        _become(
            program,
            "run",
            "--config",
            str(_config(args.config)),
            environment=_environment(args, offline=noun == "node" and args.offline),
        )
    if args.action == "check":
        config = _config(args.config)
        return {
            "config": str(config),
            "checks": _run(program, "check", "--config", str(config)),
        }
    path = args.path.expanduser()
    if path.exists() and not args.force:
        raise InvalidInput(
            f"`{path}` already exists",
            hint="pass --force to overwrite it",
            param="path",
        )
    arguments = ["init", "--path", str(path), "--listen", address(args.host, args.port)]
    if noun == "node":
        for model in args.model:
            served, _, reference = model.partition("=")
            if not served or not reference:
                raise InvalidInput(
                    f"--model takes NAME=MODEL, not `{model}`",
                    hint="name the model as clients will, like "
                    "tabicl=general-forecast/tabicl-v2-classifier",
                    param="model",
                )
        arguments += ["--data", str(args.data), "--python", sys.executable]
        arguments += [part for model in args.model for part in ("--model", model)]
    else:
        arguments += [part for node in args.node for part in ("--node", node)]
    if args.force:
        arguments.append("--force")
    _run(program, *arguments)
    out.status("Wrote", f"{path}, which holds the client token")
    steps = [
        f"gf admin {noun} check --config {path}",
        f"gf admin {noun} serve --config {path}",
    ]
    if noun == "router":
        steps.insert(
            0, f"list the replication token in {path} in each node's replication_tokens"
        )
    elif not os.environ.get("GF_LOG_KEY"):
        steps.insert(0, "export GF_LOG_KEY=$(openssl rand -hex 32)")
    return {"path": str(path), "next": steps}


def main(argv: list[str] | None = None) -> None:
    """`python -m generalforecast.serve`: `gf serve`, with errors as JSON."""
    from generalforecast import cli

    cli.main(["serve", "--json", *(sys.argv[1:] if argv is None else argv)])
