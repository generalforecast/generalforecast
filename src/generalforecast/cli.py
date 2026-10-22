# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Installed command parser and safe terminal presentation for gf."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import NoReturn

from generalforecast.commands import (
    _CREDENTIAL_VALUES,
    COLLECTIONS,
    CLIError,
    execute,
    read_data,
    require,
)
from generalforecast.deployments.contracts import ServiceError
from generalforecast.package_commands import PackageCommands
from generalforecast.packages import Package, PackageError, PackageSpec
from generalforecast.serve.listener import add_arguments as add_serve_arguments
from generalforecast.transport import RelayedError


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        safe = (
            message
            if message.startswith("the following arguments are required:")
            else "invalid arguments; run gf --help"
        )
        emit({"error": {"code": "UsageError", "message": safe}}, error=True)
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    """Builds one parser shared by the Python and compiled entry points."""
    common = _Parser(add_help=False)
    for flag in ("url", "base-url", "key-env", "project", "profile", "output"):
        common.add_argument("--" + flag, default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--timeout", type=float, default=argparse.SUPPRESS)
    root = _Parser(
        prog="gf",
        parents=[common],
        description="Immutable prediction packages and operations.",
    )
    root.add_argument("--version", action="version", version="gf 0.1.1")
    sub = root.add_subparsers(dest="group", required=True)

    def command(title, help):
        return sub.add_parser(title, parents=[common], help=help)

    command("login", "Verify a configured API key and save non-secret profile metadata")
    init = command("init", "Create a package directory without downloading models")
    init.add_argument("directory")
    init.add_argument("--name")
    init.add_argument("--model", required=True)
    for flag in ("target", "context", "id", "time", "frequency"):
        init.add_argument("--" + flag)
    init.add_argument("--series", nargs="+")
    init.add_argument("--horizon", type=int)

    def package_flags(p):
        p.add_argument("path", nargs="?", default=".")
        p.add_argument("--data", help="Schema sample relative to package directory")
        p.add_argument("--locked", action="store_true")
        p.add_argument("--frozen", action="store_true")

    check = command("check", "Offline validation and lock generation")
    package_flags(check)
    check.add_argument(
        "--online",
        action="store_true",
        help="Explicit online model metadata qualification",
    )
    check.add_argument("--offline", action="store_true", help="Default: no network")
    check.add_argument("--upgrade", action="store_true")
    push = command(
        "push", "Publish once, optionally patch an isolated remote dev session"
    )
    package_flags(push)
    push.add_argument("--env", default="staging")
    push.add_argument("--runtime", default="local")
    push.add_argument("--watch", action="store_true")
    push.add_argument("--resume", help="Reconnect to a dev session without publishing")
    push.add_argument("--wait", action="store_true", help="Readiness is always awaited")
    push.add_argument("--interval", type=float, default=1)
    push.add_argument("--watch-seconds", type=float, default=3600)
    push.add_argument("--ttl", type=int, default=3600)
    for title in ("status", "logs", "diff", "delete", "promote", "rollback"):
        p = command(title, "Inspect or operate on an immutable deployment")
        p.add_argument("name")
        p.add_argument("--env", default="production")
        if title == "diff":
            p.add_argument("--path", default=".")
            p.add_argument("--data")
        if title in ("promote", "rollback"):
            p.add_argument(
                "--revision" if title == "promote" else "--to",
                dest="revision",
                type=int,
                required=True,
            )
            p.add_argument("--expected-generation", type=int)
        if title == "delete":
            p.add_argument("--yes", action="store_true", required=True)
    predict = command("predict", "Predict through a hosted environment or dev session")
    predict.add_argument("name", nargs="?")
    predict.add_argument("--path", default=".")
    predict.add_argument("--env", default="production")
    predict.add_argument("--session")
    predict.add_argument("--input", required=True)
    predict.add_argument("--context")
    predict.add_argument("--target")
    predict.add_argument("--features", nargs="+")
    serve = sub.add_parser(
        "serve",
        help="Serve a package or catalog model over the Models API on this machine",
    )
    add_serve_arguments(serve)
    datasets = command("datasets", "Logical dataset imports and transactional writes")
    datasets.add_argument(
        "action", choices=["create", "import", "append", "label", "refresh", "status"]
    )
    datasets.add_argument("name")
    datasets.add_argument("input", nargs="?")
    for flag in (
        "file",
        "source",
        "format",
        "credential-ref",
        "id",
        "column",
        "idempotency-key",
    ):
        datasets.add_argument("--" + flag)
    runtimes = command(
        "runtimes",
        "Provisioning operations, separate from authentication and inference",
    )
    runtimes.add_argument(
        "action", choices=["preflight", "connect", "status", "cleanup"]
    )
    runtimes.add_argument("name")
    runtimes.add_argument("--file")
    runtimes.add_argument("--yes", action="store_true")
    for group in COLLECTIONS:
        p = command(group, "Scoped API operations")
        p.add_argument("action")
        p.add_argument("name", nargs="?")
        p.add_argument("--file")
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--after", type=int, default=0)
        p.add_argument("--limit", type=int, default=100)
    return root


def run(argv: list[str] | None = None) -> dict:
    """Dispatches without accepting credential values in argv."""
    args = parser().parse_args(argv)
    if args.group == "init":
        root = Path(args.directory).absolute()
        spec = {"name": args.name or root.name, "model": args.model}
        for key in (
            "target",
            "context",
            "id",
            "time",
            "series",
            "horizon",
            "frequency",
        ):
            if getattr(args, key) is not None:
                spec[key] = getattr(args, key)
        PackageSpec.from_dict(spec)
        require(not root.exists(), "package directory already exists")
        require(root.parent.is_dir(), "package parent directory does not exist")
        import yaml

        root.mkdir(mode=0o700)
        (root / "gf.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        return {"name": spec["name"], "created": True}
    if args.group == "serve":
        from generalforecast.serve import run as serve

        return serve(args)
    if args.group == "check":
        require(not (args.online and args.offline), "choose offline or online")
        package = Package.load(args.path)
        data = read_data(args.data, Path(args.path)) if args.data else None
        if args.online:
            if args.locked or args.frozen:
                raise PackageError(
                    "InvalidPackage", "locked/frozen cannot resolve online"
                )
            checked = package.resolve(data=data, upgrade=args.upgrade)
        else:
            checked = package.check(
                data=data,
                locked=args.locked,
                frozen=args.frozen,
                upgrade=args.upgrade,
            )
        assert checked.lock is not None
        return checked.lock.to_dict()
    if args.group in {
        "push",
        "predict",
        "status",
        "logs",
        "diff",
        "delete",
        "promote",
        "rollback",
    }:
        return PackageCommands(args).run()
    return execute(args)


_SECRET = re.compile(
    r"(?i)(?:gf_(?:live|test|inv)_|hf_|sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{8,}|Bearer\s+\S+"
)


def redact(value):
    """Removes secret response fields and known credential patterns defensively."""
    if isinstance(value, dict):
        return {
            redact(key): "[redacted]"
            if re.search(
                r"(?i)(token|password|secret_value|credential_value|authorization|private_key)",
                key,
            )
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        values = _CREDENTIAL_VALUES | {
            os.environ.get(env, "")
            for env in (
                "GF_API_KEY",
                "GF_TOKEN",
                "GF_PLATFORM_TOKEN",
                "GF_MODAL_KEY",
                "GF_MODAL_SECRET",
            )
        }
        for secret in sorted(values, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[redacted]")
        return _SECRET.sub("[redacted]", value)
    return value


def emit(value: dict, *, error: bool = False) -> None:
    """Writes NDJSON, including watch events, without echoing request bodies."""
    print(
        json.dumps(redact(value), allow_nan=False, ensure_ascii=False),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def main() -> None:
    """Writes typed safe errors and preserves nonzero failure/interruption exits."""
    try:
        emit(run())
    except KeyboardInterrupt:
        emit({"error": {"code": "Interrupted", "message": "interrupted"}}, error=True)
        raise SystemExit(130) from None
    except (ServiceError, PackageError) as error:
        code = (
            error.code
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", error.code)
            else "ClientError"
        )
        message = (
            str(error)
            if isinstance(error, (PackageError, CLIError, RelayedError))
            else {
                "Unauthorized": "authentication failed; configure a valid key and run gf login",
                "Forbidden": "permission denied",
                "SessionRequired": "this operation needs a signed-in Google session in the console",
                "ReauthenticationRequired": "sign in again in the console",
                "Conflict": "state changed; inspect status before retrying",
                "Timeout": "readiness timed out; inspect status",
                "TransportError": "connection failed; outcome may be unknown; reconnect with --resume",
                "ReadinessFailed": "runtime did not become ready; inspect logs",
            }.get(code, "operation failed; check configuration and scoped permissions")
        )
        emit({"error": {"code": code, "message": message}}, error=True)
        raise SystemExit(1) from None
    except (OSError, ValueError, TypeError, KeyError):
        emit(
            {
                "error": {
                    "code": "ClientError",
                    "message": "invalid input or inaccessible local state",
                }
            },
            error=True,
        )
        raise SystemExit(1) from None
    except Exception:  # noqa: BLE001
        emit(
            {
                "error": {
                    "code": "ClientError",
                    "message": "unexpected client failure; no operation was confirmed",
                }
            },
            error=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
