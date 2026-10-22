# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The gf command: parser, help, error presentation and output mode.

Human output is the default for the installed command; `--json` (and any
programmatic use of `run`) keeps machine-readable NDJSON.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from generalforecast import term
from generalforecast.command_specs import (
    COLLECTIONS,
    COMMAND_HELP,
    OPERATIONS,
    SECTIONS,
    action_help,
)


def version() -> str:
    """The installed distribution's version, from its metadata (no heavy imports)."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as installed

    try:
        return installed("generalforecast")
    except PackageNotFoundError:
        return "unknown"


class _Version(argparse.Action):
    """`-V/--version`, resolved only when asked for."""

    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        help=None,
    ):
        super().__init__(option_strings, dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        sys.stdout.write(f"gf {version()}\n")
        parser.exit()


DOCS = "https://docs.generalforecast.com"
SUPPRESS = argparse.SUPPRESS
PACKAGE_GROUPS = {
    "push",
    "predict",
    "status",
    "logs",
    "diff",
    "delete",
    "promote",
    "rollback",
}
JSON = True
ARGS: argparse.Namespace | None = None
_OUT: term.Output | None = None
_ARGV: list[str] = []


def output() -> term.Output:
    """The current invocation's writers."""
    return _OUT if _OUT is not None else term.Output()


@dataclass
class Problem(Exception):
    """An expected failure, ready to print: message, cause, hints, source pointer."""

    code: str
    message: str
    cause: str | None = None
    hints: list[str] = field(default_factory=list)
    pointer: tuple | None = None


class _Formatter(argparse.HelpFormatter):
    """cargo-style help: `Usage: gf push [OPTIONS] [PATH]`, `<VALUE>` metavars."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=30, width=100)

    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = "Usage: "
            if usage is None:
                usage = " ".join(["%(prog)s", "[OPTIONS]", *_usage_parts(actions)])
        super().add_usage(usage, actions, groups, prefix)

    def start_section(self, heading):
        names = {"positional arguments": "Arguments", "options": "Options"}
        super().start_section(names.get(heading or "", heading))

    def _format_action(self, action):
        if isinstance(action, argparse._SubParsersAction):
            return "".join(self._format_action(a) for a in action._get_subactions())
        return super()._format_action(action)

    def _get_default_metavar_for_optional(self, action):
        return f"<{action.dest.upper()}>"

    def _get_default_metavar_for_positional(self, action):
        return f"<{action.dest.upper()}>"


def _usage_parts(actions) -> list[str]:
    parts = []
    for action in actions:
        if action.help is SUPPRESS:
            continue
        if action.option_strings:
            if action.required:
                parts.append(f"{action.option_strings[-1]} <{action.dest.upper()}>")
        elif isinstance(action, argparse._SubParsersAction):
            parts.append(str(action.metavar))
        else:
            name = f"<{action.dest.upper()}>"
            parts.append(f"[{name}]" if action.nargs in ("?", "*") else name)
    return parts


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args: Any, add_help: bool = True, **options: Any) -> None:
        super().__init__(*args, add_help=False, **options)
        if add_help:
            self.add_argument(
                "-h", "--help", action="help", default=SUPPRESS, help="Print help"
            )

    def format_help(self) -> str:
        out = output()
        if self.prog == "gf":
            return root_help(out)
        text = super().format_help()
        return re.sub(
            r"(?m)^(Usage:|[A-Z][A-Za-z ]*:$)",
            lambda m: out.style(m.group(1), "green", "bold"),
            text,
        )

    def error(self, message: str) -> NoReturn:
        text, hint = _usage_problem(self, message)
        if JSON:
            emit({"error": {"code": "UsageError", "message": text}}, error=True)
            raise SystemExit(2)
        out = output()
        out.error(redact(text))
        if hint:
            out.hint(hint)
        out.say()
        prog = self.prog
        if prog == "gf" and message.startswith("unrecognized arguments"):
            group = next((a for a in sys.argv[1:] if a in _choices(self)), None)
            prog = f"gf {group}" if group else prog
        if prog == "gf":
            out.say("Usage: gf <COMMAND>. Run gf --help for all commands.")
        else:
            out.say(f"Run {prog} --help for usage.")
        raise SystemExit(2)


def root_help(out: term.Output) -> str:
    """`gf --help`: grouped commands and global options."""
    lines = [
        "General Forecast: Inference for structured data foundation models",
        "",
        f"{out.style('Usage:', 'green', 'bold')} {out.style('gf', 'cyan', 'bold')} <COMMAND>",
    ]
    width = max(len(n) for _, names in SECTIONS for n in names) + 4
    for title, names in SECTIONS:
        lines += ["", out.style(f"{title}:", "green", "bold")]
        for name in names:
            label = out.style(name.ljust(width), "cyan", "bold")
            lines.append(f"  {label}{COMMAND_HELP[name]}")
    options = (
        ("--json", "Machine-readable output"),
        ("-q, --quiet", "Only print errors"),
        ("-v, --verbose", "Show requests and timings"),
        ("-V, --version", "Print version"),
    )
    lines += ["", out.style("Options:", "green", "bold")]
    for flag, text in options:
        lines.append(f"  {out.style(flag.ljust(15), 'cyan', 'bold')}{text}")
    lines += [
        "",
        f"Run gf <command> --help for details. Docs: {out.link(DOCS, out.out)}",
        "",
    ]
    return "\n".join(lines)


def _choices(p: argparse.ArgumentParser) -> list[str]:
    for action in p._actions:
        if isinstance(action, argparse._SubParsersAction):
            return list(action.choices)
    return []


def _flags(p: argparse.ArgumentParser) -> set[str]:
    found: set[str] = set()
    for action in p._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                found |= _flags(child)
        elif action.help is not SUPPRESS:
            found |= {s for s in action.option_strings if s.startswith("--")}
    return found


def _usage_problem(p: argparse.ArgumentParser, message: str) -> tuple[str, str | None]:
    """Turns argparse's message into a short error and a suggestion, never echoing values."""
    choice = re.search(r"invalid choice: '([^']*)'", message)
    if choice:
        word, names = choice.group(1), _choices(p)
        prior = _ARGV[: _ARGV.index(word)] if word in _ARGV else []
        if prior and prior[-1].startswith("-") and prior[-1] not in _flags(p):
            # The "command" is the value of an unknown flag; never echo it.
            flag = prior[-1].split("=", 1)[0]
            match = term.suggest(flag, _flags(p))
            return f"unexpected argument `{flag}`", match and f"did you mean `{match}`?"
        match = term.suggest(word, names)
        shown = word if re.fullmatch(r"[\w.-]{1,40}", word) else "..."
        if p.prog == "gf":
            text = f"unknown command `{shown}`"
        else:
            text = f"unknown action `{shown}` for `{p.prog}`"
        hint = (
            f"did you mean `{match}`?"
            if match
            else f"choose one of: {', '.join(sorted(names))}"
        )
        return text, hint
    extra = re.match(r"unrecognized arguments: (.*)", message)
    if extra:
        first = extra.group(1).split()[0]
        if first.startswith("-"):
            flag = first.split("=", 1)[0]
            match = term.suggest(flag, _flags(p))
            return f"unexpected argument `{flag}`", match and f"did you mean `{match}`?"
        return "unexpected extra argument", None
    required = re.match(r"the following arguments are required: (.*)", message)
    if required:
        names = [n.strip() for n in required.group(1).split(",")]
        if names == ["<ACTION>"]:
            return (
                f"`{p.prog}` needs an action",
                f"choose one of: {', '.join(sorted(_choices(p)))}",
            )
        plural = "s" if len(names) > 1 else ""
        return (
            f"missing required argument{plural} {', '.join(f'`{n}`' for n in names)}",
            None,
        )
    invalid = re.match(r"argument ([^:]+): invalid (\w+) value", message)
    if invalid:
        kind = {"int": "an integer", "float": "a number"}.get(
            invalid.group(2), "a valid value"
        )
        return (
            f"invalid value for `{invalid.group(1).split('/')[-1]}`: expected {kind}",
            None,
        )
    needs = re.match(r"argument ([^:]+): expected one argument", message)
    if needs:
        return f"`{needs.group(1).split('/')[-1]}` needs a value", None
    return "invalid arguments", "run gf --help"


def parser() -> argparse.ArgumentParser:
    """Builds one parser shared by the Python and compiled entry points."""
    common = _Parser(add_help=False)
    flags = common.add_argument_group("Global options")
    flags.add_argument(
        "--json", action="store_true", default=SUPPRESS, help="Machine-readable output"
    )
    flags.add_argument(
        "-q", "--quiet", action="store_true", default=SUPPRESS, help="Only print errors"
    )
    flags.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=SUPPRESS,
        help="Show requests and timings",
    )
    flags.add_argument(
        "--timeout",
        type=float,
        default=SUPPRESS,
        metavar="<SECONDS>",
        help="Request timeout",
    )
    # Compatibility only: saved profiles, explicit endpoints and named key variables.
    for flag in ("url", "base-url", "key-env", "project", "profile"):
        common.add_argument("--" + flag, default=SUPPRESS, help=SUPPRESS)
    base = common
    common = _Parser(add_help=False, parents=[base])
    common.add_argument("--output", default=SUPPRESS, help=SUPPRESS)
    root = _Parser(prog="gf", parents=[common], formatter_class=_Formatter)
    root.add_argument("-V", "--version", action=_Version)
    sub = root.add_subparsers(dest="group", required=True, metavar="<COMMAND>")

    def command(title: str, **options: Any) -> argparse.ArgumentParser:
        options.setdefault("parents", [common])
        return sub.add_parser(
            title,
            help=COMMAND_HELP.get(title),
            description=COMMAND_HELP.get(title),
            formatter_class=_Formatter,
            **options,
        )

    login = command("login")
    login.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the login URL instead of opening a browser",
    )
    command("logout")
    command("whoami")
    init = command("init")
    init.add_argument("directory", help="Directory to create")
    init.add_argument("--name", help="Package name (default: the directory name)")
    init.add_argument(
        "--model", required=True, help="Catalog model, such as tabicl-v2-classifier"
    )
    init.add_argument("--target", help="Column to predict")
    init.add_argument("--context", help="Context table for the model")
    init.add_argument("--id", help="Row identifier column")
    init.add_argument("--time", help="Timestamp column")
    init.add_argument("--frequency", help="Series frequency, such as 1d")
    init.add_argument("--series", nargs="+", help="Series identifier columns")
    init.add_argument("--horizon", type=int, help="Forecast horizon in steps")

    def package_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "path", nargs="?", default=".", help="Package directory (default: .)"
        )
        p.add_argument(
            "--data", help="Schema sample, relative to the package directory"
        )
        p.add_argument(
            "--locked", action="store_true", help="Fail if gf.lock would change"
        )
        p.add_argument("--frozen", action="store_true", help="Use gf.lock as is")

    check = command("check")
    package_flags(check)
    check.add_argument(
        "--online", action="store_true", help="Resolve model metadata online"
    )
    check.add_argument("--offline", action="store_true", help=SUPPRESS)
    check.add_argument(
        "--upgrade", action="store_true", help="Resolve the newest model version"
    )
    push = command("push")
    package_flags(push)
    push.add_argument("--env", default="staging", help="Environment (default: staging)")
    push.add_argument("--runtime", default="local", help=SUPPRESS)
    push.add_argument(
        "--watch", action="store_true", help="Patch a dev session as files change"
    )
    push.add_argument("--resume", metavar="SESSION", help="Reconnect to a dev session")
    push.add_argument("--wait", action="store_true", help=SUPPRESS)
    push.add_argument("--interval", type=float, default=1, help=SUPPRESS)
    push.add_argument("--watch-seconds", type=float, default=3600, help=SUPPRESS)
    push.add_argument(
        "--ttl",
        type=int,
        default=3600,
        metavar="<SECONDS>",
        help="Dev session lifetime",
    )
    for title in ("status", "logs", "diff", "delete", "promote", "rollback"):
        p = command(title)
        p.add_argument("name", help="Deployment name")
        p.add_argument(
            "--env", default="production", help="Environment (default: production)"
        )
        if title == "diff":
            p.add_argument("--path", default=".", help="Package directory (default: .)")
            p.add_argument("--data", help="Schema sample")
        if title in ("promote", "rollback"):
            p.add_argument(
                "--revision" if title == "promote" else "--to",
                dest="revision",
                type=int,
                required=True,
                help="Revision number",
            )
            p.add_argument("--expected-generation", type=int, help=SUPPRESS)
        if title == "delete":
            p.add_argument(
                "--yes", action="store_true", required=True, help="Confirm deletion"
            )
    predict = command("predict", parents=[base])
    predict.add_argument(
        "name", nargs="?", help="Deployment (default: the package in --path)"
    )
    predict.add_argument("--path", default=".", help="Package directory (default: .)")
    predict.add_argument(
        "--env", default="production", help="Environment (default: production)"
    )
    predict.add_argument("--session", help="Dev session to predict with")
    predict.add_argument(
        "--input", required=True, help="Rows to predict (.csv, .parquet or .json)"
    )
    predict.add_argument("--context", help="Context rows")
    predict.add_argument("--target", help="Column to predict")
    predict.add_argument("--features", nargs="+", help="Feature columns")
    predict.add_argument(
        "--output", help="Write predictions to a .csv, .parquet or .arrow file"
    )
    from generalforecast.serve.listener import add_arguments as add_serve_arguments

    serve = command("serve", parents=[base])
    add_serve_arguments(serve)
    datasets = command("datasets")
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
    runtimes = command("runtimes")
    runtimes.add_argument(
        "action", choices=["preflight", "connect", "status", "cleanup"]
    )
    runtimes.add_argument("name")
    runtimes.add_argument("--file")
    runtimes.add_argument("--yes", action="store_true")
    for group in COLLECTIONS:
        p = command(group)
        actions = p.add_subparsers(
            dest="action", required=True, metavar="<ACTION>", title="Actions"
        )
        for action in sorted(OPERATIONS[group]):
            a = actions.add_parser(
                action,
                parents=[common],
                help=action_help(group, action),
                description=action_help(group, action),
                formatter_class=_Formatter,
            )
            a.add_argument("name", nargs="?", help="Resource name")
            a.add_argument("--file", help="JSON or YAML request body (- for stdin)")
            paged = action in {"results", "list"}
            a.add_argument(
                "--offset",
                type=int,
                default=0,
                help="First result" if paged else SUPPRESS,
            )
            a.add_argument("--after", type=int, default=0, help=SUPPRESS)
            a.add_argument(
                "--limit",
                type=int,
                default=100,
                help="Results per page" if paged else SUPPRESS,
            )
            if group == "model":
                a.add_argument(
                    "--local", action="store_true", help="Use gf serve on this machine"
                )
    return root


def _catalog() -> set[str]:
    from generalforecast.models.catalog import entries

    names: set[str] = set()
    for entry in entries():
        names.add(entry.name)
        names.update(getattr(entry, "aliases", ()) or ())
    return names


def _bare(model: str) -> bool:
    return re.fullmatch(r"[a-z0-9][a-z0-9._-]*", model) is not None


def _unknown_model(model: str, pointer: tuple | None = None) -> Problem:
    match = term.suggest(model, _catalog())
    hints = [f"did you mean `{match}`?"] if match else []
    return Problem(
        "ModelNotFound",
        f"unknown model `{model}`",
        hints=[*hints, "run `gf model list` to see available models"],
        pointer=pointer,
    )


def run(argv: list[str] | None = None) -> dict:
    """Dispatches without accepting credential values in argv."""
    global ARGS
    global _ARGV
    _ARGV = sys.argv[1:] if argv is None else list(argv)
    args = parser().parse_args(argv)
    ARGS = args
    out = output()
    if args.group in {"login", "logout", "whoami"}:
        from generalforecast import account

        if args.group == "whoami":
            return account.whoami(args)
        if getattr(args, "no_browser", False):
            os.environ["GF_NO_BROWSER"] = "1"
        return getattr(account, args.group)(args, out)
    if args.group == "init":
        from generalforecast.commands import require
        from generalforecast.packages import PackageSpec

        root = Path(args.directory).absolute()
        if _bare(args.model) and args.model not in _catalog():
            raise _unknown_model(args.model)
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
        require(not root.exists(), f"{args.directory} already exists")
        require(root.parent.is_dir(), "package parent directory does not exist")
        import yaml

        root.mkdir(mode=0o700)
        (root / "gf.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        return {"name": spec["name"], "created": True}
    if args.group == "serve":
        from generalforecast.serve import run as serve

        return serve(args)
    if args.group == "check":
        from generalforecast.commands import read_data, require
        from generalforecast.packages import Package, PackageError

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
                data=data, locked=args.locked, frozen=args.frozen, upgrade=args.upgrade
            )
        assert checked.lock is not None
        return checked.lock.to_dict()
    if args.group in PACKAGE_GROUPS:
        from generalforecast.package_commands import PackageCommands

        return PackageCommands(args).run()
    from generalforecast.commands import execute

    return execute(args)


_SECRET = re.compile(
    r"(?i)(?:gf_(?:live|test|inv)_|hf_|sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{8,}|Bearer\s+\S+"
)


def redact(value):
    """Removes secret response fields and known credential patterns defensively."""
    from generalforecast.credentials import SEEN

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
        values = SEEN | {
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
    """NDJSON with --json; otherwise watch events become status lines."""
    if not JSON and "event" in value:
        from generalforecast.render import event

        event(redact(value), output())
        return
    print(
        json.dumps(redact(value), allow_nan=False, ensure_ascii=False),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


_FIXED = {
    "SessionRequired": "this operation needs a signed-in Google session in the console",
    "ReauthenticationRequired": "sign in again in the console",
    "ExpiredToken": "the login code expired before it was approved",
    "AccessDenied": "the login was denied in the browser",
}


def _service_problem(error: Any) -> Problem:
    from generalforecast.account import host
    from generalforecast.credentials import CONNECTION

    code = (
        error.code
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", error.code)
        else "ClientError"
    )
    message = _FIXED.get(code) or str(error) or code
    server, source = CONNECTION.get("server"), CONNECTION.get("source")
    name = getattr(ARGS, "name", None) if ARGS is not None else None
    group = getattr(ARGS, "group", None) if ARGS is not None else None
    if code == "NotLoggedIn":
        return Problem(code, message, hints=["run `gf login`, or set GF_API_KEY in CI"])
    if code == "Unauthorized":
        if server and source and source != "--local":
            if source == "GF_API_KEY":
                hint = f"create a new key at {server}/#/api-keys, or unset GF_API_KEY to use your login"
            elif source == "profile" or source.isupper():
                hint = f"check the key in {source}, or run `gf login`"
            else:
                hint = "run `gf login` to log in again"
            return Problem(
                code,
                f"{host(server)} rejected the API key from {source}",
                cause=f"401 Unauthorized: {message}",
                hints=[hint],
            )
        return Problem(
            code, "authentication failed", cause=message, hints=["run `gf login`"]
        )
    if code in {"ExpiredToken", "AccessDenied"}:
        return Problem(code, message, hints=["run `gf login` again"])
    if code == "CapacityExceeded":
        return Problem(
            code,
            message,
            hints=[
                f"run it on this machine or your own GPUs: `gf serve {name or '<package>'}`"
            ],
        )
    if code == "TransportError":
        hints = [
            "check your network connection, or set GF_URL for a self-hosted server"
        ]
        if group == "push" and getattr(ARGS, "watch", False):
            hints = [
                "the outcome is unknown; reconnect with `gf push --watch --resume <session>`"
            ]
        return Problem(code, message, hints=hints)
    if code == "Timeout":
        hint = (
            f"it may still finish; check `gf status {name}`"
            if name
            else "retry with a longer --timeout"
        )
        return Problem(code, message, hints=[hint])
    if code == "Conflict":
        return Problem(
            code, message, hints=["state changed; check `gf status` before retrying"]
        )
    if code == "ReadinessFailed":
        return Problem(
            code,
            message,
            hints=[f"inspect `gf logs {name}`" if name else "inspect `gf logs`"],
        )
    if code == "Forbidden":
        return Problem(code, message, hints=["ask a workspace admin for access"])
    if code == "EnvironmentCredential":
        return Problem(code, message, hints=["unset GF_API_KEY to stop using it"])
    if code == "NotFound" and group == "model" and name and _bare(name):
        match = term.suggest(name, _catalog())
        return Problem(
            code, message, hints=[f"did you mean `{match}`?"] if match else []
        )
    return Problem(code, message)


# Top-level gf.yaml keys accepted by the catalog and Python package specs.
SPEC_FIELDS = frozenset(
    {
        *("name", "model", "package_version", "task", "target", "id", "context"),
        *("inputs", "time", "series", "horizon", "frequency", "past_inputs"),
        *("future_inputs", "quantiles", "outputs", "compute", "scaling", "placement"),
    }
)


def _package_problem(error: Any) -> Problem:
    code = (
        error.code
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", error.code)
        else "ClientError"
    )
    path = Path(getattr(ARGS, "path", ".") or ".") / "gf.yaml"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        lines = []
    for number, line in enumerate(lines, 1):
        found = re.match(r"(model:\s*)(\S+)\s*$", line)
        if found and _bare(found.group(2)) and found.group(2) not in _catalog():
            column = len(found.group(1)) + 1
            shown = os.path.relpath(path) if not path.is_absolute() else str(path)
            pointer = (
                shown,
                number,
                column,
                line,
                "not in the catalog",
                len(found.group(2)),
            )
            return _unknown_model(found.group(2), pointer)
    if str(error) == "unknown fields are not supported":
        for number, line in enumerate(lines, 1):
            found = re.match(r"([A-Za-z_][A-Za-z0-9_-]*)\s*:", line)
            if found and found.group(1) not in SPEC_FIELDS:
                field = found.group(1)
                match = term.suggest(field, SPEC_FIELDS)
                shown = os.path.relpath(path) if not path.is_absolute() else str(path)
                return Problem(
                    code,
                    f"unknown field `{field}` in gf.yaml",
                    hints=[f"did you mean `{match}`?"] if match else [],
                    pointer=(shown, number, 1, line, "unknown field", len(field)),
                )
    return Problem(code, str(error))


def describe(error: BaseException) -> Problem:
    """Maps any failure to a Problem; server messages are shown, never replaced."""
    from generalforecast.deployments.contracts import ServiceError
    from generalforecast.packages.validation import PackageError

    verbose = output().verbose
    if isinstance(error, Problem):
        return error
    if isinstance(error, PackageError):
        return _package_problem(error)
    if isinstance(error, ServiceError):
        return _service_problem(error)
    if isinstance(error, FileNotFoundError):
        return Problem("NotFound", f"{error.filename or 'file'} does not exist")
    if isinstance(error, PermissionError):
        return Problem("PermissionDenied", f"permission denied: {error.filename}")
    if isinstance(error, OSError):
        return Problem(
            "ClientError", f"cannot access local state ({error.strerror or error})"
        )
    hints = [] if verbose else ["run with -v for details"]
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return Problem(
            "ClientError", "invalid input or inaccessible local state", hints=hints
        )
    return Problem(
        "ClientError",
        "unexpected client failure; no operation was confirmed",
        hints=hints,
    )


def fail(problem: Problem) -> None:
    """Prints a Problem to stderr in the current mode."""
    if JSON:
        emit({"error": {"code": problem.code, "message": problem.message}}, error=True)
        return
    out = output()
    out.error(redact(problem.message))
    if problem.pointer:
        out.pointer(*problem.pointer)
    if problem.cause:
        out.cause(redact(problem.cause))
    for hint in problem.hints:
        out.hint(redact(hint), "help" if problem.pointer else "hint")


def _trace_requests(out: term.Output) -> None:
    from generalforecast import transport

    original = transport.Transport.send

    def send(self, method, path, body, headers, **options):
        started = time.perf_counter()
        status = "error"
        try:
            result = original(self, method, path, body, headers, **options)
            status = "ok"
            return result
        finally:
            elapsed = term.seconds(time.perf_counter() - started)
            out.debug(
                f"{method} {self.url}{path.split('?', 1)[0]} {status} in {elapsed}"
            )

    transport.Transport.send = send


def main() -> None:
    """Runs one command; expected failures print without tracebacks and exit nonzero."""
    global JSON, _OUT
    argv = sys.argv[1:]
    # Fast paths: neither builds the parser nor imports the SDK.
    if argv in (["-V"], ["--version"]):
        sys.stdout.write(f"gf {version()}\n")
        return
    if argv in (["-h"], ["--help"]):
        sys.stdout.write(root_help(term.Output()))
        return
    flags = set(argv[: argv.index("--")] if "--" in argv else argv)
    JSON = "--json" in flags
    _OUT = term.Output(
        quiet=bool(flags & {"-q", "--quiet"}), verbose=bool(flags & {"-v", "--verbose"})
    )
    if _OUT.verbose:
        _trace_requests(_OUT)
    started = time.perf_counter()
    try:
        result = run()
        if JSON:
            emit(result)
        else:
            from generalforecast.render import render

            render(ARGS, result, _OUT, time.perf_counter() - started)
        _OUT.debug(f"finished in {term.seconds(time.perf_counter() - started)}")
    except KeyboardInterrupt:
        if JSON:
            emit(
                {"error": {"code": "Interrupted", "message": "interrupted"}}, error=True
            )
        else:
            _OUT.say(force=True)
            _OUT.error("interrupted")
        raise SystemExit(130) from None
    except Exception as error:  # noqa: BLE001
        if _OUT.verbose and not isinstance(error, Problem):
            import traceback

            traceback.print_exc()
        fail(describe(error))
        raise SystemExit(1) from None
    finally:
        JSON, _OUT = True, None


if __name__ == "__main__":
    main()
