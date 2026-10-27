# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `gf` command: predictors on this machine and on the hosted service.

Fifteen commands in three groups. Commands print status lines on stderr and
return their result; `generalforecast.render` alone writes results on
stdout, for a person or, under `--json`, as one JSON object. Help and version
print without importing the SDK.

Every failure, including one while writing the result, goes through
`present`, which prints it in one of the three error shapes or, under
`--json`, as the service's error object: `code` is one of the contract's wire
codes and `reason` the finer name, so a local failure reads as a service
reply does. The exit code says how the command ended:

    0    it succeeded
    1    it failed; the error says why
    2    nothing was done: the command line was refused (`ValidationError`,
         reason `usage_error`), or a question was declined or left unanswered
         (`ValidationError`, reason `aborted`)
    130  Ctrl-C interrupted it (`ValidationError`, reason `interrupted`)
    141  the reader of stdout closed the pipe; nothing more is printed
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from argparse import SUPPRESS
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from generalforecast import term
from generalforecast.command_specs import (
    ADMIN_ACTIONS,
    ADMIN_HELP,
    ADMIN_NOUNS,
    ADMIN_PREDICTOR_NOUNS,
    ADMIN_SERVE_ACTIONS,
    CLOSING,
    COMMAND_HELP,
    COMMANDS,
    CONTEXT_ACTIONS,
    DEFAULT_ENV,
    GROUPS,
    MODEL_ACTIONS,
    NODE_CONFIG,
    NODE_DATA,
    READY_SECONDS,
    ROUTER_CONFIG,
    SERVE_HOST,
    SERVE_PORT,
    TABLE_PORT,
    TAGLINE,
)
from generalforecast.errors import Error

DEFAULT_TIMEOUT = 40.0
# The exit codes the module docstring lists, besides 0 and 1.
USAGE_EXIT = 2
ABORTED_EXIT = USAGE_EXIT
INTERRUPTED_EXIT = 130
BROKEN_PIPE_EXIT = 141


def version() -> str:
    """The installed package version."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as installed

    try:
        return installed("generalforecast")
    except PackageNotFoundError:
        return "0.0.0"


# -- parser ---------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """argparse with the contract's help layout and error grammar.

    Flags are never abbreviated: a prefix that works today would change
    meaning the day a flag sharing it is added.
    """

    def __init__(self, *args: Any, add_help: bool = True, **options: Any) -> None:
        super().__init__(*args, add_help=False, allow_abbrev=False, **options)
        if add_help:
            self.add_argument(
                "-h", "--help", action="help", default=SUPPRESS, help="Print help"
            )

    def format_help(self) -> str:
        out = term.Output(out=sys.stdout, err=sys.stdout)
        if self.prog == "gf":
            return root_help(out)
        return command_help(self, out)

    def error(self, message: str) -> NoReturn:
        text, tip, hint = _usage_problem(self, message)
        raise Usage(text, tip=tip, hint=None if tip else hint)


class Usage(Error):
    """A command line the parser rejected; `tip` names a near miss."""

    code = "UsageError"

    def __init__(
        self, message: str, *, tip: str | None = None, hint: str | None = None
    ):
        """Keeps the message, a near-miss tip and the usage hint."""
        super().__init__(message, hint=hint)
        self.tip = tip


_GLOBALS = {"json", "quiet", "verbose", "timeout"}


def _globals(*, root: bool) -> _Parser:
    """The global options, with defaults only on the root parser."""
    parent = _Parser(add_help=False)
    group = parent.add_argument_group("Global options")
    default = (lambda value: value) if root else (lambda value: SUPPRESS)
    group.add_argument(
        "--json",
        action="store_true",
        default=default(False),
        help="Print results and errors as JSON",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=default(False),
        help="Show only results and errors",
    )
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=default(False),
        help="Show requests and the traceback of an internal error",
    )
    group.add_argument(
        "--timeout",
        type=float,
        metavar="<SECONDS>",
        default=default(DEFAULT_TIMEOUT),
        help=f"Seconds to wait for the service (default: {DEFAULT_TIMEOUT:g})",
    )
    return parent


def _early(argv: Sequence[str]) -> argparse.Namespace:
    """`--json`, `-q` and `-v` as the parser reads them, before the full parse.

    A command line the full parser rejects still needs them, to print its
    usage error as JSON or not; this reads only those flags, the same way.
    """
    early = _Parser(add_help=False)
    early.add_argument("--json", action="store_true")
    early.add_argument("-q", "--quiet", action="store_true")
    early.add_argument("-v", "--verbose", action="store_true")
    try:
        return early.parse_known_args(list(argv))[0]
    except Usage:
        return argparse.Namespace(json=False, quiet=False, verbose=False)


def _columns(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _listen(p: argparse.ArgumentParser, *, port: int, what: str) -> None:
    """`--host` and `--port`: where a server listens."""
    p.add_argument(
        "--host",
        default=SERVE_HOST,
        metavar="<HOST>",
        help=f"The address {what} listens on (default: {SERVE_HOST})",
    )
    p.add_argument(
        "--port",
        type=int,
        default=port,
        metavar="<PORT>",
        help=f"The port {what} listens on (default: {port})",
    )


def _offline(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading model weights",
    )


def _config(p: argparse.ArgumentParser, what: str) -> None:
    p.add_argument(
        "--config",
        type=Path,
        metavar="<FILE>",
        required=True,
        help=f"The {what} configuration file",
    )


def _wait(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--wait",
        type=float,
        default=READY_SECONDS,
        metavar="<SECONDS>",
        help=f"Seconds to wait for the environment to be ready (default: {READY_SECONDS:g})",
    )


def _env(
    p: argparse.ArgumentParser, what: str, *, default: str | None = DEFAULT_ENV
) -> None:
    p.add_argument(
        "--env",
        default=default,
        metavar="<E>",
        help=f"{what} (default: {DEFAULT_ENV})",
    )


def parser() -> _Parser:
    """The full command tree."""
    root = _Parser(prog="gf", description=TAGLINE, parents=[_globals(root=True)])
    root.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"gf {version()}",
        help="Print version",
    )
    subs = root.add_subparsers(
        dest="command", metavar="<COMMAND>", parser_class=_Parser
    )
    shared = [_globals(root=False)]

    def command(name: str, *, help: str | None = None) -> _Parser:
        text = help or COMMAND_HELP[name]
        return subs.add_parser(name, help=text, description=text, parents=shared)

    def nested(p: _Parser, metavar: str) -> argparse._SubParsersAction:
        dest = "noun" if metavar == "<NOUN>" else "action"
        return p.add_subparsers(dest=dest, metavar=metavar, parser_class=_Parser)

    def action(group: argparse._SubParsersAction, name: str, text: str) -> _Parser:
        return group.add_parser(name, help=text, description=text, parents=shared)

    def directory(
        p: _Parser, *, what: str = "The predictor directory (default: .)"
    ) -> None:
        p.add_argument("directory", nargs="?", default=".", metavar="DIR", help=what)

    def reference(
        p: _Parser, *, metavar: str = "DIR|NAME", what: str | None = None
    ) -> None:
        p.add_argument(
            "target",
            nargs="?",
            default=".",
            metavar=metavar,
            help=what or "A predictor directory or a hosted name (default: .)",
        )

    p = command("init")
    directory(p, what="Where to write gf.yaml (default: .)")
    p.add_argument("--target", metavar="<C>", help="The column to predict")
    p.add_argument(
        "--model", metavar="<M>", help="The catalog model, general-forecast/<name>"
    )
    p.add_argument(
        "--task",
        metavar="<T>",
        choices=["classification", "regression", "forecast"],
        help="classification, regression or forecast (default: from the target)",
    )
    p.add_argument("--id", metavar="<C>", help="The column that identifies a row")
    p.add_argument(
        "--inputs",
        metavar="<C,C>",
        type=_columns,
        help="The columns the model may read",
    )
    p.add_argument(
        "--exclude",
        metavar="<C,C>",
        type=_columns,
        help="Columns to remove from the inferred inputs",
    )
    p.add_argument("--time", metavar="<C>", help="The time column of a forecast")
    p.add_argument(
        "--series",
        metavar="<C,C>",
        type=_columns,
        help="The columns that name a series",
    )
    p.add_argument(
        "--frequency", metavar="<F>", help="The cadence of a forecast, like 1d or 1h"
    )
    p.add_argument(
        "--horizon", metavar="<N>", type=int, help="How many steps a forecast predicts"
    )
    p.add_argument(
        "--context",
        metavar="<FILE>",
        help="The context rows (default: the one table here)",
    )
    split = p.add_mutually_exclusive_group()
    split.add_argument(
        "--until", metavar="<DATE>", help="Rows at or after this date are queries"
    )
    split.add_argument(
        "--holdout",
        metavar="<FRACTION>",
        type=float,
        help="Hold out this fraction of rows as queries",
    )
    split.add_argument(
        "--queries",
        choices=["none"],
        metavar="none",
        help="No query rows: every row is context",
    )
    p.add_argument(
        "--yes", action="store_true", help="Take the first option of every prompt"
    )
    p.add_argument("--no-input", action="store_true", help="Fail instead of prompting")
    p.add_argument("--force", action="store_true", help="Overwrite an existing gf.yaml")

    p = command("check")
    directory(p)
    p.add_argument("--locked", action="store_true", help="Fail if gf.lock would change")
    p.add_argument(
        "--frozen", action="store_true", help="Trust gf.lock without resolving"
    )
    p.add_argument(
        "--online",
        action="store_true",
        help="Resolve the model against the catalog service",
    )
    p.add_argument(
        "--upgrade", action="store_true", help="Move the model pin to the newest digest"
    )

    p = command("eval")
    directory(p)
    p.add_argument(
        "--queries", metavar="<FILE>", help="Rows to score instead of the held-out rows"
    )
    p.add_argument(
        "--model",
        metavar="<M>",
        action="append",
        default=[],
        help="Score with another model too (repeatable)",
    )
    p.add_argument(
        "--output",
        metavar="<FILE>",
        help="Write id, target and prediction columns here",
    )

    p = command("serve")
    reference(
        p,
        metavar="DIR|MODEL",
        what="A predictor directory, its gf.lock, or a catalog model (default: .)",
    )
    _listen(p, port=SERVE_PORT, what="the Models API")
    _offline(p)

    p = command("predict")
    reference(
        p,
        what="A predictor directory, a served URL, a catalog model or a hosted name "
        "(default: .)",
    )
    p.add_argument(
        "--queries", metavar="<FILE>", required=True, help="The rows to score"
    )
    p.add_argument(
        "--context",
        metavar="<FILE>",
        help="Context rows for a predictor whose context is per request",
    )
    p.add_argument(
        "--target",
        dest="target_column",
        metavar="<C>",
        help="The column to predict; a catalog model needs it",
    )
    p.add_argument(
        "--inputs",
        metavar="<C,C>",
        type=_columns,
        help="The columns the model may read",
    )
    _env(p, "The hosted environment, for a hosted name", default=None)
    p.add_argument(
        "--output",
        metavar="<FILE>",
        help="Write the scored rows here instead of stdout",
    )

    for name in ("login", "logout", "whoami"):
        command(name)

    p = command("push")
    directory(p)
    _env(p, "The environment to deploy to")
    _wait(p)
    p.add_argument("--yes", action="store_true", help="Push without asking")

    p = command("status")
    reference(p)

    p = command("promote")
    reference(p)
    p.add_argument(
        "--from",
        dest="source",
        default="staging",
        metavar="<E>",
        help="The environment whose revision to take (default: staging)",
    )
    p.add_argument(
        "--to",
        dest="env",
        default=DEFAULT_ENV,
        metavar="<E>",
        help=f"The environment to point (default: {DEFAULT_ENV})",
    )
    _wait(p)

    p = command("rollback")
    reference(p)
    _env(p, "The environment to point")
    p.add_argument(
        "--revision",
        type=int,
        metavar="<N>",
        help="The revision to point at (default: the one before)",
    )
    _wait(p)

    model = nested(command("model"), "<COMMAND>")
    for name, text in MODEL_ACTIONS:
        q = action(model, name, text)
        if name == "list":
            q.add_argument(
                "--task",
                choices=["classification", "regression", "forecast"],
                help="Show only models for this task",
            )
        else:
            q.add_argument(
                "model",
                metavar="MODEL",
                help="A catalog model, general-forecast/<name>",
            )

    context = nested(command("context"), "<COMMAND>")
    for name, text in CONTEXT_ACTIONS:
        q = action(context, name, text)
        q.add_argument("name", metavar="NAME", help="The live context")
        if name == "create":
            q.add_argument(
                "--file",
                metavar="<FILE>",
                required=True,
                help="The context config, JSON or YAML",
            )
        elif name in ("append", "label"):
            q.add_argument(
                "file", metavar="FILE", help="The rows, as .csv, .parquet or .json"
            )
            q.add_argument(
                "--id",
                metavar="<C>",
                required=True,
                help="The column that identifies a row",
            )
            if name == "label":
                q.add_argument(
                    "--target",
                    metavar="<C>",
                    required=True,
                    help="The column holding the target values",
                )
            q.add_argument(
                "--write-id",
                metavar="<ID>",
                help="Make the write idempotent under this id",
            )

    nouns = nested(command("admin"), "<NOUN>")
    for noun in ADMIN_NOUNS:
        n = action(nouns, noun, ADMIN_HELP[noun])
        if noun in ADMIN_ACTIONS:
            acts = nested(n, "<ACTION>")
            for spec in ADMIN_ACTIONS[noun]:
                a = action(acts, spec.name, spec.help)
                if spec.target is not None:
                    a.add_argument("name", metavar="NAME", help=f"The {spec.target}")
                if spec.body:
                    a.add_argument(
                        "--file",
                        metavar="<FILE>",
                        required=True,
                        help="The request, as JSON or YAML; - reads stdin",
                    )
                if spec.destructive:
                    a.add_argument(
                        "--yes", action="store_true", help="Go ahead without asking"
                    )
                if noun == "audit":
                    a.add_argument(
                        "--after",
                        type=int,
                        default=0,
                        metavar="<N>",
                        help="Start after this event (default: 0)",
                    )
                    a.add_argument(
                        "--limit",
                        type=int,
                        default=100,
                        metavar="<N>",
                        help="Show at most this many events, up to 100 (default: 100)",
                    )
        elif noun in ADMIN_SERVE_ACTIONS:
            acts = nested(n, "<ACTION>")
            config = NODE_CONFIG if noun == "node" else ROUTER_CONFIG
            for name, text in ADMIN_SERVE_ACTIONS[noun]:
                a = action(acts, name, text)
                if name != "init":
                    _config(a, f"{noun}'s")
                    if name == "serve" and noun == "node":
                        _offline(a)
                    continue
                a.add_argument(
                    "--path",
                    type=Path,
                    default=Path(config),
                    metavar="<FILE>",
                    help=f"Where to write the configuration (default: {config})",
                )
                _listen(a, port=TABLE_PORT, what=f"the {noun}")
                if noun == "node":
                    a.add_argument(
                        "--data",
                        type=Path,
                        default=Path(NODE_DATA),
                        metavar="<DIR>",
                        help=f"Where the node keeps its tables (default: {NODE_DATA})",
                    )
                    a.add_argument(
                        "--model",
                        action="append",
                        default=[],
                        metavar="<NAME=MODEL>",
                        help="A model to serve, as NAME=general-forecast/<name> "
                        "(repeatable; default: the TabICL models)",
                    )
                else:
                    a.add_argument(
                        "--node",
                        action="append",
                        required=True,
                        metavar="<URL>",
                        help="A node's address, like http://127.0.0.1:8816 (repeatable)",
                    )
                a.add_argument(
                    "--force", action="store_true", help="Overwrite an existing file"
                )
        elif noun == "delete":
            reference(n)
            n.add_argument("--yes", action="store_true", help="Delete without asking")
        else:
            reference(
                n,
                metavar="DIR" if noun == "diff" else "DIR|NAME",
                what="The predictor directory (default: .)" if noun == "diff" else None,
            )
            _env(
                n,
                "The environment to compare with"
                if noun == "diff"
                else "The environment",
            )

    p = command("help", help="Print help for a command")
    p.add_argument(
        "topic", nargs="*", metavar="COMMAND", help="The command to describe"
    )
    return root


# -- help -----------------------------------------------------------------


def root_help(out: term.Output) -> str:
    """`gf --help`: the three groups, the global options and the closing line."""
    width = max(len(name) for name in COMMANDS) + 3
    lines = [TAGLINE, "", f"{out.style('Usage:', 'bold', 'green')} gf <COMMAND>", ""]
    for title, names in GROUPS:
        lines.append(out.style(f"{title}:", "bold", "green"))
        lines.extend(
            f"  {out.style(name.ljust(width), 'bold', 'cyan')}{COMMAND_HELP[name]}"
            for name in names
        )
        lines.append("")
    lines.append(out.style("Options:", "bold", "green"))
    lines.extend(_option_lines(parser(), out, globals_only=True))
    lines.extend(["", CLOSING, ""])
    return "\n".join(lines)


def _flag_text(action: argparse.Action) -> str:
    shorts = [s for s in action.option_strings if not s.startswith("--")]
    longs = [s for s in action.option_strings if s.startswith("--")]
    head = ", ".join(shorts + longs) if shorts else "    " + ", ".join(longs)
    if action.nargs != 0 and not isinstance(action, argparse._VersionAction):
        head += f" {action.metavar or '<' + action.dest.upper() + '>'}"
    return head


def _is_global(action: argparse.Action) -> bool:
    return action.dest in _GLOBALS


def _option_lines(
    p: argparse.ArgumentParser, out: term.Output, *, globals_only: bool
) -> list[str]:
    actions = [
        a
        for a in p._actions
        if a.option_strings
        and a.help is not SUPPRESS
        and (
            _is_global(a)
            or isinstance(a, (argparse._HelpAction, argparse._VersionAction))
        )
        == globals_only
    ]
    if not actions:
        return []
    heads = [_flag_text(a) for a in actions]
    width = max(len(head) for head in heads) + 2
    return [
        f"  {out.style(head.ljust(width), 'bold', 'cyan')}{a.help or ''}".rstrip()
        for head, a in zip(heads, actions, strict=True)
    ]


def _subparsers(p: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    return next(
        (a for a in p._actions if isinstance(a, argparse._SubParsersAction)), None
    )


def _positionals(p: argparse.ArgumentParser) -> list[argparse.Action]:
    return [
        a
        for a in p._actions
        if not a.option_strings and not isinstance(a, argparse._SubParsersAction)
    ]


def _positional_text(action: argparse.Action) -> str:
    name = action.metavar or action.dest.upper()
    if action.nargs in ("?", "*"):
        return f"[{name}]"
    return f"<{name}>"


def usage(p: argparse.ArgumentParser) -> str:
    """One line: the program, `[OPTIONS]` when it has any, then its positionals."""
    parts: list[str] = [p.prog]
    options = [
        a
        for a in p._actions
        if a.option_strings
        and not _is_global(a)
        and not isinstance(a, argparse._HelpAction)
    ]
    required = [a for a in options if a.required]
    if options:
        parts.append("[OPTIONS]")
    parts.extend(_flag_text(a).strip() for a in required)
    subs = _subparsers(p)
    if subs is not None:
        parts.append(str(subs.metavar or "<COMMAND>"))
    parts.extend(_positional_text(a) for a in _positionals(p))
    return " ".join(parts)


def command_help(p: argparse.ArgumentParser, out: term.Output) -> str:
    """`gf <command> --help`: description, usage, commands, arguments, options."""
    lines = [
        p.description or "",
        "",
        f"{out.style('Usage:', 'bold', 'green')} {usage(p)}",
        "",
    ]
    subs = _subparsers(p)
    if subs is not None:
        title = (
            "Actions"
            if subs.metavar == "<ACTION>"
            else ("Nouns" if subs.metavar == "<NOUN>" else "Commands")
        )
        lines.append(out.style(f"{title}:", "bold", "green"))
        width = max(len(c.dest) for c in subs._choices_actions) + 3
        lines.extend(
            f"  {out.style(c.dest.ljust(width), 'bold', 'cyan')}{c.help}"
            for c in subs._choices_actions
        )
        lines.append("")
    positionals = _positionals(p)
    if positionals:
        lines.append(out.style("Arguments:", "bold", "green"))
        heads = [_positional_text(a) for a in positionals]
        width = max(len(h) for h in heads) + 2
        lines.extend(
            f"  {out.style(h.ljust(width), 'bold', 'cyan')}{a.help or ''}".rstrip()
            for h, a in zip(heads, positionals, strict=True)
        )
        lines.append("")
    options = _option_lines(p, out, globals_only=False)
    if options:
        lines.append(out.style("Options:", "bold", "green"))
        lines.extend(options)
        lines.append("")
    lines.append(out.style("Global options:", "bold", "green"))
    lines.extend(_option_lines(p, out, globals_only=True))
    lines.extend(["", CLOSING, ""])
    return "\n".join(lines)


def _find(
    p: argparse.ArgumentParser, words: Sequence[str]
) -> argparse.ArgumentParser | None:
    for word in words:
        subs = _subparsers(p)
        if subs is None or word not in subs.choices:
            return None
        p = subs.choices[word]
    return p


# -- usage errors ---------------------------------------------------------


def _choices(p: argparse.ArgumentParser) -> list[str]:
    subs = _subparsers(p)
    return sorted(subs.choices) if subs is not None else []


def _flags(p: argparse.ArgumentParser) -> list[str]:
    return [s for a in p._actions for s in a.option_strings]


def _usage_problem(
    p: argparse.ArgumentParser, message: str
) -> tuple[str, str | None, str | None]:
    """Argparse's message as `(error, tip, hint)`, never echoing a secret-looking value."""
    argv = _ARGV
    usage_hint = (
        "run 'gf --help' for the list of commands"
        if p.prog == "gf"
        else f"run '{p.prog} --help' for usage"
    )
    choice = re.search(r"invalid choice: '([^']*)'", message)
    if choice:
        word, names = choice.group(1), _choices(p)
        prior = argv[: argv.index(word)] if word in argv else []
        if prior and prior[-1].startswith("-") and prior[-1] not in _flags(p):
            flag = prior[-1].split("=", 1)[0]
            match = term.suggest(flag, _flags(p))
            return (
                f"unexpected argument '{flag}'",
                match and f"a similar argument exists: '{match}'",
                usage_hint,
            )
        match = term.suggest(word, names)
        shown = word if re.fullmatch(r"[\w.-]{1,40}", word) else "..."
        subs = _subparsers(p)
        noun = (
            "command"
            if p.prog == "gf"
            else "noun"
            if subs is not None and subs.metavar == "<NOUN>"
            else "action"
        )
        text = (
            f"unknown {noun} `{shown}`"
            if p.prog == "gf"
            else f"unknown {noun} `{shown}` for `{p.prog}`"
        )
        return text, match and f"a similar {noun} exists: '{match}'", usage_hint
    extra = re.match(r"unrecognized arguments: (.*)", message)
    if extra:
        first = extra.group(1).split()[0]
        words = [w for w in argv if not w.startswith("-")]
        deepest = p
        for n in range(len(words), 0, -1):
            found = _find(p, words[:n])
            if found is not None:
                deepest = found
                break
        if deepest is not p:
            usage_hint = f"run '{deepest.prog} --help' for usage"
        if first.startswith("-"):
            flag = first.split("=", 1)[0]
            match = term.suggest(flag, _flags(deepest))
            return (
                f"unexpected argument '{flag}'",
                match and f"a similar argument exists: '{match}'",
                usage_hint,
            )
        shown = first if re.fullmatch(r"[\w./-]{1,40}", first) else "..."
        return f"unexpected argument `{shown}`", None, usage_hint
    required = re.match(r"the following arguments are required: (.*)", message)
    if required:
        names = [n.strip() for n in required.group(1).split(",")]
        if names and names[0] in ("<COMMAND>", "<ACTION>", "<NOUN>"):
            kind = names[0].strip("<>").lower()
            return (
                f"`{p.prog}` needs {'an' if kind == 'action' else 'a'} {kind}",
                None,
                f"choose one of: {', '.join(_choices(p))}",
            )
        shown = []
        for name in names:
            action = next(
                (
                    a
                    for a in p._actions
                    if name in a.option_strings or a.dest == name or a.metavar == name
                ),
                None,
            )
            shown.append(
                _flag_text(action).strip()
                if action and action.option_strings
                else f"<{name}>"
            )
        return (
            "the following required arguments were not provided:\n" + "\n".join(shown),
            None,
            usage_hint,
        )
    exclusive = re.match(r"argument ([^:]+): not allowed with argument (.+)", message)
    if exclusive:
        first, second = (
            exclusive.group(1).split("/")[-1],
            exclusive.group(2).split("/")[-1],
        )
        return (
            f"the argument '{first}' cannot be used with '{second}'",
            None,
            usage_hint,
        )
    invalid = re.match(r"argument ([^:]+): invalid (\w+) value: '([^']*)'", message)
    if invalid:
        flag, kind, value = (
            invalid.group(1).split("/")[-1],
            invalid.group(2),
            invalid.group(3),
        )
        expected = {
            "int": "an integer",
            "float": "a number",
            "_columns": "comma-separated column names",
        }.get(kind, kind)
        shown = value if re.fullmatch(r"[\w.,-]{1,40}", value) else "..."
        return (
            f"invalid value '{shown}' for '{flag}': expected {expected}",
            None,
            usage_hint,
        )
    expected = re.match(r"argument ([^:]+): expected one argument", message)
    if expected:
        flag = expected.group(1).split("/")[-1]
        return (
            f"a value is required for '{flag}' but none was supplied",
            None,
            usage_hint,
        )
    return message, None, usage_hint


# -- errors ---------------------------------------------------------------


def error_object(error: BaseException) -> dict[str, Any]:
    """The `--json` error object, shaped as the service's.

    `code` is one of the contract's wire codes, with the specific name in
    `reason`, so a local failure and a service reply read the same way.
    """
    from generalforecast.account import wire_code

    code, reason = wire_code(error)
    message = getattr(error, "message", None) or str(error) or "internal error"
    body: dict[str, Any] = {"code": code, "message": message}
    if reason is not None:
        body["reason"] = reason
    for field in ("param", "because", "hint", "tip", "limit", "actual", "request_id"):
        value = getattr(error, field, None)
        if value is not None:
            body[field] = value
    return {"error": body}


def _pointer(error: BaseException, root: Path) -> term.Pointer | None:
    """The `-->` block for an error that carries a line and column into a file.

    A relative path is the manifest's, so it is read from the predictor
    directory the command named.
    """
    line, column = getattr(error, "line", None), getattr(error, "column", None)
    if not line or not column:
        return None
    path = getattr(error, "path", None) or "gf.yaml"
    source = ""
    try:
        source = (root / path).expanduser().read_text().splitlines()[line - 1]
    except (OSError, IndexError, UnicodeDecodeError):
        pass
    match = re.match(r"[^\s,\]\}#]+", source[column - 1 :]) if source else None
    width = term.visible(match.group(0)) if match else 1
    label = getattr(error, "because", None) or ""
    return term.Pointer(path, line, column, source, label, width)


def _oserror(error: OSError) -> Error:
    from generalforecast.errors import InvalidInput, NotFound

    filename = getattr(error, "filename", None)
    reason = error.strerror or str(error) or type(error).__name__
    reason = reason[:1].lower() + reason[1:]
    if filename is not None:
        kind = NotFound if isinstance(error, FileNotFoundError) else InvalidInput
        return kind(f"cannot open `{filename}`: {reason}")
    return InvalidInput(reason)


def _known(error: BaseException) -> bool:
    """Whether a failure says what went wrong, rather than being our bug."""
    return isinstance(error, Error) or (
        isinstance(getattr(error, "code", None), str)
        and isinstance(getattr(error, "message", None), str)
    )


def _silence_stdout() -> None:
    """Points stdout at /dev/null, so the interpreter's last flush cannot fail."""
    try:
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, sys.stdout.fileno())
        os.close(null)
    except (OSError, ValueError, AttributeError):
        pass


def present(
    error: BaseException,
    out: term.Output,
    *,
    json: bool = False,
    root: Path = Path("."),
) -> int:
    """Prints any failure in the contract's grammar and returns its exit code.

    The module docstring lists the exit codes and the wire codes and reasons
    of the failures that are not errors of their own: a usage error, a
    declined question and Ctrl-C.
    """
    from generalforecast.render import emit, redact

    if isinstance(error, BrokenPipeError):
        _silence_stdout()
        return BROKEN_PIPE_EXIT
    exit_code = 1
    if isinstance(error, KeyboardInterrupt):
        error, exit_code = Error("interrupted", code="Interrupted"), INTERRUPTED_EXIT
        # Ctrl-C cuts a line short, such as a question's: the error, or its
        # JSON object, starts a line of its own.
        out.say(force=True)
    elif isinstance(error, Usage):
        exit_code = USAGE_EXIT
    elif isinstance(error, term.Aborted):
        exit_code = ABORTED_EXIT
    elif isinstance(error, OSError) and not _known(error):
        error = _oserror(error)
    if not _known(error):
        if out.verbose:
            import traceback

            out.say("".join(traceback.format_exception(error)).rstrip(), force=True)
        error = Error("internal error", code="Internal", hint="run again with -v")
    try:
        if json:
            emit(error_object(error), error=True)
            return exit_code
        hint = getattr(error, "hint", None)
        tip = getattr(error, "tip", None)
        pointer = _pointer(error, root)
        because = getattr(error, "because", None)
        limit, actual = getattr(error, "limit", None), getattr(error, "actual", None)
        if because is None and limit is not None and actual is not None:
            because = (
                f"the limit is {term.count(limit)}; "
                f"the request sent {term.count(actual)}"
            )
        if hint is None:
            hint = _unauthorized_hint(error)
        out.error(
            redact(str(getattr(error, "message", error))),
            hint=None if tip else hint,
            because=None if pointer is not None else because,
            tip=tip,
            pointer=pointer,
            request_id=getattr(error, "request_id", None),
        )
    except BrokenPipeError:
        return BROKEN_PIPE_EXIT
    except (OSError, AttributeError):
        # stderr is closed or gone; the exit code still says what happened.
        pass
    return exit_code


def _unauthorized_hint(error: BaseException) -> str | None:
    """What to do about a refused credential, from the contract's code and reason."""
    from generalforecast.account import wire_code

    code, reason = wire_code(error)
    if code != "Unauthorized":
        return None
    if reason in (
        "not_logged_in",
        "expired_token",
        "invalid_login",
        "credential_inactive",
    ):
        return "run `gf login`"
    message = str(getattr(error, "message", ""))
    if reason in ("forbidden", "access_denied") or message.startswith(
        "permission denied"
    ):
        return "run `gf login` again, or use a key that has this permission"
    return None


# -- dispatch -------------------------------------------------------------

# The command line being parsed, for the near-miss tips of usage errors.
_ARGV: list[str] = []


def _incomplete(root: _Parser, args: argparse.Namespace) -> None:
    """Rejects a group named without its action, as argparse leaves it to us."""
    path = [args.command]
    if args.command in ("model", "context"):
        path.append(args.action)
    elif args.command == "admin":
        path.append(args.noun)
        if args.noun is not None and args.noun not in ADMIN_PREDICTOR_NOUNS:
            path.append(args.action)
    if None in path:
        known = path[: path.index(None)]
        sub = _find(root, known)
        assert sub is not None
        nested = _subparsers(sub)
        assert nested is not None
        sub.error(f"the following arguments are required: {nested.metavar}")


def run(args: argparse.Namespace, root: _Parser, out: term.Output) -> dict | None:
    """Runs one parsed command and returns its result."""
    if args.command is None:
        out.out.write(root_help(term.Output(out=out.out, err=out.out)))
        return None
    if args.command == "help":
        topic = _find(root, args.topic)
        if topic is None:
            if args.topic:
                match = term.suggest(args.topic[0], COMMANDS)
                raise Usage(
                    f"unknown command `{args.topic[0]}`",
                    tip=match and f"a similar command exists: '{match}'",
                    hint=None if match else "run 'gf --help' for the list of commands",
                )
            topic = root
        out.out.write(topic.format_help())
        return None
    _incomplete(root, args)
    if args.command in ("login", "logout", "whoami"):
        from generalforecast import account

        if args.command == "whoami":
            return account.whoami(args)
        return getattr(account, args.command)(args, out)
    if args.command in ("model", "context", "admin"):
        from generalforecast.commands import execute

        return execute(args, out)
    from generalforecast.predictor_commands import execute as predictors

    return predictors(args, out)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs one command; failures print in the error grammar and exit nonzero."""
    global _ARGV
    _ARGV = list(sys.argv[1:] if argv is None else argv)
    flags = _early(_ARGV)
    json = flags.json
    out = term.Output(quiet=flags.json or flags.quiet, verbose=flags.verbose)
    args: argparse.Namespace | None = None
    try:
        root = parser()
        args = root.parse_args(_ARGV)
        json = args.json
        out = term.Output(quiet=args.json or args.quiet, verbose=args.verbose)
        result = run(args, root, out)
        if result is not None:
            from generalforecast.render import emit, render

            if json:
                emit(result)
            else:
                render(args, result, out)
        sys.stdout.flush()
    except SystemExit:
        # Help and version print through argparse, which ignores a closed
        # pipe; the flush finds it.
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            _silence_stdout()
            raise SystemExit(BROKEN_PIPE_EXIT) from None
        raise
    except BaseException as error:  # noqa: BLE001
        # An error's file is the manifest of the predictor the command named.
        named = vars(args) if args is not None else {}
        where = named.get("directory") or named.get("target") or "."
        manifests = Path(where) if "://" not in where else Path(".")
        raise SystemExit(present(error, out, json=json, root=manifests)) from None


if __name__ == "__main__":
    main()
