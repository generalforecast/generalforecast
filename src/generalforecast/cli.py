# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `gf` command: predictors on this machine and on the hosted service.

Twelve commands in three groups plus `admin`. Human output follows one
grammar (status lines, lock changes, three error shapes) drawn by
`generalforecast.term`; `--json` prints results and errors as JSON objects
instead. Help and version print without importing the SDK.
"""

from __future__ import annotations

import argparse
import json
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
    CLOSING,
    COMMAND_HELP,
    COMMANDS,
    CONTEXT_ACTIONS,
    GROUPS,
    MODEL_ACTIONS,
    TAGLINE,
    UNNAMED_ACTIONS,
)

JSON = False
ARGS: argparse.Namespace | None = None
_OUT: term.Output | None = None
_ARGV: list[str] = []
DEFAULT_TIMEOUT = 40.0
DEFAULT_ENV = "production"

try:  # contract: provided by session D
    from generalforecast.errors import Error
except ImportError:

    class Error(Exception):  # type: ignore[no-redef]
        """A failure the CLI shows in one of its three shapes.

        `line` and `column` point into `path` (default `gf.yaml`) for shape
        two; `because`, `limit` and `actual` make shape three.
        """

        def __init__(
            self,
            code: str,
            message: str,
            *,
            param: str | None = None,
            hint: str | None = None,
            limit: int | None = None,
            actual: int | None = None,
            request_id: str | None = None,
            because: str | None = None,
            line: int | None = None,
            column: int | None = None,
            path: str | None = None,
        ) -> None:
            """Keeps every field the error object and the renderer read."""
            super().__init__(message)
            self.code, self.message = code, message
            self.param, self.hint = param, hint
            self.limit, self.actual = limit, actual
            self.request_id, self.because = request_id, because
            self.line, self.column, self.path = line, column, path

        def __str__(self) -> str:
            """The CLI's error shape without colour."""
            lines = [f"error: {self.message}"]
            if self.because:
                lines.append(f"  because: {self.because}")
            if self.hint:
                lines.append(f"  hint: {self.hint}")
            if self.request_id:
                lines.append(f"  request {self.request_id}")
            return "\n".join(lines)


def version() -> str:
    """The installed package version."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as installed

    try:
        return installed("generalforecast")
    except PackageNotFoundError:
        return "0.0.0"


def output() -> term.Output:
    """The current invocation's renderer."""
    return _OUT if _OUT is not None else term.Output()


# -- parser ---------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """argparse with the contract's help layout and error grammar."""

    def __init__(self, *args: Any, add_help: bool = True, **options: Any) -> None:
        super().__init__(*args, add_help=False, **options)
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
        raise Usage(text, tip=tip, hint=hint, prog=self.prog)


class Usage(Exception):
    """A command line the parser rejected."""

    def __init__(self, message: str, *, tip: str | None, hint: str | None, prog: str):
        """Keeps the message, a near-miss tip and the usage hint."""
        super().__init__(message)
        self.message, self.tip, self.hint, self.prog = message, tip, hint, prog


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


def _columns(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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

    def directory(
        p: _Parser, *, what: str = "The predictor directory (default: .)"
    ) -> None:
        p.add_argument("directory", nargs="?", default=".", metavar="DIR", help=what)

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
    p.add_argument(
        "target",
        nargs="?",
        default=".",
        metavar="DIR|MODEL",
        help="A predictor directory or a catalog model (default: .)",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="<HOST>",
        help="The address to listen on (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="<PORT>",
        help="The port to listen on (default: 8000)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading model weights",
    )

    p = command("predict")
    p.add_argument(
        "target",
        nargs="?",
        default=".",
        metavar="DIR|NAME",
        help="A predictor directory, a served URL or a hosted name (default: .)",
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
        help="The column to predict, for a bare catalog model",
    )
    p.add_argument(
        "--inputs",
        metavar="<C,C>",
        type=_columns,
        help="The columns the model may read, for a bare catalog model",
    )
    p.add_argument(
        "--env",
        default=DEFAULT_ENV,
        metavar="<E>",
        help=f"The hosted environment (default: {DEFAULT_ENV})",
    )
    p.add_argument(
        "--output",
        metavar="<FILE>",
        help="Write the scored rows here instead of stdout",
    )

    for name in ("login", "logout", "whoami"):
        command(name)

    p = command("push")
    directory(p)
    p.add_argument(
        "--env",
        default=DEFAULT_ENV,
        metavar="<E>",
        help=f"The environment to deploy to (default: {DEFAULT_ENV})",
    )
    p.add_argument("--yes", action="store_true", help="Push without asking")

    p = command("status")
    p.add_argument(
        "target",
        nargs="?",
        default=".",
        metavar="DIR|NAME",
        help="A predictor directory or a hosted name (default: .)",
    )

    p = command("promote")
    p.add_argument(
        "target",
        nargs="?",
        default=".",
        metavar="DIR|NAME",
        help="A predictor directory or a hosted name (default: .)",
    )
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

    p = command("rollback")
    p.add_argument(
        "target",
        nargs="?",
        default=".",
        metavar="DIR|NAME",
        help="A predictor directory or a hosted name (default: .)",
    )
    p.add_argument(
        "--env",
        default=DEFAULT_ENV,
        metavar="<E>",
        help=f"The environment to point (default: {DEFAULT_ENV})",
    )
    p.add_argument(
        "--revision",
        type=int,
        metavar="<N>",
        help="The revision to point at (default: the one before)",
    )

    model = command("model")
    actions = model.add_subparsers(
        dest="action", metavar="<COMMAND>", parser_class=_Parser
    )
    for action, text in MODEL_ACTIONS:
        q = actions.add_parser(action, help=text, description=text, parents=shared)
        if action == "list":
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

    context = command("context")
    actions = context.add_subparsers(
        dest="action", metavar="<COMMAND>", parser_class=_Parser
    )
    for action, text in CONTEXT_ACTIONS:
        q = actions.add_parser(action, help=text, description=text, parents=shared)
        q.add_argument("name", metavar="NAME", help="The live context")
        if action == "create":
            q.add_argument(
                "--file",
                metavar="<FILE>",
                required=True,
                help="The context config, JSON or YAML",
            )
        elif action in ("append", "label"):
            q.add_argument(
                "file", metavar="FILE", help="The rows, as .csv, .parquet or .json"
            )
            q.add_argument(
                "--id",
                metavar="<C>",
                required=True,
                help="The column that identifies a row",
            )
            if action == "label":
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

    admin = command("admin")
    nouns = admin.add_subparsers(dest="noun", metavar="<NOUN>", parser_class=_Parser)
    for noun in ADMIN_NOUNS:
        text = ADMIN_HELP[noun]
        n = nouns.add_parser(noun, help=text, description=text, parents=shared)
        if noun in ADMIN_ACTIONS:
            acts = n.add_subparsers(
                dest="action", metavar="<ACTION>", parser_class=_Parser
            )
            for action, line in ADMIN_ACTIONS[noun]:
                a = acts.add_parser(action, help=line, description=line, parents=shared)
                if (
                    action not in UNNAMED_ACTIONS or noun == "runtimes"
                ) and noun != "sso":
                    a.add_argument("name", metavar="NAME", help=f"The {noun}")
                a.add_argument("--file", metavar="<FILE>", help="A JSON or YAML body")
                a.add_argument(
                    "--yes", action="store_true", help="Confirm a destructive action"
                )
        elif noun == "node":
            acts = n.add_subparsers(
                dest="action", metavar="<ACTION>", parser_class=_Parser
            )
            a = acts.add_parser(
                "serve",
                help="Serve one node of a cluster",
                description="Serve one node of a cluster",
                parents=shared,
            )
            a.add_argument(
                "target",
                nargs="?",
                default=".",
                metavar="DIR|MODEL",
                help="A predictor directory or a catalog model (default: .)",
            )
            a.add_argument(
                "--node",
                action="append",
                metavar="<NAME>",
                help="A node of the cluster (repeatable)",
            )
            a.add_argument(
                "--config",
                type=Path,
                metavar="<FILE>",
                help="Run from this configuration file",
            )
            a.add_argument(
                "--init",
                type=Path,
                metavar="<FILE>",
                help="Write a configuration file here and exit",
            )
            a.add_argument(
                "--force",
                action="store_true",
                help="Let --init overwrite an existing file",
            )
            a.add_argument(
                "--check",
                action="store_true",
                help="Validate the configuration and exit",
            )
            a.add_argument(
                "--request-timeout",
                type=float,
                default=60.0,
                metavar="<SECONDS>",
                help="Seconds a request may take (default: 60)",
            )
            a.add_argument(
                "--host",
                default="127.0.0.1",
                metavar="<HOST>",
                help="The address to listen on (default: 127.0.0.1)",
            )
            a.add_argument(
                "--port",
                type=int,
                default=8000,
                metavar="<PORT>",
                help="The port to listen on (default: 8000)",
            )
            a.add_argument(
                "--offline",
                action="store_true",
                help="Fail instead of downloading model weights",
            )
        elif noun == "router":
            acts = n.add_subparsers(
                dest="action", metavar="<ACTION>", parser_class=_Parser
            )
            a = acts.add_parser(
                "serve",
                help="Route requests to nodes",
                description="Route requests to nodes",
                parents=shared,
            )
            a.add_argument(
                "--upstream",
                action="append",
                default=[],
                metavar="<URL>",
                help="A node to route to (repeatable)",
            )
            a.add_argument(
                "--host",
                default="127.0.0.1",
                metavar="<HOST>",
                help="The address to listen on (default: 127.0.0.1)",
            )
            a.add_argument(
                "--port",
                type=int,
                default=8000,
                metavar="<PORT>",
                help="The port to listen on (default: 8000)",
            )
            a.add_argument(
                "--request-timeout",
                type=float,
                default=60.0,
                metavar="<SECONDS>",
                help="Seconds a routed request may take (default: 60)",
            )
        elif noun == "logs":
            n.add_argument(
                "target",
                nargs="?",
                default=".",
                metavar="DIR|NAME",
                help="A predictor directory or a hosted name (default: .)",
            )
            n.add_argument(
                "--env",
                default=DEFAULT_ENV,
                metavar="<E>",
                help=f"The environment (default: {DEFAULT_ENV})",
            )
        elif noun == "diff":
            n.add_argument(
                "target",
                nargs="?",
                default=".",
                metavar="DIR",
                help="The predictor directory (default: .)",
            )
            n.add_argument(
                "--env",
                default=DEFAULT_ENV,
                metavar="<E>",
                help=f"The environment to compare with (default: {DEFAULT_ENV})",
            )
        elif noun == "delete":
            n.add_argument(
                "target",
                nargs="?",
                default=".",
                metavar="DIR|NAME",
                help="A predictor directory or a hosted name (default: .)",
            )
            n.add_argument("--yes", action="store_true", help="Delete without asking")

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
    lines.extend(_option_lines(parser(), out, globals_only=True, root=True))
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
    return action.dest in {"json", "quiet", "verbose", "timeout"}


def _option_lines(
    p: argparse.ArgumentParser,
    out: term.Output,
    *,
    globals_only: bool,
    root: bool = False,
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
    lines.extend(
        [
            "",
            CLOSING.replace("<command>", p.prog.removeprefix("gf ").split(" ")[0])
            if False
            else CLOSING,
            "",
        ]
    )
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
    usage_hint = (
        "run 'gf --help' for the list of commands"
        if p.prog == "gf"
        else f"run '{p.prog} --help' for usage"
    )
    choice = re.search(r"invalid choice: '([^']*)'", message)
    if choice:
        word, names = choice.group(1), _choices(p)
        prior = _ARGV[: _ARGV.index(word)] if word in _ARGV else []
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
        noun = "command" if p.prog == "gf" else "action"
        text = (
            f"unknown {noun} `{shown}`"
            if p.prog == "gf"
            else f"unknown {noun} `{shown}` for `{p.prog}`"
        )
        return text, match and f"a similar {noun} exists: '{match}'", usage_hint
    extra = re.match(r"unrecognized arguments: (.*)", message)
    if extra:
        first = extra.group(1).split()[0]
        words = [w for w in _ARGV if not w.startswith("-")]
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
        return (
            f"unexpected argument `{first if re.fullmatch(r'[\\w./-]{1,40}', first) else '...'}`",
            None,
            usage_hint,
        )
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


# -- results and errors ---------------------------------------------------


def redact(value: Any) -> Any:
    """Removes secret response fields and known credential values."""
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
        return value
    return value


def emit(value: dict, *, error: bool = False) -> None:
    """One JSON object on stdout (results) or stderr (errors)."""
    print(
        json.dumps(redact(value), allow_nan=False, ensure_ascii=False),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def error_object(error: BaseException) -> dict[str, Any]:
    """The `--json` error object: code, message and the optional fields it carries."""
    code = getattr(error, "code", None) or "Internal"
    message = getattr(error, "message", None) or str(error) or "internal error"
    body: dict[str, Any] = {"code": code, "message": message}
    for field in ("param", "hint", "limit", "actual", "request_id"):
        value = getattr(error, field, None)
        if value is not None:
            body[field] = value
    return {"error": body}


def _pointer(error: BaseException) -> term.Pointer | None:
    """The `-->` block for an error that carries a line and column into a file."""
    line, column = getattr(error, "line", None), getattr(error, "column", None)
    if not line or not column:
        return None
    path = getattr(error, "path", None) or "gf.yaml"
    root = Path(
        getattr(ARGS, "directory", None) or getattr(ARGS, "target", None) or "."
    )
    candidates = (
        [Path(path), root / path] if not Path(path).is_absolute() else [Path(path)]
    )
    source = ""
    for candidate in candidates:
        try:
            source = candidate.read_text().splitlines()[line - 1]
            break
        except (OSError, IndexError, UnicodeDecodeError):
            continue
    match = re.match(r"[^\s,\]\}#]+", source[column - 1 :]) if source else None
    width = len(match.group(0)) if match else 1
    label = getattr(error, "because", None) or ""
    return term.Pointer(path, line, column, source, label, width)


def _oserror(error: OSError) -> Error:
    filename = getattr(error, "filename", None)
    reason = (error.strerror or type(error).__name__).lower()
    if filename:
        return Error(
            "NotFound" if isinstance(error, FileNotFoundError) else "InvalidInput",
            f"cannot read `{filename}`: {reason}",
        )
    return Error("InvalidInput", reason)


def present(error: BaseException, out: term.Output) -> int:
    """Prints any failure in the contract's grammar and returns the exit code."""
    if isinstance(error, term.Aborted):
        return 1
    if isinstance(error, KeyboardInterrupt):
        if JSON:
            emit(
                {"error": {"code": "Interrupted", "message": "interrupted"}}, error=True
            )
        else:
            out.say(force=True)
            out.error("interrupted")
        return 130
    if isinstance(error, Usage):
        if JSON:
            emit(
                {"error": {"code": "UsageError", "message": redact(error.message)}},
                error=True,
            )
        else:
            out.error(
                redact(error.message),
                tip=error.tip,
                hint=None if error.tip else error.hint,
            )
        return 2
    if isinstance(error, OSError) and not hasattr(error, "code"):
        error = _oserror(error)
    known = isinstance(error, Error) or (
        hasattr(error, "code") and hasattr(error, "message")
    )
    if not known:
        if out.verbose:
            import traceback

            out.say("".join(traceback.format_exception(error)).rstrip(), force=True)
        if JSON:
            emit(
                {
                    "error": {
                        "code": "Internal",
                        "message": "internal error",
                        "hint": "run again with -v",
                    }
                },
                error=True,
            )
        else:
            out.error("internal error", hint="run again with -v")
        return 1
    if JSON:
        emit(error_object(error), error=True)
        return 1
    pointer = _pointer(error)
    hint = getattr(error, "hint", None)
    if hint is None and getattr(error, "code", None) == "NotLoggedIn":
        hint = "run `gf login`"
    if hint is None and str(getattr(error, "message", "")).startswith(
        "permission denied"
    ):
        hint = "run `gf login` again, or use a key that has this permission"
    because = getattr(error, "because", None)
    limit, actual = getattr(error, "limit", None), getattr(error, "actual", None)
    if because is None and limit is not None and actual is not None:
        because = (
            f"the limit is {term.count(limit)}; the request sent {term.count(actual)}"
        )
    out.error(
        redact(str(getattr(error, "message", error))),
        hint=hint,
        because=None if pointer is not None else because,
        pointer=pointer,
        request_id=getattr(error, "request_id", None),
    )
    return 1


# -- dispatch -------------------------------------------------------------


def run(argv: Sequence[str]) -> dict | None:
    """Parses and runs one command, returning its result for `--json`."""
    global ARGS
    root = parser()
    args = root.parse_args(list(argv))
    ARGS = args
    out = output()
    if args.command is None:
        sys.stdout.write(root_help(term.Output(out=sys.stdout, err=sys.stdout)))
        return None
    if args.command == "help":
        topic = _find(root, args.topic)
        if topic is None:
            if args.topic:
                match = term.suggest(args.topic[0], COMMANDS)
                raise Usage(
                    f"unknown command `{args.topic[0]}`",
                    tip=match and f"a similar command exists: '{match}'",
                    hint="run 'gf --help' for the list of commands",
                    prog="gf",
                )
            topic = root
        sys.stdout.write(topic.format_help())
        return None
    for group_name in ("model", "context", "admin"):
        if (
            args.command == group_name
            and getattr(args, "action" if group_name != "admin" else "noun", None)
            is None
        ):
            sub = _find(root, [group_name])
            assert sub is not None
            nested = _subparsers(sub)
            metavar = nested.metavar if nested is not None else "<COMMAND>"
            sub.error(f"the following arguments are required: {metavar}")
    if args.command == "admin" and args.noun in ADMIN_ACTIONS and args.action is None:
        sub = _find(root, ["admin", args.noun])
        assert sub is not None
        sub.error("the following arguments are required: <ACTION>")
    if (
        args.command == "admin"
        and args.noun in ("node", "router")
        and args.action is None
    ):
        sub = _find(root, ["admin", args.noun])
        assert sub is not None
        sub.error("the following arguments are required: <ACTION>")
    if args.command in ("login", "logout", "whoami"):
        from generalforecast import account

        return (
            getattr(account, args.command)(args, out)
            if args.command != "whoami"
            else account.whoami(args)
        )
    if args.command in ("model", "context", "admin"):
        from generalforecast.commands import execute

        return execute(args, out)
    from generalforecast.predictor_commands import execute as predictors

    return predictors(args, out)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs one command; failures print in the error grammar and exit nonzero."""
    global JSON, _OUT, _ARGV
    _ARGV = list(sys.argv[1:] if argv is None else argv)
    argv = _ARGV
    if argv in (["-V"], ["--version"]):
        sys.stdout.write(f"gf {version()}\n")
        return
    if argv in ([], ["-h"], ["--help"], ["help"]):
        sys.stdout.write(root_help(term.Output(out=sys.stdout, err=sys.stdout)))
        return
    flags = set(argv[: argv.index("--")] if "--" in argv else argv)
    JSON = "--json" in flags
    _OUT = term.Output(
        quiet=JSON or bool(flags & {"-q", "--quiet"}),
        verbose=bool(flags & {"-v", "--verbose"}),
    )
    try:
        result = run(argv)
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001
        raise SystemExit(present(error, _OUT)) from None
    if result is None:
        return
    if JSON:
        emit(result)
    else:
        from generalforecast.render import render

        assert ARGS is not None
        render(ARGS, result, _OUT)


if __name__ == "__main__":
    main()
