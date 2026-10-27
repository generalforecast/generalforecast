# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Renders a command's result for a person.

Commands that stream status lines print as they go; this module draws the
results that are tables or records: `whoami`, `status`, `model list`,
`model show`, `context show` and the `admin` collections.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from generalforecast import term


def render(args: argparse.Namespace, result: Any, out: term.Output) -> None:
    """Draws `result` for `args.command` on stdout."""
    if not isinstance(result, dict):
        return
    handler = _HANDLERS.get(args.command)
    if handler is not None:
        handler(args, result, out)


def _scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return term.count(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def generic(result: dict, out: term.Output) -> None:
    """A record as key/value fields, or a list of records as a table."""
    lists = [(key, value) for key, value in result.items() if isinstance(value, list)]
    if (
        len(lists) == 1
        and len(result) == 1
        and all(isinstance(item, dict) for item in lists[0][1])
    ):
        records(lists[0][1], out)
        return
    out.fields([(key, _scalar(value)) for key, value in result.items()], indent=0)


def records(
    items: list[dict], out: term.Output, columns: list[str] | None = None
) -> None:
    """Records as a table with a quiet header row."""
    if not items:
        out.line("none")
        return
    keys = columns or [
        key for key in items[0] if not isinstance(items[0][key], (dict, list))
    ]
    rows = [[_scalar(item.get(key)) for key in keys] for item in items]
    out.table(
        rows,
        header=[out.quiet_text(key, out.out) for key in keys],
        indent=0,
        stream=out.out,
    )


def _whoami(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    workspace = result.get("workspace")
    if isinstance(workspace, dict):
        result = result | {
            "workspace": workspace.get("display_name") or workspace.get("id")
        }
    pairs = [
        (label, _scalar(result[key]))
        for key, label in (
            ("email", "account"),
            ("account", "account"),
            ("workspace", "workspace"),
            ("project", "project"),
            ("server", "server"),
            ("source", "credentials"),
        )
        if result.get(key) is not None
    ]
    out.fields(pairs or [(k, _scalar(v)) for k, v in result.items()], indent=0)


def _status(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    environments = result.get("environments") or []
    if isinstance(environments, dict):
        environments = list(environments.values())
    records(
        [
            {
                "environment": e.get("name"),
                "revision": e.get("revision"),
                "status": e.get("status"),
                "runtime": e.get("runtime"),
            }
            for e in environments
        ],
        out,
    )


def _model(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    if args.action == "list":
        rows = [
            {
                "model": m["id"],
                "task": m["task"],
                "context rows": m["limits"]["context_rows"],
                "queries": m["limits"]["queries"],
                "inputs": m["limits"]["inputs"],
                "cached": m["cached"],
            }
            for m in result.get("models", [])
        ]
        records(rows, out)
    elif args.action == "show":
        limits = result.get("limits", {})
        out.fields(
            [
                ("model", result["id"]),
                ("task", result["task"]),
                ("context rows", _scalar(limits.get("context_rows"))),
                ("queries", _scalar(limits.get("queries"))),
                ("inputs", _scalar(limits.get("inputs"))),
                ("cached", _scalar(result.get("cached"))),
            ],
            indent=0,
        )


def _context(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    if args.action == "show":
        generic(result, out)


def _admin(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    if args.noun in ("node", "router", "logs", "diff", "delete"):
        return
    generic(result, out)


_HANDLERS = {
    "whoami": _whoami,
    "status": _status,
    "model": _model,
    "context": _context,
    "admin": _admin,
}
