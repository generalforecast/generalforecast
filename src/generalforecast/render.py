# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Writes a command's result: for a person, or as one JSON object under `--json`.

Commands print status lines on stderr as they go and return their result;
this module is the only writer of results on stdout. `emit` prints the
result as JSON, `render` draws it for a person: the scored rows of
`predict`, the metrics of `eval`, log lines, and the tables and records of
`whoami`, `status`, `model`, `context` and `admin`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from generalforecast import term
from generalforecast.command_specs import ADMIN_PREDICTOR_NOUNS, ADMIN_SERVE_ACTIONS
from generalforecast.errors import printable

# Response fields whose values are credentials, whatever the endpoint.
_SECRET_FIELD = re.compile(
    r"(?i)(token|password|secret_value|credential_value|authorization|private_key)"
)
# Environment variables that hold credentials, redacted wherever they appear.
_SECRET_ENVS = (
    "GF_API_KEY",
    "GF_TOKEN",
    "GF_PLATFORM_TOKEN",
    "GF_MODAL_KEY",
    "GF_MODAL_SECRET",
)


def jsonable(value: Any) -> Any:
    """`value` as plain JSON values.

    A data frame becomes its records, numpy scalars and paths their Python
    values, dates ISO strings, and a float JSON cannot hold, such as an
    undefined metric's NaN, becomes null.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(value, pandas.DataFrame):
        return jsonable(value.to_dict(orient="records"))
    if pandas is not None and value is pandas.NaT:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item") and type(value).__module__ == "numpy":
        return jsonable(value.item())
    raise TypeError(f"{type(value).__name__} is not a result value")


def redact(value: Any) -> Any:
    """Removes secret response fields and known credential values."""
    from generalforecast.credentials import SEEN

    if isinstance(value, dict):
        return {
            redact(key): "[redacted]" if _SECRET_FIELD.search(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        values = SEEN | {os.environ.get(env, "") for env in _SECRET_ENVS}
        for secret in sorted(values, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[redacted]")
        return value
    return value


def emit(value: dict, *, error: bool = False) -> None:
    """One JSON object on stdout (results) or stderr (errors)."""
    print(
        json.dumps(redact(jsonable(value)), allow_nan=False, ensure_ascii=False),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def render(args: argparse.Namespace, result: Any, out: term.Output) -> None:
    """Draws `result` for `args.command` on stdout."""
    if not isinstance(result, dict):
        return
    handler = _HANDLERS.get(args.command)
    if handler is not None:
        handler(args, result, out)


def _scalar(value: Any) -> str:
    """One value of a server's reply as a person reads it, with no terminal escapes."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return term.count(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return printable(json.dumps(jsonable(value), ensure_ascii=False))
    return printable(str(value))


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
    out.fields(
        [(printable(str(key)), _scalar(value)) for key, value in result.items()],
        indent=0,
    )


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
        header=[out.quiet_text(printable(str(key)), out.out) for key in keys],
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
            }
            for e in environments
        ],
        out,
    )


def _model(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    def row(model: dict) -> dict:
        limits = model.get("limits") or {}
        return {
            "model": model.get("id"),
            "task": model.get("task"),
            "context rows": limits.get("context_rows"),
            "queries": limits.get("queries"),
            "inputs": limits.get("inputs"),
            "cached": model.get("cached"),
        }

    if args.action == "list":
        records([row(m) for m in result.get("models", [])], out)
    elif args.action == "show":
        out.fields(
            [(key, _scalar(value)) for key, value in row(result).items()], indent=0
        )


def _context(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    if args.action == "show":
        generic(result, out)


def metrics(result: dict, out: term.Output) -> None:
    """The metrics table: names left, values right, one column per model, at column 14."""
    table = result.get("metrics") or {}
    models = list(table)
    names: list[str] = []
    for values in table.values():
        for metric in values:
            if metric not in names:
                names.append(metric)

    def cell(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else "-"

    rows = [[metric, *(cell(table[m].get(metric)) for m in models)] for metric in names]
    header = ["", *(m.split("/", 1)[-1] for m in models)] if len(models) > 1 else None
    out.say()
    out.table(
        rows,
        indent=term.LABEL_WIDTH,
        gap=3,
        align=["<", *([">"] * len(models))],
        header=header,
        stream=out.out,
    )


def _eval(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    metrics(result, out)


def _predict(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    frame = result.get("predictions")
    if frame is None:
        return
    for line in frame.to_csv(index=False).rstrip("\n").split("\n"):
        out.line(line)


def _logs(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    for line in result.get("lines", []):
        out.line(printable(str(line)))


def _serving(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    for line in result.get("checks", []):
        out.line(str(line))
    steps = result.get("next") or []
    for index, step in enumerate(steps):
        out.line(f"{'next: ' if index == 0 else '      '}{step}")


def _admin(args: argparse.Namespace, result: dict, out: term.Output) -> None:
    if args.noun == "logs":
        _logs(args, result, out)
    elif args.noun in ADMIN_SERVE_ACTIONS:
        _serving(args, result, out)
    elif args.noun not in ADMIN_PREDICTOR_NOUNS:
        generic(result, out)


_HANDLERS: dict[str, Callable[[argparse.Namespace, dict, term.Output], None]] = {
    "whoami": _whoami,
    "status": _status,
    "model": _model,
    "context": _context,
    "eval": _eval,
    "predict": _predict,
    "admin": _admin,
}
