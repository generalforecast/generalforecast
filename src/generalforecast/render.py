# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Human output for gf results and watch events; --json bypasses this module.

Status lines (right-aligned verbs) go to stderr; data (tables, fields) to stdout.
"""

import argparse
import datetime
import time
from typing import Any

from generalforecast.account import date, host
from generalforecast.term import Output, seconds

PREVIEW_ROWS = 20
_PREFERRED = (
    "name",
    "id",
    "job",
    "package",
    "deployment",
    "model",
    "task",
    "status",
    "state",
    "rows",
    "project",
    "created_at",
    "updated_at",
    "expires_at",
)


def render(
    args: argparse.Namespace | None, result: Any, out: Output, elapsed: float
) -> None:
    """Prints one command's result for people."""
    group = getattr(args, "group", None)
    handler = _HANDLERS.get(group or "")
    if handler is not None:
        handler(args, result, out, elapsed)
    elif isinstance(result, dict):
        generic(args, result, out)


def _none(*_: Any) -> None:
    return None


def _whoami(args: Any, result: dict, out: Output, elapsed: float) -> None:
    key, workspace = result.get("key") or {}, result.get("workspace")
    out.line(result.get("email") or f"API key {key.get('name') or key.get('id')}")
    expires = date(key.get("expires_at"))
    detail = ", ".join(
        x for x in (result.get("source"), expires and f"expires {expires}") if x
    )
    out.fields(
        [
            (
                "workspace",
                workspace and f"{workspace['display_name']} ({workspace['id']})",
            ),
            ("server", host(result["server"])),
            ("key", f"{key.get('suffix')} ({detail})" if detail else key.get("suffix")),
        ],
        indent=2,
    )


def _init(args: Any, result: dict, out: Output, elapsed: float) -> None:
    directory = args.directory if args is not None else result["name"]
    shown = directory if directory.startswith(("/", ".", "~")) else f"./{directory}"
    out.verb("Created", f"package {result['name']} in {shown}")
    out.say()
    out.say("Next:")
    comment = out.style("# add a schema sample, then", "dim", stream=out.err)
    out.say(f"  cd {directory}  {comment}")
    out.say("  gf check --data sample.csv")


def _check(args: Any, lock: dict, out: Output, elapsed: float) -> None:
    spec, model = lock.get("spec") or {}, lock.get("model") or {}
    columns = (lock.get("schema") or {}).get("columns")
    detail = f"{lock.get('task')}, {model.get('name') or spec.get('model')}"
    if isinstance(columns, list):
        detail += f", {len(columns)} columns"
    out.verb("Checked", f"{spec.get('name')} ({detail})")
    out.verb("Locked", "gf.lock", f"in {seconds(elapsed)}")


def _serve(args: Any, result: dict, out: Output, elapsed: float) -> None:
    out.verb("Checked", result["model"], f"in {seconds(elapsed)}")
    out.note(f"start it with `gf serve {result['target']}`")


def _push(args: Any, result: dict, out: Output, elapsed: float) -> None:
    if "session" in result:
        out.verb("Stopped", f"dev session {result['session'].get('id')}")
        return
    revision, environment = result["revision"], result["environment"]
    name = revision["deployment"]
    out.verb(
        "Deployed",
        f"{name} r{revision['number']} to {environment['name']}",
        f"in {seconds(elapsed)}",
    )
    if environment.get("status") != "ready":
        out.warning(f"{environment['name']} is {environment.get('status')}")
    from generalforecast.credentials import CONNECTION

    out.say()
    out.say(f"  Predict: gf predict {name} --input <file>")
    if CONNECTION.get("server") and CONNECTION.get("source") != "--local":
        out.say(
            f"  Console: {out.link(CONNECTION['server'] + '/#/deployments/' + name)}"
        )


def _predict(args: Any, result: dict, out: Output, elapsed: float) -> None:
    rows = result.get("predictions") or []
    model = result.get("model")
    detail = f"({model}, r{result.get('revision')}, {seconds(elapsed)})"
    noun = "row" if len(rows) == 1 else "rows"
    out.verb("Predicted", f"{len(rows)} {noun} with {result.get('deployment')}", detail)
    written = getattr(args, "output", None)
    if written and written != "json":
        out.verb("Wrote", written)
        return
    if not rows:
        return
    headers = list(rows[0])
    out.line()
    out.table(headers, [[r.get(h) for h in headers] for r in rows[:PREVIEW_ROWS]])
    if len(rows) > PREVIEW_ROWS:
        out.note(f"\n... {len(rows) - PREVIEW_ROWS} more rows")
    out.say()
    out.say(f"Write all rows to a file with {out.code('--output predictions.csv')}")


def _status(args: Any, result: dict, out: Output, elapsed: float) -> None:
    env = result.get("environment") or {}
    out.fields(
        [
            ("deployment", env.get("deployment")),
            ("environment", env.get("name")),
            ("revision", env.get("revision") and f"r{env['revision']}"),
            ("status", env.get("status")),
            ("runtime", env.get("runtime")),
            ("generation", env.get("generation")),
        ]
    )


def _logs(args: Any, result: dict, out: Output, elapsed: float) -> None:
    lines = result.get("logs")
    if not isinstance(lines, list):
        generic(args, result, out)
        return
    if not lines:
        out.note("No logs yet.")
    for line in lines:
        if isinstance(line, dict):
            stamp = line.get("time") or line.get("timestamp")
            text = line.get("message") or line.get("line") or line
            out.line(f"{out.style(str(stamp), 'dim')} {text}" if stamp else str(text))
        else:
            out.line(str(line))


def _diff(args: Any, result: dict, out: Output, elapsed: float) -> None:
    changed = result.get("changed") or []
    if not changed:
        out.verb("Unchanged", f"{args.name} matches the deployed package")
        return
    out.verb("Changed", f"{len(changed)} fields differ from {args.name} ({args.env})")
    for item in changed:
        out.line(f"  {out.style('~', 'yellow')} {item}")


def _environment_change(verb: str):
    def handler(args: Any, result: dict, out: Output, elapsed: float) -> None:
        out.verb(
            verb,
            f"{result.get('deployment')} r{result.get('revision')} in {result.get('name')}",
            f"in {seconds(elapsed)}",
        )

    return handler


def _delete(args: Any, result: dict, out: Output, elapsed: float) -> None:
    out.verb("Deleted", f"deployment {args.name}")


def generic(args: Any, result: dict, out: Output) -> None:
    """Tables for collections, aligned fields for one resource."""
    lists = [(k, v) for k, v in result.items() if isinstance(v, list)]
    if len(lists) == 1 and all(isinstance(v, dict) for v in lists[0][1]):
        noun, items = lists[0]
        table(noun, items, out)
        return
    out.fields(flatten(result))


def table(noun: str, items: list[dict], out: Output) -> None:
    """A collection as columns, with a dim count footer."""
    if not items:
        out.note(f"No {noun}.")
        return
    keys = [k for k in items[0] if _scalar(items[0][k])]
    columns = [k for k in _PREFERRED if k in keys] or keys
    columns = columns[:7]
    rows = [[_value(k, item.get(k)) for k in columns] for item in items]
    out.table(
        [c.removesuffix("_at") if c.endswith("_at") else c for c in columns], rows
    )
    out.say()
    out.note(f"{len(items)} {noun if len(items) != 1 else noun.rstrip('s')}.")


def flatten(value: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Nested objects as dotted labels; lists of objects as counts."""
    pairs: list[tuple[str, Any]] = []
    for key, item in value.items():
        label = prefix + key
        if isinstance(item, dict):
            pairs += flatten(item, label + ".")
        elif isinstance(item, list) and not all(_scalar(x) for x in item):
            pairs.append((label, f"{len(item)} items"))
        else:
            pairs.append((label, _value(key, item)))
    return pairs


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _value(key: str, value: Any) -> Any:
    if key.endswith("_at") and isinstance(value, (int, float)) and value > 1e9:
        return ago(value)
    return value


def ago(stamp: float, now: float | None = None) -> str:
    """`3m ago`, `2h ago`, `yesterday` or a date."""
    delta = (time.time() if now is None else now) - stamp
    if delta < 0:
        ahead = -delta
        if ahead < 3600:
            return f"in {max(1, int(ahead // 60))}m"
        if ahead < 172800:
            return f"in {int(ahead // 3600)}h"
        return f"in {int(ahead // 86400)} days"
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 172800:
        return "yesterday"
    return datetime.datetime.fromtimestamp(stamp, datetime.UTC).date().isoformat()


def event(value: dict, out: Output) -> None:
    """One `gf push --watch` event as a status line."""
    kind, session = value.get("event"), value.get("session")
    session_id = session.get("id") if isinstance(session, dict) else session
    digest = str(value.get("package") or value.get("serving_package") or "")[:19]
    if kind == "watching":
        out.verb("Watching", f"dev session {session_id} (Ctrl-C to stop)")
    elif kind == "patching":
        out.verb("Patching", f"dev session {session_id}", digest)
    elif kind == "patch_rejected":
        out.warning(f"patch rejected ({value.get('code')}); still serving {digest}")
    elif kind == "reconnecting":
        out.warning(f"connection lost; reconnecting (attempt {value.get('attempt')})")
    elif kind in {"disconnected", "conflict"}:
        out.warning(f"{kind} from dev session {session_id}")
        out.hint(f"resume with gf push --watch --resume {value.get('resume')}")
    elif kind == "stopped":
        out.verb("Stopped", f"dev session {session_id}")
    elif kind == "cleanup_failed":
        out.warning(f"could not stop dev session {session_id}; {value.get('action')}")
    else:
        out.note(str(value))


_HANDLERS = {
    "login": _none,
    "logout": _none,
    "whoami": _whoami,
    "init": _init,
    "check": _check,
    "serve": _serve,
    "push": _push,
    "predict": _predict,
    "status": _status,
    "logs": _logs,
    "diff": _diff,
    "delete": _delete,
    "promote": _environment_change("Promoted"),
    "rollback": _environment_change("Rolled back"),
}
