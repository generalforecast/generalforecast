# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The predictor commands: init, check, eval, serve, predict, push and the hosted pointers.

`gf init` asks for what it cannot infer and writes `gf.yaml`. `gf check`
resolves the manifest against the context and the model registry through
`generalforecast.predictors` and writes `gf.lock`. The hosted commands use
the SDK's transport and the hosted predictor routes. Each command returns
its result for `generalforecast.render` to write.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.compute as _compute

from generalforecast import term
from generalforecast.command_specs import DEFAULT_ENV
from generalforecast.errors import AmbiguousReference, Error, InvalidInput, NotFound

if TYPE_CHECKING:
    from generalforecast.predictors import Resolution

# pyarrow.compute builds its kernels at import, so its stubs name none of them.
pc: Any = _compute
MANIFEST = "gf.yaml"
LOCK = "gf.lock"
TABLE_SUFFIXES = (".csv", ".parquet", ".json")


def _table(path: Path) -> Any:
    from generalforecast.commands import read_data

    return read_data(str(path), Path.cwd())


def _registry() -> Any:
    from generalforecast.commands import model_registry

    return model_registry()


def _previous_lock(root: Path) -> dict[str, Any] | None:
    from generalforecast.predictors import lock

    return lock.read_lock(root)


def _short(digest: str | None) -> str:
    if not digest:
        return "no digest"
    return digest.split(":")[-1][:7]


def _shown_value(value: Any) -> str:
    if value is None:
        return "whole table"
    if type(value) is int:
        return term.count(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _shown_changes(
    changes: Sequence[tuple[str, str, Any, Any]], model: str
) -> list[tuple[str, str, Any, Any]]:
    """Lock changes as the CLI lists them.

    The library diffs dotted keys with raw values. The listing joins the model
    and its digest into one `model` line with short digests, every changed
    selection field into one `selection` line, counts in thousands, and
    orders the lines model, inputs, selection, then the rest.
    """
    shown: list[tuple[str, str, Any, Any]] = []
    pinned = {
        key: (old, new)
        for _, key, old, new in changes
        if key in ("model", "model_digest")
    }
    selection = [
        (key.removeprefix("selection."), old, new)
        for _, key, old, new in changes
        if key.startswith("selection.")
    ]
    for op, key, old, new in changes:
        if key in ("model", "model_digest"):
            if not pinned:
                continue
            old_digest, new_digest = pinned.get("model_digest", (None, None))
            if "model" in pinned:
                before, after = pinned["model"]
                line = (
                    f"{before} {_short(old_digest)}",
                    f"{after} {_short(new_digest)}",
                )
            else:
                line = (f"{model} {_short(old_digest)}", _short(new_digest))
            shown.append(("~", "model", *line))
            pinned = {}
        elif key.startswith("selection."):
            if not selection:
                continue
            parts = [
                f"{field} {_shown_value(old)} → {_shown_value(new)}"
                for field, old, new in selection
            ]
            shown.append(("~", "selection", None, ", ".join(parts)))
            selection = []
        elif key == "input":
            shown.append((op, key, old, new))
        else:
            shown.append(
                (
                    op,
                    key,
                    _shown_value(old) if old is not None else None,
                    _shown_value(new) if new is not None else None,
                )
            )
    order = {"model": 0, "input": 1, "selection": 2}
    return sorted(shown, key=lambda change: order.get(change[1], 3))


def _query_rule(spec: Any, query_rows: int) -> str:
    context = spec.context
    if context.until is not None:
        return f"rows at or after {context.until}"
    if context.holdout is not None:
        return f"a random {context.holdout.fraction:g} of the rows"
    return "rows with no target"


def predictors() -> Any:
    """The predictor library, `generalforecast.predictors`."""
    from generalforecast import predictors as library

    return library


def predictor_class() -> Any:
    """`generalforecast.Predictor`, the class `gf eval` and `gf predict` run."""
    from generalforecast.predictor import Predictor

    return Predictor


# -- helpers --------------------------------------------------------------


def _shown(root: Path) -> str:
    text = os.path.relpath(root)
    if text == ".":
        return "."
    if text.startswith(".."):
        return str(root)
    return text if text.startswith((".", "/")) else f"./{text}"


def is_local(reference: str) -> bool:
    """Whether a hosted command's DIR|NAME argument names a directory here.

    A path names one; so does a bare name with a `gf.yaml` beneath it, whose
    manifest then names the hosted predictor.
    """
    if re.match(r"^https?://", reference):
        return False
    if reference.startswith((".", "~")) or "/" in reference or os.sep in reference:
        return True
    return (Path(reference) / MANIFEST).is_file()


def _tables_in(root: Path) -> list[Path]:
    return (
        sorted(p for p in root.iterdir() if p.is_file() and p.suffix in TABLE_SUFFIXES)
        if root.is_dir()
        else []
    )


def _output_path(
    text: str | None, *inputs: str | Path | None, predictor: Path | None = None
) -> Path | None:
    """The `--output` file, checked before any scoring.

    Its directory must exist, and it must be none of the files the command
    reads, which writing it would destroy: the queries and context it was
    given, and a local predictor's manifest, lock and snapshot.
    """
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.parent.is_dir():
        raise NotFound(
            f"cannot write `{text}`: directory `{path.parent}` does not exist",
            param="output",
        )
    if path.is_dir():
        raise InvalidInput(f"cannot write `{text}`: it is a directory", param="output")
    read = list(inputs)
    if predictor is not None:
        read += [predictor / MANIFEST, predictor / LOCK]
        snapshot = predictors().load(predictor).context.snapshot
        if snapshot is not None:
            read.append(predictor / snapshot)
    for given in read:
        if given is not None and Path(given).expanduser().resolve() == path.resolve():
            raise InvalidInput(
                f"--output `{text}` is also an input; it would be overwritten",
                hint="write the scored rows to another file",
                param="output",
            )
    return path


def _mode(path: Path) -> int:
    """The mode a written file takes: the one it replaces, else what the umask gives."""
    if path.exists():
        return path.stat().st_mode & 0o777
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _write_frame(frame: Any, path: Path) -> None:
    """Writes rows as their suffix says, atomically: a reader sees all or nothing."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif path.suffix == ".json":
            frame.to_json(temporary, orient="records")
        else:
            frame.to_csv(temporary, index=False)
        os.chmod(temporary, _mode(path))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


# -- the commands ---------------------------------------------------------


def execute(args: argparse.Namespace, out: term.Output) -> dict | None:
    """Runs one predictor command."""
    if args.command == "serve":
        from generalforecast.serve import listener

        listener.serve(args)
    return getattr(PredictorCommands(args, out), args.command)()


class PredictorCommands:
    """The predictor commands, bound to one invocation's arguments and renderer."""

    def __init__(self, args: argparse.Namespace, out: term.Output) -> None:
        """Binds the arguments and the renderer."""
        self.args, self.out = args, out
        self.library = predictors()

    # -- shared -----------------------------------------------------------

    @cached_property
    def client(self) -> Any:
        """The authenticated SDK client, built on first use."""
        from generalforecast.commands import client_for

        return client_for(self.args)

    @property
    def root(self) -> Path:
        """The predictor directory a DIR argument names."""
        return Path(self.args.directory).expanduser()

    def reference(self) -> tuple[str, Path | None]:
        """A DIR|NAME argument as `(hosted name, local root or None)`."""
        target = self.args.target
        if is_local(target):
            root = Path(target).expanduser()
            return self.library.load(root).name, root
        return target, None

    def call(
        self, method: str, predictor: str, suffix: str = "", payload: dict | None = None
    ) -> dict:
        """One request on the predictor's hosted routes."""
        from generalforecast.service.contracts import name

        return self.client._call(
            method,
            f"/predictors/{name(predictor)}" + suffix,
            "predictor",
            predictor,
            payload,
        )

    def confirm(self, question: str) -> None:
        """Asks `question [y/N]` unless `--yes`; raises `Aborted` on anything but yes."""
        term.Prompter(self.out, yes=self.args.yes).confirm(question)

    # -- init -------------------------------------------------------------

    def init(self) -> dict:
        """Writes a predictor manifest, asking for what the context does not say."""
        args, out = self.args, self.out
        root = self.root
        manifest = root / MANIFEST
        if manifest.exists() and not args.force:
            raise Error(
                "InvalidInput",
                f"{MANIFEST} already exists in {_shown(root)}",
                hint="pass --force to overwrite it, or run `gf check` to validate it",
            )
        if args.context:
            source = Path(args.context).expanduser()
            if not source.is_file():
                raise Error(
                    "NotFound",
                    f"context file `{args.context}` does not exist",
                    hint="pass --context with a .csv, .parquet or .json file",
                )
            try:
                relative = source.resolve().relative_to(root.resolve())
            except ValueError:
                raise Error(
                    "InvalidInput",
                    f"context file `{args.context}` is outside {_shown(root)}",
                    hint=f"move it into {_shown(root)}; a predictor's snapshot lives beside its {MANIFEST}",
                ) from None
        else:
            tables = _tables_in(root)
            if len(tables) != 1:
                if not tables:
                    raise Error(
                        "NotFound",
                        f"no context table in {_shown(root)}",
                        hint="pass --context FILE, or put a .csv, .parquet or .json file here",
                    )
                raise Error(
                    "InvalidInput",
                    f"{_shown(root)} has {len(tables)} tables",
                    because="the context must be one file: "
                    + ", ".join(t.name for t in tables),
                    hint="pass --context FILE to pick one",
                )
            source = tables[0]
            relative = source.relative_to(root)
        info = self.library.inspect_table(source)
        out.status(
            "Found",
            f"{relative}  {term.plural(info.rows, 'row')}, {term.plural(len(info.columns), 'column')}",
        )
        out.say(force=True)
        prompter = term.Prompter(out, yes=args.yes, no_input=args.no_input)
        registry = _registry()
        options = {c.name: term.Option(c.name, c.name, c.kind) for c in info.columns}

        target = args.target
        if target is not None and target not in options:
            raise Error(
                "InvalidInput",
                f"target column `{target}` is not in {relative}",
                hint=f"{relative} has {', '.join(info.names)}",
                param="target",
            )
        if target is None:
            first = info.targets[:1]
            candidates = [
                options[c] for c in (*first, *(c for c in info.names if c not in first))
            ]
            target = prompter.choose("Target", candidates, flag="--target")
        else:
            out.collapsed("Target", target)

        column = info.column(target)
        model = args.model
        if model is not None:
            model = registry.canonical(model)
            task, why = registry.task_of(model), "from --model"
        elif args.task is not None:
            task, why = args.task, "from --task"
        elif args.time or args.frequency or args.horizon:
            flag = next(
                f
                for f, v in (
                    ("--time", args.time),
                    ("--frequency", args.frequency),
                    ("--horizon", args.horizon),
                )
                if v
            )
            task, why = "forecast", f"from {flag}"
        else:
            task = self.library.task_for(column, info.rows)
            why = f"{target} has {term.count(column.distinct)} distinct {column.kind} values"
        out.collapsed("Task", f"{task}  {out.quiet_text(why)}")
        forecast = task == "forecast"
        time_column, series, frequency, horizon = (
            args.time,
            args.series,
            args.frequency,
            args.horizon,
        )
        if forecast:
            if time_column is None:
                times = [options[c] for c in info.times] or [
                    options[c] for c in info.names if c != target
                ]
                time_column = prompter.choose("Time", times, flag="--time")
            else:
                out.collapsed("Time", time_column)
            if series is None:
                rest = [
                    options[c] for c in info.names if c not in (target, time_column)
                ]
                series = prompter.choose_many(
                    "Series",
                    rest,
                    flag="--series",
                    selected=[
                        i
                        for i, o in enumerate(rest)
                        if info.column(o.value).kind in ("text", "category")
                    ],
                )
            else:
                out.collapsed("Series", ", ".join(series))
            if frequency is None:
                frequency = prompter.ask("Frequency", "1d", flag="--frequency")
            else:
                out.collapsed("Frequency", str(frequency))
            if horizon is None:
                horizon = int(
                    prompter.ask(
                        "Horizon", "7", flag="--horizon", check=_horizon_problem
                    )
                )
            else:
                out.collapsed("Horizon", str(horizon))

        if model is None:
            default = registry.default(task)
            listed = sorted(registry.catalog(task=task), key=lambda m: m.id != default)
            catalog = [term.Option(m.id, m.id, m.task) for m in listed]
            model = prompter.choose("Model", catalog, flag="--model")
        else:
            out.collapsed("Model", model)

        taken = {target, time_column, *(series or ())} - {None}
        identifier = args.id
        if identifier is None:
            ids = [options[c] for c in info.ids if c not in taken]
            identifier = prompter.choose(
                "Id", [*ids, term.Option(None, "none", "")], flag="--id"
            )
        else:
            if identifier not in options:
                raise Error(
                    "InvalidInput",
                    f"id column `{identifier}` is not in {relative}",
                    param="id",
                )
            out.collapsed("Id", identifier)
        taken.add(identifier)

        remaining = [options[c] for c in info.names if c not in taken]
        if args.inputs is not None:
            missing = [c for c in args.inputs if c not in options]
            if missing:
                raise Error(
                    "InvalidInput",
                    f"input column `{missing[0]}` is not in {relative}",
                    param="inputs",
                )
            inputs: list[str] | None = list(args.inputs)
            out.collapsed("Inputs", ", ".join(inputs))
        else:
            preselected = [
                i
                for i, o in enumerate(remaining)
                if o.value not in (args.exclude or ())
            ]
            chosen = prompter.choose_many(
                "Inputs", remaining, flag="--inputs", selected=preselected
            )
            inputs = (
                None
                if chosen == [o.value for o in remaining] and not args.exclude
                else chosen
            )

        empty = column.empty
        time_name = time_column or (info.times[0] if info.times else None)
        cap = registry.limits(model).queries
        context: dict[str, Any] = {"snapshot": str(relative)}
        if args.until is not None:
            if time_name is None:
                raise Error(
                    "InvalidInput",
                    "--until needs a time column",
                    because=f"{relative} has no timestamp column to compare dates against",
                    hint="pass --holdout FRACTION for a random split",
                    param="until",
                )
            context["until"] = args.until
            out.collapsed("Queries", f"rows at or after {args.until}")
        elif args.holdout is not None:
            context["holdout"] = {"fraction": args.holdout, "seed": 0}
            out.collapsed("Queries", f"a random {args.holdout:g} of the rows")
        elif args.queries == "none":
            out.collapsed("Queries", "none")
        else:
            rules = []
            if empty:
                rules.append(
                    term.Option(
                        "empty",
                        "rows with no target",
                        term.plural(empty, "row"),
                        f"{term.count(empty)} rows with no target",
                    )
                )
            if time_name is not None:
                rules.append(term.Option("date", "rows after a date", ""))
            rules.extend(
                [
                    term.Option("fraction", "a random fraction", ""),
                    term.Option("none", "none", ""),
                ]
            )
            rule = prompter.choose(
                "Queries", rules, flag="--until, --holdout or --queries none"
            )
            if rule == "date":
                assert time_name is not None
                last, rows_after = _dates(_table(source).column(time_name))
                until = prompter.ask(
                    "Until",
                    last,
                    flag="--until",
                    note=lambda v: term.plural(rows_after(v), "row"),
                )
                context["until"] = until
            elif rule == "fraction":
                fraction = prompter.ask(
                    "Holdout",
                    _holdout_default(info.rows, cap),
                    flag="--holdout",
                    check=_fraction_problem,
                    note=lambda v: _holdout_note(float(v), info.rows, cap, model),
                )
                context["holdout"] = {"fraction": float(fraction), "seed": 0}

        name = (
            root.resolve().name
            if root.resolve().name
            and re.fullmatch(r"[a-z][a-z0-9-]{0,62}", root.resolve().name)
            else "predictor"
        )
        spec: dict[str, Any] = {"name": name, "model": model, "target": target}
        if identifier:
            spec["id"] = identifier
        if forecast:
            spec.update(
                time=time_column,
                series=list(series or ()),
                frequency=_number(frequency),
                horizon=horizon,
            )
        elif "until" in context:
            spec["time"] = time_name
        if inputs is not None:
            spec["inputs"] = inputs
        elif args.exclude:
            spec["inputs"] = {"exclude": list(args.exclude)}
        spec["context"] = context
        path = self.library.write_manifest(
            self.library.PredictorSpec.from_dict(spec), root, force=args.force
        )
        out.say(force=True)
        out.status("Initialized", f"predictor {name} in {_shown(root)}/{MANIFEST}")
        return {"name": name, "path": str(path), "manifest": spec}

    # -- check ------------------------------------------------------------

    def check(self) -> dict:
        """Resolves the manifest, prints what was inferred and writes `gf.lock`."""
        args, out = self.args, self.out
        started = time.perf_counter()
        root = self.root
        spec = self.library.load(root)
        if args.frozen:
            # The lock must be exactly what check would write now, as every
            # other reader of it requires; nothing is printed or written.
            self.library.verified(spec, root, online=args.online)
            out.status(
                "Verified",
                f"{LOCK} in {_shown(root)}",
                seconds=time.perf_counter() - started,
            )
            return {"name": spec.name, "frozen": True}
        resolution = self.library.resolve(
            spec,
            root,
            online=args.online,
            upgrade=args.upgrade,
        )
        self.print_resolution(resolution)
        out.status("Runs", _qualification(resolution).describe())
        out.table(self.roles(resolution, spec), indent=2)
        changes = _shown_changes(resolution.lock_changes, resolution.model)
        out.changes(changes)
        if args.locked:
            from generalforecast.predictors.lock import lock_document

            # No lock, another version or a digest that does not match the
            # content: nothing to compare with, so the lock would change.
            lock_document(root)
        if args.locked and changes:
            raise Error(
                "InvalidInput",
                f"{LOCK} would change",
                because=(
                    f"{len(changes)} change"
                    f"{' is' if len(changes) == 1 else 's are'}"
                    " listed above"
                ),
                hint="run `gf check` without --locked to write them",
            )
        if args.locked:
            out.status("Verified", LOCK, seconds=time.perf_counter() - started)
        else:
            self.library.write_lock(resolution, root)
            out.status("Locked", LOCK, seconds=time.perf_counter() - started)
        return _resolution_dict(resolution)

    def roles(self, resolution: Resolution, spec: Any) -> list[list[str]]:
        """The role table: key, value and a quiet note on where each came from."""
        declared = set(spec.marks)
        context = spec.context
        where = context.snapshot or context.live or context.kind
        if resolution.context.query_rows:
            where += (
                f"; {term.count(resolution.context.query_rows)} queries are "
                f"{_query_rule(spec, resolution.context.query_rows)}"
            )
        if spec.inputs is not None:
            chosen = f"listed in {MANIFEST}"
        else:
            chosen = "every other column"
            if spec.exclude:
                chosen += f" but {', '.join(spec.exclude)}"
        rows = []
        if resolution.id is not None:
            rows.append(["id", resolution.id, _from(declared, "id")])
        if resolution.task == "forecast":
            # A forecaster's covariates by role, as gf.lock records them: the
            # ones `inputs.future` lists bring values ahead; every other input
            # is a past covariate, known only up to each series' origin.
            by_role = dict(
                zip(resolution.input_names, resolution.input_roles, strict=True)
            )
            past = [name for name, role in by_role.items() if role == "past"]
            future = [name for name, role in by_role.items() if role == "future"]
            listed = f"listed in {MANIFEST}" if spec.past else chosen
            rows.append(["past", ", ".join(past) or "none", listed if past else ""])
            rows.append(
                [
                    "future",
                    ", ".join(future) or "none",
                    f"listed in {MANIFEST}" if future else "",
                ]
            )
        else:
            rows.append(["inputs", ", ".join(resolution.input_names), chosen])
        rows.append(["target", resolution.target, _from(declared, "target")])
        rows.append(
            ["context", f"{term.count(resolution.context.rows or 0)} rows", where]
        )
        return [[key, value, self.out.quiet_text(note)] for key, value, note in rows]

    def print_resolution(self, resolution: Resolution) -> None:
        """The `Resolved` and `Selection` status lines."""
        context = resolution.context
        self.out.status(
            "Resolved",
            f"{resolution.name}  {resolution.task}, {term.plural(len(resolution.inputs), 'input')}, "
            f"{term.count(context.rows or 0)} context rows, "
            f"{term.count(context.query_rows or 0)} query rows",
        )
        selection = resolution.selection
        if selection.strategy == "exact":
            text = f"whole table  {term.count(context.rows or 0)} rows, no neighbors"
        elif selection.strategy == "adaptive":
            text = selection.describe(context.rows)
        else:
            text = f"retrieved  core {term.count(selection.core or 0)} {selection.kind}, {term.count(selection.neighbors)} nearest per query"
        self.out.status("Selection", text)

    # -- eval -------------------------------------------------------------

    def eval(self) -> dict:
        """Scores the query rows with the predictor; returns the metrics."""
        args, out = self.args, self.out
        output = _output_path(args.output, args.queries, predictor=self.root)
        predictor = predictor_class()(str(self.root))
        resolution = predictor.explain()
        context = resolution.context
        out.status(
            "Resolved",
            f"{resolution.name}  {resolution.task}, {term.count(context.rows)} context rows, {term.count(context.query_rows)} query rows",
        )
        started = time.perf_counter()
        evaluation = predictor.eval(queries=args.queries, models=tuple(args.model))
        rows = (
            len(evaluation.rows) if evaluation.rows is not None else context.query_rows
        )
        out.status(
            "Predicted", term.plural(rows, "row"), seconds=time.perf_counter() - started
        )
        if output is not None:
            _write_frame(evaluation.rows, output)
            out.status("Wrote", args.output)
        return {"name": resolution.name, "metrics": evaluation.metrics, "rows": rows}

    # -- predict ----------------------------------------------------------

    def predict(self) -> dict:
        """Scores rows with a local, served, ad hoc or hosted predictor."""
        from generalforecast.predictor import resolve_reference

        args, out = self.args, self.out
        reference = args.target
        try:
            kind, where = resolve_reference(reference, env=args.env)
        except AmbiguousReference:
            raise AmbiguousReference(
                f"`{reference}` is both a directory with gf.yaml and a hosted predictor name",
                hint=f"pass ./{reference} for the local predictor, "
                f"or --env {DEFAULT_ENV} for the hosted one",
            ) from None
        if kind == "catalog" and not args.target_column:
            raise InvalidInput(
                "a catalog model needs a target",
                hint="pass '--target C' and optionally '--inputs C,C'",
                param="target",
            )
        if kind == "catalog" and args.context is None:
            raise InvalidInput(
                "a catalog model needs context rows",
                hint="pass '--context FILE' with rows whose target is known",
                param="context",
            )
        output = _output_path(
            args.output,
            args.queries,
            args.context,
            predictor=where if kind == "local" else None,
        )
        predictor = predictor_class()(
            reference,
            env=(args.env or DEFAULT_ENV) if kind == "hosted" else args.env,
            target=args.target_column,
            inputs=args.inputs,
        )
        started = time.perf_counter()
        frame = predictor.predict(args.queries, context=args.context)
        out.status(
            "Predicted",
            term.plural(len(frame), "row"),
            seconds=time.perf_counter() - started,
        )
        if output is not None:
            _write_frame(frame, output)
            out.status("Wrote", args.output)
        return {"rows": len(frame), "predictions": frame}

    # -- hosted -----------------------------------------------------------

    def push(self) -> dict:
        """Uploads the locked predictor as a revision and points an environment at it."""
        from generalforecast.contracts import Environment, PromoteRequest, Revision
        from generalforecast.service.contracts import ServiceError
        from generalforecast.service.sdk import pushed

        args, out = self.args, self.out
        root = self.root
        spec = self.library.load(root)
        if _previous_lock(root) is None:
            raise Error(
                "InvalidInput",
                f"no version 2 {LOCK} in {_shown(root)}",
                hint="run `gf check` first",
            )
        checked = pushed(root)
        lock = checked.lock.to_dict()
        predictor = lock["predictor"]
        name, env = spec.name, args.env
        try:
            self.call("GET", name)
            exists = True
        except ServiceError as error:
            if error.status != 404:
                raise
            exists = False
        current: Environment | None = None
        if exists:
            try:
                current = Environment.from_dict(
                    self.call("GET", name, f"/environments/{env}")
                )
            except ServiceError as error:
                if error.status != 404:
                    raise
        number = (current.revision or 0) + 1 if current else 1
        context = predictor.get("context") or {}
        resources = spec.resources
        accelerator = (resources.accelerator if resources else None) or "L4"
        hardware = (
            f"{accelerator}, "
            f"{(resources.replicas_min if resources else None) or 0}-"
            f"{(resources.replicas_max if resources else None) or 1} replicas, "
            f"idle {(resources.idle_seconds if resources else None) or 300}s"
        )
        plan = [
            ("name", name),
            ("revision", str(number)),
            ("environment", env),
            (
                "model",
                f"{predictor.get('model')}@{_short(predictor.get('model_digest', ''))}",
            ),
            (
                "context",
                f"{context.get('kind')}, {term.count(context.get('rows', 0))} rows"
                + (f", {_short(context['digest'])}" if context.get("digest") else ""),
            ),
            ("hardware", hardware),
            ("first call", _cost_line(accelerator)),
        ]
        # The plan is what a confirmation answers, so it shows whenever one is asked.
        out.table(plan, indent=2, force=not args.yes)
        self.confirm("Push?")
        snapshot = None
        if checked.snapshot is not None:
            from generalforecast.commands import upload_snapshot

            out.status(
                "Uploading",
                f"{name} revision {number}  snapshot, {term.count(checked.snapshot.num_rows)} rows, {_bytes(checked.snapshot.nbytes)}",
            )
            snapshot = upload_snapshot(self.client, checked.snapshot)["digest"]
            if snapshot != checked.lock.data_digest:
                raise Error(
                    "InvalidInput",
                    "the uploaded snapshot differs from gf.lock",
                    hint="run `gf check` again",
                )
        else:
            out.status("Uploading", f"{name} revision {number}  {context.get('kind')}")
        started = time.perf_counter()
        if not exists:
            self.client._call("POST", "/predictors", "predictor", name, {"name": name})
        revision = Revision.from_dict(
            self.call("POST", name, "/revisions", {"lock": lock, "snapshot": snapshot})
        )
        if current is None:
            current = Environment.from_dict(
                self.call(
                    "POST",
                    name,
                    "/environments",
                    {"name": env},
                )
            )
        request = PromoteRequest(revision.number, current.generation)
        promoted = Environment.from_dict(
            self.call("POST", name, f"/environments/{env}/promote", request.to_dict())
        )
        ready = self.wait(promoted)
        out.status(
            "Deployed",
            f"{name} revision {revision.number} to {env}",
            seconds=time.perf_counter() - started,
        )
        return {
            "name": name,
            "revision": revision.to_dict(),
            "environment": ready.to_dict(),
        }

    def wait(self, environment: Any) -> Any:
        """Polls the environment until it is ready, for at most `--wait` seconds.

        `--timeout` bounds each request; `--wait` bounds the whole start, which
        a cold start of a GPU replica can take most of a minute or more.
        """
        from generalforecast.contracts import Environment

        deadline = time.monotonic() + self.args.wait
        current = environment
        while current.status != "ready":
            if current.status in ("failed", "stopped"):
                raise Error(
                    "Internal",
                    f"environment {environment.name} failed to start",
                    hint=f"run `gf admin logs {environment.predictor} --env {environment.name}`",
                )
            if time.monotonic() >= deadline:
                raise Error(
                    "Internal",
                    f"environment {environment.name} is not ready yet",
                    hint=f"run `gf status {environment.predictor}` to follow it, or pass a longer '--wait'",
                )
            time.sleep(0.25)
            current = Environment.from_dict(
                self.call(
                    "GET", environment.predictor, f"/environments/{environment.name}"
                )
            )
        return current

    def status(self) -> dict:
        """The predictor's environments."""
        name, _ = self.reference()
        predictor = self.call("GET", name)
        environments = self.call("GET", name, "/environments")
        return {
            "name": name,
            "predictor": predictor,
            "environments": environments.get("environments", environments),
        }

    def promote(self) -> dict:
        """Points `--to` at the revision `--from` serves."""
        from generalforecast.contracts import Environment

        name, _ = self.reference()
        source = Environment.from_dict(
            self.call("GET", name, f"/environments/{self.args.source}")
        )
        if source.revision is None:
            raise Error(
                "InvalidInput",
                f"{self.args.source} serves no revision of {name}",
                hint=f"run `gf push --env {self.args.source}` first",
            )
        return self._point(name, self.args.env, source.revision, "Promoted")

    def rollback(self) -> dict:
        """Points `--env` at an earlier revision."""
        from generalforecast.contracts import Environment

        name, _ = self.reference()
        current = Environment.from_dict(
            self.call("GET", name, f"/environments/{self.args.env}")
        )
        revision = (
            self.args.revision
            if self.args.revision is not None
            else (current.revision or 1) - 1
        )
        if revision < 1:
            raise Error(
                "InvalidInput",
                f"{self.args.env} serves revision {current.revision} of {name}; there is nothing earlier",
                hint="pass '--revision N' to pick one",
            )
        return self._point(name, self.args.env, revision, "Pointed")

    def _point(self, name: str, env: str, revision: int, verb: str) -> dict:
        from generalforecast.contracts import Environment, PromoteRequest
        from generalforecast.service.contracts import ServiceError

        started = time.perf_counter()
        try:
            current = Environment.from_dict(
                self.call("GET", name, f"/environments/{env}")
            )
        except ServiceError as error:
            if error.status != 404:
                raise
            current = Environment.from_dict(
                self.call(
                    "POST",
                    name,
                    "/environments",
                    {"name": env},
                )
            )
        request = PromoteRequest(revision, current.generation)
        promoted = Environment.from_dict(
            self.call("POST", name, f"/environments/{env}/promote", request.to_dict())
        )
        ready = self.wait(promoted)
        self.out.status(
            verb,
            f"{env} at {name} revision {revision}"
            if verb == "Pointed"
            else f"{name} revision {revision} to {env}",
            seconds=time.perf_counter() - started,
        )
        return {"name": name, "environment": ready.to_dict()}

    def logs(self) -> dict:
        """An environment's recent log lines."""
        name, _ = self.reference()
        return self.call("GET", name, f"/environments/{self.args.env}/logs")

    def diff(self) -> dict:
        """The lock changes between this directory and what an environment serves."""
        from generalforecast.contracts import Environment

        name, root = self.reference()
        if root is None:
            raise Error(
                "InvalidInput",
                "diff needs a predictor directory",
                hint="run it inside the predictor, or pass its path",
            )
        spec = self.library.load(root)
        resolution = self.library.resolve(spec, root)
        environment = Environment.from_dict(
            self.call("GET", name, f"/environments/{self.args.env}")
        )
        if environment.revision is None:
            raise Error("NotFound", f"{self.args.env} serves no revision of {name}")
        remote = self.call("GET", name, f"/revisions/{environment.revision}")
        from generalforecast.predictors import lock

        changes = _shown_changes(
            lock.changes(remote.get("lock") or None, lock.document(resolution)),
            resolution.model,
        )
        if changes:
            self.out.changes(changes)
        else:
            self.out.status(
                "Unchanged",
                f"{name} matches {self.args.env} revision {environment.revision}",
            )
        return {
            "name": name,
            "revision": environment.revision,
            "changes": [list(c) for c in changes],
        }

    def delete(self) -> dict:
        """Deletes a hosted predictor and every environment of it."""
        name, _ = self.reference()
        self.confirm(f"Delete {name} and its environments?")
        result = self.call("DELETE", name)
        self.out.status("Deleted", f"predictor {name}")
        return result if isinstance(result, dict) else {"name": name, "deleted": True}


def _from(declared: set[str], key: str) -> str:
    return f"from {MANIFEST}" if key in declared else ""


HOLDOUT_FRACTION = 0.2


def _holdout_default(rows: int, cap: int) -> str:
    """The holdout `--yes` takes: a fifth of the rows, capped at one call's queries."""
    if round(HOLDOUT_FRACTION * rows) <= cap:
        return f"{HOLDOUT_FRACTION:g}"
    return f"{cap / rows:.4g}"


def _holdout_note(fraction: float, rows: int, cap: int, model: str) -> str:
    held = round(fraction * rows)
    note = term.plural(held, "row")
    if held == cap and round(HOLDOUT_FRACTION * rows) > cap:
        note += f", the most {model} scores in one call"
    return note


def _fraction_problem(text: str) -> str | None:
    """What is wrong with a typed holdout fraction, or None."""
    try:
        value = float(text)
    except ValueError:
        return "must be a number between 0 and 1, like 0.2"
    return None if 0 < value < 1 else "must be between 0 and 1, like 0.2"


def _horizon_problem(text: str) -> str | None:
    """What is wrong with a typed horizon, or None."""
    return (
        None
        if text.isascii() and text.isdigit() and int(text) >= 1
        else "must be a whole number of steps"
    )


def _dates(column: Any) -> tuple[str, Callable[[str], int]]:
    """A time column's last date, and a count of its rows at or after a date.

    The column is read once; the count runs on every keystroke of the prompt.
    """
    last = pc.max(column).as_py()
    shown = (
        str(last)[:10]
        if last is not None
        else datetime.datetime.now(tz=datetime.UTC).date().isoformat()
    )
    text = column.cast("string")

    def after(value: str) -> int:
        return int(pc.sum(pc.greater_equal(text, value)).as_py() or 0)

    return shown, after


def _number(value: Any) -> Any:
    if isinstance(value, str) and re.fullmatch(r"\d+", value):
        return int(value)
    return value


def _bytes(size: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1000
    return f"{size:.1f} GB"


def _cost_line(accelerator: str) -> str:
    rates = {"cpu": 0.0002, "L4": 0.0004, "A10G": 0.0006, "L40S": 0.0010}
    starts = {"cpu": 15, "L4": 40, "A10G": 40, "L40S": 45}
    return f"about ${rates.get(accelerator, 0.0004):.4f} after a cold start of about {starts.get(accelerator, 40)}s"


def _qualification(resolution: Resolution) -> Any:
    from generalforecast.predictors.models import qualification

    return qualification(resolution.model)


def _resolution_dict(resolution: Resolution) -> dict:
    return {
        "name": resolution.name,
        "model": resolution.model,
        "model_digest": resolution.model_digest,
        "task": resolution.task,
        "target": resolution.target,
        "target_type": resolution.target_type,
        "id": resolution.id,
        "inputs": [
            {"name": n, "type": t, "role": r}
            for (n, t), r in zip(resolution.inputs, resolution.input_roles, strict=True)
        ],
        "context": {
            "kind": resolution.context.kind,
            "rows": resolution.context.rows,
            "query_rows": resolution.context.query_rows,
            "digest": resolution.context.digest,
        },
        "selection": resolution.selection.to_dict(),
        "runs": _qualification(resolution).to_dict(),
        "changes": [list(c) for c in resolution.lock_changes],
    }
