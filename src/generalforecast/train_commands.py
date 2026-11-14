# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fine-tuning commands: train and runs.

`gf train` reads a local table, asks for what its flags and the table do
not settle, uploads the table as a snapshot and starts a run on the hosted
service, then follows the run's lineage until it ends. `gf runs` lists the
project's runs, follows one again and cancels one. Each command returns its
result for `generalforecast.render` to write.
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, get_args

from generalforecast import term
from generalforecast.errors import Conflict, Error, InvalidInput, NotFound, printable
from generalforecast.models.derived import KINDS as METHODS


@dataclass(frozen=True)
class FineTunable:
    """A base the service fine-tunes.

    Attributes:
        methods: The methods a run on it takes, of `METHODS`.
        predictors: Whether hosted predictors serve its fine-tunes, as well
            as the Models API.
    """

    methods: tuple[str, ...]
    predictors: bool

    @property
    def served(self) -> str:
        """Where its fine-tunes serve, as the base selector says."""
        return "predictors and API" if self.predictors else "API only"


# The bases the service fine-tunes and the methods each takes. This mirrors
# the service's rule (gfi `FINE_TUNABLE`), which refuses any other base, or
# a method its base does not take, with 422 UnsupportedCapability; gf train
# offers only what that rule accepts. TabICL v2 comes first: it is the
# default, and the one base whose fine-tunes hosted predictors serve.
FINE_TUNABLE: Mapping[str, FineTunable] = MappingProxyType(
    {
        **{
            f"general-forecast/tabicl-v2-{kind}": FineTunable(METHODS, True)
            for kind in ("classifier", "regressor")
        },
        **{
            f"general-forecast/kumo-tabular-{size}-{kind}": FineTunable(METHODS, False)
            for size in ("small", "medium", "large")
            for kind in ("classifier", "regressor")
        },
        **{
            f"general-forecast/mitra-2-{kind}": FineTunable(("lora",), False)
            for kind in ("classifier", "regressor")
        },
    }
)
TASKS = ("classification", "regression")

# The plan the quickstart uses and the benchmark validated.
STEPS = 400
LEARNING_RATE = 3e-6
# The learning rate of full fine-tuning, which the benchmark validated apart.
FULL_LEARNING_RATE = 1e-7
LORA_RANK = 16
EPISODE_ROWS = 8192
CONTEXT_FRACTION = 0.9375
EVAL_STEPS = 25
PATIENCE = 6
HOLDOUT = 0.2
BUDGET_GPU_HOURS = 2.0
# How many standard errors above 0 a check's gain must be for its step to be
# kept, when the run does not say: the service's default.
KEEP_STANDARD_ERRORS = 2.0
# The longest name a snapshot's prefix takes, so `<prefix>-<12 hex>` is a name.
SNAPSHOT_PREFIX = 50

STATE_VERBS = {
    "queued": "Queued",
    "setting_up": "Setting up",
    "training": "Training",
    "preempted": "Preempted",
    "cancelling": "Cancelling",
    "finished": "Finished",
    "cancelled": "Cancelled",
}
_NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")


def components() -> tuple[str, ...]:
    """The components full fine-tuning trains by name, as the API spells them."""
    from generalforecast._api import RunComponent

    return get_args(RunComponent)


def _registry() -> Any:
    from generalforecast.commands import model_registry

    return model_registry()


def _table(path: Path) -> Any:
    from generalforecast.commands import read_data

    return read_data(str(path), Path.cwd())


def default_name(path: Path) -> str:
    """The fine-tune name a table's file suggests: its stem as a lowercase name."""
    from generalforecast.models.catalog import entries

    stem = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    stem = re.sub(r"^[^a-z]+", "", stem)[:63].rstrip("-") or "fine-tune"
    if stem in {entry.name for entry in entries()}:
        stem = f"{stem[:60]}-ft"
    return stem


def name_problem(text: str) -> str | None:
    """What is wrong with a fine-tune name, or None."""
    from generalforecast.models.catalog import entries

    if not _NAME.fullmatch(text):
        return "must be lowercase letters, digits and dashes, starting with a letter"
    if text in {entry.name for entry in entries()}:
        return "is a catalog model's name"
    return None


def snapshot_name(name: str, digest: str) -> str:
    """The snapshot a table is uploaded as: `<name>-<first 12 hex of its digest>`.

    The rule `gf push` names its snapshots by, so the same table under the
    same name is the same snapshot; a name is cut to keep the whole a name.
    """
    from generalforecast._operations import snapshot_name as named

    return named(name[:SNAPSHOT_PREFIX].rstrip("-"), digest)


def method_text(method: Mapping[str, Any]) -> str:
    """A run's method as a person reads it: `lora rank 16`, `full column_encoder, ...`."""
    kind = str(method.get("type"))
    if kind == "lora":
        text = f"lora rank {method.get('rank', LORA_RANK)}"
        return text + (" with an input adapter" if method.get("input_adapter") else "")
    if kind == "full":
        return f"full {', '.join(method.get('components') or components())}"
    return kind


def plan_text(plan: Mapping[str, Any]) -> str:
    """The plan line: steps, learning rate, episodes, checks and patience."""
    episodes = plan.get("episode_sampling") or {}
    return (
        f"{term.count(plan['steps'])} steps at learning rate {plan['learning_rate']:g}, "
        f"{term.count(episodes['rows'])}-row episodes, "
        f"a check every {plan['eval_steps']} steps, "
        f"patience {plan['early_stopping_patience']}"
    )


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _percent(value: float) -> str:
    return f"{value:+.1f}%"


class Follower:
    """Writes a run's lineage events as status lines, as they arrive.

    A check's line gives its step, the select rows' loss and the gain over
    the base with its standard error, and marks a step kept by the
    service's rule: a gain more than `threshold` standard errors above 0,
    and a select loss below every step kept before it.
    """

    def __init__(self, out: term.Output, model: str, threshold: float) -> None:
        """Binds the renderer, the run's model and its keep threshold."""
        self.out, self.model, self.threshold = out, model, threshold
        self.best: float | None = None
        self.failure: str | None = None

    def kept(self, body: Mapping[str, Any]) -> bool:
        """Whether the check in a metrics event keeps its step."""
        gain = body.get("check_gain")
        error = body.get("check_standard_error")
        loss = body.get("validation_loss")
        if gain is None or error is None or loss is None:
            return False
        if gain <= self.threshold * error:
            return False
        if self.best is not None and loss >= self.best:
            return False
        self.best = loss
        return True

    def check(self, body: Mapping[str, Any]) -> str | None:
        """A check's status message, or None for a stretch of training with no check."""
        loss = body.get("validation_loss")
        if loss is None:
            return None
        text = f"step {term.count(int(body.get('steps', 0)))}  select loss {loss:.4f}"
        gain, error = body.get("check_gain"), body.get("check_standard_error")
        if gain is not None and error is not None:
            text += f", gain {_signed(gain)} ± {error:.4f}"
        if self.kept(body):
            text += f", {self.out.style('kept', 'bold', 'green')}"
        return text

    def event(self, event: Mapping[str, Any]) -> None:
        """Writes one lineage event; a failure's message is kept for the error."""
        kind, body = event.get("kind"), event.get("body") or {}
        if kind == "state":
            state = body.get("state")
            reason = printable(str(body.get("reason") or ""))
            verb = STATE_VERBS.get(str(state))
            if verb is not None:
                self.out.status(verb, self.model + (f"  {reason}" if reason else ""))
        elif kind == "metrics":
            text = self.check(body)
            if text is not None:
                self.out.status("Checked", text)
        elif kind == "resumed":
            self.out.status("Resumed", self.model)
        elif kind == "published":
            model = printable(str(body.get("model")))
            self.out.status("Published", f"{model}  {_kept(body)}")
        elif kind == "failure":
            found = body.get("message") or body.get("error")
            self.failure = printable(str(found)) if found else None

    def follow(self, events: Iterable[Mapping[str, Any]]) -> None:
        """Writes every event; Ctrl-C stops following and leaves the run running."""
        try:
            for event in events:
                self.event(event)
        except KeyboardInterrupt:
            self.out.say(force=True)
            self.out.say(
                f"the run carries on; follow it again with `gf runs watch {self.model}`",
                force=True,
            )
            raise


def _kept(published: Mapping[str, Any]) -> str:
    step = published.get("kept_step")
    if step is None:
        return "no step kept"
    if step == 0:
        return "kept step 0, no check beat the base"
    return f"kept step {term.count(int(step))}"


def threshold(run: Mapping[str, Any]) -> float:
    """How many standard errors above 0 a check's gain must be to keep its step."""
    evaluation = run.get("evaluation") or {}
    found = (evaluation.get("early_stopping_threshold") or {}).get("standard_errors")
    return KEEP_STANDARD_ERRORS if found is None else float(found)


def report_rows(run: Mapping[str, Any]) -> tuple[list[tuple[str, str]], str | None]:
    """What a run published, as rows, and the `next:` step that serves it.

    The rows name the version and its kept step, then the report: its
    rows, the base's and the kept step's loss on them (log loss for a
    classifier), and the gain relative to the base with its 95% interval.
    """
    published = run.get("published")
    if not published:
        return [], None
    model = printable(str(published.get("model")))
    step = published.get("kept_step")
    rows = [
        ("version", model),
        ("kept step", "-" if step is None else term.count(int(step))),
    ]
    report = (published.get("report") or {}).get("data")
    if report:
        base, kept = float(report["base"]), float(report["kept"])
        classifier = "log_loss" in ((report.get("metrics") or {}).get("base") or {})
        rows.append(("report rows", term.count(int(report["rows"]))))
        rows.append(
            ("log loss" if classifier else "loss", f"{base:.4f} base, {kept:.4f} kept")
        )
        if base:
            low, high = (_percent(100 * end / base) for end in report["ci95"])
            gain = _percent(100 * report["gain"] / base)
            rows.append(("gain", f"{gain}, 95% CI {low} to {high}"))
    tunable = FINE_TUNABLE.get(str(run.get("base")))
    if tunable is not None and tunable.predictors:
        step_text = f"put `model: {model}` in gf.yaml, then run `gf push`"
    else:
        step_text = f"predict with `{model}` through the Models API"
    return rows, step_text


def execute(args: argparse.Namespace, out: term.Output) -> dict | None:
    """Runs `gf train` or `gf runs <action>`."""
    commands = TrainCommands(args, out)
    if args.command == "train":
        return commands.train()
    return getattr(commands, args.action)()


class TrainCommands:
    """The fine-tuning commands, bound to one invocation's arguments and renderer."""

    def __init__(self, args: argparse.Namespace, out: term.Output) -> None:
        """Binds the arguments and the renderer."""
        self.args, self.out = args, out

    @cached_property
    def client(self) -> Any:
        """The authenticated SDK client, built on first use."""
        from generalforecast.commands import client_for

        return client_for(self.args)

    def source(self) -> Path:
        """The table TABLE names, else the one table in this directory."""
        from generalforecast.predictor_commands import _tables_in

        if self.args.table is not None:
            path = Path(self.args.table).expanduser()
            if not path.is_file():
                raise NotFound(
                    f"table `{self.args.table}` does not exist",
                    hint="pass a .csv, .parquet or .json file",
                )
            return path
        tables = _tables_in(Path.cwd())
        if not tables:
            raise NotFound(
                "no table here to train on",
                hint="pass TABLE, a .csv, .parquet or .json file",
            )
        if len(tables) > 1:
            raise InvalidInput(
                f"this directory has {len(tables)} tables",
                because="training reads one: " + ", ".join(t.name for t in tables),
                hint="pass TABLE to pick one",
            )
        return tables[0]

    def train(self) -> dict:
        """Uploads a table and fine-tunes a base on it, asking for what the flags leave open."""
        from generalforecast.predictors.tables import (
            inspect_frame,
            target_candidates,
            task_for,
        )
        from generalforecast.tabular import table_digest

        args, out = self.args, self.out
        path = self.source()
        table = _table(path)
        info = inspect_frame(table, path=str(path))
        out.status(
            "Read",
            f"{path.name}  {term.plural(info.rows, 'row')}, "
            f"{term.plural(len(info.columns), 'column')}",
        )
        out.say()
        prompter = term.Prompter(out, yes=args.yes, no_input=args.no_input)
        registry = _registry()
        columns = {c.name: term.Option(c.name, c.name, c.kind) for c in info.columns}

        base = args.base
        declared: tuple[str, str] | None = None
        if base is not None:
            base = registry.canonical(base)
            _tunable(base)
            declared = registry.task_of(base), "from --base"
            if args.task is not None and args.task != declared[0]:
                raise InvalidInput(
                    f"{base} does {declared[0]}, not {args.task}",
                    hint="drop '--task', or pass a --base that does it",
                    param="task",
                )
        elif args.task is not None:
            declared = args.task, "from --task"

        target = args.target_column
        if target is not None:
            if target not in columns:
                raise InvalidInput(
                    f"target column `{target}` is not in {path.name}",
                    hint=f"{path.name} has {', '.join(info.names)}",
                    param="target",
                )
            out.collapsed("Target", target)
        else:
            tasks = (declared[0],) if declared is not None else TASKS
            names = dict.fromkeys(
                name for task in tasks for name in target_candidates(info, task)
            )
            if not names:
                raise InvalidInput(
                    f"no column of {path.name} can be the target",
                    hint="pass '--target C' with a column of the table",
                    param="target",
                )
            target = prompter.choose(
                "Target", [columns[c] for c in names], flag="--target"
            )

        column = info.column(target)
        if declared is not None:
            task, why = declared
        else:
            task = task_for(column, info.rows)
            why = f"{target} has {term.count(column.distinct)} distinct {column.kind} values"
        out.collapsed("Task", f"{task}  {out.quiet_text(why)}")

        bases = [ref for ref in FINE_TUNABLE if registry.task_of(ref) == task]
        if base is None:
            base = prompter.choose(
                "Base",
                [
                    term.Option(
                        ref,
                        ref,
                        FINE_TUNABLE[ref].served,
                        f"{ref}  {out.quiet_text(FINE_TUNABLE[ref].served)}",
                    )
                    for ref in bases
                ],
                flag="--base",
            )
        else:
            out.collapsed(
                "Base", f"{base}  {out.quiet_text(FINE_TUNABLE[base].served)}"
            )

        method = self.method(prompter, base)

        if args.name is not None:
            if (wrong := name_problem(args.name)) is not None:
                raise InvalidInput(f"name {wrong}", param="name")
            name = args.name
            out.collapsed("Name", name)
        else:
            name = prompter.ask(
                "Name", default_name(path), flag="--name", check=name_problem
            )

        plan, evaluation, budget = self.plan(method)
        out.say()
        out.table(
            [
                ("plan", plan_text(plan)),
                ("holdout", f"{evaluation['holdout']:g} of the rows"),
                ("budget", f"{budget['gpu_hours']:g} GPU hours"),
            ],
            indent=2,
            force=not args.yes,
        )
        prompter.confirm("Train?")

        snapshot = self.snapshot(
            snapshot_name(name, table_digest(table)), table, target=target, task=task
        )
        run = self.client.runs.create(
            name,
            base=base,
            data=[snapshot],
            method=method,
            plan=plan,
            evaluation=evaluation,
            budget=budget,
        )
        model = printable(str(run["model"]))
        out.status("Started", f"{model}  {base}, {method_text(method)}")
        if args.detach:
            return run
        return self.follow(run)

    def method(self, prompter: term.Prompter, base: str) -> dict[str, Any]:
        """The run's method: `--method` and its options, else chosen among the base's."""
        method = self._method(prompter, base)
        if self.args.method is not None:
            self.out.collapsed("Method", method_text(method))
        return method

    def _method(self, prompter: term.Prompter, base: str) -> dict[str, Any]:
        args = self.args
        taken = FINE_TUNABLE[base].methods
        kind = args.method
        if kind is None:
            details = {
                "lora": f"low-rank adapters, rank {args.rank or LORA_RANK}",
                "input_adapter": "an adapter in front of the inputs",
                "full": "every weight of the components named",
            }
            shown = {
                "type": None,
                "rank": args.rank or LORA_RANK,
                "input_adapter": args.input_adapter,
                "components": args.components,
            }
            kind = prompter.choose(
                "Method",
                [
                    term.Option(m, m, details[m], method_text(shown | {"type": m}))
                    for m in taken
                ],
                flag="--method",
            )
        elif kind not in taken:
            raise Error(
                "UnsupportedCapability",
                f"{base} does not take {kind} fine-tuning",
                hint=f"pass '--method' with one of {', '.join(taken)}",
                param="method",
            )
        for flag, value, wanted in (
            ("--rank", args.rank, "lora"),
            ("--input-adapter", args.input_adapter or None, "lora"),
            ("--components", args.components, "full"),
        ):
            if value is not None and kind != wanted:
                raise InvalidInput(
                    f"{flag} applies to {wanted} fine-tuning, not {kind}",
                    param=flag.lstrip("-"),
                )
        if kind == "lora":
            rank = LORA_RANK if args.rank is None else args.rank
            if rank < 1:
                raise InvalidInput("--rank must be 1 or more", param="rank")
            method: dict[str, Any] = {"type": "lora", "rank": rank}
            if args.input_adapter:
                if "input_adapter" not in taken:
                    raise Error(
                        "UnsupportedCapability",
                        f"{base} does not take an input adapter",
                        param="input-adapter",
                    )
                method["input_adapter"] = True
            return method
        if kind == "full":
            every = components()
            chosen = list(args.components or every)
            unknown = [c for c in chosen if c not in every]
            if unknown:
                raise InvalidInput(
                    f"`{unknown[0]}` is not a component",
                    hint=f"name some of {', '.join(every)}",
                    param="components",
                )
            return {"type": "full", "components": chosen}
        return {"type": kind}

    def plan(self, method: Mapping[str, Any]) -> tuple[dict, dict, dict]:
        """The run's plan, evaluation and budget, from the flags and their defaults."""
        args = self.args
        rate = args.learning_rate
        if rate is None:
            rate = FULL_LEARNING_RATE if method["type"] == "full" else LEARNING_RATE
        for flag, value in (
            ("--steps", args.steps),
            ("--episode-rows", args.episode_rows),
            ("--eval-steps", args.eval_steps),
            ("--patience", args.patience),
        ):
            if value is not None and value < 1:
                raise InvalidInput(f"{flag} must be 1 or more", param=flag.lstrip("-"))
        if not rate > 0:
            raise InvalidInput("--learning-rate must be above 0", param="learning-rate")
        holdout = HOLDOUT if args.holdout is None else args.holdout
        if not 0 < holdout < 1:
            raise InvalidInput("--holdout must be between 0 and 1", param="holdout")
        hours = (
            BUDGET_GPU_HOURS if args.budget_gpu_hours is None else args.budget_gpu_hours
        )
        if not hours > 0:
            raise InvalidInput(
                "--budget-gpu-hours must be above 0", param="budget-gpu-hours"
            )
        plan = {
            "steps": args.steps or STEPS,
            "learning_rate": rate,
            "episode_sampling": {
                "rows": args.episode_rows or EPISODE_ROWS,
                "context_fraction": CONTEXT_FRACTION,
            },
            "eval_steps": args.eval_steps or EVAL_STEPS,
            "early_stopping_patience": args.patience or PATIENCE,
        }
        return plan, {"holdout": holdout}, {"gpu_hours": hours}

    def snapshot(self, name: str, table: Any, *, target: str, task: str) -> str:
        """Uploads the table as snapshot `name`, or reuses the one already named so.

        The name holds the table's digest, so a ready snapshot of that name
        holds this table; it is reused when it was uploaded for the same
        target and task.
        """
        out = self.out
        try:
            found = self.client.snapshots.get(name)
        except NotFound:
            found = None
        if found is not None and found.get("state") in (
            "deleted",
            "rejected",
            "expired",
        ):
            raise Conflict(
                f"snapshot {name} was {found['state']}, and its name is not reused",
                hint="pass '--name' to train under another name",
                param="name",
            )
        if found is not None and found.get("state") == "ready":
            metadata = found.get("metadata") or {}
            if (metadata.get("target"), metadata.get("task")) != (target, task):
                raise Conflict(
                    f"snapshot {name} holds this table for target "
                    f"`{metadata.get('target')}` ({metadata.get('task')})",
                    hint="pass '--name' to train another fine-tune on it",
                    param="target",
                )
            out.status(
                "Reused",
                f"snapshot {name}  {term.plural(int(found.get('rows') or 0), 'row')}",
            )
            return name
        started = time.perf_counter()
        made = self.client.snapshots.create(name, table, target=target, task=task)
        out.status(
            "Uploaded",
            f"snapshot {name}  {term.plural(int(made.get('rows') or table.num_rows), 'row')}",
            seconds=time.perf_counter() - started,
        )
        return name

    def follow(self, run: Mapping[str, Any]) -> dict:
        """Follows a run's events until it ends, and returns it as it ended.

        A failed run raises, with the reason the service gave.
        """
        model = printable(str(run["model"]))
        follower = Follower(self.out, model, threshold(run))
        follower.follow(self.client.runs.watch(model))
        ended = dict(self.client.runs.get(model))
        if ended.get("state") == "failed":
            raise Error(
                "Internal",
                f"run {model} failed"
                + (
                    f": {printable(str(ended['reason']))}"
                    if ended.get("reason")
                    else ""
                ),
                because=follower.failure,
            )
        return ended

    def list(self) -> dict:
        """The project's runs, only those in `--state` when it is given."""
        return {"runs": self.client.runs.list(state=self.args.state)}

    def watch(self) -> dict:
        """Follows a run again, from its first event, until it ends."""
        return self.follow(self.client.runs.get(self.args.model))

    def cancel(self) -> dict:
        """Cancels a run; with `--save`, a training run publishes what it trained."""
        model = self.args.model
        term.Prompter(self.out, yes=self.args.yes).confirm(f"Cancel run {model}?")
        run = dict(self.client.runs.cancel(model, save=self.args.save))
        verb = STATE_VERBS.get(str(run.get("state")), "Cancelled")
        self.out.status(verb, str(run.get("model", model)))
        return run


def _tunable(base: str) -> FineTunable:
    """The base's fine-tuning entry; any other base is refused as the service refuses it."""
    found = FINE_TUNABLE.get(base)
    if found is None:
        raise Error(
            "UnsupportedCapability",
            f"{base} cannot be fine-tuned",
            hint=f"fine-tune one of {', '.join(FINE_TUNABLE)}",
            param="base",
        )
    return found


def run_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Runs as `gf runs list` shows them, one record each."""
    return [
        {
            "model": run.get("model"),
            "state": run.get("state"),
            "base": run.get("base"),
            "method": method_text(run.get("method") or {}),
            "kept step": (run.get("published") or {}).get("kept_step"),
        }
        for run in runs
    ]
