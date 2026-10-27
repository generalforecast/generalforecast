# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Renders every `gf` command and error shape in a 100-column pseudo-terminal.

    python -m tests.product_cli.render_suite OUTDIR [--png]

Each case runs the CLI in a child process under a pty, against fixtures
written to a temporary directory, with the model registry, the SDK's
`Predictor` and the hosted service replaced by fakes whose numbers match the
interface contract; the predictor library is the real one, and the fixture
locks are what it wrote under those fakes, made stale on purpose. The child's output is kept with
its ANSI codes (`<case>.ansi`), replayed through a small terminal emulator to
plain text (`<case>.txt`), joined into `suite.txt`, and, with `--png` and
Pillow installed, drawn with a monospace font (`<case>.png`).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import pty
import random
import re
import select
import struct
import subprocess
import sys
import termios
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

COLUMNS = 100
ROWS = 60
ROOT = Path(__file__).resolve().parents[2]
TABICL = "general-forecast/tabicl-v2-classifier"
KUMO = "general-forecast/kumo-tabular-medium-classifier"
DIGEST = "sha256:7a01c2e" + "0" * 57
SERVED_DIGEST = "sha256:5fb6a4c" + "0" * 57
REQUEST_ID = "r-3abb2401f119428db148927c6a0d358c"


@dataclass(frozen=True)
class Case:
    """One rendered invocation."""

    name: str
    argv: tuple[str, ...]
    cwd: str = "churn"
    keys: tuple[tuple[str, bytes], ...] = ()
    seconds: tuple[float, ...] = (0.0, 0.21)
    snapshot_after: str | None = None
    env: dict[str, str] = field(default_factory=dict)


CASES: tuple[Case, ...] = (
    Case("help", ("--help",)),
    Case("help-init", ("init", "--help")),
    Case("help-check", ("check", "--help")),
    Case("help-eval", ("eval", "--help")),
    Case("help-serve", ("serve", "--help")),
    Case("help-predict", ("predict", "--help")),
    Case("help-login", ("login", "--help")),
    Case("help-logout", ("logout", "--help")),
    Case("help-whoami", ("whoami", "--help")),
    Case("help-push", ("push", "--help")),
    Case("help-status", ("status", "--help")),
    Case("help-promote", ("promote", "--help")),
    Case("help-rollback", ("rollback", "--help")),
    Case("help-model", ("model", "--help")),
    Case("help-context", ("context", "--help")),
    Case("help-admin", ("admin", "--help")),
    Case("help-admin-key", ("admin", "key", "--help")),
    Case("help-admin-node", ("admin", "node", "serve", "--help")),
    Case(
        "init",
        ("init",),
        cwd="init/churn",
        keys=(
            ("Target", b"\r"),
            ("Model", b"\r"),
            ("Id", b"\r"),
            ("Inputs", b"\r"),
            ("Queries", b"\r"),
        ),
        snapshot_after="Target",
    ),
    Case("init-yes", ("init", "--yes"), cwd="init-yes/churn"),
    Case(
        "init-forecast",
        ("init", "--target", "units", "--task", "forecast"),
        cwd="init-forecast/sales",
        keys=(
            ("Time", b"\r"),
            ("Series", b"\r"),
            ("Frequency", b"\r"),
            ("Horizon", b"\x7f"),
            ("Horizon", b"\r"),
            ("Horizon", b"5"),
            ("Horizon", b"\r"),
            ("Model", b"\r"),
            ("Id", b"\r"),
            ("Queries", b"\r"),
            ("Until", b"\r"),
        ),
    ),
    Case(
        "init-holdout",
        ("init", "--target", "churned", "--id", "customer_id", "--model", TABICL),
        cwd="init-holdout/churn",
        keys=(
            ("Inputs", b"\r"),
            ("Queries", b"\r"),
            *(("Holdout", b"\x7f") for _ in range(7)),
            *(("Holdout", key.encode()) for key in "0.03"),
            ("Holdout", b"\r"),
        ),
    ),
    Case(
        "init-flags",
        (
            "init",
            "--target",
            "churned",
            "--model",
            TABICL,
            "--id",
            "customer_id",
            "--holdout",
            "0.2",
            "--yes",
        ),
        cwd="init-flags/churn",
    ),
    Case("init-exists", ("init", "churn"), cwd="."),
    Case("check", ("check",), cwd="orders", seconds=(0.0, 0.21)),
    Case("check-whole", ("check",), seconds=(0.0, 0.21)),
    Case("check-typo", ("check",), cwd="typo"),
    Case("check-cadence", ("check",), cwd="sales"),
    Case("check-bare", ("check",), cwd="bare"),
    Case("check-locked", ("check", "--locked"), cwd="orders-stale"),
    Case("eval", ("eval",), seconds=(0.0, 0.19)),
    Case(
        "eval-models",
        (
            "eval",
            "--model",
            "general-forecast/mitra-2-classifier",
            "--output",
            "scored.parquet",
        ),
        seconds=(0.0, 0.19),
    ),
    Case(
        "predict",
        ("predict", "--queries", "queries.csv", "--output", "scored.parquet"),
        seconds=(0.0, 0.21),
    ),
    Case(
        "predict-capacity",
        ("predict", "churn", "--queries", "queries.csv"),
        cwd="fresh",
    ),
    Case(
        "predict-json",
        ("--json", "predict", "churn", "--queries", "queries.csv"),
        cwd="fresh",
    ),
    Case("push", ("push", "--env", "staging", "--yes"), seconds=(0.0, 41.0)),
    Case("push-ask", ("push",), keys=(("Push?", b"y\n"),), seconds=(0.0, 41.0)),
    Case("status", ("status", "churn"), cwd="fresh"),
    Case("promote", ("promote", "churn"), cwd="fresh", seconds=(0.0, 3.2)),
    Case("rollback", ("rollback", "churn"), cwd="fresh", seconds=(0.0, 2.8)),
    Case("whoami", ("whoami",)),
    Case("model-list", ("model", "list", "--task", "classification")),
    Case("model-show", ("model", "show", TABICL)),
    Case("model-pull", ("model", "pull", TABICL)),
    Case("model-bare", ("model", "show", "tabicl-classifier")),
    Case("context-show", ("context", "show", "orders-live")),
    Case(
        "context-append",
        ("context", "append", "orders-live", "queries.csv", "--id", "customer_id"),
    ),
    Case("admin-key-list", ("admin", "key", "list")),
    Case("admin-logs", ("admin", "logs", "churn"), cwd="fresh"),
    Case("admin-diff", ("admin", "diff")),
    Case("error-unknown-command", ("chekc",)),
    Case("error-unknown-argument", ("init", "--contxt", "x.csv")),
    Case("error-missing-argument", ("predict",)),
    Case("error-internal", ("eval",), env={"GF_SUITE_INTERNAL": "1"}),
    Case("error-internal-verbose", ("-v", "eval"), env={"GF_SUITE_INTERNAL": "1"}),
    Case("error-json", ("--json", "check"), cwd="typo"),
)


# -- fixtures ---------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _churn_rows(
    count: int = 2052, queries: int = 212, *, target: str = "churned"
) -> tuple[list[str], list[list[Any]]]:
    rng = random.Random(0)
    rows = []
    for i in range(count):
        churned = "" if i >= count - queries else rng.choice(["0", "1"])
        rows.append(
            [
                1000 + i,
                round(rng.uniform(5, 500), 2),
                rng.choice(["basic", "pro", "team"]),
                rng.randint(0, 40),
                churned,
            ]
        )
    return ["customer_id", "spend", "plan", "signups", target], rows


def write_fixtures(root: Path) -> None:
    """Writes every predictor directory the cases use."""
    import yaml

    churn = root / "churn"
    churn.mkdir(parents=True)
    header, rows = _churn_rows()
    _write_csv(churn / "context.csv", header, rows)
    _write_csv(churn / "queries.csv", header[:-1], [r[:-1] for r in rows[-1000:]] * 1)
    (churn / "gf.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "churn",
                "model": TABICL,
                "target": "churned",
                "id": "customer_id",
                "context": {"snapshot": "context.csv"},
            },
            sort_keys=False,
        )
    )
    lock(churn)
    for name in ("init", "init-yes", "init-flags", "fresh"):
        fresh = root / name / "churn" if name != "fresh" else root / name
        fresh.mkdir(parents=True)
        _write_csv(fresh / "context.csv", header, rows)
    labelled = root / "init-holdout" / "churn"
    labelled.mkdir(parents=True)
    _write_csv(labelled / "context.csv", *_churn_rows(queries=0))
    series = root / "init-forecast" / "sales"
    series.mkdir(parents=True)
    _write_csv(
        series / "context.csv",
        ["date", "store", "units"],
        [
            [f"2025-01-{day:02d}", store, day * 2 + (store == "a")]
            for day in range(1, 31)
            for store in ("a", "b")
        ],
    )

    typo = root / "typo"
    typo.mkdir()
    header, rows = _churn_rows(target="churn")
    _write_csv(
        typo / "context.csv",
        ["customer_id", "spend", "plan", "churn"],
        [[r[0], r[1], r[2], r[4]] for r in rows],
    )
    (typo / "gf.yaml").write_text(
        f"name: churn\nmodel: {TABICL}\ntarget: churned\ncontext:\n  snapshot: context.csv\n"
    )

    sales = root / "sales"
    sales.mkdir()
    _write_csv(
        sales / "context.csv",
        ["date", "sku", "units"],
        [[f"2025-01-{d:02d}", "a", d] for d in range(1, 29)],
    )
    (sales / "gf.yaml").write_text(
        "name: sales\nmodel: general-forecast/chronos-2\ntarget: units\ncontext:\n  snapshot: context.csv\n"
    )

    bare = root / "bare"
    bare.mkdir()
    _write_csv(bare / "context.csv", header, rows)
    (bare / "gf.yaml").write_text(
        "name: bare\nmodel: tabicl-classifier\ntarget: churn\ncontext:\n  snapshot: context.csv\n"
    )

    for orders in (root / "orders", root / "orders-stale"):
        _write_orders(orders)


def lock(root: Path, stale: Any = None) -> dict:
    """Writes `gf.lock` as `gf check` would under the fakes; `stale` edits it first.

    Returns:
        The lock document written.
    """
    import yaml

    from generalforecast import predictors
    from generalforecast.predictors import _registry

    _registry.use(FakeRegistry())
    try:
        resolution = predictors.resolve(predictors.load(root), root)
        path = predictors.write_lock(resolution, root)
    finally:
        _registry.use(None)
    document = yaml.safe_load(path.read_text())
    if stale is not None:
        from generalforecast.predictors.lock import lock_digest

        # Stale as a lock gf check wrote before something changed: its
        # digest still matches its content.
        stale(document["predictor"])
        document["digest"] = lock_digest(document)
        path.write_text(yaml.safe_dump(document, sort_keys=False))
    return document


def _write_orders(orders: Path) -> None:
    import yaml

    orders.mkdir()
    rng = random.Random(1)
    order_rows = [
        [
            100000 + i,
            round(rng.uniform(5, 900), 2),
            rng.randint(1, 9),
            rng.choice(["ads", "search", "referral"]),
            rng.choice([0, 1]),
        ]
        for i in range(10_000)
    ]
    _write_csv(
        orders / "context.csv",
        ["order_id", "amount", "items", "signup_source", "returned"],
        order_rows,
    )
    (orders / "gf.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "orders",
                "model": TABICL,
                "target": "returned",
                "id": "order_id",
                "context": {"snapshot": "context.csv"},
            },
            sort_keys=False,
        )
    )
    lock(orders, _before_signup_source)


def _before_signup_source(predictor: dict) -> None:
    """The lock a check wrote before `signup_source` existed and the cap was 4,000."""
    predictor["inputs"] = [
        i for i in predictor["inputs"] if i["name"] != "signup_source"
    ]
    predictor["selection"].update(core=1000, neighbors=3000)


# -- fakes ------------------------------------------------------------------


class FakeTime:
    """`perf_counter` returns the scripted instants; `sleep` does nothing."""

    def __init__(self, instants: tuple[float, ...]) -> None:
        self.instants = list(instants)
        self.now = 0.0

    def perf_counter(self) -> float:
        if self.instants:
            self.now = self.instants.pop(0)
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        return None


class ModelInfo:
    def __init__(self, id: str, task: str) -> None:
        self.id, self.task = id, task


class FakeRegistry:
    """The catalog the contract's examples assume."""

    MODELS = (
        (TABICL, "classification"),
        ("general-forecast/tabicl-v2-regressor", "regression"),
        ("general-forecast/mitra-2-classifier", "classification"),
        (KUMO, "classification"),
        ("general-forecast/kumo-tabular-medium-regressor", "regression"),
        ("general-forecast/chronos-2", "forecast"),
    )
    ALIASES: ClassVar[dict[str, str]] = {
        "tabicl-classifier": TABICL,
        "tabicl-v2-classifier": TABICL,
        "chronos-2": "general-forecast/chronos-2",
    }

    def catalog(self, task: str | None = None) -> list[ModelInfo]:
        return [ModelInfo(i, t) for i, t in self.MODELS if task is None or t == task]

    def default(self, task: str) -> str:
        from generalforecast.models.catalog import DEFAULTS

        return DEFAULTS[task]

    def canonical(self, ref: str) -> str:
        from generalforecast.errors import InvalidInput, NotFound

        if ref.partition("@")[0] in dict(self.MODELS):
            return ref
        if ref in self.ALIASES:
            raise InvalidInput(
                f"`{ref}` is not a catalog model id; use {self.ALIASES[ref]}",
                param="model",
            )
        raise NotFound(
            f"model `{ref}` is not in the catalog",
            hint="run `gf model list` to see the catalog",
        )

    def task_of(self, ref: str) -> Any:
        return dict(self.MODELS)[self.canonical(ref).partition("@")[0]]

    def limits(self, ref: str) -> Any:
        from generalforecast.predictors import _registry

        return _registry.Limits(6000, 100, 32)

    def pin(self, ref: str) -> Any:
        from dataclasses import replace

        from generalforecast.predictors import _registry

        return replace(_registry.CatalogRegistry().pin(ref), digest=DIGEST, files=())

    def confirm(self, ref: str) -> None:
        return None

    def is_cached(self, ref: str) -> bool:
        return False

    def pull(self, ref: str, *, progress: Any = None) -> Path:
        return Path.home() / ".cache/general-forecast" / ref.split("/")[-1]


class FakeEvaluation:
    def __init__(self, metrics: dict[str, dict[str, float]], rows: Any) -> None:
        self.metrics, self.rows = metrics, rows


class FakePredictor:
    """`generalforecast.Predictor` with the contract's numbers."""

    def __init__(
        self,
        reference: str,
        *,
        env: str | None = None,
        target: str | None = None,
        inputs: Any = None,
    ) -> None:
        self.reference, self.env = reference, env

    def explain(self) -> Any:
        from generalforecast.predictors import ContextInfo, Resolution, SelectionInfo

        return Resolution(
            name="churn",
            model=TABICL,
            model_digest=DIGEST,
            task="classification",
            target="churned",
            target_type="0/1",
            classes=("0", "1"),
            id="customer_id",
            time=None,
            series=(),
            inputs=(("spend", "float"), ("plan", "text"), ("signups", "int")),
            inferred={},
            context=ContextInfo("snapshot", 1840, 212, None),
            selection=SelectionInfo("exact"),
        )

    def eval(
        self, queries: Any = None, *, models: tuple[str, ...] = ()
    ) -> FakeEvaluation:
        import pandas as pd

        if os.environ.get("GF_SUITE_INTERNAL"):
            raise RuntimeError("the fixture asked for an internal failure")
        metrics = {TABICL: {"accuracy": 0.934, "log loss": 0.187, "auroc": 0.961}}
        for model in models:
            metrics[model] = {"accuracy": 0.921, "log loss": 0.203, "auroc": 0.948}
        rows = pd.DataFrame(
            {"customer_id": range(212), "churned": [0] * 212, "label": [0] * 212}
        )
        return FakeEvaluation(metrics, rows)

    def predict(self, queries: Any, context: Any = None, *, as_of: Any = None) -> Any:
        import pandas as pd

        from generalforecast.errors import Error

        if self.env is not None:
            raise Error(
                "CapacityExceeded",
                f"queries exceed the limit for {TABICL}",
                because="this model accepts 100 queries per call; the request sent 1,000",
                hint="split the queries, or push a predictor and call it in batches",
                limit=100,
                actual=1000,
                request_id=REQUEST_ID,
            )
        return pd.DataFrame(
            {
                "customer_id": range(1000, 2000),
                "label": [0, 1] * 500,
                "p_1": [0.2, 0.8] * 500,
            }
        )


class FakeClient:
    """The hosted service's predictor, context and identity routes."""

    def __init__(self, fixtures: Path) -> None:
        self.fixtures = fixtures
        self.generation = 3
        self.next_revision = 8
        self.contexts = self
        self.identity = self
        self.project = "churn-lab"

    def served_lock(self) -> dict:
        """The lock production serves: churn as pushed before `signups` existed."""
        import yaml

        document = yaml.safe_load((self.fixtures / "churn" / "gf.lock").read_text())
        predictor = document["predictor"]
        predictor["model_digest"] = SERVED_DIGEST
        predictor["inputs"] = [i for i in predictor["inputs"] if i["name"] != "signups"]
        return document

    def _call(
        self,
        method: str,
        path: str,
        kind: str,
        ident: str,
        payload: Any = None,
        **_: Any,
    ) -> dict:
        if path.endswith("/revisions"):
            return {
                "predictor": ident,
                "number": self.next_revision,
                "predictor_digest": "sha256:" + "a" * 64,
                "model": TABICL,
                "created_at": 0.0,
                "status": "ready",
            }
        if path.endswith("/environments"):
            return {
                "environments": [
                    {
                        "predictor": ident,
                        "name": "production",
                        "revision": 6,
                        "generation": 3,
                        "status": "ready",
                    },
                    {
                        "predictor": ident,
                        "name": "staging",
                        "revision": 7,
                        "generation": 5,
                        "status": "ready",
                    },
                ]
            }
        if "/environments/" in path and path.endswith("/logs"):
            return {
                "lines": [
                    "2026-10-24T05:12:01Z loaded general-forecast/tabicl-v2-classifier",
                    "2026-10-24T05:12:03Z compiled 12 programs in 6.8s",
                    "2026-10-24T05:14:40Z predicted 1,000 rows in 0.21s",
                ]
            }
        if "/environments/" in path:
            env = path.split("/environments/")[1].split("/")[0]
            revision = 6 if env == "production" else 7
            if payload and "revision" in payload:
                revision = payload["revision"]
            elif method == "GET":
                self.next_revision = revision + 1
            return {
                "predictor": ident,
                "name": env,
                "revision": revision,
                "generation": self.generation,
                "status": "ready",
            }
        if path.startswith("/predictors/") and "/revisions/" in path:
            return {"lock": self.served_lock()}
        return {"name": ident, "environments": ["production", "staging"]}

    def upload_snapshot(self, table: Any) -> dict:
        from generalforecast.tabular import table_digest

        return {"digest": table_digest(table), "rows": table.num_rows}

    # live contexts
    def get(self, name: str) -> Any:
        return _Wire(
            {
                "name": name,
                "rows": 48210,
                "labelled": 46100,
                "updated": "2026-10-23",
                "digest": "sha256:9c1d2e",
            }
        )

    def append(self, context: str, data: Any, *, id: str, idempotency_key: str) -> dict:
        return {"name": context, "accepted": data.num_rows, "write_id": idempotency_key}

    def label(
        self, context: str, data: Any, *, id: str, column: str, idempotency_key: str
    ) -> dict:
        return {"name": context, "accepted": data.num_rows, "write_id": idempotency_key}

    def create(self, config: dict) -> Any:
        return _Wire(config)

    # identity
    def manage(self, action: str, payload: Any = None) -> dict:
        if action == "keys.list":
            return {
                "keys": [
                    {
                        "id": "k-2f91",
                        "name": "ci",
                        "created": "2026-09-30",
                        "expires_at": "2027-09-30",
                    },
                    {
                        "id": "k-7ab3",
                        "name": "laptop",
                        "created": "2026-10-02",
                        "expires_at": "2027-10-02",
                    },
                ]
            }
        return {
            "email": "tejas@generalforecast.com",
            "workspace": "general-forecast",
            "key": {"id": "k-7ab3", "name": "laptop"},
        }


class _Wire:
    def __init__(self, value: dict) -> None:
        self.value = value

    def to_dict(self) -> dict:
        return dict(self.value)


def install_fakes(case: Case, fixtures: Path) -> None:
    """Replaces the libraries behind the CLI with the fakes above."""
    from generalforecast import account, commands, predictor_commands
    from generalforecast.predictors import _registry

    setattr(predictor_commands, "time", FakeTime(case.seconds))  # noqa: B010
    setattr(predictor_commands, "predictor_class", lambda: FakePredictor)  # noqa: B010
    setattr(commands, "model_registry", lambda: FakeRegistry())  # noqa: B010
    _registry.use(FakeRegistry())
    client = FakeClient(fixtures)
    setattr(commands, "client_for", lambda args, settings=None: client)  # noqa: B010
    setattr(commands, "connection_for", lambda args: None)  # noqa: B010
    setattr(  # noqa: B010
        account,
        "whoami",
        lambda args: {
            "email": "tejas@generalforecast.com",
            "workspace": "general-forecast",
            "project": "churn-lab",
            "server": "https://app.generalforecast.com",
            "source": "gf login",
        },
    )


def run_case(name: str, fixtures: Path, *, json: bool = False) -> None:
    """Child side: fakes in, then `gf <argv>` in the fixture directory, `--json` first if asked."""
    case = next(c for c in CASES if c.name == name)
    os.chdir(fixtures / case.cwd)
    os.environ.update(case.env)
    from generalforecast import cli

    install_fakes(case, fixtures)
    try:
        cli.main(["--json", *case.argv] if json else list(case.argv))
    except SystemExit as error:
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(error.code) from None
    sys.stdout.flush()
    sys.stderr.flush()


# -- the pseudo-terminal ------------------------------------------------------


def capture(case: Case, fixtures: Path) -> tuple[bytes, bytes | None]:
    """Runs one case under a pty; returns its bytes and the mid-prompt snapshot."""
    fd, slave = pty.openpty()
    env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "FORCE_COLOR")}
    env.update(TERM="xterm-256color", COLUMNS=str(COLUMNS), PYTHONPATH=str(ROOT))
    child = subprocess.Popen(
        [sys.executable, __file__, "--case", case.name, "--fixtures", str(fixtures)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLUMNS, 0, 0))
    output = bytearray()
    snapshot: bytes | None = None
    pending = list(case.keys)
    while True:
        ready, _, _ = select.select([fd], [], [], 30)
        if not ready:
            break
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        output.extend(chunk)
        plain = Screen(COLUMNS).feed(bytes(output)).text()
        if (
            snapshot is None
            and case.snapshot_after
            and _prompt_ready(bytes(output), plain, case.snapshot_after)
        ):
            snapshot = bytes(output)
        if pending and _prompt_ready(bytes(output), plain, pending[0][0]):
            os.write(fd, pending[0][1])
            pending.pop(0)
    child.wait(timeout=60)
    os.close(fd)
    return bytes(output), snapshot


def _prompt_ready(output: bytes, plain: str, label: str) -> bool:
    """Whether the prompt `label` is on screen and drawn whole.

    A `label [y/N]` question is whole once that line is. Every other prompt is
    a frame the renderer closes with `term.FRAME_END`, so a selector whose
    header arrived ahead of its option rows, or an edit line still being
    written, is neither snapshotted nor answered; the frame's last line is the
    edit line `label ...`, or an option row under the header `label`.
    """
    from generalforecast.term import FRAME_END

    lines = [line for line in plain.split("\n") if line.strip()]
    if not lines:
        return False
    last = lines[-1]
    if last == f"{label} [y/N]":
        return True
    if not output.endswith(FRAME_END.encode()):
        return False
    if last.startswith(label):
        return True
    if not last.startswith("  "):
        return False
    header = next((line for line in reversed(lines) if not line.startswith("  ")), "")
    return header.split(" ")[0] == label


class Screen:
    """Enough of a terminal to replay cursor-up redraws and keep SGR styles."""

    _CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
    _OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

    def __init__(self, columns: int) -> None:
        self.columns = columns
        self.lines: list[list[tuple[str, str]]] = [[]]
        self.row = 0
        self.col = 0
        self.style = ""

    def feed(self, data: bytes) -> Screen:
        text = data.decode("utf-8", "replace")
        i = 0
        while i < len(text):
            char = text[i]
            if char == "\x1b":
                osc = self._OSC.match(text, i)
                if osc:
                    i = osc.end()
                    continue
                csi = self._CSI.match(text, i)
                if csi:
                    self._control(csi.group(1), csi.group(2))
                    i = csi.end()
                    continue
                i += 1
                continue
            if char == "\r":
                self.col = 0
            elif char == "\n":
                self.row += 1
                while len(self.lines) <= self.row:
                    self.lines.append([])
            elif char == "\x07":
                pass
            else:
                line = self.lines[self.row]
                while len(line) < self.col:
                    line.append((" ", ""))
                if self.col < len(line):
                    line[self.col] = (char, self.style)
                else:
                    line.append((char, self.style))
                self.col += 1
            i += 1
        return self

    def _control(self, params: str, final: str) -> None:
        if final == "m":
            codes = [c for c in params.split(";") if c] or ["0"]
            for code in codes:
                if code == "0":
                    self.style = ""
                else:
                    self.style = ";".join(
                        sorted(set(filter(None, self.style.split(";"))) | {code})
                    )
        elif final == "A":
            self.row = max(0, self.row - int(params or "1"))
        elif final == "B":
            self.row += int(params or "1")
            while len(self.lines) <= self.row:
                self.lines.append([])
        elif final == "J":
            self.lines[self.row] = self.lines[self.row][: self.col]
            del self.lines[self.row + 1 :]
        elif final == "K":
            self.lines[self.row] = self.lines[self.row][: self.col]

    def rows(self) -> list[list[tuple[str, str]]]:
        rows = list(self.lines)
        while rows and not rows[-1]:
            rows.pop()
        return rows

    def text(self) -> str:
        return "\n".join("".join(c for c, _ in line).rstrip() for line in self.rows())


# -- rendering --------------------------------------------------------------

_PALETTE = {
    "31": (255, 85, 85),
    "32": (80, 250, 123),
    "33": (241, 250, 140),
    "36": (139, 233, 253),
}


def png(screen: Screen, title: str, path: Path) -> bool:
    """Draws the screen with DejaVu Sans Mono; False when Pillow is missing."""
    try:
        from PIL import Image, ImageDraw, ImageFont  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ]
    size = 15
    try:
        regular = ImageFont.truetype(fonts[0], size)
        bold = ImageFont.truetype(fonts[1], size)
    except OSError:
        regular = bold = ImageFont.load_default()
    width = regular.getbbox("M")[2]
    height = size + 6
    rows = [[(c, "") for c in title], [], *screen.rows()]
    size_px = (int(width * COLUMNS + 32), int(height * (len(rows) + 1) + 16))
    image = Image.new("RGB", size_px, (24, 24, 27))
    draw = ImageDraw.Draw(image)
    for r, line in enumerate(rows):
        for c, (char, style) in enumerate(line):
            codes = set(style.split(";")) - {""}
            colour = next(
                (_PALETTE[k] for k in codes if k in _PALETTE), (229, 229, 229)
            )
            if "2" in codes:
                colour = (128, 128, 136)
            draw.text(
                (16 + c * width, 8 + r * height),
                char,
                font=bold if "1" in codes else regular,
                fill=colour,
            )
    image.save(path)
    return True


def render(outdir: Path, *, pngs: bool = False) -> dict[str, str]:
    """Runs every case; returns `{case: plain text}` and writes the files."""
    import tempfile

    outdir.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    suite: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        fixtures = Path(temporary) / "fixtures"
        write_fixtures(fixtures)
        for case in CASES:
            raw, snapshot = capture(case, fixtures)
            title = f"$ gf {' '.join(case.argv)}"
            (outdir / f"{case.name}.ansi").write_bytes(raw)
            screen = Screen(COLUMNS).feed(raw)
            text = screen.text()
            texts[case.name] = text
            (outdir / f"{case.name}.txt").write_text(text + "\n")
            suite.append(f"{title}\n{text}\n")
            if pngs:
                png(screen, title, outdir / f"{case.name}.png")
            if snapshot is not None:
                shot = Screen(COLUMNS).feed(snapshot)
                texts[f"{case.name}-prompt"] = shot.text()
                (outdir / f"{case.name}-prompt.txt").write_text(shot.text() + "\n")
                suite.append(f"{title}  (at the first prompt)\n{shot.text()}\n")
                if pngs:
                    png(
                        shot,
                        f"{title}  (at the first prompt)",
                        outdir / f"{case.name}-prompt.png",
                    )
    (outdir / "suite.txt").write_text("\n".join(suite))
    return texts


def main() -> None:
    """Command line: an output directory, optionally `--png`, or `--case` for the child."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("outdir", nargs="?")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--fixtures")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.case:
        run_case(args.case, Path(args.fixtures), json=args.json)
        return
    if not args.outdir:
        parser.error("an output directory is required")
    render(Path(args.outdir), pngs=args.png)


if __name__ == "__main__":
    main()
