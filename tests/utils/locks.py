# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Predictor directories, locks and compiled plans built the way `gf check` builds them."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from generalforecast.contracts import PredictorLock
from generalforecast.predictors import load, resolve, write_lock
from generalforecast.predictors.lock import document
from generalforecast.predictors.runtime.compile import compile_lock
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.context import served_context
from generalforecast.tabular import arrow_bytes

CLASSIFIER = "general-forecast/tabicl-v2-classifier"
SNAPSHOT = "history.arrow"


def directory(
    root: Path,
    data: pa.Table,
    *,
    context: str = "snapshot",
    model: str = CLASSIFIER,
    target: str = "y",
    split: dict[str, Any] | None = None,
    **manifest: Any,
) -> Path:
    """A predictor directory with `gf.yaml`, the rows as its snapshot, and `gf.lock`.

    `context` is `snapshot`, `request` or `live`; a request or live
    manifest lists its inputs, every column without a role unless given.
    `split` is a snapshot's `until` or `holdout`.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / SNAPSHOT).write_bytes(arrow_bytes(data))
    body: dict[str, Any] = {"name": "test-model", "model": model, "target": target}
    body.update(manifest)
    if context == "snapshot":
        body["context"] = {"snapshot": SNAPSHOT, **(split or {})}
    else:
        body["context"] = (
            {"request": {}} if context == "request" else {"live": "events"}
        )
        roles = {
            target,
            *(body.get(key) for key in ("id", "time") if body.get(key)),
            *(body.get("series") or ()),
        }
        body.setdefault(
            "inputs", [name for name in data.column_names if name not in roles]
        )
    (root / "gf.yaml").write_text(yaml.safe_dump(body, sort_keys=False))
    write_lock(resolve(load(root), root), root)
    return root


def lock_of(root: Path) -> PredictorLock:
    """The directory's `gf.lock` as hosted serving receives it."""
    return PredictorLock.from_dict(yaml.safe_load((root / "gf.lock").read_text()))


def compiled(
    root: Path,
    data: pa.Table,
    *,
    outputs: Any = None,
    **options: Any,
) -> CompiledPlan:
    """The compiled plan hosted serving binds for the directory's lock and rows."""
    directory(root, data, **options)
    lock = lock_of(root)
    rows = None if lock.context_kind == "request" else served_context(lock, data)
    return compile_lock(lock, rows, outputs=outputs)


def ad_hoc(rows: pa.Table, **manifest: Any) -> PredictorLock:
    """The lock of an ad hoc request, resolved against its rows as the hosted route does."""
    from generalforecast.predictors.spec import PredictorSpec

    spec = PredictorSpec.from_dict(
        {"name": "ad-hoc", "context": {"request": {}}, **manifest}
    )
    return PredictorLock.from_dict(document(resolve(spec, rows=rows)))
