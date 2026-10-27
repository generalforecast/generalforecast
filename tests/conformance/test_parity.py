# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One gf.yaml, the same predictions and the same lock digest on every path.

This is the product's promise as a test. One predictor directory, for a
classifier, a regressor and a forecaster with past and future covariates,
type overrides, an `until` or `holdout` split and a declared selection, is
predicted through:

- `gf.Predictor(directory)`, in process;
- `gf predict directory`, the CLI;
- `gf serve directory`, the gf-server binary;
- hosted publishing: the pushed lock and rows, compiled with `compile_lock`,
  published with `published_entry` and run by `PredictorExecutor`, as the
  hosted plane runs a revision.

Every path must answer the same predictions, floats within tolerance, and
know the predictor by the same lock digest. The models are the tiny ones the
conformance tests load in both engines.
"""

import datetime
import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import generalforecast as gf
from generalforecast import cli, hosting
from generalforecast.predictor import _frame_of_result
from generalforecast.predictors import load, resolve, write_lock
from generalforecast.serve import models as serve_models
from generalforecast.service.sdk import pushed
from tests.conformance.test_surfaces import Served, _predicted, _same, engine
from tests.utils.server import build

__all__ = ["engine"]
START = datetime.date(2025, 3, 1)


def _days(count: int) -> list[datetime.date]:
    return [START + datetime.timedelta(days=day) for day in range(count)]


def classifier_rows() -> pa.Table:
    days = _days(24)
    return pa.table(
        {
            "id": [f"c{row}" for row in range(24)],
            "day": days,
            "spend": [float((row * 7) % 11) for row in range(24)],
            "zip": [100 + row % 3 for row in range(24)],
            "plan": [("basic", "pro", "team")[row % 3] for row in range(24)],
            "note": [f"n{row}" for row in range(24)],
            "churned": [row % 2 if row % 9 else None for row in range(24)],
        }
    )


def regressor_rows() -> pa.Table:
    return pa.table(
        {
            "id": list(range(30)),
            "size": [float(row % 7) for row in range(30)],
            "zip": [200 + row % 4 for row in range(30)],
            "price": [float(10 + 3 * (row % 7) + row % 4) for row in range(30)],
        }
    )


def forecaster_rows() -> pa.Table:
    days = _days(20)
    return pa.table(
        {
            "store": [store for store in ("north", "south") for _ in days],
            "date": [day for _ in range(2) for day in days],
            "units": [
                float(10 + step + 4 * index + step % 3)
                for index in range(2)
                for step in range(20)
            ],
            "footfall": [float((step * 3) % 5) for _ in range(2) for step in range(20)],
            "promo": [float(step % 2) for _ in range(2) for step in range(20)],
        }
    )


CASES = {
    "classifier": (
        classifier_rows,
        {
            "model": "general-forecast/tabicl-v2-classifier",
            "target": "churned",
            "id": "id",
            "time": "day",
            "inputs": {"exclude": ["note"], "types": {"zip": "category"}},
            "selection": {"core": 6, "neighbors": 4, "seed": 3},
            "context": {"snapshot": "rows.parquet", "until": "2025-03-21"},
        },
    ),
    "regressor": (
        regressor_rows,
        {
            "model": "general-forecast/tabicl-v2-regressor",
            "target": "price",
            "id": "id",
            "inputs": {"types": {"zip": "category"}},
            "quantiles": [0.1, 0.5, 0.9],
            "selection": {"core": 8, "neighbors": 6, "kind": "random", "seed": 1},
            "context": {
                "snapshot": "rows.parquet",
                "holdout": {"fraction": 0.2, "seed": 5},
            },
        },
    ),
    "forecaster": (
        forecaster_rows,
        {
            "model": "general-forecast/chronos-2",
            "target": "units",
            "time": "date",
            "series": ["store"],
            "horizon": 3,
            "inputs": {"past": ["footfall"], "future": ["promo"]},
            "context": {"snapshot": "rows.parquet", "until": "2025-03-18"},
        },
    ),
}


def _directory(root: Path, case: str) -> Path:
    rows, manifest = CASES[case]
    root.mkdir(parents=True)
    pq.write_table(rows(), root / "rows.parquet")
    (root / "gf.yaml").write_text(
        yaml.safe_dump({"name": case, **manifest}, sort_keys=False)
    )
    write_lock(resolve(load(root), root), root)
    return root


def _queries(root: Path) -> pa.Table:
    """The predictor's own query rows, in the columns its lock reads.

    A tabular query carries the id and inputs; a forecast query its series,
    time and known-future inputs, over the horizon after the context.
    """
    predictor = gf.Predictor(root)
    queries = predictor._query_frame
    assert queries is not None and queries.num_rows
    document = yaml.safe_load((root / "gf.lock").read_text())["predictor"]
    if document["task"] == "forecast":
        future = [c["name"] for c in document["inputs"] if c["role"] == "future"]
        return queries.select([*document["series"], document["time"], *future])
    names = [column["name"] for column in document["inputs"]]
    return queries.select([document["id"], *names])


def _records(frame: Any) -> list[dict[str, Any]]:
    table = frame if isinstance(frame, pa.Table) else pa.Table.from_pandas(frame)
    return [
        {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in row.items()
        }
        for row in table.to_pylist()
    ]


def _outputs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The prediction columns alone: label, value, probabilities and quantiles."""
    names = [
        column
        for column in frame.columns
        if column in ("label", "value") or column.startswith(("p_", "q_"))
    ]
    return _records(frame[names].reset_index(drop=True))


def _cli(root: Path, queries: pa.Table, monkeypatch: Any) -> pd.DataFrame:
    source, output = (
        root.parent / f"{root.name}-q.parquet",
        root.parent / f"{root.name}-out.parquet",
    )
    pq.write_table(queries, source)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setenv("NO_COLOR", "1")
    try:
        cli.main(
            ["predict", str(root), "--queries", str(source), "--output", str(output)]
        )
    except SystemExit as error:
        assert not error.code, sys.stderr.getvalue()
    return pd.read_parquet(output)


@pytest.mark.parametrize("case", list(CASES))
def test_every_path_predicts_the_same_and_knows_the_same_lock(
    tmp_path, engine, monkeypatch, case
):
    root = _directory(tmp_path / case, case)
    lock = yaml.safe_load((root / "gf.lock").read_text())
    digest = lock["digest"]
    queries = _queries(root)

    local = gf.Predictor(root).predict(queries)
    command = _cli(root, queries, monkeypatch)

    sent = pushed(root)
    assert sent.lock.digest == digest
    plan = hosting.compile_lock(sent.lock, sent.snapshot)
    published = hosting.published_entry(plan)
    hosted = hosting.PredictorExecutor().prepare(plan, sent.snapshot).predict(queries)
    hosted_frame = _frame_of_result({"predictions": hosted.to_pylist()}, plan.spec.id)

    server = Served(build(), str(root))
    try:
        served_rows = _predicted(server, case, queries)
        described = json.loads((server.directory / "models.json").read_text())[0][
            "predictor"
        ]
    finally:
        server.stop()
    served_frame = _frame_of_result({"predictions": served_rows}, plan.spec.id)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(serve_models, "checkpoint", lambda *_, **__: None)
        (tmp_path / "entry").mkdir()
        entry = serve_models.served(str(root), tmp_path / "entry")
    assert {**entry["predictor"], "snapshot": None} == {
        **published["predictor"],
        "snapshot": None,
    }
    assert entry["description"] == published["description"]
    for known in (
        published["predictor"]["lock_digest"],
        described["lock_digest"],
        plan.lock_digest,
    ):
        assert known == digest
    if case != "forecaster":
        assert plan.spec.selection is not None
        assert plan.spec.selection.strategy == "retrieved"
        assert published["predictor"]["strategy"] == plan.spec.selection.strategy_spec()

    expected = _outputs(local)
    assert expected
    for name, frame in (
        ("cli", command),
        ("hosted", hosted_frame),
        ("served", served_frame),
    ):
        _same(_outputs(frame), expected, name)
