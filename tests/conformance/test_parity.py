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

The forecasters include 500 series that end on different days, each with
its own known-future promotions, two declared related groups, and one whose
context comes with each request, its lock checked without rows.

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


def dated_rows() -> pa.Table:
    """The classifier's rows with a time of day input, which JSON carries as ISO text."""
    start = datetime.datetime(2025, 3, 1, 8, tzinfo=datetime.UTC)
    seen = [
        (start + datetime.timedelta(hours=7 * row)).replace(tzinfo=None)
        for row in range(24)
    ]
    return classifier_rows().append_column("seen", pa.array(seen, pa.timestamp("us")))


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


def request_rows() -> pa.Table:
    """The forecaster's rows with each series' last three targets blank: its queries."""
    rows = forecaster_rows()
    late = [
        day >= START + datetime.timedelta(days=17) for day in rows["date"].to_pylist()
    ]
    units = [
        None if blank else value
        for blank, value in zip(late, rows["units"].to_pylist(), strict=True)
    ]
    return rows.set_column(
        rows.schema.get_field_index("units"), "units", pa.array(units)
    )


def many_rows() -> pa.Table:
    """500 series ending on three different days, each with its own promotions.

    Each series' three rows past its end have no target: they are the
    queries, with the known-future promotions of that series alone.
    """
    columns: dict[str, list] = {"sku": [], "date": [], "units": [], "promo": []}
    for sku in range(500):
        end = 16 - sku % 3
        for step in range(end + 3):
            columns["sku"].append(f"sku-{sku:03d}")
            columns["date"].append(START + datetime.timedelta(days=step))
            columns["units"].append(
                None if step >= end else float(5 + sku % 7 + step % 4)
            )
            columns["promo"].append(float((sku * 3 + step) % 2))
    return pa.table(columns)


def grouped_rows() -> pa.Table:
    """Two regions of four stores, each region's stores forecast together."""
    columns: dict[str, list] = {
        "region": [],
        "store": [],
        "date": [],
        "units": [],
        "promo": [],
    }
    for region in ("east", "west"):
        for store in range(4):
            for step in range(19):
                columns["region"].append(region)
                columns["store"].append(f"{region}-{store}")
                columns["date"].append(START + datetime.timedelta(days=step))
                columns["units"].append(
                    None if step >= 16 else float(10 + store + step % 5)
                )
                columns["promo"].append(float((step + (region == "west")) % 2))
    return pa.table(columns)


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
    "request": (
        request_rows,
        {
            "model": "general-forecast/chronos-2",
            "target": "units",
            "time": "date",
            "series": ["store"],
            "frequency": "1d",
            "horizon": 3,
            "inputs": {"past": ["footfall"], "future": ["promo"]},
            "context": {"request": {}},
        },
    ),
    # A request-context classifier checked without rows records no kind for
    # `seen`: every path reads its ISO text as UTC timestamps, not as text.
    "request-classifier-dt": (
        dated_rows,
        {
            "model": "general-forecast/tabicl-v2-classifier",
            "target": "churned",
            "id": "id",
            "inputs": ["spend", "plan", "seen"],
            "context": {"request": {}},
        },
    ),
    "many": (
        many_rows,
        {
            "model": "general-forecast/chronos-2",
            "target": "units",
            "time": "date",
            "series": ["sku"],
            "horizon": 3,
            "inputs": {"future": ["promo"]},
            "context": {"snapshot": "rows.parquet"},
        },
    ),
    "grouped": (
        grouped_rows,
        {
            "model": "general-forecast/chronos-2",
            "target": "units",
            "time": "date",
            "series": ["region", "store"],
            "related": ["region"],
            "horizon": 3,
            "inputs": {"future": ["promo"]},
            "context": {"snapshot": "rows.parquet"},
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
    document = yaml.safe_load((root / "gf.lock").read_text())["predictor"]
    if document["context"]["kind"] == "request":
        queries = _request(root)[1]
    else:
        queries = gf.Predictor(root)._query_frame
    assert queries is not None and queries.num_rows
    if document["task"] == "forecast":
        future = [c["name"] for c in document["inputs"] if c["role"] == "future"]
        return queries.select([*document["series"], document["time"], *future])
    names = [column["name"] for column in document["inputs"]]
    return queries.select([document["id"], *names])


def _request(root: Path) -> tuple[pa.Table, pa.Table]:
    """A request-context predictor's call: the labeled rows, and the rows to predict."""
    document = yaml.safe_load((root / "gf.lock").read_text())["predictor"]
    rows = pq.read_table(root / "rows.parquet")
    blank = pa.array([value is None for value in rows[document["target"]].to_pylist()])
    labeled = pa.array(
        [value is not None for value in rows[document["target"]].to_pylist()]
    )
    return rows.filter(labeled), rows.filter(blank)


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


def _ascending(frame: pd.DataFrame) -> None:
    """Every row's quantiles rise with their levels, as every path rearranges them."""
    levels = sorted(
        (float(column[2:]), column)
        for column in frame.columns
        if column.startswith("q_")
    )
    for row in frame[[column for _, column in levels]].itertuples(index=False):
        assert list(row) == sorted(row), row


def _cli(
    root: Path, queries: pa.Table, monkeypatch: Any, context: pa.Table | None = None
) -> pd.DataFrame:
    source, output = (
        root.parent / f"{root.name}-q.parquet",
        root.parent / f"{root.name}-out.parquet",
    )
    pq.write_table(queries, source)
    given = []
    if context is not None:
        pq.write_table(context, root.parent / f"{root.name}-c.parquet")
        given = ["--context", str(root.parent / f"{root.name}-c.parquet")]
    errors = io.StringIO()
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", errors)
    monkeypatch.setenv("NO_COLOR", "1")
    try:
        cli.main(
            [
                "predict",
                str(root),
                "--queries",
                str(source),
                "--output",
                str(output),
                *given,
            ]
        )
    except SystemExit as error:
        assert not error.code, errors.getvalue()
    return pd.read_parquet(output)


@pytest.mark.server
@pytest.mark.parametrize("case", list(CASES))
def test_every_path_predicts_the_same_and_knows_the_same_lock(
    tmp_path, engine, monkeypatch, case
):
    root = _directory(tmp_path / case, case)
    lock = yaml.safe_load((root / "gf.lock").read_text())
    digest = lock["digest"]
    queries = _queries(root)
    # A request-context predictor takes its labeled rows with each call.
    context = (
        _request(root)[0] if lock["predictor"]["context"]["kind"] == "request" else None
    )

    local = gf.Predictor(root).predict(queries, context=context)
    command = _cli(root, queries, monkeypatch, context)

    sent = pushed(root)
    assert sent.lock.digest == digest
    plan = hosting.compile_lock(sent.lock, sent.snapshot)
    published = hosting.published_entry(plan)
    hosted = (
        hosting.PredictorExecutor()
        .prepare(plan, sent.snapshot)
        .predict(
            queries,
            context=None
            if context is None
            else hosting.served_context(sent.lock, context),
        )
    )
    hosted_frame = _frame_of_result({"predictions": hosted.to_pylist()}, plan.spec.id)

    server = Served(build(), str(root))
    try:
        served_rows = _predicted(server, case, queries, context)
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
    if case == "many":
        assert len(queries) == 1500
        assert len({row["date"] for row in queries.to_pylist()}) == 5
    if case in ("classifier", "regressor"):
        assert plan.spec.selection is not None
        assert plan.spec.selection.strategy == "retrieved"
        assert published["predictor"]["strategy"] == plan.spec.selection.strategy_spec()

    expected = _outputs(local)
    assert expected
    _ascending(local)
    for name, frame in (
        ("cli", command),
        ("hosted", hosted_frame),
        ("served", served_frame),
    ):
        _same(_outputs(frame), expected, name)


# Hand edits to a checked lock, each re-digested so the lock is
# self-consistent; every entry point must refuse every one alike.
def _forge(root: Path, change: Any) -> None:
    from generalforecast.predictors.lock import lock_digest

    document = yaml.safe_load((root / "gf.lock").read_text())
    change(document)
    if document.get("lock_version") == 1:
        document["digest"] = lock_digest(document)
    (root / "gf.lock").write_text(yaml.safe_dump(document, sort_keys=False))


FORGERIES = {
    "selection past the cap": lambda d: d["predictor"]["selection"].update(core=10**9),
    "selection seed": lambda d: d["predictor"]["selection"].update(seed=99),
    "outputs narrowed": lambda d: d["predictor"].update(outputs=["label"]),
    "input kind": lambda d: d["predictor"]["inputs"][0].update(
        type="text" if d["predictor"]["inputs"][0]["type"] != "text" else "category"
    ),
    "classes": lambda d: d["predictor"].update(classes=[0, 1, 2]),
    "resources": lambda d: d.update(resources={"accelerator": "L4"}),
    "context rows": lambda d: d["predictor"]["context"].update(rows=3),
    "another format": lambda d: d.update(lock_version=2),
}


@pytest.mark.parametrize("forgery", list(FORGERIES))
def test_every_entry_point_refuses_a_forged_lock(tmp_path, monkeypatch, forgery):
    from generalforecast.predictors import verified
    from generalforecast.predictors.validation import PredictorError
    from generalforecast.service.contracts import ServiceError

    monkeypatch.setattr(serve_models, "checkpoint", lambda *a, **k: None)
    root = _directory(tmp_path / "classifier", "classifier")
    _forge(root, FORGERIES[forgery])
    with pytest.raises(gf.InvalidInput):
        verified(load(root), root)
    for flag in ("--frozen", "--locked"):
        with pytest.raises(SystemExit) as exited:
            cli.main(["check", flag, "-q", str(root)])
        assert exited.value.code != 0
    with pytest.raises(PredictorError):
        serve_models.served(str(root), tmp_path / "entry")
    with pytest.raises(ServiceError):
        pushed(root)
    with pytest.raises(gf.InvalidInput):
        gf.Predictor(root)
    # Hosted serving holds the pushed lock to its manifest and rows alike.
    document = yaml.safe_load((root / "gf.lock").read_text())
    rows = pq.read_table(root / "rows.parquet")
    with pytest.raises(PredictorError):
        hosting.compile_lock(
            document,
            hosting.served_context(document, rows),
            manifest=load(root).to_dict(),
        )


@pytest.mark.server
def test_any_number_of_series_and_queries_predict_alike_past_one_call(tmp_path, engine):
    """2,100 series, 10,500 queries: past one call's bound, every path agrees.

    gf.Predictor in process, a gf serve reached through gf.Predictor(url)
    and hosted serving each split the queries into calls of at most
    `MAX_QUERIES` rows, every series whole within one.
    """
    from generalforecast.contracts import MAX_QUERIES
    from generalforecast.predictor import _query_batches

    columns: dict[str, list] = {"sku": [], "date": [], "units": []}
    for sku in range(2100):
        for step in range(12 + sku % 3):
            columns["sku"].append(f"s{sku:04d}")
            columns["date"].append(START + datetime.timedelta(days=step))
            columns["units"].append(
                None if step >= 7 + sku % 3 else float((sku + step) % 9)
            )
    root = tmp_path / "wide"
    root.mkdir()
    pq.write_table(pa.table(columns), root / "rows.parquet")
    manifest = {
        "name": "wide",
        "model": "general-forecast/chronos-2",
        "target": "units",
        "time": "date",
        "series": ["sku"],
        "horizon": 5,
        "context": {"snapshot": "rows.parquet"},
    }
    (root / "gf.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    write_lock(resolve(load(root), root), root)
    queries = _queries(root)
    assert queries.num_rows == 10_500 > MAX_QUERIES

    local = gf.Predictor(root).predict(queries)
    sent = pushed(root)
    plan = hosting.compile_lock(sent.lock, sent.snapshot)
    prepared = hosting.PredictorExecutor().prepare(plan, sent.snapshot)
    hosted = pd.concat(
        [
            _frame_of_result(
                {"predictions": prepared.predict(queries.take(rows)).to_pylist()}, None
            ).set_index(pd.Index(rows))
            for rows in _query_batches(queries, ["sku"], MAX_QUERIES)
        ]
    ).sort_index()
    server = Served(build(), str(root))
    try:
        served = gf.Predictor(f"{server.url}/wide").predict(queries)
    finally:
        server.stop()
    expected = _outputs(local)
    _same(_outputs(hosted.reset_index(drop=True)), expected, "hosted")
    _same(_outputs(served), expected, "served")
    # A served forecast's frame is a local one: its series and time lead it.
    assert list(served.columns) == list(local.columns)
    assert list(local.columns[:2]) == ["sku", "date"]
    for name in ("sku", "date"):
        assert [str(value) for value in served[name]] == [
            str(value) for value in local[name]
        ], name
