# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The public SDK surface: exports, errors, `Predictor` resolution and evaluation."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import generalforecast as gf
from generalforecast import errors, tabular
from generalforecast.native import tables
from generalforecast.predictor import (
    Evaluation,
    PredictRequest,
    classification_metrics,
    forecast_metrics,
    regression_metrics,
    resolve_reference,
)

MODEL = "general-forecast/tabicl-v2-classifier"
REGRESSOR = "general-forecast/tabicl-v2-regressor"
CONTEXT = [
    {"customer_id": 1, "spend": 10.0, "plan": "a", "churned": 0},
    {"customer_id": 2, "spend": 20.0, "plan": "b", "churned": 1},
    {"customer_id": 3, "spend": 30.0, "plan": "a", "churned": 0},
    {"customer_id": 4, "spend": 40.0, "plan": "b", "churned": 1},
]


def test_top_level_exports_are_exactly_the_contract() -> None:
    assert set(gf.__all__) == {
        "Predictor",
        "Classifier",
        "Regressor",
        "Forecaster",
        "Error",
        "InvalidInput",
        "NotFound",
        "Unauthorized",
        "CapacityExceeded",
        "ModelNotReady",
        "AmbiguousReference",
        "Internal",
        "Conflict",
        "Unavailable",
        "Client",
        "AsyncClient",
        "engine",
        "errors",
        "predictors",
        "__version__",
    }
    for name in gf.__all__:
        assert getattr(gf, name) is not None
    for name in ("Strategy", "GroupStrategy", "local", "Table", "Package"):
        with pytest.raises(AttributeError):
            getattr(gf, name)
    assert gf.engine.Table is tables.Table
    assert gf.errors is errors
    assert issubclass(gf.CapacityExceeded, gf.Error)


def test_errors_render_the_cli_shapes_without_colour() -> None:
    one = gf.InvalidInput(
        "gf.yaml already exists in ./churn",
        hint="pass --force to overwrite it, or run `gf check` to validate it",
    )
    assert str(one) == (
        "error: gf.yaml already exists in ./churn\n"
        "  hint: pass --force to overwrite it, or run `gf check` to validate it"
    )
    three = gf.CapacityExceeded(
        f"queries exceed the limit for {MODEL}",
        because="this model accepts 100 queries per call; the request sent 1,000",
        hint="split the queries, or push a predictor and call it in batches",
        limit=100,
        actual=1000,
        request_id="r-3abb2401f119428db148927c6a0d358c",
    )
    assert str(three) == (
        f"error: queries exceed the limit for {MODEL}\n"
        "  because: this model accepts 100 queries per call; the request sent 1,000\n"
        "  hint: split the queries, or push a predictor and call it in batches\n"
        "  request r-3abb2401f119428db148927c6a0d358c"
    )
    assert three.code == "CapacityExceeded"
    assert three.to_dict() == {
        "code": "CapacityExceeded",
        "message": f"queries exceed the limit for {MODEL}",
        "hint": "split the queries, or push a predictor and call it in batches",
        "limit": 100,
        "actual": 1000,
        "request_id": "r-3abb2401f119428db148927c6a0d358c",
    }
    long = gf.InvalidInput(
        "predictor sales has no cadence",
        because="general-forecast/chronos-2 forecasts series, and gf.yaml declares "
        "no `time` column so there is nothing to step along",
    )
    lines = str(long).splitlines()
    assert lines[1].startswith("  because: ")
    assert all(line.startswith(" " * len("  because: ")) for line in lines[2:])


def test_server_error_objects_map_to_the_classes() -> None:
    mapped = errors.from_wire(
        {
            "code": "ValidationError",
            "message": "column plan has text values",
            "param": "inputs.plan",
        },
        status=422,
        request_id="r-1",
    )
    assert type(mapped) is gf.InvalidInput
    assert (mapped.param, mapped.request_id) == ("inputs.plan", "r-1")
    assert (
        type(errors.from_wire({"code": "ContextCapacityExceeded", "message": "x"}))
        is gf.CapacityExceeded
    )
    assert type(errors.from_wire({"code": "NotFound", "message": "x"})) is gf.NotFound
    assert (
        type(errors.from_wire({"code": "Unauthenticated", "message": "x"}))
        is gf.Unauthorized
    )
    assert (
        type(errors.from_wire({"code": "ModelNotReady", "message": "x"}))
        is gf.ModelNotReady
    )
    assert (
        type(errors.from_wire({"code": "Weird", "message": "x"}, status=500))
        is gf.Internal
    )
    assert errors.convert(RuntimeError("engine")) is None
    # A ValueError the engine's data libraries raise is the caller's; one
    # raised by any other code is a bug, and stays itself.
    with pytest.raises(gf.InvalidInput), errors.boundary():
        pd.DataFrame({"a": [1], "b": [1, 2]})
    with pytest.raises(ValueError, match="a bug") as bug, errors.boundary():
        raise ValueError("a bug")
    assert not isinstance(bug.value, gf.Error)
    with pytest.raises(gf.InvalidInput), errors.boundary():
        pa.array([1, "a"]).cast(pa.int64())


def test_bare_model_names_are_rejected_with_the_canonical_hint() -> None:
    for bare in ("tabicl-classifier", "tabicl-v2-classifier"):
        with pytest.raises(gf.InvalidInput) as caught:
            gf.Predictor(bare, target="churned", context=CONTEXT).predict(CONTEXT)
        assert str(caught.value) == (
            f"error: `{bare}` is not a catalog model id; use {MODEL}"
        )
    with pytest.raises(gf.NotFound):
        gf.Predictor("general-forecast/no-such-model", target="churned")


def test_references_resolve_by_the_contract_rule(tmp_path: Path, monkeypatch) -> None:
    assert resolve_reference("churn") == ("hosted", "churn")
    assert resolve_reference(MODEL) == ("catalog", MODEL)
    assert resolve_reference("http://10.0.0.5:8000/churn") == (
        "served",
        ("http://10.0.0.5:8000", "churn"),
    )
    (tmp_path / "churn").mkdir()
    (tmp_path / "churn" / "gf.yaml").write_text(
        f"name: churn\nmodel: {MODEL}\ntarget: churned\n"
    )
    assert resolve_reference(tmp_path / "churn") == ("local", tmp_path / "churn")
    monkeypatch.chdir(tmp_path)
    assert resolve_reference("./churn") == ("local", Path("churn"))
    with pytest.raises(gf.AmbiguousReference) as caught:
        resolve_reference("churn")
    assert str(caught.value) == (
        "error: `churn` is both a directory with gf.yaml and a hosted predictor name\n"
        "  hint: pass ./churn for the local predictor, or env= for the hosted one"
    )
    with pytest.raises(gf.NotFound):
        resolve_reference("./missing")
    with pytest.raises(gf.InvalidInput):
        gf.Predictor("other", env="dev")


def test_predictor_repr_shows_name_model_inputs_context_and_env() -> None:
    hosted = gf.Predictor("churn")
    assert hosted.env == "production"
    assert (
        repr(hosted)
        == "Predictor(churn, model=?, inputs=?, context=?, env='production')"
    )
    assert gf.Predictor("churn", env="staging").env == "staging"
    ad_hoc = gf.Predictor(
        MODEL, target="churned", inputs=["spend", "plan"], context=CONTEXT
    )
    assert ad_hoc.env is None
    assert repr(ad_hoc) == (
        f"Predictor(ad-hoc, model={MODEL}, inputs=2, context=request 4 rows, env=None)"
    )
    assert (ad_hoc.context.kind, ad_hoc.context.rows) == ("request", 4)
    served = gf.Predictor("http://10.0.0.5:8000/churn")
    assert served.env == "http://10.0.0.5:8000"


def test_requests_name_columns_and_classes_in_their_errors() -> None:
    with pytest.raises(gf.InvalidInput) as caught:
        PredictRequest.build(
            model=MODEL, target="churn", context=CONTEXT, queries=CONTEXT
        )
    assert str(caught.value) == (
        "error: target column `churn` is not in the context\n  hint: did you mean `churned`?"
    )
    assert caught.value.param == "target"
    with pytest.raises(gf.InvalidInput) as caught:
        PredictRequest.build(
            model=MODEL,
            target="churned",
            context=CONTEXT,
            queries=CONTEXT,
            inputs=["spend", "age"],
        )
    assert str(caught.value) == (
        "error: input column `age` is not in the context\n"
        "  hint: the context has customer_id, spend, plan, churned"
    )
    assert caught.value.param == "inputs.age"
    request = PredictRequest.build(
        model=MODEL,
        target="churned",
        context=CONTEXT,
        queries=[{"spend": 1.0, "plan": "a"}],
        id="customer_id",
    )
    assert (request.inputs, request.classes, request.task) == (
        ("spend", "plan"),
        (0, 1),
        "classification",
    )
    payload = request.payload()
    assert (
        payload["model"] == MODEL
        and payload["inputs"] == ["spend", "plan"]
        and payload["id"] == "customer_id"
    )
    assert "task" not in payload and "options" not in payload
    # as_of goes out in RFC 3339 UTC however it is given (contract C1).
    naive = dt.datetime(2025, 1, 1)  # noqa: DTZ001 (a naive time is UTC)
    for given in ("2025-01-01T01:00:00+01:00", 1735689600, naive):
        assert PredictRequest.declared_payload([{"x": 1}], as_of=given) == {
            "queries": [{"x": 1}],
            "as_of": "2025-01-01T00:00:00Z",
        }
    with pytest.raises(gf.InvalidInput):
        PredictRequest.declared_payload([{"x": 1}], as_of="2025-01-01")
    with pytest.raises(gf.CapacityExceeded) as caught:
        PredictRequest.build(
            model=MODEL,
            target="churned",
            context=CONTEXT,
            queries=[{"spend": 1.0, "plan": "a"}] * 101,
            id="customer_id",
        )
    assert str(caught.value) == (
        f"error: queries exceed the limit for {MODEL}\n"
        "  because: this model accepts 100 queries per call; the request sent 101\n"
        "  hint: split the queries, or push a predictor and call it in batches"
    )
    assert (caught.value.limit, caught.value.actual) == (100, 101)
    with pytest.raises(gf.InvalidInput) as caught:
        PredictRequest.build(
            model=MODEL,
            target="churned",
            context=CONTEXT,
            queries=CONTEXT,
            exclude=["spend", "plan"],
            id="customer_id",
        )
    assert (
        caught.value.message == "no input columns are left after the target and roles"
    )
    with pytest.raises(gf.InvalidInput) as caught:
        PredictRequest.build(
            model="general-forecast/chronos-2",
            target="churned",
            context=CONTEXT,
            queries=CONTEXT,
        )
    assert caught.value.param == "time"


def test_frame_reads_every_row_shape_and_names_what_it_cannot(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    expected = pa.Table.from_pylist(rows)
    assert tabular.frame(rows) == expected
    assert tabular.frame(pd.DataFrame(rows)) == expected
    assert tabular.frame(expected) == expected
    assert tabular.frame({"a": [1, 2], "b": ["x", "y"]}) == expected
    csv = tmp_path / "rows.csv"
    csv.write_text("a,b\n1,x\n2,y\n")
    assert tabular.frame(csv).to_pylist() == rows
    parquet = tmp_path / "rows.parquet"
    import pyarrow.parquet as pq

    pq.write_table(expected, parquet)
    assert tabular.frame(str(parquet)) == expected
    (tmp_path / "rows.json").write_text('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')
    assert tabular.frame(tmp_path / "rows.json").to_pylist() == rows
    array = tabular.frame(np.array([[1.0, 2.0]]))
    assert array.column_names == ["f0", "f1"]
    with pytest.raises(gf.InvalidInput) as caught:
        tabular.frame(tmp_path / "rows.xml")
    assert caught.value.message == f"no file at `{tmp_path / 'rows.xml'}`"
    (tmp_path / "rows.xml").write_text("<a/>")
    with pytest.raises(gf.InvalidInput) as caught:
        tabular.frame(tmp_path / "rows.xml")
    assert (
        caught.value.hint
        == "pass a .csv, .parquet, .json, .jsonl, .arrow or .feather file"
    )
    with pytest.raises(gf.InvalidInput):
        tabular.frame(None)
    with pytest.raises(gf.InvalidInput):
        tabular.frame(42)


def test_selection_settings_resolve_against_the_model_cap() -> None:
    whole = tables.default_selection(1_000, cap=60_000)
    assert (whole.core, whole.neighbors, whole.whole) == (None, 0, True)
    drawn = tables.default_selection(100_000, cap=60_000, timed=True)
    assert (drawn.core, drawn.neighbors, drawn.kind, drawn.recent) == (
        15_000,
        45_000,
        "stratified",
        180_000,
    )
    fractions = tables.Retrieved(core=0.25, neighbors=0.75, kind="stratified").resolved(
        1_000
    )
    assert (fractions.core, fractions.neighbors) == (250, 750)
    assert (
        str(tables.Retrieved(core=None, neighbors=0).strategy(10)) == "Strategy.exact()"
    )
    assert (
        tables.Retrieved(core=16, neighbors=0, kind="random").strategy(10) is not None
    )
    assert (
        tables.retrieved_of(
            {"core": 0.5, "neighbors": 0.5}, 100, timed=False, rows=1
        ).core
        == 50
    )
    with pytest.raises(gf.InvalidInput):
        tables.Retrieved(core=None, neighbors=4)
    with pytest.raises(gf.InvalidInput):
        tables.Retrieved(core=1.5)
    with pytest.raises(gf.InvalidInput):
        tables.Retrieved(kind=cast(Any, "nearest"))
    with pytest.raises(gf.InvalidInput) as caught:
        tables.retrieved_of({"rows": 3}, 100, timed=False, rows=1)
    assert caught.value.param == "selection.rows"
    assert tables.Grouping(alone=True).strategy() is not None
    assert tables.Grouping(random=4, seed=1).strategy() is not None
    assert tables.Grouping(given=2).strategy() is not None
    with pytest.raises(gf.InvalidInput):
        tables.Grouping()
    with pytest.raises(gf.InvalidInput):
        tables.Grouping(alone=True, given=2)
    assert (
        tables.seconds("7d") == 7 * 86_400
        and tables.seconds("1h") == 3_600
        and tables.seconds(90) == 90
        and tables.seconds("15min") == tables.seconds("15m") == 900
    )
    # One parser with predictor frequencies: no fractional spans, no
    # calendar alias passed off as its mean length.
    for spelled in ("1.5h", "MS", "W-MON", "soon"):
        with pytest.raises(gf.InvalidInput):
            tables.seconds(spelled, "frequency")
    with pytest.raises(gf.InvalidInput):
        tables.TimeOptions(bucket="soon")
    with pytest.raises(gf.InvalidInput):
        tables.Schema(target="y", inputs=["a"], exclude=["b"])
    with pytest.raises(gf.InvalidInput):
        tables.Schema(target="y", id="y")
    with pytest.raises(gf.InvalidInput):
        tables.TableOptions(page_size=0)
    with pytest.raises(gf.InvalidInput):
        tables.SeriesSchema(target="y", time="t", series=["a", "b"], frequency="1d")
    with pytest.raises(gf.InvalidInput):
        tables.SeriesSchema(target="y", time="t", series="a")


def test_engine_tables_check_their_settings_before_touching_the_engine() -> None:
    with pytest.raises(gf.InvalidInput) as caught:
        gf.engine.Table("orders", model=MODEL)
    assert caught.value.message == "a table needs a target: pass target= or schema="
    with pytest.raises(gf.InvalidInput) as caught:
        gf.engine.Table("sales", model="general-forecast/chronos-2", target="units")
    assert caught.value.hint == "use gf.engine.SeriesTable for a forecaster"
    with pytest.raises(gf.InvalidInput):
        gf.engine.Table("orders", model="tabicl-v2-classifier", target="returned")
    table = gf.engine.Table(
        "orders",
        model=MODEL,
        target="returned",
        schema=gf.engine.Schema(target="returned", id="order_id"),
    )
    assert (
        repr(table)
        == f"Table('orders', model={MODEL!r}, target='returned', inputs=?, rows=0)"
    )
    with pytest.raises(gf.InvalidInput) as caught:
        table.predict([{"order_id": 1}])
    assert caught.value.hint == "insert context rows first"
    with pytest.raises(gf.InvalidInput) as caught:
        table.insert([{"order_id": 1, "spend": 2.0}])
    assert caught.value.message == "target column `returned` is not in the rows"
    assert table.insert([]) is None and len(table) == 0


def test_metrics_match_hand_computed_values() -> None:
    truth = [0, 1, 1, 0, 1]
    labels = [0, 1, 0, 0, 1]
    probabilities = {0: [0.9, 0.2, 0.6, 0.7, 0.1], 1: [0.1, 0.8, 0.4, 0.3, 0.9]}
    metrics = classification_metrics(truth, labels, probabilities)
    assert metrics["accuracy"] == pytest.approx(0.8)
    expected_loss = -np.mean(np.log([0.9, 0.8, 0.4, 0.7, 0.9]))
    assert metrics["log loss"] == pytest.approx(expected_loss)
    assert metrics["auroc"] == pytest.approx(1.0)
    three = classification_metrics(
        ["a", "b", "c", "a"],
        ["a", "b", "a", "a"],
        {"a": [1, 0, 1, 1], "b": [0, 1, 0, 0], "c": [0, 0, 0, 0]},
    )
    assert three["macro f1"] == pytest.approx((0.8 + 1.0 + 0.0) / 3)
    regression = regression_metrics(
        [1.0, 2.0, 3.0], [1.0, 2.0, 5.0], {0.1: [0.0, 1.0, 2.0], 0.9: [2.0, 3.0, 6.0]}
    )
    assert regression["mae"] == pytest.approx(2 / 3)
    assert regression["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert regression["pinball 0.1"] == pytest.approx(np.mean([0.1, 0.1, 0.1]))
    assert regression["pinball 0.9"] == pytest.approx(np.mean([0.1, 0.1, 0.3]))
    forecast = forecast_metrics(
        [3.0, 4.0],
        {0.5: [3.0, 5.0], 0.1: [2.0, 4.0], 0.9: [4.0, 6.0]},
        history={"s": [1.0, 2.0, 3.0]},
        series=["s", "s"],
    )
    assert forecast["mase"] == pytest.approx(0.5)
    pinballs = [np.mean([0.0, 0.5]), np.mean([0.1, 0.0]), np.mean([0.1, 0.2])]
    assert forecast["wql"] == pytest.approx(2 * np.mean(pinballs) * 2 / 7)


def test_eval_scores_a_tiny_split_with_known_metrics(monkeypatch) -> None:
    import generalforecast.predictor as predictor_module

    def fake_run(request: PredictRequest, *, download: bool = True) -> pd.DataFrame:
        spend = request.queries.column("spend").to_pylist()
        p1 = [0.9 if value >= 20 else 0.2 for value in spend]
        frame = pd.DataFrame(
            {
                "label": [int(p >= 0.5) for p in p1],
                "p_0": [1 - p for p in p1],
                "p_1": p1,
            }
        )
        if request.model == REGRESSOR:
            frame = pd.DataFrame({"value": spend, "q_0.5": spend})
        return frame

    monkeypatch.setattr(predictor_module, "run_local", fake_run)
    predictor = gf.Predictor(
        MODEL, target="churned", inputs=["spend", "plan"], context=CONTEXT
    )
    held = [
        {"customer_id": 5, "spend": 5.0, "plan": "a", "churned": 0},
        {"customer_id": 6, "spend": 25.0, "plan": "b", "churned": 1},
        {"customer_id": 7, "spend": 35.0, "plan": "a", "churned": 0},
    ]
    evaluation = predictor.eval(held)
    assert isinstance(evaluation, Evaluation)
    scores = evaluation.metrics[MODEL]
    assert scores["accuracy"] == pytest.approx(2 / 3)
    assert scores["log loss"] == pytest.approx(-np.mean(np.log([0.8, 0.9, 0.1])))
    assert scores["auroc"] == pytest.approx(0.75)
    assert list(evaluation.rows.columns) == ["churned", "prediction"]
    assert evaluation.rows["prediction"].tolist() == [0, 1, 1]
    assert str(evaluation).splitlines() == [
        "             accuracy   0.667",
        "             log loss   0.877",
        "             auroc      0.750",
    ]
    with pytest.raises(gf.InvalidInput) as caught:
        predictor.eval(
            pd.DataFrame({"spend": [1.0], "plan": ["a"], "churned": [np.nan]})
        )
    assert (
        caught.value.message
        == "query rows have empty `churned` values; eval needs the truth"
    )


def test_eval_without_queries_holds_out_a_fraction_deterministically(
    monkeypatch,
) -> None:
    import generalforecast.predictor as predictor_module

    seen = []

    def fake_run(request: PredictRequest, *, download: bool = True) -> pd.DataFrame:
        seen.append((len(request.context), len(request.queries)))
        return pd.DataFrame({"value": [0.0] * len(request.queries)})

    monkeypatch.setattr(predictor_module, "run_local", fake_run)
    context = [{"x": float(i), "y": float(i)} for i in range(10)]
    predictor = gf.Predictor(REGRESSOR, target="y", context=context)
    first = predictor.eval()
    second = predictor.eval()
    assert seen == [(8, 2), (8, 2)]
    assert first.rows["y"].tolist() == second.rows["y"].tolist()
    assert first.metrics[REGRESSOR]["mae"] == pytest.approx(np.mean(first.rows["y"]))


def test_local_predictors_read_the_manifest_and_split_the_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "churn"
    root.mkdir()
    (root / "gf.yaml").write_text(
        f"name: churn\nmodel: {MODEL}\ntarget: churned\nid: customer_id\n"
        "inputs:\n  exclude: [note]\ncontext:\n  snapshot: context.csv\n"
    )
    (root / "context.csv").write_text(
        "customer_id,spend,plan,note,churned\n1,10,a,x,0\n2,20,b,y,1\n3,30,a,z,0\n4,40,b,w,1\n5,50,a,v,\n"
    )
    predictor = gf.Predictor(root)
    assert predictor.name == "churn" and predictor.env is None
    assert (predictor.context.kind, predictor.context.rows) == ("snapshot", 4)
    assert (
        repr(predictor)
        == f"Predictor(churn, model={MODEL}, inputs=2, context=snapshot 4 rows, env=None)"
    )
    request = predictor.request(
        [{"customer_id": 9, "spend": 1.0, "plan": "a", "note": "q"}]
    )
    assert request.inputs == ("spend", "plan") and request.id == "customer_id"
    resolution = predictor.explain()
    assert [name for name, _ in resolution.inputs] == ["spend", "plan"]
    assert resolution.classes == (0, 1)
    import generalforecast.predictor as predictor_module

    def fake_run(request: PredictRequest, *, download: bool = True) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": request.queries.column("customer_id").to_pylist(),
                "label": [1],
                "p_0": [0.1],
                "p_1": [0.9],
            }
        )

    monkeypatch.setattr(predictor_module, "run_local", fake_run)
    scored = predictor.predict(predictor._query_frame)
    assert scored["id"].tolist() == [5] and scored["label"].tolist() == [1]
    saved = gf.Predictor(
        MODEL, target="churned", inputs=["spend", "plan"], context=CONTEXT
    ).save(tmp_path / "saved")
    text = (saved / "gf.yaml").read_text()
    assert (
        "name: saved" in text
        and f"model: {MODEL}" in text
        and "snapshot: context.parquet" in text
    )
    assert gf.Predictor(saved).context.rows == 4
    with pytest.raises(gf.InvalidInput):
        gf.Predictor(root, env="staging")
    (root / "gf.yaml").write_text(
        f"name: churn\nmodel: {MODEL}\ncontext:\n  snapshot: context.csv\n"
    )
    with pytest.raises(gf.InvalidInput) as caught:
        gf.Predictor(root)
    assert caught.value.message == "gf.yaml has no `target`"


def test_estimators_use_canonical_ids_and_the_sklearn_hint(monkeypatch) -> None:
    from generalforecast.native import sklearn_shim

    monkeypatch.setattr(sklearn_shim, "MISSING", ImportError("no sklearn"))
    with pytest.raises(gf.InvalidInput) as caught:
        gf.Classifier()
    assert str(caught.value) == (
        "error: the estimators need scikit-learn, which is not installed\n"
        '  hint: pip install "generalforecast[sklearn]"'
    )
    monkeypatch.setattr(sklearn_shim, "MISSING", None)
    with pytest.raises(gf.InvalidInput) as caught:
        gf.Regressor("tabicl-regressor")
    assert caught.value.message == (
        f"`tabicl-regressor` is not a catalog model id; use {REGRESSOR}"
    )
    assert caught.value.hint is None
    monkeypatch.setattr("generalforecast.native.models.is_cached", lambda ref: True)
    assert gf.Classifier().model == MODEL
    assert gf.Regressor().model == REGRESSOR
    assert gf.Forecaster().model == "general-forecast/chronos-2"
    assert gf.Classifier().get_params()["download"] is True


def test_first_download_prints_one_line_with_the_cache_path(
    monkeypatch, capsys
) -> None:
    from generalforecast.native import estimators as native_estimators

    monkeypatch.setattr("generalforecast.native.models.is_cached", lambda ref: False)
    monkeypatch.setattr(
        "generalforecast.native.models.cache_path", lambda ref: Path("/cache/tabicl")
    )
    monkeypatch.setattr(native_estimators, "_load_soon", lambda model, download: None)
    native_estimators._noted.clear()
    gf.Classifier()
    gf.Classifier()
    assert capsys.readouterr().err == f"Downloading {MODEL} to /cache/tabicl\n"
    with pytest.raises(gf.NotFound) as caught:
        gf.Classifier(download=False)
    assert str(caught.value) == (
        f"error: model {MODEL} is not cached\n"
        f"  hint: run `gf model pull {MODEL}`, or pass download=True"
    )


def test_eval_scores_in_batches_of_the_query_cap(monkeypatch) -> None:
    import generalforecast.predictor as predictor_module
    from generalforecast.predictors import _registry

    batches: list[int] = []

    def fake_run(request: PredictRequest, *, download: bool = True) -> pd.DataFrame:
        batches.append(len(request.queries))
        return pd.DataFrame({"value": request.queries.column("x").to_pylist()})

    monkeypatch.setattr(predictor_module, "run_local", fake_run)
    cap = _registry.limits(REGRESSOR).queries
    context = [{"x": float(i), "y": float(i)} for i in range(10)]
    held = [{"x": float(i), "y": float(i)} for i in range(2 * cap + 5)]
    evaluation = gf.Predictor(REGRESSOR, target="y", context=context).eval(held)
    assert batches == [cap, cap, 5]
    assert len(evaluation.rows) == 2 * cap + 5
    assert evaluation.metrics[REGRESSOR]["mae"] == 0.0


def test_a_local_predictor_without_query_rows_asks_for_queries(
    tmp_path: Path, monkeypatch
) -> None:
    import generalforecast.predictor as predictor_module

    root = tmp_path / "flat"
    root.mkdir()
    (root / "context.csv").write_text(
        "x,y\n" + "".join(f"{i},{i % 2}\n" for i in range(20))
    )
    (root / "gf.yaml").write_text(
        f"name: flat\nmodel: {MODEL}\ntarget: y\ncontext:\n  snapshot: context.csv\n"
    )
    predictor = gf.Predictor(root)
    with pytest.raises(errors.InvalidInput) as caught:
        predictor.eval()
    assert caught.value.message == "predictor flat has no query rows to score"
    assert "`gf eval --queries FILE`" in (caught.value.hint or "")

    def fake_run(request: PredictRequest, *, download: bool = True) -> pd.DataFrame:
        truth = request.queries.column("y").to_pylist()
        return pd.DataFrame(
            {
                "label": truth,
                "p_0": [1.0 if t == 0 else 0.0 for t in truth],
                "p_1": [1.0 if t == 1 else 0.0 for t in truth],
            }
        )

    monkeypatch.setattr(predictor_module, "run_local", fake_run)
    evaluation = predictor.eval([{"x": 1.0, "y": 1}, {"x": 2.0, "y": 0}])
    assert evaluation.metrics[MODEL]["accuracy"] == 1.0


def test_served_predictions_echo_the_id_column(monkeypatch) -> None:
    predictor = gf.Predictor("http://127.0.0.1:1/churn")
    posted: list[Any] = []
    monkeypatch.setattr(
        type(predictor),
        "_served_get",
        lambda self, path: {
            "name": "churn",
            "id": "customer_id",
            "task": "classification",
        },
    )

    def served_post(self: Any, path: str, payload: Any) -> dict[str, Any]:
        posted.append(path)
        return {
            "predictions": [
                {
                    "customer_id": 1297,
                    "label": 1,
                    "probabilities": {"0": 0.4, "1": 0.6},
                }
            ],
            "task": "classification",
        }

    monkeypatch.setattr(type(predictor), "_served_post", served_post)
    frame = predictor.predict([{"customer_id": 1297, "spend": 74.8}])
    assert list(frame.columns) == ["id", "label", "p_0", "p_1"]
    assert frame["id"].tolist() == [1297]
    assert posted == ["/predictors/churn/predict"]


def test_forecast_eval_scales_each_series_by_its_own_history(monkeypatch) -> None:
    import datetime as dt

    import generalforecast.predictor as predictor_module

    def fake_run(request: PredictRequest, *, download: bool = True) -> pd.DataFrame:
        return pd.DataFrame({"value": [0.0] * len(request.queries)})

    monkeypatch.setattr(predictor_module, "run_local", fake_run)
    day = dt.date(2025, 1, 1)
    # Two series share the first series column's value; their history
    # arrives out of time order. Each steps by 1 (east) or 4 (west) a day.
    context = [
        {
            "region": "r",
            "store": store,
            "date": day + dt.timedelta(days=d),
            "y": step * d,
        }
        for store, step in (("west", 4.0), ("east", 1.0))
        for d in (3, 0, 2, 1)
    ]
    held = [
        {"region": "r", "store": store, "date": day + dt.timedelta(days=d), "y": 8.0}
        for store in ("east", "west")
        for d in (4, 5)
    ]
    predictor = gf.Predictor(
        "general-forecast/chronos-2",
        target="y",
        time="date",
        series=["region", "store"],
        horizon=2,
        context=context,
    )
    scores = predictor.eval(held).metrics["general-forecast/chronos-2"]
    # |8 - 0| over each series' naive error: 8 for east, 2 for west.
    assert scores["mase"] == pytest.approx((8 + 8 + 2 + 2) / 4)
