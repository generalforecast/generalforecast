# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One manifest, one lock, one pipeline: what gf check locks is what every path binds."""

import datetime
import json
from dataclasses import replace

import pyarrow as pa
import pytest
import yaml

from generalforecast.contracts import (
    ContextVersion,
    PredictionRequest,
    context_version,
)
from generalforecast.errors import InvalidInput
from generalforecast.models.catalog import CHRONOS2_QUANTILES
from generalforecast.predictors import load, resolve
from generalforecast.predictors.lock import document
from generalforecast.predictors.runtime import PredictorExecutor
from generalforecast.predictors.runtime.compile import compile_lock
from generalforecast.predictors.runtime.context import live_context, served_context
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.runtime.schema import infer_schema
from generalforecast.predictors.validation import NAME, PredictorError, name
from generalforecast.serve.models import published_entry
from generalforecast.service.sdk import pushed
from tests.utils.engine import FakeEngine
from tests.utils.locks import ad_hoc, directory, lock_of

CLASSIFIER = "general-forecast/tabicl-v2-classifier"
FORECASTER = "general-forecast/chronos-2"


def churn() -> pa.Table:
    return pa.table(
        {
            "id": list(range(12)),
            "day": [
                datetime.date(2025, 8, 25) + datetime.timedelta(days=i)
                for i in range(12)
            ],
            "spend": [float(i) for i in range(12)],
            "note": [f"n{i}" for i in range(12)],
            "y": [i % 2 for i in range(10)] + [None, None],
        }
    )


def sales(days: int = 12) -> pa.Table:
    start = datetime.date(2025, 1, 1)
    return pa.table(
        {
            "store": [store for store in ("north", "south") for _ in range(days)],
            "date": [
                start + datetime.timedelta(days=d)
                for _ in range(2)
                for d in range(days)
            ],
            "units": [float(d + 3 * s) for s in range(2) for d in range(days)],
            "footfall": [float(d % 5) for _ in range(2) for d in range(days)],
            "promo": [float(d % 2) for _ in range(2) for d in range(days)],
        }
    )


@pytest.mark.parametrize(
    "split",
    [None, {"until": "2025-09-01"}, {"holdout": {"fraction": 0.3, "seed": 2}}],
)
def test_push_sends_exactly_the_context_rows_the_lock_digest_names(tmp_path, split):
    root = directory(
        tmp_path / "churn",
        churn(),
        id="id",
        time="day",
        inputs={"exclude": ["note"]},
        split=split,
    )
    lock = lock_of(root)
    sent = pushed(root)
    assert sent.snapshot is not None
    from generalforecast.tabular import table_digest

    assert table_digest(sent.snapshot) == lock.data_digest
    assert sent.snapshot.num_rows == lock.context["rows"]
    assert sent.snapshot.column_names == ["id", "day", "spend", "y"]
    assert sent.snapshot["y"].null_count == 0
    if split and "until" in split:
        assert max(sent.snapshot["day"].to_pylist()) < datetime.date(2025, 9, 1)


def test_push_refuses_a_stale_lock_or_a_changed_snapshot(tmp_path):
    from generalforecast.service.contracts import ServiceError

    root = directory(tmp_path / "churn", churn(), id="id")
    manifest = root / "gf.yaml"
    manifest.write_text(manifest.read_text() + "selection: {seed: 5}\n")
    with pytest.raises(ServiceError) as caught:
        pushed(root)
    assert caught.value.code == "LockRequired"
    other = directory(tmp_path / "other", churn(), id="id")
    from generalforecast.tabular import arrow_bytes

    (other / "history.arrow").write_bytes(
        arrow_bytes(churn().set_column(2, "spend", pa.array([9.0] * 12)))
    )
    with pytest.raises(ServiceError) as caught:
        pushed(other)
    assert caught.value.code == "SnapshotMismatch"


def test_resources_travel_in_the_lock(tmp_path):
    root = directory(
        tmp_path / "churn",
        churn(),
        id="id",
        resources={"accelerator": "L40S", "replicas": {"min": 1, "max": 3}},
    )
    assert lock_of(root).resources == {
        "accelerator": "L40S",
        "replicas_min": 1,
        "replicas_max": 3,
    }


def test_a_forecasters_unmarked_inputs_are_past_covariates_everywhere(
    tmp_path, monkeypatch
):
    FakeEngine(CHRONOS2_QUANTILES).install(monkeypatch)
    root = directory(
        tmp_path / "sales",
        sales(),
        model=FORECASTER,
        target="units",
        time="date",
        series=["store"],
        horizon=2,
        inputs={"future": ["promo"]},
    )
    lock = lock_of(root)
    roles = {column["name"]: column["role"] for column in lock.predictor["inputs"]}
    assert roles == {"footfall": "past", "promo": "future"}
    plan = compile_lock(lock, served_context(lock, sales()))
    assert plan.spec.past_inputs == ("footfall",)
    assert plan.spec.future_inputs == ("promo",)
    entry = published_entry(plan)["predictor"]
    assert entry["forecast"]["past_inputs"] == ["footfall"]
    assert entry["forecast"]["future_inputs"] == ["promo"]
    assert entry["inputs"] == ["footfall", "promo"]


def test_published_entries_type_times_by_their_rows_and_keep_categories(tmp_path):
    root = directory(
        tmp_path / "sales",
        sales(),
        model=FORECASTER,
        target="units",
        time="date",
        series=["store"],
        horizon=2,
        inputs={"past": ["footfall"], "future": ["promo"]},
    )
    lock = lock_of(root)
    plan = compile_lock(lock, served_context(lock, sales()))
    entry = published_entry(plan)["predictor"]
    columns = {column["name"]: column for column in entry["schema"]["columns"]}
    assert columns["date"]["type"] == "date" and columns["date"]["role"] == "time"
    assert columns["store"]["type"] == "string"
    assert entry["forecast"]["frequency"] == 86400.0
    assert entry["forecast"]["calendar"] is None
    assert entry["lock_digest"] == lock.digest
    typed = directory(
        tmp_path / "typed",
        pa.table({"zip": [1, 2, 3, 1], "x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]}),
        inputs={"types": {"zip": "category"}},
    )
    lock = lock_of(typed)
    plan = compile_lock(lock, served_context(lock, typed_rows()))
    columns = {
        column["name"]: column
        for column in published_entry(plan)["predictor"]["schema"]["columns"]
    }
    assert columns["zip"]["semantic"] == "category"


def typed_rows() -> pa.Table:
    return pa.table({"zip": [1, 2, 3, 1], "x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})


def test_a_univariate_forecast_locks_and_publishes(tmp_path):
    rows = sales().select(["store", "date", "units"])
    root = directory(
        tmp_path / "sales",
        rows,
        model=FORECASTER,
        target="units",
        time="date",
        series=["store"],
        horizon=2,
    )
    lock = lock_of(root)
    assert lock.predictor["inputs"] == []
    entry = published_entry(compile_lock(lock, served_context(lock, rows)))
    assert entry["predictor"]["forecast"]["past_inputs"] == []
    assert entry["predictor"]["inputs"] == []


def test_monthly_contexts_lock_a_calendar_frequency_and_forecast_on_it(
    tmp_path, monkeypatch
):
    FakeEngine(CHRONOS2_QUANTILES).install(monkeypatch)
    months = [datetime.date(2023, m, 1) for m in range(1, 13)]
    rows = pa.table({"month": months, "units": [float(i) for i in range(12)]})
    root = directory(
        tmp_path / "monthly",
        rows,
        model=FORECASTER,
        target="units",
        time="month",
        horizon=3,
    )
    lock = lock_of(root)
    assert lock.predictor["frequency"] == "MS"
    plan = compile_lock(lock, served_context(lock, rows))
    assert published_entry(plan)["predictor"]["forecast"]["calendar"] == "MS"
    prepared = PredictorExecutor().prepare(plan, served_context(lock, rows))
    queries = pa.table({"month": [datetime.date(2024, m, 1) for m in (1, 2, 3)]})
    assert prepared.predict(queries).num_rows == 3
    with pytest.raises(Exception, match="horizon"):
        prepared.predict(
            pa.table(
                {
                    "month": [
                        datetime.date(2024, 1, 31),
                        datetime.date(2024, 3, 2),
                        datetime.date(2024, 4, 1),
                    ]
                }
            )
        )


def test_live_forecasts_compile_with_any_declared_frequency():
    from generalforecast.predictors.models import CatalogResolver

    pin = CatalogResolver().resolve(FORECASTER)
    for cadence, step in (
        ("15min", 900.0),
        ("1ms", 0.001),
        ("MS", 1.0),
        ("W-MON", 1.0),
    ):
        plan = ExecutionPlan.build(
            "live-sales",
            pin.resolved,
            task="forecast",
            target="y",
            mode="request",
            time="t",
            horizon=2,
            frequency=cadence,
            past_inputs=["p"],
        )
        typed = replace(
            plan,
            target_type="float",
            inputs=replace(plan.inputs, types=(("p", "float"),)),
            role_types=(("t", "timestamp"),),
        )
        assert infer_schema(typed, pin, "forecast", None).frequency == step


def test_gf_check_and_the_runtime_refuse_the_same_irregular_times(tmp_path):
    rows = pa.table(
        {
            "date": [datetime.date(2025, 1, d) for d in (1, 2, 3, 3, 4)],
            "units": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with pytest.raises(InvalidInput) as caught:
        directory(
            tmp_path / "dup",
            rows,
            model=FORECASTER,
            target="units",
            time="date",
            horizon=2,
        )
    assert caught.value.message == "duplicate forecast time within a series"
    gappy = pa.table(
        {
            "date": [datetime.date(2025, 1, d) for d in (1, 2, 3, 10, 11)],
            "units": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with pytest.raises(InvalidInput) as caught:
        directory(
            tmp_path / "gap",
            gappy,
            model=FORECASTER,
            target="units",
            time="date",
            horizon=2,
        )
    assert (
        caught.value.message == "forecast frequency is irregular; resample explicitly"
    )


def test_check_admits_quantiles_as_the_runtime_does(tmp_path):
    rows = sales().select(["store", "date", "units"])
    with pytest.raises(InvalidInput) as caught:
        directory(
            tmp_path / "levels",
            rows,
            model="general-forecast/timesfm-3",
            target="units",
            time="date",
            series=["store"],
            horizon=2,
            quantiles=[0.123],
        )
    assert "quantile" in caught.value.message


def test_models_without_a_published_digest_are_not_locked(tmp_path):
    with pytest.raises(InvalidInput) as caught:
        directory(
            tmp_path / "limix",
            typed_rows(),
            model="general-forecast/limix-2-classifier",
        )
    assert caught.value.param == "model"
    assert "cannot be locked" in caught.value.message


def test_ad_hoc_selection_is_honoured(monkeypatch):
    fake = FakeEngine().install(monkeypatch)
    rows = pa.table(
        {"x": [float(i) for i in range(40)], "y": [i % 2 for i in range(40)]}
    )
    lock = ad_hoc(
        rows,
        model=CLASSIFIER,
        target="y",
        inputs=["x"],
        selection={"core": 8, "neighbors": 4, "seed": 1},
    )
    selection = lock.predictor["selection"]
    assert (selection["strategy"], selection["core"], selection["seed"]) == (
        "retrieved",
        8,
        1,
    )
    plan = compile_lock(lock, rows)
    prepared = PredictorExecutor().prepare(plan, None)
    prepared.predict(pa.table({"x": [1.0]}), context=rows)
    [strategy] = fake.strategies
    assert strategy is not None and json.loads(strategy)["core"]["rows"] == 8


def test_live_contexts_bind_through_the_one_served_context(tmp_path, monkeypatch):
    FakeEngine().install(monkeypatch)
    root = directory(tmp_path / "live", typed_rows(), context="live", inputs=["x"])
    lock = lock_of(root)
    snapshot = {
        "rows": [
            {"key": str(i), "features": {"x": float(i), "zip": 1}, "label": i % 2}
            for i in range(6)
        ],
        "through_sequence": 7,
        "knowledge_at": 1700000000.0,
    }
    rows, version = live_context(lock, snapshot)
    assert rows.column_names == ["x", "y"]
    assert version == ContextVersion("live", 7, 6, 1700000000.0, name="events")
    plan = compile_lock(lock, rows)
    assert (
        PredictorExecutor().prepare(plan, rows).predict(pa.table({"x": [1.0]})).num_rows
        == 1
    )


def test_context_versions_name_a_live_context_apart_from_its_kind():
    version = ContextVersion("live", 3, 10, name="events")
    assert version.to_dict() == {
        "source": "live",
        "version": 3,
        "rows": 10,
        "as_of": None,
        "name": "events",
    }
    assert context_version({"source": "live/events", "version": 3}) == ContextVersion(
        "live", 3, name="events"
    )
    with pytest.raises(PredictorError):
        ContextVersion("live/events")


def test_prediction_requests_carry_as_of():
    request = PredictionRequest.from_dict(
        {"queries": [{"x": 1}], "as_of": "2025-01-01T00:00:00Z"}
    )
    assert request.as_of == "2025-01-01T00:00:00Z"
    assert request.to_dict() == {"queries": [{"x": 1}], "as_of": "2025-01-01T00:00:00Z"}
    with pytest.raises(PredictorError):
        PredictionRequest.from_dict({"queries": [{"x": 1}], "as_of": "soon"})


def test_one_name_rule_for_chosen_names():
    from generalforecast.predictors.spec import NAME as SPEC_NAME
    from generalforecast.service.contracts import NAME as SERVICE_NAME

    assert SPEC_NAME is NAME is SERVICE_NAME
    assert name("churn-2") == "churn-2"
    for bad in ("Churn", "churn_2", "2churn", ""):
        with pytest.raises(PredictorError):
            name(bad)


def test_the_lock_document_round_trips_through_yaml(tmp_path):
    root = directory(tmp_path / "churn", churn(), id="id")
    value = yaml.safe_load((root / "gf.lock").read_text())
    assert value == document(resolve(load(root), root))


def test_serving_refuses_a_lock_pinned_to_another_artifact(tmp_path):
    from generalforecast.predictors.lock import lock_digest

    root = directory(tmp_path / "churn", churn(), id="id")
    value = yaml.safe_load((root / "gf.lock").read_text())
    value["predictor"]["model_digest"] = "sha256:" + "1" * 64
    value["digest"] = lock_digest(value)
    with pytest.raises(PredictorError) as caught:
        compile_lock(value, served_context(value, churn()))
    assert caught.value.code == "ModelMismatch"


def test_calendar_periods_win_over_equal_gaps():
    from generalforecast.predictors import frequency

    years = [datetime.date(2021 + i, 1, 1) for i in range(3)]
    assert frequency.infer({"a": years}) == "YS"
    quarters = [datetime.date(2024, 7, 1), datetime.date(2024, 10, 1)]
    assert frequency.infer({"a": quarters}) == "QS"
    months = [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)]
    assert frequency.infer({"a": months}) == "MS"
    weeks = [
        datetime.date(2024, 1, 1) + datetime.timedelta(days=7 * i) for i in range(4)
    ]
    assert frequency.infer({"a": weeks}) == "1w"
    zoned = datetime.datetime(
        2024, 1, 31, 19, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))
    )
    stepped = frequency.grid(zoned, 2, "MS")
    assert [stamp.isoformat() for stamp in stepped] == [
        "2024-03-01T00:00:00+00:00",
        "2024-04-01T00:00:00+00:00",
    ]


def test_validation_names_what_is_malformed(tmp_path):
    from generalforecast.predictors.lock import lock_digest, validate

    root = directory(tmp_path / "churn", churn(), id="id")
    good = yaml.safe_load((root / "gf.lock").read_text())

    def broken(change):
        value = yaml.safe_load(yaml.safe_dump(good))
        change(value)
        value["digest"] = lock_digest(value)
        with pytest.raises(PredictorError):
            validate(value)

    broken(lambda v: v["predictor"].pop("id_type"))
    broken(lambda v: v["predictor"]["context"].update(until=5))
    broken(lambda v: v["predictor"].update(classes="ab"))
    broken(lambda v: v["predictor"].update(outputs=["nonsense"]))
    broken(lambda v: v["predictor"]["inputs"].append(dict(v["predictor"]["inputs"][0])))
    broken(lambda v: v["predictor"]["selection"].update(strategy="nearest"))


def test_a_snapshot_without_its_target_is_a_clear_error(tmp_path):
    from generalforecast.service.contracts import ServiceError
    from generalforecast.tabular import arrow_bytes

    root = directory(tmp_path / "churn", churn(), id="id")
    (root / "history.arrow").write_bytes(arrow_bytes(churn().drop(["y"])))
    with pytest.raises(ServiceError) as caught:
        pushed(root)
    assert "target column `y` is not in the context" in caught.value.message


def test_frozen_checks_and_local_predictors_refuse_a_stale_lock(tmp_path, monkeypatch):
    import generalforecast as gf
    from generalforecast.predictors import load_lock
    from generalforecast.tabular import arrow_bytes

    root = directory(tmp_path / "churn", churn(), id="id")
    (root / "history.arrow").write_bytes(
        arrow_bytes(churn().set_column(2, "spend", pa.array([7.0] * 12)))
    )
    with pytest.raises(InvalidInput) as caught:
        gf.Predictor(root)
    assert caught.value.message.endswith("is out of date")
    manifest = root / "gf.yaml"
    manifest.write_text(manifest.read_text() + "selection: {seed: 2}\n")
    with pytest.raises(InvalidInput) as caught:
        load_lock(root)
    assert caught.value.message == "gf.lock was written for another gf.yaml"


def test_unordered_quantiles_lock_in_order(tmp_path):
    rows = pa.table(
        {"x": [float(i) for i in range(10)], "price": [float(i) for i in range(10)]}
    )
    root = directory(
        tmp_path / "price",
        rows,
        model="general-forecast/tabicl-v2-regressor",
        target="price",
        quantiles=[0.9, 0.1],
    )
    assert lock_of(root).predictor["quantiles"] == [0.1, 0.9]
