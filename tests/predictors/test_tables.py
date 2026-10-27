# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pytest

from generalforecast.errors import InvalidInput
from generalforecast.predictors import PredictorSpec, inspect_table, split
from generalforecast.predictors.spec import Context, Holdout
from generalforecast.predictors.tables import inspect_frame
from tests.predictors.conftest import MODEL, write_context


def test_inspect_table_kinds_and_candidates(tmp_path: Path):
    path = write_context(tmp_path / "churn")
    info = inspect_table(path)
    assert info.rows == 200
    assert info.kinds() == {
        "customer_id": "int",
        "spend": "float",
        "plan": "category",
        "signups": "int",
        "signed_up": "timestamp",
        "churned": "0/1",
        "note": "text",
    }
    assert info.column("customer_id").unique
    assert info.column("churned").empty == 20
    assert info.targets[0] == "churned"
    assert "plan" in info.targets and "customer_id" not in info.targets
    assert info.ids[0] == "customer_id"
    assert info.times == ("signed_up",)


KINDS = [
    (pa.array([True, False, None]), "bool", 1),
    (pa.array([0, 1, 1, 0]), "0/1", 0),
    (pa.array([0.0, 1.0, None]), "0/1", 1),
    (pa.array([1, 2, 3, 4]), "int", 0),
    (pa.array([1.5, 2.5, float("nan")]), "float", 1),
    (pa.array(["0", "1", "1"]), "0/1", 0),
    (pa.array(["yes", "no", ""]), "bool", 1),
    (pa.array(["a", "b", "a", "  "]), "category", 1),
    (pa.array([f"row {index}" for index in range(100)]), "text", 0),
    (pa.array(["2025-01-01", "2025-01-02", None]), "timestamp", 1),
]


@pytest.mark.parametrize(("values", "kind", "empty"), KINDS)
def test_kinds(values, kind, empty):
    info = inspect_frame(pa.table({"x": values}))
    assert info.column("x").kind == kind
    assert info.column("x").empty == empty


def test_inspect_json_and_parquet(tmp_path: Path):
    table = pa.table({"k": [1, 2, 3], "y": [0, 1, 0]})
    import pyarrow.parquet as pq

    pq.write_table(table, tmp_path / "t.parquet")
    (tmp_path / "t.json").write_text(
        '[{"k": 1, "y": 0}, {"k": 2, "y": 1}, {"k": 3, "y": 0}]'
    )
    for name in ("t.parquet", "t.json"):
        info = inspect_table(tmp_path / name)
        assert info.kinds() == {"k": "int", "y": "0/1"}
        assert info.ids == ("k",) and info.targets == ("y",)


def test_inspect_errors(tmp_path: Path):
    with pytest.raises(InvalidInput) as caught:
        inspect_table(tmp_path / "missing.csv")
    assert caught.value.message == "`missing.csv` does not exist"
    (tmp_path / "x.txt").write_text("a")
    with pytest.raises(InvalidInput) as caught:
        inspect_table(tmp_path / "x.txt")
    assert (
        caught.value.message == "`x.txt` is not a .csv, .parquet, .arrow or .json file"
    )
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({"a": pa.array([], pa.int64())}), tmp_path / "empty.parquet"
    )
    with pytest.raises(InvalidInput) as caught:
        inspect_table(tmp_path / "empty.parquet")
    assert caught.value.message == "`empty.parquet` has no rows"


def spec(**changes) -> PredictorSpec:
    values: dict = {"name": "t", "model": MODEL, "target": "y"}
    values["context"] = Context("snapshot", "c.csv")
    values.update(changes)
    return PredictorSpec(**values)


TABLE = pa.table(
    {
        "t": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
        "y": [0, 1, 0, 1],
    }
)


def test_split_keeps_empty_targets_as_queries_and_partitions_the_labeled_rows():
    table = pa.table({"y": [0, None, 1, None]})
    context, queries = split(
        table, spec(context=Context("snapshot", "c.csv", holdout=Holdout(0.5)))
    )
    assert context.num_rows == 1 and queries.num_rows == 3
    assert queries["y"].null_count == 2


def test_until_is_honoured_when_some_targets_are_blank():
    days = [f"2025-08-{day:02d}" for day in range(1, 32)] + [
        f"2025-09-{day:02d}" for day in range(1, 30)
    ]
    labels: list[float | None] = [float(index) for index in range(len(days))]
    labels[5] = None
    table = pa.table({"t": days, "x": [1.0] * len(days), "y": labels})
    context, queries = split(
        table, spec(time="t", context=Context("snapshot", "c.csv", until="2025-09-01"))
    )
    assert max(context["t"].to_pylist()) == "2025-08-31"
    assert context.num_rows == 30
    assert queries.num_rows == 30
    assert "2025-08-06" in queries["t"].to_pylist()


def test_split_until():
    context, queries = split(
        TABLE, spec(time="t", context=Context("snapshot", "c.csv", until="2025-03-01"))
    )
    assert context["t"].to_pylist() == ["2025-01-01", "2025-02-01"]
    assert queries["t"].to_pylist() == ["2025-03-01", "2025-04-01"]


def test_split_holdout_is_seeded():
    one = split(
        TABLE, spec(context=Context("snapshot", "c.csv", holdout=Holdout(0.5, 3)))
    )
    two = split(
        TABLE, spec(context=Context("snapshot", "c.csv", holdout=Holdout(0.5, 3)))
    )
    assert one[1].num_rows == 2 and one[0].num_rows == 2
    assert one[1].to_pylist() == two[1].to_pylist()
    three = split(
        TABLE, spec(context=Context("snapshot", "c.csv", holdout=Holdout(0.25, 3)))
    )
    assert three[1].num_rows == 1


def test_split_without_a_rule_keeps_everything():
    context, queries = split(TABLE, spec())
    assert context.num_rows == 4 and queries.num_rows == 0


def test_split_missing_target_names_the_columns():
    with pytest.raises(InvalidInput) as caught:
        split(TABLE, spec(target="label"))
    assert caught.value.message == "target column `label` is not in the context"
    assert caught.value.hint == "the context has t, y"


def test_task_for_reads_the_target_kind():
    from generalforecast.predictors.tables import ColumnInfo, task_for

    def column(kind: str, distinct: int) -> ColumnInfo:
        return ColumnInfo(name="y", kind=kind, unique=False, empty=0, distinct=distinct)

    assert task_for(column("int", 3), 178) == "classification"
    assert task_for(column("int", 20), 100_000) == "classification"
    assert task_for(column("int", 40), 1_000) == "classification"
    assert task_for(column("int", 60), 1_000) == "regression"
    assert task_for(column("float", 3), 178) == "regression"
    for kind in ("0/1", "bool", "category", "text"):
        assert task_for(column(kind, 2), 10) == "classification"


def test_whole_number_floats_keep_integer_labels():
    from generalforecast.predictors.tables import labels, whole_numbers

    floats = pa.chunked_array([pa.array([0.0, 1.0, None, 1.0])])
    cast = whole_numbers(floats)
    assert cast is not None and cast.to_pylist() == [0, 1, None, 1]
    assert whole_numbers(pa.chunked_array([pa.array([0.5, 1.0])])) is None
    assert whole_numbers(pa.chunked_array([pa.array([1, 2])])) is None
    assert whole_numbers(pa.chunked_array([pa.array([None], pa.float64())])) is None
    assert labels(floats).to_pylist() == [0, 1, 1]
    assert labels(pa.chunked_array([pa.array(["a", "", None, "b"])])).to_pylist() == [
        "a",
        "b",
    ]


def _store_sales(stores: int = 3, days: int = 20, *, ragged: bool = False) -> pa.Table:
    import datetime as dt

    rows: dict[str, list] = {"date": [], "store": [], "sales": []}
    for store in range(stores):
        for day in range(days - (store if ragged else 0)):
            rows["date"].append(dt.date(2024, 1, 1) + dt.timedelta(days=day))
            rows["store"].append(f"s{store}")
            rows["sales"].append(float(10 + store + day % 7))
    return pa.table(rows)


def test_target_candidates_follow_the_task():
    from generalforecast.predictors import target_candidates

    sales = inspect_frame(_store_sales())
    assert target_candidates(sales, "forecast") == ("sales",)
    assert target_candidates(sales, "regress") == ("sales",)
    assert target_candidates(sales, "classify") == ("store",)
    housing = inspect_frame(
        pa.table(
            {
                "house_id": list(range(40)),
                "rooms": [1 + i % 5 for i in range(40)],
                "area": [50.0 + i % 9 for i in range(40)],
                "zone": [("a", "b")[i % 2] for i in range(40)],
                "price": [100.0 + 3 * i for i in range(40)],
            }
        )
    )
    # Floats before ints, later before earlier; never the id.
    assert target_candidates(housing, "regress") == ("price", "area", "rooms")
    assert target_candidates(housing, "classify") == ("zone", "rooms")
    labeled = inspect_frame(
        pa.table({"y": [1, 2, 3, 1] * 10, "x": [0.5 * i for i in range(40)]})
    )
    assert target_candidates(labeled, "regression")[0] == "y"
    texts = inspect_frame(pa.table({"note": [f"n{i}" for i in range(40)]}))
    assert target_candidates(texts, "classify") == ()
    with pytest.raises(ValueError, match="unknown task"):
        target_candidates(sales, "cluster")


def test_default_until_holds_out_exactly_the_horizon_of_every_series():
    import datetime as dt

    from generalforecast.predictors import default_until
    from generalforecast.predictors.tables import split_by

    table = _store_sales()
    until = default_until(table, time="date", series=["store"], horizon=4)
    assert until == "2024-01-17"
    context, queries = split_by(table, "sales", time="date", until=until)
    assert queries.group_by("store").aggregate([("sales", "count")])[
        "sales_count"
    ].to_pylist() == [4, 4, 4]
    assert context.num_rows == 3 * 16
    # One frequency given; one series; timestamps give a UTC timestamp.
    hourly = pa.table(
        {"ts": [dt.datetime(2024, 3, 9, h, tzinfo=dt.UTC) for h in range(10)]}
    )
    assert (
        default_until(hourly, time="ts", series=[], horizon=3, frequency="1h")
        == "2024-03-09T07:00:00"
    )
    # Ragged ends leave some series a partial horizon; too few times; a gap.
    assert (
        default_until(
            _store_sales(ragged=True), time="date", series=["store"], horizon=4
        )
        is None
    )
    assert (
        default_until(_store_sales(days=4), time="date", series=["store"], horizon=4)
        is None
    )
    gappy = _store_sales(stores=1).filter(pa.array([i != 5 for i in range(20)]))
    assert default_until(gappy, time="date", series=["store"], horizon=4) is None


def test_predictor_local_reads_a_bare_name_as_a_directory(tmp_path, monkeypatch):
    from generalforecast.errors import NotFound
    from generalforecast.predictor import Predictor

    monkeypatch.chdir(tmp_path)
    (tmp_path / "churn").mkdir()
    with pytest.raises(NotFound, match=r"no gf\.yaml in `churn`") as raised:
        Predictor.local("churn")
    assert "gf init" in str(raised.value.hint)
    with pytest.raises(NotFound, match="no directory"):
        Predictor.local("absent")
