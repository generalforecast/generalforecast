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


def test_split_prefers_empty_targets():
    table = pa.table({"y": [0, None, 1, None]})
    context, queries = split(
        table, spec(context=Context("snapshot", "c.csv", holdout=Holdout(0.5)))
    )
    assert context.num_rows == 2 and queries.num_rows == 2
    assert queries["y"].null_count == 2


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
