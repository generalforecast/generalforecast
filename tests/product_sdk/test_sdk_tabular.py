import dataclasses
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import feather, parquet

import generalforecast as gf
from generalforecast.deployments.sdk import Dataset, prediction_payload, table


@pytest.mark.parametrize(
    "suffix", ["csv", "tsv", "parquet", "arrow", "feather", "json", "jsonl"]
)
def test_local_file_formats_use_foundation_physical_normalization(tmp_path, suffix):
    path = tmp_path / ("queries." + suffix)
    data = pa.table({"x": [1, 2], "z": [3.0, 4.0]})
    if suffix in {"csv", "tsv"}:
        separator = "," if suffix == "csv" else "\t"
        path.write_text(f"x{separator}z\n1{separator}3.0\n2{separator}4.0\n")
    elif suffix == "parquet":
        parquet.write_table(data, path)
    elif suffix == "feather":
        feather.write_feather(data, path)
    elif suffix == "arrow":
        with (
            path.open("wb") as stream,
            pa.ipc.new_stream(stream, data.schema) as writer,
        ):
            writer.write_table(data)
    elif suffix == "json":
        path.write_text(json.dumps(data.to_pylist()))
    else:
        path.write_text("\n".join(json.dumps(row) for row in data.to_pylist()))
    normalized = table(path)
    assert normalized.equals(data)
    assert prediction_payload(path)["queries"] == data.to_pylist()


def test_numpy_requires_explicit_feature_and_context_column_order():
    queries = np.array([[5, 2], [8, 4]])
    context = np.array([[3, 1, 0], [4, 2, 1]])
    with pytest.raises(gf.ServiceError, match="explicit feature column order"):
        prediction_payload(queries)
    with pytest.raises(gf.ServiceError, match="explicit feature column order"):
        prediction_payload(queries, features=["b", "a"], context=context)
    payload = prediction_payload(
        queries,
        features=["b", "a"],
        context=context,
        context_columns=["b", "a", "y"],
        target="y",
    )
    assert payload["queries"] == [{"b": 5, "a": 2}, {"b": 8, "a": 4}]
    assert payload["context"][1] == {"b": 4, "a": 2, "y": 1}
    assert payload["features"] == ["b", "a"]


def test_pandas_polars_lazyframes_and_typed_results():
    pandas = pytest.importorskip("pandas")
    polars = pytest.importorskip("polars")
    pandas_data = pandas.DataFrame({"x": [2, 1]})
    polars_data = polars.DataFrame({"x": [2, 1]})
    assert table(pandas_data).equals(table(polars_data))
    assert table(polars_data.lazy()).equals(table(polars_data))
    with pytest.raises(gf.PackageError):
        table(polars.LazyFrame({"x": range(1000001)}))
    result = gf.PredictionResult(
        pa.table({"value": [0.2, 0.8], "id": ["b", "a"]}), "catalog/model", "regression"
    )
    assert result.to_pandas()["id"].tolist() == ["b", "a"]
    assert result.to_polars()["value"].to_list() == [0.2, 0.8]
    assert result.to_numpy().shape == (2, 2)
    assert result.to_arrow().column_names == ["value", "id"]


def test_immutable_nested_snapshots_and_specs():
    source = {"nested": {"items": [1, 2]}}
    snapshot = gf.Snapshot.from_dict(source)
    source["nested"]["items"].append(3)
    snapshot["nested"]["items"].clear()
    assert snapshot.to_dict() == {"nested": {"items": [1, 2]}}
    dataset = Dataset.from_response(None, "history", {"sequence": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataset.name = "new"
    features = ["x"]
    targets = [["y", "numeric"]]
    spec = gf.DatasetSpec("history", features, targets)
    features.append("leaked")
    targets[0][0] = "changed"
    assert spec.features == ("x",) and spec.targets == (("y", "numeric"),)


def test_invalid_local_sources_are_safe_and_never_open_pipes(tmp_path):
    path = tmp_path / "pipe.csv"
    os.mkfifo(path)
    with pytest.raises(gf.ServiceError, match="regular file"):
        table(path)
    link = tmp_path / "link.csv"
    link.symlink_to(path)
    with pytest.raises(gf.ServiceError):
        table(link)
    with pytest.raises(gf.ServiceError):
        table("https://example.com?token=sensitive")


def test_root_estimators_are_lazy_and_owned_by_the_estimator_module(monkeypatch):
    owner = SimpleNamespace(
        Classifier=object(), Regressor=object(), Forecaster=object()
    )
    for name in ("Classifier", "Regressor", "Forecaster"):
        monkeypatch.setitem(gf.__dict__, name, None)
    monkeypatch.setitem(sys.modules, "generalforecast.estimators", owner)
    monkeypatch.setattr(gf, "estimators", owner, raising=False)
    assert gf.__getattr__("Classifier") is owner.Classifier
    assert gf.__getattr__("Regressor") is owner.Regressor
    assert gf.__getattr__("Forecaster") is owner.Forecaster
