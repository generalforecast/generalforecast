# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Package execution contract tests. Injected numeric kernels are not model evidence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import yaml

from generalforecast.deployments.contracts import ServiceError
from generalforecast.deployments.interfaces import PreparedExecution
from generalforecast.deployments.package_runtime import PackageExecutor
from generalforecast.packages import ModelPin, Package, PackageLock
from generalforecast.packages.validation import digest
from generalforecast.tabular import arrow_bytes


def checked(root, data, **changes):
    root.mkdir(exist_ok=True)
    spec = {
        "name": "test-model",
        "model": "tabicl-classifier",
        "context": "history.arrow",
        "target": "y",
        **changes,
    }
    (root / "gf.yaml").write_text(yaml.safe_dump(spec))
    (root / "history.arrow").write_bytes(arrow_bytes(data))
    package = Package.load(root)
    return package.check(data=data if package.spec.context.mode != "snapshot" else None)


@pytest.fixture
def data():
    return pa.table(
        {
            "id": ["a", "b", "c", "d"],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [False, False, True, True],
        }
    )


@pytest.fixture
def kernels(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "generalforecast.deployments.package_runtime._load",
        lambda *args: (SimpleNamespace(), None),
    )

    def fit(model, x, y, count):
        calls.append(np.asarray(x).copy())
        return count, float(np.mean(np.asarray(y)))

    def predict(model, fitted, x, classes):
        count, cache = fitted
        if count:
            return np.tile([0.25, 0.75], (len(x), 1))
        return np.tile(np.arange(999, dtype=float) + cache, (len(x), 1))

    monkeypatch.setattr("generalforecast.models.inference.fit", fit)
    monkeypatch.setattr("generalforecast.models.inference.predict", predict)
    return calls


def test_prepare_warms_once_preserves_ids_outputs_and_immutable_state(
    tmp_path, data, kernels
):
    package = checked(tmp_path, data, id="id", outputs=["probabilities"])
    prepared = PackageExecutor().prepare(package.lock, data)
    assert prepared.context_rows == 4 and len(kernels) == 1
    assert isinstance(prepared, PreparedExecution) and prepared.usable
    q = pa.table({"id": ["q"], "x": [1.5]})
    assert prepared.predict(q).to_pylist() == [
        {"id": "q", "probabilities": {"false": 0.25, "true": 0.75}}
    ]
    prepared.predict(q)
    assert len(kernels) == 1
    assert prepared.predict(q).schema.metadata == {
        b"generalforecast.context_rows": b"4"
    }
    with pytest.raises(FrozenInstanceError):
        prepared.context_rows = 100  # ty: ignore[invalid-assignment]


def test_full_snapshot_checked_before_explicit_selection(tmp_path, data, kernels):
    package = checked(
        tmp_path,
        data,
        id="id",
        context={
            "source": "history.arrow",
            "selection": "most_recent",
            "order_by": "x",
            "max_rows": 3,
        },
    )
    prepared = PackageExecutor().prepare(package.lock, data)
    assert prepared.context_rows == 3
    assert kernels[0][:, 0].tolist() == [1, 2, 3]
    with pytest.raises(ServiceError, match="snapshot digest") as error:
        PackageExecutor().prepare(package.lock, data.slice(1))
    assert error.value.code == "SnapshotMismatch"


def test_dev_replacement_does_not_mutate_existing_context(tmp_path, data, kernels):
    first = checked(tmp_path / "first", data, id="id")
    old = PackageExecutor().prepare(first.lock, data)
    changed = data.set_column(1, "x", pa.array([10.0, 11.0, 12.0, 13.0]))
    second = checked(tmp_path / "second", changed, id="id")
    new = PackageExecutor().prepare(second.lock, changed)
    assert old.lock != new.lock and old._context is not new._context
    assert kernels[0][:, 0].tolist() == [0, 1, 2, 3]
    assert kernels[1][:, 0].tolist() == [10, 11, 12, 13]
    assert old.predict(pa.table({"x": [1.0]})).num_rows == 1
    with pytest.raises(ServiceError):
        PackageExecutor().prepare(second.lock, data)
    assert old.context_rows == 4


@pytest.mark.parametrize("mode", ["request", "dataset/events"])
def test_dynamic_context_validation_and_no_override(tmp_path, data, kernels, mode):
    package = checked(tmp_path, data, context=mode, id="id")
    executor = PackageExecutor()
    prepared = executor.prepare(package.lock, None if mode == "request" else data)
    q = pa.table({"x": [1.0]})
    if mode == "request":
        with pytest.raises(ServiceError) as error:
            prepared.predict(q)
        assert error.value.code == "ContextRequired"
        prepared.predict(q, context=data)
        prepared.predict(q, context=data)
        assert len(kernels) == 2 and prepared._context is None
    else:
        with pytest.raises(ServiceError):
            prepared.predict(q, context=data)
        assert len(kernels) == 1


@pytest.mark.parametrize(
    "queries",
    [{"x": [1]}, {"x": [None]}, {"x": [1.0], "y": [True]}, {"unknown": [1.0]}],
)
def test_queries_reject_physical_drift_missing_leakage_and_unknown(
    tmp_path, data, kernels, queries
):
    package = checked(tmp_path, data, id="id")
    prepared = PackageExecutor().prepare(package.lock, data)
    with pytest.raises(ServiceError) as error:
        prepared.predict(pa.table(queries))
    assert error.value.status == 422


def test_forged_capability_rehashed_lock_still_fails_admission(tmp_path, data, kernels):
    package = checked(tmp_path, data, id="id")
    value = package.lock.to_dict()
    value["model"]["max_context_rows"] += 1
    value["model_digest"] = digest(ModelPin.from_dict(value["model"]).identity())
    value["package"] = digest({k: v for k, v in value.items() if k != "package"})
    lock = PackageLock.from_dict(value)
    with pytest.raises(ServiceError) as error:
        PackageExecutor().prepare(lock, data)
    assert error.value.code == "ModelMismatch" and kernels == []


def test_missing_bound_context_and_unexpected_request_context(tmp_path, data, kernels):
    package = checked(tmp_path, data, id="id")
    with pytest.raises(ServiceError):
        PackageExecutor().prepare(package.lock, None)
    package = checked(tmp_path, data, context="request", id="id")
    with pytest.raises(ServiceError):
        PackageExecutor().prepare(package.lock, data)


def test_actual_labeled_rows_metadata_for_request_selection(tmp_path, data, kernels):
    data = data.set_column(2, "y", pa.array([None, False, True, True]))
    package = checked(tmp_path, data, context="request", id="id")
    prepared = PackageExecutor().prepare(package.lock, None)
    result = prepared.predict(pa.table({"x": [1.0]}), context=data)
    assert result.schema.metadata == {b"generalforecast.context_rows": b"3"}
    assert prepared.context_rows is None


def test_query_intermediates_batched_and_capacity_rejected_not_sampled(
    tmp_path, data, kernels
):
    package = checked(tmp_path, data, id="id")
    prepared = PackageExecutor().prepare(package.lock, data)
    q = pa.table({"id": [str(i) for i in range(257)], "x": [1.0] * 257})
    result = prepared.predict(q)
    assert result.num_rows == 257 and result["id"].equals(q["id"])
    assert len(kernels) == 1
    with pytest.raises(ServiceError) as error:
        prepared.predict(pa.table({"x": [1.0] * 10_001}))
    assert error.value.code == "CapacityExceeded"


def test_schema_free_request_and_regression_projection(tmp_path, kernels):
    data = pa.table({"x": [1.0, 2.0], "y": [3.0, 5.0]})
    package = checked(
        tmp_path,
        data,
        context="request",
        model="tabicl-regressor",
        outputs=["quantiles", "value"],
        quantiles=[0.1, 0.5, 0.9],
    )
    lock = replace(package.lock, schema=type(package.lock.schema)())
    value = lock.to_dict()
    value["package"] = digest({k: v for k, v in value.items() if k != "package"})
    prepared = PackageExecutor().prepare(PackageLock.from_dict(value), None)
    result = prepared.predict(pa.table({"x": [1.5]}), context=data)
    assert result.column_names == ["quantiles", "value"]
    assert result.to_pylist() == [
        {"quantiles": {"0.1": 103.0, "0.5": 503.0, "0.9": 903.0}, "value": 503.0}
    ]


def test_concurrent_request_contexts_do_not_mutate_bound_production(tmp_path, kernels):
    first = pa.table({"x": [1.0, 2.0], "y": [3.0, 5.0]})
    second = first.set_column(1, "y", pa.array([7.0, 9.0]))
    request = checked(
        tmp_path / "request", first, context="request", model="tabicl-regressor"
    )
    snapshot = checked(tmp_path / "snapshot", first, model="tabicl-regressor")
    executor = PackageExecutor()
    remote = executor.prepare(request.lock, None)
    production = executor.prepare(snapshot.lock, first)
    queries = pa.table({"x": [1.5]})

    def predict(i):
        result = remote.predict(queries, context=first if i % 2 == 0 else second)
        return result["value"][0].as_py()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(predict, range(8))) == [503.0, 507.0] * 4
    assert production.predict(queries)["value"][0].as_py() == 503.0
    assert remote._context is None and remote.context_rows is None
    assert production.context_rows == 2


def test_target_free_schema_free_request_can_warm_without_guessing_target(
    tmp_path, kernels
):
    (tmp_path / "gf.yaml").write_text("name: test-model\nmodel: tabicl-classifier\n")
    package = Package.load(tmp_path).check(write=False)
    assert package.lock is not None
    prepared = PackageExecutor().prepare(package.lock, None)
    assert prepared.lock.spec.target is None and prepared._context is None
    assert prepared.context_rows is None and not kernels
    with pytest.raises(ServiceError, match="explicit target binding") as error:
        prepared.predict(
            pa.table({"x": [1.0]}), context=pa.table({"x": [1.0], "y": [True]})
        )
    assert error.value.code == "InvalidPackage"


@pytest.mark.parametrize("context", ["history.arrow", "dataset/history", "request"])
def test_target_free_schema_bearing_or_bound_context_is_rejected(
    tmp_path, data, kernels, context
):
    package = checked(tmp_path, data, context=context)
    lock = replace(package.lock, spec=replace(package.lock.spec, target=None))
    value = lock.to_dict()
    value["package"] = digest({k: v for k, v in value.items() if k != "package"})
    with pytest.raises(ServiceError, match="explicit target"):
        PackageExecutor().prepare(
            replace(lock, package_digest=value["package"]),
            None if context == "request" else data,
        )


def test_bad_native_output_and_private_error_are_typed(
    tmp_path, data, kernels, monkeypatch
):
    package = checked(tmp_path, data, id="id")
    prepared = PackageExecutor().prepare(package.lock, data)
    monkeypatch.undo()
    monkeypatch.setattr(
        "generalforecast.models.inference._tabular",
        lambda _: (None, lambda *args: np.array([[np.nan, 1.0]])),
    )
    with pytest.raises(ServiceError) as error:
        prepared.predict(pa.table({"x": [1.0]}))
    assert error.value.code == "ModelOutputError"

    def private(*args):
        raise RuntimeError("credential=must-not-leak")

    monkeypatch.setattr("generalforecast.deployments.package_runtime._load", private)
    with pytest.raises(ServiceError) as error:
        PackageExecutor().prepare(package.lock, data)
    assert error.value.code == "InferenceFailed" and "credential" not in str(
        error.value
    )
