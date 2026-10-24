# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import socket

import pytest
import yaml

from generalforecast.deployments.predictor_runtime.compile import (
    compile_dir,
    compile_plan,
    plan_from_dir,
    validate_queries,
)
from generalforecast.deployments.predictor_runtime.lock import ExecutionLock
from generalforecast.deployments.predictor_runtime.plan import ExecutionPlan
from generalforecast.deployments.predictor_runtime.schema import select_context
from generalforecast.predictors.models import CatalogResolver
from generalforecast.predictors.validation import PredictorError, digest
from generalforecast.tabular import table_digest, to_arrow


def plan_dir(tmp_path, **changes):
    spec = {
        "name": "fraud",
        "model": "general-forecast/tabicl-v2-classifier",
        "target": "y",
        "context": "history.csv",
        **changes,
    }
    (tmp_path / "gf.yaml").write_text(yaml.safe_dump(spec))
    (tmp_path / "history.csv").write_text("id,x,y\na,1,false\nb,2,true\nc,3,false\n")
    return tmp_path


def check(tmp_path, data=None, **options):
    plan, snapshot = plan_from_dir(tmp_path)
    table = snapshot if snapshot is not None else to_arrow(data) if data else None
    return plan, compile_plan(plan, table, **options)


def test_snapshot_compiles_offline_and_is_immutable(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline compilation attempted network or weight loading")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    plan_dir(tmp_path, id="id")
    plan, table = plan_from_dir(tmp_path)
    lock = compile_plan(plan, table)
    assert lock.task == "classification"
    assert lock.schema.features == ("x",)
    assert lock.schema.classes == (False, True)
    assert lock.schema.columns[0].role == "id"
    assert lock.model.resolved.startswith(
        "general-forecast/tabicl-v2-classifier@sha256:"
    )
    assert compile_dir(tmp_path) == lock
    assert compile_plan(plan, table, previous=lock) == lock
    assert select_context(plan, table, lock.model).num_rows == 3
    assert table_digest(table) == lock.data_digest
    assert (
        len(
            {
                lock.source_digest,
                lock.data_digest,
                lock.model_digest,
                lock.lock_digest,
            }
        )
        == 4
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(plan, "name", "changed")  # noqa: B010
    wire = lock.to_dict()
    wire["spec"]["name"] = "changed"
    assert lock.spec.name == "fraud"


def test_inspects_physical_strings_without_semantic_or_leakage_guesses(tmp_path):
    lock = compile_dir(plan_dir(tmp_path))
    assert lock.schema.features == ("id", "x")
    field = lock.schema.columns[0]
    assert field.type == "string" and field.semantic is None
    assert field.role == "input"


@pytest.mark.parametrize("context", ["request", "live/events"])
def test_non_snapshot_data_binds_schema_not_current_rows_or_content(tmp_path, context):
    plan, first = check(
        plan_dir(tmp_path, context=context, id="id"),
        {"id": ["a", "b"], "x": [1, 2], "y": [False, True]},
    )
    second = compile_plan(
        plan,
        to_arrow({"id": ["a", "b", "c"], "x": [4, 5, 6], "y": [False, True, False]}),
        previous=first,
    )
    assert first == second
    assert first.data_digest is None
    assert first.schema.rows is None and first.context_rows is None


def test_regression_quantile_resolution_is_checked_before_runtime(tmp_path):
    plan_dir(
        tmp_path,
        model="general-forecast/tabicl-v2-regressor",
        outputs=["value", "quantiles"],
        quantiles=[0.1234],
    )
    with pytest.raises(PredictorError) as error:
        compile_dir(tmp_path)
    assert error.value.code == "UnsupportedCapability"


def test_forecast_total_covariates_are_bounded(tmp_path):
    columns = [f"feature{i}" for i in range(9)]
    plan_dir(
        tmp_path,
        model="general-forecast/chronos-2",
        target="y",
        context="live/events",
        time="t",
        horizon=2,
        past_inputs=columns,
    )
    data = {**{column: [1, 2] for column in columns}, "t": [1, 2], "y": [3, 4]}
    with pytest.raises(PredictorError) as error:
        check(tmp_path, data)
    assert error.value.code == "CapacityExceeded"


def test_forecast_input_overlap_is_one_physical_channel(tmp_path):
    columns = [f"feature{i}" for i in range(8)]
    plan_dir(
        tmp_path,
        model="general-forecast/chronos-2",
        target="y",
        context="live/events",
        time="t",
        past_inputs=columns,
        future_inputs=list(reversed(columns)),
        horizon=2,
    )
    data = {**{column: [1, 2] for column in columns}, "t": [1, 2], "y": [3, 4]}
    plan, lock = check(tmp_path, data)
    assert plan.forecast_inputs == tuple(columns)
    assert lock.task == "forecast"
    assert [c.name for c in lock.schema.columns if c.role == "future"] == columns
    assert ExecutionLock.from_dict(lock.to_dict()) == lock
    assert compile_plan(plan, to_arrow(data), previous=lock) == lock


@pytest.mark.parametrize("role", ["target", "id", "time", "series"])
@pytest.mark.parametrize("input_role", ["past_inputs", "future_inputs"])
def test_forecast_overlap_does_not_weaken_reserved_roles(role, input_role):
    value = {"name": "forecast", "model": "general-forecast/chronos-2", "target": "y"}
    value[role] = ["reserved"] if role == "series" else "reserved"
    value[input_role] = ["reserved"]
    with pytest.raises(PredictorError):
        ExecutionPlan.from_dict(value)


def test_existing_schema_requires_explicit_upgrade_to_change(tmp_path):
    plan, first = check(
        plan_dir(tmp_path, context="live/events", id="id"),
        {"id": ["a", "b"], "x": [1, 2], "y": [False, True]},
    )
    drift = to_arrow({"id": ["a", "b"], "x": [1.5, 2.5], "y": [False, True]})
    with pytest.raises(PredictorError) as error:
        compile_plan(plan, drift, previous=first)
    assert error.value.code == "SchemaChanged"
    upgraded = compile_plan(plan, drift, previous=first, upgrade=True)
    assert upgraded.schema.columns[1].type == "float64"


def test_lock_integrity_versions_and_unknown_fields(tmp_path):
    lock = compile_dir(plan_dir(tmp_path))
    assert ExecutionLock.from_dict(lock.to_dict()) == lock
    for key, value in (
        ("lock", "sha256:" + "0" * 64),
        ("lock_version", 9),
        ("token", "never-persist"),
    ):
        wire = lock.to_dict()
        wire[key] = value
        with pytest.raises(PredictorError):
            ExecutionLock.from_dict(wire)
    wire = lock.to_dict()
    wire["model"]["files"]["config.json"] = "0" * 64
    wire["lock"] = digest({key: value for key, value in wire.items() if key != "lock"})
    with pytest.raises(PredictorError, match="artifact digest"):
        ExecutionLock.from_dict(wire)


def test_model_resolution_never_silently_upgrades(tmp_path):
    plan, table = plan_from_dir(plan_dir(tmp_path))
    lock = compile_plan(plan, table)
    original_pin = lock.model
    new_pin = dataclasses.replace(original_pin, max_features=31)

    class Resolver:
        def resolve(self, requested, *, offline=True):
            return new_pin

    kept = compile_plan(plan, table, resolver=Resolver(), previous=lock)
    assert kept.model == original_pin
    upgraded = compile_plan(
        plan, table, resolver=Resolver(), previous=lock, upgrade=True
    )
    assert upgraded.model == new_pin
    with pytest.raises(PredictorError, match="explicit upgrade"):
        compile_plan(plan, table, resolver=Resolver(), previous=lock, offline=False)


def test_ambiguity_and_unsupported_tasks_fail(tmp_path):
    pin = CatalogResolver().resolve("general-forecast/tabicl-v2-classifier")

    class Resolver:
        def resolve(self, requested, *, offline=True):
            return dataclasses.replace(pin, tasks=("classification", "regression"))

    with pytest.raises(PredictorError) as error:
        compile_dir(plan_dir(tmp_path), resolver=Resolver())
    assert error.value.code == "AmbiguousTask"
    assert (
        compile_dir(plan_dir(tmp_path, task="classification"), resolver=Resolver()).task
        == "classification"
    )
    with pytest.raises(PredictorError) as error:
        compile_dir(plan_dir(tmp_path, task="forecast"), upgrade=True)
    assert error.value.code == "UnsupportedTask"


def test_requested_model_digest_must_match():
    with pytest.raises(PredictorError) as error:
        CatalogResolver().resolve(
            "general-forecast/tabicl-v2-classifier@sha256:" + "0" * 64
        )
    assert error.value.code == "ModelMismatch"
    with pytest.raises(PredictorError) as error:
        CatalogResolver().resolve("unknown")
    assert error.value.code == "ModelResolutionRequired"


def test_online_qualification_is_explicit_metadata_only(monkeypatch):
    calls = []
    pin = CatalogResolver().resolve("general-forecast/tabicl-v2-classifier")

    def model_info(self, repository, *, revision):
        calls.append((repository, revision))
        return type("Info", (), {"sha": revision})()

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", model_info)
    assert (
        CatalogResolver().resolve(
            "general-forecast/tabicl-v2-classifier", offline=False
        )
        == pin
    )
    assert calls == [(pin.repository, pin.revision)]


def test_request_and_live_context(tmp_path):
    lock = compile_dir(plan_dir(tmp_path, context=None, target=None))
    assert not lock.schema.columns and lock.data_digest is None
    with pytest.raises(PredictorError) as error:
        compile_dir(plan_dir(tmp_path, context="live/transactions"))
    assert error.value.code == "SchemaRequired"
    lock = compile_dir(
        plan_dir(
            tmp_path,
            context="live/transactions",
            id="id",
            target={"column": "y", "type": "boolean"},
            inputs={"types": {"id": "string", "x": "int64"}},
        )
    )
    assert lock.schema.features == ("x",)
    assert lock.data_digest is None
    assert lock.spec.context.source == "live/transactions"


def test_live_context_changes_do_not_pin_dataset_contents(tmp_path):
    plan, _ = plan_from_dir(plan_dir(tmp_path, context="live/transactions", id="id"))
    first = compile_plan(
        plan,
        to_arrow([{"id": "a", "x": 1, "y": True}, {"id": "b", "x": 2, "y": False}]),
    )
    second = compile_plan(
        plan,
        to_arrow([{"id": "a", "x": 9, "y": True}, {"id": "b", "x": 8, "y": False}]),
    )
    assert first == second
    assert first.data_digest is None


def test_explicit_context_selection_no_first_n_fallback(tmp_path):
    pin = dataclasses.replace(
        CatalogResolver().resolve("general-forecast/tabicl-v2-classifier"),
        max_context_rows=2,
    )

    class Resolver:
        def resolve(self, requested, *, offline=True):
            return pin

    with pytest.raises(PredictorError) as error:
        compile_dir(plan_dir(tmp_path), resolver=Resolver())
    assert error.value.code == "CapacityExceeded"
    plan_dir(
        tmp_path,
        id="id",
        inputs={"exclude": ["x"]},
        context={
            "source": "history.csv",
            "selection": "most_recent",
            "order_by": "x",
            "max_rows": 2,
        },
    )
    # Keep an actual feature while explicitly excluding the order column.
    (tmp_path / "history.csv").write_text(
        "id,x,z,y\nc,3,4,false\na,1,2,true\nb,2,3,true\n"
    )
    plan, table = plan_from_dir(tmp_path)
    lock = compile_plan(plan, table, resolver=Resolver())
    assert select_context(plan, table, lock.model)["id"].to_pylist() == ["b", "c"]
    assert lock.context_rows == 2


def test_query_validation_is_locked(tmp_path):
    lock = compile_dir(plan_dir(tmp_path, id="id"))
    assert validate_queries(lock, [{"id": "q", "x": 3}]).num_rows == 1
    assert validate_queries(lock, [{"x": 3, "id": "q"}]).column_names == ["id", "x"]
    for data in ([{"y": True, "x": 3}], [{"x": 3.5}], [{"id": "q"}]):
        with pytest.raises(PredictorError):
            validate_queries(lock, data)


def test_forecast_contract_and_frequency(tmp_path):
    plan_dir(
        tmp_path,
        model="general-forecast/chronos-2",
        target="y",
        time="t",
        series=["store"],
        horizon=2,
    )
    (tmp_path / "history.csv").write_text(
        "store,t,y\na,1,3\na,2,5\na,3,4\nb,1,7\nb,2,8\n"
    )
    lock = compile_dir(tmp_path)
    assert lock.task == "forecast"
    assert lock.schema.frequency == 1
    assert not lock.schema.features
    (tmp_path / "history.csv").write_text("store,t,y\na,1,3\na,2,5\na,4,4\n")
    with pytest.raises(PredictorError) as error:
        compile_dir(tmp_path)
    assert error.value.code == "AmbiguousFrequency"


@pytest.mark.parametrize(
    "change",
    [
        {"horizon": 100},
        {"outputs": ["probabilities"]},
        {"quantiles": [0.123], "outputs": ["quantiles"]},
        {"time": "missing"},
    ],
)
def test_forecast_unsupported_contract_fails(tmp_path, change):
    plan_dir(
        tmp_path,
        **(
            {
                "model": "general-forecast/chronos-2",
                "target": "y",
                "time": "x",
                "horizon": 2,
            }
            | change
        ),
    )
    (tmp_path / "history.csv").write_text("x,y\n1,3\n2,5\n3,4\n")
    with pytest.raises(PredictorError):
        compile_dir(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"secrets": {"key": "secret"}},
        {"runtime": "azure"},
        {"compute": {"token": "secret"}},
        {"task": "auto"},
        {"context": "s3://bucket/key.csv"},
        {"context": {"source": "history.csv", "max_rows": 1}},
        {"context": "../other.csv"},
        {"inputs": {"include": ["y"]}},
        {"id": "y"},
    ],
)
def test_invalid_or_secret_bearing_intent_fails(tmp_path, change):
    with pytest.raises(PredictorError):
        compile_dir(plan_dir(tmp_path, **change))


def test_roundtrip_maximal_spec():
    value = {
        "name": "fraud",
        "model": "general-forecast/tabicl-v2-classifier",
        "task": "classification",
        "target": {"column": "y", "type": "boolean"},
        "id": "id",
        "context": {
            "source": "live/transactions",
            "selection": "most_recent",
            "order_by": "accepted_at",
            "max_rows": 100,
        },
        "inputs": {
            "include": ["x"],
            "types": {"x": "int64"},
            "missing_values": "reject",
        },
        "outputs": ["label"],
        "compute": {"cpu": 4, "memory": "16Gi"},
        "placement": {"region": "eastus", "residency": ["eastus"]},
        "scaling": {"min_replicas": 1, "max_replicas": 4},
    }
    spec = ExecutionPlan.from_dict(value)
    assert ExecutionPlan.from_dict(spec.to_dict()) == spec
    assert "runtime" not in spec.to_dict()
