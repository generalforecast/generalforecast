# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import socket

import pytest
import yaml

from generalforecast.packages import (
    CatalogResolver,
    Package,
    PackageError,
    PackageLock,
    PackageSpec,
)
from generalforecast.packages.validation import digest
from generalforecast.tabular import table_digest


def package(tmp_path, **changes):
    spec = {
        "name": "fraud",
        "model": "tabicl-classifier",
        "target": "y",
        "context": "history.csv",
        **changes,
    }
    (tmp_path / "gf.yaml").write_text(yaml.safe_dump(spec))
    (tmp_path / "history.csv").write_text("id,x,y\na,1,false\nb,2,true\nc,3,false\n")
    return Package.load(tmp_path)


def test_snapshot_compiles_offline_and_is_immutable(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline compilation attempted network or weight loading")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    value = package(tmp_path, id="id")
    checked = value.check()
    assert value.lock is None
    assert checked.lock.task == "classification"
    assert checked.lock.schema.features == ("x",)
    assert checked.lock.schema.classes == (False, True)
    assert checked.lock.schema.columns[0].role == "id"
    assert checked.lock.model.resolved.startswith(
        "catalog/tabicl-v2-classifier@sha256:"
    )
    assert Package.load(tmp_path).check(locked=True).lock == checked.lock
    assert Package.load(tmp_path).compile() == checked.lock
    assert checked.context_table().num_rows == 3
    assert table_digest(checked.snapshot_table()) == checked.lock.data_digest
    assert (
        len(
            {
                checked.lock.source_digest,
                checked.lock.data_digest,
                checked.lock.model_digest,
                checked.lock.package_digest,
            }
        )
        == 4
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        checked.spec.name = "changed"
    wire = checked.lock.to_dict()
    wire["spec"]["name"] = "changed"
    assert checked.spec.name == "fraud"


def test_inspects_physical_strings_without_semantic_or_leakage_guesses(tmp_path):
    checked = package(tmp_path).check()
    assert checked.lock.schema.features == ("id", "x")
    field = checked.lock.schema.columns[0]
    assert field.type == "string" and field.semantic is None
    assert field.role == "input"


@pytest.mark.parametrize("context", ["request", "dataset/events"])
def test_non_snapshot_data_binds_schema_not_current_rows_or_content(tmp_path, context):
    value = package(tmp_path, context=context, id="id")
    first = value.check(data={"id": ["a", "b"], "x": [1, 2], "y": [False, True]})
    second = Package.load(tmp_path).check(
        data={"id": ["a", "b", "c"], "x": [4, 5, 6], "y": [False, True, False]},
        locked=True,
    )
    assert first.lock == second.lock
    assert first.lock.data_digest is None
    assert first.lock.schema.rows is None and first.lock.context_rows is None


def test_regression_quantile_resolution_is_checked_before_runtime(tmp_path):
    value = package(
        tmp_path,
        model="tabicl-regressor",
        outputs=["value", "quantiles"],
        quantiles=[0.1234],
    )
    with pytest.raises(PackageError) as error:
        value.check()
    assert error.value.code == "UnsupportedCapability"


def test_forecast_total_covariates_are_bounded(tmp_path):
    columns = [f"feature{i}" for i in range(9)]
    value = package(
        tmp_path,
        model="chronos-2",
        target="y",
        context="dataset/events",
        time="t",
        horizon=2,
        past_inputs=columns,
    )
    data = {**{column: [1, 2] for column in columns}, "t": [1, 2], "y": [3, 4]}
    with pytest.raises(PackageError) as error:
        value.check(data=data)
    assert error.value.code == "CapacityExceeded"


def test_forecast_input_overlap_is_one_physical_channel(tmp_path):
    columns = [f"feature{i}" for i in range(8)]
    value = package(
        tmp_path,
        model="chronos-2",
        target="y",
        context="dataset/events",
        time="t",
        past_inputs=columns,
        future_inputs=list(reversed(columns)),
        horizon=2,
    )
    data = {**{column: [1, 2] for column in columns}, "t": [1, 2], "y": [3, 4]}
    checked = value.check(data=data)
    assert checked.spec.forecast_inputs == tuple(columns)
    assert checked.lock.task == "forecast"
    assert [
        c.name for c in checked.lock.schema.columns if c.role == "future"
    ] == columns
    assert PackageLock.from_dict(checked.lock.to_dict()) == checked.lock
    assert Package.load(tmp_path).check(data=data, locked=True).lock == checked.lock


@pytest.mark.parametrize("role", ["target", "id", "time", "series"])
@pytest.mark.parametrize("input_role", ["past_inputs", "future_inputs"])
def test_forecast_overlap_does_not_weaken_reserved_roles(role, input_role):
    value = {"name": "forecast", "model": "chronos-2", "target": "y"}
    value[role] = ["reserved"] if role == "series" else "reserved"
    value[input_role] = ["reserved"]
    with pytest.raises(PackageError):
        PackageSpec.from_dict(value)


def test_existing_schema_requires_explicit_upgrade_to_change(tmp_path):
    value = package(tmp_path, context="dataset/events", id="id")
    value.check(data={"id": ["a", "b"], "x": [1, 2], "y": [False, True]})
    drift = {"id": ["a", "b"], "x": [1.5, 2.5], "y": [False, True]}
    previous = (tmp_path / "gf.lock").read_bytes()
    with pytest.raises(PackageError) as error:
        Package.load(tmp_path).check(data=drift)
    assert error.value.code == "SchemaChanged"
    assert (tmp_path / "gf.lock").read_bytes() == previous
    upgraded = Package.load(tmp_path).check(data=drift, upgrade=True)
    assert isinstance(upgraded.lock, PackageLock)
    assert upgraded.lock.schema.columns[1].type == "float64"


def test_locked_and_frozen_are_read_only_and_detect_changed_source(tmp_path):
    checked = package(tmp_path, id="id").check()
    before = (tmp_path / "gf.lock").stat().st_mtime_ns
    checked.check(locked=True)
    checked.check(frozen=True)
    assert (tmp_path / "gf.lock").stat().st_mtime_ns == before
    (tmp_path / "history.csv").write_text("id,x,y\na,9,false\nb,2,true\n")
    for mode in ("locked", "frozen"):
        with pytest.raises(PackageError, match="differs") as error:
            checked.check(**{mode: True})
        assert error.value.code == "LockOutdated"
    assert (tmp_path / "gf.lock").stat().st_mtime_ns == before


def test_manifest_change_requires_reload(tmp_path):
    original = package(tmp_path)
    (tmp_path / "gf.yaml").write_text("name: other\nmodel: tabicl-classifier\n")
    with pytest.raises(PackageError) as error:
        original.check()
    assert error.value.code == "SourceChanged"


@pytest.mark.parametrize("mode", ["frozen", "locked"])
def test_modes_require_existing_lock_and_forbid_upgrade(tmp_path, mode):
    value = package(tmp_path)
    with pytest.raises(PackageError) as error:
        value.check(**{mode: True})
    assert error.value.code == "LockRequired"
    with pytest.raises(PackageError):
        value.check(**{mode: True}, upgrade=True)
    with pytest.raises(PackageError):
        value.check(**{mode: True}, offline=False)


def test_lock_integrity_versions_and_unknown_fields(tmp_path):
    lock = package(tmp_path).check().lock
    assert PackageLock.from_dict(lock.to_dict()) == lock
    for key, value in (
        ("package", "sha256:" + "0" * 64),
        ("lock_version", 2),
        ("token", "never-persist"),
    ):
        wire = lock.to_dict()
        wire[key] = value
        with pytest.raises(PackageError):
            PackageLock.from_dict(wire)
    wire = lock.to_dict()
    wire["model"]["files"]["config.json"] = "0" * 64
    wire["package"] = digest(
        {key: value for key, value in wire.items() if key != "package"}
    )
    with pytest.raises(PackageError, match="artifact digest"):
        PackageLock.from_dict(wire)


def test_model_resolution_never_silently_upgrades(tmp_path):
    checked = package(tmp_path).check()
    original_pin = checked.lock.model
    new_pin = dataclasses.replace(original_pin, max_features=31)

    class Resolver:
        def resolve(self, requested, *, offline=True):
            return new_pin

    assert checked.check(resolver=Resolver(), write=False).lock.model == original_pin
    assert (
        checked.check(resolver=Resolver(), upgrade=True, write=False).lock.model
        == new_pin
    )
    with pytest.raises(PackageError, match="explicit upgrade"):
        checked.resolve(resolver=Resolver(), write=False)


def test_ambiguity_and_unsupported_tasks_fail(tmp_path):
    pin = CatalogResolver().resolve("tabicl-classifier")

    class Resolver:
        def resolve(self, requested, *, offline=True):
            return dataclasses.replace(pin, tasks=("classification", "regression"))

    with pytest.raises(PackageError) as error:
        package(tmp_path).check(resolver=Resolver())
    assert error.value.code == "AmbiguousTask"
    assert (
        package(tmp_path, task="classification").check(resolver=Resolver()).lock.task
        == "classification"
    )
    with pytest.raises(PackageError) as error:
        package(tmp_path, task="forecast").check(upgrade=True)
    assert error.value.code == "UnsupportedTask"


def test_requested_model_digest_must_match():
    with pytest.raises(PackageError) as error:
        CatalogResolver().resolve("catalog/tabicl-v2-classifier@sha256:" + "0" * 64)
    assert error.value.code == "ModelMismatch"
    with pytest.raises(PackageError) as error:
        CatalogResolver().resolve("unknown")
    assert error.value.code == "ModelResolutionRequired"


def test_online_qualification_is_explicit_metadata_only(monkeypatch):
    calls = []
    pin = CatalogResolver().resolve("tabicl-classifier")

    def model_info(self, repository, *, revision):
        calls.append((repository, revision))
        return type("Info", (), {"sha": revision})()

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", model_info)
    assert CatalogResolver().resolve("tabicl-classifier", offline=False) == pin
    assert calls == [(pin.repository, pin.revision)]


def test_request_and_live_context(tmp_path):
    value = package(tmp_path, context=None, target=None).check()
    assert not value.lock.schema.columns and value.lock.data_digest is None
    package(tmp_path, context="dataset/transactions")
    with pytest.raises(PackageError) as error:
        Package.load(tmp_path).check()
    assert error.value.code == "SchemaRequired"
    value = package(
        tmp_path,
        context="dataset/transactions",
        id="id",
        target={"column": "y", "type": "boolean"},
        inputs={"types": {"id": "string", "x": "int64"}},
    ).check()
    assert value.lock.schema.features == ("x",)
    assert value.lock.data_digest is None
    assert value.lock.spec.context.source == "dataset/transactions"


def test_live_context_changes_do_not_pin_dataset_contents(tmp_path):
    value = package(tmp_path, context="dataset/transactions", id="id")
    first = value.check(
        data=[{"id": "a", "x": 1, "y": True}, {"id": "b", "x": 2, "y": False}],
        write=False,
    )
    second = value.check(
        data=[{"id": "a", "x": 9, "y": True}, {"id": "b", "x": 8, "y": False}],
        write=False,
    )
    assert first.lock == second.lock
    assert first.lock.data_digest is None


def test_explicit_context_selection_no_first_n_fallback(tmp_path):
    pin = dataclasses.replace(
        CatalogResolver().resolve("tabicl-classifier"), max_context_rows=2
    )

    class Resolver:
        def resolve(self, requested, *, offline=True):
            return pin

    with pytest.raises(PackageError) as error:
        package(tmp_path).check(resolver=Resolver())
    assert error.value.code == "CapacityExceeded"
    selected = package(
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
    checked = selected.check(resolver=Resolver())
    assert checked.context_table()["id"].to_pylist() == ["b", "c"]
    assert checked.lock.context_rows == 2


def test_query_validation_is_locked(tmp_path):
    value = package(tmp_path, id="id").check()
    assert value.validate_queries([{"id": "q", "x": 3}]).num_rows == 1
    assert value.validate_queries([{"x": 3, "id": "q"}]).column_names == ["id", "x"]
    for data in ([{"y": True, "x": 3}], [{"x": 3.5}], [{"id": "q"}]):
        with pytest.raises(PackageError):
            value.validate_queries(data)


def test_forecast_contract_and_frequency(tmp_path):
    value = package(
        tmp_path, model="chronos-2", target="y", time="t", series=["store"], horizon=2
    )
    (tmp_path / "history.csv").write_text(
        "store,t,y\na,1,3\na,2,5\na,3,4\nb,1,7\nb,2,8\n"
    )
    checked = value.check()
    assert checked.lock.task == "forecast"
    assert checked.lock.schema.frequency == 1
    assert not checked.lock.schema.features
    (tmp_path / "history.csv").write_text("store,t,y\na,1,3\na,2,5\na,4,4\n")
    with pytest.raises(PackageError) as error:
        Package.load(tmp_path).check()
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
    value = package(
        tmp_path,
        **({"model": "chronos-2", "target": "y", "time": "x", "horizon": 2} | change),
    )
    (tmp_path / "history.csv").write_text("x,y\n1,3\n2,5\n3,4\n")
    with pytest.raises(PackageError):
        value.check()


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
    with pytest.raises(PackageError):
        package(tmp_path, **change).check()


def test_roundtrip_maximal_spec():
    value = {
        "name": "fraud",
        "model": "tabicl-classifier",
        "task": "classification",
        "target": {"column": "y", "type": "boolean"},
        "id": "id",
        "context": {
            "source": "dataset/transactions",
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
    spec = PackageSpec.from_dict(value)
    assert PackageSpec.from_dict(spec.to_dict()) == spec
    assert "runtime" not in spec.to_dict()
