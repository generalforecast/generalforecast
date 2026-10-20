import dataclasses
import json

import pyarrow as pa
import pytest

from generalforecast.contracts import (
    ContextVersion,
    DatasetImportRef,
    DevSession,
    Environment,
    PatchRequest,
    PredictionResult,
    PromoteRequest,
    Revision,
    RuntimePlan,
)
from generalforecast.packages import (
    ModelPin,
    Package,
    PackageError,
    PackageLock,
    PythonLock,
)
from generalforecast.packages.validation import canonical, digest

HASH = "sha256:" + "a" * 64


@pytest.fixture
def python_lock(tmp_path):
    (tmp_path / "gf.yaml").write_text(
        "name: custom-score\ntask: regression\ncontext: {source: none}\n"
        "model:\n  kind: python\n  code: [predictor.py]\n"
        "  dependencies: []\n  artifacts: [probe.json]\n"
        "  inputs: [{name: x, kind: float64, nullable: false}]\n"
        "  outputs: [{name: value, kind: float64, nullable: false}]\n"
    )
    (tmp_path / "predictor.py").write_text(
        "raise AssertionError('contract validation must not execute this code')\n"
    )
    (tmp_path / "probe.json").write_text(
        '{"schema":[{"name":"x","kind":"float64","nullable":false}],"rows":[[1.0]]}'
    )
    compiled = Package.load(tmp_path).check().lock
    assert isinstance(compiled, PythonLock)
    return compiled


def lock(tmp_path):
    (tmp_path / "gf.yaml").write_text("name: fraud\nmodel: tabicl-classifier\n")
    compiled = Package.load(tmp_path).check().lock
    assert isinstance(compiled, PackageLock)
    return compiled


@pytest.mark.parametrize(
    "value",
    [
        Revision("fraud", 1, HASH, "catalog/tabicl-v2-classifier@" + HASH, 1),
        Environment("fraud", "production", 1, 0, "local"),
        PromoteRequest(1, 0),
        ContextVersion("dataset/events", "transaction-1", 12, 1),
        DatasetImportRef(
            "events",
            "s3://bucket/events.parquet",
            "parquet",
            "secret/warehouse",
            HASH,
            (("x", "value"),),
        ),
        RuntimePlan(
            "local",
            "production",
            HASH,
            HASH,
            None,
            "us-east-1",
            ("us-east-1",),
            2,
            "8Gi",
            1,
            2,
            ("secret/warehouse",),
        ),
        DevSession("dev-1", "fraud", 1, 0, HASH, 100),
    ],
)
def test_typed_values_roundtrip_exact_wire_and_are_immutable(value):
    serialized = json.loads(canonical(value.to_dict()))
    assert type(value).from_dict(serialized) == value
    with pytest.raises(PackageError):
        type(value).from_dict({**serialized, "unexpected_secret": "redacted"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.extra = True


def test_environment_promotion_and_rollback_are_cas_and_do_not_mutate_production():
    production = Environment("fraud", "production", 1, 0, "local", "ready")
    promoted = production.promote(PromoteRequest(2, 0))
    assert production.revision == 1 and production.generation == 0
    assert promoted.revision == 2 and promoted.generation == 1
    with pytest.raises(PackageError) as error:
        promoted.promote(PromoteRequest(3, 0))
    assert error.value.code == "Conflict"
    rollback = promoted.promote(PromoteRequest(1, 1))
    assert rollback.revision == 1 and rollback.generation == 2


def test_development_patch_cas_expiry_base_revision_and_deployment(tmp_path):
    compiled = lock(tmp_path)
    request = PatchRequest(0, 1, compiled)
    assert PatchRequest.from_dict(request.to_dict()) == request
    session = DevSession("dev-1", "fraud", 1, 0, HASH, 100, "ready")
    patched = session.apply_patch(request, now=10)
    assert patched.generation == 1 and patched.package_digest == compiled.package_digest
    assert session.generation == 0 and session.package_digest == HASH
    for stale, now in (
        (request, 10),
        (dataclasses.replace(request, expected_generation=1, base_revision=2), 10),
        (dataclasses.replace(request, expected_generation=1), 100),
    ):
        with pytest.raises(PackageError) as error:
            patched.apply_patch(stale, now=now)
        assert error.value.code == "Conflict"
    with pytest.raises(PackageError):
        dataclasses.replace(session, deployment="other").apply_patch(request, now=10)


def test_patch_snapshot_identity_must_match(tmp_path):
    (tmp_path / "gf.yaml").write_text(
        "name: fraud\nmodel: tabicl-classifier\ntarget: y\ncontext: history.csv\n"
    )
    (tmp_path / "history.csv").write_text("x,y\n1,true\n2,false\n")
    compiled = Package.load(tmp_path).check().lock
    assert isinstance(compiled, PackageLock)
    assert (
        PatchRequest(0, 1, compiled, compiled.data_digest).snapshot
        == compiled.data_digest
    )
    for snapshot in (None, HASH):
        with pytest.raises(PackageError):
            PatchRequest(0, 1, compiled, snapshot)


def test_runtime_plan_derivation_is_pure_and_preserves_placement(tmp_path):
    compiled = lock(tmp_path)
    plan = RuntimePlan.for_package(
        compiled,
        runtime="byoc",
        environment="production",
        secret_refs=("secret/warehouse",),
    )
    assert plan.package_digest == compiled.package_digest
    assert plan.model_digest == compiled.model_digest
    assert plan.cpu == compiled.cpu and plan.memory == compiled.memory
    assert plan.region is None and plan.residency == ()
    with pytest.raises(PackageError):
        dataclasses.replace(plan, region="eu-west-1", residency=("us-east-1",))
    with pytest.raises(PackageError):
        dataclasses.replace(plan, min_replicas=3, max_replicas=1)


def test_python_patch_roundtrip_requires_no_catalog_snapshot(python_lock):
    request = PatchRequest(0, 1, python_lock)
    assert PatchRequest.from_dict(request.to_dict()) == request
    assert request.snapshot is None
    session = DevSession("dev-1", "custom-score", 1, 0, HASH, 100, "ready")
    patched = session.apply_patch(request, now=10)
    assert patched.generation == 1
    assert patched.package_digest == python_lock.package_digest
    assert session.package_digest == HASH
    for snapshot in (HASH, "invalid"):
        with pytest.raises(PackageError):
            PatchRequest(0, 1, python_lock, snapshot)
    with pytest.raises(PackageError):
        PatchRequest.from_dict({**request.to_dict(), "snapshot": False})
    wire = request.to_dict()
    wire["lock"]["source_digest"] = HASH
    with pytest.raises(PackageError):
        PatchRequest.from_dict(wire)


def test_python_runtime_plan_has_no_fabricated_catalog_identity(python_lock):
    plan = RuntimePlan.for_package(
        python_lock, runtime="local", environment="production"
    )
    assert plan.package_digest == python_lock.package_digest
    assert plan.model_digest is None and plan.data_digest is None
    assert plan.region is None and plan.residency == ()
    assert (plan.cpu, plan.memory) == (1, "256Mi")
    assert plan.min_replicas == plan.max_replicas == 1
    assert RuntimePlan.from_dict(json.loads(canonical(plan.to_dict()))) == plan
    for change in (
        {"data_digest": HASH},
        {"region": "eastus"},
        {"residency": ("eastus",)},
        {"cpu": 2},
        {"memory": "1Gi"},
        {"min_replicas": 0},
        {"max_replicas": 2},
        {"secret_refs": ("secret/customer-key",)},
    ):
        with pytest.raises(PackageError):
            dataclasses.replace(plan, **change)


def test_python_lock_never_reaches_the_catalog_loader(python_lock, monkeypatch):
    from generalforecast.deployments.package_runtime.validation import admit
    from generalforecast.packages import CatalogResolver

    def forbidden(*args, **kwargs):
        raise AssertionError("Python admission must not resolve catalog models")

    monkeypatch.setattr(CatalogResolver, "resolve", forbidden)
    with pytest.raises(PackageError) as error:
        admit(python_lock)
    assert error.value.code == "UnsupportedCapability"


@pytest.mark.parametrize(
    "source",
    [
        "s3://user:password@bucket/table",
        "https://bucket/table?signature=redacted",
        "s3://bucket/table?signature=redacted",
        "s3://bucket/../private",
        "s3://bucket/%2e%2e/private",
        "s3://bucket/%40password",
        "file:///etc/passwd",
    ],
)
def test_dataset_import_rejects_inline_credentials_signed_urls_and_traversal(source):
    with pytest.raises(PackageError):
        DatasetImportRef("events", source, "parquet")


def test_prediction_nested_outputs_have_conversion_and_provenance():
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    table = pa.table(
        {
            "label": [False, True],
            "probabilities": [{"false": 0.7, "true": 0.3}, {"false": 0.1, "true": 0.9}],
        }
    )
    result = PredictionResult(
        table,
        "tabicl-v2-classifier",
        "classification",
        ContextVersion("dataset/events", 3, 42),
        "fraud",
        1,
        "production",
    )
    assert result.to_arrow().equals(table)
    assert result.to_pandas().shape == (2, 2)
    assert result.to_polars().shape == (2, 2)
    assert result.to_numpy().shape == (2, 2)
    wire = json.loads(canonical(result.to_dict()))
    parsed = PredictionResult.from_dict(wire)
    assert parsed.to_dict() == wire
    assert parsed.context is not None
    assert parsed.context.version == 3
    empty = PredictionResult(pa.table({}), "tabicl", "classification")
    assert empty.to_numpy().shape == (0, 0)


def test_untrusted_lock_capabilities_require_admission_even_with_valid_content_hash(
    tmp_path,
):
    compiled = lock(tmp_path)
    assert compiled.verify_model() == compiled.model
    forged = compiled.to_dict()
    forged["model"]["max_context_rows"] *= 10
    forged["model_digest"] = digest(ModelPin.from_dict(forged["model"]).identity())
    forged["package"] = digest(
        {key: value for key, value in forged.items() if key != "package"}
    )
    admitted = type(compiled).from_dict(forged)
    with pytest.raises(PackageError) as error:
        admitted.verify_model()
    assert error.value.code == "ModelMismatch"
