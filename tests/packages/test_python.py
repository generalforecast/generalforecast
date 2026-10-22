# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import hashlib
import io
import json
import socket
import tarfile
import zipfile

import pyarrow as pa
import pytest
import yaml

from generalforecast.packages import (
    CatalogResolver,
    Package,
    PackageError,
    PackageLock,
    PackageSpec,
    PythonLock,
    PythonPackageSpec,
    parse_spec,
)
from generalforecast.packages.schema import TableSchema as CatalogSchema
from generalforecast.packages.validation import bytes_digest, canonical, digest
from generalforecast.python_packages.api import Field, Table, TableSchema
from generalforecast.python_packages.compiler import PythonRevision


@pytest.fixture
def authored():
    return {
        "package_version": 1,
        "name": "custom-score",
        "task": "regression",
        "model": {
            "kind": "python",
            "code": ["predictor.py"],
            "dependencies": [],
            "artifacts": ["weights.json", "probe.json"],
            "inputs": [{"name": "x", "kind": "float64", "nullable": False}],
            "outputs": [
                {"name": "value", "kind": "float64", "nullable": False},
                {"name": "ready", "kind": "boolean", "nullable": False},
            ],
        },
        "context": {"source": "none"},
        "compute": {"cpu": 1, "memory": "256Mi"},
        "scaling": {"min_replicas": 1, "max_replicas": 1},
    }


@pytest.fixture
def root(tmp_path, authored):
    (tmp_path / "gf.yaml").write_text(yaml.safe_dump(authored))
    (tmp_path / "predictor.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.executed').write_text('unexpected execution')\n"
        "import customer_dependency_that_must_never_be_imported\n"
        "raise RuntimeError('compilation must not execute this file')\n"
    )
    (tmp_path / "weights.json").write_text('{"bias":2.0}')
    (tmp_path / "probe.json").write_bytes(
        canonical({"schema": authored["model"]["inputs"], "rows": [[1.0], [2.0]]})
    )
    return tmp_path


def resign(wire, *, python=False):
    if python:
        manifest = wire["python"]["lock"]
        manifest["revision"] = digest(
            {k: v for k, v in manifest.items() if k != "revision"}
        )
    wire["package"] = digest({k: v for k, v in wire.items() if k != "package"})
    return wire


def replace_archive(wire, archive):
    wire = copy.deepcopy(wire)
    source = bytes_digest(archive)
    wire["python"]["source"] = source
    wire["python"]["lock"]["archive"] = source
    wire["source_digest"] = digest({"spec": wire["spec"], "archive": source})
    return resign(wire, python=True)


def checked_lock(root) -> PythonLock:
    lock = Package.load(root).check().lock
    assert isinstance(lock, PythonLock)
    return lock


def test_python_check_is_offline_pure_and_has_no_catalog_metadata(root, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Python compilation attempted catalog resolution or network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    monkeypatch.setattr(CatalogResolver, "resolve", forbidden)
    package = Package.load(root)
    checked = package.check()
    assert package.lock is None
    assert isinstance(checked.spec, PythonPackageSpec)
    assert isinstance(checked.lock, PythonLock)
    assert checked.spec.context.mode == "none"
    assert checked.lock.context_rows == 0 and checked.lock.data_digest is None
    assert checked.lock.kind == "python"
    assert checked.lock.outputs == ("value", "ready")
    assert checked.lock.identity.startswith("python@sha256:")
    assert not hasattr(checked.lock, "model")
    assert not hasattr(checked.lock, "model_digest")
    assert not hasattr(checked.spec, "target")
    assert not (root / "predictor.executed").exists()
    assert not (root / "gf-python.lock").exists()
    assert not (root / "gf-python-source.tar").exists()
    assert Package.load(root).compile() == checked.lock
    assert checked.check(offline=False, resolver=CatalogResolver()).lock == checked.lock
    with pytest.raises(PackageError, match="isolated publishing"):
        checked.resolve()
    with pytest.raises(dataclasses.FrozenInstanceError):
        checked.spec.__setattr__("name", "changed")
    wire = checked.lock.to_dict()
    wire["python"]["lock"]["spec"]["inputs"][0]["name"] = "changed"
    assert checked.lock.schema.inputs.fields[0].name == "x"


@pytest.mark.parametrize("task", ["regression", "classification", "forecast"])
def test_exact_v2_wire_roundtrip_and_verified_archive(root, authored, task):
    authored["task"] = task
    (root / "gf.yaml").write_text(yaml.safe_dump(authored))
    checked = Package.load(root).check()
    lock = checked.lock
    assert isinstance(lock, PythonLock)
    archive = checked.source_archive()
    revision = lock.verify_source(archive)
    assert isinstance(revision, PythonRevision)
    assert revision.spec.task == task
    assert lock.identity == "python@" + revision.digest
    wire = lock.to_dict()
    assert set(wire) == {
        "lock_version",
        "compiler_version",
        "kind",
        "spec",
        "python",
        "source_digest",
        "data_digest",
        "task",
        "schema",
        "outputs",
        "context_rows",
        "compute",
        "package",
    }
    assert wire["lock_version"] == 2 and wire["compiler_version"] == "python-1"
    assert wire["source_digest"] == digest(
        {"spec": wire["spec"], "archive": wire["python"]["source"]}
    )
    assert wire["package"] == digest({k: v for k, v in wire.items() if k != "package"})
    assert wire["python"]["source"] == bytes_digest(archive)
    assert set(wire["python"]) == {"kind", "lock", "source"}
    assert wire["schema"] == {
        "inputs": authored["model"]["inputs"],
        "outputs": authored["model"]["outputs"],
    }
    assert "task" not in wire["spec"]["model"]
    assert "version" not in wire["spec"]["model"]
    assert wire["spec"]["model"]["artifacts"] == ["probe.json", "weights.json"]
    assert PythonLock.from_dict(wire) == lock
    assert PackageLock.from_dict(wire) == lock
    assert PackageLock.load(root) == lock
    assert parse_spec(wire["spec"]) == checked.spec
    with pytest.raises(PackageError):
        lock.verify_source(archive + b"tampered")


def test_catalog_v1_serialization_is_stable():
    spec = PackageSpec.from_dict({"name": "v1-fixture", "model": "tabicl-classifier"})
    pin = CatalogResolver().resolve(spec.model, offline=True)
    lock = PackageLock.build(
        spec=spec,
        model=pin,
        task="classification",
        schema=CatalogSchema(),
        source_digest="sha256:" + "1" * 64,
        data_digest=None,
        outputs=("label",),
        context_rows=None,
    )
    assert (
        hashlib.sha256(canonical(lock.to_dict())).hexdigest()
        == "3b363c2990bad0d9ad8c64cd56cf9b3842a4790c7c3218645c3e318c97228869"
    )
    assert lock.kind == "catalog" and lock.identity == lock.model.resolved
    assert "kind" not in lock.to_dict()
    assert isinstance(PackageLock.from_dict(lock.to_dict()), PackageLock)
    assert parse_spec(spec.to_dict()) == spec


@pytest.mark.parametrize("mode", ["locked", "frozen"])
def test_python_read_only_modes_and_immutable_artifact_revisions(root, mode):
    with pytest.raises(PackageError) as error:
        Package.load(root).check(locked=mode == "locked", frozen=mode == "frozen")
    assert error.value.code == "LockRequired"
    checked = Package.load(root).check()
    assert isinstance(checked.lock, PythonLock)
    before = (root / "gf.lock").read_bytes()
    mtime = (root / "gf.lock").stat().st_mtime_ns
    assert (
        checked.check(locked=mode == "locked", frozen=mode == "frozen").lock
        == checked.lock
    )
    assert (root / "gf.lock").stat().st_mtime_ns == mtime
    (root / "weights.json").write_text('{"bias":4.0}')
    with pytest.raises(PackageError) as error:
        Package.load(root).check(locked=mode == "locked", frozen=mode == "frozen")
    assert error.value.code == "LockOutdated"
    with pytest.raises(PackageError):
        checked.source_archive()
    with pytest.raises(PackageError):
        checked.compile()
    assert (root / "gf.lock").read_bytes() == before
    changed = Package.load(root).check(write=False)
    assert isinstance(changed.lock, PythonLock)
    assert changed.lock != checked.lock
    assert (root / "gf.lock").read_bytes() == before
    assert checked.lock.to_dict()["python"] != changed.lock.to_dict()["python"]
    assert Package.load(root).compile(frozen=False) == changed.lock


def test_python_source_changes_and_schema_upgrades_are_explicit(root, authored):
    checked = Package.load(root).check()
    assert isinstance(checked.lock, PythonLock)
    (root / "predictor.py").write_text("raise RuntimeError('new source')\n")
    changed = Package.load(root).check()
    assert isinstance(changed.lock, PythonLock)
    assert changed.lock.package_digest != checked.lock.package_digest
    before = (root / "gf.lock").read_bytes()
    authored["model"]["inputs"][0]["kind"] = "int64"
    (root / "gf.yaml").write_text(yaml.safe_dump(authored))
    (root / "probe.json").write_bytes(
        canonical({"schema": authored["model"]["inputs"], "rows": [[1]]})
    )
    with pytest.raises(PackageError) as error:
        changed.check()
    assert error.value.code == "SourceChanged"
    with pytest.raises(PackageError) as error:
        Package.load(root).check()
    assert error.value.code == "SchemaChanged"
    assert (root / "gf.lock").read_bytes() == before
    upgraded = Package.load(root).check(upgrade=True)
    assert isinstance(upgraded.lock, PythonLock)
    assert upgraded.lock.schema.inputs.fields[0].kind == "int64"
    assert checked.lock.schema.inputs.fields[0].kind == "float64"
    with pytest.raises(PackageError) as error:
        Package.load(root).check(frozen=True, upgrade=True)
    assert error.value.code == "InvalidPackage"


def test_python_checks_detect_concurrent_source_changes(root, monkeypatch):
    from generalforecast.packages import compiler

    original = compiler.compile_python
    calls = 0

    def changing(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "weights.json").write_text('{"bias":3.0}')
        return original(*args)

    monkeypatch.setattr(compiler, "compile_python", changing)
    with pytest.raises(PackageError) as error:
        Package.load(root).check()
    assert error.value.code == "SourceChanged"
    assert not (root / "gf.lock").exists()


@pytest.mark.parametrize(
    "data",
    [
        None,
        "request",
        {"dataset": "events"},
        {"source": "weights.json"},
        {"source": "none", "selection": "all"},
    ],
)
def test_python_requires_distinct_no_context(authored, data):
    authored["context"] = data
    with pytest.raises(PackageError):
        parse_spec(authored)


@pytest.mark.parametrize(
    "key",
    [
        "target",
        "id",
        "inputs",
        "outputs",
        "time",
        "series",
        "horizon",
        "frequency",
        "past_inputs",
        "future_inputs",
        "quantiles",
        "placement",
        "secret",
    ],
)
def test_python_rejects_catalog_and_unknown_fields(authored, key):
    authored[key] = "unused"
    with pytest.raises(PackageError):
        parse_spec(authored)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("compute", "cpu", 2),
        ("compute", "cpu", True),
        ("compute", "cpu", 1.0),
        ("compute", "memory", "1Gi"),
        ("scaling", "min_replicas", 0),
        ("scaling", "max_replicas", 2),
        ("scaling", "max_replicas", True),
        ("model", "kind", "catalog"),
        ("model", "task", "regression"),
        ("model", "version", "gf.python/v1"),
        ("model", "target", "value"),
        ("model", "code", []),
        ("model", "code", ["predictor.py", "predictor.py"]),
        ("model", "code", ["../predictor.py"]),
        ("model", "code", ["predictor.py", "*.py"]),
        ("model", "artifacts", ["weights.json"]),
        ("model", "artifacts", ["predictor.py", "probe.json"]),
        ("model", "dependencies", ["native-cp312-linux.whl"]),
        ("model", "dependencies", ["https://example.invalid/package-py3-none-any.whl"]),
        ("model", "artifacts", [".env", "probe.json"]),
    ],
)
def test_python_strict_intent_and_fixed_worker_policy(authored, section, key, value):
    authored[section][key] = value
    with pytest.raises(PackageError):
        parse_spec(authored)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(rows=[]),
        lambda p: p.update(rows=[[1.0]] * 101),
        lambda p: p.update(rows=[[None]]),
        lambda p: p.update(rows=[[1]]),
        lambda p: p.update(rows=[[float("nan")]]),
        lambda p: p.update(rows=[[float("inf")]]),
        lambda p: p.update(rows=[[1.0, 2.0]]),
        lambda p: p.update(extra=True),
        lambda p: p["schema"][0].update(kind="int64"),
        lambda p: p["schema"][0].update(nullable=True),
        lambda p: p["schema"][0].update(name="wrong"),
    ],
)
def test_python_probe_is_required_typed_bounded_data_only(root, authored, change):
    probe = {"schema": authored["model"]["inputs"], "rows": [[1.0]]}
    change(probe)
    (root / "probe.json").write_text(json.dumps(probe))
    with pytest.raises(PackageError):
        Package.load(root).check()
    assert not (root / "gf.lock").exists()
    assert not (root / "predictor.executed").exists()


@pytest.mark.parametrize(
    "raw",
    [b"not JSON", b'{"schema":[],"rows":[],"rows":[[1.0]]}', b"\xff", b"[" * 1500],
)
def test_python_invalid_probe_diagnostics_are_safe(root, raw):
    (root / "probe.json").write_bytes(raw)
    with pytest.raises(PackageError):
        Package.load(root).check()


def test_python_rejects_symlinks_and_embedded_private_key_artifacts(root):
    (root / "weights.json").unlink()
    (root / "weights.json").symlink_to(root / "probe.json")
    with pytest.raises(PackageError) as error:
        Package.load(root).check()
    assert error.value.code == "UnsafePath"
    (root / "weights.json").unlink()
    (root / "weights.json").write_text(
        "-----BEGIN PRIVATE KEY-----\ninvalid-test-material\n"
    )
    with pytest.raises(PackageError, match="credential"):
        Package.load(root).check()


@pytest.mark.parametrize(
    "hook",
    [
        "ok.py",
        "startup.pth",
        "extension.so",
        "generalforecast/native/__init__.py",
        "sitecustomize.py",
    ],
)
def test_python_offline_wheel_validation_never_installs(root, authored, hook):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as wheel:
        wheel.writestr(hook, "raise RuntimeError('wheel must never execute')")
        wheel.writestr(
            "small-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    filename = "small-1.0-py3-none-any.whl"
    (root / filename).write_bytes(stream.getvalue())
    authored["model"]["dependencies"] = [filename]
    (root / "gf.yaml").write_text(yaml.safe_dump(authored))
    if hook == "ok.py":
        assert checked_lock(root).python
    else:
        with pytest.raises(PackageError):
            Package.load(root).check()


@pytest.mark.parametrize(
    "path,value",
    [
        (("task",), "classification"),
        (("source_digest",), "sha256:" + "1" * 64),
        (("data_digest",), "sha256:" + "1" * 64),
        (("context_rows",), 1),
        (("context_rows",), False),
        (("outputs",), ["ready", "value"]),
        (("compute", "cpu"), True),
        (("schema", "inputs", 0, "kind"), "int64"),
        (("schema", "outputs", 0, "nullable"), 0),
        (("python", "source"), "sha256:" + "1" * 64),
        (("python", "lock", "dependency_digest"), "sha256:" + "1" * 64),
        (("python", "lock", "files", 0, "size"), True),
        (("python", "lock", "files", 0, "path"), "other.py"),
        (("python", "lock", "files", 0, "role"), "artifact"),
        (("python", "lock", "spec", "task"), "classification"),
    ],
)
def test_python_lock_rejects_rehashed_declaration_mismatches(root, path, value):
    wire = checked_lock(root).to_dict()
    parent = wire
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    resign(wire, python=True)
    with pytest.raises(PackageError):
        PackageLock.from_dict(wire)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lock_version", 2.0),
        ("lock_version", 3),
        ("kind", "unknown"),
        ("compiler_version", "1"),
        ("model_digest", "sha256:" + "1" * 64),
    ],
)
def test_python_lock_strict_discriminator_and_unknown_fields(root, key, value):
    wire = checked_lock(root).to_dict()
    wire[key] = value
    resign(wire)
    with pytest.raises(PackageError):
        PackageLock.from_dict(wire)


def test_python_lock_metadata_is_not_archive_admission(root):
    checked = Package.load(root).check()
    assert isinstance(checked.lock, PythonLock)
    archive = checked.source_archive()
    wire = checked.lock.to_dict()
    wire["python"]["lock"]["files"][0]["digest"] = "sha256:" + "1" * 64
    lock = PackageLock.from_dict(resign(wire, python=True))
    assert isinstance(lock, PythonLock)
    with pytest.raises(PackageError, match="manifest mismatch"):
        lock.verify_source(archive)


@pytest.mark.parametrize(
    "path,kind",
    [
        ("../predictor.py", "file"),
        ("predictor.py", "duplicate"),
        ("predictor.py", "symlink"),
    ],
)
def test_python_archive_admission_rejects_unsafe_members_even_when_rehashed(
    root, path, kind
):
    checked = Package.load(root).check()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo(path)
        if kind == "symlink":
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(info, io.BytesIO(b""))
        if kind == "duplicate":
            tar.addfile(info, io.BytesIO(b""))
    archive = stream.getvalue()
    assert isinstance(checked.lock, PythonLock)
    lock = PackageLock.from_dict(replace_archive(checked.lock.to_dict(), archive))
    assert isinstance(lock, PythonLock)
    with pytest.raises(PackageError):
        lock.verify_source(archive)


def test_python_query_conversion_is_exact_and_has_no_training_context(root):
    checked = Package.load(root).check()
    query = pa.table({"x": [1.0, 2.0]})
    result = checked.validate_queries(query)
    assert result.to_pydict() == query.to_pydict()
    assert result.schema.field("x").nullable is False
    table = Table(TableSchema((Field("x", "float64"),)), ((3.0,),))
    assert checked.validate_queries(table).to_pydict() == {"x": [3.0]}
    for data in (
        {"x": [1]},
        {"x": [None]},
        {"wrong": [1.0]},
        {"x": [1.0], "extra": [2.0]},
        {"x": [float("nan")]},
        {"x": [float("inf")]},
        {"x": [1.0] * 4097},
    ):
        with pytest.raises(PackageError):
            checked.validate_queries(data)
    with pytest.raises(PackageError):
        checked.check(data=query)
    with pytest.raises(PackageError):
        checked.snapshot_table()
    with pytest.raises(PackageError):
        checked.context_table()
    with pytest.raises(PackageError):
        Package(root, checked.spec).source_archive()


def test_python_optional_dataframe_adapters_use_exact_scalar_schema(root):
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    checked = Package.load(root).check()
    for query in (
        pd.DataFrame({"x": [1.0, 2.0]}),
        pl.DataFrame({"x": [1.0, 2.0]}),
        pl.DataFrame({"x": [1.0, 2.0]}).lazy(),
    ):
        assert checked.validate_queries(query).to_pydict() == {"x": [1.0, 2.0]}
