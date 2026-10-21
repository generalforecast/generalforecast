# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
import zipfile
from dataclasses import replace

import pytest

from generalforecast.packages.validation import PackageError, canonical, digest
from generalforecast.python_packages.api import Field, Table, TableSchema
from generalforecast.python_packages.compiler import (
    PythonRevision,
    PythonSpec,
    compile_python,
    patch_kind,
    unpack,
    wheel_files,
)
from tests.product_python.conftest import INPUT, wheel


def test_deterministic_explicit_archive(package_root, spec, revision):
    (package_root / ".env").write_text("DO_NOT_CAPTURE=private")
    (package_root / "unrelated.py").write_text("raise Exception('must not import')")
    (package_root / "predictor.py").chmod(0o700)
    repeated = compile_python(package_root, spec)
    assert revision == repeated
    assert set(unpack(revision.archive)) == {
        "predictor.py",
        "bias.json",
        "small_math-1.0-py3-none-any.whl",
    }
    revision.write(package_root)
    assert PythonRevision.load(package_root) == revision
    assert revision.inspect()["files"][0]["role"] == "code"


def test_compilation_does_not_execute(package_root, spec):
    marker = package_root / "executed"
    (package_root / "predictor.py").write_text(
        f"open({str(marker)!r}, 'w').write('executed')\nclass Model: pass"
    )
    compile_python(package_root, spec)
    assert not marker.exists()


@pytest.mark.parametrize(
    "path",
    [
        "../predictor.py",
        "/etc/passwd",
        "a/../predictor.py",
        ".env",
        "secrets/key",
        "a\\b.py",
        "a//b.py",
    ],
)
def test_unsafe_allowlists(spec, path):
    with pytest.raises(PackageError):
        PythonSpec.from_dict({**spec.to_dict(), "artifacts": [path]})


def test_symlink_file_and_parent_rejected(package_root, spec):
    outside = package_root.parent / (package_root.name + "-outside.py")
    outside.write_text("class Model: pass")
    (package_root / "predictor.py").unlink()
    (package_root / "predictor.py").symlink_to(outside)
    with pytest.raises(PackageError, match="unsafe"):
        compile_python(package_root, spec)
    alias = package_root.parent / (package_root.name + "-alias")
    alias.symlink_to(package_root, target_is_directory=True)
    with pytest.raises(PackageError):
        compile_python(alias, spec)


def test_hashes_cover_source_dependencies_artifacts(package_root, spec, revision):
    (package_root / "bias.json").write_text('{"bias":3.0}')
    artifact = compile_python(package_root, spec)
    assert artifact.digest != revision.digest
    assert patch_kind(revision, artifact) == "hot-swap"
    (package_root / "predictor.py").write_text("class Model: pass\n")
    code = compile_python(package_root, spec)
    assert code.digest != artifact.digest
    (package_root / "small_math-1.0-py3-none-any.whl").write_bytes(wheel(1.0))
    dependency = compile_python(package_root, spec)
    assert dependency.digest != code.digest
    assert patch_kind(code, dependency) == "rebuild"


def test_tampered_archive_and_lock(revision):
    with pytest.raises(PackageError):
        replace(revision, archive=revision.archive[:-1] + b"x").inspect()
    manifest = json.loads(revision.lock)
    manifest["spec"]["outputs"][0]["kind"] = "string"
    with pytest.raises(PackageError, match="digest mismatch"):
        replace(revision, lock=canonical(manifest)).inspect()
    manifest["revision"] = digest(
        {k: v for k, v in manifest.items() if k != "revision"}
    )
    assert (
        replace(revision, lock=canonical(manifest)).inspect()["spec"]["outputs"][0][
            "kind"
        ]
        == "string"
    )


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.REGTYPE]
)
def test_malicious_tar(kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("../escape")
        member.type = kind
        archive.addfile(member)
    with pytest.raises(PackageError):
        unpack(stream.getvalue())


@pytest.mark.parametrize(
    "path", ["../escape", "evil.pth", "generalforecast/__init__.py", "sitecustomize.py"]
)
def test_wheel_escape_and_hooks(path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(path, "pass")
    with pytest.raises(PackageError):
        wheel_files(stream.getvalue())


def test_unknown_keys_credentials_syntax(package_root, spec):
    with pytest.raises(PackageError):
        PythonSpec.from_dict({**spec.to_dict(), "api_key": "forbidden"})
    (package_root / "predictor.py").write_text("class Model(:")
    with pytest.raises(PackageError, match="syntax"):
        compile_python(package_root, spec)
    (package_root / "predictor.py").write_text("# -----BEGIN PRIVATE KEY-----")
    with pytest.raises(PackageError, match="credential"):
        compile_python(package_root, spec)


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("small_math/keys.py", "KEY = 'AKIA" + "A" * 16 + "'\n" + "pass\n" * 64),
        ("small_math/credentials/token.py", "TOKEN = None\n"),
    ],
    ids=["payload", "path"],
)
def test_compressed_wheel_credentials_rejected(package_root, spec, path, payload):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(path, payload)
        archive.writestr(
            "small_math-1.0.dist-info/WHEEL",
            "Root-Is-Purelib: true\nTag: py3-none-any\n",
        )
    raw = stream.getvalue()
    assert b"AKIA" not in raw
    (package_root / "small_math-1.0-py3-none-any.whl").write_bytes(raw)
    with pytest.raises(PackageError, match="credential"):
        compile_python(package_root, spec)


@pytest.mark.parametrize("value", [True, 1, float("inf"), float("nan"), None, "1"])
def test_typed_input_values(value):
    with pytest.raises(ValueError):
        Table(INPUT, ((value,),))


def test_typed_tables():
    schema = TableSchema((Field("id", "int64"), Field("optional", "string", True)))
    table = Table(schema, ((1, None),))
    assert Table.from_dict(table.to_dict()) == table
    with pytest.raises(ValueError):
        Table(schema, ((2**64, None),))
    with pytest.raises(ValueError):
        Table(schema, ((True, None),))
    with pytest.raises(ValueError):
        TableSchema((Field("x", "string"), Field("x", "string")))


def test_aggregate_table_bytes_bound():
    schema = TableSchema((Field("x", "string"),))
    with pytest.raises(ValueError, match="aggregate byte"):
        Table(schema, (("x" * 32000,), ("y" * 32000,)))
    with pytest.raises(ValueError, match="row bound"):
        Table(schema, (("x",),) * 4097)


@pytest.mark.parametrize("bound", ["bytes", "files"])
def test_expanded_dependencies_have_an_aggregate_bound(
    package_root, spec, monkeypatch, bound
):
    from generalforecast.python_packages import compiler

    if bound == "bytes":
        monkeypatch.setattr(compiler, "MAX_ARCHIVE", 16000)
    dependencies = []
    for index in range(2):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_index in range(1 if bound == "bytes" else 2048):
                archive.writestr(
                    f"library{index}/file{file_index}.py",
                    "#" * (9000 if bound == "bytes" else 1),
                )
            archive.writestr(
                f"library{index}.dist-info/WHEEL",
                "Root-Is-Purelib: true\nTag: py3-none-any\n",
            )
        path = f"library{index}-1.0-py3-none-any.whl"
        (package_root / path).write_bytes(stream.getvalue())
        dependencies.append(path)
    message = "expanded Python runtime" if bound == "bytes" else "too many files"
    with pytest.raises(PackageError, match=message):
        compile_python(package_root, replace(spec, dependencies=tuple(dependencies)))


def test_dependency_file_directory_conflicts(package_root, spec):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("library.py", "pass")
        archive.writestr("library.py/child.py", "pass")
        archive.writestr(
            "library.dist-info/WHEEL", "Root-Is-Purelib: true\nTag: py3-none-any\n"
        )
    (package_root / spec.dependencies[0]).write_bytes(stream.getvalue())
    with pytest.raises(PackageError, match="conflicts with a directory"):
        compile_python(package_root, spec)
