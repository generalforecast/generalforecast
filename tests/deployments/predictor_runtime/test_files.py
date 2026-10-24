# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import stat

import pytest

from generalforecast.deployments.predictor_runtime.compile import (
    compile_dir,
    plan_from_dir,
)
from generalforecast.predictors.files import atomic_write, parse_yaml, read, relative
from generalforecast.predictors.validation import PredictorError


@pytest.mark.parametrize(
    "path",
    [
        "../x",
        "/tmp/x",
        "data/../x",
        "data//x",
        ".env",
        ".aws/credentials",
        "secrets/key.csv",
        "C:\\data.csv",
        "data\\file.csv",
        "s3://bucket/file",
        "data/${TOKEN}.csv",
    ],
)
def test_unsafe_and_credential_paths_fail(path):
    with pytest.raises(PredictorError):
        relative(path)


def test_symlinks_directories_manifest_and_data_are_never_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "history.csv").write_text("x,y\n1,true\n2,false\n")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    (tmp_path / "gf.yaml").symlink_to(real / "history.csv")
    for path in ("linked/history.csv", "gf.yaml"):
        with pytest.raises(PredictorError) as error:
            read(tmp_path, path)
        assert error.value.code == "UnsafePath"


def test_package_root_can_have_a_symlinked_parent(tmp_path):
    actual = tmp_path / "actual"
    predictor = actual / "predictor"
    predictor.mkdir(parents=True)
    (predictor / "gf.yaml").write_text(
        "name: linked-root\nmodel: general-forecast/tabicl-v2-classifier\ntarget: y\n"
        "context: request\n"
    )
    (tmp_path / "linked").symlink_to(actual, target_is_directory=True)

    loaded, _ = plan_from_dir(tmp_path / "linked" / "predictor")

    assert loaded.name == "linked-root"


def test_lock_atomicity_private_mode_and_failed_replace_cleanup(tmp_path, monkeypatch):
    atomic_write(tmp_path, "gf.lock", b"old")
    assert stat.S_IMODE((tmp_path / "gf.lock").stat().st_mode) == 0o600

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        atomic_write(tmp_path, "gf.lock", b"new")
    assert (tmp_path / "gf.lock").read_bytes() == b"old"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["gf.lock"]


def test_atomic_write_refuses_symlink_destination(tmp_path):
    victim = tmp_path / "victim"
    victim.write_bytes(b"private")
    (tmp_path / "gf.lock").symlink_to(victim)
    with pytest.raises(PredictorError):
        atomic_write(tmp_path, "gf.lock", b"new")
    assert victim.read_bytes() == b"private"


@pytest.mark.parametrize(
    "data",
    [
        b"name: first\nname: second",
        b"x: &a [1]\ny: *a",
        b"x: !!python/object:object {}",
        b"[not, an, object]",
    ],
)
def test_strict_yaml(data):
    with pytest.raises(PredictorError):
        parse_yaml(data)


def test_compiler_never_recursively_captures_files_or_dotenv(tmp_path):
    (tmp_path / "gf.yaml").write_text(
        "name: test\nmodel: general-forecast/tabicl-v2-classifier\n"
    )
    (tmp_path / ".env").write_text("PRIVATE=not-a-token\n")
    (tmp_path / "unrelated.py").write_text("raise RuntimeError('must not execute')")
    first = compile_dir(tmp_path)
    (tmp_path / ".env").write_text("PRIVATE=changed\n")
    (tmp_path / "unrelated.py").write_text("raise RuntimeError('changed')")
    second = compile_dir(tmp_path)
    assert first == second
    assert "PRIVATE" not in str(first.to_dict())


def test_fifo_and_large_source_are_refused_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "fifo.csv")
    with pytest.raises(PredictorError):
        read(tmp_path, "fifo.csv")
    (tmp_path / "large").write_bytes(b"12345")
    with pytest.raises(PredictorError):
        read(tmp_path, "large", limit=4)
