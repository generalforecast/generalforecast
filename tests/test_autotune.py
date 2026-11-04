# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Workers load, and with recording on also record, autotune results kept per GPU and jaxlib."""

import os
import subprocess
from pathlib import Path

import pytest

from generalforecast import autotune

GPU = "NVIDIA H100 80GB HBM3"


def test_results_are_kept_per_gpu_and_jaxlib(tmp_path: Path) -> None:
    kept = autotune.results_file(tmp_path, GPU, "0.12.0.1")
    assert kept == tmp_path / "nvidia-h100-80gb-hbm3-jaxlib-0.12.0.1.textproto"
    # XLA writes text only to a path that ends in `.textproto`.
    assert autotune.recording(kept, 42).name == (
        "nvidia-h100-80gb-hbm3-jaxlib-0.12.0.1.recording-42.textproto"
    )


def test_a_worker_loads_the_kept_results_and_records_beside_them(
    tmp_path: Path,
) -> None:
    kept = autotune.results_file(tmp_path, GPU)
    assert autotune.autotune_flags({}, GPU) == []
    assert autotune.autotune_flags({"GF_AUTOTUNE": str(tmp_path)}, GPU) == []
    recording = {"GF_AUTOTUNE": str(tmp_path), "GF_AUTOTUNE_RECORD": "1"}
    assert autotune.autotune_flags(recording, GPU) == [
        f"--xla_gpu_dump_autotune_results_to={autotune.recording(kept, os.getpid())}"
    ]
    kept.write_text("version: 3\n")
    assert autotune.autotune_flags({"GF_AUTOTUNE": str(tmp_path)}, GPU) == [
        f"--xla_gpu_load_autotune_results_from={kept}"
    ]
    assert autotune.autotune_flags(recording, GPU) == [
        f"--xla_gpu_load_autotune_results_from={kept}",
        f"--xla_gpu_dump_autotune_results_to={autotune.recording(kept, os.getpid())}",
    ]
    # A run that names results itself keeps them.
    own = {
        "GF_AUTOTUNE": str(tmp_path),
        "XLA_FLAGS": "--xla_gpu_load_autotune_results_from=x",
    }
    assert autotune.autotune_flags(own, GPU) == []


def test_keeping_takes_each_gpu_types_largest_whole_recording(tmp_path: Path) -> None:
    h100 = autotune.results_file(tmp_path, GPU, "1")
    l4 = autotune.results_file(tmp_path, "NVIDIA L4", "1")
    autotune.recording(h100, 11).write_text("results {\n  a: 1\n}\n")
    autotune.recording(h100, 12).write_text(
        "results {\n  a: 1\n}\nresults {\n  b: 2\n}\n"
    )
    # Cut off as its worker wrote it: not whole, though larger.
    autotune.recording(h100, 13).write_text(
        "results {\n  a: 1\n}\nresults {\n  b: 2\n}\nresults {\n  c"
    )
    autotune.recording(l4, 7).write_text("results {")
    (tmp_path / "unrelated.recording-x.textproto").write_text("x")
    assert autotune.keep_autotune(tmp_path) == [h100]
    assert h100.read_text().count("results") == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [h100.name, "unrelated.recording-x.textproto"]
    )


def test_without_a_visible_gpu_nothing_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert autotune.gpu_name() is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    asked = []

    def missing(command, **_):
        asked.append(command)
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert autotune.gpu_name() is None
    assert asked[0][-1] == "--id=1"
