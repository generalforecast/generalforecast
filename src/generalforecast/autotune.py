# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Recorded autotune results, so every worker on a GPU type compiles to the same kernels.

XLA's GPU autotuner times candidate kernels for each fusion and matrix
multiply and keeps the fastest, and which is fastest varies from process to
process. So two fresh workers of one gf build compile the same program to
different kernels, and predict differently: on covertype past the cap with
TabICL v2 on an H100, by up to 0.08 in probability, and one row in 500
changed class. Programs a worker loads from the compile cache keep the
kernels they were compiled with, but a program compiled anew is tuned anew.

A worker given `GF_AUTOTUNE` loads the results recorded for its GPU and
jaxlib from that directory, when there are some, and compiles every program
they cover to the recorded kernels without timing any: workers that load one
file predict identically, bitwise, as fast as tuned ones, and compiling takes
no memory for candidates. With `GF_AUTOTUNE_RECORD=1` it also records
everything it tunes or loads, to a file of its own beside them, which
`keep_autotune` makes the kept results once the recording workers end, as a
pool's prebuild does.

This module does not import JAX, so the plane can keep results without it.
"""

import importlib.metadata
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

AUTOTUNE_ENV = "GF_AUTOTUNE"
RECORD_ENV = "GF_AUTOTUNE_RECORD"
SUFFIX = ".textproto"
RECORDING = ".recording-"


def gpu_name() -> str | None:
    """The first visible GPU's name as the driver gives it, or None without one.

    Read before JAX starts its backend, which takes XLA's flags only then.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    index = (visible or "0").split(",")[0].strip()
    if visible == "" or index in ("", "-1"):
        return None
    command = [
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
        f"--id={index}",
    ]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    name = done.stdout.strip().splitlines()[0].strip() if done.stdout.strip() else ""
    return name if done.returncode == 0 and name else None


def results_file(directory: Path, gpu: str, jaxlib: str | None = None) -> Path:
    """Where `directory` keeps the results recorded for `gpu` under this jaxlib, or `jaxlib`."""
    release = jaxlib or importlib.metadata.version("jaxlib")
    slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    return directory / f"{slug}-jaxlib-{release}{SUFFIX}"


def recording(kept: Path, pid: int) -> Path:
    """Where process `pid` records the results it tunes or loads for `kept`.

    It ends in `.textproto`, as XLA writes text only for that suffix.
    """
    return kept.with_name(f"{kept.name.removesuffix(SUFFIX)}{RECORDING}{pid}{SUFFIX}")


def autotune_flags(
    environ: Mapping[str, str] | None = None, gpu: str | None = None
) -> list[str]:
    """XLA's flags to load, and with `GF_AUTOTUNE_RECORD=1` record, this GPU's results; none without `GF_AUTOTUNE`.

    A run whose `XLA_FLAGS` already name autotune results keeps its own.
    """
    if environ is None:
        environ = os.environ
    root = environ.get(AUTOTUNE_ENV)
    if not root or "autotune_results" in environ.get("XLA_FLAGS", ""):
        return []
    gpu = gpu or gpu_name()
    if gpu is None:
        return []
    kept = results_file(Path(root), gpu)
    flags = []
    if kept.is_file():
        flags.append(f"--xla_gpu_load_autotune_results_from={kept}")
    if environ.get(RECORD_ENV) == "1":
        kept.parent.mkdir(parents=True, exist_ok=True)
        flags.append(
            f"--xla_gpu_dump_autotune_results_to={recording(kept, os.getpid())}"
        )
    return flags


def use_autotune_results() -> list[str]:
    """Adds `autotune_flags` to `XLA_FLAGS`; it takes effect only before JAX starts its first backend."""
    flags = autotune_flags()
    if flags:
        os.environ["XLA_FLAGS"] = " ".join(
            [os.environ.get("XLA_FLAGS", ""), *flags]
        ).strip()
    return flags


def _complete(path: Path) -> bool:
    """Whether a recording reads as whole text: its braces close and it ends a line."""
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return False
    return (
        bool(text.strip())
        and text.endswith("\n")
        and text.count("{") == text.count("}")
    )


def keep_autotune(directory: Path) -> list[Path]:
    """Makes each GPU type's largest whole recording in `directory` its kept results, and drops the rest.

    Call it once the recording workers have ended: a worker writes its
    recording again after each compile. Returns the files it kept.
    """
    recordings: dict[Path, list[Path]] = {}
    for path in directory.glob(f"*{RECORDING}*{SUFFIX}"):
        base, _, pid = path.name.removesuffix(SUFFIX).rpartition(RECORDING)
        if base and pid.isdigit():
            recordings.setdefault(path.with_name(base + SUFFIX), []).append(path)
    kept_files = []
    for kept, paths in sorted(recordings.items()):
        whole = [path for path in paths if _complete(path)]
        if whole:
            largest = max(whole, key=lambda path: path.stat().st_size)
            os.replace(largest, kept)
            kept_files.append(kept)
        for path in paths:
            path.unlink(missing_ok=True)
    return kept_files
