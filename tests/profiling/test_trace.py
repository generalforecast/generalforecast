# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pytest

from generalforecast.profiling import trace
from generalforecast.profiling.trace import (
    STEP_EVENT,
    STEPS_FILE,
    Trace,
    summarize,
    summarize_trace,
)

HOST, GPU = 1, 2
MAIN, COMPUTE, COPIES_IN, COPIES_OUT, BLAS = 100, 10, 11, 12, 13


def _meta(pid: int, name: str, tid: int | None = None) -> dict:
    kind = "process_name" if tid is None else "thread_name"
    event = {"ph": "M", "pid": pid, "name": kind, "args": {"name": name}}
    if tid is not None:
        event["tid"] = tid
    return event


def _event(pid: int, tid: int, name: str, ts: float, dur: float, **args) -> dict:
    return {
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": args,
    }


# A trace as XLA's CUPTI tracer writes one: the host's threads in one
# process and the GPU's streams in another, copies named by direction with
# their bytes in `memcpy_details`, kernels naming their program.
GPU_TRACE = [
    _meta(HOST, "/host:CPU"),
    _meta(GPU, "/device:GPU:0 (NVIDIA H100)"),
    _meta(GPU, "Stream #1(Compute)", COMPUTE),
    _meta(GPU, "Stream #2(MemcpyH2D)", COPIES_IN),
    _meta(GPU, "Stream #3(MemcpyD2H)", COPIES_OUT),
    _meta(GPU, "Stream #4(Compute)", BLAS),
    _event(HOST, MAIN, STEP_EVENT, 100, 1000, step_num="0"),
    _event(HOST, MAIN, STEP_EVENT, 1200, 300, step_num="1"),
    # The host's own copy call, which must not count as a device copy.
    _event(HOST, MAIN, "cuMemcpyHtoDAsync", 110, 5),
    _event(
        GPU,
        COPIES_IN,
        "MemcpyH2D",
        120,
        30,
        memcpy_details="kind_src:pinned kind_dst:device size:4096 dest:0 async:1",
    ),
    _event(GPU, COMPUTE, "fusion_1", 200, 100, hlo_module="jit_a", hlo_op="fusion.1"),
    _event(GPU, COMPUTE, "fusion_2", 350, 50, hlo_module="jit_a", hlo_op="fusion.2"),
    # Overlaps fusion_2 on another stream, so the device's busy time merges them.
    _event(GPU, BLAS, "gemm", 380, 100, hlo_module="jit_b", hlo_op="dot.3"),
    _event(
        GPU,
        COPIES_OUT,
        "MemcpyD2H",
        900,
        20,
        memcpy_details="kind_src:device kind_dst:pinned size:512 dest:1 async:1",
    ),
    # Outside every step: in the kernel and program tables only.
    _event(GPU, COMPUTE, "fusion_1", 2000, 100, hlo_module="jit_a", hlo_op="fusion.1"),
]

STEPS = [
    {
        "step": 0,
        "started": 10.0001,
        "ended": 10.0011,
        "seconds": 0.001,
        "stages": {},
        "parts": {
            "wait": {"calls": 1, "seconds": 0.0007},
            "pad": {"calls": 2, "seconds": 0.0001},
        },
    },
    {
        "step": 1,
        "started": 10.0012,
        "ended": 10.0015,
        "seconds": 0.0003,
        "stages": {},
        "parts": {"wait": {"calls": 1, "seconds": 0.0002}},
    },
]


def test_a_gpu_trace_summarises_per_step_kernel_and_program() -> None:
    parsed = Trace.parse(GPU_TRACE)
    assert parsed.processes[GPU].startswith("/device:")
    assert [event.name for event in parsed.device_events()] == [
        "MemcpyH2D",
        "fusion_1",
        "fusion_2",
        "gemm",
        "MemcpyD2H",
        "fusion_1",
    ]

    summary = summarize_trace(parsed, STEPS)
    first, second = summary.steps.to_pylist()
    us = 1e-6
    assert first["step"] == 0
    assert first["wall_s"] == pytest.approx(0.001)
    assert first["trace_s"] == pytest.approx(1000 * us)
    # [120, 150] + [200, 300] + [350, 480] + [900, 920]
    assert first["device_s"] == pytest.approx(280 * us)
    assert first["idle_s"] == pytest.approx(720 * us)
    assert first["lead_s"] == pytest.approx(20 * us)
    assert first["tail_s"] == pytest.approx(180 * us)
    assert first["kernels"] == 3
    assert (first["h2d"], first["h2d_bytes"], first["h2d_s"]) == (
        1,
        4096,
        pytest.approx(30 * us),
    )
    assert (first["d2h"], first["d2h_bytes"], first["d2h_s"]) == (
        1,
        512,
        pytest.approx(20 * us),
    )
    assert first["gap_s"] == pytest.approx(420 * us)
    assert first["gap_at_s"] == pytest.approx(380 * us)
    assert first["gap_after"] == "gemm"
    assert (first["wait_s"], first["pad_s"]) == (0.0007, 0.0001)

    # A step the device did nothing in.
    assert second["device_s"] == 0.0
    assert second["idle_s"] == pytest.approx(300 * us)
    assert second["lead_s"] == pytest.approx(300 * us)
    assert (second["kernels"], second["h2d"], second["d2h"], second["gap_s"]) == (
        0,
        0,
        0,
        0.0,
    )
    assert second["gap_after"] == ""
    assert (second["wait_s"], second["pad_s"]) == (0.0002, 0.0)

    kernels = {row["kernel"]: row for row in summary.kernels.to_pylist()}
    assert set(kernels) == {"fusion_1", "fusion_2", "gemm"}
    assert (kernels["fusion_1"]["count"], kernels["fusion_1"]["seconds"]) == (
        2,
        pytest.approx(200 * us),
    )
    assert kernels["fusion_1"]["mean_s"] == pytest.approx(100 * us)
    assert kernels["fusion_1"]["share"] == pytest.approx(200 / 350)
    assert summary.kernels.column("seconds").to_pylist() == sorted(
        summary.kernels.column("seconds").to_pylist(), reverse=True
    )

    programs = {row["program"]: row for row in summary.programs.to_pylist()}
    assert programs["jit_a"] == {
        "program": "jit_a",
        "runs": 3,
        "kernels": 3,
        "seconds": pytest.approx(250 * us),
        "share": pytest.approx(250 / 350),
    }
    assert programs["jit_b"]["kernels"] == 1


def test_a_cpu_trace_takes_the_hosts_thunks_as_its_kernels() -> None:
    # Without a device process, XLA's CPU thunks carry the program's name.
    cpu = [
        _meta(HOST, "/host:CPU"),
        _meta(HOST, "tf_XLAPjRtCpuClient/7", 7),
        _event(HOST, MAIN, STEP_EVENT, 0, 100, step_num="0"),
        _event(HOST, MAIN, "PjitFunction(f)", 5, 60),
        _event(
            HOST,
            7,
            "ynn_fusion",
            10,
            20,
            hlo_module="jit_f",
            hlo_op="ynn_fusion",
            run_id="-5",
        ),
        _event(
            HOST, 7, "copy.1", 30, 10, hlo_module="jit_f", hlo_op="copy.1", run_id="-5"
        ),
        _event(
            HOST,
            7,
            "ynn_fusion",
            60,
            20,
            hlo_module="jit_f",
            hlo_op="ynn_fusion",
            run_id="-4",
        ),
    ]
    steps = [{"step": 0, "started": 1.0, "ended": 1.0001, "parts": {}}]
    summary = summarize_trace(Trace.parse(cpu), steps)
    (row,) = summary.steps.to_pylist()
    assert (row["kernels"], row["h2d"], row["d2h"]) == (3, 0, 0)
    assert row["device_s"] == pytest.approx(50e-6)
    assert row["gap_s"] == pytest.approx(20e-6)
    assert row["gap_after"] == "copy.1"
    (program,) = summary.programs.to_pylist()
    assert (program["program"], program["runs"], program["kernels"]) == ("jit_f", 2, 3)


def test_the_command_reads_the_latest_run_and_writes_the_tables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "plugins" / "profile" / "2026_10_19_12_00_00"
    run.mkdir(parents=True)
    with gzip.open(run / "host.trace.json.gz", "wt") as file:
        json.dump({"displayTimeUnit": "ns", "traceEvents": GPU_TRACE}, file)
    (run / "perfetto_trace.json.gz").write_bytes(b"not a trace")
    (tmp_path / STEPS_FILE).write_text(json.dumps({"steps": STEPS}))

    assert trace.main([str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "gemm" in printed and "jit_a" in printed
    for name in ("steps", "kernels", "programs"):
        assert (tmp_path / f"{name}.parquet").exists()
    assert trace.main([str(tmp_path), "--csv"]) == 0
    assert "gemm" in (tmp_path / "kernels.csv").read_text()
    assert summarize(tmp_path).steps.num_rows == 2

    assert trace.main([str(tmp_path / "missing")]) == 1
