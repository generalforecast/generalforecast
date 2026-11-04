# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A build for an x86-64 level above baseline refuses, on import, a CPU without it."""

from pathlib import Path

import pytest

from generalforecast import _cpu

V3_FLAGS = _cpu.LEVELS["x86-64-v3"] | {"fpu", "sse", "sse2"}


def test_a_baseline_build_checks_nothing(tmp_path: Path) -> None:
    assert _cpu.built_level(tmp_path / "cpu-level") is None
    assert _cpu.missing_features(None, frozenset(), "x86_64") == []


def test_a_level_names_each_feature_the_cpu_lacks() -> None:
    assert _cpu.missing_features("x86-64-v3", V3_FLAGS, "x86_64") == []
    lacking = V3_FLAGS - {"avx2", "fma", "abm"}
    assert _cpu.missing_features("x86-64-v3", lacking, "x86_64") == [
        "abm",
        "avx2",
        "fma",
    ]
    assert _cpu.missing_features("x86-64-v2", lacking, "x86_64") == []
    assert _cpu.missing_features("x86-64-v3", V3_FLAGS, "aarch64") == [
        "an x86-64 CPU, not aarch64"
    ]
    assert _cpu.missing_features("x86-64-v9", V3_FLAGS, "x86_64") == [
        "x86-64-v9, a level this gf does not know"
    ]


def test_flags_come_from_the_first_processor(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nflags\t\t: fpu avx2 fma\n\nprocessor\t: 1\nflags\t\t: fpu\n"
    )
    assert _cpu.cpu_flags(cpuinfo) == {"fpu", "avx2", "fma"}
    assert _cpu.cpu_flags(tmp_path / "absent") is None


def test_a_cpu_without_the_level_is_refused_with_what_it_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "cpu-level"
    marker.write_text("x86-64-v3\n")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags\t: " + " ".join(sorted(V3_FLAGS - {"avx2"})) + "\n")
    monkeypatch.setattr(_cpu, "MARKER", marker)
    monkeypatch.setattr(_cpu, "CPUINFO", cpuinfo)
    monkeypatch.setattr(_cpu.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="is for x86-64-v3, and this CPU lacks avx2"):
        _cpu.check()
    cpuinfo.write_text("flags\t: " + " ".join(sorted(V3_FLAGS)) + "\n")
    _cpu.check()
