# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The x86-64 level a build of gf's native code needs, checked on import.

A build for a level above baseline x86-64, as the hosted plane makes with
`RUSTFLAGS=-C target-cpu=x86-64-v3`, may use the level's instructions
anywhere in gf-server, gf-router and the native module, before any check of
their own could run. `scripts/stage_servers.sh` records the level in the
package as `bin/cpu-level`, and importing `generalforecast`, which runs on
any x86-64 CPU, refuses a CPU without the level's features, naming them,
rather than letting a process die of an illegal instruction mid-request.
"""

import platform
from pathlib import Path

# Each level's features, as Linux's /proc/cpuinfo names them: `pni` is SSE3,
# `abm` holds LZCNT, and `lahf_lm` LAHF and SAHF in 64-bit mode.
_V2 = frozenset({"cx16", "lahf_lm", "popcnt", "pni", "sse4_1", "sse4_2", "ssse3"})
LEVELS: dict[str, frozenset[str]] = {
    "x86-64-v2": _V2,
    "x86-64-v3": _V2
    | {"abm", "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "xsave"},
}

MARKER = Path(__file__).with_name("bin") / "cpu-level"
CPUINFO = Path("/proc/cpuinfo")


def built_level(marker: Path = MARKER) -> str | None:
    """The level this install's native code was built for, or None for baseline."""
    try:
        level = marker.read_text().strip()
    except OSError:
        return None
    return level or None


def cpu_flags(cpuinfo: Path = CPUINFO) -> frozenset[str] | None:
    """The first processor's feature flags, or None where Linux does not list them."""
    try:
        text = cpuinfo.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name.strip() == "flags":
            return frozenset(value.split())
    return None


def missing_features(
    level: str | None, flags: frozenset[str] | None, machine: str
) -> list[str]:
    """What a CPU lacks of `level`: nothing for baseline or unlisted flags."""
    if level is None:
        return []
    if level not in LEVELS:
        return [f"{level}, a level this gf does not know"]
    if machine.lower() not in ("x86_64", "amd64"):
        return [f"an x86-64 CPU, not {machine}"]
    if flags is None:
        return []
    return sorted(LEVELS[level] - flags)


def check() -> None:
    """Refuses this CPU if it lacks the level the native code was built for.

    Raises:
        RuntimeError: Naming the level and the missing features.
    """
    level = built_level(MARKER)
    if level is None:
        return
    missing = missing_features(level, cpu_flags(CPUINFO), platform.machine())
    if missing:
        raise RuntimeError(
            f"this build of generalforecast is for {level}, and this CPU lacks"
            f" {', '.join(missing)}; install a build for baseline x86-64"
        )
