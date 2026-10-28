# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A wheel installs no program but `gf`: gf-server and gf-router live only under gf.

The release builds the wheel and runs this test on it with `GF_WHEEL`
naming it; without one, only the rule's own test runs.
"""

import os
import zipfile
from pathlib import Path

import pytest

SERVERS = ("gf-server", "gf-router")


def wheel_problems(names: list[str]) -> list[str]:
    """What breaks the rule in a wheel's file names (its RECORD)."""
    problems = []
    for name in names:
        parts = name.split("/")
        if len(parts) >= 3 and parts[0].endswith(".data") and parts[1] == "scripts":
            problems.append(f"{name} installs a program beside Python")
        if parts[-1] in SERVERS and parts[:-1] != ["generalforecast", "bin"]:
            problems.append(f"{name} is a server outside generalforecast/bin")
    for server in SERVERS:
        if f"generalforecast/bin/{server}" not in names:
            problems.append(f"the wheel carries no generalforecast/bin/{server}")
    return problems


def record(wheel: Path) -> list[str]:
    """The file names a wheel's RECORD lists."""
    with zipfile.ZipFile(wheel) as archive:
        listed = next(
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        )
        rows = archive.read(listed).decode().splitlines()
    return [row.split(",")[0] for row in rows if row]


def test_the_release_wheel_installs_only_gf():
    wheel = os.environ.get("GF_WHEEL")
    if not wheel:
        pytest.skip("set GF_WHEEL to a built wheel")
    assert wheel_problems(record(Path(wheel))) == []


def test_the_rule_catches_programs_outside_gf():
    good = [
        "generalforecast/__init__.py",
        "generalforecast/bin/gf-server",
        "generalforecast/bin/gf-router",
        "generalforecast-0.1.13.dist-info/entry_points.txt",
    ]
    assert wheel_problems(good) == []
    assert wheel_problems(
        [
            "generalforecast/bin/gf-server",
            "generalforecast-0.1.13.data/scripts/gf-native",
            "generalforecast-0.1.13.data/scripts/gf-router",
        ]
    ) == [
        "generalforecast-0.1.13.data/scripts/gf-native installs a program beside Python",
        "generalforecast-0.1.13.data/scripts/gf-router installs a program beside Python",
        "generalforecast-0.1.13.data/scripts/gf-router is a server outside generalforecast/bin",
        "the wheel carries no generalforecast/bin/gf-router",
    ]
