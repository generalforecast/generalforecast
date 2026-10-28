# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conventions every tracked file keeps, checked over the repository itself."""

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracked(*patterns: str) -> list[Path]:
    """The files git tracks that match `patterns`."""
    names = subprocess.run(
        ["git", "ls-files", "-z", "--", *patterns],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    return [ROOT / name.decode() for name in names if name]


def versions(workspace: dict, project: dict, crates: dict[str, dict]) -> list[str]:
    """What breaks the one-version rule: the workspace's version is the package's and every crate's."""
    problems = []
    version = workspace.get("workspace", {}).get("package", {}).get("version")
    if version is None:
        problems.append("Cargo.toml has no workspace.package version")
    if project.get("project", {}).get("version") != version:
        problems.append(
            f"pyproject.toml's version {project.get('project', {}).get('version')} "
            f"is not the workspace's {version}"
        )
    for name, crate in crates.items():
        if crate.get("package", {}).get("version") != {"workspace": True}:
            problems.append(f"{name} does not take version.workspace")
    return problems


def test_one_version_for_the_package_and_every_crate():
    def read(path: Path) -> dict:
        return tomllib.loads(path.read_text())

    crates = {
        str(path.relative_to(ROOT)): read(path)
        for path in tracked("crates/*/Cargo.toml")
    }
    assert crates
    assert (
        versions(read(ROOT / "Cargo.toml"), read(ROOT / "pyproject.toml"), crates) == []
    )
    # A deliberately broken tree fails the check.
    broken = versions(
        {"workspace": {"package": {"version": "0.1.13"}}},
        {"project": {"version": "0.1.12"}},
        {"crates/x/Cargo.toml": {"package": {"version": "0.1.13"}}},
    )
    assert broken == [
        "pyproject.toml's version 0.1.12 is not the workspace's 0.1.13",
        "crates/x/Cargo.toml does not take version.workspace",
    ]
