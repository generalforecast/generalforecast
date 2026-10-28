# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conventions every tracked file keeps, checked over the repository itself."""

import re
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


# A comment line that is only a rule (`# -----`, `// =====`, `/* *** */`), or a
# title between rules (`# -- name ----`, `# --- Name ---`): a banner.
_BANNER = re.compile(
    r"^\s*(?:#+|//+|--|/\*+|\*|;+|<!--)\s*"
    r"(?:(?:-{3,}|={3,}|#{3,}|\*{3,})\s*(?:\*/|-->)?"
    r"|(?:-{2,}|={2,}|#{2,}|\*{2,})\s*\S.*?\s*(?:-{2,}|={2,}|#{2,}|\*{2,})\s*(?:\*/|-->)?)\s*$"
)
# The characters the formatting rules keep out of every file: the em dash,
# and the center dot and bullet used as separators.
_FORBIDDEN = {
    chr(0x2014): "an em dash",
    chr(0xB7): "a center dot",
    chr(0x2022): "a bullet",
}


def formatting(text: str) -> list[str]:
    """Where `text` breaks the formatting rules: banner comments, em dashes, center dots."""
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        if _BANNER.match(line):
            problems.append(f"{number}: a banner comment")
        for character, name in _FORBIDDEN.items():
            if character in line:
                problems.append(f"{number}: {name}")
    return problems


def test_every_tracked_text_file_keeps_the_formatting_rules():
    broken = {}
    for path in tracked():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue
        if problems := formatting(text):
            broken[str(path.relative_to(ROOT))] = problems
    assert broken == {}


def test_the_formatting_rules_catch_what_they_name():
    assert formatting("# a plain comment\nx = 1  # trailing\n") == []
    assert formatting(
        "# -- facts ------\n"
        "# --- Worker ---\n"
        "// ===========\n"
        "  # ----------\n"
        "/* ***** */\n"
        f"# a pause {chr(0x2014)} then more\n"
        f"Searching {chr(0xB7)} 3s\n"
        f"{chr(0x2022)} item\n"
    ) == [
        "1: a banner comment",
        "2: a banner comment",
        "3: a banner comment",
        "4: a banner comment",
        "5: a banner comment",
        "6: an em dash",
        "7: a center dot",
        "8: a bullet",
    ]
