# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conventions every tracked file keeps, checked over the repository itself."""

import ast
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


def test_a_test_that_starts_a_server_without_its_marker_fails(monkeypatch):
    import pytest

    from tests.utils import processes

    # This test carries no `server` marker, so starting anything is refused
    # before a process exists.
    assert processes.MARKED is False
    with pytest.raises(processes.UnmarkedServer, match="mark it"):
        processes.start(["true"])
    monkeypatch.setattr(processes, "MARKED", True)
    process = processes.start(["true"])
    processes.stop(process)


# Errors whose message is their second argument, after the code.
_CODED = {
    "ServiceError",
    "InputError",
    "PredictorError",
    "RelayedError",
    "FrequencyError",
}
# Errors whose message is their first argument.
_PLAIN = {
    "Error",
    "ValueError",
    "TypeError",
    "KeyError",
    "RuntimeError",
    "OSError",
    "LookupError",
    "PermissionError",
    "NotImplementedError",
    "SystemExit",
    "InvalidInput",
    "NotFound",
    "AmbiguousReference",
    "Usage",
}
# Names a message may start with that read as ordinary words.
_NAMES = {
    "Azure",
    "Classifier",
    "Feather",
    "Forecaster",
    "Mitra",
    "Modal",
    "Polars",
    "Python",
    "Regressor",
    "Timestral",
}
# Messages a caller's own checks expect word for word: scikit-learn's
# estimator checks read this one.
_VERBATIM = {"Unknown label type: {}"}


def _literal(node: ast.AST) -> str | None:
    """A literal's text, `{}` for each interpolated part; None for no literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(part.value) if isinstance(part, ast.Constant) else "{}"
            for part in node.values
        )
    return None


def _message_nodes(call: ast.Call) -> list[ast.expr]:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    # `help` is a diagnostic's only on `report` and `warning`; argparse's
    # help texts are descriptions, capitalized as cargo's own --help is.
    found = [
        keyword.value
        for keyword in call.keywords
        if keyword.arg in ("message", "hint", "because")
        or (keyword.arg == "help" and name in ("report", "warning"))
    ]
    if name in _CODED or (name == "Error" and len(call.args) >= 2):
        found += call.args[1:2]
    elif name in _PLAIN:
        found += call.args[:1]
    elif name in ("require", "ensure", "report"):
        found += call.args[1:2]
    elif name in ("error", "warning") and isinstance(func, ast.Attribute):
        found += call.args[:1]
    return found


def style(message: str) -> list[str]:
    """What a message breaks of cargo's style: a capitalized first word, a final period."""
    problems = []
    word = (message.split(maxsplit=1) or [""])[0].rstrip(",:;")
    if re.fullmatch(r"[A-Z][a-z]+", word) and word not in _NAMES:
        problems.append("starts with a capitalized word")
    if message.rstrip().endswith(".") and not message.rstrip().endswith("..."):
        problems.append("ends with a period")
    return problems


def python_messages(source: str) -> list[str]:
    """Each message literal in Python `source` that breaks the style, as `line: text`."""
    broken = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for message in _message_nodes(node):
                text = _literal(message)
                if text is None or text.startswith("{}") or text in _VERBATIM:
                    continue
                if style(text):
                    broken.append(f"{message.lineno}: {text}")
    return broken


# A Rust error's message literal: an ApiError or RequestError, a refusal
# helper, an `Err(...)` or `map_err` closure, a `bail!`, or a Display `write!`.
_RUST_MESSAGE = re.compile(
    r"(?:ApiError::(?:new|validation)|RequestError::(?:new|invalid)|\binvalid|\brefused"
    r"|\brequire|\bErr|\bbail!|write!\(\s*f,)\(\s*(?:\"[A-Z][A-Za-z]+\",\s*)?"
    r"(?:&?format!\(\s*)?\"((?:[^\"\\\\]|\\\\.)*)\""
)


def rust_messages(source: str) -> list[str]:
    """Each error message literal in Rust `source` that breaks the style, as `line: text`."""
    broken = []
    for match in _RUST_MESSAGE.finditer(source):
        text = match.group(1)
        if text.startswith("{") or not style(text):
            continue
        broken.append(f"{source[: match.start()].count(chr(10)) + 1}: {text}")
    return broken


def test_every_error_message_follows_cargo_style():
    broken = {}
    for path in tracked("src/*.py"):
        if path.name != "_api.py" and (found := python_messages(path.read_text())):
            broken[str(path.relative_to(ROOT))] = found
    for path in tracked("crates/*/src/*.rs"):
        if found := rust_messages(path.read_text()):
            broken[str(path.relative_to(ROOT))] = found
    assert broken == {}


def test_the_style_check_refuses_a_capitalized_or_full_stopped_message():
    assert python_messages(
        "raise ServiceError('ValidationError', 'Invalid port', 422)\n"
        "ensure(ok, 'the port is invalid.')\n"
        "raise InvalidInput(f'Bad {name}', hint='Try again.')\n"
        "out.warning('Something broke')\n"
        "raise Error('Internal', 'Environment failed')\n"
    ) == [
        "1: Invalid port",
        "2: the port is invalid.",
        "3: Try again.",
        "3: Bad {}",
        "4: Something broke",
        "5: Environment failed",
    ]
    assert (
        python_messages(
            "raise InvalidInput('gf.yaml has no `target`', hint='add one')\n"
            "raise ValueError(f'{name} is invalid')\n"
            "ensure(ok, 'Python runtime placement needs one worker')\n"
        )
        == []
    )
    assert rust_messages(
        'ApiError::validation("Invalid port")\n'
        'Err(format!("the port {port} is invalid."))\n'
        'RequestError::new("ValidationError", "Bad input", 422)\n'
        'ApiError::validation("as_of is an RFC 3339 time")\n'
    ) == ["1: Invalid port", "2: the port {port} is invalid.", "3: Bad input"]


def notebook_api(source: str) -> list[str]:
    """What notebook code uses of `gf` that gf does not have: names, and keywords of calls."""
    import inspect

    import generalforecast as gf

    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )
    missing = []
    for node in ast.walk(ast.parse(code)):
        target = node.func if isinstance(node, ast.Call) else node
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "gf"
        ):
            continue
        if target.attr not in gf.__all__:
            missing.append(f"gf.{target.attr}")
            continue
        if isinstance(node, ast.Call):
            accepted = inspect.signature(getattr(gf, target.attr)).parameters
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in accepted:
                    missing.append(f"gf.{target.attr}({keyword.arg}=)")
    return sorted(set(missing))


def test_every_notebook_uses_only_the_api_gf_has():
    import json

    notebooks = tracked("examples/*.ipynb")
    assert notebooks
    for path in notebooks:
        cells = json.loads(path.read_text())["cells"]
        source = "\n".join(
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
        )
        assert notebook_api(source) == [], path.name


def test_the_notebook_check_catches_a_removed_name_or_keyword():
    assert notebook_api(
        "%pip install generalforecast\n"
        "client = gf.local(allow_downloads=True)\n"
        'model = gf.Regressor("general-forecast/tabicl-v2-regressor", allow_downloads=True)\n'
        "fine = gf.Classifier(download=True)\n"
    ) == ["gf.Regressor(allow_downloads=)", "gf.local"]
