# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""generalforecast/_core.pyi against the extension, so editors show the Rust docstrings."""

import ast
import inspect
from pathlib import Path
from typing import Any

import generalforecast._core as core

STUB = Path(core.__file__).with_name("_core.pyi")


def _stub_docs(body: list[ast.stmt], prefix: str = "") -> dict[str, str | None]:
    """Each stub item's docstring by dotted name, attributes' included."""
    docs: dict[str, str | None] = {}
    for node, after in zip(body, [*body[1:], None], strict=True):
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            if not node.name.startswith("__"):
                docs[prefix + node.name] = ast.get_docstring(node)
            if isinstance(node, ast.ClassDef):
                docs |= _stub_docs(node.body, f"{prefix}{node.name}.")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = after.value if isinstance(after, ast.Expr) else None
            text = value.value if isinstance(value, ast.Constant) else None
            docs[prefix + node.target.id] = (
                inspect.cleandoc(text) if isinstance(text, str) else None
            )
    return docs


def _runtime_doc(name: str) -> str | None:
    value: Any = core
    for part in name.split("."):
        value = getattr(value, part)
    if isinstance(value, int):
        # A constant's docstring is its type's.
        return None
    return inspect.cleandoc(value.__doc__) if value.__doc__ else None


def test_stub_docstrings_match_the_extension() -> None:
    docs = _stub_docs(ast.parse(STUB.read_text()).body)

    assert docs == {name: _runtime_doc(name) for name in docs}
