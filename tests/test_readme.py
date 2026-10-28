# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every Python block of README.md runs, on the tiny models each catalog id stands for.

The release runs the same test on its wheel with `GF_README_MODELS=catalog`,
where the blocks load the real checkpoints.
"""

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from generalforecast.engine import LocalEngine

README = Path(__file__).resolve().parents[1] / "README.md"
# The tiny model the engine's workers load for each catalog id the README names.
TINY = {
    "general-forecast/tabicl-v2-classifier": "tests/tabicl-classifier",
    "general-forecast/chronos-2": "tests/chronos2-catalog-forecaster",
}


def blocks(text: str) -> list[str]:
    """The README's ```python blocks, in order."""
    return re.findall(r"^```python\n(.*?)^```$", text, flags=re.DOTALL | re.MULTILINE)


@pytest.fixture
def models() -> Iterator[None]:
    if os.environ.get("GF_README_MODELS") == "catalog":
        yield
        return
    tiny = LocalEngine(models=TINY)

    def local(surface: Any = "local", allow_downloads: bool = False) -> LocalEngine:
        return tiny.gated(surface, allow_downloads)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("generalforecast.engine.local", local)
        patch.setattr("generalforecast.native.estimators._shared", {})
        yield
    tiny.close()


def test_every_python_block_of_the_readme_runs(models):
    found = blocks(README.read_text())
    assert found, "README.md has no python block"
    for index, block in enumerate(found):
        exec(compile(block, f"README.md block {index + 1}", "exec"), {})  # noqa: S102


def test_blocks_are_read_as_written():
    assert blocks("x\n```python\na = 1\n```\n```sh\nls\n```\n") == ["a = 1\n"]
