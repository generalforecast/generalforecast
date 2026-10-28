# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def reference_group() -> None:
    """Skips these tests unless the reference dependency group is installed.

    The weekly reference job sets `GF_REFERENCE_REQUIRED`, so a missing
    package fails it instead of skipping every test.
    """
    required = os.environ.get("GF_REFERENCE_REQUIRED") == "1"
    for module in (
        "torch",
        "tabicl",
        "sdm",
        "tabpfn",
        "inference",
        "autogluon.tabular",
        "chronos",
        "timestral",
        "timesfm3",
    ):
        if required:
            importlib.import_module(module)
        else:
            pytest.importorskip(module)
