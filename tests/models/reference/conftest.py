# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def reference_group() -> None:
    """Skips these tests unless the reference dependency group is installed."""
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
        pytest.importorskip(module)
