# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Importing the package loads neither sklearn nor pandas; the estimators do, on first use."""

import subprocess
import sys

_PROBE = """
import sys
import warnings
warnings.simplefilter("error")
import generalforecast as gf
loaded = sorted(name for name in ("sklearn", "pandas", "jax", "polars") if name in sys.modules)
print(gf.__version__, *loaded)
"""


def test_importing_generalforecast_loads_no_estimator_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", _PROBE],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    version, *loaded = result.stdout.split()
    assert version
    assert loaded == [], f"imported at import time: {loaded}"
    assert result.stderr == ""
