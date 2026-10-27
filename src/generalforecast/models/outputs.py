# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How every surface spells class keys and reads quantiles.

The local SDK, predictors and the Models API all present a model's outputs the
same way: classes keyed by one spelling, and quantile levels read from the
model's own output grid. Standard library only.
"""

from collections.abc import Sequence
from typing import Any

# Quantiles returned when a caller asks for none.
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
# A regressor's outputs: quantile levels k/1000 for k = 1 .. 999.
REGRESSION_STEP = 0.001
REGRESSION_LEVELS = 999


def class_key(label: Any) -> str:
    """A class's key in a probabilities mapping: `true`/`false` or `str(label)`."""
    return str(label).lower() if type(label) is bool else str(label)


def probabilities(classes: Sequence[Any], row: Sequence[float]) -> dict[str, float]:
    """One row's class probabilities, keyed by class."""
    return {
        class_key(label): float(value)
        for label, value in zip(classes, row[: len(classes)], strict=True)
    }


def quantile_index(level: float, grid: Sequence[float] | None = None) -> int:
    """The output column of a quantile level.

    Args:
        level: The level, in (0, 1).
        grid: A forecaster's output levels, or None for a regressor's k/1000 grid.

    Raises:
        ValueError: If the model does not output that level.
    """
    if grid is not None:
        if level not in grid:
            raise ValueError(f"the model does not output the {level} quantile")
        return list(grid).index(level)
    index = round(level / REGRESSION_STEP)
    if (
        abs(level / REGRESSION_STEP - index) > 1e-9
        or not 1 <= index <= REGRESSION_LEVELS
    ):
        raise ValueError("quantiles use levels k/1000 for k=1..999")
    return index - 1


def quantiles(
    row: Sequence[float], levels: Sequence[float], grid: Sequence[float] | None = None
) -> dict[str, float]:
    """One prediction's requested quantiles, keyed by level."""
    return {str(level): float(row[quantile_index(level, grid)]) for level in levels}
