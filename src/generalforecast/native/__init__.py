# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""General Forecast inference: serving tabular foundation models."""

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Seen by type checkers and editors only; at runtime `__getattr__` imports them.
    from generalforecast.native.estimators import Classifier as Classifier
    from generalforecast.native.estimators import Regressor as Regressor
    from generalforecast.native.local_client import LocalClient as LocalClient
    from generalforecast.native.local_client import local as local
    from generalforecast.native.tables import Grouping as Grouping
    from generalforecast.native.tables import Retrieved as Retrieved
    from generalforecast.native.tables import Schema as Schema
    from generalforecast.native.tables import SeriesSchema as SeriesSchema
    from generalforecast.native.tables import SeriesTable as SeriesTable
    from generalforecast.native.tables import Table as Table
    from generalforecast.native.tables import TableOptions as TableOptions
    from generalforecast.native.tables import TimeOptions as TimeOptions

try:
    __version__ = version("generalforecast")
except PackageNotFoundError:  # A checkout imported without being installed.
    __version__ = "0.0.0"

__all__ = [
    "Grouping",
    "LocalClient",
    "Retrieved",
    "Schema",
    "SeriesSchema",
    "SeriesTable",
    "Table",
    "TableOptions",
    "TimeOptions",
    "__version__",
    "local",
]

# Exports load on first use, so metadata tools such as `gf check` never import
# the native core, JAX or sklearn.
_EXPORTS = {
    "Classifier": "generalforecast.native.estimators",
    "Regressor": "generalforecast.native.estimators",
    "LocalClient": "generalforecast.native.local_client",
    "local": "generalforecast.native.local_client",
    **dict.fromkeys(
        (
            "Grouping",
            "Retrieved",
            "Schema",
            "SeriesSchema",
            "SeriesTable",
            "Table",
            "TableOptions",
            "TimeOptions",
        ),
        "generalforecast.native.tables",
    ),
}


def __getattr__(name: str) -> Any:
    """Imports an export on first use."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
