"""General Forecast inference: serving tabular foundation models."""

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Seen by type checkers and editors only; at runtime `__getattr__` imports them.
    from generalforecast._core import GroupStrategy as GroupStrategy
    from generalforecast._core import Strategy as Strategy
    from generalforecast.columns import Column as Column
    from generalforecast.native.client import Answer as Answer
    from generalforecast.native.client import Definition as Definition
    from generalforecast.native.client import Described as Described
    from generalforecast.native.client import RebuildWait as RebuildWait
    from generalforecast.native.client import SeriesDefinition as SeriesDefinition
    from generalforecast.native.client import SeriesTable as SeriesTable
    from generalforecast.native.client import Table as Table
    from generalforecast.native.estimators import Classifier as Classifier
    from generalforecast.native.estimators import Regressor as Regressor
    from generalforecast.native.local_client import LocalClient as LocalClient
    from generalforecast.native.local_client import LocalSeriesTable as LocalSeriesTable
    from generalforecast.native.local_client import LocalTable as LocalTable
    from generalforecast.native.local_client import local as local

try:
    __version__ = version("generalforecast")
except PackageNotFoundError:  # A checkout imported without being installed.
    __version__ = "0.0.0"

# The estimators need sklearn, an optional extra, so a star import leaves them
# out; `generalforecast.Classifier` and `generalforecast.Regressor` still import them on first use.
__all__ = [
    "Answer",
    "Column",
    "Definition",
    "Described",
    "GroupStrategy",
    "LocalClient",
    "LocalSeriesTable",
    "LocalTable",
    "RebuildWait",
    "SeriesDefinition",
    "SeriesTable",
    "Strategy",
    "Table",
    "__version__",
    "local",
]


# Exports load on first use, so metadata tools such as `gf check` never import
# the native core, JAX or sklearn.
_EXPORTS = {
    "GroupStrategy": "generalforecast._core",
    "Strategy": "generalforecast._core",
    "Column": "generalforecast.columns",
    "Classifier": "generalforecast.native.estimators",
    "Regressor": "generalforecast.native.estimators",
    **dict.fromkeys(
        (
            "Answer",
            "Definition",
            "Described",
            "RebuildWait",
            "SeriesDefinition",
            "SeriesTable",
            "Table",
        ),
        "generalforecast.native.client",
    ),
    **dict.fromkeys(
        ("LocalClient", "LocalSeriesTable", "LocalTable", "local"),
        "generalforecast.native.local_client",
    ),
}


def __getattr__(name: str) -> Any:
    """Imports an export's module on first use and caches the attribute."""
    if (module := _EXPORTS.get(name)) is None:
        raise AttributeError(f"module 'generalforecast' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
