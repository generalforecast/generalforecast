# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""General Forecast: predictors on tabular foundation models.

Exports `Predictor`, the estimators `Classifier`, `Regressor` and
`Forecaster`, the hosted `Client` and `AsyncClient`, the error classes, and
the `engine`, `errors` and `predictors` submodules.
Everything loads on first use, so `import generalforecast` stays cheap.
"""

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from generalforecast.errors import (
    AmbiguousReference,
    CapacityExceeded,
    Conflict,
    Error,
    Internal,
    InvalidInput,
    ModelNotReady,
    NotFound,
    Unauthorized,
    Unavailable,
    Unsupported,
)

if TYPE_CHECKING:
    from generalforecast import engine as engine
    from generalforecast import errors as errors
    from generalforecast import predictors as predictors
    from generalforecast.async_client import AsyncClient as AsyncClient
    from generalforecast.client import Client as Client
    from generalforecast.estimators import Classifier as Classifier
    from generalforecast.estimators import Forecaster as Forecaster
    from generalforecast.estimators import Regressor as Regressor
    from generalforecast.predictor import Predictor as Predictor

try:
    __version__ = version("generalforecast")
except PackageNotFoundError:  # A checkout imported without being installed.
    __version__ = "0.0.0"

__all__ = [
    "AmbiguousReference",
    "AsyncClient",
    "CapacityExceeded",
    "Classifier",
    "Client",
    "Conflict",
    "Error",
    "Forecaster",
    "Internal",
    "InvalidInput",
    "ModelNotReady",
    "NotFound",
    "Predictor",
    "Regressor",
    "Unauthorized",
    "Unavailable",
    "Unsupported",
    "__version__",
    "engine",
    "errors",
    "predictors",
]

_EXPORTS = {
    "Predictor": ("generalforecast.predictor", "Predictor"),
    "Client": ("generalforecast.client", "Client"),
    "AsyncClient": ("generalforecast.async_client", "AsyncClient"),
    "Classifier": ("generalforecast.estimators", "Classifier"),
    "Regressor": ("generalforecast.estimators", "Regressor"),
    "Forecaster": ("generalforecast.estimators", "Forecaster"),
    "engine": ("generalforecast.engine", None),
    "errors": ("generalforecast.errors", None),
    "predictors": ("generalforecast.predictors", None),
}


def __getattr__(name: str) -> Any:
    """Imports an export on first use."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Lists the public exports."""
    return sorted(__all__)
