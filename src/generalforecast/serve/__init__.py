# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A standalone model server for prediction packages and catalog models.

`gf serve` loads one locked package or catalog model and answers the hosted
Models API routes with the same request and response envelopes, so the SDK
works against both. It holds no accounts, revisions or database; put auth
beyond a single bearer key and versioning in front of it.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from generalforecast.serve.app import Application as Application
    from generalforecast.serve.listener import Server as Server
    from generalforecast.serve.listener import add_arguments as add_arguments
    from generalforecast.serve.listener import main as main
    from generalforecast.serve.listener import run as run
    from generalforecast.serve.models import CatalogModel as CatalogModel
    from generalforecast.serve.models import PackageModel as PackageModel
    from generalforecast.serve.models import Served as Served
    from generalforecast.serve.models import load as load

_EXPORTS = {
    "Application": "app",
    "CatalogModel": "models",
    "PackageModel": "models",
    "Served": "models",
    "Server": "listener",
    "add_arguments": "listener",
    "load": "models",
    "main": "listener",
    "run": "listener",
}
__all__ = [
    "Application",
    "CatalogModel",
    "PackageModel",
    "Served",
    "Server",
    "add_arguments",
    "load",
    "main",
    "run",
]


def __getattr__(name: str) -> Any:
    """Imports model runtime modules only when they are used."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib

    return getattr(
        importlib.import_module(f"generalforecast.serve.{_EXPORTS[name]}"), name
    )
