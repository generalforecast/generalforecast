# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""gf SDK: versioned HTTP clients and the existing native notebook engine."""

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from generalforecast.async_client import AsyncClient as AsyncClient
    from generalforecast.client import Client as Client
    from generalforecast.contracts import ContextVersion as ContextVersion
    from generalforecast.contracts import DatasetImportRef as DatasetImportRef
    from generalforecast.contracts import DevSession as DevSession
    from generalforecast.contracts import Environment as Environment
    from generalforecast.contracts import PatchRequest as PatchRequest
    from generalforecast.contracts import PredictionResult as PredictionResult
    from generalforecast.contracts import PromoteRequest as PromoteRequest
    from generalforecast.contracts import Revision as Revision
    from generalforecast.contracts import RuntimePlan as RuntimePlan
    from generalforecast.deployments.contracts import ServiceError as ServiceError
    from generalforecast.deployments.sdk import Dataset as Dataset
    from generalforecast.deployments.sdk import DatasetSpec as DatasetSpec
    from generalforecast.deployments.sdk import Deployment as Deployment
    from generalforecast.deployments.sdk import Development as Development
    from generalforecast.deployments.sdk import ImportReceipt as ImportReceipt
    from generalforecast.deployments.sdk import Job as Job
    from generalforecast.deployments.sdk import JobSpec as JobSpec
    from generalforecast.deployments.sdk import RevisionDetails as RevisionDetails
    from generalforecast.deployments.sdk import Runtime as Runtime
    from generalforecast.deployments.sdk import Snapshot as Snapshot
    from generalforecast.deployments.sdk import WriteReceipt as WriteReceipt
    from generalforecast.estimators import Classifier as Classifier
    from generalforecast.estimators import Regressor as Regressor
    from generalforecast.local_models import LocalModels as LocalModels
    from generalforecast.native import Answer as Answer
    from generalforecast.native import Column as Column
    from generalforecast.native import Definition as Definition
    from generalforecast.native import Described as Described
    from generalforecast.native import GroupStrategy as GroupStrategy
    from generalforecast.native import LocalClient as LocalClient
    from generalforecast.native import LocalSeriesTable as LocalSeriesTable
    from generalforecast.native import LocalTable as LocalTable
    from generalforecast.native import RebuildWait as RebuildWait
    from generalforecast.native import SeriesDefinition as SeriesDefinition
    from generalforecast.native import SeriesTable as SeriesTable
    from generalforecast.native import Strategy as Strategy
    from generalforecast.native import Table as Table
    from generalforecast.native import local as local
    from generalforecast.packages import Package as Package
    from generalforecast.packages import PackageError as PackageError
    from generalforecast.packages import PackageLock as PackageLock
    from generalforecast.packages import PackageSpec as PackageSpec
    from generalforecast.packages import PythonLock as PythonLock
    from generalforecast.transport import HostedOnlyError as HostedOnlyError
    from generalforecast.transport import Profile as Profile

try:
    __version__ = version("generalforecast")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Answer",
    "AsyncClient",
    "Classifier",
    "Client",
    "Column",
    "ContextVersion",
    "Dataset",
    "DatasetImportRef",
    "DatasetSpec",
    "Definition",
    "Deployment",
    "Described",
    "DevSession",
    "Development",
    "Environment",
    "Forecaster",
    "GroupStrategy",
    "HostedOnlyError",
    "ImportReceipt",
    "Job",
    "JobSpec",
    "LocalClient",
    "LocalModels",
    "LocalSeriesTable",
    "LocalTable",
    "Package",
    "PackageError",
    "PackageLock",
    "PackageSpec",
    "PatchRequest",
    "PredictionResult",
    "Profile",
    "PromoteRequest",
    "PythonLock",
    "RebuildWait",
    "Regressor",
    "Revision",
    "RevisionDetails",
    "Runtime",
    "RuntimePlan",
    "SeriesDefinition",
    "SeriesTable",
    "ServiceError",
    "Snapshot",
    "Strategy",
    "Table",
    "WriteReceipt",
    "__version__",
    "local",
]


# Exports load on first use, so `gf check` imports neither HTTP clients nor the
# native engine; sklearn stays optional for the estimators.
_EXPORTS = {
    "Described": "generalforecast.native",
    "LocalSeriesTable": "generalforecast.native",
    "LocalTable": "generalforecast.native",
    "RebuildWait": "generalforecast.native",
    "SeriesTable": "generalforecast.native",
    "Table": "generalforecast.native",
    "Answer": "generalforecast.native",
    "AsyncClient": "generalforecast.async_client",
    "Client": "generalforecast.client",
    "Column": "generalforecast.native",
    "ContextVersion": "generalforecast.contracts",
    "Dataset": "generalforecast.deployments.sdk",
    "DatasetImportRef": "generalforecast.contracts",
    "DatasetSpec": "generalforecast.deployments.sdk",
    "Definition": "generalforecast.native",
    "Deployment": "generalforecast.deployments.sdk",
    "DevSession": "generalforecast.contracts",
    "Development": "generalforecast.deployments.sdk",
    "Environment": "generalforecast.contracts",
    "GroupStrategy": "generalforecast.native",
    "ImportReceipt": "generalforecast.deployments.sdk",
    "Job": "generalforecast.deployments.sdk",
    "JobSpec": "generalforecast.deployments.sdk",
    "LocalClient": "generalforecast.native",
    "LocalModels": "generalforecast.local_models",
    "Package": "generalforecast.packages",
    "PackageError": "generalforecast.packages",
    "PackageLock": "generalforecast.packages",
    "PackageSpec": "generalforecast.packages",
    "PatchRequest": "generalforecast.contracts",
    "PredictionResult": "generalforecast.contracts",
    "Profile": "generalforecast.transport",
    "HostedOnlyError": "generalforecast.transport",
    "PromoteRequest": "generalforecast.contracts",
    "PythonLock": "generalforecast.packages",
    "Revision": "generalforecast.contracts",
    "RevisionDetails": "generalforecast.deployments.sdk",
    "Runtime": "generalforecast.deployments.sdk",
    "RuntimePlan": "generalforecast.contracts",
    "SeriesDefinition": "generalforecast.native",
    "ServiceError": "generalforecast.deployments.contracts",
    "Snapshot": "generalforecast.deployments.sdk",
    "Strategy": "generalforecast.native",
    "WriteReceipt": "generalforecast.deployments.sdk",
    "local": "generalforecast.native",
    "Classifier": "generalforecast.estimators",
    "Forecaster": "generalforecast.estimators",
    "Regressor": "generalforecast.estimators",
}


def __getattr__(name: str) -> Any:
    """Imports an export's module on first use and caches the attribute."""
    if (module := _EXPORTS.get(name)) is None:
        raise AttributeError(f"module 'generalforecast' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
