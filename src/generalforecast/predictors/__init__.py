# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Predictors: the manifest (`gf.yaml`), its resolution and the lock (`gf.lock`).

Exports load lazily so reading a manifest never imports Arrow.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from generalforecast.predictors.lock import load_lock as load_lock
    from generalforecast.predictors.lock import write_lock as write_lock
    from generalforecast.predictors.resolution import ContextInfo as ContextInfo
    from generalforecast.predictors.resolution import Resolution as Resolution
    from generalforecast.predictors.resolution import resolve as resolve
    from generalforecast.predictors.resolution import verified as verified
    from generalforecast.predictors.selection import SelectionInfo as SelectionInfo
    from generalforecast.predictors.spec import PredictorSpec as PredictorSpec
    from generalforecast.predictors.spec import load as load
    from generalforecast.predictors.spec import write_manifest as write_manifest
    from generalforecast.predictors.tables import TableInfo as TableInfo
    from generalforecast.predictors.tables import default_until as default_until
    from generalforecast.predictors.tables import inspect_table as inspect_table
    from generalforecast.predictors.tables import split as split
    from generalforecast.predictors.tables import (
        target_candidates as target_candidates,
    )
    from generalforecast.predictors.tables import task_for as task_for

__all__ = [
    "ContextInfo",
    "PredictorSpec",
    "Resolution",
    "SelectionInfo",
    "TableInfo",
    "default_until",
    "inspect_table",
    "load",
    "load_lock",
    "resolve",
    "split",
    "target_candidates",
    "task_for",
    "verified",
    "write_lock",
    "write_manifest",
]

_MODULES = {
    "PredictorSpec": "generalforecast.predictors.spec",
    "load": "generalforecast.predictors.spec",
    "write_manifest": "generalforecast.predictors.spec",
    "TableInfo": "generalforecast.predictors.tables",
    "inspect_table": "generalforecast.predictors.tables",
    "split": "generalforecast.predictors.tables",
    "task_for": "generalforecast.predictors.tables",
    "target_candidates": "generalforecast.predictors.tables",
    "default_until": "generalforecast.predictors.tables",
    "Resolution": "generalforecast.predictors.resolution",
    "ContextInfo": "generalforecast.predictors.resolution",
    "SelectionInfo": "generalforecast.predictors.selection",
    "resolve": "generalforecast.predictors.resolution",
    "verified": "generalforecast.predictors.resolution",
    "write_lock": "generalforecast.predictors.lock",
    "load_lock": "generalforecast.predictors.lock",
}


def __getattr__(name: str) -> Any:
    """Imports the module that defines a name on first use."""
    import importlib

    if (module := _MODULES.get(name)) is None:
        raise AttributeError(name)
    return getattr(importlib.import_module(module), name)
