# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Catalog lookups the SDK needs, over `generalforecast.models.catalog`.

The SDK never imports `generalforecast.models.registry`: that module loads
models for execution and belongs to the worker. The catalog is metadata, and
`generalforecast.models.artifacts` (imported lazily) owns the cache.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from generalforecast.errors import Error, InvalidInput, NotFound
from generalforecast.models import catalog as _catalog
from generalforecast.models.catalog import Limits, ModelEntry, ModelRefError

ORGANIZATION = _catalog.ORGANIZATION
CATALOG_ID = re.compile(r"^[a-z0-9-]+/[a-z0-9.-]+(@.+)?$")
Task = Literal["classification", "regression", "forecast"]


def _base(ref: str) -> str:
    return re.split(r"[@*]", ref, maxsplit=1)[0]


def _entry(ref: str) -> ModelEntry:
    try:
        return _catalog.reference(ref).entry
    except ModelRefError as error:
        raise InvalidInput(str(error), param="model") from None


def _capability(ref: str) -> _catalog.Capability:
    """What the model can do; a custom model's is read from its loaded contract."""
    parsed = _catalog.reference(ref)
    if not parsed.custom:
        return parsed.entry.capability
    from generalforecast.models import custom

    try:
        return custom.capability(parsed)
    except ModelRefError as error:
        raise InvalidInput(str(error), param="model") from None


def canonical(ref: str) -> str:
    """The canonical id of a model reference, with its variant kept.

    A custom model's `module:attribute` is its own canonical spelling.
    """
    return str(_catalog.reference(ref))


def limits(ref: str) -> Limits:
    """The model's context row, query and input limits."""
    return _capability(ref).limits


def task_of(ref: str) -> Task:
    """The task the model performs."""
    return _capability(ref).task


def quantiles(ref: str) -> tuple[float, ...]:
    """The levels a forecaster's outputs are at, if declared; empty otherwise."""
    return _capability(ref).quantiles


def _artifacts() -> Any:
    from generalforecast.models import artifacts

    return artifacts


def is_cached(ref: str) -> bool:
    """Whether the model's files are in the local cache."""
    return bool(_artifacts().cached(_entry(ref)))


def pull(ref: str, *, progress: Any = None) -> Path:
    """Downloads the model's files if needed and returns their directory."""
    artifacts = _artifacts()
    try:
        return Path(
            artifacts.checkpoint(_entry(ref), allow_downloads=True, progress=progress)
        )
    except artifacts.ArtifactError as error:
        if error.code == "NotFound":
            raise NotFound(str(error), hint=error.hint) from None
        raise


def cache_path(ref: str) -> Path:
    """Where the model's files live in the cache."""
    return Path(_artifacts().checkpoint(_entry(ref)))


def catalog(task: Task | None = None) -> list[ModelEntry]:
    """The catalog entries, newest first, for one task or all."""
    return [
        entry
        for entry in reversed(_catalog.entries())
        if task is None or entry.capability.task == task
    ]


def is_catalog_id(ref: str) -> bool:
    """Whether `ref` has the shape `org/name[@variant]` of a catalog id."""
    return bool(CATALOG_ID.match(ref))


def known_org(ref: str) -> bool:
    """Whether a catalog-shaped `ref` names a registry organization."""
    return _base(ref).partition("/")[0] == ORGANIZATION


def is_bare_name(ref: Any) -> bool:
    """Whether `ref` is a model's old or short name, which the catalog rejects."""
    return isinstance(ref, str) and _base(ref) in _catalog.ALIASES


def is_custom(ref: str) -> bool:
    """Whether `ref` names a custom model by its `module:attribute`."""
    return _catalog.reference(ref).custom


def is_catalog_model(ref: Any) -> bool:
    """Whether `ref` names a catalog model, not a custom `module:attribute` spec."""
    if not isinstance(ref, str) or "/" not in _base(ref):
        return False
    try:
        canonical(ref)
    except Error:
        return False
    return True


def entry(ref: str) -> ModelEntry:
    """The catalog entry of a model reference."""
    return _entry(ref)


def short(ref: str) -> str:
    """The name after the organization, without a variant or digest."""
    base = _base(ref)
    return base.partition("/")[2] if "/" in base else base


def not_cached(ref: str) -> NotFound:
    """The error for a model whose files are not in the cache."""
    id = canonical(ref)
    return NotFound(
        f"model {id} is not cached",
        hint=f"run `gf model pull {id}`, or pass download=True",
    )


__all__ = [
    "Limits",
    "cache_path",
    "canonical",
    "catalog",
    "entry",
    "is_bare_name",
    "is_cached",
    "is_catalog_id",
    "is_catalog_model",
    "is_custom",
    "known_org",
    "limits",
    "not_cached",
    "pull",
    "quantiles",
    "short",
    "task_of",
]
