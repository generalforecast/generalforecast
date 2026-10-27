# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The model registry the predictor layer reads, behind one seam.

Ids, tasks and limits come from `generalforecast.models.catalog`, the same
functions `generalforecast.models.registry` exports, without importing model
execution; the pin a lock records is the catalog entry projected under the v1
serving policy. The installed registry is swapped with `use`, which is how the
rendered CLI suite runs `gf check` against the contract's numbers without
forking any of this.
"""

from typing import Any, Protocol

from generalforecast.errors import InvalidInput, NotFound
from generalforecast.models import catalog
from generalforecast.predictors import models

ORGANIZATION = catalog.ORGANIZATION
Task = catalog.Task
Limits = catalog.Limits


class Registry(Protocol):
    """What the predictor layer asks a model registry."""

    def canonical(self, ref: str) -> str:
        """The canonical `general-forecast/<name>` id, pin and variant kept."""
        ...

    def task_of(self, ref: str) -> Task:
        """Whether the model classifies, regresses or forecasts."""
        ...

    def limits(self, ref: str) -> Limits:
        """The per-call bounds of a model."""
        ...

    def pin(self, ref: str) -> models.ModelPin:
        """The artifact and capability a lock records for the model."""
        ...

    def confirm(self, ref: str) -> None:
        """Checks the pin against the upstream registry; raises `PredictorError`."""
        ...


class CatalogRegistry:
    """The built-in catalog."""

    def canonical(self, ref: str) -> str:
        return catalog.canonical(ref, "predictor")

    def task_of(self, ref: str) -> Task:
        return catalog.task_of(ref)

    def limits(self, ref: str) -> Limits:
        return catalog.limits(ref)

    def pin(self, ref: str) -> models.ModelPin:
        parsed = catalog.parse(self.canonical(ref))
        return models.project(parsed.entry, parsed, catalog.V1_POLICY)

    def confirm(self, ref: str) -> None:
        models.CatalogResolver().resolve(self.canonical(ref), offline=False)


_CATALOG = CatalogRegistry()
_installed: Registry | None = None


def use(registry: Registry | None) -> None:
    """Installs the registry the predictor layer reads; None restores the catalog."""
    global _installed
    _installed = registry


def registry() -> Registry:
    """The installed registry."""
    return _installed if _installed is not None else _CATALOG


def canonical(ref: Any) -> str:
    """The canonical `general-forecast/<name>` id of a model reference.

    Args:
        ref: A model id, optionally pinned (`@sha256:...`) and with a variant
            suffix (`@K`, `*N`).

    Returns:
        The same reference spelled canonically.

    Raises:
        InvalidInput: With `param="model"`, if the reference is not a string,
            is a bare or old name, or names no catalog model.
    """
    if type(ref) is not str or not ref:
        raise InvalidInput(
            message="`model` must be a catalog id",
            param="model",
            hint=f"use {ORGANIZATION}/tabicl-v2-classifier, or run `gf model list`",
        )
    try:
        return registry().canonical(ref)
    except (InvalidInput, NotFound) as error:
        raise InvalidInput(
            message=error.message,
            param="model",
            hint=error.hint,
            because=error.because,
        ) from None


def task_of(ref: Any) -> Task:
    """Whether the model classifies, regresses or forecasts."""
    return registry().task_of(canonical(ref))


def limits(ref: Any) -> Limits:
    """The per-call bounds of a model."""
    return registry().limits(canonical(ref))


def pin(ref: Any, *, online: bool = False) -> models.ModelPin:
    """The artifact and capability a lock records for a model.

    Args:
        ref: The model reference.
        online: Also confirm the pin against the upstream registry.

    Raises:
        InvalidInput: If the reference is not a catalog id.
        models.PredictorError: If `online` and the upstream registry disagrees.
    """
    id = canonical(ref)
    pin = registry().pin(id)
    if online and pin.files:
        registry().confirm(id)
    return pin
