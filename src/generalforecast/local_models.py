# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native caller-context inference without management or persistent writes."""

import sys
from pathlib import Path
from typing import Any

from generalforecast.deployments.boundary import Request, Resource, Scope
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.interfaces import InferenceRequest, InferenceResult
from generalforecast.models.catalog import resource_name

_SPEC_ONLY = (
    "local models run in the engine's workers; give a model spec, such as "
    "'tabicl-classifier' or 'package.module:function'"
)


def code_digest(model: str) -> str:
    """The `sha256:` of the module file a trusted `module:function` spec names.

    It finds the file on `sys.path`, which the workers inherit, as their
    import would, without importing anything, so the parent never runs model
    code.

    Raises:
        PackageError: If no source file holds the module.
    """
    import hashlib

    from generalforecast.packages.validation import ensure

    origin = _source(model.partition(":")[0])
    ensure(
        origin is not None, "trusted model module has no source file", "ModelNotFound"
    )
    assert origin is not None
    return "sha256:" + hashlib.sha256(origin.read_bytes()).hexdigest()


def _source(module: str) -> Path | None:
    """The source file of a module, found as a path import finds it.

    Each name is a regular package or a module file in the first location
    that holds one, or else a namespace package over every location that
    holds its directory.
    """
    locations = [Path(entry or ".") for entry in sys.path]
    origin = None
    for part in module.split("."):
        origin, namespace = None, []
        for location in locations:
            package = location / part
            if (package / "__init__.py").is_file():
                origin, locations = package / "__init__.py", [package]
                break
            if (location / f"{part}.py").is_file():
                origin, locations = location / f"{part}.py", []
                break
            if package.is_dir():
                namespace.append(package)
        else:
            locations = namespace
        if origin is None and not locations:
            return None
    return origin


def custom_model(model: Any) -> bool:
    """Whether a model spec names trusted `module:function` code, not a built-in model.

    Raises:
        TypeError: If the model is not a spec.
        PackageError: If the spec is malformed.
    """
    from generalforecast.models.catalog import ModelRefError, parse
    from generalforecast.packages.validation import PackageError

    if not isinstance(model, str):
        raise TypeError(_SPEC_ONLY)
    try:
        return parse(model, trusted=True).trusted
    except ModelRefError as error:
        raise PackageError(error.code, str(error)) from None


def native_model(model: Any, task: str) -> str:
    """The spec the engine's workers load a model by, checked for `task`.

    Names, aliases and `@K`/`*N` variants resolve offline through the shared
    catalog; the engine verifies pinned files as its workers load them.
    `module:function` names trusted local code, which the workers check.
    """
    from generalforecast.models.catalog import parse
    from generalforecast.packages.validation import ensure

    if not custom_model(model):
        ensure(
            parse(model).entry.capability.task == task,
            "model does not support estimator task",
        )
    return model


class LocalModels:
    """Runs the SAME catalog port as the remote Models API."""

    def __init__(self, *, allow_downloads: bool = False) -> None:
        """Defaults to verified cached weights; downloads require explicit opt-in."""
        self._catalog = CatalogService(allow_downloads=allow_downloads)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Validates the shared envelope, then invokes genuine native inference."""
        checked = Request.parse(
            Request(
                Scope("local", "local-inference"),
                Resource("model", resource_name(request.get("model", ""))),
                dict(request),
            ).to_dict()
        )
        return self._catalog.infer(checked.scope, request)

    def list(self) -> list[dict]:
        """Lists curated metadata without downloading weights."""
        return self._catalog.list(Scope("local", "local-models"))
