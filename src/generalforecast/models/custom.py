# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Custom models: the user's own, named by a `module:attribute` spec.

The attribute, imported and called, returns `(functions, parameters)` written
against `generalforecast.models.contract` (a classifier or regressor) or
`generalforecast.models.series` (a forecaster): what `registry.register`
loaders return. The engine's workers load it by the spec through
`generalforecast.models.registry.load_engine_model`, which checks it against
its contract once as it first loads; this process reads its properties for
the task and limits a table needs before the workers have it. Importing this
module imports no model code; a custom model's does on first use.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any

from generalforecast.models.catalog import Capability, ModelRef, ModelRefError, parse

Model = tuple[Any, Any]


def load(spec: str) -> Model:
    """Imports a custom model's attribute and calls it for its functions and parameters.

    Args:
        spec: The model's `module:attribute`.

    Returns:
        The model's contract calls and parameters.

    Raises:
        ModelRefError: `InvalidModel` if the spec is not a custom model's or
            the attribute does not return `(functions, parameters)` written
            against a contract; `UnknownModel` if the module or attribute
            does not import.
    """
    from generalforecast.models.contract import TabularFunctions
    from generalforecast.models.series import SeriesFunctions

    ref = parse(spec)
    if not ref.custom:
        raise ModelRefError(
            "InvalidModel", f"{spec!r} is not a custom model's module:attribute"
        )
    module, attribute = ref.name.split(":")
    try:
        imported = importlib.import_module(module)
    except ImportError as error:
        raise ModelRefError(
            "UnknownModel", f"custom model {spec!r} does not import: {error}"
        ) from None
    try:
        target = getattr(imported, attribute)
    except AttributeError:
        raise ModelRefError(
            "UnknownModel",
            f"custom model {spec!r} does not import: module {module!r} has no "
            f"attribute {attribute!r}",
        ) from None
    try:
        loaded = target() if callable(target) else target
    except Exception as error:
        raise ModelRefError(
            "InvalidModel",
            f"custom model {spec!r} failed to load: {type(error).__name__}: {error}",
        ) from error
    if not (
        isinstance(loaded, tuple)
        and len(loaded) == 2
        and isinstance(loaded[0], TabularFunctions | SeriesFunctions)
    ):
        raise ModelRefError(
            "InvalidModel",
            f"custom model {spec!r} is not a model: its attribute returns "
            "(functions, parameters), the functions a TabularFunctions or "
            f"SeriesFunctions; got {type(loaded).__name__}",
        )
    return loaded


def checked(spec: str) -> Model:
    """Loads a custom model and checks it against its contract, as a worker's first load does.

    Raises:
        ModelRefError: As `load`; `InvalidModel` naming the contract check
            the model fails, if it fails one.
    """
    functions, parameters = load(spec)
    from generalforecast.models import conformance
    from generalforecast.models.series import SeriesFunctions

    try:
        if isinstance(functions, SeriesFunctions):
            conformance.check_series(functions, parameters)
        else:
            conformance.check(functions, parameters)
    except AssertionError as error:
        name = str(error).split(":", 1)[0]
        raise ModelRefError(
            "InvalidModel",
            f"custom model {spec!r} fails the contract's `{name}` check; run "
            "generalforecast.models.conformance.check on it for the report",
        ) from None
    return functions, parameters


@functools.cache
def _capability(spec: str) -> Capability:
    from generalforecast.models.series import SeriesFunctions

    functions, parameters = load(spec)
    properties = functions.properties(parameters)
    if isinstance(functions, SeriesFunctions):
        return Capability("forecast", quantiles=tuple(properties.quantiles))
    return Capability(
        "regression" if properties.regression else "classification",
        class_slots=None if properties.regression else properties.class_slots,
        context_rows=properties.context_rows,
        context_features=properties.context_features,
        context_classes=properties.context_classes,
    )


def capability(ref: ModelRef) -> Capability:
    """What a model can do: its catalog entry's, or read from a custom model's properties.

    A custom model's properties are read once per process, which loads the
    model here as a worker does.

    Raises:
        ModelRefError: As `load`, for a custom model.
    """
    return _capability(ref.name) if ref.custom else ref.entry.capability
