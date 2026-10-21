# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Models by name: built-in checkpoints, or trusted `module:function` specs.

A built-in model goes by its canonical name, such as `tabicl-v2-classifier`,
an alias such as `tabicl-classifier`, or its repository id, such as
`general-forecast/tabicl-v2-classifier`; `generalforecast.models.catalog` holds each one's
pinned files, capability and qualification. Its checkpoint loads through
`generalforecast.models.artifacts`, verified against those pins, from `GF_CHECKPOINTS`
(one subdirectory per model) on machines without internet or else from Hugging
Face. Mitra-2, Chronos-2 and Timestral load from converted mirrors under
`general-forecast/`; TimesFM 3, whose license forbids redistribution, converts
at load from Google's repository into a directory with a verified derived
manifest.

`@K` and `*N` wrap a classifier or regressor in the existing
`generalforecast.models.many_class` and `generalforecast.models.ensemble` models, wherever it runs.
"""

import dataclasses
import importlib
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from generalforecast.models import catalog
from generalforecast.models.catalog import ModelEntry, ModelRef, ModelRefError, Surface
from generalforecast.models.contract import TabularFunctions
from generalforecast.models.series import SeriesFunctions

TabularModel = tuple[TabularFunctions, Any]
SeriesModel = tuple[SeriesFunctions, Any]
Model = TabularModel | SeriesModel

ORGANIZATION = catalog.ORGANIZATION

# Each family's loader and contract calls, as `module:attribute`.
_FAMILIES: dict[str, tuple[str, str]] = {
    "tabicl": (
        "generalforecast.models.tabicl.checkpoint:load_tabicl",
        "generalforecast.models.tabicl.model:tabicl_functions",
    ),
    "kumo": (
        "generalforecast.models.kumo.checkpoint:load_kumo",
        "generalforecast.models.kumo.model:kumo_functions",
    ),
    "tabpfn": (
        "generalforecast.models.tabpfn.checkpoint:load_tabpfn",
        "generalforecast.models.tabpfn.model:tabpfn_functions",
    ),
    "limix": (
        "generalforecast.models.limix.checkpoint:load_limix",
        "generalforecast.models.limix.model:limix_functions",
    ),
    "mitra": (
        "generalforecast.models.mitra.checkpoint:load_mitra",
        "generalforecast.models.mitra.model:mitra_functions",
    ),
    "chronos2": (
        "generalforecast.models.chronos2.checkpoint:load_chronos2",
        "generalforecast.models.chronos2.model:chronos2_functions",
    ),
    "timestral": (
        "generalforecast.models.timestral.checkpoint:load_timestral",
        "generalforecast.models.timestral.model:timestral_functions",
    ),
    "timesfm3": (
        "generalforecast.models.timesfm3.checkpoint:load_timesfm",
        "generalforecast.models.timesfm3.model:timesfm_functions",
    ),
}

# The pinned revision of each built-in checkpoint, read from the catalog.
REVISIONS: dict[str, str] = {
    entry.name: entry.artifact.revision
    for entry in catalog.entries()
    if entry.artifact.repository == entry.repository_id
}


def _attribute(path: str) -> Any:
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def _checkpoint(name: str) -> Path:
    """A built-in checkpoint's verified directory, for trusted local use."""
    return _artifacts().checkpoint(catalog.ENTRIES[name])


def _artifacts() -> Any:
    from generalforecast.models import artifacts

    return artifacts


def _native(entry: ModelEntry, surface: Surface, allow_downloads: bool) -> Model:
    """A built-in model's contract calls and verified, traced parameters."""
    loader, functions = _FAMILIES[entry.family]
    directory = _artifacts().checkpoint(
        entry, surface=surface, allow_downloads=allow_downloads
    )
    parameters = _attribute(loader)(directory, **dict(entry.options))
    if surface != "local":
        import jax

        from generalforecast.runtime.device import configure, device_settings

        parameters = configure(parameters, device_settings(jax.default_backend()))
    return _trained(
        entry.name, _traced(entry.name, (_attribute(functions), parameters))
    )


# Package and remote processes share one verified copy of each model, and one
# set of wrapped contract calls per variant, which compiles once.
@lru_cache(maxsize=4)
def _served(name: str, surface: Surface, allow_downloads: bool) -> Model:
    return _native(catalog.ENTRIES[name], surface, allow_downloads)


@lru_cache(maxsize=16)
def _variant(
    name: str,
    surface: Surface,
    allow_downloads: bool,
    codes: int | None,
    estimators: int | None,
) -> Model:
    functions, parameters = _served(name, surface, allow_downloads)
    return wrap(functions, parameters, ModelRef(name, codes, estimators))


def load(
    spec: str, *, surface: Surface = "local", allow_downloads: bool = True
) -> Model:
    """Loads a model's contract calls and parameters.

    Args:
        spec: A built-in model's name, alias or repository id, such as
            `tabicl-classifier`, or, for local use only, `module:function`
            naming a function that returns `(functions, parameters)`; then
            `@K` for K error-correcting code estimators, which serve more
            classes than the classifier has slots, then `*N` for an ensemble
            of N estimators: each at most once, in that order. A forecaster
            takes neither.
        surface: Where the model runs. Local use trusts `module:function` and
            checkpoints without recorded hashes; packages and remote serving
            need the model's qualification gates, every pinned file and the
            serving policy's bounds on codes and estimators.
        allow_downloads: Whether missing pinned files may download.

    Returns:
        The model's contract calls and parameters.

    Raises:
        ValueError: If the spec is malformed, names no model or one not
            qualified where it runs, or puts error-correcting codes on a
            regressor or estimators on a forecaster.
    """
    return load_ref(
        catalog.parse(spec, trusted=surface == "local"),
        surface=surface,
        allow_downloads=allow_downloads,
    )


def load_ref(
    ref: ModelRef, *, surface: Surface = "local", allow_downloads: bool = True
) -> Model:
    """Loads a parsed model reference, as `load` does a spec."""
    if ref.trusted:
        if surface != "local":
            raise ModelRefError(
                "UntrustedModel", "module:function models run only locally"
            )
        module, function = ref.name.split(":", 1)
        functions, parameters = getattr(importlib.import_module(module), function)()
    else:
        entry = ref.entry
        if not entry.allows(surface, variant=ref.variant):
            raise ModelRefError(
                "UnsupportedModel",
                f"{ref} is not qualified for {surface} use: {entry.gates.evidence}",
            )
        if surface != "local" and (refused := catalog.V1_POLICY.admits(ref)):
            raise ModelRefError("CapacityExceeded", f"serving policy allows {refused}")
        if surface != "local":
            return _variant(
                entry.name, surface, allow_downloads, ref.codes, ref.estimators
            )
        functions, parameters = _native(entry, surface, allow_downloads)
    return wrap(functions, parameters, ref)


def wrap(functions: Any, parameters: Any, ref: ModelRef) -> Model:
    """A model in the `@K` codes and `*N` ensemble a reference asks for."""
    if isinstance(functions, SeriesFunctions):
        if ref.variant:
            raise ModelRefError(
                "UnsupportedCapability",
                f"a forecaster takes no codes or estimators: {ref}",
            )
        return functions, parameters
    if ref.codes is not None:
        from generalforecast.models.many_class import ManyClass, many_class_functions

        if functions.properties(parameters).regression:
            raise ModelRefError(
                "UnsupportedCapability",
                f"error-correcting codes need a classifier: {ref}",
            )
        functions = many_class_functions(functions)
        parameters = ManyClass(parameters, ref.codes)
    if ref.estimators is not None:
        from generalforecast.models.ensemble import Ensemble, ensemble_functions

        functions = ensemble_functions(functions)
        parameters = Ensemble(parameters, ref.estimators)
    return functions, parameters


def load_tabular(spec: str) -> TabularModel:
    """Loads a tabular model, as `load`, refusing a forecaster.

    Args:
        spec: A model spec, as `load` takes.

    Returns:
        The model's contract calls and parameters.

    Raises:
        ValueError: If the spec names a forecaster, which only a series table
            can use, or `load` refuses it.
    """
    functions, parameters = load(spec)
    if isinstance(functions, SeriesFunctions):
        raise ValueError(  # noqa: TRY004
            f"{spec!r} is a forecaster, which only a series table uses"
        )
    return functions, parameters


# The built-in models, by repository id.
BUILT_IN: dict[str, Callable[[], Model]] = {
    entry.repository_id: partial(load, entry.name) for entry in catalog.entries()
}


def _trained(name: str, model: Model) -> Model:
    """A built-in model whose properties say the largest contexts it was trained on."""
    short = name.removeprefix(f"{ORGANIZATION}/")
    functions, parameters = model
    known = catalog.ENTRIES.get(short)
    if known is None or isinstance(functions, SeriesFunctions):
        return model
    capability = known.capability
    if capability.context_rows is None:
        return model

    def properties(parameters: Any) -> Any:
        return dataclasses.replace(
            functions.properties(parameters),
            context_rows=capability.context_rows,
            context_features=capability.context_features,
            context_classes=capability.context_classes,
        )

    return functions._replace(properties=properties), parameters


def _traced(name: str, model: Model) -> Model:
    """A loaded checkpoint, once its weights trace through every contract call.

    Loaders check that every parameter is there, not its shape, so a damaged
    or mismatched tensor would otherwise surface at a table's first fit, as a
    shape error naming no checkpoint.

    Raises:
        ValueError: Naming the checkpoint, if its weights do not fit its settings.
    """
    import jax
    import jax.numpy as jnp

    from generalforecast.models.conformance import Table, predict
    from generalforecast.models.series import TARGET, SeriesGroup

    functions, parameters = model
    rows, columns = 8, 2

    def run(parameters: Any) -> Any:
        if isinstance(functions, SeriesFunctions):
            properties = functions.properties(parameters)
            steps = 2 * properties.patch
            group = SeriesGroup(
                jnp.zeros((2, steps), jnp.float32),
                jnp.ones((2, steps), bool),
                jnp.full(2, TARGET, jnp.int32),
            )
            return functions.forecast(parameters, group, properties.output_patch)
        regression = functions.properties(parameters).regression
        targets = (
            jnp.zeros(rows, jnp.float32)
            if regression
            else jnp.arange(rows, dtype=jnp.int32) % 2
        )
        features = jnp.zeros((rows, columns), jnp.float32)
        data = Table(features, targets, features[:2], 0 if regression else 2)
        return predict(functions, parameters, data)

    try:
        jax.eval_shape(run, parameters)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"checkpoint {name!r} has weights that do not fit its settings: {error}"
        ) from None
    return model
