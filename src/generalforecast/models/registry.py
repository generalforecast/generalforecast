# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Models by canonical id: built-in checkpoints.

A built-in model goes by its canonical id alone, such as
`general-forecast/tabicl-v2-classifier`; `generalforecast.models.catalog` holds
each one's pinned files, capability, limits and qualification, and the one
table of old spellings, `catalog.ALIASES`, which `canonical` refuses with the
canonical id as the hint. A checkpoint loads through
`generalforecast.models.artifacts`, verified against its pins, from
`GF_CHECKPOINTS` (one subdirectory per model) on machines without internet or
else from the Hugging Face cache, which `pull` fills at the pinned revision.
Mitra-2, Chronos-2 and Timestral load from converted mirrors under
`general-forecast/`; TimesFM 3, whose license forbids redistribution, converts
at load from Google's repository into a directory with a verified derived
manifest.

`@K` and `*N` wrap a classifier or regressor in the existing
`generalforecast.models.many_class` and `generalforecast.models.ensemble`
models, wherever it runs.
"""

import dataclasses
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from generalforecast.errors import NotFound
from generalforecast.models import catalog as _catalog
from generalforecast.models.catalog import (
    Limits,
    ModelEntry,
    ModelRef,
    ModelRefError,
    Surface,
    Task,
    canonical,
    limits,
    task_of,
)
from generalforecast.models.contract import TabularFunctions
from generalforecast.models.series import SeriesFunctions

TabularModel = tuple[TabularFunctions, Any]
SeriesModel = tuple[SeriesFunctions, Any]
Model = TabularModel | SeriesModel
Progress = Callable[[str, str], None]

ORGANIZATION = _catalog.ORGANIZATION

__all__ = [
    "Limits",
    "ModelInfo",
    "canonical",
    "catalog",
    "is_cached",
    "limits",
    "load",
    "pull",
    "task_of",
]


@dataclass(frozen=True)
class ModelInfo:
    """What `catalog` says about one model, for a listing or a prompt.

    Attributes:
        id: The canonical id, `general-forecast/<name>`.
        task: `classification`, `regression` or `forecast`.
        context_rows: Most context rows one call binds.
        queries: Most rows (for a forecaster, series) one call scores.
        inputs: Most input columns (for a forecaster, covariates).
        license: The SPDX id of the weights' license, or None if unrecorded.
        local: Whether it runs on this machine.
        hosted: Whether a pushed predictor may serve it.
    """

    id: str
    task: Task
    context_rows: int
    queries: int
    inputs: int
    license: str | None
    local: bool
    hosted: bool


_ref = _catalog.reference


def is_cached(ref: str) -> bool:
    """Whether a model's pinned files are already on this machine."""
    return _artifacts().cached(_ref(ref).entry)


def pull(ref: str, *, progress: Progress | None = None) -> Path:
    """Downloads a model's pinned files and verifies them.

    Args:
        ref: The model's canonical id.
        progress: Receives each status line as `(verb, message)`, such as
            `("Pulling", "general-forecast/chronos-2  (1.2 GB) to ~/.cache/huggingface")`;
            by default the line goes to stderr. Nothing is reported when the
            files are already cached.

    Returns:
        The verified checkpoint directory.

    Raises:
        NotFound: If downloads are off (`GF_OFFLINE=1`) and the files are
            not cached.
        ArtifactError: If the files differ from their pins or will not download.
    """
    artifacts = _artifacts()
    try:
        return artifacts.checkpoint(
            _ref(ref).entry, allow_downloads=True, progress=progress
        )
    except artifacts.ArtifactError as error:
        if error.code == "NotFound":
            raise NotFound(str(error), hint=error.hint) from None
        raise


def catalog(task: Task | None = None) -> list[ModelInfo]:
    """The built-in models, the recommended model of each task first.

    Args:
        task: Only models of this task, or every model.
    """
    return [
        ModelInfo(
            entry.id,
            entry.capability.task,
            entry.limits.context_rows,
            entry.limits.queries,
            entry.limits.inputs,
            entry.artifact.license,
            entry.allows("local"),
            entry.hostable and entry.gates.remote,
        )
        for entry in _catalog.entries()
        if task is None or entry.capability.task == task
    ]


# Each family's loader and contract calls, as `module:attribute`.
# Each built-in family's package under `generalforecast.models`, its checkpoint
# loader in `checkpoint` and its contract calls in `model`, imported on first use.
_FAMILIES: dict[str, tuple[str, str, str]] = {
    "tabicl": ("tabicl", "load_tabicl", "tabicl_functions"),
    "kumo": ("kumo", "load_kumo", "kumo_functions"),
    "tabpfn": ("tabpfn", "load_tabpfn", "tabpfn_functions"),
    "limix": ("limix", "load_limix", "limix_functions"),
    "mitra": ("mitra", "load_mitra", "mitra_functions"),
    "chronos2": ("chronos2", "load_chronos2", "chronos2_functions"),
    "timestral": ("timestral", "load_timestral", "timestral_functions"),
    "timesfm3": ("timesfm3", "load_timesfm", "timesfm_functions"),
}

# The pinned revision of each built-in checkpoint, by canonical id.
REVISIONS: dict[str, str] = {
    entry.id: entry.artifact.revision
    for entry in _catalog.entries()
    if entry.artifact.repository == entry.id
}


def _family(family: str) -> tuple[Any, Any]:
    """A family's checkpoint loader and contract calls, imported on first use."""
    package, loader, functions = _FAMILIES[family]
    base = f"generalforecast.models.{package}"
    return (
        getattr(importlib.import_module(f"{base}.checkpoint"), loader),
        getattr(importlib.import_module(f"{base}.model"), functions),
    )


IMPLEMENTATIONS: dict[str, Callable[[], tuple[Any, Any]]] = {}
"""The engine's own implementations by id: each loader returns `(functions, parameters)`.

Worker processes serve these next to the catalog's models; a test's small
models register here. User model specs never reach this registry: `load`
resolves catalog ids only.
"""


def register(id: str, loader: Callable[[], tuple[Any, Any]]) -> None:
    """Registers an engine implementation under `id` for `load_engine_model`."""
    IMPLEMENTATIONS[id] = loader


def _checkpoint(id: str) -> Path:
    """A built-in checkpoint's verified directory, for trusted local use."""
    return _artifacts().checkpoint(_catalog.ENTRIES[id])


def _artifacts() -> Any:
    from generalforecast.models import artifacts

    return artifacts


def _native(entry: ModelEntry, surface: Surface, allow_downloads: bool) -> Model:
    """A built-in model's contract calls and verified, traced parameters."""
    loader, functions = _family(entry.family)
    directory = _artifacts().checkpoint(
        entry, surface=surface, allow_downloads=allow_downloads
    )
    parameters = loader(directory, **dict(entry.options))
    if surface != "local":
        import jax

        from generalforecast.runtime.device import configure, device_settings

        parameters = configure(parameters, device_settings(jax.default_backend()))
    return _trained(entry, _traced(entry.id, (functions, parameters)))


# Predictor and remote processes share one verified copy of each model, and one
# set of wrapped contract calls per variant, which compiles once.
@lru_cache(maxsize=4)
def _served(id: str, surface: Surface, allow_downloads: bool) -> Model:
    return _native(_catalog.ENTRIES[id], surface, allow_downloads)


@lru_cache(maxsize=16)
def _variant(
    id: str,
    surface: Surface,
    allow_downloads: bool,
    codes: int | None,
    estimators: int | None,
) -> Model:
    functions, parameters = _served(id, surface, allow_downloads)
    return wrap(functions, parameters, ModelRef(id, codes, estimators))


def load(
    spec: str, *, surface: Surface = "local", allow_downloads: bool = True
) -> Model:
    """Loads a model's contract calls and parameters.

    Args:
        spec: A built-in model's canonical id, such as
            `general-forecast/tabicl-v2-classifier`; then `@K` for K error-correcting code
            estimators, which serve more classes than the classifier has
            slots, then `*N` for an ensemble of N estimators: each at most
            once, in that order. A forecaster takes neither. Or a custom
            model's `module:attribute`, on the local surface alone.
        surface: Where the model runs. Local use takes checkpoints without
            recorded hashes; predictors and remote serving
            need the model's qualification gates, every pinned file and the
            serving policy's bounds on codes and estimators.
        allow_downloads: Whether missing pinned files may download; the first
            download reports one `Pulling` line on stderr.

    Returns:
        The model's contract calls and parameters.

    Raises:
        ValueError: If the spec is malformed, names no model or one by an old
            spelling or one not qualified where it runs, or puts
            error-correcting codes on a regressor or estimators on a
            forecaster.
    """
    return load_ref(
        _catalog.parse(spec),
        surface=surface,
        allow_downloads=allow_downloads,
    )


def load_ref(
    ref: ModelRef, *, surface: Surface = "local", allow_downloads: bool = True
) -> Model:
    """Loads a parsed model reference, as `load` does a spec."""
    _catalog.admit(ref, surface)
    if ref.custom:
        from generalforecast.models import custom

        return custom.load(ref.name)
    entry = ref.entry
    if surface != "local":
        return _variant(entry.id, surface, allow_downloads, ref.codes, ref.estimators)
    functions, parameters = _native(entry, surface, allow_downloads)
    return wrap(functions, parameters, ref)


def load_engine_model(spec: str) -> Model:
    """Loads what the engine's workers were started with.

    A spec is a catalog id, a custom model's `module:attribute` or a
    registered implementation's id (see `register`). A custom model is
    checked against its contract as it loads, so one the engine would run
    wrongly fails to load instead.
    """
    loader = IMPLEMENTATIONS.get(spec)
    if loader is not None:
        functions, parameters = loader()
        return wrap(functions, parameters, ModelRef(spec, None, None, None))
    ref = _catalog.parse(spec)
    if ref.custom:
        from generalforecast.models import custom

        return custom.checked(spec)
    return load_ref(ref)


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


# The built-in models, by canonical id.
BUILT_IN: dict[str, Callable[[], Model]] = {
    entry.id: partial(load, entry.id) for entry in _catalog.entries()
}


def _trained(entry: ModelEntry, model: Model) -> Model:
    """A built-in model whose properties say the largest contexts it was trained on."""
    functions, parameters = model
    capability = entry.capability
    if isinstance(functions, SeriesFunctions) or capability.context_rows is None:
        return model

    def properties(parameters: Any) -> Any:
        return dataclasses.replace(
            functions.properties(parameters),
            context_rows=capability.context_rows,
            context_features=capability.context_features,
            context_classes=capability.context_classes,
        )

    return functions._replace(properties=properties), parameters


def _traced(id: str, model: Model) -> Model:
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
            f"checkpoint {id!r} has weights that do not fit its settings: {error}"
        ) from None
    return model
