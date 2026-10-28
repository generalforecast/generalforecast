# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The models the conformance suite runs on, each a small random instance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from generalforecast.models.conformance import Table, predict, table
from generalforecast.models.contract import TabularFunctions
from generalforecast.models.series import SeriesFunctions

__all__ = [
    "CASES",
    "FEATURES",
    "RUNTIME_CASES",
    "SERIES_CASES",
    "Case",
    "SeriesCase",
    "Table",
    "predict",
    "table",
]

FEATURES = 5


class Case(NamedTuple):
    """A model to check: its contract calls and a builder for random parameters.

    `cache_drift` bounds how far a float16, bfloat16 or int8 cache may move
    predictions.
    """

    name: str
    regression: bool
    functions: Callable[[], TabularFunctions]
    build: Callable[[int], Any]
    cache_drift: float = 5e-3


def _tabicl_functions() -> TabularFunctions:
    from generalforecast.models.tabicl.model import tabicl_functions

    return tabicl_functions


def _tabicl(regression: bool) -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from tests.models.tabicl.tiny import tiny_tabicl

        return tiny_tabicl(regression=regression, seed=seed)

    return build


def _retrieval(seed: int) -> Any:
    """A tiny TabICL classifier with a retrieval decoder in place of its MLP."""
    import dataclasses

    from generalforecast.nn.linear import LinearParameters, MLPParameters
    from generalforecast.nn.qassmax import QASSMaxParameters
    from generalforecast.nn.retrieval import RetrievalParameters
    from tests.models.tabicl.tiny import ICL, tiny_tabicl

    model = tiny_tabicl(regression=False, seed=seed)
    generator = np.random.default_rng(seed + 1)
    heads, head_dim, hidden = 2, 8, 4

    def linear(outputs: int, inputs: int) -> LinearParameters:
        weight = generator.normal(size=(outputs, inputs)) / np.sqrt(inputs)
        bias = 0.1 * generator.normal(size=outputs)
        return LinearParameters(
            jnp.asarray(weight, jnp.float32), jnp.asarray(bias, jnp.float32)
        )

    decoder = RetrievalParameters(
        query=linear(heads * head_dim, ICL),
        key=linear(heads * head_dim, ICL),
        num_heads=heads,
        scale=QASSMaxParameters(
            base=MLPParameters(linear(hidden, 1), linear(heads * head_dim, hidden)),
            query=MLPParameters(linear(hidden, head_dim), linear(head_dim, hidden)),
        ),
    )
    return model._replace(icl=dataclasses.replace(model.icl, decoder=decoder))


def _tabpfn_functions() -> TabularFunctions:
    from generalforecast.models.tabpfn.model import tabpfn_functions

    return tabpfn_functions


def _tabpfn(regression: bool, release: str = "3") -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from tests.models.tabpfn.tiny import tiny_tabpfn

        return tiny_tabpfn(regression=regression, seed=seed, release=release)

    return build


def _kumo_functions() -> TabularFunctions:
    from generalforecast.models.kumo.model import kumo_functions

    return kumo_functions


def _kumo(regression: bool) -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from tests.models.kumo.tiny import tiny_kumo

        return tiny_kumo(regression=regression, seed=seed)

    return build


def _limix_functions() -> TabularFunctions:
    from generalforecast.models.limix.model import limix_functions

    return limix_functions


def _limix(regression: bool) -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from tests.models.limix.tiny import tiny_limix

        return tiny_limix(regression=regression, seed=seed)

    return build


def _mitra_functions() -> TabularFunctions:
    from generalforecast.models.mitra.model import mitra_functions

    return mitra_functions


def _mitra(regression: bool) -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from tests.models.mitra.tiny import tiny_mitra

        return tiny_mitra(seed=seed, regression=regression)

    return build


class SeriesCase(NamedTuple):
    """A forecaster to check: its contract calls, a builder for random parameters, and how far its results may move between equivalent calls."""

    name: str
    functions: Callable[[], SeriesFunctions]
    build: Callable[[int], Any]
    tolerance: float = 1e-5


def _chronos2_functions() -> SeriesFunctions:
    from generalforecast.models.chronos2.model import chronos2_functions

    return chronos2_functions


def _chronos2(seed: int) -> Any:
    from tests.models.chronos2.tiny import tiny_chronos2

    return tiny_chronos2(seed=seed)


def _ensemble(
    functions: Callable[[], TabularFunctions], build: Callable[[int], Any]
) -> tuple[Callable[[], TabularFunctions], Callable[[int], Any]]:
    """Three estimators of a model."""
    from generalforecast.models.ensemble import Ensemble, ensemble_functions

    return lambda: ensemble_functions(functions()), lambda seed: Ensemble(
        build(seed), 3
    )


def _many_class(
    functions: Callable[[], TabularFunctions], build: Callable[[int], Any]
) -> tuple[Callable[[], TabularFunctions], Callable[[int], Any]]:
    """Four error-correcting code estimators of a classifier."""
    from generalforecast.models.many_class import ManyClass, many_class_functions

    return (
        lambda: many_class_functions(functions()),
        lambda seed: ManyClass(build(seed), 4),
    )


CASES = [
    Case("tabicl-classes", False, _tabicl_functions, _tabicl(False)),
    Case("tabicl-values", True, _tabicl_functions, _tabicl(True)),
    Case("retrieval-classes", False, _tabicl_functions, _retrieval),
    Case("tabpfn-classes", False, _tabpfn_functions, _tabpfn(False)),
    Case("tabpfn-values", True, _tabpfn_functions, _tabpfn(True)),
    # Int8 also quantizes the retrieval decoder's keys, which TabPFN keeps exact;
    # on the real TabPFN-3.5 classifier that moves probabilities by about 0.008.
    Case("tabpfn-3.5-classes", False, _tabpfn_functions, _tabpfn(False, "3.5"), 1e-2),
    Case("tabpfn-3.5-values", True, _tabpfn_functions, _tabpfn(True, "3.5")),
    Case("kumo-classes", False, _kumo_functions, _kumo(False)),
    Case("kumo-values", True, _kumo_functions, _kumo(True)),
    Case("limix-classes", False, _limix_functions, _limix(False)),
    Case("limix-values", True, _limix_functions, _limix(True)),
    Case("mitra-classes", False, _mitra_functions, _mitra(False)),
    Case("mitra-values", True, _mitra_functions, _mitra(True)),
    Case("ensemble-classes", False, *_ensemble(_tabicl_functions, _tabicl(False))),
    Case("ensemble-values", True, *_ensemble(_kumo_functions, _kumo(True))),
    Case("many-classes", False, *_many_class(_kumo_functions, _kumo(False))),
]

# The runtime's paging, decoding, worker units and local tables depend on a
# model only through its caches and states, so their tests run on one case of
# each layout: every family, a retrieval decoder, ensembles and
# error-correcting codes, classes and values both.
RUNTIME_CASES = [
    case
    for case in CASES
    if case.name
    in {
        "tabicl-values",
        "retrieval-classes",
        "tabpfn-classes",
        "tabpfn-3.5-values",
        "kumo-values",
        "limix-classes",
        "mitra-values",
        "ensemble-classes",
        "many-classes",
    }
]


def _timestral_functions() -> SeriesFunctions:
    from generalforecast.models.timestral.model import timestral_functions

    return timestral_functions


def _timestral(symmetric: bool) -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from tests.models.timestral.tiny import tiny_timestral

        return tiny_timestral(seed=seed, symmetric=symmetric)

    return build


def _timesfm_functions() -> SeriesFunctions:
    from generalforecast.models.timesfm3.model import timesfm_functions

    return timesfm_functions


def _timesfm(seed: int) -> Any:
    from tests.models.timesfm3.tiny import tiny_timesfm

    return tiny_timesfm(seed=seed)


SERIES_CASES = [
    # Under jaxlib 0.12, padding moves one value of the tiny random Chronos-2
    # by 1.2e-5 on a CPU, past the default by rounding alone.
    SeriesCase("chronos2", _chronos2_functions, _chronos2, tolerance=2e-5),
    SeriesCase("timesfm3", _timesfm_functions, _timesfm),
    SeriesCase("timestral", _timestral_functions, _timestral(True)),
    SeriesCase("timestral-plain", _timestral_functions, _timestral(False)),
]


def __getattr__(name: str) -> Callable[[], tuple[TabularFunctions, Any]]:
    """Each case at seed 0 as a model spec target, e.g. `tests.models.conformance.cases:kumo_values`."""
    for case in CASES:
        if case.name.replace("-", "_").replace(".", "_") == name:
            return lambda: (case.functions(), case.build(0))
    raise AttributeError(name)
