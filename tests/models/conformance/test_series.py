"""Contract checks every forecaster must pass, run on small random instances."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from generalforecast.models.conformance import check_series, group
from generalforecast.models.series import SeriesFunctions
from generalforecast.nn.attention import AttentionKernel
from generalforecast.runtime.device import DeviceSettings, configure
from tests.models.conformance.cases import SERIES_CASES, SeriesCase

FLOAT16 = jnp.dtype("float16")


@pytest.fixture(params=SERIES_CASES, ids=[case.name for case in SERIES_CASES])
def case(request: pytest.FixtureRequest) -> SeriesCase:
    return request.param


@pytest.fixture
def functions(case: SeriesCase) -> SeriesFunctions:
    return case.functions()


@pytest.fixture
def model(case: SeriesCase) -> Any:
    return case.build(0)


def test_functions_follow_the_contract(functions: SeriesFunctions) -> None:
    assert isinstance(functions, SeriesFunctions)


def test_models_pass_the_conformance_check(
    functions: SeriesFunctions, model: Any
) -> None:
    check_series(functions, model)


def test_groups_in_one_bucket_share_one_compile(
    functions: SeriesFunctions, model: Any
) -> None:
    traces: list[object] = []
    patch = functions.properties(model).patch

    def pipeline(model: Any, data: Any) -> jax.Array:
        traces.append(data.num_rows)
        return functions.forecast(model, data, patch)

    compiled = jax.jit(pipeline)
    for rows in (2, 3):
        data = group(rows=3, steps=4 * patch, horizon=patch)._replace(num_rows=rows)
        compiled(model, data)

    assert len(traces) == 1


def test_parameters_set_the_compute_dtype(
    functions: SeriesFunctions, model: Any
) -> None:
    half = configure(model, DeviceSettings(FLOAT16, FLOAT16, AttentionKernel()))
    patch = functions.properties(model).patch
    data = group(steps=4 * patch, horizon=patch)

    predictions = functions.forecast(half, data, patch)

    assert predictions.dtype == jnp.float32
    assert bool(jnp.isfinite(predictions).all())
