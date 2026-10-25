# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Models API forecasting for every catalog forecaster, over shared aligned groups.

The requested reference picks the model; its quantiles, patch and row limit
come from the loaded weights, and provenance names the model that ran.
"""

from collections.abc import Sequence
from typing import Literal

from generalforecast._core import ForecastRequest
from generalforecast.engine import Forecasts
from generalforecast.models.catalog import ModelEntry, ModelRef, ModelRefError
from generalforecast.service.boundary import JsonObject
from generalforecast.service.catalog_artifacts import artifact
from generalforecast.service.contracts import ServiceError

Surface = Literal["local", "remote", "self-hosted"]


def provenance(entry: ModelEntry) -> JsonObject:
    """Identifies the forecaster's pinned upstream bytes and license declaration."""
    return {
        **artifact(
            entry.artifact.repository,
            entry.artifact.revision,
            str(entry.artifact.license),
            dict(entry.artifact.files),
        ),
        "license_source": entry.artifact.license_source,
    }


def _unsupported(levels: Sequence[float], available: Sequence[float]) -> None:
    if not set(levels) <= set(available):
        raise ServiceError(
            "UnsupportedQuantiles",
            "the model only returns its own published quantile levels",
            422,
        )


def run(
    ref: ModelRef,
    request: ForecastRequest,
    *,
    surface: Surface = "remote",
    allow_downloads: bool = False,
) -> Forecasts:
    """A validated request's forecasts from the process engine, refused as the Models API answers."""
    from generalforecast import _core, engine
    from generalforecast.models.artifacts import ArtifactError

    try:
        return engine.local(surface, allow_downloads).forecast(ref, request)
    except ArtifactError as error:
        raise ServiceError(error.code, str(error), 503) from None
    except ModelRefError as error:
        raise ServiceError(error.code, str(error), 422) from None
    except engine.ModelOutputError:
        raise ServiceError(
            "ModelOutputError", "native forecaster returned invalid forecasts", 503
        ) from None
    except _core.RequestError as error:
        raise ServiceError(*error.args) from None


def checked(
    ref: ModelRef,
    request: ForecastRequest,
    *,
    surface: Surface = "remote",
    allow_downloads: bool = False,
) -> Forecasts:
    """A request's forecasts, at levels the model outputs and the request selects.

    What only the loaded weights know is checked here: a catalog model's
    levels against those it declares, and the request's against them.
    """
    forecasts = run(ref, request, surface=surface, allow_downloads=allow_downloads)
    declared = ref.entry.capability.quantiles
    if declared and forecasts.levels != tuple(declared):
        raise ServiceError(
            "UnsupportedModel", f"{ref.entry.name} checkpoint capabilities changed", 503
        )
    _unsupported(request.levels, forecasts.levels)
    return forecasts
