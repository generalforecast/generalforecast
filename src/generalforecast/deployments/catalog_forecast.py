"""Models API forecasting for every catalog forecaster, over shared aligned groups.

The requested reference picks the model; its quantiles, patch and row limit
come from the loaded weights, and provenance names the model that ran.
"""

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Literal, cast

from generalforecast.deployments.boundary import JsonObject
from generalforecast.deployments.catalog import quantile_levels
from generalforecast.deployments.catalog_artifacts import artifact
from generalforecast.deployments.contracts import (
    ServiceError,
    fields,
    number,
    require,
    text,
)
from generalforecast.deployments.interfaces import InferenceResult
from generalforecast.models import inference
from generalforecast.models.catalog import (
    V1_POLICY,
    ModelEntry,
    ModelRef,
    ModelRefError,
)
from generalforecast.models.grouping import AlignmentError, aligned_group
from generalforecast.models.outputs import quantile_index

Surface = Literal["remote", "self-hosted"]


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


def validate(
    request: JsonObject, quantiles: Sequence[float] = ()
) -> tuple[JsonObject, dict[str, dict[int | float, JsonObject]]]:
    """Rejects leakage, mixed time axes and invalid known-future shapes.

    `quantiles` are the model's levels when they are known before loading it.
    """
    require(
        not request["features"], "forecast uses series covariates, not tabular features"
    )
    fields(request["options"], {"series"})
    spec = fields(
        request["options"]["series"],
        {"step", "horizon", "quantiles", "known_covariates"},
    )
    step = number(spec["step"], 0, 1e12, "step")
    require(step > 0, "step must be positive")
    horizon = spec["horizon"]
    require(
        type(horizon) is int and 1 <= horizon <= V1_POLICY.horizon,
        f"horizon must be 1..{V1_POLICY.horizon}",
    )
    levels = quantile_levels(spec["quantiles"])
    if quantiles:
        _unsupported(levels, quantiles)
    known = spec["known_covariates"]
    require(
        type(known) is list and len(known) <= V1_POLICY.covariates,
        "invalid known_covariates",
    )
    for covariate in known:
        text(covariate, "known covariate")
    require(len(known) == len(set(known)), "known covariates must be unique")
    require(request["target"] not in known, "target cannot be a known covariate")
    series: dict[str, dict[int | float, JsonObject]] = {}
    context_keys = set()
    clock = None
    for row in request["context"]:
        fields(row, {"key", "series", "time", "value", "covariates"})
        key = text(row["key"], "key")
        require(key not in context_keys, "duplicate context key")
        context_keys.add(key)
        identity = text(row["series"], "series")
        instant = number(row["time"], -1e12, 1e12, "time")
        if clock is None:
            clock = type(instant)
        require(
            type(instant) is clock,
            "cannot mix integer periods and floating UTC seconds",
        )
        points = series.setdefault(identity, {})
        require(instant not in points, "duplicate series time")
        points[instant] = row
        if row["value"] is not None:
            number(row["value"], -1e12, 1e12, "value")
        require(
            type(row["covariates"]) is dict
            and len(row["covariates"]) <= V1_POLICY.covariates,
            "invalid covariates",
        )
        for covariate in row["covariates"]:
            text(covariate, "covariate")
        for value in row["covariates"].values():
            if value is not None:
                number(value, -1e12, 1e12, "covariate")
    covariates = set(known) | {
        key
        for points in series.values()
        for row in points.values()
        for key in row["covariates"]
    }
    require(
        len(covariates) <= V1_POLICY.covariates,
        f"at most {V1_POLICY.covariates} covariates",
    )
    require(
        not covariates & (series.keys() | {request["target"]}),
        "target series cannot be covariates",
    )
    require(
        clock is not int or type(step) is int, "integer periods require an integer step"
    )
    if not any(
        row["value"] is not None
        for points in series.values()
        for row in points.values()
    ):
        raise ServiceError(
            "EmptyContext", "no usable series observations; no forecast", 503
        )
    query_keys = set()
    for query in request["queries"]:
        fields(query, {"key", "series", "origin", "group", "future_covariates"})
        key = text(query["key"], "key")
        require(key not in query_keys, "duplicate query key")
        query_keys.add(key)
        identity = text(query["series"], "series")
        origin = number(query["origin"], -1e12, 1e12, "origin")
        require(type(origin) is clock, "query and context must use the same time axis")
        times = [origin + step * index for index in range(horizon + 1)]
        require(
            all(later > earlier for earlier, later in pairwise(times)),
            "step is below the time axis precision",
        )
        group = query["group"]
        require(
            type(group) is list and 1 <= len(group) <= V1_POLICY.group_series,
            f"group must contain 1..{V1_POLICY.group_series} series",
        )
        for member in group:
            text(member, "group series")
        require(
            len(set(group)) == len(group) and identity in group, "invalid query group"
        )
        require(set(group) <= series.keys(), "group series missing from context")
        for member in group:
            points = series[member]
            require(
                all(instant <= origin for instant in points), "future target leakage"
            )
            positions = [(origin - instant) / step for instant in points]
            require(
                all(
                    math.isfinite(position) and 0 <= position < V1_POLICY.context_rows
                    for position in positions
                ),
                "series span exceeds context bound",
            )
            require(
                all(abs(position - round(position)) < 1e-6 for position in positions),
                "series times must align to step and origin",
            )
            if not any(row["value"] is not None for row in points.values()):
                raise ServiceError(
                    "EmptyContext", "query group has no usable series observations", 503
                )
        future = fields(query["future_covariates"], set(known))
        for values in future.values():
            require(
                type(values) is list and len(values) == horizon,
                "known-future values must cover the complete horizon",
            )
            for value in values:
                number(value, -1e12, 1e12, "future covariate")
    return spec, series


def model(
    ref: ModelRef, *, surface: Surface = "remote", allow_downloads: bool = False
) -> Any:
    """The requested forecaster, verified against its pins for the surface."""
    from generalforecast.models import registry
    from generalforecast.models.artifacts import ArtifactError

    try:
        loaded = registry.load_ref(
            ref, surface=surface, allow_downloads=allow_downloads
        )
    except ArtifactError as error:
        raise ServiceError(error.code, str(error), 503) from None
    except ModelRefError as error:
        raise ServiceError(error.code, str(error), 422) from None
    functions, parameters = loaded
    declared = ref.entry.capability.quantiles
    if declared and tuple(cast(Any, functions.properties(parameters)).quantiles) != (
        declared
    ):
        raise ServiceError(
            "UnsupportedModel", f"{ref.entry.name} checkpoint capabilities changed", 503
        )
    return loaded


def infer(
    request: JsonObject,
    *,
    ref: ModelRef,
    surface: Surface = "remote",
    allow_downloads: bool = False,
) -> InferenceResult:
    """Forecasts related targets and shared past/known-future covariates."""
    spec, series = validate(request, ref.entry.capability.quantiles)

    def groups_for(patch: int) -> list:
        return [
            aligned_group(
                series,
                query["group"],
                spec["known_covariates"],
                query["origin"],
                spec["step"],
                spec["horizon"],
                query["future_covariates"],
                patch,
            )
            for query in request["queries"]
        ]

    try:
        # Alignment fails before any weights load; the model's patch pads time.
        groups_for(1)
    except AlignmentError as error:
        raise ServiceError(error.code, str(error), 422) from None
    try:
        loaded = model(ref, surface=surface, allow_downloads=allow_downloads)
        properties = loaded[0].properties(loaded[1])
        levels = tuple(properties.quantiles)
        _unsupported(spec["quantiles"], levels)
        groups = groups_for(properties.patch)
        limit = getattr(properties, "max_rows", None)
        if limit is not None and any(g.values.shape[0] > limit for g in groups):
            raise ServiceError(
                "CapacityExceeded", f"{ref.entry.name} reads at most {limit} rows", 422
            )
        outputs = []
        for query, group in zip(request["queries"], groups, strict=True):
            try:
                result = inference.forecast(loaded, group, spec["horizon"])
            except ValueError:
                raise ServiceError(
                    "ModelOutputError",
                    "native forecaster returned invalid forecasts",
                    503,
                ) from None
            target = result[query["group"].index(query["series"])]
            outputs.append(
                {
                    "key": query["key"],
                    "series": query["series"],
                    "origin": query["origin"],
                    "times": [
                        query["origin"] + spec["step"] * (index + 1)
                        for index in range(spec["horizon"])
                    ],
                    "quantiles": {
                        str(level): target[:, quantile_index(level, levels)].tolist()
                        for level in spec["quantiles"]
                    },
                }
            )
    except ServiceError:
        raise
    except Exception:  # noqa: BLE001
        raise ServiceError(
            "InferenceFailed", "native forecast inference failed", 503
        ) from None
    return {
        "task": "forecast",
        "outputs": outputs,
        "artifact": {**provenance(ref.entry), "verified": True},
        "context_provenance": {
            "source": "request",
            "supplied_rows": len(request["context"]),
            "usable_observations": sum(
                row["value"] is not None
                for points in series.values()
                for row in points.values()
            ),
            "persisted": False,
            "covariate_policy": "shared-within-query-group",
            "time_axis": "integer-periods"
            if type(request["context"][0]["time"]) is int
            else "utc-seconds",
        },
    }
