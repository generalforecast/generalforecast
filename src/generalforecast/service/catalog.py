# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded request-context Models API; no management or ingestion dependency."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Literal, cast

from generalforecast import _core
from generalforecast._core import ForecastRequest
from generalforecast.models.catalog import (
    V1_POLICY,
    ModelEntry,
    ModelRef,
    entries,
    parse,
)
from generalforecast.service.boundary import (
    Capability,
    JsonObject,
    Scope,
)
from generalforecast.service.catalog_artifacts import artifact
from generalforecast.service.contracts import ServiceError, require

MAX_CONTEXT = V1_POLICY.context_rows
MAX_QUERIES = V1_POLICY.query_rows
MAX_FEATURES = V1_POLICY.features


ServingSurface = Literal["remote", "self-hosted"]


def _refused(error: _core.RequestError) -> ServiceError:
    code, message, status = error.args
    return ServiceError(code, message, status)


def _ref(model: str, surface: ServingSurface = "remote") -> ModelRef:
    """A model reference the surface may serve, never code or an unqualified model.

    Hosted serving takes remote-qualified models. A self-hosted server also
    takes a pinned model whose license only its operator can grant. The gates
    are `gf_core::catalog::resolve`'s, which every surface shares.
    """
    try:
        return parse(_core.resolve_model(model, surface))
    except _core.RequestError as error:
        raise _refused(error) from None


def forecast_request(
    request: JsonObject,
    levels: Sequence[float],
) -> ForecastRequest:
    """A predictor's or estimator's bound forecast context, validated as a request is.

    The predictor's lock admitted its model and bounds its context, so neither
    the serving gates nor a request's size bounds apply; `levels` are those
    the loaded forecaster outputs.
    """
    try:
        payload = json.dumps(request, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError):
        raise ServiceError("ValidationError", "expected JSON value", 422) from None
    try:
        parsed = _core.parse_inference(
            payload.encode("utf-8"),
            "local",
            limits="predictor",
            levels=list(levels),
        )
    except _core.RequestError as error:
        raise _refused(error) from None
    require(isinstance(parsed, ForecastRequest), "expected a forecast request")
    return cast(ForecastRequest, parsed)


def _artifact(entry: ModelEntry) -> JsonObject:
    found = artifact(
        entry.artifact.repository,
        entry.artifact.revision,
        str(entry.artifact.license),
        dict(entry.artifact.files),
    )
    if entry.artifact.license_source and entry.artifact.license_source != "LICENSE":
        found["license_source"] = entry.artifact.license_source
    return found


def _capabilities(task: str) -> list[JsonObject]:
    return [
        asdict(Capability(task, "supported")),
        asdict(Capability("request-context", "supported")),
        asdict(
            Capability(
                "related-series",
                "unsupported",
                "UnsupportedCapability",
                "tabular models do not forecast; related series need a forecaster",
            )
        ),
    ]


class CatalogService:
    """Stateless curated models port; authorization belongs to the route boundary."""

    def __init__(
        self,
        *,
        allow_downloads: bool = False,
        runtime: Literal["cpu", "gpu"] = "cpu",
        surface: ServingSurface = "remote",
    ):
        """Uses cached checkpoints unless an operator explicitly enables downloads.

        `runtime` is what model metadata reports: where inference executes.
        `surface` is "self-hosted" only for a server its operator runs.
        """
        require(type(allow_downloads) is bool, "allow_downloads must be boolean")
        require(runtime in ("cpu", "gpu"), "runtime must be cpu or gpu")
        require(surface in ("remote", "self-hosted"), "invalid serving surface")
        self.allow_downloads, self.runtime = allow_downloads, runtime
        self.surface: ServingSurface = surface

    def list(self, scope: Scope) -> list[JsonObject]:
        """Lists immutable model metadata without downloads or context retention."""
        require(isinstance(scope, Scope), "expected validated scope")
        return [
            self.get(scope, entry.id)
            for entry in entries()
            if entry.gates.allows("remote")
        ]

    def get(self, scope: Scope, model: str) -> JsonObject:
        """Describes support and provenance, not serving readiness or tenancy."""
        require(isinstance(scope, Scope), "expected validated scope")
        ref = _ref(model, self.surface)
        entry = ref.entry
        bounds = {
            "context_rows": MAX_CONTEXT,
            "query_rows": MAX_QUERIES,
            "features": MAX_FEATURES,
        }
        described: JsonObject = {
            "name": entry.id,
            "repository": entry.artifact.repository,
            "revision": entry.artifact.revision,
            "license": entry.artifact.license,
            "task": entry.capability.task,
            "artifact": _artifact(entry),
            "artifact_verification": "required-before-load",
            "runtime": self.runtime,
        }
        if not entry.capability.tabular:
            return {
                **described,
                "license_source": "pinned-model-card",
                "bounds": {
                    **bounds,
                    "horizon": V1_POLICY.horizon,
                    "group_series": V1_POLICY.group_series,
                    "covariates": V1_POLICY.covariates,
                    "quantiles": list(entry.capability.quantiles),
                },
                "capabilities": [
                    asdict(Capability(capability, "supported"))
                    for capability in (
                        "forecast",
                        "related-series",
                        "known-future-covariates",
                        "past-covariates",
                        "request-context",
                    )
                ],
            }
        if ref.variant:
            described["variant"] = {"codes": ref.codes, "estimators": ref.estimators}
        return {
            **described,
            "bounds": {
                **bounds,
                "classes": entry.capacity(ref) or 0,
                "quantile_step": entry.capability.quantile_step,
                "codes": V1_POLICY.max_codes,
                "estimators": V1_POLICY.max_estimators,
                "members": V1_POLICY.max_members,
            },
            "capabilities": _capabilities(entry.capability.task),
        }


def quantile_levels(value: object) -> list[float]:
    """Validates requested quantile levels: strictly increasing, inside (0, 1)."""
    try:
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        raise ServiceError("ValidationError", "invalid quantile levels", 422) from None
    try:
        return _core.quantile_levels(payload)
    except _core.RequestError as error:
        raise _refused(error) from None
