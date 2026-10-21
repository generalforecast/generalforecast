"""Bounded request-context Models API; no management or ingestion dependency."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal, cast

from generalforecast import _core
from generalforecast._core import ForecastRequest, TabularRequest
from generalforecast.deployments.boundary import (
    Capability,
    JsonObject,
    Scope,
    json_bytes,
)
from generalforecast.deployments.catalog_artifacts import artifact
from generalforecast.deployments.contracts import ServiceError, require
from generalforecast.deployments.interfaces import InferenceRequest, InferenceResult
from generalforecast.models.catalog import (
    V1_POLICY,
    ModelEntry,
    ModelRef,
    ModelRefError,
    entries,
    parse,
)
from generalforecast.models.outputs import probabilities, quantiles

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


def inference_request(
    request: object, surface: ServingSurface
) -> TabularRequest | ForecastRequest:
    """A Models API request, validated by `gf_core::request` before any load.

    The single validator every surface shares: shape, bounds, the model's
    gates and capability, class order, quantile levels and aligned forecast
    groups, with the Models API's error codes, messages and statuses.
    """
    payload = json_bytes(request)
    try:
        return _core.parse_inference(payload, surface)
    except _core.RequestError as error:
        raise _refused(error) from None


def forecast_request(
    request: JsonObject,
    levels: Sequence[float],
    admission: Literal["catalog", "trusted"] = "catalog",
    digest: str | None = None,
) -> ForecastRequest:
    """A package's or estimator's bound forecast context, validated as a request is.

    The package's lock admitted its model and bounds its context, so neither
    the serving gates nor a request's size bounds apply; `levels` are those
    the loaded forecaster outputs. A `trusted` admission also takes the
    caller's own `module:function` forecaster, at `levels`, whose code
    `digest` names.
    """
    try:
        payload = json.dumps(request, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError):
        raise ServiceError("ValidationError", "expected JSON value", 422) from None
    try:
        parsed = _core.parse_inference(
            payload.encode("utf-8"),
            "local",
            limits="package",
            levels=list(levels),
            admission=admission,
            digest=digest,
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
            self.get(scope, entry.name)
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
            "name": entry.name,
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
                "covariate_policy": "shared-within-query-group",
                "time_axis": "all integer periods or all floating UTC seconds",
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

    def infer(self, scope: Scope, request: InferenceRequest) -> InferenceResult:
        """Validates before model access and never writes context, query or scores."""
        require(isinstance(scope, Scope), "expected validated scope")
        parsed = inference_request(request, self.surface)
        model = self.get(scope, parsed.model)
        if isinstance(parsed, ForecastRequest):
            from generalforecast.deployments.catalog_forecast import infer

            return infer(
                parsed,
                ref=parse(parsed.model),
                surface=self.surface,
                allow_downloads=self.allow_downloads,
            )
        return self._tabular(parsed, model)

    def check(
        self, scope: Scope, request: InferenceRequest
    ) -> tuple[InferenceRequest, JsonObject]:
        """The request's shape and bounds, and its model's metadata, before any load."""
        require(isinstance(scope, Scope), "expected validated scope")
        parsed = inference_request(request, self.surface)
        return (
            cast(InferenceRequest, json.loads(json_bytes(request))),
            self.get(scope, parsed.model),
        )

    def _tabular(self, request: TabularRequest, model: JsonObject) -> InferenceResult:
        """Scores a validated request's queries against its labeled context."""
        classes = request.classes
        predictions = self._run(parse(request.model), request)
        outputs = []
        for key, prediction in zip(request.query_keys, predictions, strict=True):
            if request.task == "classification":
                outputs.append(
                    {"key": key, "probabilities": probabilities(classes, prediction)}
                )
            else:
                outputs.append(
                    {"key": key, "quantiles": quantiles(prediction, request.levels)}
                )
        return {
            "task": request.task,
            "outputs": outputs,
            "artifact": {**model["artifact"], "verified": True},
            "context_provenance": {
                "source": "request",
                "supplied_rows": request.supplied_rows,
                "usable_labeled_rows": len(request.context),
                "features": list(request.features),
                "target": request.target,
                "persisted": False,
                "classes": list(classes),
            },
        }

    def _run(self, ref: ModelRef, request: TabularRequest) -> Any:
        """The model's checked outputs for a validated request, from the process engine."""
        from generalforecast import _core, engine
        from generalforecast.models.artifacts import ArtifactError

        try:
            return engine.local(self.surface, self.allow_downloads).tabular(
                ref, request
            )
        except ArtifactError as error:
            raise ServiceError(error.code, str(error), 503) from None
        except ModelRefError as error:
            raise ServiceError(error.code, str(error), 422) from None
        except engine.ModelOutputError:
            raise ServiceError(
                "ModelOutputError", "native model returned invalid outputs", 503
            ) from None
        except _core.RequestError as error:
            raise ServiceError(*error.args) from None
        except ValueError as error:
            raise ServiceError("UnsupportedClassCount", str(error), 422) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native inference failed", 503
            ) from None


def create_catalog() -> CatalogService:
    """Zero-arg provider factory with explicitly authorized pinned model downloads."""
    return CatalogService(allow_downloads=True)


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
