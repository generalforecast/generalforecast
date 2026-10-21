"""Bounded request-context Models API; no management or ingestion dependency."""

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal, cast

from generalforecast.deployments.boundary import (
    Capability,
    JsonObject,
    Scope,
    json_object,
)
from generalforecast.deployments.catalog_artifacts import artifact
from generalforecast.deployments.contracts import (
    ServiceError,
    fields,
    number,
    require,
    text,
)
from generalforecast.deployments.interfaces import InferenceRequest, InferenceResult
from generalforecast.models.catalog import (
    V1_POLICY,
    ModelEntry,
    ModelRef,
    ModelRefError,
    entries,
    parse,
)
from generalforecast.models.outputs import (
    DEFAULT_QUANTILES,
    class_order,
    probabilities,
    quantile_index,
    quantiles,
)

MAX_CONTEXT = V1_POLICY.context_rows
MAX_QUERIES = V1_POLICY.query_rows
MAX_FEATURES = V1_POLICY.features


ServingSurface = Literal["remote", "self-hosted"]


def _ref(model: str, surface: ServingSurface = "remote") -> ModelRef:
    """A model reference the surface may serve, never code or an unqualified model.

    Hosted serving takes remote-qualified models. A self-hosted server also
    takes a pinned model whose license only its operator can grant.
    """
    text(model, "model")
    try:
        ref = parse(model)
    except ModelRefError as error:
        if error.code in ("UnknownModel", "UntrustedModel", "InvalidModel"):
            raise ServiceError(
                "ModelNotFound", "model is not in the curated catalog", 404
            ) from None
        raise ServiceError(error.code, str(error), 422) from None
    gates = ref.entry.gates
    if surface == "self-hosted" and not ref.entry.hostable:
        if ref.variant:
            raise ServiceError(
                "UnsupportedCapability", f"{ref} variants are not qualified", 422
            )
        if not ref.entry.allows("self-hosted"):
            raise ServiceError(
                "UnsupportedModel", f"{ref.entry.name} has no pinned files", 422
            )
    else:
        if not ref.entry.hostable:
            raise ServiceError(
                "UnsupportedModel",
                f"{ref.entry.name} is local-only: its license does not permit hosted serving",
                422,
            )
        if not gates.allows("remote"):
            raise ServiceError(
                "ModelNotFound", "model is not qualified for remote serving", 404
            )
        if not gates.allows("remote", variant=ref.variant):
            raise ServiceError(
                "UnsupportedCapability",
                f"{ref} variants are not qualified remotely",
                422,
            )
    if (refused := V1_POLICY.admits(ref)) is not None:
        raise ServiceError("CapacityExceeded", f"serving policy allows {refused}", 422)
    return ref


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
        value, model = self.check(scope, request)
        if value["task"] == "forecast":
            from generalforecast.deployments.catalog_forecast import infer

            return infer(
                cast(JsonObject, value),
                ref=_ref(value["model"], self.surface),
                surface=self.surface,
                allow_downloads=self.allow_downloads,
            )
        return self._tabular(value, model)

    def check(
        self, scope: Scope, request: InferenceRequest
    ) -> tuple[InferenceRequest, JsonObject]:
        """The request's shape and bounds, and its model's metadata, before any load."""
        require(isinstance(scope, Scope), "expected validated scope")
        value = json_object(request)
        fields(
            value,
            {"model", "task", "target", "features", "context", "queries", "options"},
        )
        model = self.get(scope, value["model"])
        require(
            value["task"] in ("classification", "regression", "forecast"),
            "invalid task",
        )
        if value["task"] != model["task"]:
            raise ServiceError(
                "UnsupportedTask", "selected model does not support this task", 422
            )
        text(value["target"], "target")
        features = value["features"]
        require(
            type(features) is list and len(features) <= MAX_FEATURES,
            "invalid feature list",
        )
        for feature in features:
            text(feature, "feature")
        require(len(set(features)) == len(features), "features must be unique")
        require(value["target"] not in features, "target cannot be a feature")
        require(type(value["context"]) is list, "context must be a list")
        require(len(value["context"]) <= MAX_CONTEXT, "context capacity exceeded")
        require(type(value["queries"]) is list, "queries must be a list")
        require(0 < len(value["queries"]) <= MAX_QUERIES, "query capacity exceeded")
        fields(
            value["options"],
            set(),
            {"series"} if value["task"] == "forecast" else {"quantiles"},
        )
        return cast(InferenceRequest, value), model

    def _tabular(self, request: InferenceRequest, model: JsonObject) -> InferenceResult:
        features = request["features"]
        require(bool(features), "tabular inference requires features")
        context = _rows(request["context"], features, scoring=False)
        queries = _rows(request["queries"], features, scoring=True)
        labeled = [row for row in context if "label" in row]
        if not labeled:
            raise ServiceError("EmptyContext", "no usable labeled rows; no score", 503)
        labels = [row["label"] for row in labeled]
        classes: tuple = ()
        if request["task"] == "classification":
            require(not request["options"], "classification has no quantile options")
            kind = type(labels[0])
            require(
                kind in (str, bool, int),
                "classification labels must be strings, booleans or integers",
            )
            require(
                all(type(label) is kind for label in labels),
                "class labels must have one type",
            )
            if kind is str:
                for label in labels:
                    text(label, "label")
            classes = class_order(labels)
            capacity = model["bounds"]["classes"]
            if len(classes) > capacity:
                raise ServiceError(
                    "UnsupportedClassCount",
                    f"this model serves at most {capacity} classes; "
                    "@K error-correcting codes serve more",
                    422,
                )
            targets: list = [classes.index(label) for label in labels]
            levels: list[float] = []
        else:
            targets = [number(label, -1e12, 1e12, "label") for label in labels]
            levels = quantile_levels(
                request["options"].get("quantiles", list(DEFAULT_QUANTILES))
            )
            try:
                for level in levels:
                    quantile_index(level)
            except ValueError as error:
                raise ServiceError("ValidationError", str(error), 422) from None
        ref = _ref(request["model"], self.surface)
        predictions = self._predict(
            ref,
            [[row["features"][feature] for feature in features] for row in labeled],
            targets,
            [[row["features"][feature] for feature in features] for row in queries],
            len(classes),
        )
        outputs = []
        for row, prediction in zip(queries, predictions, strict=True):
            if classes:
                outputs.append(
                    {
                        "key": row["key"],
                        "probabilities": probabilities(classes, prediction),
                    }
                )
            else:
                outputs.append(
                    {"key": row["key"], "quantiles": quantiles(prediction, levels)}
                )
        return {
            "task": request["task"],
            "outputs": outputs,
            "artifact": {**model["artifact"], "verified": True},
            "context_provenance": {
                "source": "request",
                "supplied_rows": len(context),
                "usable_labeled_rows": len(labeled),
                "features": list(features),
                "target": request["target"],
                "persisted": False,
                "classes": list(classes),
            },
        }

    def _predict(
        self,
        ref: ModelRef,
        features: Sequence[Sequence[float]],
        targets: Sequence[Any],
        queries: Sequence[Sequence[float]],
        classes: int,
    ) -> Any:
        """Fits and decodes genuine supplied context, retaining only model parameters."""
        import jax.numpy as jnp

        from generalforecast.models import inference, registry
        from generalforecast.models.artifacts import ArtifactError

        try:
            loaded = cast(
                "registry.TabularModel",
                registry.load_ref(
                    ref, surface=self.surface, allow_downloads=self.allow_downloads
                ),
            )
        except ArtifactError as error:
            raise ServiceError(error.code, str(error), 503) from None
        except ModelRefError as error:
            raise ServiceError(error.code, str(error), 422) from None
        try:
            fitted = inference.fit(
                loaded,
                jnp.asarray(features, dtype=jnp.float32),
                jnp.asarray(targets, dtype=jnp.int32 if classes else jnp.float32),
                classes,
            )
        except ValueError as error:
            raise ServiceError("UnsupportedClassCount", str(error), 422) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native inference failed", 503
            ) from None
        try:
            return inference.predict(
                loaded, fitted, jnp.asarray(queries, dtype=jnp.float32), classes
            ).tolist()
        except ValueError:
            raise ServiceError(
                "ModelOutputError", "native model returned invalid outputs", 503
            ) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native inference failed", 503
            ) from None


def create_catalog() -> CatalogService:
    """Zero-arg provider factory with explicitly authorized pinned model downloads."""
    return CatalogService(allow_downloads=True)


def quantile_levels(value: object) -> list[float]:
    """Validates requested quantile levels: strictly increasing, inside (0, 1)."""
    require(type(value) is list and 0 < len(value) <= 99, "invalid quantile levels")
    levels = [number(level, 0, 1, "quantile") for level in cast(list, value)]
    require(all(0 < level < 1 for level in levels), "quantiles must be within (0,1)")
    require(levels == sorted(set(levels)), "quantiles must strictly increase")
    return levels


def _rows(
    rows: list[JsonObject], features: list[str], *, scoring: bool
) -> list[JsonObject]:
    keys = set()
    for row in rows:
        fields(row, {"key", "features"}, set() if scoring else {"label", "event_at"})
        key = text(row["key"], "key")
        require(key not in keys, "duplicate row key")
        keys.add(key)
        fields(row["features"], set(features))
        for value in row["features"].values():
            number(value, -1e12, 1e12, "feature")
        if "event_at" in row:
            number(row["event_at"], -1e12, 1e12, "event_at")
    return rows
