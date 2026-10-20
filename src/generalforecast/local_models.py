"""Native caller-context inference without management or persistent writes."""

from typing import Any

from generalforecast.deployments.boundary import Request, Resource, Scope
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.interfaces import InferenceRequest, InferenceResult
from generalforecast.models.catalog import resource_name


def native_model(model: Any, task: str, *, allow_downloads: bool = False) -> Any:
    """Loads a built-in model by name, or uses explicit native model functions.

    Names, aliases and `@K`/`*N` variants resolve offline through the shared
    catalog, and pinned files are verified before loading; downloads happen
    only when requested. Custom native tuples are local only.
    """
    if not isinstance(model, str):
        return model
    from generalforecast.models import registry
    from generalforecast.models.artifacts import ArtifactError
    from generalforecast.models.catalog import ModelRefError, parse
    from generalforecast.packages.validation import PackageError, ensure

    try:
        ref = parse(model)
        ensure(
            ref.entry.capability.task == task, "model does not support estimator task"
        )
        return registry.load_ref(ref, allow_downloads=allow_downloads)
    except (ModelRefError, ArtifactError) as error:
        raise PackageError(error.code, str(error)) from None


class LocalModels:
    """Runs the SAME catalog port as the remote Models API."""

    def __init__(self, *, allow_downloads: bool = False) -> None:
        """Defaults to verified cached weights; downloads require explicit opt-in."""
        self._catalog = CatalogService(allow_downloads=allow_downloads)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Validates the shared envelope, then invokes genuine native inference."""
        checked = Request.parse(
            Request(
                Scope("local", "local-inference"),
                Resource("model", resource_name(request.get("model", ""))),
                dict(request),
            ).to_dict()
        )
        return self._catalog.infer(checked.scope, request)

    def list(self) -> list[dict]:
        """Lists curated metadata without downloading weights."""
        return self._catalog.list(Scope("local", "local-models"))
