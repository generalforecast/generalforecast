"""What `gf serve` serves: one catalog model or one locked prediction package."""

import threading
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow as pa

from generalforecast.deployments.boundary import JsonObject, Scope
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.contracts import ServiceError, fields, require
from generalforecast.deployments.interfaces import InferenceRequest
from generalforecast.deployments.package_runtime import (
    PackageExecutor,
    PreparedPackage,
)
from generalforecast.deployments.package_runtime.requests import (
    bind_request,
    one_shot_lock,
    prediction,
    request_table,
)
from generalforecast.deployments.package_runtime.validation import (
    MAX_QUERY_ROWS,
    admit,
)
from generalforecast.models.catalog import ModelRef, ModelRefError, parse
from generalforecast.packages import Package, PackageLock, PythonLock
from generalforecast.packages.files import parse_yaml
from generalforecast.packages.validation import PackageError, ensure

MAX_CONTEXT_ROWS = 100_000


class Served(Protocol):
    """One loaded model behind the Models API routes."""

    name: str

    def warm(self) -> None:
        """Loads verified weights and bound context once, before readiness."""
        ...

    def describe(self, scope: Scope) -> JsonObject:
        """The `GET models/{name}` result."""
        ...

    def infer(self, scope: Scope, payload: JsonObject) -> JsonObject:
        """The `POST models/{name}/infer` result."""
        ...

    def predict(self, scope: Scope, payload: JsonObject) -> JsonObject:
        """The `POST predictions` result."""
        ...


def _not_served() -> ServiceError:
    return ServiceError("ModelNotFound", "model is not served here", 404)


class CatalogModel:
    """A curated catalog reference, served by the hosted Models API's own port.

    The operator runs it, so it also serves pinned models whose license
    forbids hosting them on General Forecast infrastructure.
    """

    def __init__(self, ref: ModelRef, *, allow_downloads: bool) -> None:
        """Rejects references a self-hosted Models API would not serve."""
        self.ref = ref
        self.name = ref.entry.name
        self.catalog = CatalogService(
            allow_downloads=allow_downloads, surface="self-hosted"
        )
        self.executor = PackageExecutor(allow_downloads=allow_downloads)
        self.allow_downloads = allow_downloads
        self.catalog.get(Scope("local", "load"), str(ref))

    def _admitted(self, model: object) -> None:
        try:
            ref = parse(model) if type(model) is str else None
        except ModelRefError:
            ref = None
        if ref != self.ref:
            raise _not_served()

    def warm(self) -> None:
        """Loads the pinned weights the request paths use, through their caches."""
        from generalforecast.deployments import catalog_forecast
        from generalforecast.models import registry
        from generalforecast.models.artifacts import ArtifactError

        if not self.ref.entry.capability.tabular:
            catalog_forecast.model(
                self.ref, surface="self-hosted", allow_downloads=self.allow_downloads
            )
            return
        try:
            registry.load_ref(
                self.ref, surface="self-hosted", allow_downloads=self.allow_downloads
            )
        except ArtifactError as error:
            raise ServiceError(error.code, str(error), 503) from None

    def describe(self, scope: Scope) -> JsonObject:
        """The hosted catalog description of exactly this reference."""
        return self.catalog.get(scope, str(self.ref))

    def infer(self, scope: Scope, payload: JsonObject) -> JsonObject:
        """The hosted `CatalogService.infer`, limited to the served reference."""
        self._admitted(payload.get("model"))
        return dict(self.catalog.infer(scope, cast(InferenceRequest, payload)))

    def predict(self, scope: Scope, payload: JsonObject) -> JsonObject:
        """The hosted one-shot `/predictions`, limited to the served reference."""
        self._admitted(payload.get("model"))
        if not self.ref.entry.gates.allows("package", variant=self.ref.variant):
            raise ServiceError(
                "UnsupportedCapability",
                f"packages cannot lock {self.name}; POST models/{self.name}/infer",
                422,
            )
        lock, context = one_shot_lock(payload)
        pin = lock.model
        if (pin.name, pin.codes, pin.estimators) != (
            self.ref.name,
            self.ref.codes,
            self.ref.estimators,
        ):
            raise _not_served()
        prepared = self.executor.prepare(lock, None)
        table = prepared.predict(
            request_table(payload["queries"], lock, max_rows=MAX_QUERY_ROWS),
            context=context,
        )
        return prediction(table, lock, context.num_rows)


class PackageModel:
    """A locked catalog package; its context is bound once and shared read-only."""

    def __init__(
        self, lock: PackageLock, context: pa.Table | None, *, allow_downloads: bool
    ) -> None:
        """Admits the lock now; weights and context load in `warm`."""
        self.lock = admit(lock)
        mode = self.lock.spec.context.mode
        ensure(
            mode != "dataset",
            "dataset context needs the hosted data plane; lock a snapshot or "
            "request-context package to serve it",
            "UnsupportedCapability",
        )
        ensure(
            (context is None) == (mode == "request"),
            "snapshot packages are served from their package directory",
            "ContextRequired",
        )
        self.name = self.lock.spec.name
        self.context = context
        self.executor = PackageExecutor(allow_downloads=allow_downloads)
        self.prepared: PreparedPackage | None = None
        self.schema_free = mode == "request" and not self.lock.schema.columns
        self._warming = threading.Lock()

    def warm(self) -> None:
        """Loads weights and warms bound context exactly once."""
        with self._warming:
            if self.prepared is None:
                self.prepared = self.executor.prepare(self.lock, self.context)

    def describe(self, scope: Scope) -> JsonObject:
        """Locked identity, roles and outputs, without context or weights."""
        lock = self.lock
        return {
            "name": self.name,
            "kind": "package",
            "model": lock.model.resolved,
            "task": lock.task,
            "target": lock.spec.target,
            "features": list(lock.schema.features),
            "outputs": list(lock.outputs),
            "context": lock.spec.context.mode,
            "context_rows": None
            if self.prepared is None
            else self.prepared.context_rows,
            "package_digest": lock.package_digest,
        }

    def infer(self, scope: Scope, payload: JsonObject) -> JsonObject:
        """Packages answer `/predictions`, which keeps their locked outputs."""
        raise ServiceError(
            "UnsupportedCapability",
            "packages serve POST predictions with the package name as model",
            422,
        )

    def predict(self, scope: Scope, payload: JsonObject) -> JsonObject:
        """The `/predictions` body, checked against the lock instead of compiled."""
        fields(
            payload,
            {"model", "task", "target", "features", "context", "queries", "options"},
        )
        if payload["model"] != self.name:
            raise _not_served()
        lock = self.lock
        if payload["task"] != lock.task:
            raise ServiceError(
                "UnsupportedTask", "package does not serve this task", 422
            )
        require(payload["options"] == {}, "package options are compiled into its lock")
        require(type(payload["features"]) is list, "features must be a list")
        require(
            type(payload["queries"]) is list
            and 0 < len(payload["queries"]) <= MAX_QUERY_ROWS,
            "queries must contain 1..10000 rows",
        )
        if not self.schema_free:
            require(payload["target"] == lock.spec.target, "target differs from lock")
            require(
                payload["features"] == list(lock.schema.features),
                "features differ from lock",
            )
        prepared = self.prepared
        if prepared is None:
            raise ServiceError("NotReady", "model is loading", 503)
        if lock.spec.context.mode != "request":
            require(payload["context"] == [], "bound context cannot be overridden")
            table = prepared.predict(
                request_table(payload["queries"], lock, max_rows=MAX_QUERY_ROWS)
            )
            return prediction(
                table,
                lock,
                prepared.context_rows or 0,
                source="snapshot",
                version=lock.data_digest,
            )
        context = request_table(payload["context"], lock, max_rows=MAX_CONTEXT_ROWS)
        if self.schema_free:
            bound = bind_request(lock, payload, context)
            prepared = self.executor.prepare(bound, None)
            lock = bound
        table = prepared.predict(
            request_table(payload["queries"], lock, max_rows=MAX_QUERY_ROWS),
            context=context,
        )
        return prediction(table, lock, context.num_rows)


def load(target: str, *, allow_downloads: bool = True) -> Served:
    """Resolves a package directory, a `gf.lock` file or a catalog model reference.

    Raises:
        PackageError: If the target is not a servable package or model.
    """
    require(type(target) is str and bool(target), "serve target is required")
    path = Path(target)
    try:
        if path.is_dir():
            package = Package.load(path)
            ensure(
                package.lock is not None,
                "gf.lock is required; run gf check first",
                "LockRequired",
            )
            lock = package.compile(frozen=True)
            _catalog_lock(lock)
            assert isinstance(lock, PackageLock)
            context = None
            if lock.spec.context.mode == "snapshot":
                context = package.snapshot_table()
            return PackageModel(lock, context, allow_downloads=allow_downloads)
        if path.is_file():
            if (path.parent / "gf.yaml").is_file():
                return load(str(path.parent), allow_downloads=allow_downloads)
            raw = path.read_bytes()
            ensure(len(raw) <= 1_048_576, "lock file is too large")
            lock = PackageLock.from_dict(parse_yaml(raw))
            _catalog_lock(lock)
            assert isinstance(lock, PackageLock)
            ensure(
                lock.spec.context.mode == "request",
                "a lone lock file serves only request-context packages; "
                "serve the package directory instead",
                "ContextRequired",
            )
            return PackageModel(lock, None, allow_downloads=allow_downloads)
        try:
            ref = parse(target)
        except ModelRefError as error:
            raise PackageError(
                "ModelNotFound",
                f"{target!r} is not a package directory, lock file or catalog model",
            ) from error
        return CatalogModel(ref, allow_downloads=allow_downloads)
    except ServiceError as error:
        raise PackageError(error.code, error.message) from None
    except OSError as error:
        raise PackageError("MissingFile", f"cannot read {target!r}") from error


def _catalog_lock(lock: Any) -> None:
    ensure(
        not isinstance(lock, PythonLock),
        "Python packages run in the hosted isolated runtime, not gf serve",
        "UnsupportedCapability",
    )
