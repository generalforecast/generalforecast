# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What `gf serve` serves: a catalog model or a locked tabular package, described as gf-server reads it."""

from pathlib import Path
from typing import Any

from generalforecast.deployments.boundary import Scope
from generalforecast.deployments.catalog import CatalogService, ServingSurface
from generalforecast.deployments.contracts import ServiceError, require
from generalforecast.deployments.package_runtime.validation import admit
from generalforecast.models.artifacts import ArtifactError, checkpoint
from generalforecast.models.catalog import ModelRef, ModelRefError, parse
from generalforecast.models.outputs import DEFAULT_QUANTILES
from generalforecast.packages import Package, PackageLock, PythonLock
from generalforecast.packages.files import parse_yaml
from generalforecast.packages.models import describe
from generalforecast.packages.schema import declared_seconds
from generalforecast.packages.validation import PackageError, ensure
from generalforecast.tabular import arrow_bytes

LOCK_BYTES = 1_048_576


def served(
    target: str, directory: Path, *, allow_downloads: bool = True
) -> dict[str, Any]:
    """The served model's entry in gf-server's models file, with its weights verified.

    The target is a package directory, a `gf.lock` file or a catalog model
    reference. A snapshot package's context rows go to `directory`, where
    gf-server binds them once into a durable table; a request-context
    package binds each request's rows into an ephemeral table.

    The operator runs the server, so it also serves pinned models whose
    license forbids hosting them on General Forecast infrastructure. A
    catalog model's description is the hosted catalog's, so
    `GET models/{name}` answers the same on both.

    Raises:
        PackageError: If the target is not a package or catalog model this
            server may serve, or its weights are missing or differ from
            their pins.
    """
    require(type(target) is str and bool(target), "serve target is required")
    path = Path(target)
    try:
        if path.is_dir() or path.is_file():
            return _package(path, directory, allow_downloads=allow_downloads)
        try:
            ref = parse(target)
        except ModelRefError as error:
            raise PackageError(
                "ModelNotFound",
                f"{target!r} is not a package directory, lock file or catalog model",
            ) from error
    except ServiceError as error:
        raise PackageError(error.code, error.message) from None
    except OSError as error:
        raise PackageError("MissingFile", f"cannot read {target!r}") from error
    return catalog_model(ref, surface="self-hosted", allow_downloads=allow_downloads)


def catalog_model(
    ref: ModelRef, *, surface: ServingSurface, allow_downloads: bool
) -> dict[str, Any]:
    """A catalog model's entry in gf-server's models file, its weights verified.

    `gf serve` passes the `self-hosted` surface, where the operator may serve
    pinned models whose license forbids hosting them on General Forecast
    infrastructure; the hosted pool passes `remote`, which keeps those out.

    Raises:
        PackageError: If the surface may not serve the model, or its weights
            are missing or differ from their pins.
    """
    try:
        description = CatalogService(surface=surface).get(
            Scope("local", "load"), str(ref)
        )
    except ServiceError as error:
        raise PackageError(error.code, error.message) from None
    _verified(ref, surface=surface, allow_downloads=allow_downloads)
    pin = describe(str(ref))
    return {
        "name": ref.entry.name,
        "reference": str(ref),
        "worker": ref.entry.name,
        "resolved": pin.resolved,
        "outputs": list(pin.outputs),
        "description": description,
        "license_source": ref.entry.artifact.license_source,
    }


def _verified(
    ref: ModelRef, *, surface: ServingSurface = "self-hosted", allow_downloads: bool
) -> None:
    try:
        checkpoint(ref.entry, surface=surface, allow_downloads=allow_downloads)
    except ArtifactError as error:
        raise PackageError(error.code, str(error)) from None


def _package(path: Path, directory: Path, *, allow_downloads: bool) -> dict[str, Any]:
    """A locked package's entry, with a snapshot's selected rows written out.

    The entry holds the package's identity and its lock's schema, which
    gf-server binds rows with through `gf_core::package`.
    """
    context = None
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
        if lock.spec.context.mode == "snapshot":
            context = package.context_table()
    elif (path.parent / "gf.yaml").is_file():
        return _package(path.parent, directory, allow_downloads=allow_downloads)
    else:
        raw = path.read_bytes()
        ensure(len(raw) <= LOCK_BYTES, "lock file is too large")
        lock = PackageLock.from_dict(parse_yaml(raw))
        _catalog_lock(lock)
        assert isinstance(lock, PackageLock)
        ensure(
            lock.spec.context.mode == "request",
            "a lone lock file serves only request-context packages; "
            "serve the package directory instead",
            "ContextRequired",
        )
    lock = admit(lock)
    mode = lock.spec.context.mode
    ensure(
        mode != "dataset",
        "dataset context needs the hosted data plane; lock a snapshot or "
        "request-context package to serve it",
        "UnsupportedCapability",
    )
    ref = parse(lock.model.resolved)
    _verified(ref, allow_downloads=allow_downloads)
    snapshot = None
    if context is not None:
        snapshot = directory / "context.arrow"
        snapshot.write_bytes(arrow_bytes(context))
    return served_entry(lock, snapshot)


def served_entry(lock: PackageLock, snapshot: Path | None = None) -> dict[str, Any]:
    """A locked package as gf-server serves it, from `gf serve` or a deployment.

    `snapshot` names the file a snapshot's rows were written to, if any were.
    """
    ref = parse(lock.model.resolved)
    mode = lock.spec.context.mode
    regression = lock.task in ("regression", "forecast")
    return {
        "name": lock.spec.name,
        "reference": str(ref),
        "worker": ref.entry.name,
        "resolved": lock.model.resolved,
        "outputs": list(lock.outputs),
        "license_source": ref.entry.artifact.license_source,
        "description": {
            "name": lock.spec.name,
            "kind": "package",
            "model": lock.model.resolved,
            "task": lock.task,
            "target": lock.spec.target,
            "features": list(lock.schema.features),
            "outputs": list(lock.outputs),
            "context": mode,
            "context_rows": lock.context_rows if mode == "snapshot" else None,
            "package_digest": lock.package_digest,
        },
        "package": {
            "schema": lock.schema.to_dict(),
            "task": lock.task,
            "target": lock.spec.target,
            "features": list(lock.schema.features),
            "id": lock.spec.id,
            "context": mode,
            "quantiles": list(lock.spec.quantiles or DEFAULT_QUANTILES)
            if regression
            else [],
            "data_digest": lock.data_digest,
            "package_digest": lock.package_digest,
            "snapshot": None if snapshot is None else str(snapshot),
            "forecast": None
            if lock.task != "forecast"
            else {
                "series": list(lock.spec.series),
                "time": lock.spec.time,
                "horizon": lock.spec.horizon,
                "past_inputs": list(lock.spec.past_inputs),
                "future_inputs": list(lock.spec.future_inputs),
                "frequency": lock.schema.frequency,
                "declared": declared_seconds(lock.spec.frequency),
            },
        },
    }


def _catalog_lock(lock: Any) -> None:
    ensure(
        not isinstance(lock, PythonLock),
        "Python packages run in the hosted isolated runtime, not gf serve",
        "UnsupportedCapability",
    )
