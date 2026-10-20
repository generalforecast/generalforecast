"""Verified in-process package execution, without HTTP, persistence or identity."""

from dataclasses import dataclass
from typing import Any, cast

import pyarrow as pa

from generalforecast.deployments.contracts import ServiceError
from generalforecast.deployments.interfaces import ResourceFree
from generalforecast.deployments.package_runtime.forecast import ForecastContext
from generalforecast.deployments.package_runtime.tabular import TabularContext
from generalforecast.deployments.package_runtime.validation import (
    admit,
    context_table,
    query_table,
)
from generalforecast.packages import PackageLock
from generalforecast.packages.python_lock import PythonLock
from generalforecast.packages.schema import TableSchema
from generalforecast.packages.validation import PackageError, ensure


def _load(lock: PackageLock, allow_downloads: bool) -> Any:
    """The locked model and variant, verified against its pins, for packages."""
    from generalforecast.models import registry
    from generalforecast.models.artifacts import ArtifactError
    from generalforecast.models.catalog import ModelRef, ModelRefError

    pin = lock.model
    ref = ModelRef(pin.name, pin.codes, pin.estimators, pin.digest)
    try:
        model = registry.load_ref(
            ref, surface="package", allow_downloads=allow_downloads
        )
    except ArtifactError as error:
        raise ServiceError(error.code, str(error), 503) from None
    except ModelRefError as error:
        raise PackageError(error.code, str(error)) from None
    if lock.task == "forecast":
        functions, parameters = model
        ensure(
            tuple(cast(Any, functions.properties(parameters)).quantiles)
            == ref.entry.capability.quantiles,
            "pinned model capabilities changed",
            "ModelMismatch",
        )
    return model


def _bind(lock: PackageLock, model: Any, context: pa.Table):
    table, schema = context_table(lock, context)
    engine = ForecastContext if lock.task == "forecast" else TabularContext
    rows = table.num_rows
    if lock.task != "forecast":
        assert lock.spec.target is not None
        rows -= table[lock.spec.target].null_count
    return engine.build(lock, model, table, schema), schema, rows


def _failure(error: PackageError) -> ServiceError:
    return ServiceError(error.code, str(error), 422)


@dataclass(frozen=True)
class PreparedPackage(ResourceFree):
    """An isolated revision's model and derived state, safe for API atomic swaps.

    In-process weights and warmed context are garbage collected, so its
    lifecycle is the explicit resource-free one.
    """

    lock: PackageLock
    _model: Any
    _context: TabularContext | ForecastContext | None
    _schema: TableSchema
    context_rows: int | None

    def predict(
        self, queries: pa.Table, *, context: pa.Table | None = None
    ) -> pa.Table:
        """Projects locked outputs in input row order, preserving a declared query ID."""
        try:
            bound, schema, rows = self._context, self._schema, self.context_rows
            if self.lock.spec.context.mode == "request":
                ensure(
                    context is not None,
                    "request context is required",
                    "ContextRequired",
                )
                assert context is not None
                ensure(
                    self.lock.spec.target is not None,
                    "schema-free request context requires explicit target binding",
                )
                bound, schema, rows = _bind(self.lock, self._model, context)
            else:
                ensure(context is None, "bound package context cannot be overridden")
            assert bound is not None
            table = query_table(self.lock, schema, queries)
            outputs = bound.predict(self.lock, self._model, table)
            columns = {}
            if self.lock.spec.id in table.column_names:
                columns[self.lock.spec.id] = table[self.lock.spec.id]
            for name in self.lock.outputs:
                columns[name] = outputs[name]
            return pa.table(columns).replace_schema_metadata(
                {b"generalforecast.context_rows": str(rows).encode("ascii")}
            )
        except ServiceError:
            raise
        except PackageError as error:
            raise _failure(error) from None
        except (ValueError, TypeError, pa.ArrowException):
            raise ServiceError(
                "ValidationError", "invalid package prediction input", 422
            ) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native package prediction failed", 503
            ) from None


class PackageExecutor:
    """The curated engine adapter; operators may inject a separate admitted executor."""

    def __init__(self, *, allow_downloads: bool = False) -> None:
        """Keeps online artifact acquisition an explicit host-operator decision."""
        self.allow_downloads = allow_downloads

    def prepare(
        self, lock: PackageLock | PythonLock, context: pa.Table | None
    ) -> PreparedPackage:
        """Admits immutable content, loads hash-verified weights and warms bound context."""
        try:
            lock = admit(lock)
            mode = lock.spec.context.mode
            ensure(
                (context is None) == (mode == "request"),
                "bound context is required only for snapshot/dataset packages",
                "ContextRequired",
            )
            model = _load(lock, self.allow_downloads)
            bound, schema, rows = None, lock.schema, None
            if context is not None:
                bound, schema, rows = _bind(lock, model, context)
            return PreparedPackage(lock, model, bound, schema, rows)
        except ServiceError:
            raise
        except PackageError as error:
            raise _failure(error) from None
        except (ValueError, TypeError, pa.ArrowException):
            raise ServiceError(
                "ValidationError", "invalid package context", 422
            ) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native package preparation failed", 503
            ) from None


__all__ = ["PackageExecutor", "PreparedPackage"]
