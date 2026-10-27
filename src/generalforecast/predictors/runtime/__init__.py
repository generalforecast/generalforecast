# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verified in-process predictor execution, without HTTP, persistence or identity.

`PredictorExecutor.prepare` binds a `CompiledPlan` (see `compile_lock`) and
its context rows; the prepared predictor scores queries. `gf.Predictor`
runs local and ad hoc predictors through it, and hosted serving runs pushed
revisions through it, so both bind and score the same way.
"""

from dataclasses import dataclass
from typing import Literal

import pyarrow as pa

from generalforecast.engine import LocalEngine
from generalforecast.models.catalog import ModelRef
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.forecast import ForecastContext
from generalforecast.predictors.runtime.schema import TableSchema
from generalforecast.predictors.runtime.tabular import TabularContext
from generalforecast.predictors.runtime.validation import (
    admit,
    context_table,
    query_table,
)
from generalforecast.predictors.validation import PredictorError, ensure
from generalforecast.service.contracts import ServiceError
from generalforecast.service.interfaces import ResourceFree


def _load(
    lock: CompiledPlan,
    allow_downloads: bool,
    surface: Literal["predictor", "local"] = "predictor",
) -> tuple[LocalEngine, ModelRef]:
    """The process engine and the locked model, verified against its pins, for predictors."""
    from generalforecast import engine as engines
    from generalforecast.models.artifacts import ArtifactError
    from generalforecast.models.catalog import ModelRefError

    pin = lock.model
    ref = ModelRef(pin.model, pin.codes, pin.estimators, pin.digest)
    engine = engines.local(surface, allow_downloads)
    try:
        engine.load(ref)
    except ArtifactError as error:
        raise ServiceError(error.code, str(error), 503) from None
    except ModelRefError as error:
        raise PredictorError(error.code, str(error)) from None
    return engine, ref


def _bind(lock: CompiledPlan, model: tuple[LocalEngine, ModelRef], context: pa.Table):
    table, schema = context_table(lock, context)
    engine = ForecastContext if lock.task == "forecast" else TabularContext
    rows = table.num_rows
    if lock.task != "forecast":
        assert lock.spec.target is not None
        rows -= table[lock.spec.target].null_count
    return engine.build(lock, model, table, schema), schema, rows


def _failure(error: PredictorError) -> ServiceError:
    return ServiceError(error.code, str(error), 422)


@dataclass(frozen=True)
class PreparedPredictor(ResourceFree):
    """An isolated revision's model and derived state, safe for API atomic swaps.

    In-process weights and warmed context are garbage collected, so its
    lifecycle is the explicit resource-free one.
    """

    lock: CompiledPlan
    _model: tuple[LocalEngine, ModelRef]
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
                ensure(
                    context is None, "a bound predictor's context cannot be overridden"
                )
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
        except PredictorError as error:
            raise _failure(error) from None
        except (ValueError, TypeError, pa.ArrowException):
            raise ServiceError(
                "ValidationError", "invalid predictor prediction input", 422
            ) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native predictor prediction failed", 503
            ) from None


class PredictorExecutor:
    """The curated engine adapter; operators may inject a separate admitted executor."""

    def __init__(
        self,
        *,
        allow_downloads: bool = False,
        surface: Literal["predictor", "local"] = "predictor",
    ) -> None:
        """Keeps online artifact acquisition an explicit host-operator decision.

        `surface` is the qualification gate models must pass: `predictor`
        for hosted serving, `local` for `gf.Predictor` on the user's machine
        (`models.CatalogResolver`).
        """
        from generalforecast.predictors.models import CatalogResolver

        self.allow_downloads = allow_downloads
        self.surface = surface
        self.resolver = CatalogResolver(surface)

    def prepare(
        self, lock: CompiledPlan, context: pa.Table | None
    ) -> PreparedPredictor:
        """Admits immutable content, loads hash-verified weights and warms bound context."""
        try:
            lock = admit(lock, self.resolver)
            mode = lock.spec.context.mode
            ensure(
                (context is None) == (mode == "request"),
                "bound context is required only for snapshot and live-context predictors",
                "ContextRequired",
            )
            model = _load(lock, self.allow_downloads, self.surface)
            bound, schema, rows = None, lock.schema, None
            if context is not None:
                bound, schema, rows = _bind(lock, model, context)
            return PreparedPredictor(lock, model, bound, schema, rows)
        except ServiceError:
            raise
        except PredictorError as error:
            raise _failure(error) from None
        except (ValueError, TypeError, pa.ArrowException):
            raise ServiceError(
                "ValidationError", "invalid predictor context", 422
            ) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native predictor preparation failed", 503
            ) from None


__all__ = ["PredictorExecutor", "PreparedPredictor"]
