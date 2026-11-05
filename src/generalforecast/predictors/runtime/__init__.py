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

from generalforecast import _core
from generalforecast.engine import LocalEngine
from generalforecast.models.catalog import ModelRef
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.context import plan_predictor, served_context
from generalforecast.predictors.runtime.forecast import ForecastContext
from generalforecast.predictors.runtime.schema import TableSchema
from generalforecast.predictors.runtime.tabular import TabularContext, bind_rows
from generalforecast.predictors.runtime.validation import (
    admit,
    context_table,
    query_table,
)
from generalforecast.predictors.validation import PredictorError, ensure
from generalforecast.service.contracts import ServiceError
from generalforecast.service.interfaces import ResourceFree

# The schema metadata key of a prediction table naming how many context rows
# the call read, as ASCII digits.
CONTEXT_ROWS_KEY = b"generalforecast.context_rows"


def _load(
    lock: CompiledPlan,
    allow_downloads: bool,
    surface: Literal["predictor", "local"] = "predictor",
) -> tuple[LocalEngine, ModelRef]:
    """The process engine and the locked model, verified against its pins, for predictors."""
    from generalforecast import engine as engines
    from generalforecast.models.artifacts import ArtifactError
    from generalforecast.models.catalog import ModelRefError, parse

    pin = lock.model
    engine = engines.local(surface, allow_downloads)
    try:
        # The pin's own spelling: a derived model's id pinned to its weights.
        ref = parse(pin.resolved)
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


def _refusal(error: _core.RequestError, message: str) -> ServiceError:
    """An engine refusal as the Models API answers it.

    One the engine answers 5xx, such as a worker starting again
    (`ModelUnavailable`) or failing (`InferenceFailed`), keeps its code,
    message and status, so a caller retries it rather than reading it as
    bad input; any other is the predictor's own invalid `message`.
    """
    code, reason, status = error.args
    if isinstance(status, int) and status >= 500:
        return ServiceError(code, reason, status)
    return ServiceError("ValidationError", message, 422)


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
                # The rows bind as served_context reads them on every path
                # (kinds, whole-number widening, ISO times, unlabeled rows
                # left out), whether or not the caller read them so first.
                rows_read = served_context(
                    {"predictor": plan_predictor(self.lock.spec)}, context
                )
                assert rows_read is not None
                bound, schema, rows = _bind(self.lock, self._model, rows_read)
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
                {CONTEXT_ROWS_KEY: str(rows).encode("ascii")}
            )
        except ServiceError:
            raise
        except PredictorError as error:
            raise _failure(error) from None
        except _core.RequestError as error:
            raise _refusal(error, "invalid predictor prediction input") from None
        except (ValueError, TypeError, pa.ArrowException):
            raise ServiceError(
                "ValidationError", "invalid predictor prediction input", 422
            ) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native predictor prediction failed", 503
            ) from None


@dataclass(frozen=True)
class CheckedPredictor(ResourceFree):
    """A revision admitted and its context checked, with no model loaded.

    What a control plane whose predictions a pool of its own serves keeps
    of a binding: the rows the context binds, never weights or a worker.
    """

    lock: CompiledPlan
    _schema: TableSchema
    context_rows: int | None

    def predict(
        self, queries: pa.Table, *, context: pa.Table | None = None
    ) -> pa.Table:
        """Refused: a checked predictor holds no model to score with."""
        raise ServiceError(
            "Unsupported", "this executor checks predictors and serves none", 501
        )


def _checked(lock: CompiledPlan, context: pa.Table) -> tuple[TableSchema, int]:
    """A context's schema and labeled rows, checked as a binding checks them, without a model."""
    table, schema = context_table(lock, context)
    rows = table.num_rows
    if lock.task != "forecast":
        assert lock.spec.target is not None
        bind_rows(lock, table, schema)
        rows -= table[lock.spec.target].null_count
    return schema, rows


class PredictorExecutor:
    """The curated engine adapter; operators may inject a separate admitted executor."""

    def __init__(
        self,
        *,
        allow_downloads: bool = False,
        surface: Literal["predictor", "local"] = "predictor",
        load: bool = True,
    ) -> None:
        """Keeps online artifact acquisition an explicit host-operator decision.

        `surface` is the qualification gate models must pass: `predictor`
        for hosted serving, `local` for `gf.Predictor` on the user's machine
        (`models.CatalogResolver`). Without `load`, `prepare` admits the
        plan and checks its context but loads no model and binds nothing in
        this process (`CheckedPredictor`): a control plane whose pool serves
        its predictions holds no weights of its own.
        """
        from generalforecast.predictors.models import CatalogResolver

        self.allow_downloads = allow_downloads
        self.surface = surface
        self.load = load
        self.resolver = CatalogResolver(surface)

    def prepare(
        self, lock: CompiledPlan, context: pa.Table | None
    ) -> PreparedPredictor | CheckedPredictor:
        """Admits immutable content, loads hash-verified weights and warms bound context.

        Without `load`, the content is admitted and the context checked
        alone (`CheckedPredictor`).
        """
        try:
            lock = admit(lock, self.resolver)
            mode = lock.spec.context.mode
            ensure(
                (context is None) == (mode == "request"),
                "bound context is required only for snapshot and live-context predictors",
                "ContextRequired",
            )
            if not self.load:
                schema, rows = lock.schema, None
                if context is not None:
                    schema, rows = _checked(lock, context)
                return CheckedPredictor(lock, schema, rows)
            model = _load(lock, self.allow_downloads, self.surface)
            bound, schema, rows = None, lock.schema, None
            if context is not None:
                bound, schema, rows = _bind(lock, model, context)
            return PreparedPredictor(lock, model, bound, schema, rows)
        except ServiceError:
            raise
        except PredictorError as error:
            raise _failure(error) from None
        except _core.RequestError as error:
            raise _refusal(error, "invalid predictor context") from None
        except (ValueError, TypeError, pa.ArrowException):
            raise ServiceError(
                "ValidationError", "invalid predictor context", 422
            ) from None
        except Exception:  # noqa: BLE001
            raise ServiceError(
                "InferenceFailed", "native predictor preparation failed", 503
            ) from None


__all__ = [
    "CONTEXT_ROWS_KEY",
    "CheckedPredictor",
    "PredictorExecutor",
    "PreparedPredictor",
]
