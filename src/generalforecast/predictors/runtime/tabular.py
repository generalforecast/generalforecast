# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Predictor tabular contexts, bound once and scored on the engine, for any model."""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from generalforecast import _core
from generalforecast.engine import Bound, Labeled, LocalEngine, ModelOutputError
from generalforecast.models.catalog import ModelRef
from generalforecast.models.outputs import (
    DEFAULT_QUANTILES,
    probabilities,
    quantile_index,
    quantiles,
)
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.schema import TableSchema
from generalforecast.predictors.validation import PredictorError
from generalforecast.service.contracts import ServiceError
from generalforecast.tabular import arrow_bytes


def _refused(error: _core.RequestError) -> PredictorError:
    code, message, _ = error.args
    return PredictorError(code, message)


def strategy(lock: CompiledPlan) -> str | None:
    """The engine strategy the plan's selection binds with, as JSON; exact when None."""
    selection = lock.spec.selection
    return None if selection is None else json.dumps(selection.strategy_spec())


def bound_order(lock: CompiledPlan, table: pa.Table) -> pa.Table:
    """The context rows in the order the binder writes them: by time, when they have one.

    A retrieved selection's neighbors lean to the newest rows by their order
    in the bound table, so every path binds them oldest first. `gf serve`
    writes its snapshot rows in this order too.
    """
    time = lock.spec.time
    if time is None or time not in table.column_names:
        return table
    return table.sort_by([(time, "ascending")])


@dataclass(frozen=True)
class TabularContext:
    """A context's `gf_core::predictor` binding and the engine context its queries read.

    The labeled rows are bound once as a table, which lives as long as the
    prepared predictor: a snapshot's or live context's for every request, a request
    context's for that request.
    """

    context: _core.PredictorContext
    classes: tuple[Any, ...]
    levels: tuple[float, ...]
    bound: Bound

    @classmethod
    def build(
        cls,
        lock: CompiledPlan,
        model: tuple[LocalEngine, ModelRef],
        table: pa.Table,
        schema: TableSchema,
    ):
        """Binds the labeled rows through `gf_core::predictor`, then as a table."""
        target = lock.spec.target
        assert target is not None
        table = bound_order(lock, table)
        try:
            context = _core.bind_predictor(
                json.dumps(schema.to_dict()).encode("utf-8"),
                "classification" if lock.task == "classification" else "regression",
                target,
                arrow_bytes(table),
            )
        except _core.RequestError as error:
            raise _refused(error) from None
        classes = tuple(context.classes)
        labeled = Labeled(
            task="classification" if classes else "regression",
            context=np.asarray(context.features, np.float32),
            targets=np.asarray(context.targets, np.int32 if classes else np.float32),
            classes=classes,
        )
        engine, ref = model
        return cls(
            context,
            classes,
            tuple(lock.spec.quantiles or DEFAULT_QUANTILES),
            engine.bind(ref, labeled, strategy(lock)),
        )

    def predict(
        self, lock: CompiledPlan, model: tuple[LocalEngine, ModelRef], table: pa.Table
    ) -> dict[str, Any]:
        """Scores the query rows in their order, in steps the engine shares."""
        try:
            matrix = self.context.encode(arrow_bytes(table))
        except _core.RequestError as error:
            raise _refused(error) from None
        engine, _ = model
        try:
            result = engine.predict(self.bound, np.asarray(matrix, np.float32))
        except ModelOutputError:
            raise ServiceError(
                "ModelOutputError", "native model returned invalid outputs", 503
            ) from None
        if self.classes:
            return {
                "label": [self.classes[i] for i in result.argmax(axis=1)],
                "probabilities": [probabilities(self.classes, row) for row in result],
            }
        median = quantile_index(0.5)
        return {
            "value": result[:, median].tolist(),
            "quantiles": [quantiles(row, self.levels) for row in result],
        }
