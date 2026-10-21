"""Package tabular contexts, bound once and scored on the engine, for any model."""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from generalforecast import _core
from generalforecast.deployments.contracts import ServiceError
from generalforecast.engine import Bound, Labeled, LocalEngine, ModelOutputError
from generalforecast.models.catalog import ModelRef
from generalforecast.models.outputs import (
    DEFAULT_QUANTILES,
    probabilities,
    quantile_index,
    quantiles,
)
from generalforecast.packages import PackageLock
from generalforecast.packages.schema import TableSchema
from generalforecast.packages.validation import PackageError
from generalforecast.tabular import arrow_bytes


def _refused(error: _core.RequestError) -> PackageError:
    code, message, _ = error.args
    return PackageError(code, message)


@dataclass(frozen=True)
class TabularContext:
    """A context's `gf_core::package` binding and the engine context its queries read.

    The labeled rows are bound once as a table, which lives as long as the
    prepared package: a snapshot's or dataset's for every request, a request
    context's for that request.
    """

    context: _core.PackageContext
    classes: tuple[Any, ...]
    levels: tuple[float, ...]
    bound: Bound

    @classmethod
    def build(
        cls,
        lock: PackageLock,
        model: tuple[LocalEngine, ModelRef],
        table: pa.Table,
        schema: TableSchema,
    ):
        """Binds the labeled rows through `gf_core::package`, then as a table."""
        target = lock.spec.target
        assert target is not None
        try:
            context = _core.bind_package(
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
            engine.bind(ref, labeled),
        )

    def predict(
        self, lock: PackageLock, model: tuple[LocalEngine, ModelRef], table: pa.Table
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
