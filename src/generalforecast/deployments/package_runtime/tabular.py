# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native tabular context fitting and reusable immutable query state, for any model."""

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from generalforecast.columns import Column, encode, prepare
from generalforecast.deployments.contracts import ServiceError
from generalforecast.models import inference
from generalforecast.models.outputs import (
    DEFAULT_QUANTILES,
    class_order,
    probabilities,
    quantile_index,
    quantiles,
)
from generalforecast.packages import PackageLock
from generalforecast.packages.schema import TableSchema
from generalforecast.packages.validation import ensure


def _columns(schema: TableSchema) -> tuple[Column, ...]:
    return tuple(
        Column(
            c.name,
            "categorical"
            if c.semantic == "category" or c.type == "string"
            else "datetime"
            if c.semantic == "datetime" or c.type in ("date", "timestamp")
            else "numeric",
        )
        for c in schema.columns
        if c.role == "input"
    )


@dataclass(frozen=True)
class TabularContext:
    """Independent fit statistics, vocabularies and immutable attention cache."""

    columns: tuple[Column, ...]
    vocabularies: tuple[tuple[tuple[str, int], ...], ...]
    classes: tuple
    state: Any
    cache: Any

    @classmethod
    def build(cls, lock: PackageLock, model: Any, table: pa.Table, schema: TableSchema):
        """Fits all selected labeled rows and blocks until their cache is ready."""
        import jax.numpy as jnp

        target = lock.spec.target
        assert target is not None
        table = table.filter(table[target].is_valid())
        ensure(table.num_rows > 0, "context has no labeled rows", "EmptyContext")
        columns = _columns(schema)
        values, _ = prepare(table, columns)
        vocabularies: list[dict[str, int]] = [{} for _ in columns]
        matrix, added = encode(values, columns, vocabularies, True)
        for index, words in added.items():
            vocabularies[index].update({word: i for i, word in enumerate(words)})
        labels = table[target].to_pylist()
        classes = class_order(labels) if lock.task == "classification" else ()
        if classes:
            ensure(len(set(labels)) >= 2, "selected context needs at least two classes")
            labels = [classes.index(value) for value in labels]
        state, cache = inference.fit(
            model,
            jnp.asarray(matrix),
            jnp.asarray(labels, dtype=jnp.int32 if classes else jnp.float32),
            len(classes),
        )
        return cls(
            columns,
            tuple(tuple(v.items()) for v in vocabularies),
            classes,
            state,
            cache,
        )

    def predict(self, lock: PackageLock, model: Any, table: pa.Table) -> dict[str, Any]:
        """Bounds native query intermediates while preserving request row order."""
        outputs: dict[str, list] = {name: [] for name in lock.outputs}
        for offset in range(0, table.num_rows, 128):
            batch = self._batch(lock, model, table.slice(offset, 128))
            for name, values in outputs.items():
                values.extend(batch[name])
        return outputs

    def _batch(self, lock: PackageLock, model: Any, table: pa.Table) -> dict[str, Any]:
        import jax.numpy as jnp

        values, _ = prepare(table, self.columns)
        matrix, _ = encode(
            values, self.columns, [dict(v) for v in self.vocabularies], False
        )
        try:
            result = inference.predict(
                model, (self.state, self.cache), jnp.asarray(matrix), len(self.classes)
            )
        except ValueError:
            raise ServiceError(
                "ModelOutputError", "native model returned invalid outputs", 503
            ) from None
        if self.classes:
            return {
                "label": [self.classes[i] for i in result.argmax(axis=1)],
                "probabilities": [probabilities(self.classes, row) for row in result],
            }
        levels = lock.spec.quantiles or DEFAULT_QUANTILES
        median = quantile_index(0.5)
        return {
            "value": result[:, median].tolist(),
            "quantiles": [quantiles(row, levels) for row in result],
        }
