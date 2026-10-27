# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A forecaster's bound series: each forecast on its own, or with the series declared related.

Every series forecasts from its own context, from its own origin, with its
own known-future values; a predictor's `related` groups are the exception,
whose series (at most `ServingPolicy.group_series` of them) forecast jointly
from one origin and share their known-future values. A prediction is one
engine forecast request with a query per queried series, its group's
context rows and its future values, so any number of series forecast in one
batched call; the engine keeps each group's context table by its
fingerprint.

Times sit on their frequency's axis, `frequency.coordinate`: UTC seconds for
a span, the period ordinal for a calendar alias such as `MS`, so a model
stepping by `frequency.step` sees evenly spaced times. gf-server's binder
(`predictors.rs`) builds the same request from the same rows.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from generalforecast.engine import LocalEngine, ModelOutputError
from generalforecast.models.catalog import ModelRef
from generalforecast.models.outputs import DEFAULT_QUANTILES, quantile_index, quantiles
from generalforecast.predictors import frequency as _frequency
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.schema import TableSchema, series_groups
from generalforecast.predictors.validation import PredictorError, ensure
from generalforecast.service.catalog import forecast_request
from generalforecast.service.contracts import ServiceError


def _levels(spec: Any, declared: Any, grid: Any) -> None:
    """Holds the loaded forecaster's levels to the catalog's and the plan's.

    A model whose catalog entry declares its levels must output exactly
    those; one that declares none publishes them in its pinned weights, and
    the plan's levels must be among them.
    """
    ensure(
        not declared or tuple(grid) == tuple(declared),
        "pinned model capabilities changed",
        "ModelMismatch",
    )
    missing = [
        level for level in spec.quantiles or DEFAULT_QUANTILES if level not in grid
    ]
    ensure(
        not missing,
        f"the model does not output quantile {missing[0]}" if missing else "",
        "UnsupportedCapability",
    )


def _label(identity: tuple) -> str:
    return ", ".join(str(value) for value in identity) or "(all rows)"


@dataclass(frozen=True)
class ForecastContext:
    """A forecaster's series, aligned once into the rows its requests repeat.

    Attributes:
        identities: Each series' values of the series columns, by index.
        index: Each series' index, by its values.
        origins: Each series' last context time, on the frequency's axis.
        step: The distance between times on that axis.
        group_of: Each series' group.
        groups: Each group's series.
        rows: Each series' context rows, as the forecast request takes them.
        options: The request's model, target and series options.
    """

    identities: tuple[tuple, ...]
    index: dict[tuple, int]
    origins: tuple[int | float, ...]
    step: int | float
    group_of: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    rows: tuple[tuple[dict[str, Any], ...], ...]
    options: dict[str, Any]

    @classmethod
    def build(
        cls,
        lock: CompiledPlan,
        model: tuple[LocalEngine, ModelRef],
        table: pa.Table,
        schema: TableSchema,
    ):
        """Validates and aligns every series, and warms the first group's forecast."""
        from generalforecast.predictors.runtime.tabular import bound_order

        spec = lock.spec
        assert (
            spec.time is not None
            and spec.target is not None
            and spec.horizon is not None
            and spec.frequency is not None
        )
        cadence = spec.frequency
        records = bound_order(lock, table).to_pylist()
        index: dict[tuple, int] = {}
        rows: list[list[dict[str, Any]]] = []
        ends: list[int | float] = []
        for key, row in enumerate(records):
            identity = tuple(row[name] for name in spec.series)
            series = index.get(identity)
            moment = _frequency.coordinate(row[spec.time], cadence)
            if series is None:
                series = index[identity] = len(index)
                rows.append([])
                ends.append(moment)
            ends[series] = max(ends[series], moment)
            rows[series].append(
                {
                    "key": str(key),
                    "series": f"gfseries:{series}",
                    "time": moment,
                    "value": row[spec.target],
                    "covariates": {
                        name: row[name]
                        for name in (*spec.past_inputs, *spec.future_inputs)
                    },
                }
            )
        identities = list(index)
        ensure(bool(identities), "a forecast context has no series", "EmptyContext")
        group_of, groups = series_groups(spec, identities)
        for members in groups:
            ensure(
                len({ends[series] for series in members}) == 1,
                "related series must share the same forecast origin",
            )
        options = {
            "model": lock.model.resolved,
            "task": "forecast",
            "target": spec.target,
            "features": [],
            "options": {
                "series": {
                    "step": _frequency.step(cadence),
                    "horizon": spec.horizon,
                    "quantiles": list(spec.quantiles or DEFAULT_QUANTILES),
                    "known_covariates": list(spec.future_inputs),
                }
            },
        }
        bound = cls(
            tuple(identities),
            index,
            tuple(ends),
            _frequency.step(cadence),
            tuple(group_of),
            tuple(groups),
            tuple(tuple(series) for series in rows),
            options,
        )
        engine, ref = model
        declared = ref.entry.capability.quantiles
        zeros = {name: [0.0] * spec.horizon for name in spec.future_inputs}
        # Every group's request is checked; the engine warms on the first.
        forecast_request(
            bound.request({series: zeros for series in range(len(identities))}),
            declared,
        )
        _, grid = engine.forecast(
            ref, forecast_request(bound.request({groups[0][0]: zeros}), declared)
        )
        _levels(spec, declared, grid)
        return bound

    def request(self, futures: dict[int, dict[str, list[Any]]]) -> dict[str, Any]:
        """The forecast request for queried series and their known-future values.

        It carries the context rows of every group a queried series is in,
        and one query per queried series, in ascending series order.
        """
        queried = sorted(futures)
        groups = sorted({self.group_of[series] for series in queried})
        context = [
            row
            for group in groups
            for series in self.groups[group]
            for row in self.rows[series]
        ]
        queries = [
            {
                "key": f"q{series}",
                "series": f"gfseries:{series}",
                "origin": self.origins[series],
                "group": [
                    f"gfseries:{member}"
                    for member in self.groups[self.group_of[series]]
                ],
                "future_covariates": futures[series],
            }
            for series in queried
        ]
        return {**self.options, "context": context, "queries": queries}

    def predict(
        self, lock: CompiledPlan, model: tuple[LocalEngine, ModelRef], table: pa.Table
    ) -> dict[str, Any]:
        """Places each query row on its series' horizon and forecasts the queried series."""
        spec = lock.spec
        assert spec.horizon is not None and spec.time is not None
        assert spec.frequency is not None
        positions: list[tuple[int, int]] = []
        seen: dict[int, set[int]] = {}
        future: dict[tuple[int, str, int], float] = {}
        for row in table.to_pylist():
            identity = tuple(row[name] for name in spec.series)
            series = self.index.get(identity)
            if series is None:
                raise PredictorError(
                    "InvalidPredictor",
                    f"query series {_label(identity)} is missing from context",
                )
            moment = _frequency.coordinate(row[spec.time], spec.frequency)
            offset = (moment - self.origins[series]) / self.step - 1
            position = round(offset)
            ensure(
                abs(offset - position) < 1e-6 and 0 <= position < spec.horizon,
                "query times must cover the locked future horizon after the context origin",
            )
            held = seen.setdefault(series, set())
            ensure(
                position not in held, "duplicate forecast query time within a series"
            )
            held.add(position)
            positions.append((series, position))
            group = self.group_of[series]
            shared = len(self.groups[group]) > 1
            for name in spec.future_inputs:
                value = row[name]
                ensure(
                    type(value) in (int, float, bool)
                    and np.isfinite(value)
                    and abs(value) <= 1e12,
                    "known future covariates must be bounded numeric values",
                )
                # A related group shares its known-future values; a series
                # on its own has its own.
                key = (-1 - group if shared else series, name, position)
                ensure(
                    key not in future or future[key] == value,
                    "related series must share named future covariates",
                    "UnsupportedCovariates",
                )
                future[key] = value
        ensure(
            all(len(times) == spec.horizon for times in seen.values()),
            "queries must include the complete horizon for each requested series",
        )

        def owner(series: int) -> int:
            group = self.group_of[series]
            return -1 - group if len(self.groups[group]) > 1 else series

        futures = {
            series: {
                name: [
                    future[(owner(series), name, step)] for step in range(spec.horizon)
                ]
                for name in spec.future_inputs
            }
            for series in seen
        }
        engine, ref = model
        declared = ref.entry.capability.quantiles
        parsed = forecast_request(self.request(futures), declared)
        try:
            result, grid = engine.forecast(ref, parsed)
        except ModelOutputError:
            raise ServiceError(
                "ModelOutputError", "native forecaster returned invalid forecasts", 503
            ) from None
        _levels(spec, declared, grid)
        rows = {series: at for at, series in enumerate(sorted(futures))}
        levels = spec.quantiles or DEFAULT_QUANTILES
        return {
            "value": [
                float(result[rows[s], p, quantile_index(0.5, grid)])
                for s, p in positions
            ],
            "quantiles": [
                quantiles(result[rows[s], p], levels, grid) for s, p in positions
            ],
        }


__all__ = ["ForecastContext", "series_groups"]
