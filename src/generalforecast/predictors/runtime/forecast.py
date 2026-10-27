# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Aligned grouped forecast contexts, with explicit complete future query rows.

Times sit on their frequency's axis, `frequency.coordinate`: UTC seconds for
a span, the period ordinal for a calendar alias such as `MS`, so a model
stepping by `frequency.step` sees evenly spaced times. gf-server's binder
(`predictors.rs`) places them the same way.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from generalforecast.engine import LocalEngine, ModelOutputError
from generalforecast.models.catalog import V1_POLICY, ModelRef
from generalforecast.models.outputs import DEFAULT_QUANTILES, quantile_index, quantiles
from generalforecast.predictors import frequency as _frequency
from generalforecast.predictors.runtime.compiled import CompiledPlan
from generalforecast.predictors.runtime.schema import TableSchema
from generalforecast.predictors.validation import ensure
from generalforecast.service.catalog import forecast_request
from generalforecast.service.contracts import ServiceError


@dataclass(frozen=True)
class ForecastContext:
    """A related group's series, aligned once into the request its forecasts repeat.

    Each forecast is that request with the queries' future inputs, so the
    engine reads the group's context table by its fingerprint: made once,
    and once per set of known future values.
    """

    identities: tuple[tuple, ...]
    origin: int | float
    step: int | float
    request: dict[str, Any]

    @classmethod
    def build(
        cls,
        lock: CompiledPlan,
        model: tuple[LocalEngine, ModelRef],
        table: pa.Table,
        schema: TableSchema,
    ):
        """Validates and aligns a related group, binding it as a table."""
        spec = lock.spec
        assert (
            spec.time is not None
            and spec.target is not None
            and spec.horizon is not None
        )
        identities = tuple(
            dict.fromkeys(
                tuple(row[name] for name in spec.series) for row in table.to_pylist()
            )
        )
        ensure(
            1 <= len(identities) <= V1_POLICY.group_series,
            "grouped forecast supports at most eight explicitly related series",
            "CapacityExceeded",
        )
        assert spec.frequency is not None
        cadence = spec.frequency
        ends: dict[tuple, int | float] = {}
        rows = []
        for index, row in enumerate(table.to_pylist()):
            identity = tuple(row[name] for name in spec.series)
            moment = _frequency.coordinate(row[spec.time], cadence)
            ends[identity] = max(moment, ends.get(identity, moment))
            rows.append(
                {
                    "key": str(index),
                    "series": f"gfseries:{identities.index(identity)}",
                    "time": moment,
                    "value": row[spec.target],
                    "covariates": {
                        name: row[name]
                        for name in (*spec.past_inputs, *spec.future_inputs)
                    },
                }
            )
        ensure(
            len(set(ends.values())) == 1,
            "related series must share the same forecast origin",
        )
        origin = next(iter(ends.values()))
        step = _frequency.step(cadence)
        query = {
            "key": "warm",
            "series": "gfseries:0",
            "origin": origin,
            "group": [f"gfseries:{i}" for i in range(len(identities))],
            "future_covariates": {
                name: [0.0] * spec.horizon for name in spec.future_inputs
            },
        }
        request = {
            "model": lock.model.resolved,
            "task": "forecast",
            "target": spec.target,
            "features": [],
            "context": rows,
            "queries": [query],
            "options": {
                "series": {
                    "step": step,
                    "horizon": spec.horizon,
                    "quantiles": list(spec.quantiles or DEFAULT_QUANTILES),
                    "known_covariates": list(spec.future_inputs),
                }
            },
        }
        engine, ref = model
        engine.forecast(ref, forecast_request(request, ref.entry.capability.quantiles))
        return cls(identities, origin, step, request)

    def predict(
        self, lock: CompiledPlan, model: tuple[LocalEngine, ModelRef], table: pa.Table
    ) -> dict[str, Any]:
        """Aligns future rows and forecasts the queried series of the bound group."""
        spec = lock.spec
        assert spec.horizon is not None and spec.time is not None
        positions: list[tuple[int, int]] = []
        seen: dict[int, set[int]] = {}
        future: dict[tuple[str, int], float] = {}
        for row in table.to_pylist():
            identity = tuple(row[name] for name in spec.series)
            ensure(identity in self.identities, "query series is missing from context")
            series = self.identities.index(identity)
            assert spec.frequency is not None
            moment = _frequency.coordinate(row[spec.time], spec.frequency)
            offset = (moment - self.origin) / self.step - 1
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
            for name in spec.future_inputs:
                value = row[name]
                ensure(
                    type(value) in (int, float, bool)
                    and np.isfinite(value)
                    and abs(value) <= 1e12,
                    "known future covariates must be bounded numeric values",
                )
                key = (name, position)
                ensure(
                    key not in future or future[key] == value,
                    "grouped targets must share named future covariates",
                    "UnsupportedCovariates",
                )
                future[key] = value
        ensure(
            all(len(times) == spec.horizon for times in seen.values()),
            "queries must include the complete horizon for each requested series",
        )
        members = tuple(sorted(seen))
        template = self.request["queries"][0]
        queries = [
            {
                **template,
                "key": f"q{member}",
                "series": f"gfseries:{member}",
                "future_covariates": {
                    name: [future[(name, i)] for i in range(spec.horizon)]
                    for name in spec.future_inputs
                },
            }
            for member in members
        ]
        engine, ref = model
        declared = ref.entry.capability.quantiles
        parsed = forecast_request({**self.request, "queries": queries}, declared)
        try:
            result, grid = engine.forecast(ref, parsed)
        except ModelOutputError:
            raise ServiceError(
                "ModelOutputError", "native forecaster returned invalid forecasts", 503
            ) from None
        ensure(
            grid == tuple(declared),
            "pinned model capabilities changed",
            "ModelMismatch",
        )
        rows = {member: index for index, member in enumerate(members)}
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
