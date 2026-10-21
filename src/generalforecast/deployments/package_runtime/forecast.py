"""Aligned grouped forecast contexts, with explicit complete future query rows."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from generalforecast.deployments import catalog_forecast as native
from generalforecast.deployments.contracts import ServiceError
from generalforecast.models import inference
from generalforecast.models.catalog import V1_POLICY
from generalforecast.models.grouping import AlignmentError, aligned_group, instant
from generalforecast.models.outputs import DEFAULT_QUANTILES, quantile_index, quantiles
from generalforecast.packages import PackageLock
from generalforecast.packages.schema import TableSchema
from generalforecast.packages.validation import ensure


def _run(model: Any, group: Any, horizon: int) -> np.ndarray:
    try:
        return inference.forecast(model, group, horizon)
    except ValueError:
        raise ServiceError(
            "ModelOutputError", "native forecaster returned invalid forecasts", 503
        ) from None


def _levels(model: Any) -> tuple[float, ...]:
    functions, parameters = model
    return tuple(functions.properties(parameters).quantiles)


@dataclass(frozen=True)
class ForecastContext:
    """Independent aligned series tensors and a warmed native forecast."""

    identities: tuple[tuple, ...]
    origin: int | float
    step: int | float
    group: Any
    warmed: np.ndarray

    @classmethod
    def build(cls, lock: PackageLock, model: Any, table: pa.Table, schema: TableSchema):
        """Validates and warms a related group without inventing a reusable encoder."""
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
        ends: dict[tuple, int | float] = {}
        rows = []
        for index, row in enumerate(table.to_pylist()):
            identity = tuple(row[name] for name in spec.series)
            moment = instant(row[spec.time])
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
        assert schema.frequency is not None
        step = (
            int(schema.frequency)
            if type(origin) is int and schema.frequency.is_integer()
            else schema.frequency
        )
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
            "model": lock.model.name,
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
        settings, series = native.validate(request, _levels(model))
        try:
            group = aligned_group(
                series,
                query["group"],
                settings["known_covariates"],
                origin,
                step,
                spec.horizon,
                query["future_covariates"],
                model[0].properties(model[1]).patch,
            )
        except AlignmentError as error:
            raise ServiceError(error.code, str(error), 422) from None
        # A forecaster may have no separable context encoder. Warm the complete native kernel
        # and retain its immutable aligned context tensors instead of inventing a KV cache.
        warmed = _run(model, group, spec.horizon)
        return cls(identities, origin, step, group, warmed)

    def predict(self, lock: PackageLock, model: Any, table: pa.Table) -> dict[str, Any]:
        """Aligns future rows and reruns only when known future values affect outputs."""
        import jax.numpy as jnp

        spec = lock.spec
        assert spec.horizon is not None and spec.time is not None
        positions: list[tuple[int, int]] = []
        seen: dict[int, set[int]] = {}
        future: dict[tuple[str, int], float] = {}
        for row in table.to_pylist():
            identity = tuple(row[name] for name in spec.series)
            ensure(identity in self.identities, "query series is missing from context")
            series = self.identities.index(identity)
            offset = (instant(row[spec.time]) - self.origin) / self.step - 1
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
        result = self.warmed
        if spec.future_inputs:
            covariates = sorted(set(spec.past_inputs) | set(spec.future_inputs))
            values = np.zeros(self.group.future.shape, np.float32)
            for name in spec.future_inputs:
                values[len(self.identities) + covariates.index(name)] = [
                    future[(name, i)] for i in range(spec.horizon)
                ]
            result = _run(
                model, self.group._replace(future=jnp.asarray(values)), spec.horizon
            )
        levels = spec.quantiles or DEFAULT_QUANTILES
        grid = _levels(model)
        return {
            "value": [
                float(result[s, p, quantile_index(0.5, grid)]) for s, p in positions
            ],
            "quantiles": [quantiles(result[s, p], levels, grid) for s, p in positions],
        }
