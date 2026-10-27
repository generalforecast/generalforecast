# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Series: groups encoded, extended and kept in slots with their patches first, and forecast from them or whole."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from generalforecast import protocol
from generalforecast._core import CacheKey
from generalforecast.models.series import SeriesGroup
from generalforecast.runtime.device import store_cache
from generalforecast.runtime.memory import tree_bytes
from generalforecast.runtime.paging import CacheLayout
from generalforecast.runtime.series import (
    buckets,
    fixed_leaves,
    horizon_bucket,
    laid_group,
    patch_bucket,
    series_decode,
    series_extend,
)
from generalforecast.worker.calls import _signature
from generalforecast.worker.records import Forecast, SeriesUnit, _SeriesModel

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker


def _grid_of(unit: SeriesUnit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A unit's grid as host arrays, from the bindings' or a frame's form."""
    grid: Any = getattr(unit, "grid", unit)
    return (
        np.asarray(grid.values, dtype=np.float32),
        np.asarray(grid.observed, dtype=bool),
        np.asarray(grid.roles, dtype=np.int32),
    )


def _future_of(forecast: Forecast) -> tuple[np.ndarray, np.ndarray]:
    """A forecast's known future as host arrays."""
    if isinstance(forecast, protocol.Forecast):
        return forecast.future, forecast.future_observed
    return (
        np.asarray(forecast.future.values, dtype=np.float32),
        np.asarray(forecast.future.observed, dtype=bool),
    )


class Series:
    """How a worker forecasts series units."""

    def _forecast(
        self: "Worker", units: Sequence[SeriesUnit], forecasts: Sequence[Forecast]
    ) -> dict[int, np.ndarray]:
        """Forecasts every series from the unit it reads.

        Units a reusable model may keep encode once, in one call per model and
        shape, and pack into their slots with the patches first; their
        forecasts decode from the slots, every group of a model, layout and
        horizon in one call. Units forecast whole, and every unit of a model
        that keeps no cache, run the model's `forecast` with the group's known
        future, one call per model, shape and horizon. Forecasts of one unit
        over one padded horizon share a call, whichever members they are of.

        Returns:
            Each request's forecasts, `(series, horizon, quantiles)`.
        """
        groups: dict[int, SeriesGroup] = {}
        for index, unit in enumerate(units):
            properties = self.series_properties(self._tables[unit.table].model)
            values, observed, roles = _grid_of(unit)
            rows, steps = buckets(
                values.shape[0], values.shape[1], properties, self._minimum
            )
            groups[index] = laid_group(values, observed, roles, rows, steps)
            self._fits_device_series(self._tables[unit.table].model, rows, steps)
            if unit.action == "transient":
                self.stats.transient += 1
        self._encode_series(units, groups)
        whole: dict[tuple[int, int], list[int]] = defaultdict(list)
        cached: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, forecast in enumerate(forecasts):
            unit = units[forecast.unit]
            properties = self.series_properties(self._tables[unit.table].model)
            horizon = horizon_bucket(forecast.horizon, properties)
            if unit.action == "transient" or not properties.reusable:
                whole[forecast.unit, horizon].append(index)
            else:
                cached[forecast.unit, horizon].append(index)
        results: dict[int, np.ndarray] = {}

        def keep(item: tuple[int, int], outputs: Any) -> None:
            unit, _ = item
            # A member's rows are its target's, then one per feature column.
            features = self._tables[units[unit].table].features
            outputs = np.asarray(outputs)
            for index in whole.get(item, []) + cached.get(item, []):
                forecast = forecasts[index]
                row = forecast.target * (1 + features)
                results[index] = outputs[row, : forecast.horizon]

        self._forecast_whole(units, forecasts, groups, whole, keep)
        self._decode_series(units, cached, keep)
        collected: dict[int, list[np.ndarray]] = defaultdict(list)
        for index, forecast in enumerate(forecasts):
            collected[forecast.request].append(results[index])
        return {request: np.stack(rows) for request, rows in collected.items()}

    def _forecast_whole(
        self: "Worker",
        units: Sequence[SeriesUnit],
        forecasts: Sequence[Forecast],
        groups: dict[int, SeriesGroup],
        whole: dict[tuple[int, int], list[int]],
        keep: Callable[[tuple[int, int], Any], None],
    ) -> None:
        """Runs `forecast` for the groups forecast whole, alike ones together.

        Each unit's group takes the known future its forecasts share over
        the padded horizon.
        """
        alike: dict[tuple, list[tuple[tuple[int, int], SeriesGroup]]] = defaultdict(
            list
        )
        for (unit, horizon), indices in whole.items():
            model = self._tables[units[unit].table].model
            padded = groups[unit]
            future, future_observed = _future_of(forecasts[indices[0]])
            # A table without known columns sends no future rows.
            known = future.shape[0] > 0
            group = laid_group(
                *_grid_of(units[unit]),
                padded.values.shape[0],
                padded.values.shape[1],
                future if known else None,
                future_observed if known else None,
                horizon if known else None,
            )
            alike[(model, horizon, _signature((group,)))].append(
                ((unit, horizon), group)
            )
        for (model, horizon, signature), members in alike.items():
            entry = self._series(model)
            forecast = entry.functions.forecast

            def build(forecast: Any = forecast, horizon: int = horizon) -> Any:
                return jax.vmap(lambda p, g: forecast(p, g, horizon), in_axes=(None, 0))

            compiled = self.programs.compiled(("forecast", model, horizon), build)
            items = [(group,) for _, group in members]

            def estimate(
                items: list[tuple] = items, entry: Any = entry, horizon: int = horizon
            ) -> int:
                shape = jax.eval_shape(
                    lambda p, g: entry.functions.forecast(p, g, horizon),
                    entry.parameters,
                    items[0][0],
                )
                return tree_bytes(items[0]) + tree_bytes(shape)

            self._run_items(
                "forecast",
                ("forecast", model, horizon, signature),
                compiled,
                (entry.parameters,),
                items,
                estimate,
                [item for item, _ in members],
                keep,
            )

    def _encode_series(
        self: "Worker", units: Sequence[SeriesUnit], groups: dict[int, SeriesGroup]
    ) -> None:
        """Encodes the series units to keep, and packs each into its slots, patches first; extends the units grown from a kept one."""
        alike: dict[tuple, list[int]] = defaultdict(list)
        extensions = []
        for index, unit in enumerate(units):
            model = self._tables[unit.table].model
            if unit.action == "reuse":
                self.stats.reused += 1
                continue
            if unit.action == "extend":
                extensions.append(index)
                continue
            if unit.action != "encode" or not self.series_properties(model).reusable:
                continue
            alike[(model, _signature((groups[index],)))].append(index)
        self._extend_series(units, groups, extensions)
        for (model, signature), indices in alike.items():
            entry = self._series(model)
            encode = entry.functions.encode
            assert encode is not None
            compiled = self.programs.compiled(
                ("series_encode", model),
                lambda encode=encode: jax.vmap(encode, in_axes=(None, 0)),
            )
            items = [(groups[index],) for index in indices]

            def estimate(
                items: list[tuple] = items,
                encode: Any = encode,
                entry: _SeriesModel = entry,
            ) -> int:
                shape = jax.eval_shape(encode, entry.parameters, items[0][0])
                return tree_bytes(items[0]) + 2 * tree_bytes(shape)

            def keep(index: int, cache: Any) -> None:
                unit = units[index]
                if self._settings is not None:
                    cache = store_cache(cache, self._settings)
                rows, steps = groups[index].values.shape
                model = self._tables[unit.table].model
                layout = self._series_layout(model, rows)
                patches = steps // self.series_properties(model).patch
                self._store(unit.key, unit.slots, layout, patches, cache)
                self._fixed[unit.key] = fixed_leaves(layout, cache)
                self.stats.encoded += 1

            self._run_items(
                "series_encode",
                ("series_encode", model, signature),
                compiled,
                (entry.parameters,),
                items,
                estimate,
                indices,
                keep,
            )

    def _extend_series(
        self: "Worker",
        units: Sequence[SeriesUnit],
        groups: dict[int, SeriesGroup],
        indices: list[int],
    ) -> None:
        """Grows each extended unit's kept encoding by the grid's new patches, read from the slots that hold it, alike ones in one call.

        The grown encoding packs into the unit's own slots with as many
        patches as the source held plus the new ones; the source stays as it
        was. The prefix counts as reused, the unit as extended.
        """
        alike: dict[tuple, list[tuple[int, Any, CacheKey]]] = defaultdict(list)
        for index in indices:
            unit = units[index]
            model = self._tables[unit.table].model
            assert unit.source is not None
            source = self._contexts[unit.source]
            group = groups[index]
            grown = unit.grown
            new = SeriesGroup(
                group.values[:, -grown:],
                group.observed[:, -grown:],
                group.roles,
                None,
                None,
                group.num_rows,
            )
            grown_patches = grown // self.series_properties(model).patch
            held = self._patch_bucket(model, source.rows)
            stored_at = self._patch_bucket(model, source.rows + grown_patches)
            alike[
                (
                    model,
                    source.layout,
                    held,
                    grown_patches,
                    stored_at,
                    _signature((new,)),
                )
            ].append((index, new, unit.source))
        for (
            model,
            layout,
            held,
            grown_patches,
            stored_at,
            _,
        ), members in alike.items():
            entry = self._series(model)
            stored = False

            def build(
                entry: _SeriesModel = entry,
                stored: bool = stored,
                layout: CacheLayout = layout,
                stored_at: int = stored_at,
            ) -> Any:
                return series_extend(entry.functions, stored, layout, stored_at)

            compiled = self.programs.compiled(
                ("series_extend", model, layout, stored_at), build
            )
            # Every source's bucket is read; its real patches are the count.
            positions = np.arange(held, dtype=np.int32)
            blocks = np.stack(
                [
                    np.asarray(self._contexts[source].slots, dtype=np.int32).reshape(
                        -1, layout.slots_per_row
                    )
                    for _, _, source in members
                ]
            )
            device_source, places = self._device.source(blocks)
            items = [
                (
                    places[at],
                    positions,
                    np.int32(self._contexts[source].rows),
                    self._fixed[source],
                    new,
                )
                for at, (_, new, source) in enumerate(members)
            ]

            def estimate(
                layout: CacheLayout = layout,
                patches: int = held + stored_at,
                items: list[tuple] = items,
            ) -> int:
                return (
                    2 * patches * layout.row_bytes
                    + tree_bytes(items[0][3])
                    + tree_bytes(items[0][4])
                )

            def keep(
                index: int,
                cache: Any,
                layout: CacheLayout = layout,
                grown_patches: int = grown_patches,
            ) -> None:
                unit = units[index]
                assert unit.source is not None
                if self._settings is not None:
                    cache = store_cache(cache, self._settings)
                patches = self._contexts[unit.source].rows + grown_patches
                self._store(unit.key, unit.slots, layout, patches, cache)
                self._fixed[unit.key] = fixed_leaves(layout, cache)
                self.stats.extended += 1
                self.stats.reused += 1

            self._run_items(
                "series_extend",
                ("series_extend", model, layout, held, stored_at, grown_patches),
                compiled,
                (entry.parameters, device_source),
                items,
                estimate,
                [index for index, _, _ in members],
                keep,
            )

    def _patch_bucket(self: "Worker", model: int, patches: int) -> int:
        """The patches an entry of `patches` real ones is stored at, as the node sizes it."""
        return patch_bucket(patches, self.series_properties(model), self._minimum)

    def _decode_series(
        self: "Worker",
        units: Sequence[SeriesUnit],
        cached: dict[tuple[int, int], list[int]],
        keep: Callable[[tuple[int, int], Any], None],
    ) -> None:
        """Decodes each unit over each padded horizon from its slots, alike ones in one call."""
        alike: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        for unit, horizon in cached:
            model = self._tables[units[unit].table].model
            context = self._contexts[units[unit].key]
            bucket = self._patch_bucket(model, context.rows)
            alike[(model, context.layout, bucket, horizon)].append((unit, horizon))
        for (model, layout, patches, horizon), items_keys in alike.items():
            entry = self._series(model)
            # Forecasters read caches as `encode` returned them, so stored
            # dtypes are loaded back first.
            stored = False

            def build(
                entry: _SeriesModel = entry,
                stored: bool = stored,
                layout: CacheLayout = layout,
                horizon: int = horizon,
            ) -> Any:
                return series_decode(entry.functions, stored, layout, horizon)

            compiled = self.programs.compiled(
                ("series_decode", model, layout, horizon), build
            )
            # Every entry's bucket is read; its real patches are the count,
            # so one program serves every count in the bucket.
            positions = np.arange(patches, dtype=np.int32)
            keys = [units[unit].key for unit, _ in items_keys]
            blocks = np.stack(
                [
                    np.asarray(self._contexts[key].slots, dtype=np.int32).reshape(
                        -1, layout.slots_per_row
                    )
                    for key in keys
                ]
            )
            source, places = self._device.source(blocks)
            items = [
                (
                    places[at],
                    positions,
                    np.int32(self._contexts[key].rows),
                    self._fixed[key],
                )
                for at, key in enumerate(keys)
            ]

            def estimate(
                layout: CacheLayout = layout,
                patches: int = patches,
                fixed: Any = self._fixed[keys[0]],
            ) -> int:
                return 2 * patches * layout.row_bytes + tree_bytes(fixed)

            self._run_items(
                "series_decode",
                ("series_decode", model, layout, patches, horizon),
                compiled,
                (entry.parameters, source),
                items,
                estimate,
                items_keys,
                keep,
            )
