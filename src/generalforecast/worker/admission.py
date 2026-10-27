# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Memory admission: what a model takes, measured once, and refusing what would not fit.

Sizes come from tracing and compiling the model, never from running it, and
are kept for later processes under the model's spec. A context, encode or
fit the device could not hold is refused up front, naming what reads less.
"""

import logging
import sys
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from generalforecast.constants import (
    AUTOTUNE_SHARE,
    FIT_CELL_BYTES,
    FIT_ROW_SCALE,
    MEASURED_ROWS,
)
from generalforecast.runtime.memory import (
    EncodeProfile,
    FitProfile,
    MemoryProfile,
    NoMemoryAnalysis,
    WorkingProfile,
    decode_profile,
    encode_bytes,
    encode_profile,
    encode_widths,
    fit_profile,
    memory_profile,
)
from generalforecast.runtime.paging import CacheLayout, cache_layout
from generalforecast.runtime.series import (
    SeriesBytes,
    abstract_group,
    series_bytes,
    series_layout,
)
from generalforecast.worker.records import Unit

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


class Admission:
    """How a worker measures its models and refuses work past its memory."""

    def profile(self: "Worker", model: int, features: int) -> MemoryProfile:
        """A loaded model's sizes for tables with `features` columns, as stored."""
        if (model, features) not in self._profiles:
            entry = self._tabular(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._profiles[model, features] = self._measured(
                model,
                ("profile", features),
                lambda: memory_profile(
                    entry.functions, entry.parameters, features, cache_dtype
                ),
            )
        return self._profiles[model, features]

    def series_profile(self: "Worker", model: int) -> SeriesBytes:
        """A loaded forecaster's cache bytes per row and patch, as stored."""
        if model not in self._series_profiles:
            entry = self._series(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._series_profiles[model] = series_bytes(
                entry.functions, entry.parameters, cache_dtype
            )
        return self._series_profiles[model]

    def _fits_device_series(self: "Worker", model: int, rows: int, steps: int) -> None:
        """Refuses a group whose cache alone would not fit the device's memory.

        Raises:
            ValueError: Naming what reads less.
        """
        sizes = self.series_profile(model)
        patches = steps // self.series_properties(model).patch
        bytes = patches * (sizes.per_row * rows + sizes.per_patch)
        bytes += sizes.fixed_per_row * rows + sizes.fixed
        if bytes > self.budget.device:
            raise ValueError(
                f"a group of {rows} rows over {steps} steps takes {bytes >> 20} MiB "
                f"as the model caches it, but the device holds "
                f"{self.budget.device >> 20} MiB; use smaller groups, fewer "
                "covariates, or a device with more memory"
            )

    def check_series(self: "Worker", model: int, features: int, rows: int = 2) -> None:
        """Traces a forecast of a small group, without running it.

        Raises:
            ValueError: If the model refuses such a group.
        """
        entry = self._series(model)
        properties = entry.functions.properties(entry.parameters)
        group = abstract_group(rows * (1 + features), 2 * properties.patch)
        try:
            jax.eval_shape(
                lambda parameters, group: entry.functions.forecast(
                    parameters, group, properties.output_patch
                ),
                entry.parameters,
                group,
            )
        except ValueError as error:
            # A model's cap on a group's rows, worded as the planner words
            # its refusal; anything else is the model's own complaint.
            if "passes the model's" not in str(error):
                raise
            raise ValueError(
                f"a group of {rows} series takes {rows * (1 + features)} rows with "
                f"the covariates, past what the model serves per group ({error})"
            ) from error

    def _series_layout(self: "Worker", model: int, rows: int) -> CacheLayout:
        """How a forecaster's caches pack into slots, for groups of `rows` padded rows."""
        if (model, rows) not in self._series_layouts:
            entry = self._series(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._series_layouts[model, rows] = series_layout(
                entry.functions, entry.parameters, rows, self._slot_bytes, cache_dtype
            )
        return self._series_layouts[model, rows]

    def decode_profile(self: "Worker", model: int) -> WorkingProfile:
        """A loaded model's decode of its smallest query chunk, measured once."""
        if model not in self._decodes:
            entry = self._tabular(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._decodes[model] = self._measured(
                model,
                ("decode", self._size(1)),
                lambda: decode_profile(
                    entry.functions,
                    entry.parameters,
                    self._stage("decode", model, 0),
                    self._size(1),
                    cache_dtype,
                ),
            )
        return self._decodes[model]

    def encode_profile(
        self: "Worker", model: int, features: int, fit: bool = False
    ) -> EncodeProfile | None:
        """A loaded model's encode of a context, measured once for widths around `features`.

        With `fit`, the encode fits the context's own state first. None where
        XLA cannot measure it.
        """
        widths = encode_widths(features)
        if (model, fit, widths) not in self._encodes:
            entry = self._tabular(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype

            def measure() -> EncodeProfile | None:
                try:
                    return encode_profile(
                        entry.functions, entry.parameters, fit, cache_dtype, widths
                    )
                except Exception as error:  # noqa: BLE001
                    logger.warning(
                        "could not measure the model's encode at widths %s (%s); "
                        "each context's is measured at its own size",
                        widths,
                        error,
                    )
                    return None

            self._encodes[model, fit, widths] = self._measured(
                model, ("encode_profile", fit, widths), measure
            )
        return self._encodes[model, fit, widths]

    def encode_bytes(
        self: "Worker", model: int, features: int, rows: int, fit: bool = False
    ) -> int | None:
        """Bytes a loaded model's encode of a context of `rows` padded rows takes at most.

        From the model's profile where it holds, and otherwise measured at the
        context's own width and size. None where XLA gives no measure.

        Raises:
            ValueError: If the encode cannot even be compiled to measure it.
        """
        per_row = self.profile(model, features).cache_bytes_per_row
        if rows >= MEASURED_ROWS:
            profile = self.encode_profile(model, features, fit)
            if profile is not None:
                return profile.working_bytes(per_row, rows)
        key = (model, fit, features, rows)
        if key not in self._encode_bytes:
            self._remember(model, ("encode", features, rows, fit))
            entry = self._tabular(model)
            try:
                cache_dtype = (
                    None if self._settings is None else self._settings.cache_dtype
                )
                measured = self._measured(
                    model,
                    ("encode", features, rows, fit),
                    lambda: encode_bytes(
                        entry.functions,
                        entry.parameters,
                        features,
                        rows,
                        fit,
                        cache_dtype,
                    ),
                )
            except NoMemoryAnalysis as error:
                logger.warning(
                    "could not measure the model's encode (%s); contexts are "
                    "refused only by their decode",
                    error,
                )
                measured = None
            except Exception as error:
                raise ValueError(
                    f"could not compile the encode of a context of {rows} rows to "
                    f"measure it ({error}); use a strategy that reads fewer rows, "
                    "such as neighbors, or a device with more memory"
                ) from error
            self._encode_bytes[key] = measured
        return self._encode_bytes[key]

    def fit_profile(self: "Worker", model: int, features: int) -> FitProfile:
        """What a loaded model's fit takes for tables with `features` columns, measured once."""
        if (model, features) not in self._fit_profiles:
            entry = self._tabular(model)
            self._fit_profiles[model, features] = fit_profile(
                entry.functions, entry.parameters, features, self._fit_cache
            )
        return self._fit_profiles[model, features]

    def fit_rows(self: "Worker", model: int, features: int) -> tuple[int, int]:
        """The most rows a fit takes within the call budget, for tables with `features` columns.

        Each is a padded size, so a fit padded to its size stays within it.

        Args:
            model: A loaded model's id.
            features: Number of feature columns.

        Returns:
            The most rows a state alone is fit on, and the most a fit embeds
            in the call that fits its state; `sys.maxsize` for any.
        """
        profile = self.fit_profile(model, features)
        return (
            self._padded_floor(profile.state_rows(self._call_bytes)),
            self._padded_floor(profile.fused_rows(self._call_bytes)),
        )

    def fit_bound(self: "Worker", model: int, features: int) -> int:
        """The most rows a table's fit takes before `fit_rows` is measured, without compiling.

        A padded size at which a fit embedding its rows takes the call budget
        even at `FIT_CELL_BYTES` a cell and `FIT_ROW_SCALE` times its rows'
        representations, far more than any built-in model's fit holds.

        Args:
            model: A loaded model's id.
            features: Number of feature columns.

        Returns:
            The rows, a padded size or none.
        """
        rows = self.profile(model, features).row_bytes
        per_row = FIT_CELL_BYTES * (features + 4) + FIT_ROW_SCALE * rows
        return self._padded_floor(self._call_bytes // per_row)

    def _padded_floor(self: "Worker", rows: int) -> int:
        """The largest padded size of at most `rows` rows, 0 if none, `sys.maxsize` for any."""
        if rows >= sys.maxsize:
            return sys.maxsize
        if rows < self._minimum:
            return 0
        return self._minimum << ((rows // self._minimum).bit_length() - 1)

    def _summary_chunk(self: "Worker", model: int, width: int) -> int | None:
        """Cells a fit's column summary reads a step within the call budget, if it has one."""
        if not self.properties(model).column_summary:
            return None
        return self.fit_profile(model, width).summary_chunk(self._call_bytes)

    def check(self: "Worker", model: int, num_classes: int, features: int) -> None:
        """Traces a fit for a table of this task and width, without running it.

        Args:
            model: A loaded model's id.
            num_classes: Number of classes, or 0 for regression.
            features: Number of feature columns.

        Raises:
            ValueError: If the model refuses such a table, so it is refused
                when created rather than at its first fit.
        """
        if (model, num_classes, features) in self._checked:
            return
        size = self._size(1)

        def check() -> bool:
            targets = jnp.zeros(size, jnp.int32 if num_classes else jnp.float32)
            jax.eval_shape(
                self._stage("fit", model, num_classes),
                self._models[model].parameters,
                jnp.zeros((size, features), jnp.float32),
                targets,
                1,
                None,
            )
            return True

        # Tracing the model takes most of a second, so a table it accepted is
        # remembered for later processes; one it refuses is traced again.
        self._measured(model, ("check", num_classes, features, size), check)
        self._checked.add((model, num_classes, features))

    def layout(self: "Worker", model: int, features: int) -> CacheLayout:
        """How a loaded model's contexts pack into slots, for tables with `features` columns.

        Raises:
            ValueError: If a slot cannot hold one of the model's context rows.
        """
        if (model, features) not in self._layouts:
            entry = self._tabular(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._layouts[model, features] = self._measured(
                model,
                ("layout", features),
                lambda: cache_layout(
                    entry.functions,
                    entry.parameters,
                    features,
                    self._slot_bytes,
                    cache_dtype,
                ),
            )
        return self._layouts[model, features]

    def _fits_device(self: "Worker", unit: Unit) -> None:
        """Refuses a unit whose encode, or whose context and decode, would not fit the device.

        Beside the pool, the device must hold the context's encode: its inputs,
        temporaries and cache, which the call's lone item is taken from in
        place. It must also hold the working memory of decoding the
        smallest chunk of queries against the context. A kept context's decode
        reads it, and any own state, from the pool into that working memory; a
        transient context's decode takes a copy of it into its call, beside
        the one held. On an accelerator, XLA's autotuner also needs headroom.
        Otherwise the request would fail for want of memory, after a long fit,
        with nothing said about what to do instead.

        Raises:
            ValueError: Naming the strategies that read fewer rows.
        """
        table = self._tables[unit.table]
        per_row = self.profile(table.model, table.features).cache_bytes_per_row
        size = self._size(len(unit.rows))
        context = self._context_bytes(unit)
        decode = self.decode_profile(table.model)
        working = decode.working_bytes(per_row, size)
        measured = self.encode_bytes(
            table.model, table.features, size, table.own is not None
        )
        encode = 0 if measured is None else measured
        if unit.action == "transient":
            needed = max(encode, 2 * context + working)
        else:
            # A unit's own state is read from its slots and unpacked beside it.
            state = 0 if table.own is None else 2 * table.own.slots * self._slot_bytes
            needed = max(encode, working + state)
        if not self.budget.cpu:
            needed += max(encode, working) // AUTOTUNE_SHARE
        room = self.budget.device - self._pool_bytes
        if needed > room:
            encoding = (
                "" if measured is None else f"encoding it {encode >> 20} MiB and "
            )
            raise ValueError(
                f"a context of {len(unit.rows)} rows takes {context >> 20} MiB as the "
                f"model caches it, {encoding}decoding "
                f"{decode.queries} queries against it {working >> 20} MiB, which "
                f"needs {needed >> 20} MiB, but the device has {room >> 20} MiB "
                "beside the contexts it keeps; use a strategy that reads fewer "
                "rows, such as neighbors, or a device with more memory"
            )

    def _context_bytes(self: "Worker", unit: Unit) -> int:
        """Bytes of a unit's context as the model caches it, padding and own state included."""
        table = self._tables[unit.table]
        profile = self.profile(table.model, table.features)
        own = 0 if table.own is None else table.own.slots * self._slot_bytes
        return self._size(len(unit.rows)) * profile.cache_bytes_per_row + own
