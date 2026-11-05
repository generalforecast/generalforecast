# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Memory admission: what a model takes, measured once, and refusing what would not fit.

Sizes come from tracing and compiling the model, never from running it, and
are kept for later processes under the model's spec. A context, encode or
fit the device could not hold is refused up front, naming what reads less.
"""

import logging
import math
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from generalforecast.constants import (
    AUTOTUNE_SHARE,
    FIT_CELL_BYTES,
    FIT_ROW_SCALE,
    MEASURE_OVERSTATE,
    MEASURED_ROWS,
    MEMORY_ROWS_LIMIT,
    MEMORY_ROWS_MEASURES,
    WORKING_MARGIN,
)
from generalforecast.runtime.memory import (
    EncodeProfile,
    FitProfile,
    MemoryProfile,
    NoMemoryAnalysis,
    WorkingProfile,
    concurrently,
    decode_profile,
    encode_bytes,
    encode_profile,
    encode_widths,
    fit_profile,
    memory_profile,
)
from generalforecast.runtime.paging import CacheLayout, cache_layout
from generalforecast.runtime.programs import padded_width
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

# The most rows `Admission.whole_rows` searches up to.
WHOLE_ROWS_LIMIT = 1 << 26


def _gather_bytes(layout: CacheLayout, size: int) -> int:
    """Bytes a paged decode's copy of a kept context of `size` padded rows takes; see `Admission._gather_bytes`."""
    return math.ceil(WORKING_MARGIN * size * layout.row_bytes)


class _Unmeasured(Exception):
    """A fit profile XLA could not measure, which is not kept."""


class Admission:
    """How a worker measures its models and refuses work past its memory."""

    def width(self: "Worker", model: int, features: int) -> int:
        """The width a loaded model runs a table of `features` columns at: padded to a shared width where its properties allow.

        A padded width is its own, so the width of a width is itself.
        """
        if not self.properties(model).padded_width:
            return features
        return padded_width(features)

    def profile(self: "Worker", model: int, features: int) -> MemoryProfile:
        """A loaded model's sizes for tables with `features` columns, as stored."""
        features = self.width(model, features)
        entry = self._tabular(model)
        cache_dtype = None if self._settings is None else self._settings.cache_dtype
        return self._profiles.get(
            (model, features),
            lambda: self._measured(
                model,
                ("profile", features),
                lambda: memory_profile(
                    entry.functions, entry.parameters, features, cache_dtype
                ),
            ),
        )

    def series_profile(self: "Worker", model: int) -> SeriesBytes:
        """A loaded forecaster's cache bytes per row and patch, as stored."""
        entry = self._series(model)
        cache_dtype = None if self._settings is None else self._settings.cache_dtype
        return self._series_profiles.get(
            model, lambda: series_bytes(entry.functions, entry.parameters, cache_dtype)
        )

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
        entry = self._series(model)
        cache_dtype = None if self._settings is None else self._settings.cache_dtype
        return self._series_layouts.get(
            (model, rows),
            lambda: series_layout(
                entry.functions, entry.parameters, rows, self._slot_bytes, cache_dtype
            ),
        )

    def decode_profile(self: "Worker", model: int) -> WorkingProfile:
        """A loaded model's decode of its smallest query chunk, measured once."""
        entry = self._tabular(model)
        cache_dtype = None if self._settings is None else self._settings.cache_dtype
        return self._decodes.get(
            model,
            lambda: self._measured(
                model,
                ("decode", self._size(1)),
                lambda: decode_profile(
                    entry.functions,
                    entry.parameters,
                    self._stage("decode", model, 0),
                    self._size(1),
                    cache_dtype,
                ),
            ),
        )

    def encode_profile(
        self: "Worker", model: int, features: int, fit: bool = False
    ) -> EncodeProfile | None:
        """A loaded model's encode of a context, measured once for widths around `features`.

        With `fit`, the encode fits the context's own state first. None where
        XLA cannot measure it.
        """
        widths = encode_widths(features)
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

        return self._encodes.get(
            (model, fit, widths),
            lambda: self._measured(model, ("encode_profile", fit, widths), measure),
        )

    def encode_bytes(
        self: "Worker",
        model: int,
        features: int,
        rows: int,
        fit: bool = False,
        exact: bool = False,
    ) -> int | None:
        """Bytes a loaded model's encode of a context of `rows` padded rows takes at most.

        From the model's profile where it holds, and otherwise, or with
        `exact`, measured at the context's own width and size. None where XLA
        gives no measure.

        Raises:
            ValueError: If the encode cannot even be compiled to measure it.
        """
        per_row = self.profile(model, features).cache_bytes_per_row
        if rows >= MEASURED_ROWS and not exact:
            profile = self.encode_profile(model, features, fit)
            if profile is not None:
                return profile.working_bytes(per_row, rows)
        entry = self._tabular(model)

        def measure() -> int | None:
            self._remember(model, ("encode", features, rows, fit))
            try:
                cache_dtype = (
                    None if self._settings is None else self._settings.cache_dtype
                )
                return self._measured(
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
                return None
            except Exception as error:
                raise ValueError(
                    f"could not compile the encode of a context of {rows} rows to "
                    f"measure it ({error}); use a strategy that reads fewer rows, "
                    "such as neighbors, or a device with more memory"
                ) from error

        return self._encode_bytes.get((model, fit, features, rows), measure)

    def fit_profile(self: "Worker", model: int, features: int) -> FitProfile:
        """What a loaded model's fit takes for tables with `features` columns, measured once.

        Kept with the model's other measures, so a later process reads it
        without tracing the four fits it measures.
        """
        entry = self._tabular(model)
        unmeasured = FitProfile(features)

        def measure() -> FitProfile:
            profile = fit_profile(
                entry.functions, entry.parameters, features, self._fit_cache
            )
            # A fit XLA could not measure is not kept for later processes,
            # which measure it again.
            if profile == unmeasured:
                raise _Unmeasured
            return profile

        def kept() -> FitProfile:
            try:
                # Keyed apart from fit profiles kept before they measured a
                # step, which would read with none and size steps by the
                # fused fit; those read as absent, and are measured again.
                key = ("fit_profile", features, "step")
                return self._measured(model, key, measure)
            except _Unmeasured:
                return unmeasured

        return self._fit_profiles.get((model, features), kept)

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
        profile = self.fit_profile(model, self.width(model, features))
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
        features = self.width(model, features)
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
        features = self.width(model, features)
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
        self._checked.get(
            (model, num_classes, features),
            lambda: self._measured(
                model, ("check", num_classes, features, size), check
            ),
        )

    def layout(self: "Worker", model: int, features: int) -> CacheLayout:
        """How a loaded model's contexts pack into slots, for tables with `features` columns.

        Raises:
            ValueError: If a slot cannot hold one of the model's context rows.
        """
        features = self.width(model, features)
        entry = self._tabular(model)
        cache_dtype = None if self._settings is None else self._settings.cache_dtype
        return self._layouts.get(
            (model, features),
            lambda: self._measured(
                model,
                ("layout", features),
                lambda: cache_layout(
                    entry.functions,
                    entry.parameters,
                    features,
                    self._slot_bytes,
                    cache_dtype,
                ),
            ),
        )

    def _fits_device(self: "Worker", unit: Unit, fused: bool = False) -> None:
        """Refuses a unit whose encode, or whose context and decode, would not fit the device.

        Beside the pool, the device must hold the context's encode: its inputs,
        temporaries and cache, which the call's lone item is taken from in
        place. It must also hold the working memory of decoding the
        smallest chunk of queries against the context, which depends on how
        the decode reads the context:

        - A kept context's paged decode gathers its rows, and any own state,
          from the pool into a copy of their own (`_gather_bytes`), beside
          that working memory.
        - A transient context decoded apart from its encode takes a copy of
          its cache into the decode's call, beside the one held: its cache
          twice.
        - With `fused`, a transient context whose few queries decode in the
          one program that gathers and encodes it (`_encode_decode`), whose
          cache lives only inside that program: its encode without the cache
          held as an output, and that decode's working memory twice. Probes
          of Mitra and TabICL v2 measured the program below this.

        A transient context pads to `_transient_size`, as its encode pads it,
        on an accelerator; on a CPU, every unit is charged at its bucket. On
        an accelerator, XLA's autotuner also needs headroom. Otherwise the
        request would fail for want of memory, after a long fit, with nothing
        said about what to do instead.

        Args:
            unit: The unit to admit.
            fused: Whether its decode will take the fused path: a transient
                context whose step's rows are on the device, read by at most
                `FUSED_QUERIES` queries.

        Raises:
            ValueError: Naming the strategies that read fewer rows.
        """
        table = self._tables[unit.table]
        per_row = self.profile(table.model, table.features).cache_bytes_per_row
        bucket = self._size(len(unit.rows))
        size = self._admitted_rows(unit)
        transient = unit.action == "transient"
        accelerator = not self.budget.cpu
        fused = fused and transient and table.own is None and accelerator
        own = 0 if table.own is None else table.own.slots * self._slot_bytes
        context = size * per_row + own
        decode = self.decode_profile(table.model)
        working = decode.working_bytes(per_row, size)
        room = self.budget.device - self._pool_bytes
        fit = table.own is not None
        # A unit's own state is read from its slots and unpacked beside it.
        state = 2 * own
        gather = (
            self._gather_bytes(table.model, table.features, bucket)
            if accelerator and unit.action not in ("transient", "fuse")
            else 0
        )

        def needs(measured: int | None) -> int:
            encode = 0 if measured is None else measured
            if fused:
                # The program's cache is a temporary the decode reads, not
                # an output beside it; unmeasured, the cache at least.
                held = context if measured is None else max(0, encode - context)
                needed = held + 2 * working
            elif transient:
                needed = max(encode, 2 * context + working)
            else:
                needed = max(encode, gather + working + state)
            if accelerator:
                needed += max(encode, working) // AUTOTUNE_SHARE
            return needed

        # A transient context is charged at its own padded size by the
        # encode profile's line, which compiles nothing. Below where that
        # line holds, and to measure it exactly, it takes the measure of its
        # bucket, which a kept context of its rows compiles, and which holds
        # at least as much: so admission compiles no new shape.
        line_rows = bucket
        if (
            size < bucket
            and size >= MEASURED_ROWS
            and self.encode_profile(table.model, table.features, fit) is not None
        ):
            line_rows = size
        measured = self.encode_bytes(table.model, table.features, line_rows, fit)
        needed = needs(measured)
        # The profile's line through small contexts can overstate a large
        # one's encode several times, so a context it refuses is measured at
        # its own size before it is refused: unless what it needs besides
        # its encode, its cache and decode, does not fit either, or the line
        # puts it so far past the device that no overstatement would bring it
        # in. Those would compile a program larger than the device to refuse.
        if needed > room and needs(None) <= room and needed <= MEASURE_OVERSTATE * room:
            exact = self.encode_bytes(
                table.model, table.features, bucket, fit, exact=True
            )
            if line_rows == bucket or measured is None:
                measured = exact
            elif exact is not None:
                measured = min(measured, exact)
            needed = needs(measured)
        encode = 0 if measured is None else measured
        if needed > room:
            encoding = (
                "" if measured is None else f"encoding it {encode >> 20} MiB and "
            )
            reading = " in one call with its encode" if fused else ""
            raise ValueError(
                f"a context of {len(unit.rows)} rows ({unit.action}, padded to "
                f"{size}) takes {context >> 20} MiB as the "
                f"model caches it, {encoding}decoding "
                f"{decode.queries} queries against it{reading} {working >> 20} MiB, "
                f"which needs {needed >> 20} MiB, but the device has "
                f"{room >> 20} MiB beside the contexts it keeps; use a strategy "
                "that reads fewer rows, such as neighbors, or a device with "
                "more memory"
            )

    def _admitted_rows(self: "Worker", unit: Unit) -> int:
        """Rows admission charges a unit's context at: as its encode pads it.

        A transient context gathered from pages pads to `_transient_size`
        on an accelerator; any other, and every one on a CPU, to its bucket.
        """
        table = self._tables[unit.table]
        if unit.action == "transient" and table.own is None and not self.budget.cpu:
            return self._transient_size(len(unit.rows))
        return self._size(len(unit.rows))

    def _gather_bytes(self: "Worker", model: int, features: int, size: int) -> int:
        """Bytes a paged decode's copy of a kept context of `size` padded rows takes, with a margin.

        The decode reads the context's blocks from the pool into a cache of
        its own (`decodes.py`'s "a context and the working copy beside it"),
        which the decode profile, measured with the cache as an argument,
        leaves out. Probes measured the paged decode's temporaries at 1.04
        times the padded cache for Mitra-2, and at the cache and the decode's
        working memory for TabICL v2; `WORKING_MARGIN` covers the first.
        """
        return _gather_bytes(self.layout(model, features), size)

    def memory_rows(
        self: "Worker",
        model: int,
        features: int,
        beside: Callable[[], object] | None = None,
    ) -> int | None:
        """The most rows a context of a table this wide holds on this device, a padded size; None for any.

        A context past it would not fit as `_fits_device` checks: its cache
        kept in the pool, and its encode, or the decode of a chunk of queries
        and its copy of the context's slots, beside the pool; or, put off for
        want of room in the pool and decoded apart from its encode, its
        encode or its cache twice and that decode. The fused path a put-off
        context's few queries may take holds less, so is not what bounds
        it. So an adaptive table resolves
        against it as against the model's own cap, reading a core and each
        query's neighbors rather than a context the device cannot hold.
        Padded sizes grow from the smallest measured, bucket by bucket, until
        one does not fit, by the model's profiles; where even that one does
        not, they shrink from it until one does, so a model whose contexts
        are large a row reads fewer rows rather than none. A size the
        profiles refuse is measured at its own size before it ends the
        search, since the profiles' line through small contexts can
        overstate a large one.
        None on a CPU, whose memory the host's caches share, and where XLA
        gives no measure.

        `beside`, if given, runs on a thread of its own beside the encode
        profile's compiles when the decode profile is measured already, as
        for a model's second width: compiles the caller needs next then fill
        a machine the four alone leave half idle. Beside both profiles'
        eight, as for a model's first width, they would only slow both, and
        `beside` is left for the caller to run when it needs it.
        """
        if self.budget.cpu or not self._pool_bytes:
            return None
        # As the table's steps run: at its padded width.
        features = self.width(model, features)
        per_row = self.profile(model, features).cache_bytes_per_row
        layout = self.layout(model, features)
        if beside is not None and model in self._decodes:
            threading.Thread(
                target=beside, name="beside-memory-rows", daemon=True
            ).start()
        # The two profiles every size reads compile side by side, then are read
        # as each is kept.
        concurrently(
            [
                lambda: self.decode_profile(model),
                lambda: self.encode_profile(model, features),
            ]
        )
        decode = self.decode_profile(model)
        room = self.budget.device - self._pool_bytes
        measures = 0
        ahead_taken = False

        def fits(size: int, exact: bool) -> bool | None:
            if layout.slots(size) * self._slot_bytes > self._pool_bytes:
                return False
            encode = self.encode_bytes(model, features, size, exact=exact)
            if encode is None:
                return None
            working = decode.working_bytes(per_row, size)
            # As `_fits_device` holds a kept unit, its decode gathering its
            # slots, and a put-off one decoded apart from its encode.
            needed = max(encode, besides(size, working))
            return needed + max(encode, working) // AUTOTUNE_SHARE <= room

        def besides(size: int, working: int) -> int:
            """What a context of `size` rows holds besides its encode, kept or put off."""
            gather = _gather_bytes(layout, size)
            return max(2 * size * per_row, gather) + working

        def measurable(size: int) -> bool:
            """Whether an exact measure of `size`'s encode could make it fit.

            Not where its slots outgrow the pool, or what it holds besides its
            encode, its cache twice and a decode, outgrows the room: `fits`
            refuses it whatever its encode takes, so measuring it, a compile
            of an encode of that many rows, would be for nothing. TabICL v2's
            contexts past about 600k rows on an H100 are such, and measuring
            three of them took 5 s of every new width's first table create.
            """
            if layout.slots(size) * self._slot_bytes > self._pool_bytes:
                return False
            working = decode.working_bytes(per_row, size)
            return besides(size, working) + working // AUTOTUNE_SHARE <= room

        def ahead(size: int, larger: bool) -> None:
            """Measures `size` and the sizes the search would measure after it, side by side.

            A search that measures one size and finds it fits measures the
            next padded size, then the one after, each a compile of an encode
            many times the device's working sizes; at once they take about
            as long as one. A size the search then stops short of was
            measured for nothing.
            """
            sizes = [size]
            last = size
            while len(sizes) < MEMORY_ROWS_MEASURES - measures:
                step = self._size(last + 1) if larger else self._size(last // 2)
                if step == last or (not larger and step < self._minimum):
                    break
                last = step
                if measurable(step):
                    sizes.append(step)
                elif larger:
                    # Larger sizes hold more beside their encode still.
                    break

            def measure(rows: int) -> None:
                # A size that cannot compile raises when the search reads it.
                try:
                    self.encode_bytes(model, features, rows, exact=True)
                except ValueError:
                    pass

            concurrently([lambda rows=rows: measure(rows) for rows in sizes])

        def measured_fits(size: int, larger: bool = True) -> bool | None:
            nonlocal measures, ahead_taken
            found = fits(size, exact=False)
            if found is False and measures < MEMORY_ROWS_MEASURES:
                # A size no encode would bring in counts as measured without
                # compiling one, so the search reads the same sizes either way.
                if measurable(size):
                    if not ahead_taken:
                        ahead(size, larger)
                        ahead_taken = True
                    found = fits(size, exact=True)
                measures += 1
            return found

        size = self._size(MEASURED_ROWS)
        found = measured_fits(size)
        if found is None:
            return None
        if not found:
            while size > self._minimum:
                size = self._size(size // 2)
                found = measured_fits(size, larger=False)
                if found is None:
                    return None
                if found:
                    break
            return size
        while size < MEMORY_ROWS_LIMIT:
            larger = self._size(size + 1)
            found = measured_fits(larger)
            if found is None:
                return None
            if not found:
                return size
            size = larger
        return None

    def whole_rows(self: "Worker", model: int, features: int) -> int | None:
        """The most rows this worker can hold as one kept context on its device.

        A context of them fits the device pool, so it stays cached for every
        query that reads it, and its encode, or its decode of the smallest
        chunk of queries, fits beside the pool, with the autotuner's headroom,
        as `_fits_device` holds a unit to. An `adaptive_whole` table past its
        model's cap reads whole within it; so does one past `memory_rows`
        where that is fewer than 16,384, and past it every query of such a
        table shares a core of this many rows. None where the model's encode
        cannot be measured, so such a table reads as an adaptive one does.
        """
        features = self.width(model, features)
        profile = self.encode_profile(model, features)
        if profile is None:
            return None
        per_row = self.profile(model, features).cache_bytes_per_row
        layout = self.layout(model, features)
        decode = self.decode_profile(model)
        room = self.budget.device - self._pool_bytes

        def fits(rows: int) -> bool:
            size = self._size(rows)
            # The context's slots as its rows pack into blocks of them.
            if layout.slots(size) * self._slot_bytes > self._pool_bytes:
                return False
            encode = profile.working_bytes(per_row, size)
            working = decode.working_bytes(per_row, size)
            if self.budget.cpu:
                return max(encode, working) <= room
            # The paged decode's copy of the context's slots, beside its
            # working memory, as `_fits_device` holds a kept unit to.
            gather = _gather_bytes(layout, size)
            needed = max(encode, gather + working)
            return needed + max(encode, working) // AUTOTUNE_SHARE <= room

        if not fits(1):
            return None
        low, high = 1, WHOLE_ROWS_LIMIT
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle):
                low = middle
            else:
                high = middle - 1
        return low

    def _input_bytes(self: "Worker", unit: Unit) -> int:
        """Bytes of a unit's padded rows as its encode reads them: their representations."""
        table = self._tables[unit.table]
        profile = self.profile(table.model, table.features)
        return self._size(len(unit.rows)) * profile.row_bytes

    def _context_bytes(self: "Worker", unit: Unit) -> int:
        """Bytes of a unit's context as the model caches it, padding and own state included."""
        table = self._tables[unit.table]
        profile = self.profile(table.model, table.features)
        own = 0 if table.own is None else table.own.slots * self._slot_bytes
        return self._size(len(unit.rows)) * profile.cache_bytes_per_row + own
