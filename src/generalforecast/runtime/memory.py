# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from generalforecast.models.contract import TabularFunctions
from generalforecast.runtime.device import to_cache_dtype, with_summary_chunk

logger = logging.getLogger(__name__)

# Context sizes the cache is measured at. Tracing does no arithmetic, so any two work.
_SMALL, _LARGE = 8, 16
# Context sizes a decode or an encode is measured at: compiling allocates
# nothing, and at these what grows with the context's rows, such as attention
# scores, dominates.
_WORKING_ROWS = (2048, 4096)
# The fewest context rows a fit from those sizes holds for.
MEASURED_ROWS = _WORKING_ROWS[0]
# Table widths a model's calls are measured at, once per model. An encode's
# temporaries can grow faster than its cache with width, so a wider table's
# encode is measured at the table's own width.
_WORKING_WIDTHS = (4, 16)
# Headroom over the estimate, which XLA's choices at other sizes can exceed.
_WORKING_MARGIN = 1.05
# Rows a model's fit is measured at, once per model and table width. Below a
# few thousand rows, what XLA holds does not yet grow with them in a line.
_FIT_ROWS = (4096, 16384)
# Headroom over a fit's measure, which XLA's choices between the sizes it was
# measured at, and past them, can exceed.
_FIT_MARGIN = 1.1
# Rows a step of a measured fit's column summary reads, so what the fit holds
# per row besides its steps shows.
_FIT_STEP_ROWS = 8
# The fewest rows a step of a streamed column summary reads, whatever the call
# budget leaves it: fewer would loop over the rows for little memory saved.
MIN_SUMMARY_ROWS = 64
# The fewest rows a call budget limits a state's fit to: fewer barely describe
# a table, and a call past the budget at that size runs anyway, as any call
# too large for it alone does.
MIN_FIT_ROWS = 4096


def tree_bytes(tree: Any) -> int:
    """Bytes held by the arrays in `tree`, which may be abstract shapes.

    Args:
        tree: Pytree of arrays or `jax.ShapeDtypeStruct`s.

    Returns:
        The total size in bytes.
    """
    return sum(
        math.prod(leaf.shape) * jnp.dtype(leaf.dtype).itemsize
        for leaf in jax.tree.leaves(tree)
    )


@dataclass(frozen=True)
class MemoryProfile:
    """Sizes of a model's cached state for one table width.

    Attributes:
        state_bytes: Table state, which does not depend on the number of rows,
            for a table without categorical columns. A table that summarizes
            each context keeps one beside each unit's rows.
        row_bytes: One row representation.
        cache_bytes_per_row: Context cache per context row.
        cache_fixed_bytes: Context cache that does not grow with rows.
        categorical_state_bytes: Table state for a table with categorical
            columns, which some models keep more for.
    """

    state_bytes: int
    row_bytes: int
    cache_bytes_per_row: int
    cache_fixed_bytes: int
    categorical_state_bytes: int = 0

    def table_state_bytes(self, categorical: bool) -> int:
        """State of a table with or without categorical columns."""
        return self.categorical_state_bytes if categorical else self.state_bytes

    def context_bytes(self, rows: int) -> int:
        """Context cache size for `rows` context rows."""
        return self.cache_fixed_bytes + rows * self.cache_bytes_per_row


def _per_row(small: int, large: int, what: str) -> int:
    """Bytes per row from sizes at the two measured row counts."""
    growth, remainder = divmod(large - small, _LARGE - _SMALL)
    if remainder:
        raise ValueError(f"{what} does not grow by a whole number of bytes per row")
    return growth


def _traced(
    functions: TabularFunctions,
    model: Any,
    num_features: int,
    rows: int,
    cache_dtype: jnp.dtype | None,
    categorical: bool = True,
) -> tuple[Any, Any, Any]:
    """Abstract state, context rows and cache for a table of `rows` rows, by tracing.

    With `categorical`, the table has categorical columns, which some states
    keep more for, so the state is the most a table's takes.
    """
    properties = functions.properties(model)
    target_dtype = jnp.float32 if properties.regression else jnp.int32
    # Sizes do not depend on the class count, and a small one fits any model.
    classes = 0 if properties.regression else min(properties.class_slots, 2)

    # Traced with a row count, as the worker calls the model, so caches carry
    # their lengths as they do there.
    def fit(model: Any, features: Any, targets: Any, count: Any, mask: Any) -> Any:
        return functions.fit(
            model, features, targets, classes, num_rows=count, categorical=mask
        )

    def encode(model: Any, state: Any, rows: Any, targets: Any, count: Any) -> Any:
        cache = functions.encode(model, state, rows, targets, num_rows=count)
        return cache if cache_dtype is None else to_cache_dtype(cache, cache_dtype)

    features = jax.ShapeDtypeStruct((rows, num_features), jnp.float32)
    targets = jax.ShapeDtypeStruct((rows,), target_dtype)
    count = jax.ShapeDtypeStruct((), jnp.int32)
    mask = jax.ShapeDtypeStruct((num_features,), jnp.bool_) if categorical else None
    state, context_rows = jax.eval_shape(fit, model, features, targets, count, mask)
    cache = jax.eval_shape(encode, model, state, context_rows, targets, count)
    if cache_dtype is not None:
        cache = jax.eval_shape(partial(to_cache_dtype, dtype=cache_dtype), cache)
    return state, context_rows, cache


def memory_profile(
    functions: TabularFunctions,
    model: Any,
    num_features: int,
    cache_dtype: jnp.dtype | None = None,
) -> MemoryProfile:
    """Measures a model's state, row and cache sizes by tracing, without running it.

    Args:
        functions: The model's contract calls.
        model: Model parameters, real or abstract.
        num_features: Number of feature columns in the table.
        cache_dtype: Dtype the runtime stores caches in, or None for the dtype
            `encode` returns.

    Returns:
        The sizes for tables with `num_features` columns.
    """

    def sizes(rows: int) -> tuple[int, int, int, int]:
        state, context_rows, cache = _traced(
            functions, model, num_features, rows, cache_dtype, categorical=False
        )
        categorical, _, _ = _traced(functions, model, num_features, rows, cache_dtype)
        return (
            tree_bytes(state),
            tree_bytes(categorical),
            tree_bytes(context_rows),
            tree_bytes(cache),
        )

    small_state, small_categorical, small_rows, small_cache = sizes(_SMALL)
    large_state, large_categorical, large_rows, large_cache = sizes(_LARGE)
    if (small_state, small_categorical) != (large_state, large_categorical):
        raise ValueError("table state size depends on the number of rows")
    cache_per_row = _per_row(small_cache, large_cache, "the context cache")
    return MemoryProfile(
        state_bytes=small_state,
        row_bytes=_per_row(small_rows, large_rows, "a row representation"),
        cache_bytes_per_row=cache_per_row,
        cache_fixed_bytes=small_cache - _SMALL * cache_per_row,
        categorical_state_bytes=small_categorical,
    )


@dataclass(frozen=True)
class WorkingProfile:
    """Working memory a call against one context takes, from the cache's size.

    A decode works beside the context it reads, and an encode makes the
    context, so the working memory of either per context row follows the
    cache's bytes per row, which `MemoryProfile` gives for any
    width. These coefficients, measured once per model, turn one into the other.

    Attributes:
        queries: Queries in the decode's chunk, or 0 for an encode.
        row_scale: Working memory per context row, per byte of the cache's row.
        row_bytes: Working memory per context row, whatever the cache's row.
        fixed_scale: Working memory that does not grow with the context, per
            byte of the cache's row.
        fixed_bytes: Working memory that grows with neither.
    """

    queries: int
    row_scale: float = 1.0
    row_bytes: float = 0.0
    fixed_scale: float = 0.0
    fixed_bytes: float = 0.0

    def estimate(self, cache_bytes_per_row: int, rows: int) -> float:
        """The fit's bytes for `rows` context rows, with no floor or margin."""
        per_row = self.row_scale * cache_bytes_per_row + self.row_bytes
        fixed = self.fixed_scale * cache_bytes_per_row + self.fixed_bytes
        return per_row * rows + fixed

    def working_bytes(self, cache_bytes_per_row: int, rows: int) -> int:
        """Bytes the call works in against `rows` context rows.

        Args:
            cache_bytes_per_row: The cache's bytes per row, for the table's width.
            rows: Context rows, padded.

        Returns:
            The estimate, with a margin.
        """
        per_row = self.row_scale * cache_bytes_per_row + self.row_bytes
        fixed = self.fixed_scale * cache_bytes_per_row + self.fixed_bytes
        working = max(0.0, per_row) * rows + max(0.0, fixed)
        return math.ceil(_WORKING_MARGIN * working)


@dataclass(frozen=True)
class EncodeProfile:
    """Working memory an encode of one context takes, from the cache's size, by part.

    The call holds its inputs and outputs throughout and its temporaries
    while it runs; the runtime takes a lone context's cache from its outputs'
    own buffers. The fit holds from the
    smaller of the sizes it was measured at; below that, XLA lays the call out
    differently, so `encode_bytes` measures it at the context's own size.

    Attributes:
        held: The call's inputs and outputs.
        temporaries: Its temporaries.
        outputs: Its outputs alone.
    """

    held: WorkingProfile
    temporaries: WorkingProfile
    outputs: WorkingProfile

    def working_bytes(self, cache_bytes_per_row: int, rows: int) -> int:
        """Bytes the encode takes for `rows` context rows, at most at once.

        Args:
            cache_bytes_per_row: The cache's bytes per row, for the table's width.
            rows: Context rows, padded.

        Returns:
            The estimate, at least the cache, with a margin.
        """
        held = self.held.estimate(cache_bytes_per_row, rows)
        temporaries = self.temporaries.estimate(cache_bytes_per_row, rows)
        working = max(held + temporaries, cache_bytes_per_row * rows)
        return math.ceil(_WORKING_MARGIN * working)


def _working_profiles(
    functions: TabularFunctions,
    model: Any,
    working: Callable[[int, int], tuple[int, ...]],
    queries: int,
    cache_dtype: jnp.dtype | None,
    widths: tuple[int, int] = _WORKING_WIDTHS,
) -> tuple[WorkingProfile, ...]:
    """Fits parts of a call's working memory, measured at two context sizes and two widths.

    Args:
        functions: The model's contract calls.
        model: Model parameters.
        working: The bytes of each part the call takes at a table width and
            a context size.
        queries: Queries in the call, or 0 for an encode.
        cache_dtype: Dtype the runtime stores caches in, or None.
        widths: The narrow and the wide table width measured.

    Returns:
        Each part's coefficients.

    Raises:
        Exception: Whatever compiling the call raised, where XLA cannot say.
    """
    small_rows, large_rows = _WORKING_ROWS
    measured = []
    for width in widths:
        cache = memory_profile(functions, model, width, cache_dtype)
        small, large = working(width, small_rows), working(width, large_rows)
        per_row = [
            (b - a) / (large_rows - small_rows)
            for a, b in zip(small, large, strict=True)
        ]
        fixed = [b - large_rows * row for b, row in zip(large, per_row, strict=True)]
        measured.append((cache.cache_bytes_per_row, per_row, fixed))
    (narrow, narrow_rows, narrow_fixed), (wide, wide_rows, wide_fixed) = measured
    profiles = []
    for narrow_row, wide_row, narrow_fix, wide_fix in zip(
        narrow_rows, wide_rows, narrow_fixed, wide_fixed, strict=True
    ):
        if narrow == wide:
            # The cache's rows do not grow with width, so neither do the call's.
            profiles.append(
                WorkingProfile(
                    queries,
                    row_scale=max(narrow_row, wide_row) / max(1, narrow),
                    fixed_bytes=max(narrow_fix, wide_fix),
                )
            )
            continue
        row_scale = (wide_row - narrow_row) / (wide - narrow)
        fixed_scale = (wide_fix - narrow_fix) / (wide - narrow)
        profiles.append(
            WorkingProfile(
                queries,
                row_scale=row_scale,
                row_bytes=narrow_row - row_scale * narrow,
                fixed_scale=fixed_scale,
                fixed_bytes=narrow_fix - fixed_scale * narrow,
            )
        )
    return tuple(profiles)


class NoMemoryAnalysis(RuntimeError):
    """XLA compiled a call but gives no memory analysis on this backend."""


def _analysis(function: Callable[..., Any], *arguments: Any) -> Any:
    """XLA's memory analysis of `function` compiled for these arguments."""
    analysis = jax.jit(function).lower(*arguments).compile().memory_analysis()
    if analysis is None:
        raise NoMemoryAnalysis("XLA gives no memory analysis on this backend")
    return analysis


def decode_profile(
    functions: TabularFunctions,
    model: Any,
    decode: Callable[[Any, Any, Any, Any], Any],
    queries: int,
    cache_dtype: jnp.dtype | None = None,
) -> WorkingProfile:
    """Measures a model's decode of one query chunk by compiling it, without running it.

    XLA's analysis of the decode compiled at two context sizes gives what it
    takes per context row and besides; at two table widths, how both follow
    the cache's bytes per row. The contexts are large enough that what grows
    with their rows outweighs the rest.

    Args:
        functions: The model's contract calls.
        model: Model parameters, as the runtime configured them.
        decode: Decodes as the runtime does: parameters, state, cache and
            query rows to outputs.
        queries: Queries in the chunk measured: the fewest a decode takes.
        cache_dtype: Dtype the runtime stores caches in, or None for the dtype
            `encode` returns.

    Returns:
        The coefficients; where XLA cannot say, a copy of the context per row.
    """

    def working(width: int, rows: int) -> tuple[int]:
        state, _, cache = _traced(functions, model, width, rows, cache_dtype)
        features = jax.ShapeDtypeStruct((queries, width), jnp.float32)
        embedded = jax.eval_shape(functions.embed, model, state, features)
        analysis = _analysis(decode, model, state, cache, embedded)
        return (analysis.temp_size_in_bytes + analysis.output_size_in_bytes,)

    try:
        (profile,) = _working_profiles(functions, model, working, queries, cache_dtype)
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "could not measure the model's decode (%s); its working memory "
            "is taken as a copy of the context",
            error,
        )
        return WorkingProfile(queries)
    return profile


def _encode_parts(
    functions: TabularFunctions,
    model: Any,
    fit: bool,
    width: int,
    rows: int,
    cache_dtype: jnp.dtype | None = None,
) -> tuple[int, int, int]:
    """An encode's inputs and outputs, temporaries, and outputs alone, as XLA measures them.

    With `cache_dtype`, the encode returns its cache in that dtype, as the
    runtime's does.
    """
    properties = functions.properties(model)
    target_dtype = jnp.float32 if properties.regression else jnp.int32
    classes = 0 if properties.regression else min(properties.class_slots, 2)

    def encode(model: Any, state: Any, rows: Any, targets: Any, count: Any) -> Any:
        cache = functions.encode(model, state, rows, targets, num_rows=count)
        return cache if cache_dtype is None else to_cache_dtype(cache, cache_dtype)

    def fit_encode(
        model: Any, features: Any, targets: Any, count: Any, mask: Any
    ) -> Any:
        state, rows = functions.fit(
            model, features, targets, classes, num_rows=count, categorical=mask
        )
        return state, encode(model, state, rows, targets, count)

    targets = jax.ShapeDtypeStruct((rows,), target_dtype)
    count = jax.ShapeDtypeStruct((), jnp.int32)
    if fit:
        features = jax.ShapeDtypeStruct((rows, width), jnp.float32)
        mask = jax.ShapeDtypeStruct((width,), jnp.bool_)
        arguments = (features, targets, count, mask)
        analysis = _analysis(fit_encode, model, *arguments)
    else:
        state, context_rows, _ = _traced(functions, model, width, rows, None)
        arguments = (state, context_rows, targets, count)
        analysis = _analysis(encode, model, *arguments)
    outputs = analysis.output_size_in_bytes
    return tree_bytes(arguments) + outputs, analysis.temp_size_in_bytes, outputs


def encode_widths(width: int) -> tuple[int, int]:
    """Widths to measure an encode at for a table of `width` columns, around it."""
    narrow, wide = _WORKING_WIDTHS
    return min(narrow, width), max(wide, width)


def encode_profile(
    functions: TabularFunctions,
    model: Any,
    fit: bool = False,
    cache_dtype: jnp.dtype | None = None,
    widths: tuple[int, int] = _WORKING_WIDTHS,
) -> EncodeProfile:
    """Measures a model's encode of a context by compiling it, without running it.

    As `decode_profile` measures a decode, part by part. The estimate holds
    for contexts of at least `MEASURED_ROWS` rows and for widths between the
    two measured. An encode's temporaries can grow faster than its cache with
    width, and a line through two widths falls short past them, so
    `encode_widths` gives widths around a table's.

    Args:
        functions: The model's contract calls.
        model: Model parameters, as the runtime configured them.
        fit: Whether the call fits the context's own state from its features
            first, as for a table that summarizes each context.
        cache_dtype: Dtype the runtime stores caches in, or None for the dtype
            `encode` returns.
        widths: The narrow and the wide table width measured.

    Returns:
        The parts' coefficients.

    Raises:
        NoMemoryAnalysis: Where XLA cannot say.
        Exception: Whatever compiling the encode raised.
    """
    profiles = _working_profiles(
        functions,
        model,
        lambda width, rows: _encode_parts(
            functions, model, fit, width, rows, cache_dtype
        ),
        0,
        cache_dtype,
        widths,
    )
    return EncodeProfile(*profiles)


def encode_bytes(
    functions: TabularFunctions,
    model: Any,
    width: int,
    rows: int,
    fit: bool = False,
    cache_dtype: jnp.dtype | None = None,
) -> int:
    """Measures the encode of one context of this width and size, by compiling it.

    For contexts smaller than `encode_profile`'s fit holds for.

    Args:
        functions: The model's contract calls.
        model: Model parameters, as the runtime configured them.
        width: The table's feature columns.
        rows: Context rows, padded.
        fit: Whether the call fits the context's own state first.
        cache_dtype: Dtype the runtime stores caches in, or None for the dtype
            `encode` returns.

    Returns:
        Bytes the encode takes at most at once, with a margin.

    Raises:
        NoMemoryAnalysis: Where XLA cannot say.
        Exception: Whatever compiling the encode raised.
    """
    held, temporaries, _ = _encode_parts(
        functions, model, fit, width, rows, cache_dtype
    )
    return math.ceil(_WORKING_MARGIN * (held + temporaries))


@dataclass(frozen=True)
class Cost:
    """Bytes a call takes for a table of `rows` rows.

    Attributes:
        fixed: Bytes that do not grow with the rows.
        row: Bytes per row.
    """

    fixed: float = 0.0
    row: float = 0.0

    def bytes(self, rows: int) -> float:
        """Bytes for a table of `rows` rows."""
        return self.fixed + rows * self.row

    def rows(self, budget: int) -> int:
        """The most rows `budget` bytes hold, with `_FIT_MARGIN` to spare; any if a row takes none."""
        if self.row <= 0:
            return sys.maxsize
        return max(0, int((budget / _FIT_MARGIN - self.fixed) // self.row))


@dataclass(frozen=True)
class FitProfile:
    """What a model's fit takes at one table width, as XLA measures it.

    Zero costs, as where XLA cannot say, put no limit on a fit's rows.

    Attributes:
        width: The table's columns.
        fused: A fit that embeds every row in the call that fits its state,
            reading its column summary at once.
        state: A fit of the state alone whose column summary reads a few rows
            a step: what it holds besides the steps.
    """

    width: int
    fused: Cost = Cost()
    state: Cost = Cost()

    def fused_rows(self, budget: int) -> int:
        """The most rows a fit embeds in the call that fits its state, in `budget` bytes."""
        return self.fused.rows(budget)

    def summary_chunk(self, budget: int) -> int:
        """Cells a column summary reads a step: half of `budget`, at a fused fit's bytes per row.

        At least `MIN_SUMMARY_ROWS` rows' cells, where what a fit holds besides
        its rows takes that half.
        """
        rows = self.fused.rows(budget // 2)
        return max(MIN_SUMMARY_ROWS, min(rows, sys.maxsize // self.width)) * self.width

    def state_rows(self, budget: int) -> int:
        """The most rows a state alone is fit on: the half of `budget` its summary's steps leave.

        At least `MIN_FIT_ROWS`, however small the budget.
        """
        return max(MIN_FIT_ROWS, self.state.rows(budget // 2))


def fit_profile(
    functions: TabularFunctions, model: Any, width: int, cache: Path | None = None
) -> FitProfile:
    """Measures what a model's fit takes by compiling it, without running it.

    XLA's analysis of the fit compiled at two row counts gives what it takes
    besides its rows, and per row: once as one call that also embeds every
    row, and once for the state alone with its column summary read a few rows
    a step. Fits are measured at the table's own width, since what a row
    takes need not grow with the columns in a line, as with grouped columns.

    Args:
        functions: The model's contract calls.
        model: Model parameters, as the runtime configured them.
        width: The table's columns.
        cache: A directory that keeps measures for later processes, by the
            programs measured and the device, or None to keep none.

    Returns:
        The costs, or zeros where XLA cannot say.
    """
    properties = functions.properties(model)
    target_dtype = jnp.float32 if properties.regression else jnp.int32
    classes = 0 if properties.regression else min(properties.class_slots, 2)
    few, many = _FIT_ROWS

    def lowered(rows: int, state: bool) -> jax.stages.Lowered:
        chunk = _FIT_STEP_ROWS * width if state else None

        def fit(model: Any, features: Any, targets: Any, mask: Any) -> Any:
            configured = with_summary_chunk(model, chunk)
            fitted = functions.fit(
                configured, features, targets, classes, categorical=mask
            )
            return fitted[0] if state else fitted

        # With a mask of categorical columns, which some models read apart,
        # so the fit is the most a table's takes.
        inputs = (
            jax.ShapeDtypeStruct((rows, width), jnp.float32),
            jax.ShapeDtypeStruct((rows,), target_dtype),
            jax.ShapeDtypeStruct((width,), jnp.bool_),
        )
        return jax.jit(fit).lower(model, *inputs)

    def measured(program: jax.stages.Lowered, rows: int) -> float:
        analysis = program.compile().memory_analysis()
        if analysis is None:
            raise RuntimeError("XLA gives no memory analysis on this backend")
        total = analysis.temp_size_in_bytes + analysis.output_size_in_bytes
        return total + rows * width * 4 + rows * jnp.dtype(target_dtype).itemsize

    def cost(small: float, large: float) -> Cost:
        row = max(0.0, (large - small) / (many - few))
        return Cost(max(0.0, large - row * many), row)

    try:
        programs = [
            lowered(rows, state) for state in (False, True) for rows in (few, many)
        ]
        key = hashlib.sha256(jax.devices()[0].device_kind.encode())
        for program in programs:
            key.update(program.as_text().encode())
        path = None if cache is None else cache / f"{key.hexdigest()}.json"
        if path is not None and path.is_file():
            fused, state = json.loads(path.read_text())
            return FitProfile(width, Cost(*fused), Cost(*state))
        bytes = [
            measured(program, rows)
            for program, rows in zip(programs, (few, many) * 2, strict=True)
        ]
        profile = FitProfile(width, cost(*bytes[:2]), cost(*bytes[2:]))
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "could not measure the model's fit (%s); its rows are not limited", error
        )
        return FitProfile(width)
    if path is not None:
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            written = path.with_suffix(f".{os.getpid()}.tmp")
            written.write_text(
                json.dumps(
                    [
                        dataclasses.astuple(profile.fused),
                        dataclasses.astuple(profile.state),
                    ]
                )
            )
            written.replace(path)
    return profile
