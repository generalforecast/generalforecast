# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The contract for forecasters: models that predict a series' next steps.

A forecaster differs from a tabular model in what it is asked: there is no
table state fit on every row, no row representation worth caching apart from
the context, and a query is a position in time, not a row. So forecasters
have a contract of their own, `SeriesFunctions`, beside the tabular one in
`generalforecast.models.contract`. What the two share sits below the contracts: the
attention caches, the slot layout, the pools, compiled programs and the frame
protocol.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import jax
import jax.core
import jax.numpy as jnp
from jax import Array
from jaxtyping import Int

from generalforecast.nn.attention import AttentionCache

# The roles of a group's rows: the series forecast, a covariate known only
# in the past, and a covariate known over the horizon too.
TARGET, PAST, KNOWN = 0, 1, 2


class SeriesGroup(NamedTuple):
    """Aligned rows of one attention group over one time axis.

    Rows come real first, `num_rows` of them, the rest padding. Time is
    left-padded to a multiple of the model's patch; the last step is the
    forecast origin. Unobserved steps are False in `observed`, whatever
    `values` holds there.

    Attributes:
        values: Row values, `(rows, time)`, float32 on the original scale.
        observed: Which values are observed, `(rows, time)`.
        roles: Per row, `TARGET`, `PAST` or `KNOWN`, int32.
        future: Known-future covariate values over the horizon, `(rows,
            horizon)`, zeros on rows that are not known covariates, or None
            when no row is one.
        future_observed: Which of `future` are observed, or None.
        num_rows: Real rows, or None for all; may be traced.
    """

    values: Array
    observed: Array
    roles: Array
    future: Array | None = None
    future_observed: Array | None = None
    num_rows: Array | int | None = None


@dataclass(frozen=True)
class SeriesProperties:
    """What the engine needs to know about a forecaster before running it.

    Every field is read from parameters and their static settings, so it holds
    for abstract parameters too. The values are hashable, which lets them be
    static under jit.

    Attributes:
        patch: Steps per input patch; contexts pad to a multiple.
        output_patch: Steps per output patch; horizons pad to a multiple.
        context_limit: Most context steps the model reads.
        pass_limit: Most horizon steps one pass of the network predicts, or
            None when one pass covers any horizon. `forecast` and `decode`
            loop to produce a longer horizon when the model imposes a limit.
        quantiles: The levels of the last output axis, increasing.
        reusable: Whether `encode` and `decode` exist: the context encoding
            does not read the horizon when no row is a known covariate, so a
            group with one is forecast whole.
        max_rows: Most rows in a group the model reads, or None for any number.
    """

    patch: int
    output_patch: int
    context_limit: int
    pass_limit: int | None
    quantiles: tuple[float, ...]
    reusable: bool
    max_rows: int | None = None


class Forecast(Protocol):
    """Computes quantiles for the steps after a group."""

    def __call__(self, model: Any, group: SeriesGroup, horizon: int, /) -> Array:
        """Forecasts `horizon` steps for every row, `(rows, horizon, quantiles)`."""
        ...


class Encode(Protocol):
    """Computes the context cache a horizon is decoded from."""

    def __call__(self, model: Any, group: SeriesGroup, /) -> Any:
        """Encodes a group's context; refuses a `KNOWN` row if it reads the horizon."""
        ...


class Decode(Protocol):
    """Computes quantiles from a context cache."""

    def __call__(
        self,
        model: Any,
        cache: Any,
        horizon: int,
        /,
        *,
        future: Array | None = None,
        future_observed: Array | None = None,
    ) -> Array:
        """Decodes `horizon` steps from a cache, with the horizon's known values."""
        ...


class Extend(Protocol):
    """Grows a context cache by whole patches of new steps."""

    def __call__(self, model: Any, cache: Any, group: SeriesGroup, /) -> Any:
        """Extends a cache by `group`, the new steps of the same rows in order."""
        ...


class Properties(Protocol):
    """Describes a forecaster from its parameters, once at load."""

    def __call__(self, model: Any, /) -> SeriesProperties:
        """The model's properties."""
        ...


class SeriesFunctions(NamedTuple):
    """The calls the engine makes on a forecaster, each with its own cache boundary.

    Every call works on one group with no batch axis; the engine batches with
    `jax.vmap`. A group is one attention group, the rows that may see each
    other: a multivariate series, or related series retrieval put together.
    `horizon` is a Python int, static under jit; the engine buckets horizons
    to multiples of the output patch and slices.

    Outputs are `(rows, horizon, quantiles)` float32 on the original scale,
    one row of outputs per group row, at the levels the properties name. They
    are the model's estimates as it emits them; the engine does not sort them.

    A context cache is a tree of `AttentionCache`s with the context patches
    on their first axis, plus leaves per row the decode needs, such as scales.
    The engine pages caches by patches as it pages tabular caches by rows.
    Test-time extras such as clipping and averaging over signs are the
    model's business, inside `forecast` and `decode`.

    Attributes:
        forecast: Quantiles for `horizon` steps after a group, in one pass or
            the model's own continuation past `pass_limit`. Every model has
            it; it is the parity path, and the path for groups the model
            cannot cache.
        encode: The context cache of a group, for a reusable model, or None.
        decode: Quantiles for `horizon` steps from a cache, with the horizon's
            known covariates if the group has any, or None with `encode`.
        extend: The cache grown by whole patches of new steps, exact, or None
            for a model whose encoding cannot grow. A cache the runtime reads
            back from its slots may carry `length` on its attention caches:
            the real patches, which come first; the rest are zeros to skip,
            and `extend` places the new patches right after the real ones.
            `decode` reads such a cache too.
        properties: Describes the model from its parameters, once at load.
        decode_within: `decode` for a cache whose real steps plus the horizon
            stay within `context_limit`, which the engine calls for such
            caches so the model may skip, statically, the work of passing
            the limit; or None when `decode` has none to skip. Its outputs
            must equal `decode`'s bit for bit.
    """

    forecast: Forecast
    encode: Encode | None
    decode: Decode | None
    extend: Extend | None
    properties: Properties
    decode_within: Decode | None = None


class GroupInputs(NamedTuple):
    """A group as a forecaster reads it: trimmed to its context and masked.

    Attributes:
        values: The context's last steps, 0 where unobserved, float32.
        observed: Which of them are observed, False on padding rows.
        real: Which rows are real rather than padding.
        future: Known future values over the horizon, 0 where not known,
            float32.
        future_observed: Which future values are known: observed where the
            group says, or finite if it does not, and never on padding rows.
    """

    values: Array
    observed: Array
    real: Array
    future: Array
    future_observed: Array


def group_inputs(
    group: SeriesGroup, patch: int, limit: int, horizon: int, known_only: bool = True
) -> GroupInputs:
    """A group's context, at most `limit` steps of it, and its known future.

    Args:
        group: The group.
        patch: Steps per patch, which the context must be whole patches of.
        limit: Most context steps read, the latest.
        horizon: Steps of future values when the group has none.
        known_only: Whether only known covariates' future values are known.

    Returns:
        The inputs.

    Raises:
        ValueError: If the context is not whole patches.
    """
    rows, steps = group.values.shape
    if steps % patch:
        raise ValueError(
            f"the context's {steps} steps are not whole patches of {patch}"
        )
    real = (
        jnp.ones(rows, bool)
        if group.num_rows is None
        else jnp.arange(rows) < group.num_rows
    )
    observed = (group.observed & real[:, None])[:, -limit:]
    values = jnp.where(observed, group.values[:, -limit:], 0.0).astype(jnp.float32)
    if group.future is None:
        future = jnp.zeros((rows, horizon), jnp.float32)
        future_observed = jnp.zeros((rows, horizon), bool)
    else:
        future_observed = (
            jnp.isfinite(group.future)
            if group.future_observed is None
            else group.future_observed
        )
        future_observed = future_observed & real[:, None]
        if known_only:
            future_observed = future_observed & (group.roles == KNOWN)[:, None]
        future = jnp.where(future_observed, group.future, 0.0).astype(jnp.float32)
    return GroupInputs(values, observed, real, future, future_observed)


def refuse_known(group: SeriesGroup) -> None:
    """Raises for a known covariate when roles can be seen: a group with one is forecast whole.

    Under a trace they cannot, so an encoding must poison itself instead.

    Raises:
        ValueError: If a row is a known covariate.
    """
    if not isinstance(group.roles, jax.core.Tracer) and bool(
        (group.roles == KNOWN).any()
    ):
        raise ValueError(
            "a group with a known covariate is forecast whole, not encoded"
        )


def observed_moments(
    values: Array, observed: Array, keepdims: bool = False
) -> tuple[Array, Array, Array]:
    """The count, mean and population deviation of observed values along the last axis.

    With none observed, the mean and deviation are 0.

    Args:
        values: Values.
        observed: Which are observed.
        keepdims: Whether the last axis stays, of size 1.

    Returns:
        The count, an integer, the mean and the deviation.
    """
    count = observed.sum(axis=-1, keepdims=keepdims)
    divisor = jnp.maximum(count, 1)
    mean = jnp.where(observed, values, 0.0).sum(axis=-1, keepdims=keepdims) / divisor
    centered = values - (mean if keepdims else mean[..., None])
    squares = jnp.where(observed, jnp.square(centered), 0.0)
    return count, mean, jnp.sqrt(squares.sum(axis=-1, keepdims=keepdims) / divisor)


def median_index(levels: Sequence[float]) -> int:
    """Where the median, level 0.5, sits among a forecaster's quantile levels.

    Forecasters that feed their medians back or fan their quantiles out from
    it find it by its level, since a grid need not put it in the middle.

    Args:
        levels: The quantile levels.

    Returns:
        The median's index.

    Raises:
        ValueError: If no level is 0.5.
    """
    if 0.5 not in levels:
        raise ValueError(f"the quantile levels {tuple(levels)} have no median, 0.5")
    return list(levels).index(0.5)


def real_patches(layers: Sequence[AttentionCache]) -> Int[Array, ""] | None:
    """How many of a context cache's patches are real, from its attention caches' lengths.

    A cache the runtime reads back from its slots holds its real patches
    first and zeros after, and says how many are real; one from `encode` or
    `extend` holds only real patches and says nothing.

    Args:
        layers: The cache's attention caches along time, patches first.

    Returns:
        The count, or None when every patch is real.
    """
    return layers[0].length if layers else None


def rows_first(layers: Sequence[AttentionCache]) -> tuple[AttentionCache, ...]:
    """Attention caches stored patches first, with rows first as attention along time reads them.

    A scalar length stays as it is.
    """
    return tuple(
        jax.tree.map(lambda leaf: leaf.swapaxes(0, 1) if leaf.ndim > 1 else leaf, layer)
        for layer in layers
    )


def padded_horizon(values: Array, steps: int, fill: float | bool) -> Array:
    """The first `steps` of a horizon's values, padded with `fill` if it has fewer.

    Args:
        values: Values over a horizon on their last axis.
        steps: Steps wanted.
        fill: What the steps past the horizon take.

    Returns:
        `(..., steps)`.
    """
    if values.shape[-1] >= steps:
        return values[..., :steps]
    pad = [(0, 0)] * (values.ndim - 1) + [(0, steps - values.shape[-1])]
    return jnp.pad(values, pad, constant_values=fill)
