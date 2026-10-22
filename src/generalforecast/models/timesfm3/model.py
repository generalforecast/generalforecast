# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM 3: a causal patch transformer that decodes a horizon in one pass.

Each row is normalized by the running moments of its observed values up to
each patch, and each patch's token carries its values, the next output
patch's values when they are known, and their masks. Layers attend causally
along a row's patches, then across the group's rows at each patch, then apply
a feed-forward, each with a norm after it. Every token predicts the next
output patch, and the horizon's tokens, which carry nothing, predict it from
the context; their frames are refined with the model's own predictions, the
overlapping predictions are stitched, and a linear trend the context fit, if
it explained enough of the variance, is added back.

The context's tokens never read the horizon unless a row is a known covariate,
whose tokens carry its future values and whose moments include them, so a
group without one encodes once and decodes any horizon. The detrending and the
moments read the whole context, so an encoding cannot grow.
"""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.core
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.series import (
    KNOWN,
    TARGET,
    SeriesFunctions,
    SeriesGroup,
    SeriesProperties,
    padded_horizon,
)
from generalforecast.nn.attention import (
    AttentionCache,
    AttentionParameters,
    attend,
    project_kv,
)
from generalforecast.nn.linear import (
    LinearParameters,
    ResidualBlockParameters,
    linear,
    residual_block,
)
from generalforecast.nn.norm import RMSNormParameters, rms_norm

# Most context steps the forecaster reads.
CONTEXT_LIMIT = 15360
# A deviation below this divides as 1.
DEVIATION_FLOOR = 1e-6


class TimesFMLayer(NamedTuple):
    """Attention along patches, attention across rows and a feed-forward.

    Each has a norm before it and a norm on its output, then the residual.
    """

    sequence_norm: RMSNormParameters
    sequence: AttentionParameters
    sequence_output_norm: RMSNormParameters
    variate_norm: RMSNormParameters
    variate: AttentionParameters
    variate_output_norm: RMSNormParameters
    feed_forward_norm: RMSNormParameters
    feed_forward_input: LinearParameters
    feed_forward_output: LinearParameters
    feed_forward_output_norm: RMSNormParameters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TimesFM:
    """TimesFM 3's embedding, layers and head, with its settings.

    Attributes:
        embedding: Embeds a patch's values, known next values and masks.
        layers: The transformer layers.
        head: Maps a token to its output patch's quantiles.
        patch: Steps per input patch.
        output_patch: Steps each token predicts, a multiple of `patch`.
        quantiles: The levels of the output quantiles.
        value_clip: Values and predictions are clipped to this magnitude.
        detrending: Below this ratio of the detrended to the original
            deviation, a row's linear trend is removed and added back; None
            never detrends.
        max_rows: Most rows in a group, as the model was trained.
    """

    embedding: ResidualBlockParameters
    layers: tuple[TimesFMLayer, ...]
    head: LinearParameters
    patch: int = field(metadata={"static": True})
    output_patch: int = field(metadata={"static": True})
    quantiles: tuple[float, ...] = field(metadata={"static": True})
    value_clip: float = field(metadata={"static": True})
    detrending: float | None = field(metadata={"static": True})
    max_rows: int = field(metadata={"static": True})


class Moments(NamedTuple):
    """Running moments of observed values: their count, mean and deviation."""

    count: Array
    mean: Array
    deviation: Array


class Trend(NamedTuple):
    """Each row's fitted linear trend over normalized time, and whether it applies."""

    slope: Array
    intercept: Array
    applies: Array


class Rows(NamedTuple):
    """A group's rows as the network reads them.

    Attributes:
        values: Context values, 0 where masked, `(rows, time)` float32.
        masked: Which values the network does not see; a row without any
            observed value, and a padding row, is read as observed zeros.
        roles: Each row's role.
        real: Which rows are real rather than padding.
        future: Known future values over the horizon, 0 elsewhere.
        future_masked: Which future values are not known.
        first: The first patch a real target row observes a value in, before
            rows without one are read as observed zeros, so padding on the
            left never moves where positions count from.
    """

    values: Array
    masked: Array
    roles: Array
    real: Array
    future: Array
    future_masked: Array
    first: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TimesFMCache:
    """A group's encoded context, its patches first.

    Attributes:
        sequence: Keys and values per layer's attention along patches,
            `(patches, rows, heads, head_dim)`.
        visible: Which patches are keys for each row, `(patches, rows)`: all
            but the fully masked ones a row starts with.
        first: The first patch any real row observes, where positions start.
        moments: The running moments per row at the last context patch.
        trend: Each row's trend.
        last: The last context patch's raw outputs per row, which the horizon's
            frames and the stitching start from; NaN when a row is a known
            covariate, whose cache would forecast wrongly.
        roles: Each row's role.
        real: Which rows are real.
    """

    sequence: tuple[AttentionCache, ...]
    visible: Bool[Array, " patches rows"]
    first: Int[Array, ""]
    moments: Moments
    trend: Trend
    last: Float[Array, " rows outputs"]
    roles: Int[Array, " rows"]
    real: Bool[Array, " rows"]


def _along_rows(leaf: Array) -> Array:
    """A stored attention cache leaf with rows first, as attention along time reads it; a cache's scalar length stays as it is."""
    return leaf.swapaxes(0, 1) if leaf.ndim > 1 else leaf


def _dtype(model: TimesFM) -> jnp.dtype:
    """The dtype the network computes in, set by its parameters."""
    return model.head.weight.dtype


def _safe(deviation: Array) -> Array:
    """A deviation to divide by: 1 where it is near zero."""
    return jnp.where(deviation < DEVIATION_FLOOR, 1.0, deviation)


def _update(
    moments: Moments,
    values: Float[Array, " rows patch"],
    masked: Bool[Array, " rows patch"],
) -> Moments:
    """The moments after one patch of values, as upstream accumulates them."""
    seen = (~masked).astype(jnp.float32)
    count = seen.sum(axis=-1)
    present = jnp.where(masked, 0.0, values)
    mean = present.sum(axis=-1) / jnp.maximum(count, 1.0)
    variance = jnp.where(masked, 0.0, jnp.square(values - mean[:, None])).sum(axis=-1)
    deviation = jnp.sqrt(variance / jnp.maximum(count, 1.0))
    total = moments.count + count
    safe = jnp.maximum(total, 1.0)
    new_mean = jnp.where(
        total == 0, 0.0, (moments.count * moments.mean + mean * count) / safe
    )
    spread = (
        moments.count * jnp.square(moments.deviation)
        + count * jnp.square(deviation)
        + moments.count * jnp.square(moments.mean - new_mean)
        + count * jnp.square(mean - new_mean)
    )
    new_deviation = jnp.sqrt(jnp.where(total == 0, 0.0, spread / safe))
    return Moments(total, new_mean, new_deviation)


def _running(
    model: TimesFM,
    values: Float[Array, " rows time"],
    masked: Bool[Array, " rows time"],
    before: Moments | None = None,
) -> Moments:
    """The moments of each row's observed values up to each patch, `(rows, patches)`."""
    rows = values.shape[0]
    patches = values.reshape(rows, -1, model.patch).swapaxes(0, 1)
    masks = masked.reshape(rows, -1, model.patch).swapaxes(0, 1)
    if before is None:
        before = Moments(*(jnp.zeros(rows, jnp.float32) for _ in range(3)))

    def step(moments: Moments, patch: tuple[Array, Array]) -> tuple[Moments, Moments]:
        moments = _update(moments, *patch)
        return moments, moments

    _, running = jax.lax.scan(step, before, (patches, masks))
    return Moments(*(leaf.swapaxes(0, 1) for leaf in running))


def _trend(
    model: TimesFM,
    values: Float[Array, " rows time"],
    masked: Bool[Array, " rows time"],
) -> Trend:
    """Each row's least-squares line over its observed context, and whether it applies.

    Time runs over the context length, and the line applies when it leaves
    little enough of the variance.
    """
    rows, steps = values.shape
    if model.detrending is None:
        zeros = jnp.zeros(rows, jnp.float32)
        return Trend(zeros, zeros, jnp.zeros(rows, bool))
    time = jnp.arange(-(steps - 1), 1, dtype=jnp.float32)[None] / steps
    seen = ~masked
    count = seen.sum(axis=-1, keepdims=True).astype(jnp.float32)
    sum_t = jnp.where(seen, time, 0.0).sum(axis=-1, keepdims=True)
    sum_tt = jnp.where(seen, jnp.square(time), 0.0).sum(axis=-1, keepdims=True)
    sum_y = jnp.where(seen, values, 0.0).sum(axis=-1, keepdims=True)
    sum_ty = jnp.where(seen, time * values, 0.0).sum(axis=-1, keepdims=True)
    determinant = count * sum_tt - jnp.square(sum_t)
    singular = determinant == 0
    slope = jnp.where(
        singular,
        0.0,
        (count * sum_ty - sum_t * sum_y) / jnp.where(singular, 1.0, determinant),
    )
    at_least_one = jnp.maximum(count, 1.0)
    intercept = jnp.where(
        singular,
        jnp.where(count > 0, sum_y / at_least_one, 0.0),
        (sum_y - slope * sum_t) / at_least_one,
    )
    detrended = values - (slope * time + intercept)
    mean = sum_y / at_least_one
    sum_yy = jnp.where(seen, jnp.square(values), 0.0).sum(axis=-1, keepdims=True)
    original = jnp.sqrt(jnp.maximum(sum_yy / at_least_one - jnp.square(mean), 0.0))
    mean_detrended = (
        jnp.where(seen, detrended, 0.0).sum(axis=-1, keepdims=True) / at_least_one
    )
    sum_dd = jnp.where(seen, jnp.square(detrended), 0.0).sum(axis=-1, keepdims=True)
    remaining = jnp.sqrt(
        jnp.maximum(sum_dd / at_least_one - jnp.square(mean_detrended), 0.0)
    )
    return Trend(
        slope[:, 0], intercept[:, 0], (remaining < model.detrending * original)[:, 0]
    )


def _detrended(
    trend: Trend, values: Float[Array, " rows time"], time: Float[Array, " time"]
) -> Float[Array, " rows time"]:
    """Values less their row's trend at `time`, where the trend applies."""
    line = trend.slope[:, None] * time[None] + trend.intercept[:, None]
    return jnp.where(trend.applies[:, None], values - line, values)


def _embed(
    model: TimesFM,
    values: Float[Array, " rows time"],
    masked: Bool[Array, " rows time"],
    target: Bool[Array, " rows"],
    moments: Moments,
) -> tuple[Float[Array, " rows patches model"], Bool[Array, " rows patches"]]:
    """Embeds each patch's values, a known covariate's next values, and their masks.

    Also says which patches are fully masked.
    """
    rows, steps = values.shape
    patches = steps // model.patch
    rolls = model.output_patch // model.patch
    values = values.reshape(rows, patches, model.patch)
    masked = masked.reshape(rows, patches, model.patch)
    mean, deviation = moments.mean[..., None], _safe(moments.deviation)[..., None]
    normalized = jnp.where(masked, 0.0, (values - mean) / deviation)
    pad = [(0, 0), (0, rolls), (0, 0)]
    later = jnp.pad(values, pad)
    later_masked = jnp.pad(masked, pad, constant_values=True)
    following = jnp.concatenate(
        [later[:, roll : roll + patches] for roll in range(1, rolls + 1)], axis=-1
    )
    following_masked = jnp.concatenate(
        [later_masked[:, roll : roll + patches] for roll in range(1, rolls + 1)],
        axis=-1,
    )
    following_masked = following_masked | target[:, None, None]
    following = jnp.where(following_masked, 0.0, (following - mean) / deviation)
    masks = jnp.concatenate([masked, following_masked], axis=-1)
    features = jnp.concatenate(
        [normalized, following, masks.astype(jnp.float32)], axis=-1
    )
    tokens = residual_block(model.embedding, features.astype(_dtype(model)))
    return tokens, masks.all(axis=-1)


def _first(
    masked: Bool[Array, " rows time"], real: Bool[Array, " rows"], patch: int
) -> Int[Array, ""]:
    """The first patch a real target row observes a value in, where the group starts.

    Patches before it are padding: no row reads them, and positions count from
    it. Upstream drops the steps before the targets' first observation and
    counts positions from the start of its padded batch, and the norms after
    RoPE make that count matter; counting from the targets' first patch
    matches upstream on a batch of the group alone.
    """
    observed = (~masked).reshape(masked.shape[0], -1, patch).any(axis=-1)
    return jnp.argmax((observed & real[:, None]).any(axis=0)).astype(jnp.int32)


def _visible(
    fully_masked: Bool[Array, " rows patches"], first: Int[Array, ""]
) -> Bool[Array, " rows patches"]:
    """Which patches attention reads: all but padding and a row's leading masked ones."""
    begun = jnp.cumsum(~fully_masked, axis=-1) > 0
    return begun & (jnp.arange(fully_masked.shape[-1]) >= first)


def _transform(
    model: TimesFM,
    tokens: Float[Array, " rows slots model"],
    positions: Int[Array, " slots"],
    time_mask: Bool[Array, " rows slots keys"],
    variate_mask: Bool[Array, " slots rows rows"],
    prefix: tuple[AttentionCache, ...] | None = None,
) -> tuple[Float[Array, " rows slots model"], tuple[AttentionCache, ...]]:
    """Runs the layers on tokens, with cached keys before the tokens' own.

    Args:
        model: TimesFM parameters.
        tokens: Token embeddings.
        positions: Each slot's position for RoPE.
        time_mask: Which keys, the prefix's then the tokens' own, each token reads.
        variate_mask: Which rows each row reads at each slot.
        prefix: Keys and values of earlier tokens per layer, `(rows, patches,
            heads, head_dim)`, or None.

    Returns:
        The outputs, and the tokens' own keys and values per layer.
    """
    caches: list[AttentionCache] = []
    for index, layer in enumerate(model.layers):
        normed = rms_norm(layer.sequence_norm, tokens)
        own = project_kv(layer.sequence, normed, positions=positions)
        caches.append(own)
        keys = own
        if prefix is not None:
            keys = AttentionCache(
                key=jnp.concatenate([prefix[index].key, own.key], axis=1),
                value=jnp.concatenate([prefix[index].value, own.value], axis=1),
            )
        attended = attend(
            layer.sequence, normed, keys._replace(mask=time_mask), positions=positions
        )
        tokens = rms_norm(layer.sequence_output_norm, attended) + tokens
        flipped = tokens.swapaxes(0, 1)
        normed = rms_norm(layer.variate_norm, flipped)
        keys = project_kv(layer.variate, normed)._replace(mask=variate_mask)
        attended = rms_norm(
            layer.variate_output_norm, attend(layer.variate, normed, keys)
        )
        tokens = (attended + flipped).swapaxes(0, 1)
        hidden = jnn.relu(
            linear(layer.feed_forward_input, rms_norm(layer.feed_forward_norm, tokens))
        )
        tokens = (
            rms_norm(
                layer.feed_forward_output_norm,
                linear(layer.feed_forward_output, hidden),
            )
            + tokens
        )
    return tokens, tuple(caches)


def _refined(
    model: TimesFM,
    moments: Moments,
    last: Float[Array, " rows outputs"],
    raw: Float[Array, " rows patches outputs"],
) -> Moments:
    """The horizon patches' frames, from the context's moments and the model's medians.

    The last context patch's prediction seeds the first `rolls` horizon
    patches; every `rolls` patches, the latest horizon patch's prediction, in
    the frame just refined, seeds the next.
    """
    rows, patches, _ = raw.shape
    rolls = model.output_patch // model.patch
    middle = len(model.quantiles) // 2

    def medians(outputs: Array) -> Array:
        return outputs.reshape(
            *outputs.shape[:-1], rolls, model.patch, len(model.quantiles)
        )[..., middle]

    def predicted(outputs: Array, frame: Moments) -> Array:
        values = (
            medians(outputs) * frame.deviation[:, None, None]
            + frame.mean[:, None, None]
        )
        return jnp.clip(values, -model.value_clip, model.value_clip)

    anchor = predicted(last, moments)
    unmasked = jnp.zeros((rows, model.patch), bool)
    refined = []
    for index in range(patches):
        moments = _update(moments, anchor[:, index % rolls], unmasked)
        refined.append(moments)
        if (index + 1) % rolls == 0:
            anchor = predicted(raw[:, index], moments)
    return Moments(
        *(jnp.stack(leaves, axis=1) for leaves in zip(*refined, strict=True))
    )


def _stitched(
    model: TimesFM, predictions: Float[Array, " rows patches steps quantiles"]
) -> Float[Array, " rows horizon quantiles"]:
    """Consecutive patches' predictions as one series, blended where they overlap."""
    rows, patches, steps, quantiles = predictions.shape
    overlap = steps - model.patch
    if patches == 1:
        return predictions[:, 0]
    weights = jnp.linspace(1.0, 0.0, overlap)[None, None, :, None]
    earlier = predictions[:, :-1, model.patch :]
    later = predictions[:, 1:, :overlap]
    blended = weights * earlier + (1 - weights) * later
    middles = predictions[:, 1:, overlap : model.patch]
    chunks = jnp.concatenate([blended, middles], axis=2).reshape(rows, -1, quantiles)
    return jnp.concatenate(
        [predictions[:, 0, : model.patch], chunks, predictions[:, -1, model.patch :]],
        axis=1,
    )


def _sizes(model: TimesFM, horizon: int) -> tuple[int, int, int]:
    """The patches stitched for a horizon, the horizon patches run, and their steps."""
    extract = min(2 * model.patch, model.output_patch)
    overlap = extract - model.patch
    stitched = max(math.ceil((horizon - overlap) / model.patch), 1)
    patches = stitched + model.output_patch // model.patch - 1
    return stitched, patches, extract


def _rows(model: TimesFM, group: SeriesGroup, horizon: int) -> Rows:
    """A group as the network reads it: trimmed, masked and clipped."""
    rows, steps = group.values.shape
    if steps % model.patch:
        raise ValueError(
            f"the context's {steps} steps are not whole patches of {model.patch}"
        )
    if rows > model.max_rows:
        raise ValueError(f"a group of {rows} rows passes the model's {model.max_rows}")
    limit = CONTEXT_LIMIT // model.patch * model.patch
    real = (
        jnp.ones(rows, bool)
        if group.num_rows is None
        else jnp.arange(rows) < group.num_rows
    )
    observed = (group.observed & real[:, None])[:, -limit:]
    first = _first(~observed, real & (group.roles == TARGET), model.patch)
    # A row the network would see nothing of is read as observed zeros, as
    # upstream's forecaster reads an all-missing series.
    observed = observed | ~observed.any(axis=-1, keepdims=True)
    values = jnp.where(observed, group.values[:, -limit:], 0.0).astype(jnp.float32)
    values = jnp.clip(jnp.nan_to_num(values), -model.value_clip, model.value_clip)
    if group.future is None:
        future = jnp.zeros((rows, horizon), jnp.float32)
        future_observed = jnp.zeros((rows, horizon), bool)
    else:
        future_observed = (
            jnp.isfinite(group.future)
            if group.future_observed is None
            else group.future_observed
        )
        future_observed = (
            future_observed & real[:, None] & (group.roles == KNOWN)[:, None]
        )
        future = jnp.where(future_observed, group.future, 0.0).astype(jnp.float32)
        future = jnp.clip(jnp.nan_to_num(future), -model.value_clip, model.value_clip)
    return Rows(values, ~observed, group.roles, real, future, ~future_observed, first)


def _finish(
    model: TimesFM,
    outputs: Float[Array, " rows patches outputs"],
    frames: Moments,
    trend: Trend,
    context: int,
    horizon: int,
) -> Float[Array, " rows horizon quantiles"]:
    """Predictions on the original scale from the last context patch's outputs on.

    Each patch's outputs are denormalized in its frame, consecutive patches
    are stitched, the trend comes back, and the quantiles are sorted.
    """
    rows, patches, _ = outputs.shape
    stitched, _, extract = _sizes(model, horizon)
    values = outputs * frames.deviation[..., None] + frames.mean[..., None]
    values = jnp.clip(values, -model.value_clip, model.value_clip)
    values = values.reshape(rows, patches, model.output_patch, len(model.quantiles))
    predictions = _stitched(model, values[:, :stitched, :extract])[:, :horizon]
    if model.detrending is not None:
        time = jnp.arange(1, horizon + 1, dtype=jnp.float32) / (context * model.patch)
        line = trend.slope[:, None] * time[None] + trend.intercept[:, None]
        predictions = (
            predictions + jnp.where(trend.applies[:, None], line, 0.0)[..., None]
        )
    return jnp.sort(predictions, axis=-1)


def forecast(
    model: TimesFM, group: SeriesGroup, horizon: int
) -> Float[Array, " rows horizon quantiles"]:
    """Forecasts `horizon` steps for every row of a group in one pass.

    Args:
        model: TimesFM parameters.
        group: The group; known future values ride in `future`, observed
            where `future_observed` says, or where finite without it.
        horizon: Steps to forecast.

    Returns:
        Quantiles at the model's levels for every row and step, sorted along
        the last axis as upstream's forecaster sorts them.
    """
    data = _rows(model, group, horizon)
    rows, steps = data.values.shape
    context = steps // model.patch
    _, patches, _ = _sizes(model, horizon)
    trend = _trend(model, data.values, data.masked)
    values = jnp.where(
        data.masked,
        0.0,
        _detrended(
            trend, data.values, jnp.arange(-(steps - 1), 1, dtype=jnp.float32) / steps
        ),
    )
    future = _detrended(
        trend, data.future, jnp.arange(1, horizon + 1, dtype=jnp.float32) / steps
    )
    future = padded_horizon(
        jnp.where(data.future_masked, 0.0, future), patches * model.patch, 0.0
    )
    future_masked = padded_horizon(data.future_masked, patches * model.patch, True)
    values = jnp.concatenate([values, future], axis=1)
    masked = jnp.concatenate([data.masked, future_masked], axis=1)
    target = data.roles != KNOWN
    moments = _running(model, values, masked)
    tokens, fully_masked = _embed(model, values, masked, target, moments)
    visible = _visible(fully_masked, data.first)
    total = context + patches
    time_mask = jnp.tril(jnp.ones((total, total), bool))[None] & visible[:, None, :]
    variate_mask = (visible.T & data.real[None, :])[:, None, :]
    variate_mask = jnp.broadcast_to(variate_mask, (total, rows, rows))
    positions = jnp.arange(total) - data.first
    outputs, _ = _transform(model, tokens, positions, time_mask, variate_mask)
    raw = linear(model.head, outputs).astype(jnp.float32)
    at_last = jax.tree.map(lambda leaf: leaf[:, context - 1], moments)
    refined = _refined(model, at_last, raw[:, context - 1], raw[:, context:])
    frames = jax.tree.map(
        lambda last, later: jnp.concatenate([last[:, None], later], axis=1),
        at_last,
        refined,
    )
    return _finish(model, raw[:, context - 1 :], frames, trend, context, horizon)


def encode(model: TimesFM, group: SeriesGroup) -> TimesFMCache:
    """Encodes a group's context, for any horizon to decode from.

    Args:
        model: TimesFM parameters.
        group: The group, without known covariates: their tokens carry their
            future, so a group with one is forecast whole.

    Returns:
        The context cache, patches first.

    Raises:
        ValueError: If a row is a known covariate, when that can be seen.
    """
    if not isinstance(group.roles, jax.core.Tracer) and bool(
        (group.roles == KNOWN).any()
    ):
        raise ValueError(
            "a group with a known covariate is forecast whole, not encoded"
        )
    data = _rows(model, group, 0)
    rows, steps = data.values.shape
    context = steps // model.patch
    trend = _trend(model, data.values, data.masked)
    values = jnp.where(
        data.masked,
        0.0,
        _detrended(
            trend, data.values, jnp.arange(-(steps - 1), 1, dtype=jnp.float32) / steps
        ),
    )
    moments = _running(model, values, data.masked)
    tokens, fully_masked = _embed(
        model, values, data.masked, jnp.ones(rows, bool), moments
    )
    visible = _visible(fully_masked, data.first)
    time_mask = jnp.tril(jnp.ones((context, context), bool))[None] & visible[:, None, :]
    variate_mask = jnp.broadcast_to(
        (visible.T & data.real[None, :])[:, None, :], (context, rows, rows)
    )
    outputs, caches = _transform(
        model, tokens, jnp.arange(context) - data.first, time_mask, variate_mask
    )
    last = linear(model.head, outputs[:, -1]).astype(jnp.float32)
    # A known covariate's tokens carry its future, so its cache would forecast
    # wrongly; NaN outputs, which every frame reads, make that show.
    last = jnp.where((data.roles == KNOWN).any(), jnp.nan, last)
    return TimesFMCache(
        sequence=tuple(
            jax.tree.map(lambda leaf: leaf.swapaxes(0, 1), layer) for layer in caches
        ),
        visible=visible.T,
        first=data.first,
        moments=jax.tree.map(lambda leaf: leaf[:, -1], moments),
        trend=trend,
        last=last,
        roles=data.roles,
        real=data.real,
    )


def decode(
    model: TimesFM,
    cache: TimesFMCache,
    horizon: int,
    future: Array | None = None,
    future_observed: Array | None = None,
) -> Float[Array, " rows horizon quantiles"]:
    """Forecasts `horizon` steps from an encoded context in one pass.

    Args:
        model: TimesFM parameters.
        cache: The context from `encode`.
        horizon: Steps to forecast.
        future: Unused: a group with known covariates is forecast whole.
        future_observed: Unused.

    Returns:
        Quantiles at the model's levels for every row and step, sorted.
    """
    del future, future_observed
    context, rows = cache.visible.shape
    _, patches, _ = _sizes(model, horizon)
    blank = jnp.zeros((rows, patches * model.patch), jnp.float32)
    masked = jnp.ones((rows, patches * model.patch), bool)
    unchanged = jax.tree.map(
        lambda leaf: jnp.repeat(leaf[:, None], patches, axis=1), cache.moments
    )
    tokens, _ = _embed(model, blank, masked, jnp.ones(rows, bool), unchanged)
    time_mask = jnp.concatenate(
        [
            jnp.broadcast_to(cache.visible.T[:, None, :], (rows, patches, context)),
            jnp.broadcast_to(
                jnp.tril(jnp.ones((patches, patches), bool)), (rows, patches, patches)
            ),
        ],
        axis=-1,
    )
    # The horizon's tokens are keys for every real row, since no real row is
    # fully masked from its start.
    variate_mask = jnp.broadcast_to(cache.real[None, None, :], (patches, rows, rows))
    prefix = tuple(jax.tree.map(_along_rows, layer) for layer in cache.sequence)
    positions = context + jnp.arange(patches) - cache.first
    outputs, _ = _transform(model, tokens, positions, time_mask, variate_mask, prefix)
    raw = linear(model.head, outputs).astype(jnp.float32)
    refined = _refined(model, cache.moments, cache.last, raw)
    frames = jax.tree.map(
        lambda last, later: jnp.concatenate([last[:, None], later], axis=1),
        cache.moments,
        refined,
    )
    outputs = jnp.concatenate([cache.last[:, None], raw], axis=1)
    return _finish(model, outputs, frames, cache.trend, context, horizon)


def properties(model: TimesFM) -> SeriesProperties:
    """Describes TimesFM 3 from its parameters and settings.

    One pass covers any horizon, so there is no pass limit.

    Args:
        model: TimesFM parameters, real or abstract.

    Returns:
        The model's properties.
    """
    return SeriesProperties(
        patch=model.patch,
        output_patch=model.patch,
        context_limit=CONTEXT_LIMIT,
        pass_limit=None,
        quantiles=model.quantiles,
        reusable=True,
        max_rows=model.max_rows,
    )


timesfm_functions = SeriesFunctions(
    forecast=forecast,
    encode=encode,
    decode=decode,
    extend=None,
    properties=properties,
)
