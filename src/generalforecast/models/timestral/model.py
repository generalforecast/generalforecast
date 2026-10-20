"""Timestral: a causal patch transformer that decodes blocks of query tokens.

Each row is normalized causally per patch and embedded from its values and
missing flags. The sequence is the context's patches followed by one learned
query token per horizon patch. Layers attend along time within a row, where
the context is causal and the queries read the whole context and their own
causal chain, and on some layers across the rows of a group at each patch.
The queries decode to monotone quantiles, which the last context patch's
frame maps back, then each later horizon patch moves to the frame the block
would have had with its medians as context. Horizons past a block continue
with the medians fed back as context.

The context's keys and values do not depend on the horizon unless a row is a
known covariate, whose context reads its future, so a group without one
encodes once and decodes any horizon, growing exactly by whole patches.
"""

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
    PAST,
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
    GatedMLPParameters,
    ResidualBlockParameters,
    gated_mlp,
    residual_block,
)
from generalforecast.nn.norm import RMSNormParameters, rms_norm

# Forecasts are clamped to the observed context's range widened by this many
# scales of its last patch, and fed-back medians of this size or more become 0.
RANGE_CAP = 1e4
EXTREME = 1e30


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TimestralLayer:
    """One or two pre-norm attentions, along time or across rows, then a SwiGLU.

    `along_time` is static: it says which axis each attention runs on.
    """

    norms: tuple[RMSNormParameters, ...]
    attentions: tuple[AttentionParameters, ...]
    mlp_norm: RMSNormParameters
    mlp: GatedMLPParameters
    along_time: tuple[bool, ...] = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Timestral:
    """Timestral's embedding, query token, layers and head, with its settings.

    Attributes:
        patch_embedding: Embeds a patch's values and missing flags.
        query: The forecast query token.
        layers: The transformer layers.
        final_norm: Norm of the transformer's output.
        head: Maps a query token to its patch's raw quantiles.
        patch: Steps per patch.
        block_size: Most patches one pass decodes.
        context_limit: Most context steps read.
        quantiles: The levels of the output quantiles, symmetric about 0.5.
        width_factor: Scale of the gaps between neighboring quantiles.
        minimum_scale: Floor of a row's scale.
        symmetric: Whether to forecast the negated group too and average the
            two, as upstream does by default; it doubles the work.
    """

    patch_embedding: ResidualBlockParameters
    query: Float[Array, " model"]
    layers: tuple[TimestralLayer, ...]
    final_norm: RMSNormParameters
    head: ResidualBlockParameters
    patch: int = field(metadata={"static": True})
    block_size: int = field(metadata={"static": True})
    context_limit: int = field(metadata={"static": True})
    quantiles: tuple[float, ...] = field(metadata={"static": True})
    width_factor: float = field(metadata={"static": True})
    minimum_scale: float = field(metadata={"static": True})
    symmetric: bool = field(default=True, metadata={"static": True})


class Welford(NamedTuple):
    """Running moments of observed values: their count, mean and sum of squares."""

    count: Array
    mean: Array
    m2: Array


class Scale(NamedTuple):
    """A location and scale, in float32."""

    loc: Array
    scale: Array


class Rows(NamedTuple):
    """A group's rows as the network reads them, mirrored when averaging over signs.

    Attributes:
        values: Context values, 0 where unobserved, `(rows, time)` float32.
        observed: Which values are observed, False on padding rows.
        roles: Each row's role.
        sign: Each row's sign: 0, or 1 for the negated copy of a row.
        real: Which rows are real rather than padding.
        future: Known future values over the horizon, 0 elsewhere.
        future_observed: Which future values are known.
    """

    values: Array
    observed: Array
    roles: Array
    sign: Array
    real: Array
    future: Array
    future_observed: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TimestralCache:
    """A group's encoded context, its patches first.

    Attributes:
        time: Keys and values per attention along time, `(patches, rows,
            heads, head_dim)`, keys rotated at their positions.
        valid: Which patches are keys for each row, `(patches, rows)`: the
            observed ones and those after a row's first.
        statistics: The context's running moments per row at its last step.
        lowest: The lowest observed value per row, or infinity for none.
        highest: The highest observed value per row, or minus infinity.
        nonnegative: Which rows have no negative observed value.
        first: The first patch any row of the group observes.
        roles: Each row's role.
        sign: Each row's sign.
        real: Which rows are real.

    A cache the runtime reads back from its slots may carry `length` on
    its attention caches: how many patches are real, which come first; the
    rest are zeros, which `valid` marks as no keys.
    """

    time: tuple[AttentionCache, ...]
    valid: Bool[Array, " patches rows"]
    statistics: Welford
    lowest: Float[Array, " rows"]
    highest: Float[Array, " rows"]
    nonnegative: Bool[Array, " rows"]
    first: Int[Array, ""]
    roles: Int[Array, " rows"]
    sign: Int[Array, " rows"]
    real: Bool[Array, " rows"]


def _dtype(model: Timestral) -> jnp.dtype:
    """The dtype the network computes in, set by its parameters."""
    return model.query.dtype


def _combine(a: Welford, b: Welford) -> Welford:
    """Merges two sets of running moments; no observations is the identity."""
    count = a.count + b.count
    safe = jnp.maximum(count, jnp.finfo(jnp.float32).tiny)
    delta = b.mean - a.mean
    mean = a.mean + delta * b.count / safe
    m2 = a.m2 + b.m2 + jnp.square(delta) * a.count * b.count / safe
    empty = count == 0
    return Welford(count, jnp.where(empty, 0.0, mean), jnp.where(empty, 0.0, m2))


def _running(
    values: Float[Array, " rows time"],
    observed: Bool[Array, " rows time"],
    before: Welford | None = None,
) -> Welford:
    """The moments of each row's observed values up to each step, after `before`'s."""
    steps = Welford(
        observed.astype(jnp.float32),
        jnp.where(observed, values, 0.0),
        jnp.zeros_like(values),
    )
    running = jax.lax.associative_scan(_combine, steps, axis=-1)
    if before is None:
        return running
    return _combine(jax.tree.map(lambda leaf: leaf[..., None], before), running)


def _scale_of(model: Timestral, state: Welford) -> Scale:
    """The frame of running moments: their mean, and their deviation floored."""
    variance = state.m2 / jnp.maximum(state.count - 1, 1.0)
    return Scale(state.mean, jnp.maximum(jnp.sqrt(variance), model.minimum_scale))


def _standardize(
    model: Timestral,
    values: Float[Array, " rows time"],
    observed: Bool[Array, " rows time"],
    frame: Scale,
    asinh: Array | bool = True,
) -> Float[Array, " rows time"]:
    """`asinh((values - loc) / scale)` where observed, 0 elsewhere, in float32."""
    scaled = (values - frame.loc) / frame.scale
    return jnp.where(observed, jnp.where(asinh, jnp.arcsinh(scaled), scaled), 0.0)


def _patch_frames(model: Timestral, running: Welford) -> Scale:
    """Each patch's frame, the moments at its last step, repeated over its steps."""
    ends = jax.tree.map(lambda leaf: leaf[..., model.patch - 1 :: model.patch], running)
    frame = _scale_of(model, ends)
    return Scale(*(jnp.repeat(leaf, model.patch, axis=-1) for leaf in frame))


def _scaled(
    model: Timestral,
    values: Float[Array, " rows time"],
    observed: Bool[Array, " rows time"],
    roles: Int[Array, " rows"],
    before: Welford | None = None,
) -> tuple[Float[Array, " rows time"], Welford]:
    """Normalized values and the running moments at every step.

    Rows are normalized causally per patch. A known covariate is normalized
    over its whole row instead, future included, and a known indicator row of
    zeros and ones is left as it is.

    Args:
        model: Timestral parameters.
        values: Row values, 0 where unobserved.
        observed: Which values are observed.
        roles: Each row's role.
        before: The moments of earlier steps, when `values` continue a context.

    Returns:
        The normalized values and the running moments.
    """
    running = _running(values, observed, before)
    scaled = _standardize(model, values, observed, _patch_frames(model, running))
    known = (roles == KNOWN)[:, None]
    count = jnp.maximum(observed.sum(axis=-1, keepdims=True), 1)
    loc = jnp.where(observed, values, 0.0).sum(axis=-1, keepdims=True) / count
    deviation = jnp.where(observed, jnp.square(values - loc), 0.0)
    scale = jnp.maximum(
        jnp.sqrt(deviation.sum(axis=-1, keepdims=True) / count), model.minimum_scale
    )
    indicator = ((values == 0) | (values == 1) | ~observed).all(axis=-1, keepdims=True)
    indicator = indicator & observed.any(axis=-1, keepdims=True)
    whole = Scale(jnp.where(indicator, 0.0, loc), jnp.where(indicator, 1.0, scale))
    return jnp.where(
        known, _standardize(model, values, observed, whole, ~indicator), scaled
    ), running


def _embed(
    model: Timestral,
    scaled: Float[Array, " rows time"],
    observed: Bool[Array, " rows time"],
) -> Float[Array, " rows patches model"]:
    """Embeds each patch's normalized values and missing flags."""
    rows = scaled.shape[0]
    features = jnp.concatenate(
        [
            scaled.reshape(rows, -1, model.patch),
            (~observed).reshape(rows, -1, model.patch).astype(jnp.float32),
        ],
        axis=-1,
    )
    return residual_block(model.patch_embedding, features.astype(_dtype(model)))


def _valid(
    model: Timestral,
    observed: Bool[Array, " rows time"],
    started: Bool[Array, " rows"] | None = None,
) -> Bool[Array, " rows patches"]:
    """Which patches are keys: the observed ones, and every one after a row's first."""
    patches = observed.reshape(observed.shape[0], -1, model.patch).any(axis=-1)
    begun = jnp.cumsum(patches, axis=-1) > 0
    if started is not None:
        begun = begun | started[:, None]
    return patches | begun


def _firewall(roles: Int[Array, " rows"]) -> Bool[Array, " rows rows"]:
    """Rows that may not read rows: covariates the targets, known ones the rest."""
    target, known = roles == TARGET, roles == KNOWN
    return (~target[:, None] & target[None, :]) | (known[:, None] & ~known[None, :])


def _variate_mask(
    rows: Rows | TimestralCache, valid: Bool[Array, " slots rows"]
) -> Bool[Array, " slots rows rows"]:
    """Which rows each row reads at each slot: its group's valid keys, past the firewall."""
    same = (rows.sign[:, None] == rows.sign[None, :]) & rows.real[None, :]
    mask = same & ~_firewall(rows.roles)
    mask = mask[None] & valid[:, None, :]
    return mask | jnp.eye(rows.sign.shape[0], dtype=bool)[None]


def _causal(queries: int, keys: int) -> Bool[Array, " queries keys"]:
    """Queries that end a sequence of `keys` read the keys at or before them."""
    return jnp.tril(jnp.ones((queries, keys), bool), keys - queries)


def _transform(
    model: Timestral,
    tokens: Float[Array, " rows slots model"],
    positions: Int[Array, " slots"],
    time_mask: Bool[Array, " rows slots keys"],
    variate_mask: Bool[Array, " slots rows rows"],
    prefix: tuple[AttentionCache, ...] | None = None,
) -> tuple[Float[Array, " rows slots model"], tuple[AttentionCache, ...]]:
    """Runs the layers on tokens, with cached keys before the tokens' own.

    Args:
        model: Timestral parameters.
        tokens: Token embeddings.
        positions: Each slot's position for RoPE.
        time_mask: Which keys, the prefix's then the tokens' own, each token reads.
        variate_mask: Which rows each row reads at each slot.
        prefix: Keys and values of earlier tokens per attention along time,
            `(rows, patches, heads, head_dim)`, or None.

    Returns:
        The normalized outputs, and the tokens' own keys and values per
        attention along time.
    """
    caches: list[AttentionCache] = []
    for layer in model.layers:
        for norm, attention, along_time in zip(
            layer.norms, layer.attentions, layer.along_time, strict=True
        ):
            if along_time:
                normed = rms_norm(norm, tokens)
                own = project_kv(attention, normed, positions=positions)
                keys = own
                if prefix is not None:
                    earlier = prefix[len(caches)]
                    keys = AttentionCache(
                        key=jnp.concatenate([earlier.key, own.key], axis=1),
                        value=jnp.concatenate([earlier.value, own.value], axis=1),
                    )
                caches.append(own)
                keys = keys._replace(mask=time_mask)
                tokens = tokens + attend(attention, normed, keys, positions=positions)
            else:
                flipped = tokens.swapaxes(0, 1)
                normed = rms_norm(norm, flipped)
                keys = project_kv(attention, normed)._replace(mask=variate_mask)
                tokens = (flipped + attend(attention, normed, keys)).swapaxes(0, 1)
        tokens = tokens + gated_mlp(layer.mlp, rms_norm(layer.mlp_norm, tokens))
    return rms_norm(model.final_norm, tokens), tuple(caches)


def _quantiles(
    model: Timestral, tokens: Float[Array, " rows patches model"]
) -> Float[Array, " rows steps quantiles"]:
    """Monotone quantiles per step from query tokens, in the normalized space.

    The median is free; the others fan out from it by `width_factor` times the
    probability gap to the next level inward times a softplus of the raw output.
    """
    rows, patches, _ = tokens.shape
    levels = model.quantiles
    middle = levels.index(0.5)
    raw = residual_block(model.head, tokens).astype(jnp.float32)
    raw = raw.reshape(rows, patches, model.patch, len(levels))
    widths = [0.0] * len(levels)
    for index in range(middle + 1, len(levels)):
        widths[index] = levels[index] - levels[index - 1]
    for index in range(middle):
        widths[index] = levels[index + 1] - levels[index]
    gaps = model.width_factor * jnp.asarray(widths, jnp.float32) * jnn.softplus(raw)
    median = raw[..., middle : middle + 1]
    up = jnp.cumsum(gaps[..., middle + 1 :], axis=-1)
    down = jnp.cumsum(gaps[..., :middle][..., ::-1], axis=-1)[..., ::-1]
    quantiles = jnp.concatenate([median - down, median, median + up], axis=-1)
    biggest = jnp.finfo(jnp.float32).max
    return jnp.clip(quantiles, -biggest, biggest).reshape(rows, -1, len(levels))


def _finite(values: Float[Array, " ... quantiles"]) -> Float[Array, " ... quantiles"]:
    """Infinities become the nearest finite quantile, or 0 when none is finite."""
    finite = jnp.isfinite(values)
    high = jnp.max(jnp.where(finite, values, -jnp.inf), axis=-1, keepdims=True)
    low = jnp.min(jnp.where(finite, values, jnp.inf), axis=-1, keepdims=True)
    clamped = jnp.where(
        values == jnp.inf, high, jnp.where(values == -jnp.inf, low, values)
    )
    return jnp.where(jnp.isinf(clamped), 0.0, clamped)


def _denormalize(
    quantiles: Float[Array, " rows steps quantiles"], frame: Scale
) -> Float[Array, " rows steps quantiles"]:
    """Maps normalized quantiles back through each row's frame."""
    values = jnp.sinh(quantiles) * frame.scale[:, None, None] + frame.loc[:, None, None]
    return _finite(values)


def _advancing(
    model: Timestral, state: Welford, sinh_median: Float[Array, " rows patches patch"]
) -> Scale:
    """Each horizon patch's frame, with the block's earlier medians folded in.

    The medians arrive in the last context patch's frame with the asinh undone;
    each is denormalized in its own patch's frame before its moments join.
    """
    loc, scale = [], []
    offset = state.mean
    clip = jnp.finfo(jnp.float32).max ** 0.5 / 1e3
    for index in range(sinh_median.shape[1]):
        frame = _scale_of(model, state)
        loc.append(frame.loc)
        scale.append(frame.scale)
        folded = (
            sinh_median[:, index] * frame.scale[:, None] + (frame.loc - offset)[:, None]
        )
        folded = jnp.clip(folded, -clip, clip) + offset[:, None]
        mean = folded.mean(axis=-1)
        m2 = jnp.square(folded - mean[:, None]).sum(axis=-1)
        patch = Welford(jnp.full_like(mean, model.patch), mean, m2)
        state = _combine(state, patch)
    return Scale(jnp.stack(loc, axis=1), jnp.stack(scale, axis=1))


def _refine(
    model: Timestral,
    quantiles: Float[Array, " rows steps quantiles"],
    state: Welford,
    roles: Int[Array, " rows"],
) -> Float[Array, " rows steps quantiles"]:
    """Moves a block from the last context patch's frame to the advancing frames.

    The first horizon patch keeps the context's frame; each later one takes the
    frame of the context plus the block's medians before it. Covariate rows keep
    their quantiles.
    """
    patches = quantiles.shape[1] // model.patch
    if patches <= 1:
        return quantiles
    anchor = _scale_of(model, state)
    median = quantiles[..., model.quantiles.index(0.5)].reshape(
        -1, patches, model.patch
    )
    sinh_median = (median - anchor.loc[:, None, None]) / anchor.scale[:, None, None]
    frames = _advancing(model, state, sinh_median)
    first = jnp.arange(patches) == 0
    loc = jnp.repeat(
        jnp.where(first, anchor.loc[:, None], frames.loc), model.patch, axis=-1
    )
    scale = jnp.repeat(
        jnp.where(first, anchor.scale[:, None], frames.scale), model.patch, axis=-1
    )
    normalized = (quantiles - anchor.loc[:, None, None]) / anchor.scale[:, None, None]
    refined = _finite(normalized * scale[..., None] + loc[..., None])
    return jnp.where((roles == TARGET)[:, None, None], refined, quantiles)


def _rows(model: Timestral, group: SeriesGroup, horizon: int) -> Rows:
    """A group as the network reads it: trimmed, masked, and mirrored if averaging."""
    rows, steps = group.values.shape
    if steps % model.patch:
        raise ValueError(
            f"the context's {steps} steps are not whole patches of {model.patch}"
        )
    limit = model.context_limit // model.patch * model.patch
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
        future_observed = (
            future_observed & real[:, None] & (group.roles == KNOWN)[:, None]
        )
        future = jnp.where(future_observed, group.future, 0.0).astype(jnp.float32)
    data = Rows(
        values,
        observed,
        group.roles,
        jnp.zeros(rows, jnp.int32),
        real,
        future,
        future_observed,
    )
    if not model.symmetric:
        return data
    return Rows(
        jnp.concatenate([values, -values]),
        jnp.concatenate([observed, observed]),
        jnp.concatenate([group.roles, group.roles]),
        jnp.repeat(jnp.arange(2, dtype=jnp.int32), rows),
        jnp.concatenate([real, real]),
        jnp.concatenate([future, -future]),
        jnp.concatenate([future_observed, future_observed]),
    )


def _length(cache: TimestralCache) -> Int[Array, ""] | None:
    """How many of a cache's patches are real, from its attention caches' lengths; None when all are."""
    return cache.time[0].length if cache.time else None


def _along_rows(leaf: Array) -> Array:
    """A stored attention cache leaf with rows first, as attention along time reads it; a cache's scalar length stays as it is."""
    return leaf.swapaxes(0, 1) if leaf.ndim > 1 else leaf


def _first(observed: Bool[Array, " rows time"], patch: int) -> Int[Array, ""]:
    """The first patch any row observes, or 0 when none does."""
    patches = observed.reshape(observed.shape[0], -1, patch).any(axis=(0, 2))
    return jnp.argmax(patches).astype(jnp.int32)


def _block(
    model: Timestral,
    data: Rows,
    future: Float[Array, " rows steps"],
    future_observed: Bool[Array, " rows steps"],
) -> Float[Array, " rows steps quantiles"]:
    """One pass over a context and a block of horizon patches, as upstream runs it.

    The context and the horizon together are the window: known covariates'
    context patches read their future ones, and the query tokens of the horizon
    read the context and each other.
    """
    rows, steps = data.values.shape
    context = steps // model.patch
    horizon = future.shape[1] // model.patch
    values = jnp.concatenate([data.values, future], axis=1)
    observed = jnp.concatenate([data.observed, future_observed], axis=1)
    scaled, running = _scaled(model, values, observed, data.roles)
    window = context + horizon
    tokens = jnp.concatenate(
        [
            _embed(model, scaled, observed),
            jnp.broadcast_to(model.query, (rows, horizon, model.query.shape[0])).astype(
                _dtype(model)
            ),
        ],
        axis=1,
    )
    valid = _valid(model, observed)
    known = (data.roles == KNOWN)[:, None, None]
    eye = jnp.eye(window, dtype=bool)
    encoder = (
        jnp.where(known, valid[:, None, :], _causal(window, window) & valid[:, None, :])
        | eye
    )
    queries = jnp.concatenate(
        [
            jnp.broadcast_to(
                valid[:, None, :] & (jnp.arange(window) < context),
                (rows, horizon, window),
            ),
            jnp.broadcast_to(
                _causal(horizon, horizon) | jnp.eye(horizon, dtype=bool),
                (rows, horizon, horizon),
            ),
        ],
        axis=-1,
    )
    time_mask = jnp.concatenate(
        [
            jnp.concatenate(
                [encoder, jnp.zeros((rows, window, horizon), bool)], axis=-1
            ),
            queries,
        ],
        axis=1,
    )
    slots = jnp.concatenate([valid, jnp.ones((rows, horizon), bool)], axis=1).T
    positions = jnp.concatenate([jnp.arange(window), context + jnp.arange(horizon)])
    positions = positions - _first(observed, model.patch)
    outputs, _ = _transform(
        model, tokens, positions, time_mask, _variate_mask(data, slots)
    )
    anchor = _scale_of(model, jax.tree.map(lambda leaf: leaf[:, steps - 1], running))
    return _denormalize(_quantiles(model, outputs[:, window:]), anchor)


def _range(
    values: Float[Array, " rows steps"],
    observed: Bool[Array, " rows steps"],
    before: TimestralCache | None = None,
) -> tuple[Array, Array, Array]:
    """The observed values' range and sign, folded into a cache's if given."""
    lowest = jnp.min(jnp.where(observed, values, jnp.inf), axis=-1)
    highest = jnp.max(jnp.where(observed, values, -jnp.inf), axis=-1)
    nonnegative = ((values >= 0) | ~observed).all(axis=-1)
    if before is None:
        return lowest, highest, nonnegative
    return (
        jnp.minimum(before.lowest, lowest),
        jnp.maximum(before.highest, highest),
        before.nonnegative & nonnegative,
    )


def _bounds(
    model: Timestral, state: Welford, lowest: Array, highest: Array, nonnegative: Array
) -> tuple[Array, Array, Array]:
    """The range forecasts are clamped to, and which rows are clamped at 0.

    The range is the observed values' widened by `RANGE_CAP` scales of the
    context's last frame; a row without observed values is clamped around 0
    and not at it. A NaN range, from a cache a known covariate poisoned, makes
    every forecast NaN.
    """
    reach = _scale_of(model, state).scale * RANGE_CAP
    observed = jnp.isfinite(lowest)
    low = jnp.where(observed, lowest, jnp.where(jnp.isnan(lowest), lowest, 0.0))
    high = jnp.where(observed, highest, jnp.where(jnp.isnan(highest), highest, 0.0))
    return low - reach, high + reach, observed & nonnegative


def _finish(
    model: Timestral,
    quantiles: Float[Array, " rows steps quantiles"],
    data: Rows | TimestralCache,
    low: Array,
    high: Array,
    nonnegative: Array,
) -> Float[Array, " rows steps quantiles"]:
    """Clamps target rows to their range, folds the signs together, and clips at 0."""
    target = (data.roles == TARGET)[:, None, None]
    quantiles = jnp.where(
        target, jnp.clip(quantiles, low[:, None, None], high[:, None, None]), quantiles
    )
    if model.symmetric:
        rows = quantiles.shape[0] // 2
        mirrored = -quantiles[rows:][..., ::-1]
        quantiles = 0.5 * (quantiles[:rows] + mirrored)
        target, nonnegative = target[:rows], nonnegative[:rows]
    clip = target & nonnegative[:, None, None]
    return jnp.where(clip, jnp.maximum(quantiles, 0.0), quantiles)


def _feedback(
    model: Timestral,
    quantiles: Float[Array, " rows steps quantiles"],
    roles: Int[Array, " rows"],
    future: Float[Array, " rows steps"],
    future_observed: Bool[Array, " rows steps"],
) -> tuple[Float[Array, " rows steps"], Bool[Array, " rows steps"]]:
    """What a block adds to the context: medians, known futures, or blanks."""
    median = quantiles[..., model.quantiles.index(0.5)]
    median = jnp.where(jnp.isfinite(median) & (jnp.abs(median) < EXTREME), median, 0.0)
    known, past = (roles == KNOWN)[:, None], (roles == PAST)[:, None]
    values = jnp.where(known, future, jnp.where(past, 0.0, median))
    observed = jnp.where(known, future_observed, ~past)
    return values, observed


def forecast(
    model: Timestral, group: SeriesGroup, horizon: int
) -> Float[Array, " rows horizon quantiles"]:
    """Forecasts `horizon` steps for every row of a group, block by block.

    Each block runs the network over the context and the block, as upstream
    does, and feeds its medians back as context for the next; the context
    drops its oldest steps past the model's limit.

    Args:
        model: Timestral parameters.
        group: The group; known future values ride in `future`, observed
            where `future_observed` says, or where finite without it.
        horizon: Steps to forecast.

    Returns:
        Quantiles at the model's levels for every row and step.
    """
    data = _rows(model, group, horizon)
    first = jax.tree.map(lambda leaf: leaf[:, -1], _running(data.values, data.observed))
    low, high, nonnegative = _bounds(model, first, *_range(data.values, data.observed))
    limit = model.context_limit // model.patch * model.patch
    values, observed = data.values, data.observed
    blocks = []
    done = 0
    while done < horizon:
        steps = min(model.block_size, -(-(horizon - done) // model.patch)) * model.patch
        future = padded_horizon(data.future[:, done:], steps, 0.0)
        future_observed = padded_horizon(data.future_observed[:, done:], steps, False)
        context = data._replace(values=values, observed=observed)
        quantiles = _block(model, context, future, future_observed)
        last = jax.tree.map(lambda leaf: leaf[:, -1], _running(values, observed))
        quantiles = _refine(model, quantiles, last, data.roles)
        blocks.append(quantiles)
        done += steps
        if done < horizon:
            added, added_observed = _feedback(
                model, quantiles, data.roles, future, future_observed
            )
            values = jnp.concatenate([values, added], axis=1)[:, -limit:]
            observed = jnp.concatenate([observed, added_observed], axis=1)[:, -limit:]
    quantiles = jnp.concatenate(blocks, axis=1)[:, :horizon]
    return _finish(model, quantiles, data, low, high, nonnegative)


def _grow(
    model: Timestral,
    cache: TimestralCache | None,
    data: Rows,
    values: Float[Array, " rows steps"],
    observed: Bool[Array, " rows steps"],
    fold: bool = True,
) -> TimestralCache:
    """Encodes new context patches after a cache's, or the first ones.

    Args:
        model: Timestral parameters.
        cache: The context so far, or None.
        data: The rows, for their roles, signs and realness.
        values: The new steps' values, whole patches.
        observed: Which new values are observed.
        fold: Whether the new observed values widen the cache's range; the
            medians a decode feeds back do not, as `forecast` fixes the range
            once.

    Returns:
        The cache grown by the new patches.

    Raises:
        ValueError: If the context would pass the model's limit.
    """
    rows, steps = values.shape
    patches = steps // model.patch
    earlier = 0 if cache is None else cache.valid.shape[0]
    # A cache read back from the runtime's slots holds its real patches
    # first, then padding, and says how many are real.
    real = None if cache is None else _length(cache)
    if real is None and (earlier + patches) * model.patch > model.context_limit:
        raise ValueError(
            f"a context of {(earlier + patches) * model.patch} steps passes the "
            f"model's limit of {model.context_limit}"
        )
    held = earlier if real is None else real
    before = None if cache is None else cache.statistics
    scaled, running = _scaled(model, values, observed, data.roles, before)
    tokens = _embed(model, scaled, observed)
    if cache is None:
        started = None
    elif real is None:
        started = cache.valid[-1]
    else:
        started = jnp.take(cache.valid, real - 1, axis=0)
    valid = _valid(model, observed, started)
    own = _causal(patches, patches) & valid[:, None, :] | jnp.eye(patches, dtype=bool)
    if cache is None:
        time_mask, prefix, first = own, None, _first(observed, model.patch)
    else:
        first = cache.first
        time_mask = jnp.concatenate(
            [
                jnp.broadcast_to(cache.valid.T[:, None, :], (rows, patches, earlier)),
                own,
            ],
            -1,
        )
        prefix = tuple(jax.tree.map(_along_rows, layer) for layer in cache.time)
    positions = held + jnp.arange(patches) - first
    _, caches = _transform(
        model, tokens, positions, time_mask, _variate_mask(data, valid.T), prefix
    )
    time = tuple(
        jax.tree.map(lambda leaf: leaf.swapaxes(0, 1), layer) for layer in caches
    )
    last = jax.tree.map(lambda leaf: leaf[:, -1], running)
    if cache is None:
        lowest, highest, nonnegative = _range(values, observed)
        # A known covariate's context reads the horizon, so its cache would
        # forecast wrongly; a NaN range makes that show.
        known = (data.roles == KNOWN).any()
        lowest = jnp.where(known, jnp.nan, lowest)
        return TimestralCache(
            time,
            valid.T,
            last,
            lowest,
            highest,
            nonnegative,
            first,
            data.roles,
            data.sign,
            data.real,
        )
    if fold:
        lowest, highest, nonnegative = _range(values, observed, cache)
    else:
        lowest, highest, nonnegative = cache.lowest, cache.highest, cache.nonnegative

    # The new patches go after the real ones: at the end, or over the
    # padding of a cache read back from storage, which then says how many
    # patches are real.
    def grown(old: Array, new: Array) -> Array:
        if real is None:
            return jnp.concatenate([old, new])
        buffer = jnp.concatenate([old, jnp.zeros_like(new)])
        return jax.lax.dynamic_update_slice_in_dim(buffer, new, real, axis=0)

    length = None if real is None else real + patches
    return TimestralCache(
        tuple(
            AttentionCache(
                key=grown(old.key, new.key),
                value=grown(old.value, new.value),
                length=length,
            )
            for old, new in zip(cache.time, time, strict=True)
        ),
        grown(cache.valid, valid.T),
        last,
        lowest,
        highest,
        nonnegative,
        cache.first,
        cache.roles,
        cache.sign,
        cache.real,
    )


def encode(model: Timestral, group: SeriesGroup) -> TimestralCache:
    """Encodes a group's context, for any horizon to decode from.

    Args:
        model: Timestral parameters.
        group: The group, without known covariates: their context reads the
            horizon, so a group with one is forecast whole.

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
    return _grow(model, None, data, data.values, data.observed)


def decode(
    model: Timestral,
    cache: TimestralCache,
    horizon: int,
    future: Array | None = None,
    future_observed: Array | None = None,
) -> Float[Array, " rows horizon quantiles"]:
    """Forecasts `horizon` steps from an encoded context, block by block.

    Each block's query tokens read the cached context and each other; past one
    block, the context grows by the block's medians, exactly as encoding the
    longer context would, though without dropping steps past the model's limit.
    The range forecasts are clamped to is the context's at the start, as in
    `forecast`.

    Args:
        model: Timestral parameters.
        cache: The context from `encode`.
        horizon: Steps to forecast.
        future: Must be None: a group with known covariates is forecast whole.
        future_observed: Must be None.

    Returns:
        Quantiles at the model's levels for every row and step.

    Raises:
        ValueError: If known future values are given.
    """
    if future is not None or future_observed is not None:
        raise ValueError(
            "a group with a known covariate is forecast whole, not decoded"
        )
    rows = cache.roles.shape[0]
    low, high, nonnegative = _bounds(
        model, cache.statistics, cache.lowest, cache.highest, cache.nonnegative
    )
    blocks = []
    done = 0
    while done < horizon:
        patches = min(model.block_size, -(-(horizon - done) // model.patch))
        context = cache.valid.shape[0]
        real = _length(cache)
        tokens = jnp.broadcast_to(model.query, (rows, patches, model.query.shape[0]))
        own = _causal(patches, patches) | jnp.eye(patches, dtype=bool)
        time_mask = jnp.concatenate(
            [
                jnp.broadcast_to(cache.valid.T[:, None, :], (rows, patches, context)),
                jnp.broadcast_to(own, (rows, patches, patches)),
            ],
            axis=-1,
        )
        held = context if real is None else real
        positions = held + jnp.arange(patches) - cache.first
        prefix = tuple(jax.tree.map(_along_rows, layer) for layer in cache.time)
        slots = jnp.ones((patches, rows), bool)
        outputs, _ = _transform(
            model,
            tokens.astype(_dtype(model)),
            positions,
            time_mask,
            _variate_mask(cache, slots),
            prefix,
        )
        quantiles = _denormalize(
            _quantiles(model, outputs), _scale_of(model, cache.statistics)
        )
        quantiles = _refine(model, quantiles, cache.statistics, cache.roles)
        blocks.append(quantiles)
        done += patches * model.patch
        if done < horizon:
            steps = patches * model.patch
            blank = jnp.zeros((rows, steps), jnp.float32)
            added, added_observed = _feedback(
                model, quantiles, cache.roles, blank, jnp.zeros((rows, steps), bool)
            )
            data = Rows(
                added,
                added_observed,
                cache.roles,
                cache.sign,
                cache.real,
                blank,
                blank > 0,
            )
            cache = _grow(model, cache, data, added, added_observed, fold=False)
    quantiles = jnp.concatenate(blocks, axis=1)[:, :horizon]
    return _finish(model, quantiles, cache, low, high, nonnegative)


def extend(
    model: Timestral, cache: TimestralCache, group: SeriesGroup
) -> TimestralCache:
    """Grows an encoded context by whole patches of new steps, exactly.

    Args:
        model: Timestral parameters.
        cache: The context from `encode`.
        group: The new steps of the same rows, in the same order.

    Returns:
        The cache grown by the new patches, its range widened by the observed ones.

    Raises:
        ValueError: If the context would pass the model's limit.
    """
    data = _rows(model, group, 0)
    data = data._replace(roles=cache.roles, sign=cache.sign, real=cache.real)
    return _grow(model, cache, data, data.values, data.observed)


def properties(model: Timestral) -> SeriesProperties:
    """Describes Timestral from its parameters and settings.

    Args:
        model: Timestral parameters, real or abstract.

    Returns:
        The model's properties.
    """
    return SeriesProperties(
        patch=model.patch,
        output_patch=model.patch,
        context_limit=model.context_limit,
        pass_limit=model.block_size * model.patch,
        quantiles=model.quantiles,
        reusable=True,
    )


timestral_functions = SeriesFunctions(
    forecast=forecast,
    encode=encode,
    decode=decode,
    extend=extend,
    properties=properties,
)
