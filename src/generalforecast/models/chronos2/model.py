# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-2: an encoder over patches with attention along time and across rows.

Every row of a group is standardized on its own and turned into patches of
`[time encoding, values, observed]`, followed by a register token and one
patch per output patch, which carry the horizon's time encodings and any
known future values. Each layer attends along time within a row, then across
the group's rows at each position, then applies a feed-forward. The output
patches decode to quantiles, which the row's scale maps back.
"""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float

from generalforecast.models.series import (
    SeriesFunctions,
    SeriesGroup,
    SeriesProperties,
    padded_horizon,
)
from generalforecast.nn.attention import AttentionParameters, attend, project_kv
from generalforecast.nn.linear import (
    LinearParameters,
    ResidualBlockParameters,
    linear,
    residual_block,
)
from generalforecast.nn.norm import RMSNormParameters, rms_norm

# The levels the pipeline continues long horizons along, each as its own path.
UNROLLED = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class FeedForward(NamedTuple):
    """A pre-norm feed-forward layer with ReLU, without biases."""

    norm: RMSNormParameters
    input: LinearParameters
    output: LinearParameters


class Chronos2Layer(NamedTuple):
    """Pre-norm attention along time, then across rows, then the feed-forward."""

    time_norm: RMSNormParameters
    time: AttentionParameters
    group_norm: RMSNormParameters
    group: AttentionParameters
    feed_forward: FeedForward


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Chronos2:
    """Chronos-2's embeddings, layers and head, with its static settings.

    Attributes:
        patch_embedding: Embeds a patch's time encodings, values and mask.
        register: The register token's embedding.
        layers: The encoder layers.
        final_norm: Norm of the encoder's output.
        output_embedding: Maps an output patch to its quantiles' values.
        patch: Steps per input and output patch.
        context_limit: Most context steps read.
        max_output_patches: Most output patches one pass predicts.
        quantiles: The levels of the output quantiles.
        time_scale: What step offsets are divided by in the time encoding.
    """

    patch_embedding: ResidualBlockParameters
    register: Float[Array, " model"]
    layers: tuple[Chronos2Layer, ...]
    final_norm: RMSNormParameters
    output_embedding: ResidualBlockParameters
    patch: int = field(metadata={"static": True})
    context_limit: int = field(metadata={"static": True})
    max_output_patches: int = field(metadata={"static": True})
    quantiles: tuple[float, ...] = field(metadata={"static": True})
    time_scale: float = field(metadata={"static": True})


class Scale(NamedTuple):
    """A row's location and scale, `(rows, 1)` each, in float32."""

    loc: Array
    scale: Array


def _dtype(model: Chronos2) -> jnp.dtype:
    """The dtype the network computes in, set by its parameters."""
    return model.register.dtype


def _feed_forward(parameters: FeedForward, values: Array) -> Array:
    """A pre-norm feed-forward with ReLU, added to its input."""
    hidden = jnn.relu(linear(parameters.input, rms_norm(parameters.norm, values)))
    return values + linear(parameters.output, hidden)


def _standardize(
    values: Float[Array, " rows time"], observed: Bool[Array, " rows time"]
) -> tuple[Float[Array, " rows time"], Scale]:
    """Standardizes each row over its observed steps, then takes arcsinh.

    A row without observed steps keeps location 0 and scale 1, and a constant
    row a scale of 1e-5, as upstream's instance norm does. Unobserved steps
    come out as 0.

    Args:
        values: Row values, float32.
        observed: Which values are observed.

    Returns:
        The standardized values and each row's location and scale.
    """
    count = observed.sum(axis=-1, keepdims=True)
    present = jnp.where(observed, values, 0.0)
    loc = present.sum(axis=-1, keepdims=True) / jnp.maximum(count, 1)
    squares = jnp.where(observed, jnp.square(values - loc), 0.0)
    scale = jnp.sqrt(squares.sum(axis=-1, keepdims=True) / jnp.maximum(count, 1))
    scale = jnp.where(count > 0, scale, 1.0)
    scale = jnp.where(scale == 0, 1e-5, scale)
    standardized = jnp.where(observed, jnp.arcsinh((values - loc) / scale), 0.0)
    return standardized, Scale(loc, scale)


def _patches(
    model: Chronos2,
    standardized: Float[Array, " rows time"],
    observed: Bool[Array, " rows time"],
    start: int,
) -> Float[Array, " rows patches model"]:
    """Embeds steps `start, start + 1, ...` relative to the origin as patches."""
    rows, steps = standardized.shape
    patches = steps // model.patch
    offsets = jnp.arange(start, start + steps, dtype=jnp.float32) / model.time_scale
    encoding = jnp.broadcast_to(
        offsets.reshape(1, patches, model.patch), (rows, patches, model.patch)
    )
    features = jnp.concatenate(
        [
            encoding,
            standardized.reshape(rows, patches, model.patch),
            observed.reshape(rows, patches, model.patch).astype(jnp.float32),
        ],
        axis=-1,
    )
    return residual_block(model.patch_embedding, features.astype(_dtype(model)))


def _layer(
    layer: Chronos2Layer,
    tokens: Float[Array, " rows positions model"],
    time_mask: Bool[Array, " rows positions positions"],
    group_mask: Bool[Array, " positions rows rows"],
) -> Float[Array, " rows positions model"]:
    """One layer: attention along time, across rows, then the feed-forward."""
    normed = rms_norm(layer.time_norm, tokens)
    cache = project_kv(layer.time, normed)._replace(mask=time_mask)
    tokens = tokens + attend(layer.time, normed, cache)
    flipped = tokens.swapaxes(0, 1)
    normed = rms_norm(layer.group_norm, flipped)
    cache = project_kv(layer.group, normed)._replace(mask=group_mask)
    flipped = flipped + attend(layer.group, normed, cache)
    return _feed_forward(layer.feed_forward, flipped.swapaxes(0, 1))


def _pass(
    model: Chronos2,
    values: Float[Array, " rows time"],
    observed: Bool[Array, " rows time"],
    future: Float[Array, " rows steps"],
    future_observed: Bool[Array, " rows steps"],
    real: Bool[Array, " rows"],
) -> Float[Array, " rows steps quantiles"]:
    """One pass of the network over a group, predicting the future's steps.

    Args:
        model: Chronos-2 parameters.
        values: Context values, a multiple of the patch long, 0 where unobserved.
        observed: Which context values are observed, False on padding rows.
        future: Known future values, a multiple of the patch long, 0 elsewhere.
        future_observed: Which future values are known, False on padding rows.
        real: Which rows are real rather than padding.

    Returns:
        Quantiles for every row and future step, on the original scale.
    """
    rows, steps = future.shape
    standardized, scale = _standardize(values, observed)
    context = _patches(model, standardized, observed, -values.shape[-1])
    known = jnp.where(
        future_observed, jnp.arcsinh((future - scale.loc) / scale.scale), 0.0
    )
    horizon = _patches(model, known, future_observed, 0)
    register = jnp.broadcast_to(model.register, (rows, 1, model.register.shape[0]))
    tokens = jnp.concatenate([context, register, horizon], axis=1)
    valid = jnp.concatenate(
        [
            observed.reshape(rows, -1, model.patch).any(axis=-1),
            jnp.ones((rows, 1 + steps // model.patch), bool),
        ],
        axis=1,
    )
    positions = valid.shape[1]
    time_mask = jnp.broadcast_to(valid[:, None, :], (rows, positions, positions))
    keys = (valid & real[:, None]).T
    group_mask = jnp.broadcast_to(keys[:, None, :], (positions, rows, rows))
    for layer in model.layers:
        tokens = _layer(layer, tokens, time_mask, group_mask)
    tokens = rms_norm(model.final_norm, tokens)
    outputs = residual_block(
        model.output_embedding, tokens[:, -(steps // model.patch) :]
    )
    quantiles = len(model.quantiles)
    outputs = outputs.reshape(rows, -1, quantiles, model.patch).astype(jnp.float32)
    outputs = outputs.transpose(0, 1, 3, 2).reshape(rows, steps, quantiles)
    return jnp.sinh(outputs) * scale.scale[..., None] + scale.loc[..., None]


def _mass(levels: tuple[float, ...]) -> Array:
    """Probability mass per level, half the gap to its neighbors, normalized."""
    bounds = jnp.asarray([0.0, *levels, 1.0], jnp.float32)
    mass = (bounds[2:] - bounds[:-2]) / 2
    return mass / mass.sum()


def _interpolate(
    queries: Array,
    levels: Float[Array, " ... samples"],
    values: Float[Array, " ... samples"],
) -> Float[Array, " ... queries"]:
    """Linear interpolation of sorted `values` at `queries`, constant past the ends."""
    ones = jnp.ones_like(levels[..., :1])
    levels = jnp.concatenate([0 * ones, levels, ones], axis=-1)
    values = jnp.concatenate([values[..., :1], values, values[..., -1:]], axis=-1)
    upper = (levels[..., None, :] <= queries[:, None]).sum(axis=-1)
    upper = jnp.minimum(upper, levels.shape[-1] - 1)
    lower = upper - 1
    gap = jnp.take_along_axis(levels, upper, -1) - jnp.take_along_axis(
        levels, lower, -1
    )
    weight = jnp.where(
        gap > 0, (queries - jnp.take_along_axis(levels, lower, -1)) / gap, 0.0
    )
    low, high = (jnp.take_along_axis(values, at, -1) for at in (lower, upper))
    return low + weight * (high - low)


def _weighted_quantile(
    levels: tuple[float, ...], weights: Array, samples: Float[Array, " ... samples"]
) -> Float[Array, " ... levels"]:
    """Quantiles of weighted samples, from their empirical distribution."""
    order = jnp.argsort(samples, axis=-1)
    sorted_samples = jnp.take_along_axis(samples, order, axis=-1)
    cumulative = jnp.clip(jnp.cumsum(weights[order], axis=-1), 0.0, 1.0)
    return _interpolate(jnp.asarray(levels, jnp.float32), cumulative, sorted_samples)


def _steps(model: Chronos2, remaining: int) -> int:
    """The steps the next pass predicts: whole patches, at most a pass."""
    return (
        min(math.ceil(remaining / model.patch), model.max_output_patches) * model.patch
    )


def forecast(
    model: Chronos2, group: SeriesGroup, horizon: int
) -> Float[Array, " rows horizon quantiles"]:
    """Forecasts `horizon` steps for every row of a group.

    A horizon past one pass continues as upstream's pipeline does: each of the
    `UNROLLED` levels of the last pass extends the context along its own path,
    the paths predict the next steps as separate groups, and their quantiles
    merge as a weighted distribution.

    Args:
        model: Chronos-2 parameters.
        group: The group; known future values ride in `future`, observed
            where `future_observed` says, or where finite without it.
        horizon: Steps to forecast.

    Returns:
        Quantiles at the model's levels for every row and step.
    """
    rows, steps = group.values.shape
    if steps % model.patch:
        raise ValueError(
            f"the context's {steps} steps are not whole patches of {model.patch}"
        )
    real = (
        jnp.ones(rows, bool)
        if group.num_rows is None
        else jnp.arange(rows) < group.num_rows
    )
    observed = (group.observed & real[:, None])[:, -model.context_limit :]
    values = jnp.where(observed, group.values[:, -model.context_limit :], 0.0)
    values = values.astype(jnp.float32)
    if group.future is None:
        future = jnp.zeros((rows, horizon), jnp.float32)
        future_observed = jnp.zeros((rows, horizon), bool)
    else:
        # Known where observed, or where finite if the group does not say.
        future_observed = (
            jnp.isfinite(group.future)
            if group.future_observed is None
            else group.future_observed
        )
        future_observed = future_observed & real[:, None]
        future = jnp.where(future_observed, group.future, 0.0).astype(jnp.float32)

    first = _steps(model, horizon)
    prediction = _pass(
        model,
        values,
        observed,
        padded_horizon(future, first, 0.0),
        padded_horizon(future_observed, first, False),
        real,
    )
    predictions = [prediction]
    remaining = horizon - first
    if remaining > 0:
        paths = len(UNROLLED)
        indices = jnp.asarray([model.quantiles.index(level) for level in UNROLLED])
        weights = jnp.outer(_mass(UNROLLED), _mass(model.quantiles)).reshape(-1)
        passes = jax.vmap(_pass, in_axes=(None, 0, 0, 0, 0, None))

        def repeated(array: Array) -> Array:
            return jnp.broadcast_to(array, (paths, *array.shape))

        values, observed = repeated(values), repeated(observed)
        future, future_observed = repeated(future), repeated(future_observed)
        while remaining > 0:
            done = prediction.shape[1]
            # Each path appends its level's values, known future values replacing them.
            continued = prediction[..., indices].transpose(2, 0, 1)
            known = padded_horizon(future_observed, done, False)
            continued = jnp.where(known, padded_horizon(future, done, 0.0), continued)
            values = jnp.concatenate([values, continued], axis=-1)
            observed = jnp.concatenate([observed, jnp.ones_like(known)], axis=-1)
            values = values[..., -model.context_limit :]
            observed = observed[..., -model.context_limit :]
            future, future_observed = future[..., done:], future_observed[..., done:]
            steps = _steps(model, remaining)
            sampled = passes(
                model,
                values,
                observed,
                padded_horizon(future, steps, 0.0),
                padded_horizon(future_observed, steps, False),
                real,
            )
            samples = sampled.transpose(1, 2, 0, 3).reshape(rows, steps, -1)
            prediction = _weighted_quantile(model.quantiles, weights, samples)
            predictions.append(prediction)
            remaining -= steps
    return jnp.concatenate(predictions, axis=1)[:, :horizon]


def properties(model: Chronos2) -> SeriesProperties:
    """Describes Chronos-2 from its parameters and settings.

    Its attention across rows reads the horizon's tokens in every layer, so
    no context encoding can be reused across horizons.

    Args:
        model: Chronos-2 parameters, real or abstract.

    Returns:
        The model's properties.
    """
    return SeriesProperties(
        patch=model.patch,
        output_patch=model.patch,
        context_limit=model.context_limit,
        pass_limit=model.max_output_patches * model.patch,
        quantiles=model.quantiles,
        reusable=False,
    )


chronos2_functions = SeriesFunctions(
    forecast=forecast,
    encode=None,
    decode=None,
    extend=None,
    properties=properties,
)
