# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.column_embedding import FeatureGrouping, group_columns
from generalforecast.models.prepare import precise_mean
from generalforecast.nn.fourier import fourier_features
from generalforecast.nn.linear import LinearParameters


class CellEmbeddingParameters(NamedTuple):
    """Fourier features of grouped cells, plus a missing-value term.

    Numerical and categorical group members have their own frequencies and
    projections.

    Attributes:
        frequencies: One row of frequencies per group member, for numerical members.
        linear: Projection of a numerical member's sines and cosines to the cell width.
        missing: Projection of the members' missing-value indicators, no bias.
        categorical_frequencies: Frequencies for categorical members.
        categorical_linear: Projection for categorical members.
    """

    frequencies: Float[Array, " group frequencies"]
    linear: LinearParameters
    missing: LinearParameters
    categorical_frequencies: Float[Array, " group frequencies"]
    categorical_linear: LinearParameters


def impute_means(
    features: Float[Array, " rows features"], valid: Bool[Array, " rows"]
) -> Float[Array, " features"]:
    """Mean of each column over valid rows, ignoring NaN; 0 for an empty column.

    Args:
        features: Prepared context rows, with NaN for missing values.
        valid: Which rows are real context rows rather than padding.

    Returns:
        The imputation value of each column.
    """
    return precise_mean(features, valid[:, None] & ~jnp.isnan(features))


def _project(
    fourier: Float[Array, " rows features group fourier"], linear: LinearParameters
) -> Float[Array, " rows features model"]:
    """Sums the projections of a cell's group members."""
    embedded = jnp.einsum("rcgf,df->rcd", fourier, linear.weight)
    if linear.bias is None:
        return embedded
    return embedded + fourier.shape[2] * linear.bias


def embed_cells(
    parameters: CellEmbeddingParameters,
    features: Float[Array, " rows features"],
    means: Float[Array, " features"],
    dtype: jnp.dtype,
    categorical: Bool[Array, " features"] | None = None,
    count: Int[Array, ""] | None = None,
) -> Float[Array, " rows features model"]:
    """Embeds each cell from its group's imputed values and missing indicators.

    Fourier phases are computed in float32; sines and cosines are cast to `dtype`.

    Args:
        parameters: Cell embedding parameters.
        features: Prepared rows, with NaN for missing values.
        means: Imputation values from `impute_means`.
        dtype: Dtype of the embeddings.
        categorical: Which columns are categorical, or None if none are.
        count: Leading columns kept, which groups cycle through; None for all.

    Returns:
        One embedding per cell.
    """
    missing = jnp.isnan(features)
    filled = jnp.where(missing, means, features).astype(jnp.float32)
    grouping = FeatureGrouping("same", parameters.frequencies.shape[0], offset=-1)
    index = group_columns(features.shape[-1], grouping, count)

    def fourier(frequencies: Array) -> Array:
        return fourier_features(filled[:, index], frequencies, dtype)

    if categorical is None:
        embedded = _project(fourier(parameters.frequencies), parameters.linear)
    else:
        # Each member goes through its own type's frequencies and projection.
        members = categorical[index]
        numerical, categorical_linear = parameters.linear, parameters.categorical_linear
        waves = fourier(
            jnp.where(
                members[..., None],
                parameters.categorical_frequencies,
                parameters.frequencies,
            )
        )
        chosen = members[None, :, :, None]
        embedded = jnp.einsum(
            "rcgf,df->rcd", jnp.where(chosen, 0, waves), numerical.weight
        ) + jnp.einsum(
            "rcgf,df->rcd", jnp.where(chosen, waves, 0), categorical_linear.weight
        )
        # Each member takes its own projection's bias, if that has one.
        zero = jnp.zeros(embedded.shape[-1], embedded.dtype)
        numerical_bias = zero if numerical.bias is None else numerical.bias
        categorical_bias = (
            zero if categorical_linear.bias is None else categorical_linear.bias
        )
        biases = jnp.where(members[..., None], categorical_bias, numerical_bias)
        embedded = embedded + biases.sum(axis=1).astype(dtype)
    indicators = missing[:, index].astype(dtype)
    return embedded + jnp.einsum("rcg,dg->rcd", indicators, parameters.missing.weight)
