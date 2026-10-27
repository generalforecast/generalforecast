# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.kumo.cell_embedding import (
    CellEmbeddingParameters,
    embed_cells,
)
from generalforecast.nn.attention import (
    AttentionBlockParameters,
    attention_block,
    block_kv,
    in_chunks,
)
from generalforecast.nn.induced import (
    InducedAttentionParameters,
    compress,
    compress_layers,
    expand,
)
from generalforecast.nn.norm import RMSNormParameters, rms_norm
from generalforecast.nn.target import TargetEncoder, embed_targets


class RowEmbeddingParameters(NamedTuple):
    """Kumo Tabular's row encoder: cell embedding, then column and row blocks in turn.

    Attributes:
        cells: Cell embedding.
        target_encoder: Label embedding added to every cell of a context row.
        readout: Readout tokens prepended to each row; their outputs form the row.
        columns: One induced attention per layer, across rows within each column.
        rows: One attention block per layer, across a row's readout and cell tokens.
        output_norm: Norm of the readout tokens.
    """

    cells: CellEmbeddingParameters
    target_encoder: TargetEncoder
    readout: Float[Array, " readout model"]
    columns: tuple[InducedAttentionParameters, ...]
    rows: tuple[AttentionBlockParameters, ...]
    output_norm: RMSNormParameters


ColumnStates = tuple[Float[Array, " features inducing model"], ...]


def embed_rows(
    parameters: RowEmbeddingParameters,
    features: Float[Array, " rows features"],
    means: Float[Array, " features"],
    targets: Array | None,
    states: ColumnStates | None = None,
    length: Int[Array, ""] | None = None,
    categorical: Bool[Array, " features"] | None = None,
    count: Int[Array, ""] | None = None,
) -> tuple[ColumnStates, Float[Array, " rows readout_model"]]:
    """Embeds rows, fitting each layer's column states first if none are given.

    Each column block compresses the context cells of a column into inducing
    points, so the states only exist layer by layer as the context flows through.
    Given the states, every row's representation depends only on that row.

    Args:
        parameters: Row encoder parameters.
        features: Prepared rows, with NaN for missing values.
        means: Imputation values for missing cells.
        targets: Class labels or scaled targets of the leading rows, which are the
            context rows, or None.
        states: Column states from fitting, or None to fit them on the context rows.
        length: Real context rows when fitting with padding, or None for all.
        categorical: Which columns are categorical, or None if none are.
        count: Leading columns kept; rows attend only to their cells, as if
            the rest were dropped. None for all.

    Returns:
        The column states and one representation per row.
    """
    dtype = parameters.readout.dtype
    if states is None and targets is not None:
        labeled = targets.shape[0]
        chunk = parameters.columns[0].compress.attention.kernel.summary_chunk
        if chunk is not None and labeled * features.shape[1] > chunk:
            states = _fit_streamed(
                parameters,
                features[:labeled],
                means,
                targets,
                max(1, chunk // features.shape[1]),
                length,
                categorical,
                count,
            )
    # Cells embed row by row, and their Fourier features are many times their
    # embeddings, so rows go a chunk at a time.
    chunk = parameters.rows[0].attention.kernel.block_chunk
    per_step = None if chunk is None else max(1, chunk // features.shape[1])
    cells = in_chunks(
        lambda rows: embed_cells(
            parameters.cells, rows, means, dtype, categorical, count
        ),
        (features,),
        per_step,
        axis=0,
    )
    if targets is not None:
        labels = embed_targets(parameters.target_encoder, targets)
        cells = cells.at[: targets.shape[0]].add(labels[:, None, :])

    num_rows, readout = features.shape[0], parameters.readout.shape[0]
    tokens = jnp.broadcast_to(parameters.readout, (num_rows, *parameters.readout.shape))
    keys = None if count is None else jnp.full(num_rows, readout + count)
    fitted: list[Array] = []
    for layer, (column, row) in enumerate(
        zip(parameters.columns, parameters.rows, strict=True)
    ):
        if states is None:
            if targets is None:
                raise ValueError("fitting column states needs the context targets")
            context = jnp.swapaxes(cells[: targets.shape[0]], 0, 1)
            lengths = None if length is None else jnp.full(context.shape[0], length)
            state = compress(column, context, lengths)
        else:
            state = states[layer]
        fitted.append(state)
        last = layer == len(parameters.rows) - 1

        def update(
            tokens: Array,
            cells: Array,
            keys: Array | None,
            column: InducedAttentionParameters = column,
            state: Array = state,
            row: AttentionBlockParameters = row,
            last: bool = last,
        ) -> Array:
            return _layer(column, state, row, tokens, cells, keys, last)

        # Given the column states, rows update independently, so they go a
        # chunk at a time and only a layer's input and output cells are whole.
        chunk = row.attention.kernel.block_chunk
        per_step = (
            None if chunk is None else max(1, chunk // (readout + cells.shape[1]))
        )
        sequence = in_chunks(update, (tokens, cells, keys), per_step, axis=0)
        if last:
            tokens = sequence
        else:
            tokens, cells = sequence[:, :readout], sequence[:, readout:]
    output = rms_norm(parameters.output_norm, tokens)
    return tuple(fitted), output.reshape(num_rows, -1)


def _layer(
    column: InducedAttentionParameters,
    state: Float[Array, " features inducing model"],
    row: AttentionBlockParameters,
    tokens: Float[Array, " rows readout model"],
    cells: Float[Array, " rows features model"],
    keys: Int[Array, " rows"] | None,
    last: bool,
) -> Float[Array, " rows tokens model"]:
    """One layer: cells read their column's state, then each row attends within itself.

    A row's keys are its readout tokens and its first `keys` tokens in all, or
    every token if None. The last layer updates only the readout tokens.
    """
    cells = jnp.swapaxes(expand(column, jnp.swapaxes(cells, 0, 1), state), 0, 1)
    sequence = jnp.concatenate([tokens, cells], axis=1)
    cache = block_kv(row, sequence, keys)
    if last:
        return attention_block(row, tokens, cache)
    return attention_block(row, sequence, cache)


def _fit_streamed(
    parameters: RowEmbeddingParameters,
    features: Float[Array, " rows features"],
    means: Float[Array, " features"],
    targets: Array,
    size: int,
    length: Int[Array, ""] | None,
    categorical: Bool[Array, " features"] | None,
    count: Int[Array, ""] | None,
) -> ColumnStates:
    """Each layer's column states, from at most `size` rows at a time.

    A chunk's cells run through the layers before each one, row blocks
    included; see `compress_layers`. Memory then does not grow with the
    rows, for compute that grows with the square of the layers.
    """
    dtype = parameters.readout.dtype
    readout = parameters.readout.shape[0]
    lengths = None if length is None else jnp.full(features.shape[1], length)

    def cells(start: Array, size: int, earlier: tuple[Array, ...]) -> Array:
        rows = jax.lax.dynamic_slice_in_dim(features, start, size)
        labels = jax.lax.dynamic_slice_in_dim(targets, start, size)
        hidden = embed_cells(parameters.cells, rows, means, dtype, categorical, count)
        hidden = hidden + embed_targets(parameters.target_encoder, labels)[:, None]
        tokens = jnp.broadcast_to(parameters.readout, (size, *parameters.readout.shape))
        keys = None if count is None else jnp.full(size, readout + count)
        for column, state, row in zip(
            parameters.columns, earlier, parameters.rows, strict=False
        ):
            sequence = _layer(column, state, row, tokens, hidden, keys, False)
            tokens, hidden = sequence[:, :readout], sequence[:, readout:]
        return jnp.swapaxes(hidden, 0, 1)

    return compress_layers(parameters.columns, cells, features.shape[0], size, lengths)
