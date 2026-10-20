from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from generalforecast.nn.fourier import FourierCellParameters, fourier_cells
from generalforecast.nn.induced import (
    InducedAttentionParameters,
    compress,
    compress_streamed,
    expand,
)
from generalforecast.nn.linear import LinearParameters, linear
from generalforecast.nn.target import TargetEncoder, embed_targets


class FeatureGrouping(NamedTuple):
    """How features are grouped into cells, see `group_features`.

    Attributes:
        mode: `same` or `none`.
        size: Features per group.
    """

    mode: str = "none"
    size: int = 1


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ColumnEmbeddingParameters:
    """Cell embedding, induced attention blocks and target encoder (column stage).

    Grouped cells are embedded by a linear projection or from Fourier features.
    `grouping` is static: it is part of the compiled program, not an array.
    """

    input: LinearParameters | FourierCellParameters
    blocks: tuple[InducedAttentionParameters, ...]
    target_encoder: TargetEncoder
    grouping: FeatureGrouping = field(
        default=FeatureGrouping(), metadata={"static": True}
    )


class ColumnSummary(NamedTuple):
    """Inducing points of each column block, compressed from the context rows."""

    inducing: tuple[Float[Array, " batch_features inducing model"], ...]


def group_features(
    table: Float[Array, " batch rows features *channels"],
    grouping: FeatureGrouping,
    count: Int[Array, " batch"] | None = None,
) -> Float[Array, " batch rows groups width"]:
    """Groups features the way the checkpoint was trained.

    `same` makes group i a circular shift by 2**i; size 1 or `none` keeps scalar
    groups. A table with channels after the features, such as missing-value
    flags, groups each channel, and the channels' groups follow one another.

    Args:
        table: Feature table, optionally with a channel axis.
        grouping: Grouping mode and size.
        count: Leading columns each shift cycles through, per batch row, as if
            the rest were dropped; None for all.

    Returns:
        Grouped features.
    """
    mode, group_size = grouping
    channels = table if table.ndim == 4 else table[..., None]
    if group_size == 1 or mode == "none":
        shifted = [channels]
    elif mode == "same":
        features = channels.shape[2]
        width = jnp.full(channels.shape[:1], features) if count is None else count
        position = jnp.arange(features)
        shifted = [
            jnp.take_along_axis(
                channels,
                ((position + 2**shift) % width[:, None])[:, None, :, None],
                axis=2,
            )
            for shift in range(group_size)
        ]
    else:
        raise ValueError(f"unknown feature group mode: {mode!r}")
    grouped = jnp.stack(shifted, axis=-1)
    return grouped.reshape(*grouped.shape[:3], -1)


def _cells(
    parameters: ColumnEmbeddingParameters,
    table: Float[Array, " batch rows features"],
    targets: Array | None,
    count: Int[Array, " batch"] | None,
) -> Float[Array, " batch_features rows model"]:
    """Projects grouped features to cells, columns folded into the batch axis."""
    grouped = group_features(table, parameters.grouping, count)
    grouped = jnp.transpose(grouped, (0, 2, 1, 3))
    if isinstance(parameters.input, FourierCellParameters):
        hidden = fourier_cells(parameters.input, grouped)
    else:
        hidden = linear(parameters.input, grouped)
    if targets is not None:
        vectors = embed_targets(parameters.target_encoder, targets)
        labeled = targets.shape[-1]
        hidden = hidden.at[:, :, :labeled, :].add(vectors[:, None, :, :])
    return hidden.reshape(-1, *hidden.shape[2:])


def _unfold(
    values: Float[Array, " batch_features rows model"], batch_size: int
) -> Float[Array, " batch rows features model"]:
    """Moves columns out of the batch axis, the inverse of the fold in `_cells`."""
    values = values.reshape(batch_size, -1, *values.shape[1:])
    return jnp.transpose(values, (0, 2, 1, 3))


def fit_columns(
    parameters: ColumnEmbeddingParameters,
    context: Float[Array, " batch context features *channels"],
    targets: Array,
    length: Int[Array, " batch"] | None = None,
    count: Int[Array, " batch"] | None = None,
) -> tuple[ColumnSummary, Float[Array, " batch rows features model"]]:
    """Summarizes the context columns and embeds the context cells in one pass.

    Expanding the context through each block while compressing it gives the same
    cells `embed_cells` would compute from the finished summary. A context of
    more cells than the kernel's `summary_chunk` is summarized a chunk of rows
    at a time instead, see `summarize_columns`, and its cells embed after.

    Args:
        parameters: Column stage parameters.
        context: Prepared context rows.
        targets: Class labels or scaled targets of the context rows.
        length: Real context rows per batch row, or None for all rows.
        count: Columns kept per batch row, which come first; None for all.

    Returns:
        The column summary and the context cell embeddings.
    """
    if _step_rows(parameters, context) < context.shape[1]:
        summary = summarize_columns(parameters, context, targets, length, count)
        return summary, embed_cells(parameters, summary, context, targets, count)
    hidden = _cells(parameters, context, targets, count)
    if length is not None:
        # Columns are folded into the batch axis, so each column needs its length.
        length = jnp.repeat(length, hidden.shape[0] // context.shape[0])
    states: list[Array] = []
    for block in parameters.blocks:
        state = compress(block, hidden, length)
        states.append(state)
        hidden = expand(block, hidden, state)
    return ColumnSummary(inducing=tuple(states)), _unfold(hidden, context.shape[0])


def summarize_columns(
    parameters: ColumnEmbeddingParameters,
    context: Float[Array, " batch context features"],
    targets: Array,
    length: Int[Array, " batch"] | None = None,
    count: Int[Array, " batch"] | None = None,
) -> ColumnSummary:
    """Compresses each feature column of the context into each block's inducing points.

    A context of more cells than the kernel's `summary_chunk` goes a chunk of
    rows at a time, block by block: each block's inducing points read every
    row's cells through the blocks before it, made again for each chunk, by a
    running softmax. Memory then does not grow with the rows.

    Args:
        parameters: Column stage parameters.
        context: Prepared context rows.
        targets: Class labels or scaled targets of the context rows.
        length: Real context rows per batch row, or None for all rows.
        count: Columns kept per batch row, which come first; None for all.

    Returns:
        The column summary.
    """
    rows, size = context.shape[1], _step_rows(parameters, context)
    if size >= rows:
        return fit_columns(parameters, context, targets, length, count)[0]
    if length is not None:
        # Columns are folded into the batch axis, each batch row's in turn.
        length = jnp.repeat(length, context.shape[2])
    states: list[Array] = []
    for block in parameters.blocks:

        def cells(start: Array, size: int, done: tuple = tuple(states)) -> Array:
            hidden = _cells(
                parameters,
                jax.lax.dynamic_slice_in_dim(context, start, size, axis=1),
                jax.lax.dynamic_slice_in_dim(targets, start, size, axis=1),
                count,
            )
            for earlier, state in zip(parameters.blocks, done, strict=False):
                hidden = expand(earlier, hidden, state)
            return hidden

        states.append(compress_streamed(block, cells, rows, size, length))
    return ColumnSummary(inducing=tuple(states))


def _step_rows(
    parameters: ColumnEmbeddingParameters,
    context: Float[Array, " batch context features *channels"],
) -> int:
    """Rows whose cells fit the kernel's `summary_chunk`, or every row without one."""
    chunk = parameters.blocks[0].compress.attention.kernel.summary_chunk
    if chunk is None:
        return context.shape[1]
    return max(1, chunk // (context.shape[0] * context.shape[2]))


def embed_cells(
    parameters: ColumnEmbeddingParameters,
    summary: ColumnSummary,
    rows: Float[Array, " batch rows features *channels"],
    targets: Array | None = None,
    count: Int[Array, " batch"] | None = None,
) -> Float[Array, " batch rows features model"]:
    """Embeds each cell of `rows` against the column summary.

    Rows do not update the summary, so each row's cells depend only on that row.

    Args:
        parameters: Column stage parameters.
        summary: Summary from `summarize_columns`.
        rows: Prepared rows.
        targets: Targets of the leading rows, or None for queries.
        count: Columns kept per batch row, which come first; None for all.

    Returns:
        Cell embeddings.
    """
    hidden = _cells(parameters, rows, targets, count)
    for block, state in zip(parameters.blocks, summary.inducing, strict=True):
        hidden = expand(block, hidden, state)
    return _unfold(hidden, rows.shape[0])
