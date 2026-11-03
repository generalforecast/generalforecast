# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TabFM's classifier: Fourier cells, two column and row stages, then in-context learning.

TabFM follows TabICL's three stages with two differences in shape: the
column stage (induced attention over each column's context rows) and the row
stage (attention across one row's cells, with CLS tokens) run twice, the
second pair over the first's CLS tokens and cells together; and every
attention block is a sandwich, its attention and its SwiGLU each reading
RMS-normalized inputs and normalizing their outputs before adding them
back. The context's labels enter the cells of the context rows, so each
column's inducing points read them, and again the in-context stage's rows.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.column_embedding import (
    FeatureGrouping,
    group_columns,
    group_features,
)
from generalforecast.models.common import TableState, check_class_count, compute_dtype
from generalforecast.models.contract import (
    Averaging,
    TabularFunctions,
    TabularProperties,
)
from generalforecast.models.icl import ICLCache
from generalforecast.models.prepare import batch_length
from generalforecast.models.tabicl.model import class_probabilities
from generalforecast.models.tabicl.prepare import (
    apply_numeric_snapshot,
    fit_numeric_snapshot,
    kept_columns,
)
from generalforecast.nn.attention import (
    AttentionBlockParameters,
    AttentionCache,
    attention_block,
    block_kv,
    in_chunks,
)
from generalforecast.nn.induced import InducedAttentionParameters, compress, expand
from generalforecast.nn.linear import LinearParameters, MLPParameters, linear
from generalforecast.nn.norm import RMSNormParameters, rms_norm
from generalforecast.nn.target import ClassEncoder, embed_targets

# Each cell's group reads its own column, the next, then the one three on,
# cycling through the kept columns.
_GROUP_OFFSET = -1


class FourierCells(NamedTuple):
    """Frequencies and projections of numerical and categorical cells.

    A group member's value becomes the sines and cosines of the value times
    its member's frequencies. Each member is projected on its own, by the
    categorical projection where its column is categorical, and the members'
    projections are summed.
    """

    frequencies: Float[Array, " group frequencies"]
    numerical: LinearParameters
    categorical_frequencies: Float[Array, " group frequencies"]
    categorical: LinearParameters


class ColumnStage(NamedTuple):
    """Induced attention blocks over each column's rows, then a projection and its norm."""

    blocks: tuple[InducedAttentionParameters, ...]
    output: LinearParameters
    output_norm: RMSNormParameters


class RowStage(NamedTuple):
    """Attention blocks across one row's tokens, then a norm."""

    blocks: tuple[AttentionBlockParameters, ...]
    output_norm: RMSNormParameters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TabFM:
    """TabFM's cells, column and row stages, CLS tokens and in-context stage.

    `cell_targets` holds each class slot's vector for the context's cells.
    The decoder's MLP uses the tanh approximation of GELU. `group_size` and
    `temperature` are static.
    """

    cells: FourierCells
    cell_targets: Float[Array, " classes model"]
    columns: tuple[ColumnStage, ColumnStage]
    cls_tokens: Float[Array, " cls model"]
    rows: tuple[RowStage, RowStage]
    icl_blocks: tuple[AttentionBlockParameters, ...]
    icl_targets: ClassEncoder
    icl_norm: RMSNormParameters
    decoder: MLPParameters
    group_size: int = field(metadata={"static": True})
    temperature: float = field(default=0.9, metadata={"static": True})


class TabFMColumns(NamedTuple):
    """What the state keeps of the context's columns.

    Attributes:
        inducing: Each column stage's inducing points per block, compressed
            from the context rows.
        categorical: Which columns are categorical, in the prepared order.
    """

    inducing: tuple[tuple[Array, ...], tuple[Array, ...]]
    categorical: Bool[Array, " features"]


def _grouping(model: TabFM) -> FeatureGrouping:
    """The checkpoint's feature groups."""
    return FeatureGrouping("same", model.group_size, _GROUP_OFFSET)


def _members(
    projection: LinearParameters,
    frequencies: Array,
    values: Float[Array, " batch rows features group"],
) -> Float[Array, " batch rows features group model"]:
    """Each group member's projected Fourier features, with phases in float32."""
    phase = values.astype(jnp.float32)[..., None] * frequencies.astype(jnp.float32)
    features = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)
    return linear(projection, features.astype(values.dtype))


def _cells(
    model: TabFM,
    values: Float[Array, " batch rows features"],
    targets: Int[Array, " batch labeled"] | None,
    categorical: Bool[Array, " batch features"],
    count: Int[Array, " batch"] | None,
) -> Float[Array, " batch rows features model"]:
    """Embeds each cell from its group, adding the class vectors to the labeled rows' cells."""
    grouped = group_features(values, _grouping(model), count)
    cells = model.cells
    hidden = _members(cells.numerical, cells.frequencies, grouped)
    index = group_columns(values.shape[2], _grouping(model), count)
    index = jnp.broadcast_to(index, (values.shape[0], *index.shape[-2:]))
    member = jax.vmap(lambda mask, index: mask[index])(categorical, index)
    coded = _members(cells.categorical, cells.categorical_frequencies, grouped)
    hidden = jnp.where(member[:, None, :, :, None], coded, hidden).sum(axis=-2)
    if targets is None:
        return hidden
    slots = model.cell_targets.shape[0]
    vectors = model.cell_targets[jnp.clip(targets, 0, slots - 1)].astype(hidden.dtype)
    return hidden.at[:, : targets.shape[1]].add(vectors[:, :, None, :])


def _fold(values: Float[Array, " batch rows features model"]) -> Array:
    """Columns into the batch axis: one sequence of rows a column."""
    return jnp.transpose(values, (0, 2, 1, 3)).reshape(
        -1, values.shape[1], values.shape[3]
    )


def _unfold(values: Array, batch: int) -> Float[Array, " batch rows features model"]:
    """Columns out of the batch axis, the inverse of `_fold`."""
    return jnp.transpose(values.reshape(batch, -1, *values.shape[1:]), (0, 2, 1, 3))


def _fit_columns(
    stage: ColumnStage,
    cells: Float[Array, " batch rows features model"],
    length: Int[Array, " batch"] | None,
) -> tuple[tuple[Array, ...], Float[Array, " batch rows features model"]]:
    """Compresses each column's context rows into each block's inducing points, embedding them too."""
    hidden = _fold(cells)
    if length is not None:
        # Columns are folded into the batch axis, so each column needs its length.
        length = jnp.repeat(length, cells.shape[2])
    states = []
    for block in stage.blocks:
        state = compress(block, hidden, length)
        states.append(state)
        hidden = expand(block, hidden, state)
    output = rms_norm(stage.output_norm, linear(stage.output, hidden))
    return tuple(states), _unfold(output, cells.shape[0])


def _embed_columns(
    stage: ColumnStage,
    inducing: tuple[Array, ...],
    cells: Float[Array, " batch rows features model"],
) -> Float[Array, " batch rows features model"]:
    """Each cell against its column's inducing points, so a row's cells depend on that row alone."""
    hidden = _fold(cells)
    for block, state in zip(stage.blocks, inducing, strict=True):
        hidden = expand(block, hidden, state)
    output = rms_norm(stage.output_norm, linear(stage.output, hidden))
    return _unfold(output, cells.shape[0])


def _interact(
    stage: RowStage,
    tokens: Float[Array, " batch rows tokens model"],
    length: Int[Array, " batch"] | None,
    keep: int | None,
) -> Float[Array, " batch rows kept model"]:
    """Attention across each row's tokens, the first `keep` tokens out, or all of them if None.

    Tokens past `length` are masked as keys, as if the columns they hold
    were dropped.
    """
    batch, rows, width, model_dim = tokens.shape
    flat = tokens.reshape(batch * rows, width, model_dim)
    lengths = None if length is None else jnp.repeat(length, rows)

    def run(hidden: Array, lengths: Array | None) -> Array:
        *early, last = stage.blocks
        for block in early:
            hidden = attention_block(block, hidden, block_kv(block, hidden, lengths))
        cache = block_kv(last, hidden, lengths)
        queries = hidden if keep is None else hidden[:, :keep]
        return rms_norm(stage.output_norm, attention_block(last, queries, cache))

    # Rows attend only within themselves, so they can go a chunk at a time.
    chunk = stage.blocks[0].attention.kernel.block_chunk
    per_step = None if chunk is None else max(1, chunk // width)
    output = in_chunks(run, (flat, lengths), per_step, axis=0)
    return output.reshape(batch, rows, *output.shape[1:])


def _first_rows(
    model: TabFM, cells: Float[Array, " batch rows features model"], count: Array | None
) -> Float[Array, " batch rows tokens model"]:
    """The first row stage over the CLS tokens and the first column stage's cells."""
    batch, rows = cells.shape[:2]
    cls = jnp.broadcast_to(model.cls_tokens, (batch, rows, *model.cls_tokens.shape))
    tokens = jnp.concatenate([cls.astype(cells.dtype), cells], axis=2)
    return _interact(model.rows[0], tokens, _length(model, count), None)


def _second_rows(
    model: TabFM, tokens: Float[Array, " batch rows tokens model"], count: Array | None
) -> Float[Array, " batch rows icl"]:
    """The second row stage, whose CLS tokens side by side are a row's representation."""
    num_cls = model.cls_tokens.shape[0]
    reps = _interact(model.rows[1], tokens, _length(model, count), num_cls)
    return reps.reshape(*reps.shape[:2], -1)


def _length(model: TabFM, count: Array | None) -> Array | None:
    """Tokens a row attends to across itself: its CLS tokens and kept columns."""
    return None if count is None else model.cls_tokens.shape[0] + count


def _categorical(
    snapshot_order: Int[Array, " features"],
    categorical: Bool[Array, " features"] | None,
) -> Bool[Array, " features"]:
    """Which prepared columns are categorical, in the prepared order."""
    if categorical is None:
        return jnp.zeros(snapshot_order.shape, dtype=bool)
    return categorical[snapshot_order]


def fit(
    model: TabFM,
    features: Float[Array, " rows features"],
    targets: Array,
    num_classes: int,
    num_rows: Int[Array, ""] | int | None = None,
    categorical: Bool[Array, " features"] | None = None,
) -> tuple[TableState, Float[Array, " rows icl"]]:
    """Fits the table state and embeds the context rows in the same pass.

    Preprocessing is TabICL's, as TabFM's package runs it for one estimator
    without a normalizer: mean imputation, standardization clipped at 100
    deviations, soft clipping past 4, and constant columns left out.
    Categorical columns, whose codes are standardized like any column, take
    the categorical Fourier frequencies and projection.

    Args:
        model: TabFM parameters.
        features: Context rows, with NaN for missing values.
        targets: Class labels `0 .. num_classes - 1`.
        num_classes: Number of classes in the task.
        num_rows: Real rows at the start; the rest are padding that does not
            affect the state. None means all rows. May be traced.
        categorical: Which columns hold category codes; None for none.

    Returns:
        The table state and one representation per context row.
    """
    check_class_count(properties(model), num_classes)
    snapshot = fit_numeric_snapshot(features, None, num_rows)
    prepared = apply_numeric_snapshot(snapshot, features).astype(compute_dtype(model))
    count = batch_length(snapshot.columns.count)
    length = batch_length(num_rows)
    coded = _categorical(snapshot.columns.order, categorical)
    cells = _cells(model, prepared[None], targets[None], coded[None], count)
    first, cells = _fit_columns(model.columns[0], cells, length)
    tokens = _first_rows(model, cells, count)
    second, tokens = _fit_columns(model.columns[1], tokens, length)
    rows = _second_rows(model, tokens, count)[0]
    columns = TabFMColumns(inducing=(first, second), categorical=coded)
    return TableState(snapshot=snapshot, columns=columns, num_classes=num_classes), rows


def embed(
    model: TabFM,
    state: TableState,
    features: Float[Array, " rows features"],
    targets: Array | None = None,
) -> Float[Array, " rows icl"]:
    """Embeds rows through preprocessing, both column and row stages.

    Each row's representation depends only on that row and `state`.

    Args:
        model: TabFM parameters.
        state: State from `fit`.
        features: Rows, with NaN for missing values.
        targets: Class labels of context rows, or None for queries.

    Returns:
        One representation per row.
    """
    prepared = apply_numeric_snapshot(state.snapshot, features).astype(
        compute_dtype(model)
    )
    count = batch_length(state.snapshot.columns.count)
    columns: TabFMColumns = state.columns
    labels = None if targets is None else targets[None]
    cells = _cells(model, prepared[None], labels, columns.categorical[None], count)
    cells = _embed_columns(model.columns[0], columns.inducing[0], cells)
    tokens = _first_rows(model, cells, count)
    tokens = _embed_columns(model.columns[1], columns.inducing[1], tokens)
    return _second_rows(model, tokens, count)[0]


def _encode_context(
    model: TabFM,
    context: Float[Array, " batch context icl"],
    targets: Array,
    length: Int[Array, " batch"] | None,
) -> ICLCache:
    """Each in-context block's keys and values of the labeled context rows."""
    hidden = context + embed_targets(model.icl_targets, targets).astype(context.dtype)
    layers: list[AttentionCache] = []
    for index, block in enumerate(model.icl_blocks):
        cache = block_kv(block, hidden, length)
        layers.append(cache)
        # Queries read only the last block's keys, so context rows stop there.
        if index < len(model.icl_blocks) - 1:
            hidden = attention_block(block, hidden, cache)
    return ICLCache(layers=tuple(layers))


def _decode_queries(
    model: TabFM, cache: ICLCache, queries: Float[Array, " batch queries icl"]
) -> Float[Array, " batch queries classes"]:
    """Query rows through the in-context blocks against the cache, then the decoder."""
    hidden = queries
    for block, layer in zip(model.icl_blocks, cache.layers, strict=True):
        hidden = attention_block(block, hidden, layer)
    hidden = rms_norm(model.icl_norm, hidden)
    hidden = jnn.gelu(linear(model.decoder.input, hidden), approximate=True)
    return linear(model.decoder.output, hidden)


def encode(
    model: TabFM,
    state: TableState,
    rows: Float[Array, " context icl"],
    targets: Array,
    num_rows: Int[Array, ""] | int | None = None,
) -> ICLCache:
    """Builds the cache queries read from labeled context rows.

    Args:
        model: TabFM parameters.
        state: State from `fit`; TabFM's in-context stage does not read it.
        rows: Context row representations.
        targets: Class labels of the context rows.
        num_rows: Real rows at the start; the rest are padding that queries
            ignore. None means all rows. May be traced.

    Returns:
        Keys and values per in-context block, without a batch axis.
    """
    del state
    cache = _encode_context(model, rows[None], targets[None], batch_length(num_rows))
    return jax.tree.map(lambda values: values[0], cache)


def decode(
    model: TabFM,
    state: TableState,
    cache: ICLCache,
    rows: Float[Array, " queries icl"],
) -> Float[Array, " queries classes"]:
    """Decodes query rows against a context cache.

    Args:
        model: TabFM parameters.
        state: State from `fit`; unread.
        cache: Cache from `encode`.
        rows: Query row representations.

    Returns:
        Logits over every class slot.
    """
    del state
    batched = jax.tree.map(lambda values: values[None], cache)
    return _decode_queries(model, batched, rows[None])[0]


def postprocess(
    model: TabFM,
    state: TableState,
    outputs: Float[Array, " queries classes"],
) -> Float[Array, " queries predictions"]:
    """Float32 probabilities over the task's classes, at TabFM's softmax temperature."""
    return class_probabilities(
        outputs.astype(jnp.float32), state.num_classes, model.temperature
    )


def properties(model: TabFM) -> TabularProperties:
    """Describes TabFM from its parameter shapes.

    Ensembles average logits before the tempered softmax, as TabFM's own
    classifier does by default.

    Args:
        model: TabFM parameters, real or abstract.

    Returns:
        The model's properties.
    """
    slots = model.decoder.output.weight.shape[0]
    return TabularProperties(
        regression=False,
        class_slots=slots,
        num_outputs=slots,
        averaging=Averaging("outputs"),
        # Its attention reads stored caches, scales and all.
        stored_cache=True,
        column_summary=True,
    )


tabfm_functions = TabularFunctions(
    fit=fit,
    embed=embed,
    encode=encode,
    decode=decode,
    postprocess=postprocess,
    properties=properties,
)


def tabfm_outputs(
    model: TabFM,
    table: Float[Array, " batch rows features"],
    targets: Array,
    context_size: int,
    categorical: Bool[Array, " batch features"] | None = None,
) -> Float[Array, " batch queries classes"]:
    """Runs every stage on prepared rows, context rows first, without preprocessing.

    Constant context columns are left out by the rule `fit` uses, which
    makes it the comparison point for TabFM's own network on the columns
    its preprocessing keeps.

    Args:
        model: TabFM parameters.
        table: Prepared context rows followed by query rows.
        targets: Class labels of the context rows.
        context_size: Number of leading context rows.
        categorical: Which columns hold category codes; None for none.

    Returns:
        Logits over every class slot for the query rows.
    """
    if categorical is None:
        categorical = jnp.zeros((table.shape[0], table.shape[2]), dtype=bool)

    def kept(rows: Array, coded: Array) -> tuple[Array, Array, Array]:
        columns = kept_columns(rows[:context_size])
        return rows[:, columns.order], coded[columns.order], columns.count

    table, categorical, count = jax.vmap(kept)(table, categorical)
    table = table.astype(compute_dtype(model))
    cells = _cells(model, table, targets, categorical, count)
    first, _ = _fit_columns(model.columns[0], cells[:, :context_size], None)
    cells = _embed_columns(model.columns[0], first, cells)
    tokens = _first_rows(model, cells, count)
    second, _ = _fit_columns(model.columns[1], tokens[:, :context_size], None)
    tokens = _embed_columns(model.columns[1], second, tokens)
    rows = _second_rows(model, tokens, count)
    cache = _encode_context(model, rows[:, :context_size], targets, None)
    return _decode_queries(model, cache, rows[:, context_size:])
