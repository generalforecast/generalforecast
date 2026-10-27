# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.column_embedding import (
    ColumnEmbeddingParameters,
    embed_cells,
    fit_columns,
    summarize_columns,
)
from generalforecast.models.common import TableState, check_class_count, compute_dtype
from generalforecast.models.contract import (
    Averaging,
    TabularFunctions,
    TabularProperties,
)
from generalforecast.models.icl import (
    ICLCache,
    ICLParameters,
    decode_queries,
    decode_table,
    encode_context,
    encode_table,
    icl_properties,
)
from generalforecast.models.prepare import batch_length
from generalforecast.models.row_interaction import (
    RowInteractionParameters,
    interact_rows,
)
from generalforecast.models.tabicl.prepare import (
    apply_numeric_snapshot,
    fit_numeric_snapshot,
    kept_columns,
)
from generalforecast.models.targets import scale_targets, unscale_quantiles
from generalforecast.nn.target import ValueEncoder


class TabICL(NamedTuple):
    """TabICL's column, row and in-context learning stages."""

    columns: ColumnEmbeddingParameters
    rows: RowInteractionParameters
    icl: ICLParameters

    @property
    def regression(self) -> bool:
        """Whether the model takes scalar targets and predicts quantiles."""
        return isinstance(self.icl.target_encoder, ValueEncoder)


def fit(
    model: TabICL,
    features: Float[Array, " rows features"],
    targets: Array,
    num_classes: int,
    num_rows: Int[Array, ""] | int | None = None,
    categorical: Bool[Array, " features"] | None = None,
) -> tuple[TableState, Float[Array, " rows model"]]:
    """Fits the table state and embeds the context rows in the same pass.

    TabICL v2 reads every column as numbers, so `categorical` changes nothing.
    Preprocessing statistics are float32; the network computes in the dtype of
    the parameters.

    Args:
        model: TabICL parameters.
        features: Context rows, with NaN for missing values.
        targets: Class labels `0 .. num_classes - 1`, or regression targets.
        num_classes: Number of classes in the task, or 0 for regression.
        num_rows: Real rows at the start; the rest are padding that does not
            affect the state. None means all rows. May be traced.
        categorical: Which columns are categorical; unused.

    Returns:
        The table state and one representation per context row.
    """
    del categorical
    check_class_count(properties(model), num_classes)
    snapshot = fit_numeric_snapshot(
        features, targets if model.regression else None, num_rows
    )
    prepared = apply_numeric_snapshot(snapshot, features).astype(compute_dtype(model))
    count = batch_length(snapshot.columns.count)
    summary, cells = fit_columns(
        model.columns,
        prepared[None, ...],
        scale_targets(snapshot, targets)[None, ...],
        batch_length(num_rows),
        count,
    )
    state = TableState(snapshot=snapshot, columns=summary, num_classes=num_classes)
    return state, interact_rows(model.rows, cells, count)[0]


def embed(
    model: TabICL,
    state: TableState,
    features: Float[Array, " rows features"],
    targets: Array | None = None,
) -> Float[Array, " rows model"]:
    """Embeds rows through preprocessing, the column stage and the row stage.

    Each row's representation depends only on that row and `state`.

    Args:
        model: TabICL parameters.
        state: State from `fit`.
        features: Rows, with NaN for missing values.
        targets: Targets of context rows, or None for queries.

    Returns:
        One representation per row.
    """
    prepared = apply_numeric_snapshot(state.snapshot, features).astype(
        compute_dtype(model)
    )
    scaled = None if targets is None else scale_targets(state.snapshot, targets)
    count = batch_length(state.snapshot.columns.count)
    cells = embed_cells(
        model.columns,
        state.columns,
        prepared[None, ...],
        None if scaled is None else scaled[None, ...],
        count,
    )
    return interact_rows(model.rows, cells, count)[0]


def encode(
    model: TabICL,
    state: TableState,
    rows: Float[Array, " context model"],
    targets: Array,
    num_rows: Int[Array, ""] | int | None = None,
) -> ICLCache:
    """Builds the cache queries read from labeled context rows.

    Args:
        model: TabICL parameters.
        state: State from `fit`.
        rows: Context row representations.
        targets: Class labels or regression targets of the context rows.
        num_rows: Real rows at the start; the rest are padding that queries
            ignore. None means all rows. May be traced.

    Returns:
        Keys and values per ICL block, without a batch axis. The cache records
        its length, so `decode` needs no row count.
    """
    return encode_table(
        model.icl, rows, scale_targets(state.snapshot, targets), num_rows
    )


def decode(
    model: TabICL,
    state: TableState,
    cache: ICLCache,
    rows: Float[Array, " queries model"],
) -> Float[Array, " queries outputs"]:
    """Decodes query rows against a context cache.

    Args:
        model: TabICL parameters.
        state: State from `fit`. TabICL does not read it.
        cache: Cache from `encode`.
        rows: Query row representations.

    Returns:
        Logits over every class slot, or quantiles in the scaled target space.
    """
    return decode_table(model.icl, cache, rows)


def class_probabilities(
    logits: Float[Array, "*batch classes"],
    num_classes: int,
    temperature: float = 0.9,
) -> Float[Array, "*batch observed"]:
    """Softmax over the task's classes, with TabICL's inference temperature.

    Args:
        logits: Logits over every class slot.
        num_classes: Number of classes, which take the first slots.
        temperature: Softmax temperature.

    Returns:
        Class probabilities.
    """
    return jnn.softmax(logits[..., :num_classes] / temperature, axis=-1)


def postprocess(
    model: TabICL,
    state: TableState,
    outputs: Float[Array, " queries outputs"],
) -> Float[Array, " queries predictions"]:
    """Turns decoder outputs into predictions.

    Args:
        model: TabICL parameters.
        state: State from `fit`.
        outputs: Outputs from `decode`.

    Returns:
        Float32 probabilities over the task's classes, or 999 sorted float32
        quantiles on the original target scale, where index k is level
        (k + 1) / 1000.
    """
    outputs = outputs.astype(jnp.float32)
    if model.regression:
        return unscale_quantiles(state.snapshot, outputs)
    return class_probabilities(outputs, state.num_classes)


def properties(model: TabICL) -> TabularProperties:
    """Describes TabICL from its parameter shapes.

    Ensembles average classifier logits before the tempered softmax, and
    regressors' sorted quantiles, as TabICL's own estimators do.

    Args:
        model: TabICL parameters, real or abstract.

    Returns:
        The model's properties.
    """
    averaging = Averaging("predictions" if model.regression else "outputs")
    return icl_properties(model.icl, averaging, column_summary=True)


tabicl_functions = TabularFunctions(
    fit=fit,
    embed=embed,
    encode=encode,
    decode=decode,
    postprocess=postprocess,
    properties=properties,
)


def tabicl_outputs(
    model: TabICL,
    table: Float[Array, " batch rows features"],
    targets: Array,
    context_size: int,
) -> Float[Array, " batch queries outputs"]:
    """Runs the three stages on prepared rows, context rows first.

    Skips preprocessing but for leaving out the columns `fit` leaves out, by
    the same rule, which makes it the comparison point for upstream TabICL on
    the columns its preprocessing keeps.

    Args:
        model: TabICL parameters.
        table: Prepared context rows followed by query rows.
        targets: Class labels or scaled targets of the context rows.
        context_size: Number of leading context rows.

    Returns:
        Logits or quantiles for the query rows.
    """

    def kept(rows: Array) -> tuple[Array, Array]:
        columns = kept_columns(rows[:context_size])
        return rows[:, columns.order], columns.count

    table, count = jax.vmap(kept)(table)
    summary = summarize_columns(
        model.columns, table[:, :context_size], targets, count=count
    )
    cells = embed_cells(model.columns, summary, table, targets, count)
    rows = interact_rows(model.rows, cells, count)
    cache = encode_context(model.icl, rows[:, :context_size], targets)
    return decode_queries(model.icl, cache, rows[:, context_size:])
