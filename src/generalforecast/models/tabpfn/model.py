# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.buckets import QUANTILE_LEVELS, bucket_quantiles
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
from generalforecast.models.tabpfn.buckets import regression_borders
from generalforecast.models.tabpfn.prepare import (
    fit_snapshot,
    prepare_features,
)
from generalforecast.models.targets import scale_targets, unscale_quantiles
from generalforecast.nn.target import ValueEncoder


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TabPFN:
    """TabPFN's column, row and in-context learning stages.

    `num_buckets` (None for a classifier), `temperature` and `rank_buckets` (None
    without ranks) are static.
    """

    columns: ColumnEmbeddingParameters
    rows: RowInteractionParameters
    icl: ICLParameters
    num_buckets: int | None = field(default=None, metadata={"static": True})
    temperature: float = field(default=0.9, metadata={"static": True})
    rank_buckets: int | None = field(default=None, metadata={"static": True})

    @property
    def regression(self) -> bool:
        """Whether the model takes scalar targets and predicts quantiles."""
        return isinstance(self.icl.target_encoder, ValueEncoder)


def fit(
    model: TabPFN,
    features: Float[Array, " rows features"],
    targets: Array,
    num_classes: int,
    num_rows: Int[Array, ""] | int | None = None,
    categorical: Bool[Array, " features"] | None = None,
) -> tuple[TableState, Float[Array, " rows model"]]:
    """Fits the table state and embeds the context rows in the same pass.

    TabPFN's network reads every column as numbers, so `categorical` changes
    nothing.

    Args:
        model: TabPFN parameters.
        features: Context rows, with NaN or infinities for missing values.
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
    snapshot = fit_snapshot(
        features,
        targets if model.regression else None,
        num_rows,
        model.rank_buckets,
    )
    prepared = prepare_features(snapshot, features).astype(compute_dtype(model))
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
    model: TabPFN,
    state: TableState,
    features: Float[Array, " rows features"],
    targets: Array | None = None,
) -> Float[Array, " rows model"]:
    """Embeds rows through preprocessing, the column stage and the row stage.

    Args:
        model: TabPFN parameters.
        state: State from `fit`.
        features: Rows, with NaN or infinities for missing values.
        targets: Targets of context rows, or None for queries.

    Returns:
        One representation per row.
    """
    prepared = prepare_features(state.snapshot, features).astype(compute_dtype(model))
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
    model: TabPFN,
    state: TableState,
    rows: Float[Array, " context model"],
    targets: Array,
    num_rows: Int[Array, ""] | int | None = None,
) -> ICLCache:
    """Builds the cache queries read from labeled context rows.

    Args:
        model: TabPFN parameters.
        state: State from `fit`.
        rows: Context row representations.
        targets: Class labels or regression targets of the context rows.
        num_rows: Real rows at the start; the rest are padding. May be traced.

    Returns:
        Keys and values per ICL block, and the decoder's keys and labels for a
        classifier, without a batch axis.
    """
    return encode_table(
        model.icl, rows, scale_targets(state.snapshot, targets), num_rows
    )


def decode(
    model: TabPFN,
    state: TableState,
    cache: ICLCache,
    rows: Float[Array, " queries model"],
) -> Float[Array, " queries outputs"]:
    """Decodes query rows against a context cache.

    Args:
        model: TabPFN parameters.
        state: State from `fit`. TabPFN does not read it.
        cache: Cache from `encode`.
        rows: Query row representations.

    Returns:
        Class log-probabilities over every class slot, or logits over the
        regression buckets.
    """
    return decode_table(model.icl, cache, rows)


def postprocess(
    model: TabPFN,
    state: TableState,
    outputs: Float[Array, " queries outputs"],
) -> Float[Array, " queries predictions"]:
    """Turns decoder outputs into predictions.

    Args:
        model: TabPFN parameters.
        state: State from `fit`.
        outputs: Outputs from `decode`.

    Returns:
        Float32 probabilities over the task's classes, or 999 sorted float32
        quantiles on the original target scale, where index k is level
        (k + 1) / 1000.
    """
    outputs = outputs.astype(jnp.float32)
    if model.num_buckets is None:
        logits = outputs[..., : state.num_classes] / model.temperature
        return jnn.softmax(logits, axis=-1)
    borders = jnp.asarray(regression_borders(model.num_buckets))
    # TabPFN tempers regression logits as it does class logits.
    quantiles = bucket_quantiles(outputs / model.temperature, borders)
    return unscale_quantiles(state.snapshot, quantiles)


def properties(model: TabPFN) -> TabularProperties:
    """Describes TabPFN from its parameter shapes.

    Ensembles average classifier probabilities, and regressors' distributions
    over buckets, as TabPFN's own estimators do.

    Args:
        model: TabPFN parameters, real or abstract.

    Returns:
        The model's properties.
    """
    if model.num_buckets is None:
        return icl_properties(model.icl, Averaging("predictions"), column_summary=True)
    averaging = Averaging("buckets", temperature=model.temperature)
    return icl_properties(model.icl, averaging, QUANTILE_LEVELS, column_summary=True)


tabpfn_functions = TabularFunctions(
    fit=fit,
    embed=embed,
    encode=encode,
    decode=decode,
    postprocess=postprocess,
    properties=properties,
)


def tabpfn_outputs(
    model: TabPFN,
    table: Float[Array, " batch rows features"],
    targets: Array,
    context_size: int,
) -> Float[Array, " batch queries outputs"]:
    """Runs the network on raw rows, context rows first, as TabPFN's forward does.

    Columns constant in the context are left out, as TabPFN's preprocessing
    drops them before the forward.

    Args:
        model: TabPFN parameters.
        table: Context rows followed by query rows, with NaN for missing values.
        targets: Class labels or scaled targets of the context rows.
        context_size: Number of leading context rows.

    Returns:
        Class log-probabilities or bucket logits for the query rows.
    """

    def prepare(rows: Array) -> tuple[Array, Array]:
        snapshot = fit_snapshot(rows[:context_size], rank_buckets=model.rank_buckets)
        prepared = prepare_features(snapshot, rows).astype(compute_dtype(model))
        return prepared, snapshot.columns.count

    prepared, count = jax.vmap(prepare)(table)
    summary = summarize_columns(
        model.columns, prepared[:, :context_size], targets, count=count
    )
    cells = embed_cells(model.columns, summary, prepared, targets, count)
    rows = interact_rows(model.rows, cells, count)
    cache = encode_context(model.icl, rows[:, :context_size], targets)
    return decode_queries(model.icl, cache, rows[:, context_size:])
