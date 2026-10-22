# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.contract import (
    Averaging,
    TabularFunctions,
    TabularProperties,
)
from generalforecast.models.icl import (
    ICLCache,
    ICLParameters,
    check_class_count,
    decode_queries,
    decode_table,
    encode_context,
    encode_table,
    icl_properties,
)
from generalforecast.models.kumo.cell_embedding import impute_means
from generalforecast.models.kumo.prepare import (
    KumoSnapshot,
    fit_snapshot,
    kumo_columns,
    prepare_features,
)
from generalforecast.models.kumo.row_embedding import (
    ColumnStates,
    RowEmbeddingParameters,
    embed_rows,
)
from generalforecast.models.prepare import valid_rows
from generalforecast.models.targets import scale_targets, unscale_quantiles
from generalforecast.nn.linear import LinearParameters, linear
from generalforecast.nn.target import ValueEncoder


class KumoTabular(NamedTuple):
    """NVIDIA's Kumo Tabular: row encoder, optional projection, and ICL stage."""

    rows: RowEmbeddingParameters
    project: LinearParameters | None
    icl: ICLParameters

    @property
    def regression(self) -> bool:
        """Whether the model takes scalar targets and predicts quantiles."""
        return isinstance(self.icl.target_encoder, ValueEncoder)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TableState:
    """State fit on every context row, and which columns are categorical.

    Categories, standardization, imputation and column states. `means` follows
    the prepared columns' order and `categorical` the table's. `num_classes` is
    static, like TabICL's.
    """

    snapshot: KumoSnapshot
    means: Float[Array, " features"]
    columns: ColumnStates
    categorical: Bool[Array, " features"] | None
    num_classes: int = field(metadata={"static": True})


def _prepared_order(
    snapshot: KumoSnapshot, categorical: Bool[Array, " features"] | None
) -> Bool[Array, " features"] | None:
    """The categorical mask in the prepared columns' order, kept columns first."""
    return None if categorical is None else categorical[snapshot.columns.order]


def _project(model: KumoTabular, rows: Array) -> Array:
    """Maps row encoder outputs to the ICL width, if they differ."""
    return rows if model.project is None else linear(model.project, rows)


def fit(
    model: KumoTabular,
    features: Float[Array, " rows features"],
    targets: Array,
    num_classes: int,
    num_rows: Int[Array, ""] | int | None = None,
    categorical: Bool[Array, " features"] | None = None,
) -> tuple[TableState, Float[Array, " rows model"]]:
    """Fits the table state and embeds the context rows in the same pass.

    Categorical columns hold category codes. As in NVIDIA's recipe, each code
    becomes its rank among the context's codes, with -1 for missing and unseen
    ones, is standardized like a number, and embeds through its own cell weights.

    The column blocks interleave with the row blocks, so their states can only be
    fit by running the context through the row encoder.

    Args:
        model: Kumo Tabular parameters.
        features: Context rows, with NaN for missing values.
        targets: Class labels `0 .. num_classes - 1`, or regression targets.
        num_classes: Number of classes in the task, or 0 for regression.
        num_rows: Real rows at the start; the rest are padding that does not
            affect the state. None means all rows. May be traced.
        categorical: Which columns are categorical, or None if none are.

    Returns:
        The table state and one representation per context row.
    """
    check_class_count(model.icl, num_classes)
    valid = valid_rows(features.shape[0], num_rows)
    regression_targets = targets if model.regression else None
    snapshot = fit_snapshot(features, regression_targets, valid, categorical)
    prepared = prepare_features(snapshot, features, categorical)
    means = impute_means(prepared, valid)
    length = None if num_rows is None else jnp.asarray(num_rows)
    columns, rows = embed_rows(
        model.rows,
        prepared,
        means,
        scale_targets(snapshot, targets),
        length=length,
        categorical=_prepared_order(snapshot, categorical),
        count=snapshot.columns.count,
    )
    state = TableState(snapshot, means, columns, categorical, num_classes)
    return state, _project(model, rows)


def embed(
    model: KumoTabular,
    state: TableState,
    features: Float[Array, " rows features"],
    targets: Array | None = None,
) -> Float[Array, " rows model"]:
    """Embeds rows against the fitted column states.

    Args:
        model: Kumo Tabular parameters.
        state: State from `fit`.
        features: Rows, with NaN for missing values.
        targets: Targets of context rows, or None for queries.

    Returns:
        One representation per row.
    """
    prepared = prepare_features(state.snapshot, features, state.categorical)
    scaled = None if targets is None else scale_targets(state.snapshot, targets)
    _, rows = embed_rows(
        model.rows,
        prepared,
        state.means,
        scaled,
        state.columns,
        categorical=_prepared_order(state.snapshot, state.categorical),
        count=state.snapshot.columns.count,
    )
    return _project(model, rows)


def encode(
    model: KumoTabular,
    state: TableState,
    rows: Float[Array, " context model"],
    targets: Array,
    num_rows: Int[Array, ""] | int | None = None,
) -> ICLCache:
    """Builds the cache queries read from labeled context rows.

    Args:
        model: Kumo Tabular parameters.
        state: State from `fit`.
        rows: Context row representations.
        targets: Class labels or regression targets of the context rows.
        num_rows: Real rows at the start; the rest are padding that queries
            ignore. None means all rows. May be traced.

    Returns:
        Keys and values per ICL block, with only the heads queries read.
    """
    return encode_table(
        model.icl, rows, scale_targets(state.snapshot, targets), num_rows
    )


def decode(
    model: KumoTabular,
    state: TableState,
    cache: ICLCache,
    rows: Float[Array, " queries model"],
) -> Float[Array, " queries outputs"]:
    """Decodes query rows against a context cache.

    Args:
        model: Kumo Tabular parameters.
        state: State from `fit`. Kumo Tabular does not read it.
        cache: Cache from `encode`.
        rows: Query row representations.

    Returns:
        Logits over every class slot, or quantiles in the scaled target space.
    """
    return decode_table(model.icl, cache, rows)


def postprocess(
    model: KumoTabular,
    state: TableState,
    outputs: Float[Array, " queries outputs"],
) -> Float[Array, " queries predictions"]:
    """Turns decoder outputs into predictions.

    Args:
        model: Kumo Tabular parameters.
        state: State from `fit`.
        outputs: Outputs from `decode`.

    Returns:
        Float32 probabilities over the task's classes, or 999 sorted float32
        quantiles on the original target scale.
    """
    outputs = outputs.astype(jnp.float32)
    if model.regression:
        return unscale_quantiles(state.snapshot, outputs)
    return jnn.softmax(outputs[..., : state.num_classes], axis=-1)


def properties(model: KumoTabular) -> TabularProperties:
    """Describes Kumo Tabular from its parameter shapes.

    Ensembles average classifier logits before the softmax, and take the
    mean of regressors' sorted quantiles without the top and bottom fifth of
    estimators, as Kumo Tabular's own recipe does.

    Args:
        model: Kumo Tabular parameters, real or abstract.

    Returns:
        The model's properties.
    """
    averaging = Averaging("predictions", trim=0.2) if model.regression else Averaging()
    return icl_properties(model.icl, averaging, column_summary=True)


def crowded_categories(state: TableState) -> int:
    """How many categorical columns held more categories than the state keeps.

    Their ranks are interpolated, so they no longer match NVIDIA's exactly.
    States stacked along leading axes, as ensembles stack them, count the most
    of any one.

    Args:
        state: State from `fit`.

    Returns:
        The number of such columns.
    """
    categories = state.snapshot.categories
    if categories is None or state.categorical is None:
        return 0
    crowded = np.asarray(categories.crowded & state.categorical)
    return int(crowded.sum(axis=-1).max())


kumo_functions = TabularFunctions(
    fit=fit,
    embed=embed,
    encode=encode,
    decode=decode,
    postprocess=postprocess,
    properties=properties,
    crowded_categories=crowded_categories,
)


def kumo_outputs(
    model: KumoTabular,
    table: Float[Array, " batch rows features"],
    targets: Array,
    context_size: int,
    categorical: Bool[Array, " features"] | None = None,
) -> Float[Array, " batch queries outputs"]:
    """Runs the network on prepared rows, context rows first, for one estimator.

    Skips preprocessing but for leaving out the columns constant in the context
    and putting categorical columns after numerical ones, which makes it the
    comparison point for NVIDIA's network on the columns its preprocessing
    keeps, in its order.

    Args:
        model: Kumo Tabular parameters.
        table: Prepared context rows followed by query rows, with NaN for missing.
        targets: Class labels or scaled targets of the context rows.
        context_size: Number of leading context rows.
        categorical: Which columns are categorical, or None if none are.

    Returns:
        Logits or quantiles for the query rows.
    """

    def one(rows: Array, labels: Array) -> Array:
        valid = jnp.ones(context_size, dtype=bool)
        columns = kumo_columns(rows[:context_size], valid, categorical)
        rows = rows[:, columns.order]
        kinds = None if categorical is None else categorical[columns.order]
        means = impute_means(rows[:context_size], valid)
        _, embedded = embed_rows(
            model.rows, rows, means, labels, categorical=kinds, count=columns.count
        )
        embedded = _project(model, embedded)
        cache = encode_context(model.icl, embedded[None, :context_size], labels[None])
        return decode_queries(model.icl, cache, embedded[None, context_size:])[0]

    return jax.vmap(one)(table, targets)
