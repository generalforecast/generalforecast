from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.buckets import QUANTILE_LEVELS, bucket_quantiles
from generalforecast.models.contract import (
    Averaging,
    TabularFunctions,
    TabularProperties,
)
from generalforecast.models.mitra.prepare import (
    MitraSnapshot,
    bucket_borders,
    fit_snapshot,
    prepare_features,
    target_buckets,
)
from generalforecast.models.prepare import batch_length
from generalforecast.models.targets import unscale_quantiles
from generalforecast.nn.attention import (
    AttentionBlockParameters,
    AttentionCache,
    attention_block,
    block_kv,
)
from generalforecast.nn.linear import LinearParameters, linear
from generalforecast.nn.norm import LayerNormParameters, layer_norm


class ClassTargets(NamedTuple):
    """A target embedding per class or bucket, and one for rows without a label."""

    classes: Float[Array, " classes model"]
    missing: Float[Array, " model"]


class MitraLayer(NamedTuple):
    """Attention across rows, then within each row, each with its feed-forward.

    Across rows, each token attends to the same token of the context rows.
    Within a row, the target token and the cells attend to each other.
    """

    across_rows: AttentionBlockParameters
    within_row: AttentionBlockParameters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Mitra:
    """Mitra's cell and target embeddings, layers and head.

    A regressor embeds each target by its bucket and predicts logits over the
    buckets, as a classifier does over classes; `regression` is static.
    """

    cells: LinearParameters
    targets: ClassTargets
    layers: tuple[MitraLayer, ...]
    final_norm: LayerNormParameters
    decoder: LinearParameters
    regression: bool = field(default=False, metadata={"static": True})


def _slots(model: Mitra) -> int:
    """The classes, or regression buckets, the head predicts over."""
    return model.decoder.weight.shape[0]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TableState:
    """State fit on every context row: the preprocessing snapshot.

    `num_classes` is static: postprocessing slices logits by it.
    """

    snapshot: MitraSnapshot
    num_classes: int = field(metadata={"static": True})


def _dtype(model: Mitra) -> jnp.dtype:
    """The dtype the network computes in, set by its parameters."""
    return model.cells.weight.dtype


def check_class_count(model: Mitra, num_classes: int) -> None:
    """Raises if the model cannot serve `num_classes` classes, 0 for regression."""
    if model.regression:
        if num_classes != 0:
            raise ValueError(f"regressors take num_classes=0, got {num_classes}")
        return
    slots = _slots(model)
    if not 1 <= num_classes <= slots:
        raise ValueError(f"num_classes must be in 1 .. {slots}, got {num_classes}")


def _tokens(
    model: Mitra,
    state: TableState,
    features: Float[Array, " rows features"],
    targets: Array | None,
) -> Float[Array, " rows tokens model"]:
    """A row's target token, then a token per cell, columns that vary first."""
    values = prepare_features(state.snapshot, features).astype(_dtype(model))
    cells = linear(model.cells, values[..., None])
    encoder = model.targets
    if targets is None:
        target = jnp.broadcast_to(encoder.missing, (features.shape[0], cells.shape[-1]))
    elif model.regression:
        # A target not known yet takes the mask, as a query's does.
        known = jnp.isfinite(targets)
        buckets = target_buckets(
            state.snapshot, jnp.where(known, targets, 0), _slots(model)
        )
        target = jnp.where(known[:, None], encoder.classes[buckets], encoder.missing)
    else:
        known = targets >= 0
        target = jnp.where(
            known[:, None], encoder.classes[jnp.maximum(targets, 0)], encoder.missing
        )
    return jnp.concatenate([target[:, None], cells], axis=1)


def fit(
    model: Mitra,
    features: Float[Array, " rows features"],
    targets: Array,
    num_classes: int,
    num_rows: Int[Array, ""] | int | None = None,
    categorical: Bool[Array, " features"] | None = None,
) -> tuple[TableState, Float[Array, " rows tokens_model"]]:
    """Fits the table state and embeds the context rows in the same pass.

    Mitra reads every column as numbers, so `categorical` changes nothing.

    Args:
        model: Mitra parameters.
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
    check_class_count(model, num_classes)
    snapshot = fit_snapshot(features, targets if model.regression else None, num_rows)
    state = TableState(snapshot, num_classes)
    return state, embed(model, state, features, targets)


def embed(
    model: Mitra,
    state: TableState,
    features: Float[Array, " rows features"],
    targets: Array | None = None,
) -> Float[Array, " rows tokens_model"]:
    """Embeds rows as their target and cell tokens, side by side.

    Args:
        model: Mitra parameters.
        state: State from `fit`.
        features: Rows, with NaN or infinities for missing values.
        targets: Targets of context rows, or None for queries.

    Returns:
        One representation per row.
    """
    tokens = _tokens(model, state, features, targets)
    return tokens.reshape(tokens.shape[0], -1)


def _within_row(
    block: AttentionBlockParameters,
    tokens: Float[Array, " batch rows tokens model"],
    length: Int[Array, ""],
) -> Float[Array, " batch rows tokens model"]:
    """Attention among one row's tokens, every row on its own.

    Only the target token and the columns that vary are keys, as if the
    constant columns were not in the table.
    """
    flat = tokens.reshape(-1, *tokens.shape[-2:])
    lengths = jnp.broadcast_to(length, flat.shape[:1])
    updated = attention_block(block, flat, block_kv(block, flat, lengths))
    return updated.reshape(tokens.shape)


def _layer(
    layer: MitraLayer,
    tokens: Float[Array, " batch rows tokens model"],
    cache: AttentionCache,
    length: Int[Array, ""],
) -> Float[Array, " batch rows tokens model"]:
    """One layer for rows that attend to the cached context rows."""
    tokens = attention_block(layer.across_rows, tokens, cache)
    return _within_row(layer.within_row, tokens, length)


def _as_tokens(model: Mitra, rows: Array) -> Float[Array, " 1 rows tokens model"]:
    """Row representations as tokens, in a batch of one."""
    width = model.cells.weight.shape[0]
    return rows.reshape(1, rows.shape[0], -1, width)


def _keys_per_row(state: TableState) -> Int[Array, ""]:
    """The target token and the columns that vary."""
    return state.snapshot.columns.count + 1


def encode(
    model: Mitra,
    state: TableState,
    rows: Float[Array, " context tokens_model"],
    targets: Array,
    num_rows: Int[Array, ""] | int | None = None,
) -> tuple[AttentionCache, ...]:
    """Builds the cache queries read from labeled context rows.

    Every layer keeps the keys and values of each normalized context row across
    rows; the rows then update by attending to them.

    Args:
        model: Mitra parameters.
        state: State from `fit`.
        rows: Context row representations.
        targets: Targets of the context rows; already embedded in `rows`.
        num_rows: Real rows at the start; the rest are padding. May be traced.

    Returns:
        Keys and values per layer, without a batch axis.
    """
    del targets
    length = batch_length(num_rows)
    tokens = _as_tokens(model, rows)
    caches: list[AttentionCache] = []
    for index, layer in enumerate(model.layers):
        cache = block_kv(layer.across_rows, tokens, length)
        caches.append(cache)
        # Queries read only the last layer's keys, so context rows stop there.
        if index < len(model.layers) - 1:
            tokens = _layer(layer, tokens, cache, _keys_per_row(state))
    return jax.tree.map(lambda values: values[0], tuple(caches))


def decode(
    model: Mitra,
    state: TableState,
    cache: tuple[AttentionCache, ...],
    rows: Float[Array, " queries tokens_model"],
) -> Float[Array, " queries outputs"]:
    """Decodes query rows against a context cache.

    Args:
        model: Mitra parameters.
        state: State from `fit`.
        cache: Cache from `encode`.
        rows: Query row representations.

    Returns:
        Logits over every class slot, or over the regression buckets.
    """
    batched = jax.tree.map(lambda values: values[None], cache)
    tokens = _as_tokens(model, rows)
    for layer, layer_cache in zip(model.layers, batched, strict=True):
        tokens = _layer(layer, tokens, layer_cache, _keys_per_row(state))
    target = layer_norm(model.final_norm, tokens[0, :, 0])
    return linear(model.decoder, target)


def postprocess(
    model: Mitra,
    state: TableState,
    outputs: Float[Array, " queries outputs"],
) -> Float[Array, " queries predictions"]:
    """Turns logits into predictions.

    Args:
        model: Mitra parameters.
        state: State from `fit`.
        outputs: Outputs from `decode`.

    Returns:
        Float32 probabilities over the task's classes, or 999 sorted float32
        quantiles on the original target scale, where index k is level
        (k + 1) / 1000.
    """
    logits = outputs.astype(jnp.float32)
    if not model.regression:
        return jnn.softmax(logits[..., : state.num_classes], axis=-1)
    quantiles = bucket_quantiles(logits, bucket_borders(_slots(model)))
    return unscale_quantiles(state.snapshot, quantiles)


def properties(model: Mitra) -> TabularProperties:
    """Describes Mitra from its parameter shapes.

    Ensembles average class probabilities, and regressors' distributions over
    buckets, as Mitra-2's own bagging does.

    Args:
        model: Mitra parameters, real or abstract.

    Returns:
        The model's properties.
    """
    if model.regression:
        return TabularProperties(
            regression=True,
            class_slots=0,
            num_outputs=QUANTILE_LEVELS,
            averaging=Averaging("buckets"),
            stored_cache=True,
            unlabeled_context=True,
        )
    slots = _slots(model)
    return TabularProperties(
        regression=False,
        class_slots=slots,
        num_outputs=slots,
        averaging=Averaging("predictions"),
        stored_cache=True,
        unlabeled_context=True,
    )


mitra_functions = TabularFunctions(
    fit=fit,
    embed=embed,
    encode=encode,
    decode=decode,
    postprocess=postprocess,
    properties=properties,
)
