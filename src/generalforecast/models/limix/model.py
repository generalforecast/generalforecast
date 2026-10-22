# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.models.buckets import QUANTILE_LEVELS, bucket_quantiles
from generalforecast.models.contract import (
    Averaging,
    ReadCache,
    TabularFunctions,
    TabularProperties,
)
from generalforecast.models.limix.prepare import (
    LimiXSnapshot,
    fit_snapshot,
    kept_groups,
    prepare_features,
)
from generalforecast.models.prepare import batch_length
from generalforecast.models.targets import scale_targets, unscale_quantiles
from generalforecast.nn.attention import (
    AttentionCache,
    AttentionParameters,
    attend,
    project_kv,
)
from generalforecast.nn.linear import (
    GatedMLPParameters,
    LinearParameters,
    MLPParameters,
    gated_mlp,
    linear,
    mlp,
)
from generalforecast.nn.norm import RMSNormParameters, rms_norm


class CellEmbedding(NamedTuple):
    """Each value's MLP, the missing-value vector, and the group fusion MLP.

    Each linear layer is followed by its norm; ReLU follows every norm but the
    fusion's last.
    """

    value_input: LinearParameters
    value_input_norm: RMSNormParameters
    value_output: LinearParameters
    value_output_norm: RMSNormParameters
    missing: Float[Array, " model"]
    fusion_input: LinearParameters
    fusion_input_norm: RMSNormParameters
    fusion_output: LinearParameters
    fusion_output_norm: RMSNormParameters


class ClassTargets(NamedTuple):
    """A target embedding per class, and one for rows without a label."""

    classes: Float[Array, " classes targets"]
    missing: Float[Array, " targets"]


class ValueTargets(NamedTuple):
    """An MLP and norm for scaled targets, and an embedding for rows without one."""

    mlp: MLPParameters
    norm: RMSNormParameters
    missing: Float[Array, " targets"]


class LimiXLayer(NamedTuple):
    """Attention across rows, feed-forward, then attention within each row.

    Across rows, each cell attends to its column's cells and the target tokens,
    merged into one, to theirs. Within a row, cells attend to every token and
    target tokens to the cells, whose readouts one projection mixes.
    """

    cells_across_rows: AttentionParameters
    targets_across_rows: AttentionParameters
    cell_feed_forward: GatedMLPParameters
    target_feed_forward: GatedMLPParameters
    cells_within_row: AttentionParameters
    targets_within_row: AttentionParameters
    targets_readout: LinearParameters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LimiX:
    """LimiX's cell and target embeddings, layers and one task's head.

    Every step of a layer reads tokens normalized with `layer_norm` and adds
    back its output normalized the same way. `group_size`, `target_tokens`,
    `temperature` and `code_seed` are static.
    """

    cells: CellEmbedding
    column_codes: LinearParameters
    targets: ClassTargets | ValueTargets
    task: Float[Array, " model"]
    layers: tuple[LimiXLayer, ...]
    layer_norm: RMSNormParameters
    head_adapter: MLPParameters
    head_norm: RMSNormParameters
    decoder: tuple[LinearParameters, ...]
    borders: Float[Array, " borders"] | None
    group_size: int = field(metadata={"static": True})
    target_tokens: int = field(metadata={"static": True})
    temperature: float = field(default=0.9, metadata={"static": True})
    code_seed: int = field(default=0, metadata={"static": True})

    @property
    def regression(self) -> bool:
        """Whether the model takes scalar targets and predicts quantiles."""
        return isinstance(self.targets, ValueTargets)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TableState:
    """State fit on every context row: preprocessing and each group's code.

    `num_classes` is static: postprocessing slices logits by it.
    """

    snapshot: LimiXSnapshot
    codes: Float[Array, " groups model"]
    num_classes: int = field(metadata={"static": True})


class LimiXCache(NamedTuple):
    """Each layer's keys and values across rows, for cells and for targets."""

    cells: tuple[AttentionCache, ...]
    targets: tuple[AttentionCache, ...]


def _dtype(model: LimiX) -> jnp.dtype:
    """The dtype the network computes in, set by its parameters."""
    return model.column_codes.weight.dtype


def _column_codes(model: LimiX, groups: int) -> Float[Array, " groups model"]:
    """A fixed random code per feature group, projected to the model width.

    Drawn with NumPy, so the codes, and the predictions, stay the same across
    JAX versions, whose random draws for a key have changed before.
    """
    width = model.column_codes.weight.shape[1]
    generator = np.random.default_rng(model.code_seed)
    draws = generator.standard_normal((groups, width), dtype=np.float32)
    return linear(model.column_codes, jnp.asarray(draws, dtype=_dtype(model)))


def _embed_cells(
    cells: CellEmbedding, values: Float[Array, " rows groups group_size"]
) -> Float[Array, " rows groups model"]:
    """Embeds each value, missing ones as the missing vector, then fuses groups."""
    missing = jnp.isnan(values)[..., None]
    hidden = linear(cells.value_input, jnp.where(missing, 0, values[..., None]))
    hidden = jnn.relu(rms_norm(cells.value_input_norm, hidden))
    hidden = linear(cells.value_output, hidden)
    hidden = jnn.relu(rms_norm(cells.value_output_norm, hidden))
    hidden = jnp.where(missing, cells.missing, hidden)
    fused = linear(cells.fusion_input, hidden.reshape(*hidden.shape[:-2], -1))
    fused = jnn.relu(rms_norm(cells.fusion_input_norm, fused))
    return rms_norm(cells.fusion_output_norm, linear(cells.fusion_output, fused))


def _embed_targets(
    model: LimiX, rows: int, targets: Array | None
) -> Float[Array, " rows targets"]:
    """Target embeddings, the missing one for queries and unknown targets."""
    encoder = model.targets
    if targets is None:
        return jnp.broadcast_to(encoder.missing, (rows, encoder.missing.shape[0]))
    if isinstance(encoder, ClassTargets):
        known = (targets >= 0)[:, None]
        return jnp.where(
            known, encoder.classes[jnp.maximum(targets, 0)], encoder.missing
        )
    values = targets.astype(_dtype(model))
    present = jnp.isfinite(values)[:, None]
    hidden = mlp(encoder.mlp, jnp.where(present, values[:, None], 0))
    return jnp.where(present, rms_norm(encoder.norm, hidden), encoder.missing)


def _tokens(
    model: LimiX,
    state: TableState,
    features: Float[Array, " rows features"],
    targets: Array | None,
) -> Float[Array, " rows tokens model"]:
    """A row's cell tokens, with its column codes, then its target tokens."""
    values = prepare_features(state.snapshot, features).astype(_dtype(model))
    cells = _embed_cells(model.cells, values) + state.codes
    scaled = None if targets is None else scale_targets(state.snapshot, targets)
    embedded = _embed_targets(model, features.shape[0], scaled)
    width = model.task.shape[0]
    target_tokens = embedded.reshape(-1, model.target_tokens, width) + model.task
    return jnp.concatenate([cells, target_tokens], axis=1)


def check_class_count(model: LimiX, num_classes: int) -> None:
    """Raises if the model cannot serve `num_classes` classes, 0 for regression."""
    if model.regression:
        if num_classes != 0:
            raise ValueError(f"regressors take num_classes=0, got {num_classes}")
        return
    slots = model.decoder[-1].weight.shape[0]
    if not 1 <= num_classes <= slots:
        raise ValueError(f"num_classes must be in 1 .. {slots}, got {num_classes}")


def fit(
    model: LimiX,
    features: Float[Array, " rows features"],
    targets: Array,
    num_classes: int,
    num_rows: Int[Array, ""] | int | None = None,
    categorical: Bool[Array, " features"] | None = None,
) -> tuple[TableState, Float[Array, " rows tokens_model"]]:
    """Fits the table state and embeds the context rows in the same pass.

    LimiX reads every column as numbers, so `categorical` changes nothing.

    Args:
        model: LimiX parameters.
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
    snapshot = fit_snapshot(
        features,
        model.group_size,
        targets if model.regression else None,
        num_rows,
    )
    codes = _column_codes(model, snapshot.group_scale.shape[0])
    state = TableState(snapshot=snapshot, codes=codes, num_classes=num_classes)
    return state, embed(model, state, features, targets)


def embed(
    model: LimiX,
    state: TableState,
    features: Float[Array, " rows features"],
    targets: Array | None = None,
) -> Float[Array, " rows tokens_model"]:
    """Embeds rows as their cell and target tokens, side by side.

    Args:
        model: LimiX parameters.
        state: State from `fit`.
        features: Rows, with NaN or infinities for missing values.
        targets: Targets of context rows, or None for queries.

    Returns:
        One representation per row.
    """
    tokens = _tokens(model, state, features, targets)
    return tokens.reshape(tokens.shape[0], -1)


def _split(model: LimiX, tokens: Array) -> tuple[Array, Array]:
    """Cell tokens, and the target tokens merged into one."""
    cells = tokens[..., : -model.target_tokens, :]
    merged = tokens[..., -model.target_tokens :, :]
    return cells, merged.reshape(*merged.shape[:-2], -1)


def _join(model: LimiX, cells: Array, targets: Array) -> Array:
    """Cell tokens and merged target tokens as one token axis."""
    width = cells.shape[-1]
    split = targets.reshape(*targets.shape[:-1], model.target_tokens, width)
    return jnp.concatenate([cells, split], axis=-2)


def _within_row(
    model: LimiX,
    layer: LimiXLayer,
    tokens: Float[Array, " batch rows tokens model"],
    groups: Int[Array, ""],
) -> Float[Array, " batch rows tokens model"]:
    """Attention among one row's tokens, every row on its own.

    Only the first `groups` cell tokens are keys, as if the rest were dropped.
    Positions do not enter, so the target tokens go first among the keys and a
    length masks the cells after the kept ones.
    """
    flat = tokens.reshape(-1, *tokens.shape[-2:])
    cells, targets = flat[:, : -model.target_tokens], flat[:, -model.target_tokens :]
    rows = jnp.full(flat.shape[0], groups)
    every = project_kv(
        layer.cells_within_row,
        jnp.concatenate([targets, cells], axis=1),
        rows + model.target_tokens,
    )
    cell_update = attend(layer.cells_within_row, cells, every)
    readouts = attend(
        layer.targets_within_row,
        targets,
        project_kv(layer.targets_within_row, cells, rows),
    )
    mixed = linear(layer.targets_readout, readouts.reshape(readouts.shape[0], -1))
    target_update = mixed.reshape(targets.shape)
    update = jnp.concatenate([cell_update, target_update], axis=1)
    return update.reshape(tokens.shape)


def _by_rows(
    function: Callable[[Array], Array],
    values: Float[Array, " batch rows tokens model"],
    tokens: int | None,
) -> Float[Array, " batch rows tokens features"]:
    """Applies a row-wise `function` to blocks of about `tokens` tokens' rows at a time."""
    rows = values.shape[1]
    steps = 1 if tokens is None else -(-rows * values.shape[2] // tokens)
    if steps <= 1:
        return function(values)
    size = -(-rows // steps)
    # Unrolled, so XLA reuses one block's buffers for the next.
    return jnp.concatenate(
        [function(values[:, start : start + size]) for start in range(0, rows, size)],
        axis=1,
    )


def _layer(
    model: LimiX,
    layer: LimiXLayer,
    tokens: Float[Array, " batch rows tokens model"],
    cells_cache: AttentionCache,
    targets_cache: AttentionCache,
    groups: Int[Array, ""],
) -> Float[Array, " batch rows tokens model"]:
    """One layer for rows that attend to the cached context rows.

    Each step reads normalized tokens and adds its normalized output back.
    Within a row, only the first `groups` cell tokens are keys.
    """
    norm = model.layer_norm
    cells, targets = _split(model, rms_norm(norm, tokens))
    across = _join(
        model,
        attend(layer.cells_across_rows, cells, cells_cache),
        attend(layer.targets_across_rows, targets, targets_cache),
    )
    tokens = tokens + rms_norm(norm, across)
    cells, targets = _split(model, rms_norm(norm, tokens))
    fed = _join(
        model,
        _by_rows(
            lambda cells: gated_mlp(layer.cell_feed_forward, cells),
            cells,
            layer.cells_within_row.kernel.block_chunk,
        ),
        gated_mlp(layer.target_feed_forward, targets),
    )
    tokens = tokens + rms_norm(norm, fed)
    within = _within_row(model, layer, rms_norm(norm, tokens), groups)
    return tokens + rms_norm(norm, within)


def _as_tokens(model: LimiX, rows: Array) -> Float[Array, " 1 rows tokens model"]:
    """Row representations as tokens, in a batch of one."""
    width = model.task.shape[0]
    return rows.reshape(1, rows.shape[0], -1, width)


def encode(
    model: LimiX,
    state: TableState,
    rows: Float[Array, " context tokens_model"],
    targets: Array,
    num_rows: Int[Array, ""] | int | None = None,
) -> LimiXCache:
    """Builds the cache queries read from labeled context rows.

    Every layer keeps the keys and values of each normalized context row across
    rows, for its cells and its targets; the rows then update by attending to them.

    Args:
        model: LimiX parameters.
        state: State from `fit`. The targets are already in `rows`.
        rows: Context row representations.
        targets: Targets of the context rows; already embedded in `rows`.
        num_rows: Real rows at the start; the rest are padding. May be traced.

    Returns:
        Keys and values per layer, without a batch axis.
    """
    del targets
    groups = kept_groups(state.snapshot)
    length = batch_length(num_rows)
    tokens = _as_tokens(model, rows)
    cells_caches: list[AttentionCache] = []
    targets_caches: list[AttentionCache] = []
    for index, layer in enumerate(model.layers):
        cells, merged = _split(model, rms_norm(model.layer_norm, tokens))
        cells_cache = project_kv(layer.cells_across_rows, cells, length)
        targets_cache = project_kv(layer.targets_across_rows, merged, length)
        cells_caches.append(cells_cache)
        targets_caches.append(targets_cache)
        # Queries read only the last layer's keys, so context rows stop there.
        if index < len(model.layers) - 1:
            tokens = _layer(model, layer, tokens, cells_cache, targets_cache, groups)
    cache = LimiXCache(cells=tuple(cells_caches), targets=tuple(targets_caches))
    return jax.tree.map(lambda values: values[0], cache)


def decode(
    model: LimiX,
    state: TableState,
    cache: LimiXCache,
    rows: Float[Array, " queries tokens_model"],
) -> Float[Array, " queries outputs"]:
    """Decodes query rows against a context cache.

    Args:
        model: LimiX parameters.
        state: State from `fit`.
        cache: Cache from `encode`.
        rows: Query row representations.

    Returns:
        Class logits over every class slot, or logits over the regression buckets.
    """

    # Tied to `after`, so XLA lays out each layer's keys only as that layer runs.
    def read(after: Any, part: Callable[[Any], Any]) -> Any:
        return part(jax.lax.optimization_barrier((after, cache))[1])

    return decode_reading(model, state, read, rows)


def decode_reading(
    model: LimiX,
    state: TableState,
    read: ReadCache,
    rows: Float[Array, " queries tokens_model"],
) -> Float[Array, " queries outputs"]:
    """Decodes query rows as `decode` does, reading each layer's cache as that layer runs.

    Args:
        model: LimiX parameters.
        state: State from `fit`.
        read: Reads a part of the cache from `encode`.
        rows: Query row representations.

    Returns:
        Class logits over every class slot, or logits over the regression buckets.
    """
    groups = kept_groups(state.snapshot)
    tokens = _as_tokens(model, rows)
    for index, layer in enumerate(model.layers):
        cells_cache, targets_cache = jax.tree.map(
            lambda values: values[None],
            read(tokens, lambda cache, at=index: (cache.cells[at], cache.targets[at])),
        )
        tokens = _layer(model, layer, tokens, cells_cache, targets_cache, groups)
    _, targets = _split(model, rms_norm(model.layer_norm, tokens)[0])
    hidden = rms_norm(model.head_norm, mlp(model.head_adapter, targets))
    for index, layer in enumerate(model.decoder):
        hidden = linear(layer, hidden)
        if index < len(model.decoder) - 1:
            hidden = jnn.gelu(hidden, approximate=False)
    return hidden


def postprocess(
    model: LimiX,
    state: TableState,
    outputs: Float[Array, " queries outputs"],
) -> Float[Array, " queries predictions"]:
    """Turns decoder outputs into predictions.

    Args:
        model: LimiX parameters.
        state: State from `fit`.
        outputs: Outputs from `decode`.

    Returns:
        Float32 probabilities over the task's classes, or 999 sorted float32
        quantiles on the original target scale, where index k is level
        (k + 1) / 1000.
    """
    logits = outputs.astype(jnp.float32) / model.temperature
    if model.borders is None:
        return jnn.softmax(logits[..., : state.num_classes], axis=-1)
    quantiles = bucket_quantiles(logits, model.borders.astype(jnp.float32))
    return unscale_quantiles(state.snapshot, quantiles)


def properties(model: LimiX) -> TabularProperties:
    """Describes LimiX from its parameter shapes.

    Ensembles average classifier probabilities, and regressors' tempered
    distributions over buckets, as LimiX's own predictor does.

    Args:
        model: LimiX parameters, real or abstract.

    Returns:
        The model's properties.
    """
    if model.regression:
        return TabularProperties(
            regression=True,
            class_slots=0,
            num_outputs=QUANTILE_LEVELS,
            averaging=Averaging("buckets", temperature=model.temperature),
            # Its attention reads stored caches, scales and all.
            stored_cache=True,
            unlabeled_context=True,
        )
    slots = model.decoder[-1].weight.shape[0]
    return TabularProperties(
        regression=False,
        class_slots=slots,
        num_outputs=slots,
        averaging=Averaging("predictions"),
        stored_cache=True,
        unlabeled_context=True,
    )


limix_functions = TabularFunctions(
    fit=fit,
    embed=embed,
    encode=encode,
    decode=decode,
    postprocess=postprocess,
    properties=properties,
    decode_reading=decode_reading,
)
