# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
from jax import Array
from jaxtyping import Float, Int

from generalforecast.models.contract import Averaging, TabularProperties
from generalforecast.models.prepare import batch_length
from generalforecast.nn.attention import (
    AttentionBlockParameters,
    AttentionCache,
    attention_block,
    block_kv,
)
from generalforecast.nn.linear import (
    LinearParameters,
    MLPParameters,
    ResidualMLPParameters,
    linear,
    mlp,
    residual_mlp,
)
from generalforecast.nn.norm import Norm, normalize
from generalforecast.nn.retrieval import RetrievalParameters, retrieval_cache, retrieve
from generalforecast.nn.target import (
    ClassEncoder,
    TargetEncoder,
    ValueEncoder,
    embed_targets,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ICLParameters:
    """The in-context learning stage TabICL and Kumo Tabular share.

    Target embeddings are added to the context rows, every row attends to the
    context, and the decoder emits class logits or quantiles. Context rows attend
    with every key/value head; queries may read only the first `query_kv_heads`,
    which is static. A retrieval decoder reads the context rows' outputs too.
    `head`, if set, updates the rows the decoder reads after the output norm.
    """

    blocks: tuple[AttentionBlockParameters, ...]
    target_encoder: TargetEncoder
    output_norm: Norm
    decoder: MLPParameters | LinearParameters | RetrievalParameters
    query_kv_heads: int | None = field(default=None, metadata={"static": True})
    head: ResidualMLPParameters | None = None


class ICLCache(NamedTuple):
    """Keys and values of the labeled context rows, one entry per layer.

    With a retrieval decoder, `labels` holds its keys and the one-hot labels.
    """

    layers: tuple[AttentionCache, ...]
    labels: AttentionCache | None = None


def encode_context(
    parameters: ICLParameters,
    context: Float[Array, " batch context model"],
    targets: Array,
    length: Int[Array, " batch"] | None = None,
) -> ICLCache:
    """Builds the per-layer cache for labeled context rows.

    Args:
        parameters: ICL stage parameters.
        context: Context row embeddings.
        targets: Class labels or scaled targets of the context rows.
        length: Real context rows per batch row, or None for all rows. Every
            layer's cache keeps it, so decoding masks the padded rows.

    Returns:
        The context cache, with only the key/value heads queries read.
    """
    heads = parameters.query_kv_heads
    hidden = context + embed_targets(parameters.target_encoder, targets)
    retrieval = isinstance(parameters.decoder, RetrievalParameters)
    layers: list[AttentionCache] = []
    for index, block in enumerate(parameters.blocks):
        cache = block_kv(block, hidden, length)
        if heads is None:
            layers.append(cache)
        else:
            layers.append(
                cache._replace(
                    key=cache.key[..., :heads, :], value=cache.value[..., :heads, :]
                )
            )
        # An MLP decoder reads only the queries' last outputs, so context rows stop
        # at the last keys; a retrieval decoder reads theirs too.
        if retrieval or index < len(parameters.blocks) - 1:
            hidden = attention_block(block, hidden, cache)
    if not isinstance(parameters.decoder, RetrievalParameters):
        return ICLCache(layers=tuple(layers))
    labels = retrieval_cache(
        parameters.decoder,
        _head(parameters, normalize(parameters.output_norm, hidden)),
        targets,
        _num_outputs(parameters),
        length,
    )
    return ICLCache(layers=tuple(layers), labels=labels)


def decode_queries(
    parameters: ICLParameters,
    cache: ICLCache,
    queries: Float[Array, " batch queries model"],
) -> Float[Array, " batch queries outputs"]:
    """Decodes query rows against a context cache.

    Args:
        parameters: ICL stage parameters.
        cache: Cache from `encode_context`.
        queries: Query row embeddings.

    Returns:
        Logits over every class slot, or quantiles in the scaled target space;
        log-probabilities with a retrieval decoder.
    """
    hidden = queries
    for block, layer in zip(parameters.blocks, cache.layers, strict=True):
        hidden = attention_block(block, hidden, layer)
    hidden = _head(parameters, normalize(parameters.output_norm, hidden))
    if isinstance(parameters.decoder, RetrievalParameters):
        assert cache.labels is not None
        return retrieve(parameters.decoder, hidden, cache.labels)
    if isinstance(parameters.decoder, LinearParameters):
        return linear(parameters.decoder, hidden)
    return mlp(parameters.decoder, hidden)


def _head(parameters: ICLParameters, rows: Array) -> Array:
    """The rows the decoder reads, after the optional head block."""
    return rows if parameters.head is None else residual_mlp(parameters.head, rows)


def _num_outputs(parameters: ICLParameters) -> int:
    """Class slots or quantiles the decoder emits."""
    if isinstance(parameters.decoder, RetrievalParameters):
        assert isinstance(parameters.target_encoder, ClassEncoder)
        return parameters.target_encoder.linear.weight.shape[1]
    if isinstance(parameters.decoder, LinearParameters):
        return parameters.decoder.weight.shape[0]
    return parameters.decoder.output.weight.shape[0]


def encode_table(
    parameters: ICLParameters,
    rows: Float[Array, " context model"],
    targets: Array,
    num_rows: Int[Array, ""] | int | None = None,
) -> ICLCache:
    """`encode_context` for one table, without a batch axis.

    Args:
        parameters: ICL stage parameters.
        rows: Context row representations.
        targets: Class labels or scaled targets of the context rows.
        num_rows: Real rows at the start, or None for all. May be traced.

    Returns:
        Keys and values per block, without a batch axis.
    """
    length = batch_length(num_rows)
    cache = encode_context(parameters, rows[None], targets[None], length)
    return jax.tree.map(lambda values: values[0], cache)


def decode_table(
    parameters: ICLParameters,
    cache: ICLCache,
    rows: Float[Array, " queries model"],
) -> Float[Array, " queries outputs"]:
    """`decode_queries` for one table, without a batch axis.

    Args:
        parameters: ICL stage parameters.
        cache: Cache from `encode_table`.
        rows: Query row representations.

    Returns:
        Logits over every class slot, or quantiles in the scaled target space.
    """
    batched = jax.tree.map(lambda values: values[None], cache)
    return decode_queries(parameters, batched, rows[None])[0]


def icl_properties(
    parameters: ICLParameters,
    averaging: Averaging,
    quantiles: int | None = None,
    column_summary: bool = False,
) -> TabularProperties:
    """The properties of a model with this ICL stage.

    Args:
        parameters: ICL stage parameters, real or abstract.
        averaging: How the model's ensembles combine estimators.
        quantiles: Quantiles `postprocess` returns, for a regressor whose decoder
            emits something else, such as buckets.
        column_summary: Whether the model's state pools over the context's rows.

    Returns:
        The model's properties.
    """
    regression = isinstance(parameters.target_encoder, ValueEncoder)
    num_outputs = _num_outputs(parameters)
    return TabularProperties(
        regression=regression,
        class_slots=0 if regression else num_outputs,
        num_outputs=num_outputs if quantiles is None else quantiles,
        averaging=averaging,
        # Its attention reads stored caches, scales and all.
        stored_cache=True,
        column_summary=column_summary,
    )
