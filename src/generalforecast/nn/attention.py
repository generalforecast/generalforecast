import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

from generalforecast.nn.linear import LinearParameters, MLPParameters, linear, mlp
from generalforecast.nn.norm import Norm, RMSNormParameters, normalize, rms_norm
from generalforecast.nn.qassmax import QASSMaxParameters
from generalforecast.nn.rope import Positions, RoPE, apply_rope
from generalforecast.nn.scaling import LogScaleParameters, QueryScale, scale_queries


@dataclass(frozen=True)
class AttentionKernel:
    """How attention runs on a device. It changes the program, not the model.

    Attributes:
        implementation: `xla`, or `cudnn` for flash attention. cuDNN needs a GPU,
            float16 or bfloat16, and a head size divisible by 8; otherwise
            attention falls back to XLA.
        query_chunk: Queries per step, so at most this many rows of scores exist
            at once. None attends with every query at once. cuDNN's flash
            attention never holds every score, so only XLA chunks.
        dtype: Dtype of queries, keys and values inside attention, or None for the
            query dtype. Outputs keep the query dtype.
        block_chunk: Positions an attention block takes per step, counted across
            the batch: the rows it updates, or the context rows inducing points
            read. None takes every position at once.
        summary_chunk: Cells a fit's column summary reads per step. A context
            of more cells is summarized a chunk of rows at a time, by a running
            softmax, so the fit's memory does not grow with its rows; the
            worker sets it from its call budget. None reads every cell at once.
    """

    implementation: Literal["xla", "cudnn"] = "xla"
    query_chunk: int | None = None
    dtype: jnp.dtype | None = None
    block_chunk: int | None = None
    summary_chunk: int | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AttentionParameters:
    """Q, K, V and output projections, with optional query scaling, QK-norm and RoPE.

    K and V may have fewer heads than Q (grouped-query attention), with query
    heads sharing K and V heads in order. Without an output projection, attention
    returns the heads side by side. With a `gate`, the heads are multiplied by
    `sigmoid(gate(query_rows))` first. `num_heads`, `rope` and `kernel` are
    static: they are part of the compiled program, not arrays.
    """

    query: LinearParameters
    key: LinearParameters
    value: LinearParameters
    output: LinearParameters | None
    num_heads: int = field(metadata={"static": True})
    scale: QueryScale | None = None
    query_norm: RMSNormParameters | None = None
    key_norm: RMSNormParameters | None = None
    gate: LinearParameters | None = None
    rope: RoPE | None = field(default=None, metadata={"static": True})
    kernel: AttentionKernel = field(
        default=AttentionKernel(), metadata={"static": True}
    )

    @property
    def head_dim(self) -> int:
        """Size of each head."""
        return self.query.weight.shape[0] // self.num_heads


def unpack_qkv(
    qkv_weight: Float[Array, " three_model model"],
    qkv_bias: Float[Array, " three_model"],
    output: LinearParameters,
    num_heads: int,
    scale: QueryScale | None = None,
    rope: RoPE | None = None,
    head_norm: RMSNormParameters | None = None,
) -> AttentionParameters:
    """Splits PyTorch's packed QKV projection (query, key, value stacked) apart.

    Args:
        qkv_weight: Packed query, key and value weights.
        qkv_bias: Packed biases.
        output: Output projection.
        num_heads: Number of attention heads.
        scale: Query scaling, if any.
        rope: RoPE settings, if any.
        head_norm: Per-head norm of queries and keys, if any.

    Returns:
        Attention parameters with separate projections.
    """
    query, key, value = (
        LinearParameters(weight=weight, bias=bias)
        for weight, bias in zip(
            jnp.split(qkv_weight, 3, axis=0),
            jnp.split(qkv_bias, 3, axis=0),
            strict=True,
        )
    )
    return AttentionParameters(
        query=query,
        key=key,
        value=value,
        output=output,
        num_heads=num_heads,
        scale=scale,
        query_norm=head_norm,
        key_norm=head_norm,
        rope=rope,
    )


class AttentionBlockParameters(NamedTuple):
    """A pre-norm attention layer followed by a feed-forward layer.

    Context rows are normalized with `context_norm` if set, otherwise with
    `attention_norm`.
    """

    attention_norm: Norm
    feed_forward_norm: Norm
    attention: AttentionParameters
    feed_forward: MLPParameters
    context_norm: Norm | None = None


class AttentionCache(NamedTuple):
    """Per-head keys and values of the context rows for one layer.

    `length` counts the real keys at the start of each batch row; the rest are
    padding that attention ignores. None means every key is real. Stored int8
    keys and values carry a scale per row and head; attention multiplies it back.
    `mask`, set only when decoding, says which keys each query reads, so queries
    that read different parts of one context decode together.
    """

    key: Float[Array, " batch context kv_heads head_dim"]
    value: Float[Array, " batch context kv_heads head_dim"]
    length: Int[Array, " batch"] | None = None
    key_scale: Float[Array, " batch context kv_heads"] | None = None
    value_scale: Float[Array, " batch context kv_heads"] | None = None
    mask: Bool[Array, " batch queries context"] | None = None


def _split_heads(
    values: Float[Array, " batch rows *tokens model"], head_dim: int
) -> Float[Array, " batch rows heads head_dim"]:
    """Splits the model axis into heads of size `head_dim`, each token's in turn."""
    return values.reshape(*values.shape[:2], -1, head_dim)


def _token_heads(scale: QueryScale, tokens: int) -> QueryScale:
    """Per-head query scaling repeated for each token's heads."""
    if tokens == 1:
        return scale
    if not isinstance(scale, LogScaleParameters):
        raise NotImplementedError("token axes are checked only with log query scaling")
    offset = None if scale.offset is None else jnp.tile(scale.offset, tokens)
    return LogScaleParameters(jnp.tile(scale.head_scale, tokens), offset)


def project_kv(
    parameters: AttentionParameters,
    context_rows: Float[Array, " batch context *tokens model"],
    length: Int[Array, " batch"] | None = None,
    positions: Positions | None = None,
) -> AttentionCache:
    """Projects context rows to per-head keys and values.

    Keys get RoPE, then the key norm, matching the order on q in `attend`.
    Padding rows' keys and values are zero, so whatever filled them, even NaN,
    cannot reach attention through its mask. Token axes after the rows fold
    into the heads, token by token, as in `attend`.

    Args:
        parameters: Attention parameters.
        context_rows: Normalized context rows, with any token axes.
        length: Real rows at the start of each batch row, or None for all rows.
        positions: Each row's position for RoPE, or None for its index.

    Returns:
        The keys and values.
    """
    k = _split_heads(linear(parameters.key, context_rows), parameters.head_dim)
    v = _split_heads(linear(parameters.value, context_rows), parameters.head_dim)
    if parameters.rope is not None:
        k = apply_rope(k, parameters.rope, positions)
    if parameters.key_norm is not None:
        k = rms_norm(parameters.key_norm, k)
    if length is not None:
        real = (jnp.arange(k.shape[-3]) < length[..., None])[..., None, None]
        k, v = jnp.where(real, k, 0), jnp.where(real, v, 0)
    return AttentionCache(key=k, value=v, length=length)


@functools.cache
def _gpu_generation() -> tuple[int, int] | None:
    """The default GPU's compute capability, or None without an NVIDIA GPU."""
    if jax.default_backend() != "gpu":
        return None
    capability = getattr(jax.devices()[0], "compute_capability", None)
    if capability is None:
        return None
    major, minor = (int(part) for part in str(capability).split("."))
    return major, minor


def _implementation(kernel: AttentionKernel, q: Array) -> Literal["xla", "cudnn"]:
    """The kernel's implementation if it supports `q` on this device, else XLA.

    cuDNN's flash attention needs an Ampere or newer GPU, and heads of at most
    128 before Hopper or 256 from it, so older GPUs such as T4s take XLA.
    """
    generation = _gpu_generation()
    supported = (
        generation is not None
        and generation >= (8, 0)
        and q.dtype in (jnp.float16, jnp.bfloat16)
        and q.shape[-1] % 8 == 0
        and q.shape[-1] <= (256 if generation >= (9, 0) else 128)
    )
    return kernel.implementation if supported else "xla"


def _scaled_attention(
    q: Float[Array, " batch rows heads head_dim"],
    cache: AttentionCache,
    mask: Bool[Array, " batch rows context"] | None,
) -> Float[Array, " batch rows heads head_dim"]:
    """Attention to int8 keys and values, without a dequantized copy of them.

    Their scales fold into the scores and the weights instead. As
    `jax.nn.dot_product_attention` does on the dequantized cache, the scores
    accumulate in float32 and the softmax runs in float32; the weights go back
    to the query dtype for the value product.
    """
    assert cache.key_scale is not None and cache.value_scale is not None
    batch, rows, heads, head_dim = q.shape
    kv_heads = cache.key.shape[-2]
    grouped = q.reshape(batch, rows, kv_heads, heads // kv_heads, head_dim)
    key = cache.key.astype(q.dtype)
    scores = jnp.einsum(
        "brkgd,bckd->bkgrc", grouped, key, preferred_element_type=jnp.float32
    )
    scores = scores * head_dim**-0.5
    scores = scores * jnp.moveaxis(cache.key_scale, 1, -1)[:, :, None, None]
    lowest = jnp.finfo(jnp.float32).min
    if cache.length is not None:
        real = jnp.arange(key.shape[1]) < cache.length[:, None]
        scores = jnp.where(real[:, None, None, None], scores, lowest)
    if mask is not None:
        scores = jnp.where(mask[:, None, None], scores, lowest)
    weights = jnn.softmax(scores, axis=-1)
    weights = weights * jnp.moveaxis(cache.value_scale, 1, -1)[:, :, None, None]
    attended = jnp.einsum(
        "bkgrc,bckd->brkgd",
        weights.astype(q.dtype),
        cache.value.astype(q.dtype),
        preferred_element_type=jnp.float32,
    )
    return attended.reshape(batch, rows, heads, head_dim).astype(q.dtype)


def _dot_product_attention(
    q: Float[Array, " batch rows heads head_dim"],
    cache: AttentionCache,
    kernel: AttentionKernel,
) -> Float[Array, " batch rows heads head_dim"]:
    """Scaled dot-product attention in the kernel's dtype, optionally by query chunk."""
    dtype = q.dtype if kernel.dtype is None else kernel.dtype
    output_dtype = q.dtype
    q = q.astype(dtype)
    scaled = cache.key_scale is not None and cache.value_scale is not None
    # cuDNN takes lengths but not a mask per query.
    implementation = "xla" if cache.mask is not None else _implementation(kernel, q)

    def run(queries: Array, mask: Array | None) -> Array:
        if scaled:
            return _scaled_attention(queries, cache, mask).astype(output_dtype)
        attended = jnn.dot_product_attention(
            queries,
            cache.key.astype(dtype),
            cache.value.astype(dtype),
            mask=None if mask is None else mask[:, None],
            key_value_seq_lengths=cache.length,
            implementation=implementation,
        )
        return attended.astype(output_dtype)

    flash = implementation == "cudnn" and not scaled
    chunk, rows = None if flash else kernel.query_chunk, q.shape[-3]
    if chunk is None or rows <= chunk:
        return run(q, cache.mask)

    def chunked(values: Array) -> Array:
        """Rows in chunks along a new leading axis, the last padded."""
        pads = [(0, 0), (0, -rows % chunk)] + [(0, 0)] * (values.ndim - 2)
        padded = jnp.pad(values, pads)
        return padded.reshape(values.shape[0], -1, chunk, *values.shape[2:]).swapaxes(
            0, 1
        )

    masks = None if cache.mask is None else chunked(cache.mask)
    attended = jax.lax.map(lambda pair: run(*pair), (chunked(q), masks)).swapaxes(0, 1)
    return attended.reshape(q.shape[0], -1, *attended.shape[3:])[:, :rows]


def attend(
    parameters: AttentionParameters,
    query_rows: Float[Array, " batch rows *tokens model"],
    cache: AttentionCache,
    positions: Positions | None = None,
) -> Float[Array, " batch rows *tokens output"]:
    """Multi-head attention from `query_rows` to cached keys and values.

    The order on q is RoPE, then the query norm, then query scaling; loaders must
    check that a model uses it. RoPE with QASSMax raises, since no model has
    checked that order. Keys past `cache.length`, and keys `cache.mask` leaves out
    of a query, are masked, and query scaling counts only the keys each query
    reads. `parameters.kernel` picks the implementation, chunking and dtype.

    Token axes after the rows, such as a table's columns, each attend to the
    same token of the cached rows with shared projections: they fold into the
    heads, as in `project_kv`.

    Args:
        parameters: Attention parameters.
        query_rows: Normalized rows to project into queries, with any token axes.
        cache: Keys and values from `project_kv`.
        positions: Each row's position for RoPE, or None for its index.

    Returns:
        Attention outputs after the output projection, or the heads side by
        side without one.
    """
    if cache.mask is not None:
        n_keys = cache.mask.sum(axis=-1)
    elif cache.length is not None:
        n_keys = cache.length
    else:
        n_keys = cache.key.shape[-3]
    q = _queries(parameters, query_rows, n_keys, positions)
    attended = _dot_product_attention(q, cache, parameters.kernel)
    return _output(parameters, query_rows, attended)


def _queries(
    parameters: AttentionParameters,
    query_rows: Float[Array, " batch rows *tokens model"],
    n_keys: int | Array,
    positions: Positions | None = None,
) -> Float[Array, " batch rows heads head_dim"]:
    """Per-head queries: projected, then RoPE, the query norm and query scaling."""
    if parameters.rope is not None and isinstance(parameters.scale, QASSMaxParameters):
        raise ValueError("attention with both RoPE and QASSMax has no checked order")
    q = _split_heads(linear(parameters.query, query_rows), parameters.head_dim)
    if parameters.rope is not None:
        q = apply_rope(q, parameters.rope, positions)
    if parameters.query_norm is not None:
        q = rms_norm(parameters.query_norm, q)
    if parameters.scale is not None:
        scale = _token_heads(parameters.scale, math.prod(query_rows.shape[2:-1]))
        q = scale_queries(scale, q, n_keys)
    return q


def _output(
    parameters: AttentionParameters,
    query_rows: Float[Array, " batch rows *tokens model"],
    attended: Float[Array, " batch rows heads head_dim"],
) -> Float[Array, " batch rows *tokens output"]:
    """Attended heads through the gate and the output projection."""
    heads = attended.reshape(*attended.shape[:2], *query_rows.shape[2:-1], -1)
    if parameters.gate is not None:
        heads = heads * jnn.sigmoid(linear(parameters.gate, query_rows))
    if parameters.output is None:
        return heads
    return linear(parameters.output, heads)


def attend_streamed(
    parameters: AttentionParameters,
    query_rows: Float[Array, " batch rows model"],
    keys: Callable[[Array], tuple[AttentionCache, Bool[Array, " batch context"]]],
    steps: int,
    n_keys: int | Int[Array, " batch"],
) -> Float[Array, " batch rows output"]:
    """Attention to keys made a step at a time, as `attend` to all of them at once.

    A running softmax keeps each query's largest score, its sum of weights and
    its weighted values, so no step holds more than its own keys.

    Args:
        parameters: Attention parameters, without RoPE.
        query_rows: Normalized rows to project into queries.
        keys: Makes step `index`'s keys and values, from `project_kv` without a
            length, and which of them count; each key must count in one step only.
        steps: Steps to take.
        n_keys: Keys that count in all, for query scaling.

    Returns:
        Attention outputs after the output projection.
    """
    if parameters.rope is not None:
        raise NotImplementedError("streamed attention takes no RoPE")
    q = _queries(parameters, query_rows, n_keys)
    dtype = q.dtype if parameters.kernel.dtype is None else parameters.kernel.dtype
    batch, rows, heads, head_dim = q.shape
    shape = jax.eval_shape(keys, jnp.int32(0))[0].key.shape
    kv_heads = shape[-2]
    grouped = q.astype(dtype).reshape(
        batch, rows, kv_heads, heads // kv_heads, head_dim
    )
    lowest = jnp.finfo(jnp.float32).min

    def step(index: Array, carry: tuple[Array, Array, Array]) -> tuple:
        peak, total, weighted = carry
        cache, counts = keys(index)
        # Keys that do not count are zero, so whatever made them, even NaN,
        # cannot reach the sums through their zero weights.
        kept = counts[:, :, None, None]
        key = jnp.where(kept, cache.key, 0).astype(dtype)
        value = jnp.where(kept, cache.value, 0).astype(dtype)
        scores = jnp.einsum(
            "brkgd,bckd->bkgrc", grouped, key, preferred_element_type=jnp.float32
        )
        real = counts[:, None, None, None, :]
        scores = jnp.where(real, scores * head_dim**-0.5, lowest)
        new_peak = jnp.maximum(peak, scores.max(axis=-1))
        weights = jnp.where(real, jnp.exp(scores - new_peak[..., None]), 0)
        rescale = jnp.exp(peak - new_peak)
        attended = jnp.einsum(
            "bkgrc,bckd->bkgrd",
            weights.astype(dtype),
            value,
            preferred_element_type=jnp.float32,
        )
        return (
            new_peak,
            total * rescale + weights.sum(axis=-1),
            weighted * rescale[..., None] + attended,
        )

    scores = (batch, kv_heads, heads // kv_heads, rows)
    start = (
        jnp.full(scores, lowest),
        jnp.zeros(scores, jnp.float32),
        jnp.zeros((*scores, head_dim), jnp.float32),
    )
    _, total, weighted = jax.lax.fori_loop(0, steps, step, start)
    attended = weighted / jnp.maximum(total, jnp.finfo(jnp.float32).tiny)[..., None]
    attended = jnp.moveaxis(attended, 3, 1).reshape(batch, rows, heads, head_dim)
    return _output(parameters, query_rows, attended.astype(q.dtype))


def block_kv(
    parameters: AttentionBlockParameters,
    context_rows: Float[Array, " batch context model"],
    length: Int[Array, " batch"] | None = None,
) -> AttentionCache:
    """Projects normalized context rows to an attention block's keys and values.

    Args:
        parameters: Block parameters.
        context_rows: Context rows.
        length: Real rows at the start of each batch row, or None for all rows.

    Returns:
        The keys and values.
    """
    norm = parameters.context_norm
    if norm is None:
        norm = parameters.attention_norm
    return project_kv(parameters.attention, normalize(norm, context_rows), length)


def attention_block(
    parameters: AttentionBlockParameters,
    query_rows: Float[Array, " batch rows model"],
    cache: AttentionCache,
) -> Float[Array, " batch rows model"]:
    """Pre-norm attention to `cache`, then feed-forward, each with a residual.

    Self-attention is `attention_block(p, rows, block_kv(p, rows))`.

    Args:
        parameters: Block parameters.
        query_rows: Rows that attend.
        cache: Keys and values from `block_kv`.

    Returns:
        Updated query rows.
    """
    normed = normalize(parameters.attention_norm, query_rows)
    return _feed_forward(
        parameters, query_rows + attend(parameters.attention, normed, cache)
    )


def attention_block_streamed(
    parameters: AttentionBlockParameters,
    query_rows: Float[Array, " batch rows model"],
    keys: Callable[[Array], tuple[AttentionCache, Bool[Array, " batch context"]]],
    steps: int,
    n_keys: int | Int[Array, " batch"],
) -> Float[Array, " batch rows model"]:
    """`attention_block` to keys made a step at a time; see `attend_streamed`.

    Args:
        parameters: Block parameters.
        query_rows: Rows that attend.
        keys: Makes a step's keys and values, from `block_kv` without a
            length, and which of them count.
        steps: Steps to take.
        n_keys: Keys that count in all.

    Returns:
        Updated query rows.
    """
    normed = normalize(parameters.attention_norm, query_rows)
    attended = attend_streamed(parameters.attention, normed, keys, steps, n_keys)
    return _feed_forward(parameters, query_rows + attended)


def _feed_forward(
    parameters: AttentionBlockParameters, hidden: Float[Array, " batch rows model"]
) -> Float[Array, " batch rows model"]:
    """The block's feed-forward layer, with its residual."""
    return hidden + mlp(
        parameters.feed_forward, normalize(parameters.feed_forward_norm, hidden)
    )


def in_chunks(
    function: Callable[..., Array], values: Any, chunk: int | None, axis: int
) -> Array:
    """Applies `function` to `chunk` positions of `values` along `axis` at a time.

    `function` must treat positions along `axis` independently and keep their
    count. Positions are split into equal steps of at most `chunk`, each step's
    output is written into one result, and the last step overlaps the one
    before it rather than padding.

    Args:
        function: Maps slices of `values`, as positional arguments, to outputs.
        values: A tuple of arrays, or Nones, that share `axis`.
        chunk: Positions per step, or None for all at once.
        axis: The axis to chunk.

    Returns:
        `function(*values)`.
    """
    length = next(value for value in values if value is not None).shape[axis]
    if chunk is None or length <= chunk:
        return function(*values)
    steps = -(-length // chunk)
    # Equal steps, so the last overlaps the one before by fewer than `steps` positions.
    size = -(-length // steps)

    def piece(start: Any) -> tuple:
        return tuple(
            None
            if value is None
            else jax.lax.dynamic_slice_in_dim(value, start, size, axis)
            for value in values
        )

    shape = jax.eval_shape(function, *piece(0))
    result = jnp.zeros(
        (*shape.shape[:axis], length, *shape.shape[axis + 1 :]), shape.dtype
    )

    def step(index: Array, result: Array) -> Array:
        start = jnp.minimum(index * size, length - size)
        return jax.lax.dynamic_update_slice_in_dim(
            result, function(*piece(start)), start, axis
        )

    return jax.lax.fori_loop(0, steps, step, result)
