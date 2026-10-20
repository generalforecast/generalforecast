from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from generalforecast.nn.attention import (
    AttentionBlockParameters,
    attention_block,
    block_kv,
    in_chunks,
)
from generalforecast.nn.norm import Norm, normalize


class RowInteractionParameters(NamedTuple):
    """CLS tokens, attention blocks and output norm of the row stage."""

    cls_tokens: Float[Array, " cls model"]
    blocks: tuple[AttentionBlockParameters, ...]
    output_norm: Norm


def interact_rows(
    parameters: RowInteractionParameters,
    embeddings: Float[Array, " batch rows features model"],
    count: Int[Array, " batch"] | None = None,
) -> Float[Array, " batch rows row_model"]:
    """Embeds each row by attending across its cells and reading out CLS tokens.

    Args:
        parameters: Row stage parameters.
        embeddings: Cell embeddings from the column stage.
        count: Cells each row attends to, which come first, per batch row, as
            if the rest were dropped; None for all.

    Returns:
        One embedding per row, the concatenated CLS outputs.
    """
    batch, rows, features, model_dim = embeddings.shape
    num_cls = parameters.cls_tokens.shape[0]
    cls = jnp.broadcast_to(parameters.cls_tokens, (batch, rows, num_cls, model_dim))
    tokens = jnp.concatenate([cls, embeddings], axis=2)
    tokens = tokens.reshape(batch * rows, num_cls + features, model_dim)
    # Keys past the kept cells are masked; RoPE numbers the rest as before.
    length = None if count is None else jnp.repeat(num_cls + count, rows)

    def interact(hidden: Array, length: Array | None) -> Array:
        *early, last = parameters.blocks
        for block in early:
            hidden = attention_block(block, hidden, block_kv(block, hidden, length))
        cache = block_kv(last, hidden, length)
        output = attention_block(last, hidden[:, :num_cls, :], cache)
        return normalize(parameters.output_norm, output)

    # Rows attend only within themselves, so they can go a chunk at a time.
    chunk = parameters.blocks[0].attention.kernel.block_chunk
    per_step = None if chunk is None else max(1, chunk // tokens.shape[1])
    cls_output = in_chunks(interact, (tokens, length), per_step, axis=0)
    return cls_output.reshape(batch, rows, num_cls * model_dim)
