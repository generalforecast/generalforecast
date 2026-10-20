from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jax import Array

from generalforecast.models.checkpoints import (
    load_state,
    read_all,
    read_linear,
    save_state,
)
from generalforecast.models.icl import ICLParameters
from generalforecast.models.kumo.cell_embedding import CellEmbeddingParameters
from generalforecast.models.kumo.model import KumoTabular
from generalforecast.models.kumo.row_embedding import RowEmbeddingParameters
from generalforecast.nn.attention import AttentionBlockParameters, unpack_qkv
from generalforecast.nn.induced import InducedAttentionParameters
from generalforecast.nn.linear import LinearParameters, MLPParameters
from generalforecast.nn.norm import RMSNormParameters
from generalforecast.nn.rope import RoPE
from generalforecast.nn.scaling import (
    GatedLogScaleParameters,
    LogScaleParameters,
    QueryScale,
)
from generalforecast.nn.target import ClassEncoder, TargetEncoder, ValueEncoder

# Row blocks rotate the halves of each head, with this base.
ROPE = RoPE(theta=100_000.0, interleaved=False)
# Per-head query and key norms have no scale and this epsilon.
HEAD_NORM = RMSNormParameters(weight=None, epsilon=1e-6)


def _norm(state: dict[str, Array], name: str) -> RMSNormParameters:
    """An RMSNorm scale stored under `name`."""
    return RMSNormParameters(weight=state[name])


# How each kind of block scales its queries: by the log of the key count, with
# a gate on the query as well, or not at all.
Scaling = Literal["log", "gated"] | None


def _scale(state: dict[str, Array], prefix: str, scaling: Scaling) -> QueryScale | None:
    """The query scaling stored under `prefix`, of the kind the block has."""
    if scaling is None:
        return None
    head_scale = state[f"{prefix}.head_scale"]
    if scaling == "log":
        return LogScaleParameters(head_scale=head_scale)
    gate = MLPParameters(
        input=read_linear(state, f"{prefix}.gate.0"),
        output=read_linear(state, f"{prefix}.gate.2"),
    )
    return GatedLogScaleParameters(head_scale=head_scale, gate=gate)


def _block(
    state: dict[str, Array],
    prefix: str,
    num_heads: int,
    scaling: Scaling,
    rope: RoPE | None = None,
) -> AttentionBlockParameters:
    """An attention block stored under `prefix`, in NVIDIA's layout."""
    qkv = read_linear(state, f"{prefix}.attn.qkv_lin")
    if qkv.bias is None:
        raise ValueError(f"{prefix} has no qkv bias")
    attention = unpack_qkv(
        qkv_weight=qkv.weight,
        qkv_bias=qkv.bias,
        output=read_linear(state, f"{prefix}.attn.out_lin"),
        num_heads=num_heads,
        scale=_scale(state, f"{prefix}.attn.sdpa.query_scaling", scaling),
        rope=rope,
        head_norm=HEAD_NORM,
    )
    return AttentionBlockParameters(
        attention_norm=_norm(state, f"{prefix}.query_norm.weight"),
        feed_forward_norm=_norm(state, f"{prefix}.mlp.0.weight"),
        attention=attention,
        feed_forward=MLPParameters(
            input=read_linear(state, f"{prefix}.mlp.1"),
            output=read_linear(state, f"{prefix}.mlp.3"),
        ),
        context_norm=_norm(state, f"{prefix}.key_value_norm.weight"),
    )


def _target_encoder(state: dict[str, Array], prefix: str) -> TargetEncoder:
    """A label embedding table as a class encoder, or a scalar target projection."""
    if f"{prefix}.y_emb.weight" in state:
        table = state[f"{prefix}.y_emb.weight"]
        return ClassEncoder(linear=LinearParameters(weight=table.T, bias=None))
    return ValueEncoder(linear=read_linear(state, f"{prefix}.y_lin", bias=False))


def _rows(state: dict[str, Array], settings: dict[str, Any]) -> RowEmbeddingParameters:
    """The row encoder."""
    prefix = "row_embedding"
    heads = int(settings["num_embedding_heads"])
    layers = range(int(settings["num_embedding_layers"]))
    cells = f"{prefix}.cell_embedding"
    return RowEmbeddingParameters(
        cells=CellEmbeddingParameters(
            frequencies=state[f"{cells}.num_freq"],
            linear=read_linear(state, f"{cells}.num_lin"),
            missing=read_linear(state, f"{cells}.nan_lin", bias=False),
            categorical_frequencies=state[f"{cells}.cat_freq"],
            categorical_linear=read_linear(state, f"{cells}.cat_lin"),
        ),
        target_encoder=_target_encoder(state, prefix),
        readout=state[f"{prefix}.readout_token"],
        # Compressing into inducing points scales by the key count; expanding
        # back does not; row blocks gate the scale on the query as well.
        columns=tuple(
            InducedAttentionParameters(
                inducing=state[f"{prefix}.col_blocks.{layer}.inducing_points"],
                compress=_block(
                    state, f"{prefix}.col_blocks.{layer}.inducing_block", heads, "log"
                ),
                expand=_block(
                    state, f"{prefix}.col_blocks.{layer}.output_block", heads, None
                ),
            )
            for layer in layers
        ),
        rows=tuple(
            _block(state, f"{prefix}.row_blocks.{layer}", heads, "gated", ROPE)
            for layer in layers
        ),
        output_norm=_norm(state, f"{prefix}.norm.weight"),
    )


def _icl(state: dict[str, Array], settings: dict[str, Any]) -> ICLParameters:
    """The in-context learning stage."""
    heads = int(settings["num_icl_heads"])
    kv_heads = settings.get("num_icl_key_value_heads_for_query")
    return ICLParameters(
        blocks=tuple(
            _block(state, f"icl_block.layers.{layer}", heads, "log")
            for layer in range(int(settings["num_icl_layers"]))
        ),
        target_encoder=_target_encoder(state, "icl_block"),
        output_norm=_norm(state, "icl_block.norm.weight"),
        decoder=MLPParameters(
            input=read_linear(state, "icl_block.head.0"),
            output=read_linear(state, "icl_block.head.2"),
        ),
        query_kv_heads=None if kv_heads is None else int(kv_heads),
    )


def kumo_from_state(state: dict[str, Array], settings: dict[str, Any]) -> KumoTabular:
    """Builds Kumo Tabular from NVIDIA's PyTorch state dict.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        settings: Model sizes, named like NVIDIA's `MODEL_KWARGS`.

    Returns:
        The model.

    Raises:
        ValueError: If a parameter is missing or left unread.
    """

    def build(read: dict[str, Array]) -> KumoTabular:
        # The projection exists only when the readout width differs from the ICL width.
        readout = int(settings["num_readout_tokens"]) * int(settings["cell_channels"])
        project = None
        if readout != int(settings["icl_channels"]):
            project = read_linear(read, "row_project")
        return KumoTabular(
            rows=_rows(read, settings), project=project, icl=_icl(read, settings)
        )

    return read_all(state, build, buffers=(".inv_freq",))


ARCHITECTURE = "kumo_tabular"


def load_kumo(directory: Path) -> KumoTabular:
    """Loads a Kumo Tabular checkpoint directory.

    Args:
        directory: Directory with `model.safetensors` and a `config.json` whose
            settings are named like NVIDIA's `MODEL_KWARGS`.

    Returns:
        The model.
    """
    return kumo_from_state(*load_state(directory, ARCHITECTURE))


def save_kumo(
    directory: Path, state: Mapping[str, np.ndarray], settings: Mapping[str, Any]
) -> None:
    """Writes NVIDIA's PyTorch state dict, as NumPy arrays, as a checkpoint directory.

    Args:
        directory: Directory to write.
        state: Arrays keyed by PyTorch parameter name.
        settings: The model's sizes, as in NVIDIA's `MODEL_KWARGS`.
    """
    save_state(directory, ARCHITECTURE, settings, state)
