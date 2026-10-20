from pathlib import Path
from typing import Any

from jax import Array

from generalforecast.models.checkpoints import load_state, read_all, read_linear
from generalforecast.models.column_embedding import (
    ColumnEmbeddingParameters,
    FeatureGrouping,
)
from generalforecast.models.icl import ICLParameters
from generalforecast.models.row_interaction import RowInteractionParameters
from generalforecast.models.tabicl.model import TabICL
from generalforecast.nn.attention import AttentionBlockParameters, unpack_qkv
from generalforecast.nn.induced import InducedAttentionParameters
from generalforecast.nn.linear import MLPParameters
from generalforecast.nn.norm import LayerNormParameters
from generalforecast.nn.qassmax import QASSMaxParameters
from generalforecast.nn.rope import RoPE
from generalforecast.nn.target import ClassEncoder, TargetEncoder, ValueEncoder


def _norm(state: dict[str, Array], prefix: str, bias: bool) -> LayerNormParameters:
    """LayerNorm parameters stored under `prefix`, with a bias unless the model is bias-free."""
    return LayerNormParameters(
        weight=state[f"{prefix}.weight"],
        bias=state[f"{prefix}.bias"] if bias else None,
    )


def _qassmax(state: dict[str, Array], prefix: str) -> QASSMaxParameters:
    """QASSMax parameters stored under `prefix`."""
    return QASSMaxParameters(
        base=MLPParameters(
            input=read_linear(state, f"{prefix}.base_mlp.0"),
            output=read_linear(state, f"{prefix}.base_mlp.2"),
        ),
        query=MLPParameters(
            input=read_linear(state, f"{prefix}.query_mlp.0"),
            output=read_linear(state, f"{prefix}.query_mlp.2"),
        ),
    )


def _scaled(settings: dict[str, Any], name: str) -> bool:
    """Whether the config asks for QASSMax in a stage, as `col_ssmax` and `icl_ssmax` do."""
    value = settings[name]
    if value in (None, False, "none"):
        return False
    if value == "qassmax-mlp-elementwise":
        return True
    raise ValueError(f"unknown {name}: {value!r}")


def _attention_block(
    state: dict[str, Array],
    prefix: str,
    settings: dict[str, Any],
    num_heads: int,
    scaled: bool,
    rope: RoPE | None = None,
) -> AttentionBlockParameters:
    """Attention block parameters stored under `prefix`, with QASSMax if `scaled`."""
    bias = not settings["bias_free_ln"]
    return AttentionBlockParameters(
        attention_norm=_norm(state, f"{prefix}.norm1", bias),
        feed_forward_norm=_norm(state, f"{prefix}.norm2", bias),
        attention=unpack_qkv(
            qkv_weight=state[f"{prefix}.attn.in_proj_weight"],
            qkv_bias=state[f"{prefix}.attn.in_proj_bias"],
            output=read_linear(state, f"{prefix}.attn.out_proj"),
            num_heads=num_heads,
            scale=_qassmax(state, f"{prefix}.attn.ssmax_layer") if scaled else None,
            rope=rope,
        ),
        feed_forward=MLPParameters(
            input=read_linear(state, f"{prefix}.linear1"),
            output=read_linear(state, f"{prefix}.linear2"),
        ),
    )


def _feature_grouping(settings: dict[str, Any]) -> FeatureGrouping:
    """Feature grouping from a TabICL model config."""
    raw = settings["col_feature_group"]
    if raw in (False, None, "none"):
        return FeatureGrouping()
    if raw in (True, "same"):
        return FeatureGrouping("same", int(settings["col_feature_group_size"]))
    raise ValueError(f"unknown col_feature_group: {raw!r}")


def _columns(
    state: dict[str, Array], settings: dict[str, Any]
) -> ColumnEmbeddingParameters:
    """Column stage parameters."""
    prefix = "col_embedder.tf_col.blocks"
    num_heads = int(settings["col_nhead"])
    # Only the compressing attention of each block scales its queries.
    scaled = _scaled(settings, "col_ssmax")
    return ColumnEmbeddingParameters(
        input=read_linear(state, "col_embedder.in_linear"),
        blocks=tuple(
            InducedAttentionParameters(
                inducing=state[f"{prefix}.{index}.ind_vectors"],
                compress=_attention_block(
                    state,
                    f"{prefix}.{index}.multihead_attn1",
                    settings,
                    num_heads,
                    scaled,
                ),
                expand=_attention_block(
                    state,
                    f"{prefix}.{index}.multihead_attn2",
                    settings,
                    num_heads,
                    False,
                ),
            )
            for index in range(int(settings["col_num_blocks"]))
        ),
        target_encoder=_target_encoder(state, "col_embedder.y_encoder"),
        grouping=_feature_grouping(settings),
    )


def _rows(
    state: dict[str, Array], settings: dict[str, Any]
) -> RowInteractionParameters:
    """Row stage parameters, whose attention uses RoPE over feature positions."""
    prefix = "row_interactor.tf_row.blocks"
    num_heads = int(settings["row_nhead"])
    rope = RoPE(
        theta=float(settings["row_rope_base"]),
        interleaved=bool(settings["row_rope_interleaved"]),
    )
    return RowInteractionParameters(
        cls_tokens=state["row_interactor.cls_tokens"],
        blocks=tuple(
            _attention_block(
                state, f"{prefix}.{index}", settings, num_heads, False, rope
            )
            for index in range(int(settings["row_num_blocks"]))
        ),
        output_norm=_norm(state, "row_interactor.out_ln", not settings["bias_free_ln"]),
    )


def _target_encoder(state: dict[str, Array], prefix: str) -> TargetEncoder:
    """Value encoder if the layer takes one input, else a class encoder."""
    layer = read_linear(state, prefix)
    if layer.weight.shape[1] == 1:
        return ValueEncoder(linear=layer)
    return ClassEncoder(linear=layer)


def _icl(state: dict[str, Array], settings: dict[str, Any]) -> ICLParameters:
    """In-context learning stage parameters."""
    prefix = "icl_predictor.tf_icl.blocks"
    num_heads = int(settings["icl_nhead"])
    scaled = _scaled(settings, "icl_ssmax")
    return ICLParameters(
        blocks=tuple(
            _attention_block(state, f"{prefix}.{index}", settings, num_heads, scaled)
            for index in range(int(settings["icl_num_blocks"]))
        ),
        target_encoder=_target_encoder(state, "icl_predictor.y_encoder"),
        output_norm=_norm(state, "icl_predictor.ln", not settings["bias_free_ln"]),
        decoder=MLPParameters(
            input=read_linear(state, "icl_predictor.decoder.0"),
            output=read_linear(state, "icl_predictor.decoder.2"),
        ),
    )


def tabicl_from_state(state: dict[str, Array], settings: dict[str, Any]) -> TabICL:
    """Builds TabICL from a PyTorch state dict.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        settings: TabICL model config.

    Returns:
        The model.

    Raises:
        ValueError: If a parameter is missing or left unread.
    """
    return read_all(
        state,
        lambda read: TabICL(
            columns=_columns(read, settings),
            rows=_rows(read, settings),
            icl=_icl(read, settings),
        ),
        buffers=(".rope.freqs",),
    )


def load_tabicl(directory: Path) -> TabICL:
    """Loads a TabICL v2 checkpoint directory.

    Args:
        directory: Directory with `model.safetensors` and `config.json`.

    Returns:
        The model.
    """
    state, settings = load_state(directory, "tabicl_v2")
    if settings["col_affine"]:
        raise ValueError("col_affine checkpoints are not supported")
    if not settings["norm_first"]:
        raise ValueError("post-norm checkpoints are not supported")
    return tabicl_from_state(state, settings)
