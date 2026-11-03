# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import numpy as np
from jax import Array

from generalforecast.models.checkpoints import load_state, read_all, read_linear
from generalforecast.models.tabfm.model import (
    ColumnStage,
    FourierCells,
    RowStage,
    TabFM,
)
from generalforecast.nn.attention import AttentionBlockParameters, AttentionParameters
from generalforecast.nn.induced import InducedAttentionParameters
from generalforecast.nn.linear import GatedMLPParameters, MLPParameters
from generalforecast.nn.norm import RMSNormParameters
from generalforecast.nn.rope import RoPE
from generalforecast.nn.scaling import SoftplusScaleParameters
from generalforecast.nn.target import ClassEncoder

# The architecture name a converted TabFM checkpoint's `config.json` carries.
ARCHITECTURE = "tabfm_v1"
# TabFM's RMSNorms all take this epsilon.
_EPSILON = 1e-6


def _norm(state: dict[str, Array], prefix: str) -> RMSNormParameters:
    return RMSNormParameters(weight=state[f"{prefix}.weight"], epsilon=_EPSILON)


def _block(
    state: dict[str, Array], prefix: str, num_heads: int, rope: RoPE | None = None
) -> AttentionBlockParameters:
    """A sandwich block: normalized attention and SwiGLU, each output normalized before it is added."""
    attention = f"{prefix}.attn"
    return AttentionBlockParameters(
        attention_norm=_norm(state, f"{prefix}.pre_attn_ln"),
        feed_forward_norm=_norm(state, f"{prefix}.pre_ff_ln"),
        attention=AttentionParameters(
            query=read_linear(state, f"{attention}.q_proj"),
            key=read_linear(state, f"{attention}.k_proj"),
            value=read_linear(state, f"{attention}.v_proj"),
            output=read_linear(state, f"{attention}.out_proj"),
            num_heads=num_heads,
            scale=SoftplusScaleParameters(state[f"{attention}.per_dim_scale"]),
            query_norm=_norm(state, f"{attention}.query_ln"),
            key_norm=_norm(state, f"{attention}.key_ln"),
            rope=rope,
        ),
        feed_forward=GatedMLPParameters(
            gate=read_linear(state, f"{prefix}.linear1_gate"),
            up=read_linear(state, f"{prefix}.linear1"),
            down=read_linear(state, f"{prefix}.linear2"),
        ),
        attention_output_norm=_norm(state, f"{prefix}.post_attn_ln"),
        feed_forward_output_norm=_norm(state, f"{prefix}.post_ff_ln"),
    )


def _columns(
    state: dict[str, Array], prefix: str, settings: dict[str, Any]
) -> ColumnStage:
    heads = int(settings["col_nhead"])
    blocks = f"{prefix}.tf_col.blocks"
    return ColumnStage(
        blocks=tuple(
            InducedAttentionParameters(
                inducing=state[f"{blocks}.{index}.ind_vectors"],
                compress=_block(state, f"{blocks}.{index}.mab1", heads),
                expand=_block(state, f"{blocks}.{index}.mab2", heads),
            )
            for index in range(int(settings["col_num_blocks"]))
        ),
        output=read_linear(state, f"{prefix}.out_w"),
        output_norm=_norm(state, f"{prefix}.ln_w"),
    )


def _rows(state: dict[str, Array], prefix: str, settings: dict[str, Any]) -> RowStage:
    """A row stage, whose blocks share the RoPE frequencies the checkpoint stores.

    TabFM computed them in bfloat16 when it trained, so they are read rather
    than computed again from the base.
    """
    frequencies = np.asarray(state[f"{prefix}.tf_row.rope.freqs"], dtype=np.float32)
    rope = RoPE(
        theta=100_000.0,
        interleaved=True,
        frequencies=tuple(float(value) for value in frequencies),
    )
    heads = int(settings["row_nhead"])
    return RowStage(
        blocks=tuple(
            _block(state, f"{prefix}.tf_row.blocks.{index}", heads, rope)
            for index in range(int(settings["row_num_blocks"]))
        ),
        output_norm=_norm(state, f"{prefix}.out_ln"),
    )


def tabfm_from_state(state: dict[str, Array], settings: dict[str, Any]) -> TabFM:
    """Builds TabFM's classifier from a PyTorch state dict.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        settings: TabFM's model config.

    Returns:
        The model.

    Raises:
        ValueError: If the checkpoint is a regressor's, or a parameter is
            missing or left unread.
    """
    if not settings["is_classifier"]:
        raise ValueError("TabFM's regressor predicts one value, not quantiles")

    def build(read: dict[str, Array]) -> TabFM:
        cells = "cell_embedder"
        heads = int(settings["icl_nhead"])
        return TabFM(
            cells=FourierCells(
                frequencies=read[f"{cells}.fourier_frequencies"],
                numerical=read_linear(read, f"{cells}.in_linear"),
                categorical_frequencies=read[f"{cells}.fourier_frequencies_cat"],
                categorical=read_linear(read, f"{cells}.in_linear_cat"),
            ),
            cell_targets=read[f"{cells}.y_embedder_lookup.weight"],
            columns=(
                _columns(read, "col_embedder", settings),
                _columns(read, "col_embedder_2", settings),
            ),
            cls_tokens=read["cls_tokens"],
            rows=(
                _rows(read, "row_interactor", settings),
                _rows(read, "row_interactor_2", settings),
            ),
            icl_blocks=tuple(
                _block(read, f"icl_predictor.tf_icl.blocks.{index}", heads)
                for index in range(int(settings["icl_num_blocks"]))
            ),
            icl_targets=ClassEncoder(
                linear=read_linear(read, "icl_predictor.y_encoder.projection")
            ),
            icl_norm=_norm(read, "icl_predictor.ln"),
            decoder=MLPParameters(
                input=read_linear(read, "icl_predictor.decoder.layers.0"),
                output=read_linear(read, "icl_predictor.decoder.layers.1"),
            ),
            group_size=int(settings["feature_group_size"]),
        )

    return read_all(state, build)


def load_tabfm(directory: Path) -> TabFM:
    """Loads a converted TabFM classifier checkpoint directory.

    Args:
        directory: Directory with `model.safetensors` and `config.json`.

    Returns:
        The model.
    """
    state, settings = load_state(directory, ARCHITECTURE)
    return tabfm_from_state(state, settings)
