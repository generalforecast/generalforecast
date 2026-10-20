"""Small random LimiX models, so tests run without checkpoints."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.limix.checkpoint import limix_from_state, one_task
from generalforecast.models.limix.model import LimiX

EMBED = 16
HEADS = 2
TOKENS = 2
GROUP = 2
LAYERS = 2
CLASS_SLOTS = 10
BUCKETS = 50


class _StateBuilder:
    """Random arrays under LimiX-2's PyTorch parameter names."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.arrays: dict[str, np.ndarray] = {}

    def normal(self, name: str, shape: tuple[int, ...], scale: float) -> None:
        self.arrays[name] = scale * self.rng.normal(size=shape)

    def linear(self, prefix: str, outputs: int, inputs: int) -> None:
        self.normal(f"{prefix}.weight", (outputs, inputs), 1 / math.sqrt(inputs))
        self.normal(f"{prefix}.bias", (outputs,), 0.1)

    def norm(self, name: str, dim: int) -> None:
        self.arrays[f"{name}.weight"] = 1 + 0.1 * self.rng.normal(size=dim)

    def scaling(self, prefix: str, heads: int) -> None:
        self.normal(f"{prefix}.scale_linear.weight", (heads, 1), 0.3)
        self.normal(f"{prefix}.scale_linear.bias", (heads,), 0.3)

    def attention(self, prefix: str, dim: int, heads: int) -> None:
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            self.linear(f"{prefix}.{name}", dim, dim)
        self.norm(f"{prefix}.q_norm", dim // heads)
        self.norm(f"{prefix}.k_norm", dim // heads)
        self.scaling(f"{prefix}.softmax_scaling_mlp", heads)

    def gated(self, prefix: str, dim: int) -> None:
        self.linear(f"{prefix}.gate_proj", 2 * dim, dim)
        self.linear(f"{prefix}.up_proj", 2 * dim, dim)
        self.linear(f"{prefix}.down_proj", dim, 2 * dim)

    def within_row(self, prefix: str) -> None:
        for name in (
            "sample_x_q_proj",
            "sample_all_k_proj",
            "sample_all_v_proj",
            "sample_x_out_proj",
            "yx_y_q_proj",
            "yx_x_k_proj",
            "yx_x_v_proj",
        ):
            self.linear(f"{prefix}.{name}", EMBED, EMBED)
        self.linear(f"{prefix}.yx_out_proj", TOKENS * EMBED, TOKENS * EMBED)
        for name in ("sample_x_q_norm", "sample_all_k_norm", "yx_q_norm", "yx_k_norm"):
            self.norm(f"{prefix}.{name}", EMBED // HEADS)
        self.scaling(f"{prefix}.sample_x_softmax_scaling", HEADS)
        self.scaling(f"{prefix}.yx_softmax_scaling", HEADS)


def _settings() -> dict[str, Any]:
    """The published config's settings, at the tiny model's sizes."""
    layer = {"arch": "smf", "pre_norm": False, "all_norm": True}
    return {
        "arch_version": "2.0",
        "layer_arch": "smf",
        "cls_only_start_layer": -1,
        "pre_norm": False,
        "all_norm": True,
        "tf_norm_type": "rmsnorm",
        "tf_layer_norm_use_elementwise_affine": False,
        "tf_mlp_layer_type": "gated",
        "tf_mlp_activation_fuction": "silu",
        "use_gated_norm": False,
        "separate_x_y_ffn": True,
        "use_separate_attention": True,
        "separate_attn_kv_combined": False,
        "sample_attention_cls_token_merge": "reshape",
        "scale_y_sample_attention_heads": True,
        "feature_attention_cls_token_merge": "none",
        "seq_att_use_softmax_scaling_mlp_for_q": True,
        "fea_att_use_softmax_scaling_mlp_for_q": True,
        "seq_att_use_q_norm": True,
        "seq_att_use_k_norm": True,
        "fea_att_use_q_norm": True,
        "fea_att_use_k_norm": True,
        "self_share_all_kv_heads": False,
        "cross_share_all_kv_heads": True,
        "seq_attn_isolated": False,
        "seq_attn_serial": False,
        "mask_feature_filling_type": "mask_token",
        "feature_positional_embedding_type": "subspace",
        "add_task_type": True,
        "learnable_borders": False,
        "reg_borders_use_train_quantile": False,
        "cls_y_decoder_type": "Free_MLP",
        "reg_y_decoder_type": "Free_MLP",
        "use_cls_post_adapter": True,
        "use_cls_post_adapter_residual": False,
        "use_reg_post_adapter": True,
        "use_reg_post_adapter_residual": False,
        "cls_post_adapter_pattern": "linear-4_GELU_linear-16",
        "reg_post_adapter_pattern": "linear-4_GELU_linear-16",
        "cls_y_decoder_mlp_pattern": "linear-16_GELU_linear-16_GELU_linear-10",
        "reg_y_decoder_mlp_pattern": "linear-16_GELU_linear-50",
        "preprocess_config_x": {
            "nan_handling_enabled": True,
            "normalize_on_train_only": True,
            "normalize_x": True,
            "remove_outliers": False,
            "normalize_by_used_features": True,
        },
        "encoder_config_x": {
            "feature_embedding_type": "mask_embedding",
            "numeric_embed_type": "linear",
            "use_feature_group_encoder": True,
            "feature_fusion_network_type": "original",
            "use_feature_mask_indicator": False,
            "mask_token_emb_type": "uniform_mask_emb_type",
        },
        "encoder_config_y": {
            "cls_y_encoder_type": "emby_embedding",
            "y_encoder_type": "none_embedding",
            "reg_y_numeric_embed_type": "MLP_GELU_postnorm",
            "cls_emb_learning_scalar": False,
            "mask_token_emb_type": "uniform_mask_emb_type",
        },
        "nlayers": LAYERS,
        "nhead": HEADS,
        "num_cls_tokens": TOKENS,
        "features_per_group": GROUP,
        "sa_temp_lower_bound": 0.0,
        "sa_temp_upper_bound": 0.5,
        "sa_scale_base_bound": 5.0,
        "fa_temp_lower_bound": 0.0,
        "fa_temp_upper_bound": 0.4,
        "fa_scale_base_bound": 1.0,
        "model_structure_config": {"layers": [layer] * LAYERS},
    }


def tiny_checkpoint(seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """A small random LimiX checkpoint for both tasks, as published.

    Args:
        seed: Seed for the weights.

    Returns:
        The state dict and settings.
    """
    builder = _StateBuilder(seed)
    width = TOKENS * EMBED
    builder.normal("encoder_x.0.mask_embedding.weight", (1, EMBED), 1.0)
    builder.linear("encoder_x.0.numeric_mlp.0", EMBED // 2, 1)
    builder.norm("encoder_x.0.numeric_mlp.1", EMBED // 2)
    builder.linear("encoder_x.0.numeric_mlp.3", EMBED, EMBED // 2)
    builder.norm("encoder_x.0.numeric_mlp.4", EMBED)
    builder.linear("encoder_x.0.fusion_network.0", EMBED, GROUP * EMBED)
    builder.norm("encoder_x.0.fusion_network.1", EMBED)
    builder.linear("encoder_x.0.fusion_network.3", EMBED, EMBED)
    builder.norm("encoder_x.0.fusion_network.4", EMBED)
    builder.linear("feature_positional_embedding", EMBED, EMBED // 4)
    builder.normal("cls_y_encoder.2.y_embedding.weight", (CLASS_SLOTS, width), 1.0)
    builder.normal("cls_y_encoder.2.y_mask.weight", (1, width), 1.0)
    builder.normal("reg_y_encoder.0.none_embedding.weight", (1, width), 1.0)
    builder.linear("reg_y_encoder.0.numeric_encoder.0", width // 2, 1)
    builder.linear("reg_y_encoder.0.numeric_encoder.2", width, width // 2)
    builder.norm("reg_y_encoder.0.numeric_encoder.3", width)
    builder.normal("add_task_info.token_type_embedding.weight", (3, EMBED), 1.0)
    for index in range(LAYERS):
        prefix = f"transformer_encoder.layers.{index}"
        sample = f"{prefix}.sequence_attentions.0"
        builder.attention(f"{sample}.x_attention", EMBED, HEADS)
        builder.attention(f"{sample}.y_attention", width, HEADS * TOKENS)
        builder.gated(f"{prefix}.mlp.0.x_ffn", EMBED)
        builder.gated(f"{prefix}.mlp.0.y_ffn", width)
        builder.within_row(f"{prefix}.feature_attentions.0")
    for task in ("cls", "reg"):
        builder.linear(f"{task}_post_adapter.adapter.0", 4, width)
        builder.linear(f"{task}_post_adapter.adapter.2", width, 4)
        builder.norm(f"{task}_post_adapter.layer_norm", width)
    builder.linear("cls_y_decoder.0", 16, width)
    builder.linear("cls_y_decoder.2", 16, 16)
    builder.linear("cls_y_decoder.4", CLASS_SLOTS, 16)
    builder.linear("reg_y_decoder.0", 16, width)
    builder.linear("reg_y_decoder.2", BUCKETS, 16)
    builder.arrays["_reg_borders"] = np.linspace(-4, 4, BUCKETS + 1)
    builder.normal("reg_log_widths", (BUCKETS,), 1.0)
    builder.linear("feature_decoder.0", EMBED, EMBED)
    builder.linear("feature_post_adapter.adapter.0", 4, EMBED)
    state = {
        key: jnp.asarray(value, dtype=jnp.float32)
        for key, value in builder.arrays.items()
    }
    return state, _settings()


def tiny_limix(*, regression: bool, seed: int = 0) -> LimiX:
    """A small random LimiX with LimiX-2's structure, for one task.

    Args:
        regression: Whether to build a regressor instead of a classifier.
        seed: Seed for the weights.

    Returns:
        The model.
    """
    state, settings = tiny_checkpoint(seed)
    return limix_from_state(one_task(state, regression), settings)
