"""Small random TimesFM 3 models, so tests run without checkpoints."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.timesfm3.checkpoint import timesfm_from_state
from generalforecast.models.timesfm3.model import TimesFM

EMBED = 16
HEADS = 2
LAYERS = 2
PATCH = 4
OUTPUT_PATCH = 8
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def tiny_checkpoint(seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """A small random checkpoint, as Google publishes them.

    Args:
        seed: Seed for the weights.

    Returns:
        The state dict and settings.
    """
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}

    def linear(prefix: str, outputs: int, inputs: int, bias: bool = False) -> None:
        arrays[f"{prefix}.weight"] = rng.normal(size=(outputs, inputs)) / math.sqrt(
            inputs
        )
        if bias:
            arrays[f"{prefix}.bias"] = 0.1 * rng.normal(size=outputs)

    def norm(prefix: str, size: int = EMBED) -> None:
        arrays[f"{prefix}.weight"] = 1 + 0.1 * rng.normal(size=size)

    def attention(prefix: str) -> None:
        for name in ("query", "key", "value", "out"):
            linear(f"{prefix}.{name}_proj", EMBED, EMBED)
        norm(f"{prefix}.query_ln", EMBED // HEADS)
        norm(f"{prefix}.key_ln", EMBED // HEADS)
        arrays[f"{prefix}.per_dim_scale.per_dim_scale"] = 0.1 * rng.normal(
            size=EMBED // HEADS
        )

    inputs = 2 * (PATCH + OUTPUT_PATCH)
    linear("pre_transformer_resblock.hidden_layer", EMBED, inputs)
    linear("pre_transformer_resblock.output_layer", EMBED, EMBED)
    linear("pre_transformer_resblock.residual_layer", EMBED, inputs)
    for index in range(LAYERS):
        prefix = f"transformer_stack.layers.{index}"
        for name in ("pre_seq_attn_ln", "post_seq_attn_ln", "pre_var_attn_ln"):
            norm(f"{prefix}.{name}")
        for name in ("post_var_attn_ln", "pre_ff_ln", "post_ff_ln"):
            norm(f"{prefix}.{name}")
        attention(f"{prefix}.seq_attn")
        attention(f"{prefix}.var_attn")
        linear(f"{prefix}.ff0", EMBED, EMBED)
        linear(f"{prefix}.ff1", EMBED, EMBED)
    linear("output_head", OUTPUT_PATCH * len(QUANTILES), EMBED, bias=True)
    state = {key: jnp.asarray(value, jnp.float32) for key, value in arrays.items()}
    settings = {
        "input_patch_len": PATCH,
        "output_patch_len": OUTPUT_PATCH,
        "quantiles": list(QUANTILES),
        "residual_block_config": {
            "activation": "relu",
            "dropout": 0.0,
            "hidden_dims": EMBED,
            "identity_skip": False,
            "output_dims": EMBED,
            "prenorm": "none",
            "use_bias": False,
        },
        "transformer_config": {
            "num_layers": LAYERS,
            "transformer": {
                "attention_norm": "rms",
                "causal_attention": True,
                "debug_no_masking": False,
                "deterministic": True,
                "feedforward_norm": "rms",
                "ff_activation": "relu",
                "hidden_dims": EMBED,
                "max_variates": 32,
                "model_dims": EMBED,
                "num_heads": HEADS,
                "paired_token_skip_second": False,
                "qk_norm": "rms",
                "training": True,
                "use_bias": False,
                "use_memory_efficient_attention": True,
                "use_rope_seq": True,
                "use_rope_var": False,
                "use_sdpa": True,
                "v_norm": "none",
            },
            "use_remat": True,
        },
        "input_transform": "identity",
        "linear_detrending_threshold": 0.5,
        "use_frozen_running_stats": False,
        "use_iterative_cpm_revin": True,
        "use_linear_detrending": True,
        "use_stitching": True,
        "use_variate_attention": True,
        "value_clip": 1e20,
    }
    return state, settings


def tiny_timesfm(seed: int = 0) -> TimesFM:
    """A small random TimesFM 3.

    Args:
        seed: Seed for the weights.

    Returns:
        The model.
    """
    return timesfm_from_state(*tiny_checkpoint(seed))


def forecaster() -> tuple[Any, TimesFM]:
    """A tiny TimesFM as a model spec target: `tests.models.timesfm3.tiny:forecaster`."""
    from generalforecast.models.timesfm3.model import timesfm_functions

    return timesfm_functions, tiny_timesfm()
