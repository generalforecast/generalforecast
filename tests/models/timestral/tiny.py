"""Small random Timestral models, so tests run without checkpoints."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.timestral.checkpoint import timestral_from_state
from generalforecast.models.timestral.model import Timestral

EMBED = 16
HEADS = 2
HIDDEN = 24
PATCH = 4
BLOCK_SIZE = 2
CONTEXT = 64
SCHEDULE = ("time", "time_variate")
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def tiny_checkpoint(seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """A small random checkpoint, as Mistral publishes them.

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

    def residual_block(prefix: str, inputs: int, outputs: int, bias: bool) -> None:
        linear(f"{prefix}.hidden_layer", 4 * EMBED, inputs, bias)
        linear(f"{prefix}.output_layer", outputs, 4 * EMBED, bias)
        linear(f"{prefix}.residual_layer", outputs, inputs, bias)

    residual_block("representation.patch_proj", 2 * PATCH, EMBED, True)
    arrays["representation.forecast_query"] = rng.normal(size=EMBED)
    for index, kind in enumerate(SCHEDULE):
        prefix = f"transformer.layers.{index}"
        for step in range(2 if "_" in kind else 1):
            arrays[f"{prefix}.attn_norms.{step}.scale"] = 1 + 0.1 * rng.normal(
                size=EMBED
            )
            linear(f"{prefix}.attns.{step}.wQKV", 3 * EMBED, EMBED)
            linear(f"{prefix}.attns.{step}.wO", EMBED, EMBED)
            linear(f"{prefix}.attns.{step}._gate", EMBED, EMBED)
        arrays[f"{prefix}.mlp_norm.scale"] = 1 + 0.1 * rng.normal(size=EMBED)
        linear(f"{prefix}.mlp.w12", 2 * HIDDEN, EMBED)
        linear(f"{prefix}.mlp.w3", EMBED, HIDDEN)
    arrays["transformer.out_norm.scale"] = 1 + 0.1 * rng.normal(size=EMBED)
    residual_block("head.projection", EMBED, PATCH * len(QUANTILES), False)
    state = {key: jnp.asarray(value, jnp.float32) for key, value in arrays.items()}
    settings = {
        "patch_size": PATCH,
        "d_model": EMBED,
        "num_heads": HEADS,
        "mlp_hidden_dim": HIDDEN,
        "block_schedule": list(SCHEDULE),
        "quantile_knots": list(QUANTILES),
        "block_size": BLOCK_SIZE,
        "context_length": CONTEXT,
        "rope_theta": 10000.0,
        "rope_partial_factor": 0.5,
        "norm_eps": 1e-6,
        "attn_bias": False,
        "mlp_bias": False,
        "time_attn_gate": "sdpa_sigmoid",
        "variate_attn_gate": "sdpa_sigmoid",
        "minimum_scale": 1e-6,
        "use_asinh": True,
        "stabilize_with_global": False,
        "covariate_enabled": True,
        "split_context_horizon_attn": True,
        "attend_missing_patches": True,
        "quantile_width_factor": 2.0,
    }
    return state, settings


def tiny_timestral(seed: int = 0, symmetric: bool = True) -> Timestral:
    """A small random Timestral.

    Args:
        seed: Seed for the weights.
        symmetric: Whether forecasts average over the group and its negation.

    Returns:
        The model.
    """
    return timestral_from_state(*tiny_checkpoint(seed), symmetric)


def forecaster() -> tuple[Any, Timestral]:
    """A small random Timestral with its contract calls, as a server's model spec."""
    from generalforecast.models.timestral.model import timestral_functions

    return timestral_functions, tiny_timestral()
