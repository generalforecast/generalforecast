# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small random Chronos-2 models, so tests run without checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.chronos2.checkpoint import chronos2_from_state
from generalforecast.models.chronos2.model import Chronos2

EMBED = 16
HEADS = 2
FF = 32
LAYERS = 2
PATCH = 4
CONTEXT = 64
MAX_OUTPUT_PATCHES = 4
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def tiny_checkpoint(
    seed: int = 0, patch: int = PATCH, quantiles: Sequence[float] = QUANTILES
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A small random checkpoint, as Amazon publishes them.

    Args:
        seed: Seed for the weights.
        patch: Input and output patch length.
        quantiles: Quantile levels it forecasts.

    Returns:
        The state dict and settings.
    """
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}

    def linear(prefix: str, outputs: int, inputs: int, bias: bool = True) -> None:
        arrays[f"{prefix}.weight"] = rng.normal(size=(outputs, inputs)) / math.sqrt(
            inputs
        )
        if bias:
            arrays[f"{prefix}.bias"] = 0.1 * rng.normal(size=outputs)

    def norm(prefix: str) -> None:
        arrays[f"{prefix}.weight"] = 1 + 0.1 * rng.normal(size=EMBED)

    def residual_block(prefix: str, inputs: int, outputs: int) -> None:
        linear(f"{prefix}.hidden_layer", FF, inputs)
        linear(f"{prefix}.output_layer", outputs, FF)
        linear(f"{prefix}.residual_layer", outputs, inputs)

    arrays["shared.weight"] = rng.normal(size=(2, EMBED))
    residual_block("input_patch_embedding", 3 * patch, EMBED)
    for index in range(LAYERS):
        prefix = f"encoder.block.{index}.layer"
        for step in (0, 1):
            norm(f"{prefix}.{step}.layer_norm")
            for name in "qkvo":
                linear(f"{prefix}.{step}.self_attention.{name}", EMBED, EMBED, False)
        norm(f"{prefix}.2.layer_norm")
        linear(f"{prefix}.2.mlp.wi", FF, EMBED, False)
        linear(f"{prefix}.2.mlp.wo", EMBED, FF, False)
    norm("encoder.final_layer_norm")
    residual_block("output_patch_embedding", EMBED, len(quantiles) * patch)
    state = {key: jnp.asarray(value, jnp.float32) for key, value in arrays.items()}
    settings = {
        "model_type": "t5",
        "feed_forward_proj": "relu",
        "dense_act_fn": "relu",
        "is_gated_act": False,
        "d_model": EMBED,
        "d_kv": EMBED // HEADS,
        "d_ff": FF,
        "num_layers": LAYERS,
        "num_heads": HEADS,
        "rope_theta": 10000.0,
        "layer_norm_epsilon": 1e-6,
        "dropout_rate": 0.1,
        "chronos_config": {
            "context_length": CONTEXT,
            "input_patch_size": patch,
            "input_patch_stride": patch,
            "output_patch_size": patch,
            "max_output_patches": MAX_OUTPUT_PATCHES,
            "quantiles": list(quantiles),
            "time_encoding_scale": CONTEXT,
            "use_reg_token": True,
            "use_arcsinh": True,
        },
    }
    return state, settings


def tiny_chronos2(seed: int = 0) -> Chronos2:
    """A small random Chronos-2.

    Args:
        seed: Seed for the weights.

    Returns:
        The model.
    """
    return chronos2_from_state(*tiny_checkpoint(seed))


def forecaster() -> tuple[Any, Chronos2]:
    """A tiny Chronos-2 as a model spec target: `tests.models.chronos2.tiny:forecaster`."""
    from generalforecast.models.chronos2.model import chronos2_functions

    return chronos2_functions, tiny_chronos2()


def catalog_forecaster() -> tuple[Any, Chronos2]:
    """A tiny Chronos-2 shaped as the catalog's: its patch and quantile levels.

    The model spec target `tests.models.chronos2.tiny:catalog_forecaster`
    stands in for `chronos-2`, whose published capabilities it matches.
    """
    from generalforecast.models import catalog
    from generalforecast.models.chronos2.model import chronos2_functions

    quantiles = catalog.ENTRIES["chronos-2"].capability.quantiles
    return chronos2_functions, chronos2_from_state(
        *tiny_checkpoint(patch=16, quantiles=quantiles)
    )
