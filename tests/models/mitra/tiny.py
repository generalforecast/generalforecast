"""Small random Mitra models, so tests run without checkpoints."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.mitra.checkpoint import mitra_from_state
from generalforecast.models.mitra.model import Mitra

EMBED = 16
HEADS = 2
LAYERS = 2
CLASS_SLOTS = 10
BUCKETS = 32


def tiny_checkpoint(
    seed: int = 0, regression: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A small random checkpoint, as AutoGluon publishes it.

    Args:
        seed: Seed for the weights.
        regression: Whether to build a regressor over `BUCKETS` buckets, as
            Mitra-2's, instead of a classifier.

    Returns:
        The state dict and settings.
    """
    rng = np.random.default_rng(seed)
    slots = BUCKETS if regression else CLASS_SLOTS
    arrays: dict[str, np.ndarray] = {}

    def linear(prefix: str, outputs: int, inputs: int) -> None:
        arrays[f"{prefix}.weight"] = rng.normal(size=(outputs, inputs)) / math.sqrt(
            inputs
        )
        arrays[f"{prefix}.bias"] = 0.1 * rng.normal(size=outputs)

    def norm(prefix: str) -> None:
        arrays[f"{prefix}.weight"] = 1 + 0.1 * rng.normal(size=EMBED)
        arrays[f"{prefix}.bias"] = 0.1 * rng.normal(size=EMBED)

    linear("x_embedding.x_embedding", EMBED, 1)
    arrays["y_embedding.y_embedding.weight"] = rng.normal(size=(slots, EMBED))
    arrays["y_embedding.y_mask.weight"] = rng.normal(size=(1, EMBED))
    for index in range(LAYERS):
        prefix = f"layers.{index}"
        for step in (1, 2):
            for name in "qkvo":
                linear(f"{prefix}.attention{step}.{name}", EMBED, EMBED)
        for step in (1, 2, 3, 4):
            norm(f"{prefix}.layer_norm{step}")
        for first in (1, 3):
            linear(f"{prefix}.linear{first}", 4 * EMBED, EMBED)
            linear(f"{prefix}.linear{first + 1}", EMBED, 4 * EMBED)
    norm("final_layer_norm")
    linear("final_layer", slots, EMBED)
    if regression:
        # Mass on the middle buckets, as trained heads put it: quantiles divide
        # by a bucket's mass, which would turn float32 rounding in nearly empty
        # buckets into large differences.
        arrays["final_layer.weight"] *= 0.1
        center = (np.arange(slots) - (slots - 1) / 2) / (slots / 8)
        arrays["final_layer.bias"] = -0.5 * center**2
    state = {key: jnp.asarray(value, jnp.float32) for key, value in arrays.items()}
    settings = {
        "dim": EMBED,
        "dim_output": slots,
        "n_layers": LAYERS,
        "n_heads": HEADS,
        "task": "REGRESSION" if regression else "CLASSIFICATION",
    }
    return state, settings


def tiny_mitra(seed: int = 0, regression: bool = False) -> Mitra:
    """A small random Mitra classifier, or regressor.

    Args:
        seed: Seed for the weights.
        regression: Whether to build a regressor.

    Returns:
        The model.
    """
    return mitra_from_state(*tiny_checkpoint(seed, regression))
