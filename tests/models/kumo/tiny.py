"""A small random Kumo Tabular, so model tests run without checkpoints."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.kumo.checkpoint import kumo_from_state
from generalforecast.models.kumo.model import KumoTabular

# Named like NVIDIA's MODEL_KWARGS. Readout tokens times cell channels (32) differs
# from the ICL width (16), so the row projection runs, and queries read 2 of the 4
# ICL key/value heads, like the medium and large checkpoints.
SETTINGS: dict[str, Any] = {
    "cell_channels": 16,
    "num_embedding_layers": 2,
    "num_embedding_heads": 2,
    "num_inducing_points": 8,
    "group_size": 3,
    "num_frequencies": 4,
    "num_readout_tokens": 2,
    "icl_channels": 16,
    "num_icl_layers": 2,
    "num_icl_heads": 4,
    "num_icl_key_value_heads_for_query": 2,
}
CLASS_SLOTS = 10
QUANTILES = 999
GATE_HIDDEN = 64


class _StateBuilder:
    """Random arrays under Kumo Tabular's PyTorch parameter names."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.arrays: dict[str, np.ndarray] = {}

    def normal(self, name: str, shape: tuple[int, ...], scale: float = 1.0) -> None:
        self.arrays[name] = scale * self.rng.normal(size=shape)

    def linear(self, prefix: str, outputs: int, inputs: int, bias: bool = True) -> None:
        self.normal(f"{prefix}.weight", (outputs, inputs), 1 / math.sqrt(inputs))
        if bias:
            self.normal(f"{prefix}.bias", (outputs,), 0.1)

    def norm(self, name: str, dim: int) -> None:
        self.arrays[name] = 1 + 0.1 * self.rng.normal(size=dim)

    def block(self, prefix: str, dim: int, heads: int, scaling: str | None) -> None:
        self.norm(f"{prefix}.query_norm.weight", dim)
        self.norm(f"{prefix}.key_value_norm.weight", dim)
        self.linear(f"{prefix}.attn.qkv_lin", 3 * dim, dim)
        self.linear(f"{prefix}.attn.out_lin", dim, dim)
        self.norm(f"{prefix}.mlp.0.weight", dim)
        self.linear(f"{prefix}.mlp.1", 2 * dim, dim)
        self.linear(f"{prefix}.mlp.3", dim, 2 * dim)
        if scaling is None:
            return
        scale = f"{prefix}.attn.sdpa.query_scaling"
        self.arrays[f"{scale}.head_scale"] = self.rng.uniform(size=heads)
        if scaling == "gated":
            self.linear(f"{scale}.gate.0", GATE_HIDDEN, dim // heads)
            self.linear(f"{scale}.gate.2", dim // heads, GATE_HIDDEN)


def tiny_kumo(*, regression: bool, seed: int = 0) -> KumoTabular:
    """Builds a Kumo Tabular with small random weights and the checkpoint's names.

    Args:
        regression: Whether to build a regressor instead of a classifier.
        seed: Seed for the weights.

    Returns:
        The model.
    """
    state = {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in tiny_kumo_state(regression=regression, seed=seed).items()
    }
    return kumo_from_state(state, SETTINGS)


def tiny_kumo_state(*, regression: bool, seed: int = 0) -> dict[str, np.ndarray]:
    """Random weights for `SETTINGS`, keyed like a Kumo Tabular state dict."""
    s = SETTINGS
    cells, heads, icl = s["cell_channels"], s["num_embedding_heads"], s["icl_channels"]
    groups, frequencies = s["group_size"], s["num_frequencies"]
    readout = s["num_readout_tokens"]
    builder = _StateBuilder(seed)

    embedding = "row_embedding.cell_embedding"
    for kind in ("num", "cat"):
        builder.normal(f"{embedding}.{kind}_freq", (groups, frequencies))
        builder.linear(f"{embedding}.{kind}_lin", cells, 2 * frequencies)
    builder.linear(f"{embedding}.nan_lin", cells, groups, bias=False)

    if regression:
        builder.linear("row_embedding.y_lin", cells, 1, bias=False)
        builder.linear("icl_block.y_lin", icl, 1, bias=False)
    else:
        builder.normal("row_embedding.y_emb.weight", (CLASS_SLOTS, cells))
        builder.normal("icl_block.y_emb.weight", (CLASS_SLOTS, icl))

    builder.normal("row_embedding.readout_token", (readout, cells), 0.02)
    head_dim = cells // heads
    inverse_frequencies = 1 / 100_000 ** (np.arange(0, head_dim, 2) / head_dim)
    for layer in range(s["num_embedding_layers"]):
        column = f"row_embedding.col_blocks.{layer}"
        builder.normal(f"{column}.inducing_points", (s["num_inducing_points"], cells))
        builder.block(f"{column}.inducing_block", cells, heads, scaling="log")
        builder.block(f"{column}.output_block", cells, heads, scaling=None)
        row = f"row_embedding.row_blocks.{layer}"
        builder.block(row, cells, heads, scaling="gated")
        for transform in ("query_transform", "key_transform"):
            builder.arrays[f"{row}.attn.{transform}.0.inv_freq"] = inverse_frequencies
    builder.norm("row_embedding.norm.weight", cells)

    if readout * cells != icl:
        builder.linear("row_project", icl, readout * cells)
    for layer in range(s["num_icl_layers"]):
        builder.block(f"icl_block.layers.{layer}", icl, s["num_icl_heads"], "log")
    builder.norm("icl_block.norm.weight", icl)
    builder.linear("icl_block.head.0", 2 * icl, icl)
    builder.linear(
        "icl_block.head.2", QUANTILES if regression else CLASS_SLOTS, 2 * icl
    )

    return builder.arrays


def classifier() -> tuple[Any, KumoTabular]:
    """A tiny classifier as a model spec target: `tests.models.kumo.tiny:classifier`."""
    from generalforecast.models.kumo.model import kumo_functions

    return kumo_functions, tiny_kumo(regression=False)


def regressor() -> tuple[Any, KumoTabular]:
    """A tiny regressor as a model spec target: `tests.models.kumo.tiny:regressor`."""
    from generalforecast.models.kumo.model import kumo_functions

    return kumo_functions, tiny_kumo(regression=True)
