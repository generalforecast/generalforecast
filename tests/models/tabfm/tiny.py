# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A small random TabFM classifier, so model tests run without its 6.5 GB checkpoint."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.tabfm.checkpoint import tabfm_from_state
from generalforecast.models.tabfm.model import TabFM

EMBED = 16
HEADS = 2
INDUCING = 8
CLS = 2
ICL = CLS * EMBED
FREQUENCIES = 8
CLASS_SLOTS = 10
SETTINGS = {
    "embed_dim": EMBED,
    "max_classes": CLASS_SLOTS,
    "col_num_blocks": 2,
    "col_nhead": HEADS,
    "col_num_inds": INDUCING,
    "row_num_blocks": 2,
    "row_nhead": HEADS,
    "row_num_cls": CLS,
    "icl_num_blocks": 2,
    "icl_nhead": HEADS,
    "ff_factor": 2,
    "feature_group_size": 3,
    "num_freq": FREQUENCIES,
    "is_classifier": True,
}


class _StateBuilder:
    """Random arrays under TabFM's PyTorch parameter names."""

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

    def block(self, prefix: str, dim: int) -> None:
        attention = f"{prefix}.attn"
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            self.linear(f"{attention}.{name}", dim, dim)
        self.normal(f"{attention}.per_dim_scale", (dim // HEADS,), 0.2)
        self.norm(f"{attention}.query_ln", dim // HEADS)
        self.norm(f"{attention}.key_ln", dim // HEADS)
        for name in ("pre_attn_ln", "post_attn_ln", "pre_ff_ln", "post_ff_ln"):
            self.norm(f"{prefix}.{name}", dim)
        self.linear(f"{prefix}.linear1", 2 * dim, dim)
        self.linear(f"{prefix}.linear1_gate", 2 * dim, dim)
        self.linear(f"{prefix}.linear2", dim, 2 * dim)

    def columns(self, prefix: str) -> None:
        for index in range(2):
            blocks = f"{prefix}.tf_col.blocks.{index}"
            self.normal(f"{blocks}.ind_vectors", (INDUCING, EMBED), 1.0)
            self.block(f"{blocks}.mab1", EMBED)
            self.block(f"{blocks}.mab2", EMBED)
        self.linear(f"{prefix}.out_w", EMBED, EMBED)
        self.norm(f"{prefix}.ln_w", EMBED)

    def rows(self, prefix: str) -> None:
        head_dim = EMBED // HEADS
        frequencies = 1 / 100_000.0 ** (np.arange(0, head_dim, 2) / head_dim)
        self.arrays[f"{prefix}.tf_row.rope.freqs"] = frequencies
        for index in range(2):
            self.block(f"{prefix}.tf_row.blocks.{index}", EMBED)
        self.norm(f"{prefix}.out_ln", EMBED)


def tiny_state(seed: int = 0) -> tuple[dict, dict]:
    """The weights and settings `tiny_tabfm` builds from, as a checkpoint holds them."""
    builder = _StateBuilder(seed)
    cells = "cell_embedder"
    builder.normal(f"{cells}.fourier_frequencies", (3, FREQUENCIES), 1.0)
    builder.normal(f"{cells}.fourier_frequencies_cat", (3, FREQUENCIES), 1.0)
    builder.linear(f"{cells}.in_linear", EMBED, 2 * FREQUENCIES)
    builder.linear(f"{cells}.in_linear_cat", EMBED, 2 * FREQUENCIES)
    builder.normal(f"{cells}.y_embedder_lookup.weight", (CLASS_SLOTS, EMBED), 1.0)
    builder.columns("col_embedder")
    builder.columns("col_embedder_2")
    builder.normal("cls_tokens", (CLS, EMBED), 1.0)
    builder.rows("row_interactor")
    builder.rows("row_interactor_2")
    for index in range(2):
        builder.block(f"icl_predictor.tf_icl.blocks.{index}", ICL)
    builder.norm("icl_predictor.ln", ICL)
    builder.linear("icl_predictor.y_encoder.projection", ICL, CLASS_SLOTS)
    builder.linear("icl_predictor.decoder.layers.0", 2 * ICL, ICL)
    builder.linear("icl_predictor.decoder.layers.1", CLASS_SLOTS, 2 * ICL)
    state = {
        key: jnp.asarray(value, dtype=jnp.float32)
        for key, value in builder.arrays.items()
    }
    return state, dict(SETTINGS)


def tiny_tabfm(seed: int = 0) -> TabFM:
    """Builds a TabFM classifier with small random weights and the checkpoint's structure.

    Args:
        seed: Seed for the weights.

    Returns:
        The model.
    """
    return tabfm_from_state(*tiny_state(seed))


def classifier() -> tuple[Any, TabFM]:
    """A tiny classifier registered for the engine as `tests/tabfm-classifier`."""
    from generalforecast.models.tabfm.model import tabfm_functions

    return tabfm_functions, tiny_tabfm()
