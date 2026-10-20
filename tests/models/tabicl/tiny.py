"""A small random TabICL v2, so model tests run without checkpoints."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.tabicl.checkpoint import tabicl_from_state
from generalforecast.models.tabicl.model import TabICL

EMBED = 16
HEADS = 4
INDUCING = 8
CLS = 2
ICL = CLS * EMBED
SSMAX_HIDDEN = 8
CLASS_SLOTS = 10
QUANTILES = 999


class _StateBuilder:
    """Random arrays under TabICL v2's PyTorch parameter names."""

    def __init__(self, seed: int, norm_bias: bool) -> None:
        self.rng = np.random.default_rng(seed)
        self.norm_bias = norm_bias
        self.arrays: dict[str, np.ndarray] = {}

    def normal(self, prefix: str, shape: tuple[int, ...], scale: float) -> None:
        self.arrays[prefix] = scale * self.rng.normal(size=shape)

    def linear(self, prefix: str, outputs: int, inputs: int) -> None:
        self.normal(f"{prefix}.weight", (outputs, inputs), 1 / math.sqrt(inputs))
        self.normal(f"{prefix}.bias", (outputs,), 0.1)

    def norm(self, prefix: str, dim: int) -> None:
        self.arrays[f"{prefix}.weight"] = 1 + 0.1 * self.rng.normal(size=dim)
        if self.norm_bias:
            self.normal(f"{prefix}.bias", (dim,), 0.1)

    def block(self, prefix: str, dim: int, ssmax: bool) -> None:
        self.normal(f"{prefix}.attn.in_proj_weight", (3 * dim, dim), 1 / math.sqrt(dim))
        self.normal(f"{prefix}.attn.in_proj_bias", (3 * dim,), 0.1)
        self.linear(f"{prefix}.attn.out_proj", dim, dim)
        self.linear(f"{prefix}.linear1", 2 * dim, dim)
        self.linear(f"{prefix}.linear2", dim, 2 * dim)
        self.norm(f"{prefix}.norm1", dim)
        self.norm(f"{prefix}.norm2", dim)
        if ssmax:
            ssmax_prefix = f"{prefix}.attn.ssmax_layer"
            head_dim = dim // HEADS
            self.linear(f"{ssmax_prefix}.base_mlp.0", SSMAX_HIDDEN, 1)
            self.linear(f"{ssmax_prefix}.base_mlp.2", dim, SSMAX_HIDDEN)
            self.linear(f"{ssmax_prefix}.query_mlp.0", SSMAX_HIDDEN, head_dim)
            self.linear(f"{ssmax_prefix}.query_mlp.2", head_dim, SSMAX_HIDDEN)


def tiny_tabicl(*, regression: bool, seed: int = 0) -> TabICL:
    """Builds a TabICL v2 with small random weights.

    It has the checkpoint's structure: feature groups of 3, QASSMax in the
    column compress and ICL blocks, RoPE in the row blocks, 10 class slots or
    999 quantiles, and norm biases only for the classifier.

    Args:
        regression: Whether to build a regressor instead of a classifier.
        seed: Seed for the weights.

    Returns:
        The model.
    """
    return tabicl_from_state(*tiny_state(regression=regression, seed=seed))


def tiny_state(*, regression: bool, seed: int = 0) -> tuple[dict, dict]:
    """The weights and settings `tiny_tabicl` builds from, as a checkpoint holds them."""
    builder = _StateBuilder(seed, norm_bias=not regression)
    targets = 1 if regression else CLASS_SLOTS
    outputs = QUANTILES if regression else CLASS_SLOTS

    builder.linear("col_embedder.in_linear", EMBED, 3)
    builder.linear("col_embedder.y_encoder", EMBED, targets)
    for index in range(2):
        prefix = f"col_embedder.tf_col.blocks.{index}"
        builder.normal(f"{prefix}.ind_vectors", (INDUCING, EMBED), 1.0)
        builder.block(f"{prefix}.multihead_attn1", EMBED, ssmax=True)
        builder.block(f"{prefix}.multihead_attn2", EMBED, ssmax=False)

    builder.normal("row_interactor.cls_tokens", (CLS, EMBED), 1.0)
    builder.norm("row_interactor.out_ln", EMBED)
    for index in range(2):
        builder.block(f"row_interactor.tf_row.blocks.{index}", EMBED, ssmax=False)

    builder.linear("icl_predictor.y_encoder", ICL, targets)
    builder.norm("icl_predictor.ln", ICL)
    builder.linear("icl_predictor.decoder.0", 2 * ICL, ICL)
    builder.linear("icl_predictor.decoder.2", outputs, 2 * ICL)
    for index in range(2):
        builder.block(f"icl_predictor.tf_icl.blocks.{index}", ICL, ssmax=True)

    settings = {
        "col_nhead": HEADS,
        "row_nhead": HEADS,
        "icl_nhead": HEADS,
        "col_num_blocks": 2,
        "row_num_blocks": 2,
        "icl_num_blocks": 2,
        "col_ssmax": "qassmax-mlp-elementwise",
        "icl_ssmax": "qassmax-mlp-elementwise",
        "bias_free_ln": regression,
        "row_rope_base": 100000.0,
        "row_rope_interleaved": False,
        "col_feature_group": "same",
        "col_feature_group_size": 3,
    }
    state = {
        key: jnp.asarray(value, dtype=jnp.float32)
        for key, value in builder.arrays.items()
    }
    return state, settings


def classifier() -> tuple[Any, TabICL]:
    """A tiny classifier as a model spec target: `tests.models.tabicl.tiny:classifier`."""
    from generalforecast.models.tabicl.model import tabicl_functions

    return tabicl_functions, tiny_tabicl(regression=False)


def regressor() -> tuple[Any, TabICL]:
    """A tiny regressor as a model spec target: `tests.models.tabicl.tiny:regressor`."""
    from generalforecast.models.tabicl.model import tabicl_functions

    return tabicl_functions, tiny_tabicl(regression=True)
