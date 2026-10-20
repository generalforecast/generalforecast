"""Small random TabPFN-3 and TabPFN-3.5 models, so tests run without checkpoints."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from generalforecast.models.tabpfn.buckets import regression_borders
from generalforecast.models.tabpfn.checkpoint import one_task, tabpfn_from_state
from generalforecast.models.tabpfn.model import TabPFN

EMBED = 8
HEADS = 2
CLS = 2
ICL = CLS * EMBED
HIDDEN = 4
CLASS_SLOTS = 10
BUCKETS = 100
FREQUENCIES = 4
RANK_FREQUENCIES = 2


class _StateBuilder:
    """Random arrays under TabPFN-3's PyTorch parameter names."""

    def normal(self, name: str, shape: tuple[int, ...], scale: float) -> None:
        self.arrays[name] = scale * self.rng.normal(size=shape)

    def linear(self, prefix: str, outputs: int, inputs: int, bias: bool) -> None:
        self.normal(f"{prefix}.weight", (outputs, inputs), 1 / math.sqrt(inputs))
        if bias:
            self.normal(f"{prefix}.bias", (outputs,), 0.1)

    def norm(self, name: str, dim: int) -> None:
        self.arrays[f"{name}.weight"] = 1 + 0.1 * self.rng.normal(size=dim)

    def scaling(self, prefix: str, dim: int, heads: int) -> None:
        self.linear(f"{prefix}.base_mlp.0", HIDDEN, 1, bias=True)
        self.linear(f"{prefix}.base_mlp.2", dim, HIDDEN, bias=True)
        self.linear(f"{prefix}.query_mlp.0", HIDDEN, dim // heads, bias=True)
        self.linear(f"{prefix}.query_mlp.2", dim // heads, HIDDEN, bias=True)

    def __init__(self, seed: int, release: str) -> None:
        self.rng = np.random.default_rng(seed)
        self.arrays: dict[str, np.ndarray] = {}
        self.multitask = release == "3.5"

    def attention(self, prefix: str, dim: int, scaled: bool) -> None:
        for name in ("q_projection", "k_projection", "v_projection", "out_projection"):
            self.linear(f"{prefix}.{name}", dim, dim, bias=False)
        if scaled:
            self.scaling(f"{prefix}.softmax_scaling_layer", dim, HEADS)
        if self.multitask:
            self.norm(f"{prefix}.q_norm", dim // HEADS)
            self.norm(f"{prefix}.k_norm", dim // HEADS)

    def bucket_head(self, prefix: str) -> None:
        """Puts a regression head's mass on the middle buckets, as trained ones do.

        The outer buckets are wide, and quantiles divide by a bucket's mass, so
        mass spread over them would turn float32 rounding into large
        differences between quantiles.
        """
        self.arrays[f"{prefix}.weight"] *= 0.1
        center = (np.arange(BUCKETS) - (BUCKETS - 1) / 2) / (BUCKETS / 8)
        self.arrays[f"{prefix}.bias"] = -0.5 * center**2

    def layer_norm(self, name: str, dim: int) -> None:
        self.norm(name, dim)
        self.normal(f"{name}.bias", (dim,), 0.1)

    def mlp(self, prefix: str, dim: int) -> None:
        self.linear(f"{prefix}.0", 2 * dim, dim, bias=False)
        self.linear(f"{prefix}.2", dim, 2 * dim, bias=False)

    def cross_block(self, prefix: str, scaled: bool) -> None:
        self.attention(f"{prefix}.attn", EMBED, scaled)
        self.mlp(f"{prefix}.mlp", EMBED)
        for name in ("layernorm_q", "layernorm_kv", "layernorm2"):
            self.norm(f"{prefix}.{name}", EMBED)

    def block(self, prefix: str, attention: str, dim: int, scaled: bool) -> None:
        self.attention(f"{prefix}.{attention}", dim, scaled)
        self.mlp(f"{prefix}.mlp", dim)
        self.norm(f"{prefix}.layernorm", dim)
        self.norm(f"{prefix}.layernorm_mlp", dim)


def _heads(builder: _StateBuilder) -> None:
    """Both tasks' target encoders and heads, as TabPFN-3.5 keeps them."""
    builder.normal("col_y_encoder.multiclass.embedding.weight", (CLASS_SLOTS, EMBED), 1)
    builder.linear("col_y_encoder.regression", EMBED, 1, bias=True)
    builder.layer_norm("col_y_layernorm", EMBED)
    builder.normal("icl_y_encoder.multiclass.embedding.weight", (CLASS_SLOTS, ICL), 1)
    builder.linear("icl_y_encoder.regression", ICL, 1, bias=True)
    builder.layer_norm("icl_y_layernorm", ICL)
    decoder = "heads.many_class_decoder"
    builder.linear(f"{decoder}.q_projection", 2 * HIDDEN, ICL, bias=True)
    builder.linear(f"{decoder}.k_projection", 2 * HIDDEN, ICL, bias=True)
    builder.scaling(f"{decoder}.softmax_scaling_layer", 2 * HIDDEN, 2)
    builder.linear("heads.output_projection", BUCKETS, ICL, bias=True)
    builder.bucket_head("heads.output_projection")
    for task in ("classification", "regression"):
        builder.norm(f"heads.mlp_{task}.norm", ICL)
        builder.mlp(f"heads.mlp_{task}.mlp", ICL)
    builder.arrays["heads.regression_borders"] = regression_borders(BUCKETS)


def tiny_checkpoint(
    *, regression: bool, seed: int = 0, release: str = "3", both_tasks: bool = False
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """A small random TabPFN checkpoint with the published ones' structure.

    Args:
        regression: Whether to build a regressor instead of a classifier.
        seed: Seed for the weights.
        release: `3`, or `3.5` for Fourier cells, ranks and one checkpoint
            for both tasks.
        both_tasks: Keep both tasks' parameters of a TabPFN-3.5 checkpoint, as
            published, instead of one task's, as converted.

    Returns:
        The state dict, settings and architecture name.
    """
    builder = _StateBuilder(seed, release)
    if builder.multitask:
        builder.normal("x_embed.fourier.frequencies", (3, FREQUENCIES), 2.0)
        builder.linear("x_embed.fourier.in_linear", EMBED, 2 * FREQUENCIES, False)
        builder.linear(
            "x_embed.metadata_linear", EMBED, 6 + 3 * 2 * RANK_FREQUENCIES, False
        )
        builder.layer_norm("x_embed.layernorm", EMBED)
    else:
        builder.linear("x_embed", EMBED, 6, bias=True)
    for index in range(2):
        prefix = f"feature_distribution_embedder.layers.{index}"
        builder.normal(f"{prefix}.inducing_vectors", (4, EMBED), 1.0)
        builder.cross_block(f"{prefix}.cross_attn_block1", scaled=True)
        builder.cross_block(f"{prefix}.cross_attn_block2", scaled=False)
    builder.normal("column_aggregator.cls_tokens", (CLS, EMBED), 1.0)
    builder.norm("column_aggregator.out_ln", EMBED)
    for index in range(2):
        builder.block(f"column_aggregator.blocks.{index}", "attention", EMBED, False)
    for index in range(2):
        builder.block(f"icl_blocks.{index}", "icl_attention", ICL, scaled=True)
    builder.norm("output_norm", ICL)
    if builder.multitask:
        _heads(builder)
    elif regression:
        builder.linear("col_y_encoder", EMBED, 1, bias=True)
        builder.linear("icl_y_encoder", ICL, 1, bias=True)
        builder.linear("output_projection.0", 2 * ICL, ICL, bias=True)
        builder.linear("output_projection.2", BUCKETS, 2 * ICL, bias=True)
        builder.bucket_head("output_projection.2")
    else:
        builder.normal("col_y_encoder.embedding.weight", (CLASS_SLOTS, EMBED), 1.0)
        builder.normal("icl_y_encoder.embedding.weight", (CLASS_SLOTS, ICL), 1.0)
        builder.linear("many_class_decoder.q_projection", 2 * HIDDEN, ICL, bias=True)
        builder.linear("many_class_decoder.k_projection", 2 * HIDDEN, ICL, bias=True)
        builder.scaling("many_class_decoder.softmax_scaling_layer", 2 * HIDDEN, 2)
    if not builder.multitask:
        builder.arrays["regression_borders"] = regression_borders(BUCKETS)
    settings = {
        "feature_group_size": 3,
        "use_rope": True,
        "use_nan_indicators": True,
        "icl_num_kv_heads": None,
        "icl_num_kv_heads_test": 1,
        "ff_factor": 2,
        "dist_embed_num_heads": HEADS,
        "dist_embed_num_blocks": 2,
        "feat_agg_num_heads": HEADS,
        "feat_agg_num_blocks": 2,
        "feat_agg_rope_base": 100_000,
        "icl_num_heads": HEADS,
        "nlayers": 2,
        "decoder_num_heads": 2,
    }
    settings["num_buckets"] = BUCKETS
    architecture = "tabpfn_v3"
    if builder.multitask:
        architecture = "tabpfn_v3_5"
        settings |= {
            "use_fourier_cell_embedding": True,
            "use_mlp_heads": True,
            "use_qk_norm": True,
            "y_encoder_layernorm": True,
            "feat_agg_use_softmax_scaling": False,
            "cell_ecdf_num_frequencies": RANK_FREQUENCIES,
            # Fewer edges than the tests' contexts have rows, so ranks interpolate.
            "cell_ecdf_num_buckets": 8,
        }
    state = {
        key: jnp.asarray(value, dtype=jnp.float32)
        for key, value in builder.arrays.items()
    }
    if builder.multitask and not both_tasks:
        state = one_task(state, regression)
    return state, settings, architecture


def tiny_tabpfn(*, regression: bool, seed: int = 0, release: str = "3") -> TabPFN:
    """Builds a TabPFN from `tiny_checkpoint`.

    Args:
        regression: Whether to build a regressor instead of a classifier.
        seed: Seed for the weights.
        release: `3` or `3.5`.

    Returns:
        The model.
    """
    return tabpfn_from_state(
        *tiny_checkpoint(regression=regression, seed=seed, release=release)
    )


def classifier() -> tuple[Any, TabPFN]:
    """A tiny classifier as a model spec target: `tests.models.tabpfn.tiny:classifier`."""
    from generalforecast.models.tabpfn.model import tabpfn_functions

    return tabpfn_functions, tiny_tabpfn(regression=False)


def regressor() -> tuple[Any, TabPFN]:
    """A tiny regressor as a model spec target: `tests.models.tabpfn.tiny:regressor`."""
    from generalforecast.models.tabpfn.model import tabpfn_functions

    return tabpfn_functions, tiny_tabpfn(regression=True)
