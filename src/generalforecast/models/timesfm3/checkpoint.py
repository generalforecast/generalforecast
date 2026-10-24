# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM 3 checkpoints: converted from Google's repository, then loaded.

The published files are the model's config, with the residual block's and
the transformer's settings nested, and safetensors under the module's
parameter names. The weights are under Google's TimesFM license, which
allows no commercial use; it is copied into the converted checkpoint beside
them, and nothing is rehosted.
"""

from pathlib import Path
from typing import Any, NamedTuple

from jax import Array

from generalforecast.models import catalog
from generalforecast.models.artifacts import checkpoint
from generalforecast.models.checkpoints import (
    load_state,
    read_all,
    read_linear,
)
from generalforecast.models.timesfm3.model import TimesFM, TimesFMLayer
from generalforecast.nn.attention import AttentionParameters
from generalforecast.nn.linear import ResidualBlockParameters
from generalforecast.nn.norm import RMSNormParameters
from generalforecast.nn.rope import RoPE
from generalforecast.nn.scaling import PerDimScaleParameters

ARCHITECTURE = "timesfm3"


class Published(NamedTuple):
    """A checkpoint in Google's Hugging Face repositories."""

    repository: str
    revision: str


# The upstream repository and revision each converts from, as the catalog pins.
PUBLISHED = {
    name: Published(
        catalog.ENTRIES[name].artifact.repository,
        catalog.ENTRIES[name].artifact.revision,
    )
    for name in ("general-forecast/timesfm-3",)
}

# Settings every release has, with the values the loader is checked against,
# at the top level, in the residual block and in the transformer.
SUPPORTED: dict[str, tuple[Any, ...]] = {
    "input_transform": ("identity",),
    "use_frozen_running_stats": (False,),
    "use_iterative_cpm_revin": (True,),
    "use_stitching": (True,),
    "use_variate_attention": (True,),
}
BLOCK_SUPPORTED: dict[str, tuple[Any, ...]] = {
    "activation": ("relu",),
    "identity_skip": (False,),
    "prenorm": ("none",),
    "use_bias": (False,),
}
TRANSFORMER_SUPPORTED: dict[str, tuple[Any, ...]] = {
    "attention_norm": ("rms",),
    "causal_attention": (True,),
    "debug_no_masking": (False,),
    "feedforward_norm": ("rms",),
    "ff_activation": ("relu",),
    "paired_token_skip_second": (False,),
    "qk_norm": ("rms",),
    "use_bias": (False,),
    "use_memory_efficient_attention": (True,),
    "use_rope_seq": (True,),
    "use_rope_var": (False,),
    "v_norm": ("none",),
}
# Settings that size the network or set its constants, which any value may take.
SIZES = (
    "input_patch_len",
    "output_patch_len",
    "quantiles",
    "value_clip",
    "linear_detrending_threshold",
    "use_linear_detrending",
    "residual_block_config",
    "transformer_config",
)
BLOCK_SIZES = ("hidden_dims", "output_dims", "dropout")
TRANSFORMER_SIZES = (
    "num_layers",
    "use_remat",
    "transformer",
    "model_dims",
    "hidden_dims",
    "num_heads",
    "max_variates",
    "deterministic",
    "training",
    "use_sdpa",
)


def timesfm_checkpoint(name: str) -> Path:
    """A TimesFM 3 checkpoint directory, verified, converted from the pinned published one if missing.

    With `GF_CHECKPOINTS` set, the directory must already be there, as on
    machines without internet. Otherwise it is kept under `~/.cache/generalforecast`.

    Args:
        name: A key of `PUBLISHED`, such as `general-forecast/timesfm-3`.

    Returns:
        A directory with `config.json` and `model.safetensors`.
    """
    return checkpoint(catalog.ENTRIES[name])


def _check(
    name: str,
    settings: dict[str, Any],
    supported: dict[str, tuple[Any, ...]],
    sizes: tuple[str, ...],
) -> None:
    """Raises for a setting off `supported`'s values, or one neither lists."""
    for key, values in supported.items():
        if settings.get(key) not in values:
            raise NotImplementedError(
                f"TimesFM {name}{key}={settings.get(key)!r} is not supported"
            )
    unknown = sorted(set(settings) - set(supported) - set(sizes))
    if unknown:
        raise NotImplementedError(f"TimesFM {name}settings {unknown} are not supported")


def check_supported(settings: dict[str, Any]) -> None:
    """Raises for a TimesFM 3 setting the loader has not been checked with.

    Args:
        settings: The checkpoint's config.

    Raises:
        NotImplementedError: Naming the unsupported or unknown setting.
    """
    _check("", settings, SUPPORTED, SIZES)
    _check(
        "residual block ",
        settings.get("residual_block_config", {}),
        BLOCK_SUPPORTED,
        BLOCK_SIZES,
    )
    stack = settings.get("transformer_config", {})
    _check(
        "transformer ",
        {**stack, **stack.get("transformer", {})},
        TRANSFORMER_SUPPORTED,
        TRANSFORMER_SIZES,
    )
    if settings.get("output_patch_len", 0) % max(settings.get("input_patch_len", 1), 1):
        raise NotImplementedError("TimesFM output patches must be whole input patches")


def _norm(state: dict[str, Array], name: str) -> RMSNormParameters:
    """An RMSNorm with PyTorch's default epsilon."""
    return RMSNormParameters(weight=state[f"{name}.weight"])


def _attention(
    state: dict[str, Array], prefix: str, heads: int, rope: RoPE | None
) -> AttentionParameters:
    """An attention with its query and key norms per head and per-dimension scale."""
    return AttentionParameters(
        query=read_linear(state, f"{prefix}.query_proj", bias=False),
        key=read_linear(state, f"{prefix}.key_proj", bias=False),
        value=read_linear(state, f"{prefix}.value_proj", bias=False),
        output=read_linear(state, f"{prefix}.out_proj", bias=False),
        num_heads=heads,
        scale=PerDimScaleParameters(state[f"{prefix}.per_dim_scale.per_dim_scale"]),
        query_norm=_norm(state, f"{prefix}.query_ln"),
        key_norm=_norm(state, f"{prefix}.key_ln"),
        rope=rope,
    )


def timesfm_from_state(state: dict[str, Array], settings: dict[str, Any]) -> TimesFM:
    """Builds TimesFM 3 from Google's state dict.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        settings: The config.

    Returns:
        The model.

    Raises:
        NotImplementedError: If a setting is not supported.
    """
    check_supported(settings)
    transformer = settings["transformer_config"]["transformer"]
    heads = int(transformer["num_heads"])
    rope = RoPE(theta=10000.0, interleaved=False)

    def build(read: dict[str, Array]) -> TimesFM:
        layers = []
        for index in range(int(settings["transformer_config"]["num_layers"])):
            prefix = f"transformer_stack.layers.{index}"
            layers.append(
                TimesFMLayer(
                    sequence_norm=_norm(read, f"{prefix}.pre_seq_attn_ln"),
                    sequence=_attention(read, f"{prefix}.seq_attn", heads, rope),
                    sequence_output_norm=_norm(read, f"{prefix}.post_seq_attn_ln"),
                    variate_norm=_norm(read, f"{prefix}.pre_var_attn_ln"),
                    variate=_attention(read, f"{prefix}.var_attn", heads, None),
                    variate_output_norm=_norm(read, f"{prefix}.post_var_attn_ln"),
                    feed_forward_norm=_norm(read, f"{prefix}.pre_ff_ln"),
                    feed_forward_input=read_linear(read, f"{prefix}.ff0", bias=False),
                    feed_forward_output=read_linear(read, f"{prefix}.ff1", bias=False),
                    feed_forward_output_norm=_norm(read, f"{prefix}.post_ff_ln"),
                )
            )
        return TimesFM(
            embedding=ResidualBlockParameters(
                hidden=read_linear(
                    read, "pre_transformer_resblock.hidden_layer", bias=False
                ),
                output=read_linear(
                    read, "pre_transformer_resblock.output_layer", bias=False
                ),
                residual=read_linear(
                    read, "pre_transformer_resblock.residual_layer", bias=False
                ),
                activation="relu",
            ),
            layers=tuple(layers),
            head=read_linear(read, "output_head"),
            patch=int(settings["input_patch_len"]),
            output_patch=int(settings["output_patch_len"]),
            quantiles=tuple(float(level) for level in settings["quantiles"]),
            value_clip=float(settings["value_clip"]),
            detrending=(
                float(settings["linear_detrending_threshold"])
                if settings["use_linear_detrending"]
                else None
            ),
            max_rows=int(transformer["max_variates"]),
        )

    return read_all(state, build)


def load_timesfm(directory: Path) -> TimesFM:
    """Loads a TimesFM 3 checkpoint directory, as `timesfm_checkpoint` writes it.

    Args:
        directory: Directory with `config.json` and `model.safetensors`.

    Returns:
        The model.
    """
    return timesfm_from_state(*load_state(directory, ARCHITECTURE))
