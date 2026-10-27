# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-2 checkpoints: converted from Amazon's repository, mirrored, then loaded.

The published files are a transformers config, whose `chronos_config` holds
the forecasting settings, and safetensors under the module's parameter names.
"""

import math
from pathlib import Path
from typing import Any

from jax import Array

from generalforecast.models import catalog
from generalforecast.models.artifacts import checkpoint
from generalforecast.models.checkpoints import (
    Published,
    check_settings,
    load_state,
    read_all,
    read_linear,
)
from generalforecast.models.chronos2.model import (
    Chronos2,
    Chronos2Layer,
    FeedForward,
)
from generalforecast.nn.attention import AttentionParameters
from generalforecast.nn.linear import LinearParameters, ResidualBlockParameters
from generalforecast.nn.norm import RMSNormParameters
from generalforecast.nn.rope import RoPE

ARCHITECTURE = "chronos2"


# The upstream revision the catalog's mirror was converted from.
PUBLISHED = {
    "general-forecast/chronos-2": Published(
        "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"
    ),
}

# Settings every release has, with the values the loader is checked against.
SUPPORTED: dict[str, tuple[Any, ...]] = {
    "model_type": ("t5",),
    "feed_forward_proj": ("relu",),
    "dense_act_fn": ("relu",),
    "is_gated_act": (False,),
}
FORECASTING_SUPPORTED: dict[str, tuple[Any, ...]] = {
    "use_reg_token": (True,),
    "use_arcsinh": (True,),
}
# Settings that size the network or set its constants, which any value may take.
SIZES = (
    "d_model",
    "d_kv",
    "d_ff",
    "num_layers",
    "num_heads",
    "rope_theta",
    "layer_norm_epsilon",
    "chronos_config",
)
FORECASTING_SIZES = (
    "context_length",
    "input_patch_size",
    "input_patch_stride",
    "output_patch_size",
    "max_output_patches",
    "quantiles",
    "time_encoding_scale",
)
# Settings that only train or name the model, which the loader does not read.
IGNORED = (
    "architectures",
    "chronos_pipeline_class",
    "dropout_rate",
    "dtype",
    "initializer_factor",
    "pad_token_id",
    "reg_token_id",
    "torch_dtype",
    "transformers_version",
    "use_cache",
    "vocab_size",
)


def chronos2_checkpoint(name: str) -> Path:
    """A Chronos-2 checkpoint directory, verified against the catalog's mirror pins.

    With `GF_CHECKPOINTS` set, the directory must already be there, as on
    machines without internet. Otherwise it is kept under `~/.cache/generalforecast`.

    Args:
        name: A canonical id, such as `general-forecast/chronos-2`.

    Returns:
        A directory with `config.json` and `model.safetensors`.
    """
    return checkpoint(catalog.ENTRIES[name])


def check_supported(settings: dict[str, Any]) -> None:
    """Raises for a Chronos-2 setting the loader has not been checked with.

    Args:
        settings: The checkpoint's model config, with its `chronos_config`.

    Raises:
        NotImplementedError: Naming the unsupported or unknown setting.
    """
    forecasting = settings.get("chronos_config", {})
    check_settings("Chronos-2", settings, SUPPORTED)
    check_settings("Chronos-2", forecasting, FORECASTING_SUPPORTED)
    patches = {forecasting.get(key) for key in FORECASTING_SIZES[1:4]}
    if len(patches) != 1:
        raise NotImplementedError(
            "Chronos-2 input and output patches of different sizes are not supported"
        )
    check_settings("Chronos-2", settings, SUPPORTED, known=(*SIZES, *IGNORED))
    check_settings("Chronos-2", forecasting, FORECASTING_SUPPORTED, FORECASTING_SIZES)


def _norm(state: dict[str, Array], name: str, epsilon: float) -> RMSNormParameters:
    """T5's layer norm, which is an RMSNorm with a scale."""
    return RMSNormParameters(weight=state[f"{name}.weight"], epsilon=epsilon)


def _attention(
    state: dict[str, Array], prefix: str, heads: int, rope: RoPE | None
) -> AttentionParameters:
    """An attention whose scores upstream leaves unscaled, so the queries carry the scale."""
    query = read_linear(state, f"{prefix}.q", bias=False)
    head_dim = query.weight.shape[0] // heads
    return AttentionParameters(
        query=LinearParameters(query.weight * math.sqrt(head_dim), None),
        key=read_linear(state, f"{prefix}.k", bias=False),
        value=read_linear(state, f"{prefix}.v", bias=False),
        output=read_linear(state, f"{prefix}.o", bias=False),
        num_heads=heads,
        rope=rope,
    )


def _residual_block(state: dict[str, Array], prefix: str) -> ResidualBlockParameters:
    """The three linear layers of a residual block, with their biases."""
    return ResidualBlockParameters(
        hidden=read_linear(state, f"{prefix}.hidden_layer"),
        output=read_linear(state, f"{prefix}.output_layer"),
        residual=read_linear(state, f"{prefix}.residual_layer"),
        activation="relu",
    )


def chronos2_from_state(state: dict[str, Array], settings: dict[str, Any]) -> Chronos2:
    """Builds Chronos-2 from the transformers state dict.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        settings: The model config, with its `chronos_config`.

    Returns:
        The model.

    Raises:
        NotImplementedError: If a setting is not supported.
    """
    check_supported(settings)
    forecasting = settings["chronos_config"]
    heads = int(settings["num_heads"])
    epsilon = float(settings["layer_norm_epsilon"])
    rope = RoPE(theta=float(settings["rope_theta"]), interleaved=False)

    def build(read: dict[str, Array]) -> Chronos2:
        layers = []
        for index in range(int(settings["num_layers"])):
            prefix = f"encoder.block.{index}.layer"
            layers.append(
                Chronos2Layer(
                    time_norm=_norm(read, f"{prefix}.0.layer_norm", epsilon),
                    time=_attention(read, f"{prefix}.0.self_attention", heads, rope),
                    group_norm=_norm(read, f"{prefix}.1.layer_norm", epsilon),
                    group=_attention(read, f"{prefix}.1.self_attention", heads, None),
                    feed_forward=FeedForward(
                        norm=_norm(read, f"{prefix}.2.layer_norm", epsilon),
                        input=read_linear(read, f"{prefix}.2.mlp.wi", bias=False),
                        output=read_linear(read, f"{prefix}.2.mlp.wo", bias=False),
                    ),
                )
            )
        return Chronos2(
            patch_embedding=_residual_block(read, "input_patch_embedding"),
            register=read["shared.weight"][1],
            layers=tuple(layers),
            final_norm=_norm(read, "encoder.final_layer_norm", epsilon),
            output_embedding=_residual_block(read, "output_patch_embedding"),
            patch=int(forecasting["input_patch_size"]),
            context_limit=int(forecasting["context_length"]),
            max_output_patches=int(forecasting["max_output_patches"]),
            quantiles=tuple(float(level) for level in forecasting["quantiles"]),
            time_scale=float(
                forecasting.get("time_encoding_scale") or forecasting["context_length"]
            ),
        )

    return read_all(state, build)


def load_chronos2(directory: Path) -> Chronos2:
    """Loads a Chronos-2 checkpoint directory, as `chronos2_checkpoint` writes it.

    Args:
        directory: Directory with `config.json` and `model.safetensors`.

    Returns:
        The model.
    """
    return chronos2_from_state(*load_state(directory, ARCHITECTURE))
