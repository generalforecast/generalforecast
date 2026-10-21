# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Timestral checkpoints: converted from Mistral's repositories, mirrored, then loaded.

The published files are the inference config, flat, and safetensors under
the module's parameter names, with the attention's projections fused.
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
from generalforecast.models.timestral.model import Timestral, TimestralLayer
from generalforecast.nn.attention import AttentionParameters
from generalforecast.nn.linear import (
    GatedMLPParameters,
    LinearParameters,
    ResidualBlockParameters,
)
from generalforecast.nn.norm import RMSNormParameters
from generalforecast.nn.rope import RoPE

ARCHITECTURE = "timestral"


class Published(NamedTuple):
    """A checkpoint in Mistral's Hugging Face repositories."""

    repository: str
    revision: str


# The upstream revisions the catalog's mirrors were converted from.
PUBLISHED = {
    "timestral-209m": Published(
        "mistralai/Timestral-209M", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
    ),
    "timestral-28m": Published(
        "mistralai/Timestral-28M", "180698bbf5a50dda3516aa6419798437fb03a430"
    ),
}

# Settings every release has, with the values the loader is checked against.
SUPPORTED: dict[str, tuple[Any, ...]] = {
    "attn_bias": (False,),
    "mlp_bias": (False,),
    "time_attn_gate": ("sdpa_sigmoid",),
    "variate_attn_gate": ("sdpa_sigmoid",),
    "use_asinh": (True,),
    "stabilize_with_global": (False,),
    "covariate_enabled": (True,),
    "attend_missing_patches": (True,),
}
# Settings that size the network or set its constants, which any value may take.
SIZES = (
    "patch_size",
    "d_model",
    "num_heads",
    "mlp_hidden_dim",
    "block_schedule",
    "quantile_knots",
    "block_size",
    "context_length",
    "rope_theta",
    "rope_partial_factor",
    "norm_eps",
    "minimum_scale",
    "quantile_width_factor",
)
# Settings that only pick an equivalent computation, which the loader does not read.
IGNORED = ("split_context_horizon_attn",)

# Which attention axes each layer kind runs, True along time.
LAYER_KINDS = {
    "time": (True,),
    "variate": (False,),
    "time_variate": (True, False),
    "variate_time": (False, True),
}


def timestral_checkpoint(name: str) -> Path:
    """A Timestral checkpoint directory, verified against the catalog's mirror pins.

    With `GF_CHECKPOINTS` set, the directory must already be there, as on
    machines without internet. Otherwise it is kept under `~/.cache/generalforecast`.

    Args:
        name: A catalog name, such as `timestral-209m`.

    Returns:
        A directory with `config.json` and `model.safetensors`.
    """
    return checkpoint(catalog.ENTRIES[name])


def check_supported(settings: dict[str, Any]) -> None:
    """Raises for a Timestral setting the loader has not been checked with.

    Args:
        settings: The checkpoint's config.

    Raises:
        NotImplementedError: Naming the unsupported or unknown setting.
    """
    for key, values in SUPPORTED.items():
        if settings.get(key) not in values:
            raise NotImplementedError(
                f"Timestral {key}={settings.get(key)!r} is not supported"
            )
    kinds = set(settings.get("block_schedule", ())) - set(LAYER_KINDS)
    if kinds:
        raise NotImplementedError(
            f"Timestral layer kinds {sorted(kinds)} are not supported"
        )
    if 0.5 not in settings.get("quantile_knots", ()):
        raise NotImplementedError("Timestral quantile_knots must hold the median")
    unknown = sorted(set(settings) - set(SUPPORTED) - set(SIZES) - set(IGNORED))
    if unknown:
        raise NotImplementedError(f"Timestral settings {unknown} are not supported")


def _norm(state: dict[str, Array], name: str, epsilon: float) -> RMSNormParameters:
    """An RMSNorm whose scale is stored as `scale`."""
    return RMSNormParameters(weight=state[f"{name}.scale"], epsilon=epsilon)


def _attention(
    state: dict[str, Array], prefix: str, heads: int, rope: RoPE | None
) -> AttentionParameters:
    """An attention from its fused projection, laid out `(qkv, head_dim, heads)`."""
    fused = state[f"{prefix}.wQKV.weight"]
    width = fused.shape[1]
    stacked = fused.reshape(3, width // heads, heads, width).transpose(0, 2, 1, 3)
    query, key, value = stacked.reshape(3, width, width)
    return AttentionParameters(
        query=LinearParameters(query, None),
        key=LinearParameters(key, None),
        value=LinearParameters(value, None),
        output=read_linear(state, f"{prefix}.wO", bias=False),
        num_heads=heads,
        gate=read_linear(state, f"{prefix}._gate", bias=False),
        rope=rope,
    )


def _residual_block(
    state: dict[str, Array], prefix: str, bias: bool
) -> ResidualBlockParameters:
    """The three linear layers of a residual block."""
    return ResidualBlockParameters(
        hidden=read_linear(state, f"{prefix}.hidden_layer", bias),
        output=read_linear(state, f"{prefix}.output_layer", bias),
        residual=read_linear(state, f"{prefix}.residual_layer", bias),
        activation="silu",
    )


def timestral_from_state(
    state: dict[str, Array], settings: dict[str, Any], symmetric: bool = True
) -> Timestral:
    """Builds Timestral from Mistral's state dict.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        settings: The config.
        symmetric: Whether forecasts average over the group and its negation.

    Returns:
        The model.

    Raises:
        NotImplementedError: If a setting is not supported.
    """
    check_supported(settings)
    heads = int(settings["num_heads"])
    epsilon = float(settings["norm_eps"])
    head_dim = int(settings["d_model"]) // heads
    rope = RoPE(
        theta=float(settings["rope_theta"]),
        interleaved=False,
        rotary_dim=int(head_dim * float(settings["rope_partial_factor"])),
    )

    def build(read: dict[str, Array]) -> Timestral:
        layers = []
        for index, kind in enumerate(settings["block_schedule"]):
            prefix = f"transformer.layers.{index}"
            axes = LAYER_KINDS[kind]
            # SwiGLU's fused input holds the half that multiplies, then the half SiLU takes.
            multiplies, activated = (
                LinearParameters(half, None)
                for half in read[f"{prefix}.mlp.w12.weight"].reshape(
                    2, -1, int(settings["d_model"])
                )
            )
            layers.append(
                TimestralLayer(
                    norms=tuple(
                        _norm(read, f"{prefix}.attn_norms.{step}", epsilon)
                        for step in range(len(axes))
                    ),
                    attentions=tuple(
                        _attention(
                            read,
                            f"{prefix}.attns.{step}",
                            heads,
                            rope if along_time else None,
                        )
                        for step, along_time in enumerate(axes)
                    ),
                    mlp_norm=_norm(read, f"{prefix}.mlp_norm", epsilon),
                    mlp=GatedMLPParameters(
                        gate=activated,
                        up=multiplies,
                        down=read_linear(read, f"{prefix}.mlp.w3", bias=False),
                    ),
                    along_time=axes,
                )
            )
        return Timestral(
            patch_embedding=_residual_block(read, "representation.patch_proj", True),
            query=read["representation.forecast_query"],
            layers=tuple(layers),
            final_norm=_norm(read, "transformer.out_norm", epsilon),
            head=_residual_block(read, "head.projection", False),
            patch=int(settings["patch_size"]),
            block_size=int(settings["block_size"]),
            context_limit=int(settings["context_length"]),
            quantiles=tuple(float(level) for level in settings["quantile_knots"]),
            width_factor=float(settings["quantile_width_factor"]),
            minimum_scale=float(settings["minimum_scale"]),
            symmetric=symmetric,
        )

    return read_all(state, build)


def load_timestral(directory: Path, symmetric: bool = True) -> Timestral:
    """Loads a Timestral checkpoint directory, as `timestral_checkpoint` writes it.

    Args:
        directory: Directory with `config.json` and `model.safetensors`.
        symmetric: Whether forecasts average over the group and its negation.

    Returns:
        The model.
    """
    return timestral_from_state(*load_state(directory, ARCHITECTURE), symmetric)
