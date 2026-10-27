# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from generalforecast.models.checkpoints import (
    Published,
    check_settings,
    converted_file,
    load_state,
    read_all,
    read_linear,
    read_torch_checkpoint,
    without,
)
from generalforecast.models.limix.model import (
    CellEmbedding,
    ClassTargets,
    LimiX,
    LimiXLayer,
    ValueTargets,
)
from generalforecast.nn.attention import AttentionParameters
from generalforecast.nn.linear import (
    GatedMLPParameters,
    LinearParameters,
    MLPParameters,
)
from generalforecast.nn.norm import RMSNormParameters
from generalforecast.nn.scaling import LogScaleParameters

ARCHITECTURE = "limix"


LIMIX_2 = Published(
    "stable-ai/LimiX-2", "de07b679e74a41b50b9de18251a8fa245e537440", "LimiX-2.ckpt"
)
# One file serves both tasks; each converts into its own directory.
PUBLISHED = {
    "general-forecast/limix-2-classifier": LIMIX_2,
    "general-forecast/limix-2-regressor": LIMIX_2,
}

# Settings every release has, with the values the loader is checked against.
SUPPORTED: dict[str, tuple[Any, ...]] = {
    "arch_version": ("2.0",),
    "layer_arch": ("smf",),
    "cls_only_start_layer": (-1,),
    "pre_norm": (False,),
    "all_norm": (True,),
    "tf_norm_type": ("rmsnorm",),
    "tf_layer_norm_use_elementwise_affine": (False,),
    "tf_mlp_layer_type": ("gated",),
    "tf_mlp_activation_fuction": ("silu",),
    "use_gated_norm": (False,),
    "separate_x_y_ffn": (True,),
    "use_separate_attention": (True,),
    "separate_attn_kv_combined": (False,),
    "sample_attention_cls_token_merge": ("reshape",),
    "scale_y_sample_attention_heads": (True,),
    "feature_attention_cls_token_merge": ("none",),
    "seq_att_use_softmax_scaling_mlp_for_q": (True,),
    "fea_att_use_softmax_scaling_mlp_for_q": (True,),
    "seq_att_use_q_norm": (True,),
    "seq_att_use_k_norm": (True,),
    "fea_att_use_q_norm": (True,),
    "fea_att_use_k_norm": (True,),
    "self_share_all_kv_heads": (False,),
    "cross_share_all_kv_heads": (True,),
    "seq_attn_isolated": (False,),
    "seq_attn_serial": (False,),
    "mask_feature_filling_type": ("mask_token",),
    "feature_positional_embedding_type": ("subspace",),
    "add_task_type": (True,),
    "learnable_borders": (False,),
    "reg_borders_use_train_quantile": (False,),
    "cls_y_decoder_type": ("Free_MLP",),
    "reg_y_decoder_type": ("Free_MLP",),
    "use_cls_post_adapter": (True,),
    "use_cls_post_adapter_residual": (False,),
    "use_reg_post_adapter": (True,),
    "use_reg_post_adapter_residual": (False,),
}
# Settings of the preprocessing and the cell and target embeddings.
SECTION_SUPPORTED: dict[str, dict[str, tuple[Any, ...]]] = {
    "preprocess_config_x": {
        "nan_handling_enabled": (True,),
        "normalize_on_train_only": (True,),
        "normalize_x": (True,),
        "remove_outliers": (False,),
        "normalize_by_used_features": (True,),
    },
    "encoder_config_x": {
        "feature_embedding_type": ("mask_embedding",),
        "numeric_embed_type": ("linear",),
        "use_feature_group_encoder": (True,),
        "feature_fusion_network_type": ("original",),
        "use_feature_mask_indicator": (False,),
        "mask_token_emb_type": ("uniform_mask_emb_type",),
    },
    "encoder_config_y": {
        "cls_y_encoder_type": ("emby_embedding",),
        "y_encoder_type": ("none_embedding",),
        "reg_y_numeric_embed_type": ("MLP_GELU_postnorm",),
        "cls_emb_learning_scalar": (False,),
        "mask_token_emb_type": ("uniform_mask_emb_type",),
    },
}
# Norm epsilon throughout LimiX.
_EPSILON = 1e-5
# Parameters only the imputation head reads.
_IMPUTATION = ("feature_decoder.", "feature_post_adapter.")
# Each task's parameters, by whether it regresses.
_TASK = {
    False: ("cls_y_encoder.", "cls_y_decoder.", "cls_post_adapter."),
    True: ("reg_y_encoder.", "reg_y_decoder.", "reg_post_adapter.", "_reg_borders"),
}


def one_task(state: dict[str, Any], regression: bool) -> dict[str, Any]:
    """The state without the other task's parameters and the imputation head.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        regression: Whether to keep regression rather than classification.

    Returns:
        The state for one task.
    """
    return without(state, (*_TASK[not regression], *_IMPUTATION, "reg_log_widths"))


def read_published(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Reads a published LimiX file.

    Args:
        path: A LimiX `.ckpt` file.

    Returns:
        The settings, with `arch_version`, and the arrays.
    """
    stored = read_torch_checkpoint(path)
    settings = {**stored["config"], "arch_version": stored["arch_version"]}
    return settings, stored["state_dict"]


def limix_checkpoint(id: str) -> Path:
    """A LimiX checkpoint directory, converted from the published file if missing.

    With `GF_CHECKPOINTS` set, the directory must already be there, as on
    machines without internet. Otherwise it is kept under `~/.cache/generalforecast`.

    Args:
        id: A key of `PUBLISHED`, such as `general-forecast/limix-2-classifier`.

    Returns:
        A directory with `config.json` and `model.safetensors`.
    """

    def convert(path: Path, name: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        settings, state = read_published(path)
        return ARCHITECTURE, settings, one_task(state, name.endswith("-regressor"))

    return converted_file(id, PUBLISHED[id], convert)


def check_supported(settings: dict[str, Any]) -> None:
    """Raises for a LimiX setting whose value the loader has not been checked with.

    Args:
        settings: The checkpoint's model config, with `arch_version`.

    Raises:
        NotImplementedError: Naming the unsupported setting.
    """
    check_settings("LimiX", settings, SUPPORTED)
    for section, supported in SECTION_SUPPORTED.items():
        check_settings(
            "LimiX", settings.get(section, {}), supported, prefix=f"{section}."
        )
    last = int(settings["nlayers"]) - 1
    for key in ("cls_y_emb_layer", "reg_y_emb_layer"):
        if int(settings.get(key, last)) != last:
            raise NotImplementedError(f"LimiX {key}={settings[key]!r} is not supported")
    for layer in settings["model_structure_config"]["layers"]:
        if layer["arch"] != "smf" or layer["pre_norm"] or not layer["all_norm"]:
            raise NotImplementedError(f"LimiX layer {layer!r} is not supported")


def _norm(state: dict[str, Array], prefix: str) -> RMSNormParameters:
    return RMSNormParameters(weight=state[f"{prefix}.weight"], epsilon=_EPSILON)


def _query_scale(
    state: dict[str, Array], prefix: str, lower: float, upper: float, base: float
) -> LogScaleParameters:
    """LimiX's soft-clamped scaling, `(1 + w log n) * b`, as an offset and a factor."""
    weight = np.asarray(state[f"{prefix}.scale_linear.weight"], np.float64)[:, 0]
    bias = np.asarray(state[f"{prefix}.scale_linear.bias"], np.float64)
    width = upper - lower
    slope = width / 2 * np.tanh(2 / width * weight) + (upper + lower) / 2
    offset = base / 2 * np.tanh(2 / base * bias) + base / 2
    dtype = state[f"{prefix}.scale_linear.weight"].dtype
    return LogScaleParameters(
        head_scale=jnp.asarray(offset * slope, dtype), offset=jnp.asarray(offset, dtype)
    )


def _bounds(settings: dict[str, Any], kind: str) -> tuple[float, float, float]:
    """A query scaling's clamps, `sa` across rows or `fa` within a row."""
    return (
        float(settings[f"{kind}_temp_lower_bound"]),
        float(settings[f"{kind}_temp_upper_bound"]),
        float(settings[f"{kind}_scale_base_bound"]),
    )


def _attention(
    state: dict[str, Array],
    prefix: str,
    names: tuple[str, str, str, str | None],
    norms: tuple[str, str],
    scale: tuple[str, tuple[float, float, float]],
    num_heads: int,
) -> AttentionParameters:
    """Attention whose projections, norms and scaling are named under `prefix`."""
    query, key, value, output = (
        None if name is None else f"{prefix}.{name}" for name in names
    )
    assert query is not None and key is not None and value is not None
    return AttentionParameters(
        query=read_linear(state, query),
        key=read_linear(state, key),
        value=read_linear(state, value),
        output=None if output is None else read_linear(state, output),
        num_heads=num_heads,
        scale=_query_scale(state, f"{prefix}.{scale[0]}", *scale[1]),
        query_norm=_norm(state, f"{prefix}.{norms[0]}"),
        key_norm=_norm(state, f"{prefix}.{norms[1]}"),
    )


_ACROSS_ROWS = ("q_proj", "k_proj", "v_proj", "out_proj")


def _gated(state: dict[str, Array], prefix: str) -> GatedMLPParameters:
    return GatedMLPParameters(
        gate=read_linear(state, f"{prefix}.gate_proj"),
        up=read_linear(state, f"{prefix}.up_proj"),
        down=read_linear(state, f"{prefix}.down_proj"),
    )


def _layer(state: dict[str, Array], index: int, settings: dict[str, Any]) -> LimiXLayer:
    prefix = f"transformer_encoder.layers.{index}"
    heads = int(settings["nhead"])
    tokens = int(settings["num_cls_tokens"])
    sample = f"{prefix}.sequence_attentions.0"
    feature = f"{prefix}.feature_attentions.0"
    across = ("softmax_scaling_mlp", _bounds(settings, "sa"))
    within = _bounds(settings, "fa")
    return LimiXLayer(
        cells_across_rows=_attention(
            state,
            f"{sample}.x_attention",
            _ACROSS_ROWS,
            ("q_norm", "k_norm"),
            across,
            heads,
        ),
        targets_across_rows=_attention(
            state,
            f"{sample}.y_attention",
            _ACROSS_ROWS,
            ("q_norm", "k_norm"),
            across,
            heads * tokens,
        ),
        cell_feed_forward=_gated(state, f"{prefix}.mlp.0.x_ffn"),
        target_feed_forward=_gated(state, f"{prefix}.mlp.0.y_ffn"),
        cells_within_row=_attention(
            state,
            feature,
            (
                "sample_x_q_proj",
                "sample_all_k_proj",
                "sample_all_v_proj",
                "sample_x_out_proj",
            ),
            ("sample_x_q_norm", "sample_all_k_norm"),
            ("sample_x_softmax_scaling", within),
            heads,
        ),
        targets_within_row=_attention(
            state,
            feature,
            ("yx_y_q_proj", "yx_x_k_proj", "yx_x_v_proj", None),
            ("yx_q_norm", "yx_k_norm"),
            ("yx_softmax_scaling", within),
            heads,
        ),
        targets_readout=read_linear(state, f"{feature}.yx_out_proj"),
    )


def _cells(state: dict[str, Array]) -> CellEmbedding:
    return CellEmbedding(
        value_input=read_linear(state, "encoder_x.0.numeric_mlp.0"),
        value_input_norm=_norm(state, "encoder_x.0.numeric_mlp.1"),
        value_output=read_linear(state, "encoder_x.0.numeric_mlp.3"),
        value_output_norm=_norm(state, "encoder_x.0.numeric_mlp.4"),
        missing=state["encoder_x.0.mask_embedding.weight"][0],
        fusion_input=read_linear(state, "encoder_x.0.fusion_network.0"),
        fusion_input_norm=_norm(state, "encoder_x.0.fusion_network.1"),
        fusion_output=read_linear(state, "encoder_x.0.fusion_network.3"),
        fusion_output_norm=_norm(state, "encoder_x.0.fusion_network.4"),
    )


def _targets(state: dict[str, Array], regression: bool) -> ClassTargets | ValueTargets:
    if not regression:
        return ClassTargets(
            classes=state["cls_y_encoder.2.y_embedding.weight"],
            missing=state["cls_y_encoder.2.y_mask.weight"][0],
        )
    prefix = "reg_y_encoder.0"
    return ValueTargets(
        mlp=MLPParameters(
            input=read_linear(state, f"{prefix}.numeric_encoder.0"),
            output=read_linear(state, f"{prefix}.numeric_encoder.2"),
        ),
        norm=_norm(state, f"{prefix}.numeric_encoder.3"),
        missing=state[f"{prefix}.none_embedding.weight"][0],
    )


def _pattern_linears(pattern: str) -> list[int]:
    """Where a `linear-N_GELU_...` pattern puts its linear layers in its Sequential."""
    parts = pattern.split("_")
    for index, part in enumerate(parts):
        expected = "GELU" if index % 2 else "linear"
        if part.split("-")[0] != expected:
            raise NotImplementedError(f"LimiX MLP pattern {pattern!r} is not supported")
    if len(parts) % 2 == 0:
        raise NotImplementedError(f"LimiX MLP pattern {pattern!r} is not supported")
    return list(range(0, len(parts), 2))


def _head(
    state: dict[str, Array], settings: dict[str, Any], regression: bool
) -> tuple[MLPParameters, RMSNormParameters, tuple[LinearParameters, ...]]:
    task = "reg" if regression else "cls"
    adapter = _pattern_linears(settings[f"{task}_post_adapter_pattern"])
    if len(adapter) != 2:
        raise NotImplementedError(f"LimiX {task}_post_adapter_pattern is not supported")
    prefix = f"{task}_post_adapter"
    head_adapter = MLPParameters(
        input=read_linear(state, f"{prefix}.adapter.{adapter[0]}"),
        output=read_linear(state, f"{prefix}.adapter.{adapter[1]}"),
    )
    decoder = tuple(
        read_linear(state, f"{task}_y_decoder.{index}")
        for index in _pattern_linears(settings[f"{task}_y_decoder_mlp_pattern"])
    )
    return head_adapter, _norm(state, f"{prefix}.layer_norm"), decoder


def limix_from_state(state: dict[str, Array], settings: dict[str, Any]) -> LimiX:
    """Builds LimiX from Stable AI's PyTorch state dict, for one task.

    Args:
        state: Arrays keyed by PyTorch parameter name, through `one_task`.
        settings: The model config, with `arch_version`.

    Returns:
        The model.

    Raises:
        NotImplementedError: If a setting is not supported.
        ValueError: If the state does not hold exactly one task's parameters.
    """
    check_supported(settings)
    regression = "reg_y_decoder.0.weight" in state
    if regression and "cls_y_decoder.0.weight" in state:
        raise ValueError("the state holds both tasks; keep one with `one_task`")

    def build(read: dict[str, Array]) -> LimiX:
        head_adapter, head_norm, decoder = _head(read, settings, regression)
        task = read["add_task_info.token_type_embedding.weight"][int(regression)]
        return LimiX(
            cells=_cells(read),
            column_codes=read_linear(read, "feature_positional_embedding"),
            targets=_targets(read, regression),
            task=task,
            layers=tuple(
                _layer(read, index, settings)
                for index in range(int(settings["nlayers"]))
            ),
            layer_norm=RMSNormParameters(weight=None, epsilon=_EPSILON),
            head_adapter=head_adapter,
            head_norm=head_norm,
            decoder=decoder,
            borders=read["_reg_borders"] if regression else None,
            group_size=int(settings["features_per_group"]),
            target_tokens=int(settings["num_cls_tokens"]),
        )

    return read_all(state, build)


def load_limix(directory: Path) -> LimiX:
    """Loads a LimiX checkpoint directory, as `limix_checkpoint` writes it.

    Args:
        directory: Directory with `config.json` and `model.safetensors`.

    Returns:
        The model.
    """
    metadata = json.loads((directory / "config.json").read_text())
    check_supported(metadata["config"])
    return limix_from_state(*load_state(directory, ARCHITECTURE))
