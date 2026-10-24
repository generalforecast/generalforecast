# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from huggingface_hub import hf_hub_download
from jax import Array
from safetensors import safe_open
from safetensors.numpy import load_file

from generalforecast.models.catalog import short_name
from generalforecast.models.checkpoints import (
    converted_checkpoint,
    load_state,
    read_all,
    read_torch_checkpoint,
    save_state,
)
from generalforecast.models.column_embedding import (
    ColumnEmbeddingParameters,
    FeatureGrouping,
)
from generalforecast.models.icl import ICLParameters
from generalforecast.models.row_interaction import RowInteractionParameters
from generalforecast.models.tabpfn.buckets import regression_borders
from generalforecast.models.tabpfn.model import TabPFN
from generalforecast.nn.attention import AttentionBlockParameters, AttentionParameters
from generalforecast.nn.fourier import FourierCellParameters
from generalforecast.nn.induced import InducedAttentionParameters
from generalforecast.nn.linear import (
    LinearParameters,
    MLPParameters,
    ResidualMLPParameters,
)
from generalforecast.nn.norm import LayerNormParameters, RMSNormParameters
from generalforecast.nn.qassmax import QASSMaxParameters
from generalforecast.nn.retrieval import RetrievalParameters
from generalforecast.nn.rope import RoPE
from generalforecast.nn.target import ClassEncoder, TargetEncoder, ValueEncoder


class Published(NamedTuple):
    """A checkpoint file in Prior Labs' Hugging Face repositories."""

    repository: str
    revision: str
    filename: str


TABPFN_3 = ("Prior-Labs/tabpfn_3", "24a16a89d245878b846555110985634aa2e656d7")
TABPFN_3_5 = ("Prior-Labs/tabpfn_3_5", "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e")
PUBLISHED = {
    "general-forecast/tabpfn-3-classifier": Published(
        *TABPFN_3, "tabpfn-v3-classifier-v3_default.ckpt"
    ),
    "general-forecast/tabpfn-3-regressor": Published(
        *TABPFN_3, "tabpfn-v3-regressor-v3_default.ckpt"
    ),
    # One file serves both tasks; each converts into its own directory.
    "general-forecast/tabpfn-3.5-classifier": Published(
        *TABPFN_3_5, "tabpfn-v3.5-20260909.safetensors"
    ),
    "general-forecast/tabpfn-3.5-regressor": Published(
        *TABPFN_3_5, "tabpfn-v3.5-20260909.safetensors"
    ),
}
# A multitask checkpoint's parameters for one task only, by whether it regresses.
_TASK_ONLY = {
    False: (
        "col_y_encoder.multiclass.",
        "icl_y_encoder.multiclass.",
        "heads.many_class_decoder.",
        "heads.mlp_classification.",
    ),
    True: (
        "col_y_encoder.regression.",
        "icl_y_encoder.regression.",
        "heads.output_projection.",
        "heads.mlp_regression.",
        "heads.regression_borders",
    ),
}


def one_task(state: dict[str, Any], regression: bool) -> dict[str, Any]:
    """A multitask checkpoint's state without the other task's parameters.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        regression: Whether to keep the regression task rather than classification.

    Returns:
        The state for one task; a single-task state is returned as it is.
    """
    other = _TASK_ONLY[not regression]
    return {key: value for key, value in state.items() if not key.startswith(other)}


def read_published(path: Path) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    """Reads a published TabPFN file.

    Args:
        path: A TabPFN-3 `.ckpt` or TabPFN-3.5 `.safetensors` file.

    Returns:
        The architecture name, the settings (the model config, plus the
        `inference_config` that describes preprocessing), and the arrays.
    """
    if path.suffix == ".ckpt":
        stored = read_torch_checkpoint(path)
        metadata = {key: stored[key] for key in ("config", "inference_config")}
        return stored["architecture_name"], _settings(metadata), stored["state_dict"]
    with safe_open(path, "np") as handle:
        stored = {key: json.loads(value) for key, value in handle.metadata().items()}
    return stored["architecture_name"], _settings(stored), load_file(path)


def _settings(metadata: dict[str, Any]) -> dict[str, Any]:
    return {**metadata["config"], "inference_config": metadata["inference_config"]}


def tabpfn_checkpoint(id: str) -> Path:
    """A TabPFN checkpoint directory, converted from the published file if missing.

    With `GF_CHECKPOINTS` set, the directory must already be there, as on
    machines without internet. Otherwise it is kept under `~/.cache/generalforecast`.

    Args:
        id: A key of `PUBLISHED`, such as `general-forecast/tabpfn-3-classifier`.

    Returns:
        A directory with `config.json` and `model.safetensors`.
    """
    published = PUBLISHED[id]
    name = short_name(id)

    def convert(directory: Path) -> None:
        path = hf_hub_download(
            published.repository, published.filename, revision=published.revision
        )
        architecture, settings, state = read_published(Path(path))
        state = one_task(state, regression=name.endswith("-regressor"))
        save_state(directory, architecture, settings, state)

    return converted_checkpoint(name, published.revision, convert)


# Settings every release has, with the values the loader is checked against.
SUPPORTED: dict[str, tuple[Any, ...]] = {
    "feature_group_size": (3,),
    "use_rope": (True,),
    "use_nan_indicators": (True,),
    "icl_num_kv_heads": (None,),
    "ff_factor": (2,),
}
# Settings only some releases have, by architecture.
RELEASE_SUPPORTED: dict[str, dict[str, tuple[Any, ...]]] = {
    "tabpfn_v3": {},
    "tabpfn_v3_5": {
        "use_fourier_cell_embedding": (True,),
        "use_mlp_heads": (True,),
        "use_qk_norm": (True,),
        "y_encoder_layernorm": (True,),
        "feat_agg_use_softmax_scaling": (False,),
    },
}


def check_supported(settings: dict[str, Any], architecture: str = "tabpfn_v3") -> None:
    """Raises for a TabPFN setting whose value the loader has not been checked with.

    Args:
        settings: The checkpoint's model config.
        architecture: The checkpoint's architecture name.

    Raises:
        NotImplementedError: Naming the unsupported architecture or setting.
    """
    if architecture not in RELEASE_SUPPORTED:
        raise NotImplementedError(
            f"TabPFN architecture {architecture!r} is not supported"
        )
    for key, values in {**SUPPORTED, **RELEASE_SUPPORTED[architecture]}.items():
        if settings.get(key) not in values:
            raise NotImplementedError(
                f"TabPFN {key}={settings.get(key)!r} is not supported"
            )


def _linear(state: dict[str, Array], prefix: str) -> LinearParameters:
    """Linear parameters stored under `prefix`, with an optional bias."""
    return LinearParameters(
        weight=state[f"{prefix}.weight"], bias=state.get(f"{prefix}.bias")
    )


def _mlp(state: dict[str, Array], prefix: str) -> MLPParameters:
    """A two-layer MLP stored as a PyTorch `Sequential` under `prefix`."""
    return MLPParameters(
        input=_linear(state, f"{prefix}.0"), output=_linear(state, f"{prefix}.2")
    )


def _norm(state: dict[str, Array], prefix: str) -> RMSNormParameters:
    """An RMSNorm under `prefix`, without a scale if the checkpoint has none."""
    return RMSNormParameters(weight=state.get(f"{prefix}.weight"))


def _head_norm(state: dict[str, Array], prefix: str) -> RMSNormParameters | None:
    """A per-head query or key RMSNorm under `prefix`, if the release has one."""
    if f"{prefix}.weight" not in state:
        return None
    return _norm(state, prefix)


def _layer_norm(state: dict[str, Array], prefix: str) -> LayerNormParameters | None:
    """A LayerNorm under `prefix`, if the release has one."""
    if f"{prefix}.weight" not in state:
        return None
    return LayerNormParameters(
        weight=state[f"{prefix}.weight"], bias=state.get(f"{prefix}.bias")
    )


def _scale(state: dict[str, Array], prefix: str) -> QASSMaxParameters | None:
    """Query scaling stored under `prefix`, if any."""
    if f"{prefix}.base_mlp.0.weight" not in state:
        return None
    return QASSMaxParameters(
        base=_mlp(state, f"{prefix}.base_mlp"), query=_mlp(state, f"{prefix}.query_mlp")
    )


def _attention(
    state: dict[str, Array], prefix: str, num_heads: int, rope: RoPE | None = None
) -> AttentionParameters:
    return AttentionParameters(
        query=_linear(state, f"{prefix}.q_projection"),
        key=_linear(state, f"{prefix}.k_projection"),
        value=_linear(state, f"{prefix}.v_projection"),
        output=_linear(state, f"{prefix}.out_projection"),
        num_heads=num_heads,
        scale=_scale(state, f"{prefix}.softmax_scaling_layer"),
        query_norm=_head_norm(state, f"{prefix}.q_norm"),
        key_norm=_head_norm(state, f"{prefix}.k_norm"),
        rope=rope,
    )


def _cross_block(
    state: dict[str, Array], prefix: str, num_heads: int
) -> AttentionBlockParameters:
    """A block whose queries and context have their own norms."""
    return AttentionBlockParameters(
        attention_norm=_norm(state, f"{prefix}.layernorm_q"),
        feed_forward_norm=_norm(state, f"{prefix}.layernorm2"),
        attention=_attention(state, f"{prefix}.attn", num_heads),
        feed_forward=_mlp(state, f"{prefix}.mlp"),
        context_norm=_norm(state, f"{prefix}.layernorm_kv"),
    )


def _block(
    state: dict[str, Array],
    prefix: str,
    attention: str,
    num_heads: int,
    rope: RoPE | None = None,
) -> AttentionBlockParameters:
    """A block whose queries and context share one norm."""
    return AttentionBlockParameters(
        attention_norm=_norm(state, f"{prefix}.layernorm"),
        feed_forward_norm=_norm(state, f"{prefix}.layernorm_mlp"),
        attention=_attention(state, f"{prefix}.{attention}", num_heads, rope),
        feed_forward=_mlp(state, f"{prefix}.mlp"),
    )


def _target_encoder(
    state: dict[str, Array], prefix: str, norm: str, regression: bool
) -> TargetEncoder:
    """A label embedding table as a class encoder, or a scalar target projection."""
    multitask = any(
        key.startswith((f"{prefix}.multiclass.", f"{prefix}.regression."))
        for key in state
    )
    if regression:
        name = f"{prefix}.regression" if multitask else prefix
        return ValueEncoder(linear=_linear(state, name), norm=_layer_norm(state, norm))
    name = f"{prefix}.multiclass" if multitask else prefix
    table = state[f"{name}.embedding.weight"]
    return ClassEncoder(
        linear=LinearParameters(weight=table.T, bias=None),
        norm=_layer_norm(state, norm),
    )


def _cells(
    state: dict[str, Array], settings: dict[str, Any]
) -> LinearParameters | FourierCellParameters:
    """A linear cell projection, or Fourier cell features in multitask releases."""
    if "x_embed.fourier.frequencies" not in state:
        return _linear(state, "x_embed")
    return FourierCellParameters(
        frequencies=state["x_embed.fourier.frequencies"],
        fourier=_linear(state, "x_embed.fourier.in_linear"),
        channels=_linear(state, "x_embed.metadata_linear"),
        norm=LayerNormParameters(
            weight=state["x_embed.layernorm.weight"],
            bias=state["x_embed.layernorm.bias"],
        ),
        rank_frequencies=int(settings["cell_ecdf_num_frequencies"]),
    )


def _columns(
    state: dict[str, Array], settings: dict[str, Any], regression: bool
) -> ColumnEmbeddingParameters:
    heads = int(settings["dist_embed_num_heads"])
    prefix = "feature_distribution_embedder.layers"
    return ColumnEmbeddingParameters(
        input=_cells(state, settings),
        blocks=tuple(
            InducedAttentionParameters(
                inducing=state[f"{prefix}.{index}.inducing_vectors"],
                compress=_cross_block(
                    state, f"{prefix}.{index}.cross_attn_block1", heads
                ),
                expand=_cross_block(
                    state, f"{prefix}.{index}.cross_attn_block2", heads
                ),
            )
            for index in range(int(settings["dist_embed_num_blocks"]))
        ),
        target_encoder=_target_encoder(
            state, "col_y_encoder", "col_y_layernorm", regression
        ),
        grouping=FeatureGrouping("same", int(settings["feature_group_size"])),
    )


def _rows(
    state: dict[str, Array], settings: dict[str, Any]
) -> RowInteractionParameters:
    heads = int(settings["feat_agg_num_heads"])
    rope = RoPE(theta=float(int(settings["feat_agg_rope_base"])), interleaved=False)
    return RowInteractionParameters(
        cls_tokens=state["column_aggregator.cls_tokens"],
        blocks=tuple(
            _block(state, f"column_aggregator.blocks.{index}", "attention", heads, rope)
            for index in range(int(settings["feat_agg_num_blocks"]))
        ),
        output_norm=_norm(state, "column_aggregator.out_ln"),
    )


def _retrieval(
    state: dict[str, Array], prefix: str, settings: dict[str, Any]
) -> RetrievalParameters:
    return RetrievalParameters(
        query=_linear(state, f"{prefix}.q_projection"),
        key=_linear(state, f"{prefix}.k_projection"),
        num_heads=int(settings["decoder_num_heads"]),
        scale=_scale(state, f"{prefix}.softmax_scaling_layer"),
    )


def _decoder(
    state: dict[str, Array], settings: dict[str, Any], regression: bool
) -> tuple[
    MLPParameters | LinearParameters | RetrievalParameters,
    ResidualMLPParameters | None,
]:
    """The decoder, and the block before it in multitask releases."""
    if not any(key.startswith("heads.") for key in state):
        if regression:
            return _mlp(state, "output_projection"), None
        return _retrieval(state, "many_class_decoder", settings), None
    block = "heads.mlp_regression" if regression else "heads.mlp_classification"
    head = ResidualMLPParameters(
        norm=_norm(state, f"{block}.norm"), mlp=_mlp(state, f"{block}.mlp")
    )
    if regression:
        return _linear(state, "heads.output_projection"), head
    return _retrieval(state, "heads.many_class_decoder", settings), head


def _icl(
    state: dict[str, Array], settings: dict[str, Any], regression: bool
) -> ICLParameters:
    heads = int(settings["icl_num_heads"])
    decoder, head = _decoder(state, settings, regression)
    kv_heads = settings.get("icl_num_kv_heads_test")
    return ICLParameters(
        blocks=tuple(
            _block(state, f"icl_blocks.{index}", "icl_attention", heads)
            for index in range(int(settings["nlayers"]))
        ),
        target_encoder=_target_encoder(
            state, "icl_y_encoder", "icl_y_layernorm", regression
        ),
        output_norm=_norm(state, "output_norm"),
        decoder=decoder,
        query_kv_heads=None if kv_heads is None else int(kv_heads),
        head=head,
    )


def tabpfn_from_state(
    state: dict[str, Array], settings: dict[str, Any], architecture: str = "tabpfn_v3"
) -> TabPFN:
    """Builds TabPFN from Prior Labs' PyTorch state dict, for one task.

    Args:
        state: Arrays keyed by PyTorch parameter name; a multitask checkpoint's
            state goes through `one_task` first.
        settings: The model config, with the published `inference_config`.
        architecture: The checkpoint's architecture name.

    Returns:
        The model.

    Raises:
        NotImplementedError: If a setting is not supported.
        ValueError: If the state does not hold exactly one task's parameters.
    """
    check_supported(settings, architecture)
    inference = settings.get("inference_config") or {}
    regression = any(
        key in state
        for key in ("output_projection.0.weight", "heads.output_projection.weight")
    )
    if regression and "heads.many_class_decoder.q_projection.weight" in state:
        raise ValueError("the state holds both tasks; keep one with `one_task`")

    def build(read: dict[str, Array]) -> TabPFN:
        num_buckets = None
        if regression:
            num_buckets = int(settings["num_buckets"])
            stored = read.get("heads.regression_borders")
            if stored is None:
                stored = read["regression_borders"]
            if not np.allclose(np.asarray(stored), regression_borders(num_buckets)):
                raise ValueError("the checkpoint's bucket borders are not TabPFN's")
        ranks = settings.get("cell_ecdf_num_buckets")
        return TabPFN(
            columns=_columns(read, settings, regression),
            rows=_rows(read, settings),
            icl=_icl(read, settings, regression),
            num_buckets=num_buckets,
            temperature=float(inference.get("SOFTMAX_TEMPERATURE", 0.9)),
            rank_buckets=None if ranks is None else int(ranks),
        )

    # A classifier keeps unused borders; regressors read and check theirs.
    return read_all(state, build, buffers=(".rope.freqs", "regression_borders"))


def load_tabpfn(directory: Path) -> TabPFN:
    """Loads a TabPFN checkpoint directory, as `tabpfn_checkpoint` writes it.

    Args:
        directory: Directory with `config.json` and `model.safetensors`.

    Returns:
        The model.
    """
    metadata = json.loads((directory / "config.json").read_text())
    architecture = metadata["architecture_name"]
    check_supported(metadata["config"], architecture)
    return tabpfn_from_state(*load_state(directory, architecture), architecture)
