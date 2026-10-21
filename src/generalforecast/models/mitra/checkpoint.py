# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, NamedTuple

from huggingface_hub import hf_hub_download
from jax import Array
from safetensors.numpy import load_file

from generalforecast.models import catalog
from generalforecast.models.artifacts import checkpoint
from generalforecast.models.checkpoints import (
    converted_checkpoint,
    load_state,
    read_all,
    read_linear,
    save_state,
)
from generalforecast.models.mitra.model import ClassTargets, Mitra, MitraLayer
from generalforecast.nn.attention import AttentionBlockParameters, AttentionParameters
from generalforecast.nn.linear import MLPParameters
from generalforecast.nn.norm import LayerNormParameters

ARCHITECTURE = "mitra"


class Published(NamedTuple):
    """A checkpoint in AutoGluon's Hugging Face repositories."""

    repository: str
    revision: str


PUBLISHED = {
    "mitra-classifier": Published(
        "autogluon/mitra-classifier", "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
    ),
    "mitra-regressor": Published(
        "autogluon/mitra-regressor", "5f277aa8f69042d39d6ac3612aed18bb9279bd95"
    ),
    # The upstream revisions the general-forecast mirrors were converted from.
    "mitra-2-classifier": Published(
        "autogluon/mitra-classifier-2", "edada0d20759c58ada8c8605c25f22f6e98ea5f0"
    ),
    "mitra-2-regressor": Published(
        "autogluon/mitra-regressor-2", "246c7d310fff04296a2388df5f448e3ffe809283"
    ),
}

# Settings every release has, with the values the loader is checked against.
SUPPORTED: dict[str, tuple[Any, ...]] = {"task": ("CLASSIFICATION", "REGRESSION")}
# Settings that size the network, which any value may take.
SIZES = ("dim", "dim_output", "n_layers", "n_heads")


def mitra_checkpoint(name: str) -> Path:
    """A Mitra checkpoint directory, converted from the published one if missing.

    With `GF_CHECKPOINTS` set, the directory must already be there, as on
    machines without internet. Otherwise it is kept under `~/.cache/generalforecast`.

    Args:
        name: A key of `PUBLISHED`, such as `mitra-classifier`.

    Returns:
        A directory with `config.json` and `model.safetensors`.
    """
    if name in catalog.ENTRIES and catalog.ENTRIES[name].artifact.files:
        return checkpoint(catalog.ENTRIES[name])
    published = PUBLISHED[name]

    def convert(directory: Path) -> None:
        config, weights = (
            hf_hub_download(published.repository, filename, revision=published.revision)
            for filename in ("config.json", "model.safetensors")
        )
        settings = json.loads(Path(config).read_text())
        save_state(directory, ARCHITECTURE, settings, load_file(weights))

    return converted_checkpoint(name, published.revision, convert)


def check_supported(settings: dict[str, Any]) -> None:
    """Raises for a Mitra setting the loader has not been checked with.

    Args:
        settings: The checkpoint's model config.

    Raises:
        NotImplementedError: Naming the unsupported or unknown setting.
    """
    for key, values in SUPPORTED.items():
        if settings.get(key) not in values:
            raise NotImplementedError(
                f"Mitra {key}={settings.get(key)!r} is not supported"
            )
    unknown = sorted(set(settings) - set(SUPPORTED) - set(SIZES))
    if unknown:
        raise NotImplementedError(f"Mitra settings {unknown} are not supported")
    # Mitra-1's regressor predicts one value, where generalforecast's regressors return
    # quantiles; Mitra-2's predicts a distribution over buckets.
    if settings["task"] == "REGRESSION" and int(settings["dim_output"]) == 1:
        raise NotImplementedError(
            "Mitra task='REGRESSION' with dim_output=1 is not supported"
        )


def _norm(state: dict[str, Array], prefix: str) -> LayerNormParameters:
    return LayerNormParameters(
        weight=state[f"{prefix}.weight"], bias=state[f"{prefix}.bias"]
    )


def _block(
    state: dict[str, Array], prefix: str, step: int, num_heads: int
) -> AttentionBlockParameters:
    """A layer's attention and feed-forward; step 1 is across rows, 2 within a row."""
    attention = f"{prefix}.attention{step}"
    first, second = 2 * step - 1, 2 * step
    return AttentionBlockParameters(
        attention_norm=_norm(state, f"{prefix}.layer_norm{first}"),
        feed_forward_norm=_norm(state, f"{prefix}.layer_norm{second}"),
        attention=AttentionParameters(
            query=read_linear(state, f"{attention}.q"),
            key=read_linear(state, f"{attention}.k"),
            value=read_linear(state, f"{attention}.v"),
            output=read_linear(state, f"{attention}.o"),
            num_heads=num_heads,
        ),
        feed_forward=MLPParameters(
            input=read_linear(state, f"{prefix}.linear{first}"),
            output=read_linear(state, f"{prefix}.linear{second}"),
        ),
    )


def mitra_from_state(state: dict[str, Array], settings: dict[str, Any]) -> Mitra:
    """Builds Mitra from AutoGluon's PyTorch state dict.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        settings: The model config.

    Returns:
        The model.

    Raises:
        NotImplementedError: If a setting is not supported.
    """
    check_supported(settings)
    heads = int(settings["n_heads"])

    def build(read: dict[str, Array]) -> Mitra:
        return Mitra(
            cells=read_linear(read, "x_embedding.x_embedding"),
            targets=ClassTargets(
                classes=read["y_embedding.y_embedding.weight"],
                missing=read["y_embedding.y_mask.weight"][0],
            ),
            layers=tuple(
                MitraLayer(
                    across_rows=_block(read, f"layers.{index}", 1, heads),
                    within_row=_block(read, f"layers.{index}", 2, heads),
                )
                for index in range(int(settings["n_layers"]))
            ),
            final_norm=_norm(read, "final_layer_norm"),
            decoder=read_linear(read, "final_layer"),
            regression=settings["task"] == "REGRESSION",
        )

    return read_all(state, build)


def load_mitra(directory: Path) -> Mitra:
    """Loads a Mitra checkpoint directory, as `mitra_checkpoint` writes it.

    Args:
        directory: Directory with `config.json` and `model.safetensors`.

    Returns:
        The model.
    """
    metadata = json.loads((directory / "config.json").read_text())
    check_supported(metadata["config"])
    return mitra_from_state(*load_state(directory, ARCHITECTURE))
