# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TabFM 1.0 from Google Research's tabfm package, and generalforecast's port, on the same weights."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from generalforecast.models import catalog
from generalforecast.models.tabfm.checkpoint import tabfm_from_state
from generalforecast.models.tabfm.model import TabFM, tabfm_outputs

NAME = "general-forecast/tabfm-1-classifier"


@functools.cache
def _upstream() -> torch.nn.Module:
    """The package's classifier network in float32, from the pinned weights."""
    from huggingface_hub import snapshot_download
    from tabfm import tabfm_v1_0_0_pytorch

    artifact = catalog.ENTRIES[NAME].artifact
    directory = snapshot_download(
        artifact.repository,
        revision=artifact.revision,
        allow_patterns=["classification/*"],
    )
    return tabfm_v1_0_0_pytorch.load(
        "classification", directory, device="cpu", dtype=None, use_cache=False
    )


@functools.cache
def _ours() -> TabFM:
    """generalforecast's TabFM from the same weights, read as the converted checkpoint holds them."""
    from huggingface_hub import hf_hub_download

    artifact = catalog.ENTRIES[NAME].artifact
    path = hf_hub_download(
        artifact.repository, "classification/config.json", revision=artifact.revision
    )
    settings = json.loads(Path(path).read_text())
    state = {
        key: jnp.asarray(value.float().numpy())
        for key, value in _upstream().state_dict().items()
    }
    return tabfm_from_state(state, settings)


def _kept(prepared: Any) -> np.ndarray:
    """The table without the columns constant in the context, which the package's filter drops."""
    context = prepared.table[: prepared.context_size]
    varying = np.array([len(np.unique(column)) > 1 for column in context.T], dtype=bool)
    return prepared.table[:, varying]


def upstream_outputs(prepared: Any) -> np.ndarray:
    """The package's logits for the query rows over every class slot, in one forward pass."""
    table = torch.from_numpy(_kept(prepared))[None]
    labels = torch.full(table.shape[:2], -100.0)
    labels[0, : prepared.context_size] = torch.from_numpy(prepared.targets)
    with torch.no_grad():
        logits = _upstream()(table, labels, torch.tensor([prepared.context_size]))
    return logits[0, prepared.context_size :].numpy()


def our_outputs(prepared: Any) -> np.ndarray:
    """generalforecast's logits for the query rows on the same prepared table."""
    outputs = tabfm_outputs(
        _ours(),
        jnp.asarray(prepared.table)[None],
        jnp.asarray(prepared.targets, jnp.int32)[None],
        prepared.context_size,
    )
    return np.asarray(outputs[0])
