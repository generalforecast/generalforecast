# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TabPFN from Prior Labs' tabpfn package, and generalforecast's port, on the same weights."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from generalforecast.models.tabpfn.checkpoint import (
    PUBLISHED,
    load_tabpfn,
    read_published,
    tabpfn_checkpoint,
)
from generalforecast.models.tabpfn.model import tabpfn_outputs


@functools.cache
def _upstream(name: str) -> torch.nn.Module:
    published = PUBLISHED[name]
    path = hf_hub_download(
        published.repository, published.filename, revision=published.revision
    )
    architecture, settings, state = read_published(Path(path))
    if architecture == "tabpfn_v3":
        from tabpfn.architectures import tabpfn_v3 as module
    else:
        from tabpfn.architectures import tabpfn_v3_5 as module
    config = {
        key: value for key, value in settings.items() if key != "inference_config"
    }
    model = module.get_architecture(module.parse_config(config)[0])
    model.load_state_dict(
        {key: torch.from_numpy(np.array(value)) for key, value in state.items()},
        strict=True,
    )
    return model.eval()


def _kept(prepared: Any) -> np.ndarray:
    """The table without the columns TabPFN's preprocessing drops on the context."""
    from tabpfn.preprocessing.datamodel import FeatureSchema
    from tabpfn.preprocessing.steps import RemoveConstantFeaturesStep

    context = prepared.table[: prepared.context_size]
    schema = FeatureSchema.from_only_categorical_indices([], context.shape[1])
    step = RemoveConstantFeaturesStep()
    step.fit_transform(context, schema)
    return step.transform(prepared.table).X


def upstream_outputs(name: str, regression: bool, prepared: Any) -> np.ndarray:
    """One float32 forward pass of Prior Labs' network, which preprocesses itself.

    Its estimators drop constant columns before the network, so they go first.
    """
    model = _upstream(name)
    table = _kept(prepared)
    # TabPFN-3.5 serves both tasks and takes which one per call.
    task = {}
    if name.startswith("tabpfn-3.5"):
        task = {"task_type": "regression" if regression else "multiclass"}
    with torch.no_grad():
        outputs = model(
            torch.from_numpy(table[:, None, :]),
            torch.from_numpy(prepared.targets[: prepared.context_size]),
            **task,
        )
    return outputs[:, 0].numpy()


def our_outputs(name: str, regression: bool, prepared: Any) -> np.ndarray:
    """generalforecast's network on the same rows, converted from the same published file."""
    model = load_tabpfn(tabpfn_checkpoint(name))
    targets = jnp.asarray(prepared.targets)
    if not regression:
        targets = targets.astype(jnp.int32)
    outputs = tabpfn_outputs(
        model, jnp.asarray(prepared.table)[None], targets[None], prepared.context_size
    )
    return np.asarray(outputs[0])
