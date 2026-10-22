# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mitra from AutoGluon's package, and generalforecast's port, on the same weights."""

from __future__ import annotations

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from autogluon.tabular.models.mitra._internal.config.enums import Task
from autogluon.tabular.models.mitra._internal.data.preprocessor import (
    Preprocessor,
)
from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

from generalforecast.models.mitra.checkpoint import (
    PUBLISHED,
    load_mitra,
    mitra_checkpoint,
)
from generalforecast.models.mitra.model import mitra_functions

# A constant column, which AutoGluon drops and generalforecast leaves out of attention.
CONSTANT = 2


def _table(prepared: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Context rows, query rows and context targets, with a constant column added."""
    table = np.insert(prepared.table, CONSTANT, 3.0, axis=1).astype(np.float32)
    context = prepared.context_size
    return table[:context], table[context:], prepared.targets[:context]


@functools.cache
def _upstream(name: str) -> torch.nn.Module:
    from huggingface_hub import snapshot_download

    published = PUBLISHED[name]
    directory = snapshot_download(
        published.repository,
        revision=published.revision,
        allow_patterns=["config.json", "model.safetensors"],
    )
    return Tab2D.from_pretrained(directory, device="cpu").float().eval()


def upstream_outputs(name: str, regression: bool, prepared: Any) -> np.ndarray:
    """AutoGluon's preprocessing and one float32 forward pass, without fine-tuning.

    Mirroring features and targets and shuffling classes are off, as one
    estimator of generalforecast's. Stock AutoGluon's Mitra regressor builds Mitra-1's
    one-output head, so a Mitra-2 regressor's targets go into buckets as
    autogluon/mitra-finetune has them put: through the cross-entropy path of
    AutoGluon's trainer, which that package switches on.
    """
    context, queries, targets = _table(prepared)
    network = _upstream(name)
    outputs = network.dim_output
    task = Task.REGRESSION if regression else Task.CLASSIFICATION
    masked = list(prepared.masked)
    if masked:
        # A masked row takes a target another row holds, so the targets'
        # range and classes are unchanged, and the mask is put on it after.
        targets = targets.copy()
        targets[masked] = np.delete(targets, masked)[0]
    if not regression:
        targets = targets.astype(np.int64)
    preprocessor = Preprocessor(
        dim_embedding=None,
        n_classes=outputs,
        dim_output=outputs,
        use_quantile_transformer=False,
        use_feature_count_scaling=False,
        use_random_transforms=False,
        shuffle_classes=False,
        shuffle_features=False,
        random_mirror_regression=False,
        random_mirror_x=False,
        task=task,
    ).fit(context.copy(), targets)
    support = torch.from_numpy(preprocessor.transform_X(context.copy()))[None]
    query = torch.from_numpy(preprocessor.transform_X(queries.copy()))[None]
    labels = torch.from_numpy(preprocessor.transform_y(targets))[None]
    if regression:
        borders = torch.linspace(-0.5, 1.5, outputs + 1)
        labels = torch.clamp(torch.bucketize(labels, borders) - 1, 0, outputs - 1)
    hooks = []
    if masked:

        def mask(module: Any, _: Any, output: tuple[Any, Any]) -> tuple[Any, Any]:
            support, query = output
            support = support.clone()
            support[:, masked] = module.y_mask.weight[0]
            return support, query

        hooks.append(network.y_embedding.register_forward_hook(mask))
    with torch.no_grad():
        logits = network(
            support,
            labels,
            query,
            torch.zeros(1, support.shape[-1], dtype=torch.bool),
            torch.zeros(1, support.shape[1], dtype=torch.bool),
            torch.zeros(1, query.shape[1], dtype=torch.bool),
        )
    for hook in hooks:
        hook.remove()
    return logits[0].numpy()


def our_outputs(name: str, regression: bool, prepared: Any) -> np.ndarray:
    """generalforecast's preprocessing and network on the same rows, converted from the same files."""
    context, queries, targets = _table(prepared)
    model = load_mitra(mitra_checkpoint(name))
    if regression:
        labels, num_classes = jnp.asarray(targets, jnp.float32), 0
    else:
        labels, num_classes = jnp.asarray(targets, jnp.int32), int(targets.max()) + 1
    # A target not known yet: NaN, or -1 for a class.
    if prepared.masked:
        unknown = jnp.nan if regression else -1
        labels = labels.at[jnp.asarray(prepared.masked)].set(unknown)
    state, rows = mitra_functions.fit(model, jnp.asarray(context), labels, num_classes)
    cache = mitra_functions.encode(model, state, rows, labels)
    query_rows = mitra_functions.embed(model, state, jnp.asarray(queries))
    return np.asarray(mitra_functions.decode(model, state, cache, query_rows))
