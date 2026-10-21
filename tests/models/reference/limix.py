# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LimiX-2 from Stable AI's LimiX package, and generalforecast's port, on the same weights."""

from __future__ import annotations

import dataclasses
import functools
import sys
import types
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from generalforecast.models.limix.checkpoint import (
    PUBLISHED,
    limix_checkpoint,
    load_limix,
)
from generalforecast.models.limix.model import LimiX, limix_functions
from generalforecast.models.limix.prepare import kept_groups
from generalforecast.nn.linear import linear

# The seed both sides draw the feature groups' codes with.
CODE_SEED = 7


def _cpu_package() -> None:
    """Lets LimiX's network import and run on a CPU.

    It imports Triton for its GPU norms, which a CPU never calls, and its CPU
    norm path reads a size its torch norm does not keep. The stand-in Triton is
    removed once LimiX has imported, so torch and other tests don't see it.
    """
    stand_in = "triton" not in sys.modules
    if stand_in:
        triton = types.ModuleType("triton")
        language = types.ModuleType("triton.language")
        triton.jit = lambda function=None, **_: function or (lambda inner: inner)
        language.constexpr = object
        triton.language = language
        sys.modules |= {"triton": triton, "triton.language": language}
    try:
        import model.v2_0.loading  # noqa: F401
        from model.v2_0.operators.rmsnorm import RMSNormMixedPrecision
    finally:
        if stand_in:
            del sys.modules["triton"], sys.modules["triton.language"]

    RMSNormMixedPrecision.hs = property(lambda norm: norm.normalized_shape[0])


@functools.cache
def _upstream() -> torch.nn.Module:
    from huggingface_hub import hf_hub_download

    _cpu_package()
    from model.v2_0.loading import load_from_checkpoint

    published = PUBLISHED["limix-2-classifier"]
    path = hf_hub_download(
        published.repository, published.filename, revision=published.revision
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["config"]["rmsnorm_impl"] = "torch"
    model, _ = load_from_checkpoint(checkpoint)
    return model.float().eval()


def _scaled_targets(regression: bool, prepared: Any) -> np.ndarray:
    """Context targets as LimiX's predictor passes them: standardized for regression.

    A masked row's value is NaN, which LimiX's regression encoder masks; a
    masked class takes a label another row holds, so LimiX's ranking of the
    labels is unchanged, and the mask is put on it after encoding.
    """
    targets = prepared.targets[: prepared.context_size].astype(np.float64)
    masked = list(prepared.masked)
    known = np.delete(targets, masked)
    if not regression:
        targets[masked] = known[0]
        return targets.astype(np.float32)
    scaled = (targets - known.mean()) / known.std(ddof=1)
    scaled[masked] = np.nan
    return scaled.astype(np.float32)


def _codes(groups: int, width: int) -> torch.Tensor:
    return torch.randn(
        (groups, width), generator=torch.Generator().manual_seed(CODE_SEED)
    )


def _kept(prepared: Any) -> np.ndarray:
    """The table without the columns LimiX's predictor drops before its network.

    It decides on the context and query rows together; the test tables' constant
    columns are constant over both.
    """
    from inference.v2_0.preprocess import FilterValidFeatures

    step = FilterValidFeatures()
    context = prepared.targets[: prepared.context_size]
    step.fit(prepared.table, [], 0, y=context)
    return step.transform(prepared.table)[0]


def upstream_outputs(regression: bool, prepared: Any) -> np.ndarray:
    """One float32 forward pass of Stable AI's network, which preprocesses itself."""
    model = _upstream()
    hooks = []
    if prepared.masked and not regression:
        # Class targets take LimiX's mask after their embedding, as queries'.
        step = model.cls_y_encoder[2]

        def masked(module: Any, _: Any, output: dict[str, Any]) -> None:
            output[module.out_key][:, list(prepared.masked)] = module.y_mask.weight[0]

        hooks.append(step.register_forward_hook(masked))
    with torch.no_grad():
        outputs = model(
            torch.from_numpy(_kept(prepared))[None],
            torch.from_numpy(_scaled_targets(regression, prepared))[None],
            prepared.context_size,
            task_type="Regression" if regression else "Classification",
            feature_positional_embedding_generator=torch.Generator().manual_seed(
                CODE_SEED
            ),
        )
    for hook in hooks:
        hook.remove()
    logits = outputs["reg_output"][0] if regression else outputs["cls_output"]
    return logits[0].numpy()


def our_outputs(regression: bool, prepared: Any) -> np.ndarray:
    """generalforecast's network on the same rows and codes, converted from the same file."""
    task = "regressor" if regression else "classifier"
    model: LimiX = load_limix(limix_checkpoint(f"limix-2-{task}"))
    context = jnp.asarray(prepared.table[: prepared.context_size])
    targets = jnp.asarray(prepared.targets[: prepared.context_size])
    num_classes = 0
    if not regression:
        targets = targets.astype(jnp.int32)
        num_classes = int(targets.max()) + 1
    # A target not known yet: NaN, or -1 for a class.
    if prepared.masked:
        unknown = jnp.nan if regression else -1
        targets = targets.at[jnp.asarray(prepared.masked)].set(unknown)
    state, _ = limix_functions.fit(model, context, targets, num_classes)
    width = model.column_codes.weight.shape[1]
    # Upstream draws codes for the groups of the kept columns only; the rest
    # are left out of attention, so their codes do not matter.
    groups = int(kept_groups(state.snapshot))
    draws = _codes(groups, width).numpy()
    draws = np.pad(draws, ((0, state.codes.shape[0] - groups), (0, 0)))
    codes = linear(model.column_codes, jnp.asarray(draws))
    state = dataclasses.replace(state, codes=codes)
    rows = limix_functions.embed(model, state, context, targets)
    cache = limix_functions.encode(model, state, rows, targets)
    queries = limix_functions.embed(
        model, state, jnp.asarray(prepared.table[prepared.context_size :])
    )
    return np.asarray(limix_functions.decode(model, state, cache, queries))
