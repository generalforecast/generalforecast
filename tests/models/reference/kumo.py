"""Kumo Tabular from NVIDIA's structured-data-models, and generalforecast's port, on the same weights."""

from __future__ import annotations

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sdm.models.kumo.tabular.model import MODEL_KWARGS, _KumoTabular

from generalforecast.models.kumo.checkpoint import kumo_from_state
from generalforecast.models.kumo.model import kumo_functions, kumo_outputs

REPOSITORY, REVISION = "nvidia/Kumo-Tabular", "v1.0.1"


@functools.cache
def _state(size: str, regression: bool) -> dict[str, torch.Tensor]:
    task = "regressor" if regression else "classifier"
    path = hf_hub_download(REPOSITORY, f"{size}/{task}.pt", revision=REVISION)
    return torch.load(path, map_location="cpu", weights_only=True)


def _targets(regression: bool, prepared: Any) -> np.ndarray:
    return prepared.targets if regression else prepared.targets.astype(np.int64)


def _kept(prepared: Any) -> np.ndarray:
    """Which columns the recipe's DropConstantColumns keeps, fit on the context."""
    import sdm.processing as sp
    from sdm import Stype, TableTensor

    names = [f"column{index}" for index in range(prepared.table.shape[1])]
    context = torch.from_numpy(prepared.table[: prepared.context_size])
    step = sp.DropConstantColumns()
    step.fit(TableTensor(columns={Stype.numerical: names}, numerical=context))
    kept = step.transform(
        TableTensor(columns={Stype.numerical: names}, numerical=context)
    ).columns[Stype.numerical]
    return np.isin(names, kept)


def upstream_outputs(size: str, regression: bool, prepared: Any) -> np.ndarray:
    """One float32 forward pass of NVIDIA's network, without preprocessing.

    Only the columns the recipe keeps go in, as its first step drops the rest.
    """
    kept = _kept(prepared)
    model = _KumoTabular(
        num_classes=0 if regression else 10,
        num_quantiles=999 if regression else 0,
        **MODEL_KWARGS[size],
    )
    model.load_state_dict(_state(size, regression), strict=True)
    with torch.no_grad():
        outputs = model.eval()(
            torch.from_numpy(prepared.table[:, kept]),
            torch.from_numpy(_targets(regression, prepared)),
            torch.from_numpy(prepared.categorical[kept]),
        )
    return outputs.numpy()


def our_outputs(size: str, regression: bool, prepared: Any) -> np.ndarray:
    """generalforecast's network on the same checkpoint and table."""
    state = {
        key: jnp.asarray(value.numpy())
        for key, value in _state(size, regression).items()
    }
    model = kumo_from_state(state, MODEL_KWARGS[size])
    targets = jnp.asarray(_targets(regression, prepared))
    if not regression:
        targets = targets.astype(jnp.int32)
    categorical = (
        jnp.asarray(prepared.categorical) if prepared.categorical.any() else None
    )
    outputs = kumo_outputs(
        model,
        jnp.asarray(prepared.table)[None],
        targets[None],
        prepared.context_size,
        categorical,
    )
    return np.asarray(outputs[0])


def _upstream_prepared(prepared: Any) -> Any:
    """The table after upstream's preprocessing for one estimator, fit on the context.

    Each column is standardized, clipped and soft-clipped; categorical columns
    first become their ranks among the context's categories. The columns stay
    in the order upstream leaves them: numerical, then categorical.
    """
    import sdm
    import sdm.processing as sp

    def scaled() -> list[Any]:
        return [
            sp.Standardize(eps=1e-6),
            sp.Clip(-100.0, 100.0),
            sp.ClipSigma(threshold=4.0),
        ]

    def tensor(rows: np.ndarray) -> Any:
        frame = pd.DataFrame(rows.astype(np.float64))
        frame.columns = [str(index) for index in frame.columns]
        stypes = {
            str(index): "categorical" if categorical else "numerical"
            for index, categorical in enumerate(prepared.categorical)
        }
        for name, stype in stypes.items():
            if stype == "categorical":
                frame[name] = frame[name].astype("category")
        return sdm.TableTensor.from_pandas(frame, stypes)

    pipeline = sp.StypeDispatch(
        numerical=scaled(),
        categorical=[sp.AlignCategories(sort_by="value"), sp.ToNumerical(), *scaled()],
    )
    pipeline.fit(tensor(prepared.table[: prepared.context_size]))
    output = pipeline.transform(tensor(prepared.table))
    names = output.columns[sdm.Stype.numerical]
    categorical = prepared.categorical[[int(name) for name in names]]
    table = output.numerical.numpy().astype(np.float32)
    return prepared._replace(table=table, categorical=categorical)


def upstream_recipe_outputs(size: str, regression: bool, prepared: Any) -> np.ndarray:
    """NVIDIA's preprocessing, then its network, on raw columns."""
    return upstream_outputs(size, regression, _upstream_prepared(prepared))


def our_recipe_outputs(size: str, regression: bool, prepared: Any) -> np.ndarray:
    """generalforecast's whole pipeline on raw columns, before the softmax or quantile scaling."""
    state = {
        key: jnp.asarray(value.numpy())
        for key, value in _state(size, regression).items()
    }
    model = kumo_from_state(state, MODEL_KWARGS[size])
    rows = jnp.asarray(prepared.table)
    context, queries = rows[: prepared.context_size], rows[prepared.context_size :]
    targets = jnp.asarray(_targets(regression, prepared))
    num_classes = 0
    if not regression:
        targets = targets.astype(jnp.int32)
        num_classes = int(targets.max()) + 1
    table, embedded = kumo_functions.fit(
        model,
        context,
        targets,
        num_classes,
        categorical=jnp.asarray(prepared.categorical),
    )
    cache = kumo_functions.encode(model, table, embedded, targets)
    queries = kumo_functions.embed(model, table, queries)
    return np.asarray(kumo_functions.decode(model, table, cache, queries))
