# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a hosted plane's trainer builds on: a catalog model's calls and parameters.

A plane fine-tunes a catalog model with its own training code. gf gives it
the model as the engine runs it, so what it trains is what gf serves, and
the parameter types it adapts. It reaches this module as
`generalforecast.hosting.training`; `VERSION` changes with any change to
what this module offers.

gf holds no training code: how a trainer samples, scores and updates is the
plane's, and what it publishes comes back as a derived model's weights
(`generalforecast.hosting.DerivedWeights`).

Exports:
    VERSION: this surface's version.
    load: a catalog model's contract calls and parameters.
    for_training: a model's parameters as a trainer differentiates them:
        each layer rematerialized for the backward pass, and attention's
        row chunking off.
    encode: a table's rows as the model's calls read them when gf serves
        a predictor of that table: features, targets, classes and the
        categorical mask `fit` takes.
    Encoded: what `encode` answers.
    TabularFunctions: a tabular model's contract calls.
    LinearParameters: a linear layer's weight, in the (output, input)
        layout, and bias: what low-rank fine-tuning adapts.
    COMPONENTS: the components full fine-tuning trains by name.
    COMPONENT_MAPS: each model family's components, as prefixes of its
        parameters' paths; a family missing from it cannot be fully
        fine-tuned.
    trainable: the leaves of a model's parameters full fine-tuning trains
        for some components, by path; every other leaf stays as loaded.
    replaced: a model's parameters with trained leaves, by path, in place.
    FULL_FORMAT, pack_full, unpack_full: a full fine-tune's stored
        weights, `full/1`, and how gf runs them as a derived model of kind
        `full` (`generalforecast.models.full`).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

    from generalforecast.models.contract import TabularFunctions
    from generalforecast.models.full import COMPONENTS, trainable
    from generalforecast.models.full import FORMAT as FULL_FORMAT
    from generalforecast.models.full import MAPS as COMPONENT_MAPS
    from generalforecast.models.full import combine as replaced
    from generalforecast.models.full import pack as pack_full
    from generalforecast.models.full import unpack as unpack_full
    from generalforecast.nn.linear import LinearParameters

VERSION = 4

# Full fine-tuning's names here, and theirs in `generalforecast.models.full`.
_FULL = {
    "COMPONENTS": "COMPONENTS",
    "COMPONENT_MAPS": "MAPS",
    "FULL_FORMAT": "FORMAT",
    "trainable": "trainable",
    "replaced": "combine",
    "pack_full": "pack",
    "unpack_full": "unpack",
}


def __getattr__(name: str) -> Any:
    """The parameter types and full fine-tuning, imported on first use: `hosting` stays free of JAX."""
    if name == "TabularFunctions":
        from generalforecast.models.contract import TabularFunctions

        return TabularFunctions
    if name == "LinearParameters":
        from generalforecast.nn.linear import LinearParameters

        return LinearParameters
    if name in _FULL:
        from generalforecast.models import full

        return getattr(full, _FULL[name])
    raise AttributeError(name)


def load(model: str, *, allow_downloads: bool = True) -> "tuple[TabularFunctions, Any]":
    """A catalog tabular model's contract calls and verified parameters.

    The model loads as hosted predictors load it, under the predictor
    surface's gates, so a trainer only fine-tunes what gf serves.

    Raises:
        ValueError: If `model` is not a catalog tabular model qualified for
            hosted predictors.
    """
    from generalforecast.models import registry
    from generalforecast.models.contract import TabularFunctions

    functions, parameters = registry.load(
        model, surface="predictor", allow_downloads=allow_downloads
    )
    if not isinstance(functions, TabularFunctions):
        raise ValueError(f"{model} is not a tabular model")  # noqa: TRY004
    return functions, parameters


def for_training(parameters: Any, *, remat: bool = True, chunked: bool = False) -> Any:
    """A model's parameters as a trainer runs them under `jax.grad`: the same arrays, other kernels.

    With `remat`, each layer keeps only its inputs for the backward pass and
    computes the rest again (`AttentionKernel.remat`): a step's memory is
    then about one layer's activations rather than every layer's. Without
    `chunked`, attention takes every position at once (`block_chunk`
    None): row chunking bounds a forward pass's memory, but under
    `jax.grad` its chunks become a scan that keeps every chunk's
    residuals. Neither changes what the model computes, so the parameters
    are served as loaded and only a trainer's step runs these.
    """
    import dataclasses

    import jax

    from generalforecast.nn.attention import AttentionParameters

    def kernel(node: Any) -> Any:
        if not isinstance(node, AttentionParameters):
            return node
        changed = dataclasses.replace(node.kernel, remat=remat)
        if not chunked:
            changed = dataclasses.replace(changed, block_chunk=None)
        return dataclasses.replace(node, kernel=changed)

    return jax.tree.map(
        kernel, parameters, is_leaf=lambda node: isinstance(node, AttentionParameters)
    )


@dataclass(frozen=True)
class Encoded:
    """A table's rows as a served predictor's model reads them.

    Attributes:
        task: `classification` or `regression`, as the model predicts.
        columns: The input columns, in the order their features take.
        features: The labeled rows' features, float32 `(rows, width)`; a
            datetime input takes several features, a missing value is NaN.
        targets: Each row's class code into `classes`, int32, or its
            value, float32.
        classes: The class labels in the model's output order; empty for
            a regression.
        categorical: The mask of category-code features that `fit` takes
            when gf serves the rows: None, as a predictor's bound table
            passes its features as numbers.
    """

    task: str
    columns: tuple[str, ...]
    features: "np.ndarray"
    targets: "np.ndarray"
    classes: tuple[Any, ...]
    categorical: "np.ndarray | None"


def encode(model: str, table: "pa.Table", target: str) -> Encoded:
    """A table's labeled rows as `model` reads them when gf serves them.

    The rows are a predictor's context for `model` and `target`, with every
    other column an input, as an ad hoc call or a pushed predictor that
    declares no `inputs` reads them. They are read and bound by the very
    code serving runs: the kinds the predictor's check infers, the context
    rows it serves (`served_context`), the schema it compiles and the
    binder its tabular contexts bind through. So a trainer that trains on
    these numbers trains on what a fine-tune is served: categorical words
    coded in the order they first appear, booleans as 0 and 1, datetimes
    as their parts, missing values as NaN, classes in sorted order and
    unlabeled rows left out. Unlike serving, the rows are not held to the
    model's context row cap: a trainer samples its own contexts.

    Raises:
        InvalidInput: For a target or table the predictor's check refuses.
        PredictorError: For rows the model cannot read or the binder refuses.
    """
    import numpy as np

    from generalforecast.hosting import ad_hoc_lock
    from generalforecast.predictors.runtime.compile import compile_lock
    from generalforecast.predictors.runtime.context import served_context
    from generalforecast.predictors.runtime.schema import infer_schema
    from generalforecast.predictors.runtime.tabular import bind_rows
    from generalforecast.predictors.runtime.validation import isolated
    from generalforecast.predictors.validation import PredictorError

    lock = ad_hoc_lock({"model": model, "target": target}, table)
    plan = compile_lock(lock)
    if plan.task not in ("classification", "regression"):
        raise PredictorError("UnsupportedCapability", f"{model} is not tabular")
    rows = served_context(lock, isolated(table))
    assert rows is not None
    schema = infer_schema(plan.spec, plan.model, plan.task, rows)
    context = bind_rows(plan, rows, schema)
    classes = tuple(context.classes)
    return Encoded(
        task=plan.task,
        columns=schema.features,
        features=np.asarray(context.features, np.float32),
        targets=np.asarray(context.targets, np.int32 if classes else np.float32),
        classes=classes,
        categorical=None,
    )


__all__ = [
    "COMPONENTS",
    "COMPONENT_MAPS",
    "FULL_FORMAT",
    "VERSION",
    "Encoded",
    "LinearParameters",
    "TabularFunctions",
    "encode",
    "for_training",
    "load",
    "pack_full",
    "replaced",
    "trainable",
    "unpack_full",
]
