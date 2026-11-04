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
    encode: a table's rows as the model's calls read them when gf serves
        a predictor of that table: features, targets, classes and the
        categorical mask `fit` takes.
    Encoded: what `encode` answers.
    TabularFunctions: a tabular model's contract calls.
    LinearParameters: a linear layer's weight, in the (output, input)
        layout, and bias: what low-rank and full fine-tuning adapt.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

    from generalforecast.models.contract import TabularFunctions
    from generalforecast.nn.linear import LinearParameters

VERSION = 2


def __getattr__(name: str) -> Any:
    """The parameter types, imported on first use: `hosting` stays free of JAX."""
    if name == "TabularFunctions":
        from generalforecast.models.contract import TabularFunctions

        return TabularFunctions
    if name == "LinearParameters":
        from generalforecast.nn.linear import LinearParameters

        return LinearParameters
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
    "VERSION",
    "Encoded",
    "LinearParameters",
    "TabularFunctions",
    "encode",
    "load",
]
