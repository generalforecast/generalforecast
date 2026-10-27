# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the models' contract calls share: their compute dtype, table state, class checks and tokens."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from generalforecast.models.contract import TabularProperties


def compute_dtype(model: Any) -> jnp.dtype:
    """The dtype a network computes in: its parameters', which configuring a model casts alike.

    Args:
        model: Model parameters, real or abstract.

    Returns:
        The dtype of its first floating-point parameter.

    Raises:
        ValueError: If it has none.
    """
    for leaf in jax.tree.leaves(model):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            return jnp.dtype(leaf.dtype)
    raise ValueError("the model has no floating-point parameters")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TableState:
    """State fit on every context row: its preprocessing, and what a model keeps of each column.

    Each row's representation depends on it, so changing it changes every row.
    `num_classes` is static: postprocessing slices logits by it, so it must be a
    Python int under jit, and a new value compiles a new program.

    Attributes:
        snapshot: The model's preprocessing, fit on the context.
        columns: What the model keeps of each column, such as a column
            summary, or None.
        num_classes: Classes of the task, or 0 for regression.
    """

    snapshot: Any
    columns: Any
    num_classes: int = field(metadata={"static": True})


def check_class_count(properties: TabularProperties, num_classes: int) -> None:
    """Raises if a model with these properties cannot serve `num_classes` classes.

    Args:
        properties: The model's properties.
        num_classes: Number of classes in the task, or 0 for regression.

    Raises:
        ValueError: If a regressor gets classes, or a classifier more than
            its slots or none.
    """
    if properties.regression:
        if num_classes != 0:
            raise ValueError(f"regressors take num_classes=0, got {num_classes}")
        return
    slots = properties.class_slots
    if not 1 <= num_classes <= slots:
        raise ValueError(f"num_classes must be in 1 .. {slots}, got {num_classes}")


def as_tokens(rows: Array, width: int) -> Float[Array, " 1 rows tokens width"]:
    """Row representations, tokens of `width` side by side, as tokens in a batch of one."""
    return rows.reshape(1, rows.shape[0], -1, width)
