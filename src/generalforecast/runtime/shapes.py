# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import jax.numpy as jnp
from jax import Array

# The scheduler in generalforecast._core decides padded sizes; Python pads to them.
from generalforecast._core import bucket_size

__all__ = ["bucket_size", "bucketed", "pad_rows"]


def pad_rows(values: Array, size: int, fill: float = 0) -> Array:
    """Pads the leading axis to `size` with `fill`, keeping the dtype.

    Args:
        values: Rows to pad.
        size: Padded number of rows.
        fill: Value of the padded rows.

    Returns:
        The padded rows.
    """
    rows = values.shape[0]
    if size < rows:
        raise ValueError(f"cannot pad {rows} rows to {size} rows")
    tail = jnp.full((size - rows, *values.shape[1:]), fill, dtype=values.dtype)
    return jnp.concatenate([values, tail])


def bucketed(items: list[Any]) -> list[Any]:
    """Pads items to a bucket by repeating the last, so each bucket compiles once."""
    return padded(items, bucket_size(len(items), 1))


def padded(items: list[Any], count: int) -> list[Any]:
    """Pads items to `count` by repeating the last, so a call of `count` items serves them."""
    return items + items[-1:] * (count - len(items))
