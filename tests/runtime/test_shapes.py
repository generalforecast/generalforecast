# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
import pytest

from generalforecast.runtime.shapes import bucket_size, pad_rows


@pytest.mark.parametrize(
    ("rows", "expected"), [(0, 8), (1, 8), (8, 8), (9, 16), (1000, 1024)]
)
def test_bucket_size_is_the_next_power_of_two(rows: int, expected: int) -> None:
    assert bucket_size(rows) == expected


def test_bucket_size_has_a_minimum() -> None:
    assert bucket_size(3, minimum=4) == 4
    assert bucket_size(5, minimum=4) == 8


def test_bucket_size_refuses_a_zero_minimum() -> None:
    with pytest.raises(ValueError, match="minimum"):
        bucket_size(3, minimum=0)


def test_pad_rows_refuses_to_truncate() -> None:
    with pytest.raises(ValueError, match="rows"):
        pad_rows(jnp.zeros((5, 2)), 4)
