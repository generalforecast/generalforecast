from __future__ import annotations

import jax.numpy as jnp
import numpy as np
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


def test_pad_rows_fills_rows_after_the_values() -> None:
    values = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)

    padded = pad_rows(values, 5, fill=jnp.nan)

    assert padded.shape == (5, 2)
    np.testing.assert_array_equal(np.asarray(padded[:3]), np.asarray(values))
    assert np.isnan(np.asarray(padded[3:])).all()


def test_pad_rows_keeps_the_dtype_of_targets() -> None:
    padded = pad_rows(jnp.asarray([1, 2], dtype=jnp.int32), 4, fill=0)

    assert padded.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(padded), [1, 2, 0, 0])


def test_pad_rows_refuses_to_truncate() -> None:
    with pytest.raises(ValueError, match="rows"):
        pad_rows(jnp.zeros((5, 2)), 4)
