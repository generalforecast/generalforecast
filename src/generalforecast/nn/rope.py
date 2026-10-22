# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

Positions = Int[Array, " rows"] | Int[Array, " batch rows"]


class RoPE(NamedTuple):
    """Rotary position embedding settings.

    Attributes:
        theta: RoPE base.
        interleaved: Rotate adjacent pairs (True) or the two halves of each head.
        rotary_dim: How many of each head's dimensions rotate, or None for all.
            They are the first `rotary_dim / 2` frequencies: with the
            contiguous layout, dimensions `0 .. rotary_dim / 2` of each half;
            interleaved, the first `rotary_dim` dimensions. The rest pass
            through unrotated.
    """

    theta: float
    interleaved: bool
    rotary_dim: int | None = None


def _angles(
    values: Float[Array, " batch rows heads head_dim"],
    rope: RoPE,
    positions: Positions | None,
) -> Float[Array, " *batch rows 1 half"]:
    """Rotation angle of each row position and frequency, broadcast over heads.

    Angles are float32: low-precision dtypes cannot represent large positions.
    Frequencies past `rotary_dim` are zero, so those dimensions rotate by nothing.
    """
    rows, head_dim = values.shape[-3], values.shape[-1]
    if positions is None:
        positions = jnp.arange(rows)
    dim = head_dim if rope.rotary_dim is None else rope.rotary_dim
    indices = jnp.arange(0, dim, 2, dtype=jnp.float32)
    inverse_freq = 1.0 / (rope.theta ** (indices / dim))
    inverse_freq = jnp.pad(inverse_freq, (0, head_dim // 2 - inverse_freq.shape[0]))
    angles = positions.astype(jnp.float32)[..., :, None] * inverse_freq
    return angles[..., :, None, :]


def _rotate(
    values: Float[Array, " batch rows heads head_dim"],
    rotated: Float[Array, " batch rows heads head_dim"],
    angles: Float[Array, " rows 1 head_dim"],
) -> Float[Array, " batch rows heads head_dim"]:
    """Combines values and their rotated copy, in the dtype of `values`."""
    cos = jnp.cos(angles).astype(values.dtype)
    sin = jnp.sin(angles).astype(values.dtype)
    return values * cos + rotated * sin


def apply_contiguous_rope(
    values: Float[Array, " batch rows heads head_dim"],
    rope: RoPE,
    positions: Positions | None = None,
) -> Float[Array, " batch rows heads head_dim"]:
    """RoPE that rotates the first half of each head against the second half.

    Args:
        values: Per-head queries or keys.
        rope: RoPE settings.
        positions: Each row's position, or None for its index.

    Returns:
        Rotated values.
    """
    angles = _angles(values, rope, positions)
    angles = jnp.concatenate([angles, angles], axis=-1)
    half = values.shape[-1] // 2
    rotated = jnp.concatenate([-values[..., half:], values[..., :half]], axis=-1)
    return _rotate(values, rotated, angles)


def apply_interleaved_rope(
    values: Float[Array, " batch rows heads head_dim"],
    rope: RoPE,
    positions: Positions | None = None,
) -> Float[Array, " batch rows heads head_dim"]:
    """RoPE that rotates adjacent pairs within each head.

    Args:
        values: Per-head queries or keys.
        rope: RoPE settings.
        positions: Each row's position, or None for its index.

    Returns:
        Rotated values.
    """
    angles = jnp.repeat(_angles(values, rope, positions), 2, axis=-1)
    pairs = values.reshape(*values.shape[:-1], -1, 2)
    rotated = jnp.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(values.shape)
    return _rotate(values, rotated, angles)


def apply_rope(
    values: Float[Array, " batch rows heads head_dim"],
    rope: RoPE,
    positions: Positions | None = None,
) -> Float[Array, " batch rows heads head_dim"]:
    """Applies interleaved or contiguous RoPE.

    Args:
        values: Per-head queries or keys.
        rope: RoPE settings.
        positions: Each row's position, `(rows,)` or one per batch row, or
            None for its index.

    Returns:
        Rotated values.
    """
    if rope.interleaved:
        return apply_interleaved_rope(values, rope, positions)
    return apply_contiguous_rope(values, rope, positions)
