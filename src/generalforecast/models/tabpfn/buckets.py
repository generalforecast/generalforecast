import numpy as np

# Where TabPFN's borders sit, as (bucket, border) for 5000 buckets; the rest
# mirror these around 0 at bucket 2500.
_BORDER_POINTS = (
    (0, -128.0),
    (5, -16.9),
    (20, -13.0),
    (100, -9.9),
    (200, -8.47),
    (500, -6.48),
    (1000, -4.40),
)


def regression_borders(num_buckets: int) -> np.ndarray:
    """TabPFN's bucket borders for `num_buckets` buckets, in the scaled target space.

    Args:
        num_buckets: Number of buckets.

    Returns:
        `num_buckets + 1` ascending float32 borders.
    """
    points = (
        *_BORDER_POINTS,
        (2500, 0.0),
        *((5000 - bucket, -border) for bucket, border in reversed(_BORDER_POINTS)),
    )
    positions = np.asarray([bucket for bucket, _ in points]) * (num_buckets / 5000)
    borders = np.asarray([border for _, border in points])
    return np.interp(np.arange(num_buckets + 1), positions, borders).astype(np.float32)
