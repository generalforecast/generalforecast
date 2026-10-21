# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Seeded synthetic tables shared by the tests."""

from typing import NamedTuple

import numpy as np


class SyntheticTable(NamedTuple):
    """Context rows with both kinds of label, plus unlabeled query rows."""

    features: np.ndarray
    queries: np.ndarray
    targets: np.ndarray
    labels: np.ndarray


def synthetic_table(
    context_rows: int,
    query_rows: int,
    features: int,
    *,
    seed: int = 0,
    noise: float = 0.1,
    missing: float = 0.0,
) -> SyntheticTable:
    """A seeded table with a nonlinear signal in the first three features.

    Targets are the signal plus Gaussian noise. Labels split the noisy signal
    into three classes at its terciles over all rows. A ``missing`` fraction of
    cells, context and queries alike, is set to NaN after the signal is drawn.
    """
    if features < 3:
        raise ValueError("the signal needs at least three features")
    generator = np.random.default_rng(seed)
    table = generator.normal(size=(context_rows + query_rows, features))
    signal = np.sin(table[:, 0]) + table[:, 1] * table[:, 2]
    signal = signal + noise * generator.normal(size=signal.shape)
    edges = np.quantile(signal, [1 / 3, 2 / 3])
    table[generator.random(table.shape) < missing] = np.nan
    return SyntheticTable(
        features=table[:context_rows].astype(np.float32),
        queries=table[context_rows:].astype(np.float32),
        targets=signal[:context_rows].astype(np.float32),
        labels=np.digitize(signal[:context_rows], edges).astype(np.int32),
    )
