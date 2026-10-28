# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`Predictor.eval`'s scores: classification, regression and forecast metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow.compute as _compute

from generalforecast.errors import (
    InvalidInput,
)
from generalforecast.native import models as registry

pc: Any = _compute


@dataclass(frozen=True)
class Evaluation:
    """Scores of one or more models on the same held-out rows.

    Attributes:
        metrics: `{model_id: {metric: value}}`.
        rows: The held-out rows: the id (if any), the target and one
            prediction column per model.
    """

    metrics: dict[str, dict[str, float]]
    rows: Any

    def __str__(self) -> str:
        """The metric table as `gf eval` prints it."""
        models = list(self.metrics)
        names = list(
            dict.fromkeys(key for item in self.metrics.values() for key in item)
        )
        width = max((len(name) for name in names), default=0)
        heads = [registry.short(model) for model in models]
        many = len(models) > 1
        widths = [max(len(head), 5) if many else 5 for head in heads]
        lines = []
        if many:
            lines.append(
                " " * 13
                + " " * width
                + "   "
                + "   ".join(
                    head.rjust(w) for head, w in zip(heads, widths, strict=True)
                )
            )
        for name in names:
            cells = []
            for model, w in zip(models, widths, strict=True):
                value = self.metrics[model].get(name)
                cells.append(("" if value is None else f"{value:.3f}").rjust(w))
            lines.append(" " * 13 + name.ljust(width) + "   " + "   ".join(cells))
        return "\n".join(lines)


def classification_metrics(
    truth: Sequence[Any],
    labels: Sequence[Any],
    probabilities: Mapping[Any, Sequence[float]],
) -> dict[str, float]:
    """`accuracy`, `log loss`, and `auroc` (binary) or `macro f1` (multiclass)."""
    classes = list(probabilities)
    y = list(truth)
    n = len(y)
    if n == 0:
        raise InvalidInput("no rows to evaluate")
    accuracy = sum(1 for a, b in zip(y, labels, strict=True) if a == b) / n
    eps = 1e-15
    losses = []
    for index, value in enumerate(y):
        p = probabilities[value][index] if value in probabilities else 0.0
        losses.append(-math.log(min(max(p, eps), 1 - eps)))
    out = {"accuracy": accuracy, "log loss": sum(losses) / n}
    if len(classes) == 2:
        positive = classes[1]
        scores = list(probabilities[positive])
        out["auroc"] = _auroc([1 if value == positive else 0 for value in y], scores)
    else:
        f1s = []
        for label in classes:
            tp = sum(
                1 for a, b in zip(y, labels, strict=True) if a == label and b == label
            )
            fp = sum(
                1 for a, b in zip(y, labels, strict=True) if a != label and b == label
            )
            fn = sum(
                1 for a, b in zip(y, labels, strict=True) if a == label and b != label
            )
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        out["macro f1"] = sum(f1s) / len(f1s)
    return out


def _auroc(truth: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(truth)
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    positive_ranks = sum(
        rank for rank, label in zip(ranks, truth, strict=True) if label
    )
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def regression_metrics(
    truth: Sequence[float],
    values: Sequence[float],
    quantiles: Mapping[float, Sequence[float]] | None = None,
) -> dict[str, float]:
    """`mae`, `rmse` and `pinball q` per quantile."""
    y = np.asarray(truth, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(y) == 0:
        raise InvalidInput("no rows to evaluate")
    out = {
        "mae": float(np.mean(np.abs(y - v))),
        "rmse": float(math.sqrt(np.mean((y - v) ** 2))),
    }
    for level, predicted in (quantiles or {}).items():
        out[f"pinball {level:g}"] = _pinball(
            y, np.asarray(predicted, dtype=float), level
        )
    return out


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    diff = y - q
    return float(np.mean(np.where(diff >= 0, level * diff, (level - 1) * diff)))


def _when(moment: Any) -> Any:
    """A time as it sorts: an instant for dates and datetimes, else itself."""
    from generalforecast.predictors.frequency import instant

    try:
        return instant(moment)
    except TypeError:
        return moment


def forecast_metrics(
    truth: Sequence[float],
    quantiles: Mapping[float, Sequence[float]],
    *,
    history: Mapping[Any, Sequence[float]],
    series: Sequence[Any],
) -> dict[str, float]:
    """`mase` (scaled by each series' naive in-sample error) and `wql`."""
    y = np.asarray(truth, dtype=float)
    levels = sorted(quantiles)
    median = min(levels, key=lambda level: abs(level - 0.5))
    point = np.asarray(quantiles[median], dtype=float)
    scales = {}
    for key, values in history.items():
        past = np.asarray(values, dtype=float)
        scale = np.mean(np.abs(np.diff(past))) if len(past) > 1 else float("nan")
        scales[key] = scale if scale and not math.isnan(scale) else float("nan")
    errors = []
    for value, predicted, key in zip(y, point, series, strict=True):
        scale = scales.get(key, float("nan"))
        if scale and not math.isnan(scale):
            errors.append(abs(value - predicted) / scale)
    mase = float(np.mean(errors)) if errors else float("nan")
    total = float(np.sum(np.abs(y)))
    losses = sum(
        2 * _pinball(y, np.asarray(quantiles[level], dtype=float), level) * len(y)
        for level in levels
    ) / len(levels)
    wql = float(losses / total) if total else float("nan")
    return {"mase": mase, "wql": wql}
