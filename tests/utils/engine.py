"""An engine double for surface tests: it records what surfaces run, and models nothing."""

import json
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from generalforecast import _core
from generalforecast._core import ForecastRequest, TabularRequest
from generalforecast.engine import Bound, Forecasts, Labeled
from generalforecast.models.catalog import ModelRef


def tabular_request(
    model: str, context: Any, labels: Any, queries: Any
) -> TabularRequest:
    """A request over numeric rows, as `gf_core::request` validates it locally.

    The task is the model's; labels are class labels or regression targets.
    """
    features = [f"f{index}" for index in range(np.shape(context)[1])]
    regression = model.endswith("regressor")
    payload = {
        "model": model,
        "task": "regression" if regression else "classification",
        "target": "y",
        "features": features,
        "context": [
            {
                "key": f"c{index}",
                "features": dict(zip(features, map(float, row), strict=True)),
                "label": label.item() if isinstance(label, np.generic) else label,
            }
            for index, (row, label) in enumerate(zip(context, labels, strict=True))
        ],
        "queries": [
            {
                "key": f"q{index}",
                "features": dict(zip(features, map(float, row), strict=True)),
            }
            for index, row in enumerate(queries)
        ],
        "options": {},
    }
    request = _core.parse_inference(json.dumps(payload).encode(), "local")
    assert isinstance(request, TabularRequest)
    return request


class FakeEngine:
    """Answers as a fixed model, through `generalforecast.engine.local` once installed.

    Classifiers answer 0.25 and 0.75, regressors 0 to 998 plus the mean of
    their context's targets, and forecasters count up over every query,
    period and level.
    """

    def __init__(
        self, levels: tuple[float, ...] = (0.1, 0.5, 0.9), max_rows: int | None = None
    ) -> None:
        self.properties = SimpleNamespace(
            quantiles=levels, max_rows=max_rows, num_outputs=999, patch=16
        )
        self.loads: list[tuple[str, str]] = []
        self.contexts: list[np.ndarray] = []
        self.forecasts: list[ForecastRequest] = []
        self.delay = 0.0
        self.surface = "local"

    def install(self, monkeypatch: Any) -> "FakeEngine":
        """Makes this the process engine every surface reaches."""
        monkeypatch.setattr("generalforecast.engine.local", self.local)
        return self

    def local(
        self, surface: str = "local", allow_downloads: bool = False
    ) -> "FakeEngine":
        self.surface = surface
        return self

    def load(self, ref: ModelRef) -> str:
        self.loads.append((ref.name, self.surface))
        return str(ref)

    def tabular(self, ref: ModelRef, request: TabularRequest) -> np.ndarray:
        self.contexts.append(request.context.copy())
        return self._score(request, request.queries)

    def forecast(self, ref: ModelRef, request: ForecastRequest) -> Forecasts:
        self.loads.append((ref.name, self.surface))
        limit = self.properties.max_rows
        if limit is not None and any(
            query.group.values.shape[0] > limit for query in request.queries
        ):
            raise _core.RequestError(
                "CapacityExceeded", f"{ref.name} reads at most {limit} rows", 422
            )
        self.forecasts.append(request)
        return Forecasts(
            self._count(len(request.queries), request.horizon),
            tuple(self.properties.quantiles),
        )

    def _count(self, queries: int, horizon: int) -> np.ndarray:
        shape = (queries, horizon, len(self.properties.quantiles))
        return np.arange(np.prod(shape), dtype=float).reshape(shape)

    def bind(self, ref: ModelRef, context: Labeled) -> Bound:
        self.contexts.append(context.context.copy())
        return Bound(ref, context, "fake")

    def predict(self, bound: Bound, queries: np.ndarray) -> np.ndarray:
        return self._score(bound.context, queries)

    def _score(
        self, context: Labeled | TabularRequest, queries: np.ndarray
    ) -> np.ndarray:
        time.sleep(self.delay)
        if context.classes:
            return np.tile([0.25, 0.75], (len(queries), 1))
        mean = float(np.mean(context.targets))
        return np.tile(np.arange(999, dtype=float) + mean, (len(queries), 1))
