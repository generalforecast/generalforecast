# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a worker keeps per model and table, and the forms a step's parts come in."""

import threading
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np

from generalforecast import _core, protocol
from generalforecast.models.contract import TabularFunctions
from generalforecast.models.series import SeriesFunctions
from generalforecast.runtime.paging import StateLayout

# A step comes from the bindings in-process, or decoded from a server's frame.
Step = _core.Step | protocol.Step
Fit = _core.Fit | protocol.Fit
Page = _core.Page | protocol.Page
Unit = _core.Unit | protocol.Unit
QueryBatch = _core.QueryBatch | protocol.QueryBatch
Movement = _core.Movement | protocol.Movement
SeriesUnit = _core.SeriesUnit | protocol.SeriesUnit
Forecast = _core.Forecast | protocol.Forecast


@dataclass
class _Model[Functions: (TabularFunctions, SeriesFunctions)]:
    functions: Functions
    parameters: Any


_TabularModel = _Model[TabularFunctions]
_SeriesModel = _Model[SeriesFunctions]


@dataclass
class _Table:
    model: int
    num_classes: int
    features: int
    # Which columns are categorical, or None if none are.
    categorical: np.ndarray | None = None
    # Rows every page embed pads to at least, so pages that split or shrink
    # keep one compiled embed.
    page_size: int = 1
    state: Any = None
    # How each unit's own state packs into slots, for a table that summarizes
    # each context; None for one that summarizes the table.
    own: StateLayout | None = None
    # Whether the table holds series, forecast by a forecaster.
    series: bool = False
    # Whether a fit already logged that the state keeps fewer categories than
    # a column holds.
    crowded: bool = False


class Memo[Key: Hashable, Value]:
    """Values a worker measures once per key, however many threads ask.

    The step thread and the threads that ready programs beside steps measure
    the same models; a key being measured is measured once, under a lock of
    its own, so a thread asking for it waits for that measure, and one asking
    for another key does not. A measure that raises keeps nothing, so the
    next ask measures again.
    """

    def __init__(self) -> None:
        """Starts with nothing measured."""
        self._values: dict[Key, Value] = {}
        self._guard = threading.Lock()
        self._measuring: dict[Key, threading.Lock] = {}

    def get(self, key: Key, measure: Callable[[], Value]) -> Value:
        """The value of `key`, measured with `measure` if no thread has yet."""
        if key in self._values:
            return self._values[key]
        with self._guard:
            lock = self._measuring.setdefault(key, threading.Lock())
        with lock:
            if key not in self._values:
                self._values[key] = measure()
        with self._guard:
            self._measuring.pop(key, None)
        return self._values[key]

    def __getitem__(self, key: Key) -> Value:
        """The value of a key measured already."""
        return self._values[key]
