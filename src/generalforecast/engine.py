# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The engine port: every surface's model calls, over gf-server's shards.

The Models API, `gf serve`, predictor predictions and the native estimators
hand requests `gf_core::request` validated to an `Engine` rather than calling
a model: a `TabularRequest`'s labeled context rows and query rows, or a
`ForecastRequest`'s queries, each with its aligned group. `LocalEngine`
answers them in this process through `generalforecast._core.Engine`, which is
gf-server's own engine: one shard per device slot, each a node and a worker
subprocess, sized from the hardware as `gf serve` sizes them, so concurrent
callers share steps and the workers' cores whatever contexts they bring.

- A request's context becomes an ephemeral table named `ctx-<fingerprint>`
  on the shard its fingerprint routes to, pinned while a request reads it
  and evicted by gf-server's table, byte and idle budget; a repeated context
  reuses its table, fit and encoded contexts. A forecast reads one per
  aligned group, by the group's fingerprint.
- `bind` writes a context that many requests read, such as a snapshot
  predictor's, once as a durable table, which lives as long as its `Bound`.

Models load in the workers, never here: a reference passes its surface's
gates and its pinned files are verified in this process, which imports no
model code, then every worker loads it by its spec.

The engine returns the model's own outputs, checked: class probabilities in
class code order, every quantile the regressor predicts, or forecasts over the
forecaster's levels. Callers shape them with `generalforecast.models.outputs`.
"""

from __future__ import annotations

import atexit
import json
import shutil
import sys
import tempfile
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

import numpy as np

from generalforecast.models.catalog import ModelRef, Surface, admit, parse
from generalforecast.system import usable_cpus

if TYPE_CHECKING:
    from generalforecast._core import BoundTable, ForecastRequest, TabularRequest
    from generalforecast._core import Engine as Shards


class ModelOutputError(ValueError):
    """A model returned outputs of another shape, or values no model of its task returns."""


@dataclass(frozen=True, eq=False)
class Labeled:
    """A tabular context to bind: labeled rows, without queries.

    Attributes:
        task: `classification` or `regression`.
        context: The rows' features, float32 `(rows, features)`.
        targets: Each row's class code into `classes`, int32, or its
            regression target, float32.
        classes: The class labels in code order; empty for a regressor.
    """

    task: Literal["classification", "regression"]
    context: np.ndarray
    targets: np.ndarray
    classes: tuple[Any, ...]


@dataclass(frozen=True, eq=False)
class Bound:
    """A context bound once as a durable table, which lives as long as this handle.

    Attributes:
        ref: The model the table runs.
        context: The context bound.
        table: The engine's handle on the table.
    """

    ref: ModelRef
    context: Labeled
    table: Any
    _dropper: Any = field(default=None, repr=False)


class Forecasts(NamedTuple):
    """Each query's forecast, `(queries, horizon, levels)`, and the model's levels."""

    values: np.ndarray
    levels: tuple[float, ...]


class Engine(Protocol):
    """Runs validated requests; every surface reaches models through one."""

    def tabular(self, ref: ModelRef, request: TabularRequest) -> np.ndarray:
        """Scores a request's queries against its context: `(queries, outputs)`."""
        ...

    def forecast(self, ref: ModelRef, request: ForecastRequest) -> Forecasts:
        """Forecasts each query's series over the model's own quantile levels."""
        ...

    def bind(self, ref: ModelRef, context: Labeled) -> Bound:
        """Binds a context many requests read as a durable table."""
        ...

    def predict(self, bound: Bound, queries: np.ndarray) -> np.ndarray:
        """Scores query features against a bound context: `(queries, outputs)`."""
        ...


_process: LocalEngine | None = None
_process_lock = threading.Lock()


def local(surface: Surface = "local", allow_downloads: bool = True) -> LocalEngine:
    """This process's engine, with a caller's gates.

    Every surface in a process shares it, so each model loads once in each
    worker and every caller joins the same steps.
    """
    global _process
    with _process_lock:
        if _process is None:
            _process = LocalEngine(cpu_threads=_cores())
    return _process.gated(surface, allow_downloads)


def _cores() -> int:
    """The cores this process may use, all for its one CPU worker.

    One caller at a time is the in-process case: on 8 cores one worker of
    eight started a small model in about half the time two of four did, and
    fit new contexts 1.1 to 3.3 times faster at 100 to 10,000 rows.
    """
    return usable_cpus()


def configuration(
    data: str,
    cpu_threads: int | None = None,
    tiers: Mapping[str, int | str] | None = None,
) -> str:
    """gf-server's configuration for an engine's shards, without Flight or HTTP.

    Workers start as `gf serve` starts them: one per visible GPU, else one
    per `cpu_threads` of the cores this process may use, pinned to them, as
    many as gf-server gives each if None.
    `tiers` sets gf-server's memory settings, such as `device_bytes` or
    `eviction`, in place of the sizes measured from the hardware.
    """
    command = [sys.executable, "-m", "generalforecast.worker_process"]
    return "\n".join(
        [
            'listen = "127.0.0.1:0"',
            f"data = {json.dumps(data)}",
            "tokens = []",
            *(f"{key} = {json.dumps(value)}" for key, value in (tiers or {}).items()),
            "",
            "[[workers]]",
            f"command = {json.dumps(command)}",
            "per_gpu = true",
            *([] if cpu_threads is None else [f"cpu_threads = {cpu_threads}"]),
            "",
            "[workers.models]",
            "",
        ]
    )


def _unpinned(ref: ModelRef) -> str:
    """The name workers serve a model by: its reference without the digest pin.

    `parse` admits a pin only when it is the catalog's digest, so a pinned and
    an unpinned reference name the same weights and share a model and its tables.
    """
    return str(replace(ref, digest=None))


class _Shared:
    """What engines with different gates share: the shards and the models loaded."""

    def __init__(
        self,
        models: Mapping[str, str],
        cpu_threads: int | None,
        tiers: Mapping[str, int | str] | None = None,
    ) -> None:
        from generalforecast import _core

        self.data = tempfile.mkdtemp(prefix="gf-engine-")
        self.shards: Shards = _core.Engine(configuration(self.data, cpu_threads, tiers))
        self.specs = {_unpinned(parse(name)): spec for name, spec in models.items()}
        self.loaded: set[str] = set()
        self.loading = threading.Lock()
        self._closer = weakref.finalize(self, _close, self.shards, self.data)

    def close(self) -> None:
        self._closer()


def _close(shards: Shards, data: str) -> None:
    """Stops the shards, which stops their workers, and removes their files."""
    shards.close()
    shutil.rmtree(data, ignore_errors=True)


class LocalEngine:
    """The engine in this process: gf-server's shards on an embedded runtime.

    Engines made by `gated` share this one's shards and loaded models, so
    callers with different gates still share steps.
    """

    def __init__(
        self,
        *,
        surface: Surface = "local",
        allow_downloads: bool = False,
        models: Mapping[str, str] | None = None,
        cpu_threads: int | None = None,
        tiers: Mapping[str, int | str] | None = None,
    ) -> None:
        """Starts the shards, admitting models as `surface` may run them.

        Args:
            surface: The surface whose gates a model must pass.
            allow_downloads: Whether a model's pinned files may download.
            models: The worker spec to load a reference's model from, by
                the reference's canonical spelling, in place of the
                reference itself: a test's small models, for instance.
            cpu_threads: Without GPUs, the cores each worker has to itself;
                as many as gf-server gives each if None.
            tiers: gf-server's memory settings; see `configuration`.
        """
        self._shared = _Shared(models or {}, cpu_threads, tiers)
        self.surface: Surface = surface
        self.allow_downloads = allow_downloads

    def gated(self, surface: Surface, allow_downloads: bool) -> LocalEngine:
        """This engine, admitting models with other gates."""
        engine = object.__new__(LocalEngine)
        engine._shared = self._shared
        engine.surface, engine.allow_downloads = surface, allow_downloads
        return engine

    @property
    def shards(self) -> int:
        """How many shards, and so worker processes, run the requests."""
        return self._shared.shards.shards

    def close(self) -> None:
        """Stops the shards and their workers; later calls raise."""
        self._shared.close()

    def serve(self) -> tuple[str, str]:
        """gf-server's Flight service on these shards, for this process alone.

        Returns:
            Its location and the random token it accepts; every call returns
            the same pair.
        """
        return self._shared.shards.serve()

    def load(self, ref: ModelRef) -> str:
        """Admits a reference for this engine's surface and has every worker load it.

        Returns:
            The name the workers serve it by.

        Raises:
            ModelRefError: If the surface may not run the model.
            ArtifactError: If its files are missing or do not match their pins.
        """
        admit(ref, self.surface)
        name = _unpinned(ref)
        spec = self._shared.specs.get(name)
        if spec is None:
            from generalforecast.models.artifacts import checkpoint

            checkpoint(
                ref.entry, surface=self.surface, allow_downloads=self.allow_downloads
            )
        with self._shared.loading:
            if name not in self._shared.loaded:
                self._shared.shards.add_model(name, spec or name)
                self._shared.loaded.add(name)
        return name

    def tabular(self, ref: ModelRef, request: TabularRequest) -> np.ndarray:  # noqa: D102
        values, _ = self._shared.shards.tabular(self.load(ref), request)
        return _checked(values, len(request.queries), len(request.classes))

    def forecast(self, ref: ModelRef, request: ForecastRequest) -> Forecasts:  # noqa: D102
        values, levels = self._shared.shards.forecast(self.load(ref), request)
        result = np.asarray(values)
        if (
            levels is None
            or result.shape != (len(request.queries), request.horizon, len(levels))
            or not np.isfinite(result).all()
        ):
            raise ModelOutputError("the forecaster returned invalid forecasts")
        result.setflags(write=False)
        return Forecasts(result, tuple(levels))

    def bind(self, ref: ModelRef, context: Labeled) -> Bound:  # noqa: D102
        shards = self._shared.shards
        targets = (
            np.ascontiguousarray(context.targets, np.int32)
            if context.classes
            else np.ascontiguousarray(context.targets, np.float32)
        )
        table: BoundTable = shards.bind(
            self.load(ref),
            np.ascontiguousarray(context.context, np.float32),
            targets,
            len(context.classes),
        )
        bound = Bound(ref, context, table)
        dropper = weakref.finalize(bound, _unbind, shards, table)
        # Shards stopped at exit take their tables with them.
        dropper.atexit = False
        object.__setattr__(bound, "_dropper", dropper)
        return bound

    def predict(self, bound: Bound, queries: np.ndarray) -> np.ndarray:  # noqa: D102
        values, _ = self._shared.shards.predict(
            bound.table, np.ascontiguousarray(queries, np.float32)
        )
        return _checked(values, len(queries), len(bound.context.classes))


def _unbind(shards: Shards, table: BoundTable) -> None:
    """Drops a bound table, unless its shards have stopped."""
    try:
        shards.unbind(table)
    except RuntimeError:
        pass


def _checked(values: Any, queries: int, classes: int) -> np.ndarray:
    """A tabular model's outputs, refused unless its task could return them."""
    result = np.asarray(values)
    valid = (
        result.ndim == 2
        and len(result) == queries
        and (not classes or result.shape[1] == classes)
        and bool(np.isfinite(result).all())
    )
    if valid and classes:
        valid = bool(
            (result >= 0).all() and np.allclose(result.sum(axis=1), 1, atol=1e-5)
        )
    elif valid:
        valid = bool((np.diff(result, axis=1) >= 0).all())
    if not valid:
        raise ModelOutputError("the model returned invalid outputs")
    return result


@atexit.register
def _stop() -> None:
    """Stops the process engine's workers before the interpreter goes."""
    if _process is not None:
        _process.close()


# The public engine surface, `gf.engine.Table` and its settings, lives in
# `generalforecast.native.tables` and loads on first use so importing this
# module stays cheap.
_PUBLIC = {
    name: "generalforecast.native.tables"
    for name in (
        "Grouping",
        "Retrieved",
        "Schema",
        "SeriesSchema",
        "SeriesTable",
        "Table",
        "TableOptions",
        "TimeOptions",
        "default_selection",
    )
}


def __getattr__(name: str) -> Any:
    """Imports a public engine export on first use."""
    module = _PUBLIC.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
