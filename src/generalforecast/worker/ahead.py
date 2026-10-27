# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compiling ahead and deferring: programs ready before the steps that need them.

A worker compiles at start what earlier workers recorded for its models,
compiles likely sizes in quiet gaps between steps, and on an accelerator
defers a table whose new program would hold up every other table's steps,
announcing it once the program is ready beside them.
"""

import functools
import hashlib
import logging
import os
import platform
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np

from generalforecast._core import bucket_size
from generalforecast.constants import (
    STARTED_PROGRAMS,
    TABLE_STAGES,
    WARM_RUN_ROWS,
    WARMED_PAGES,
    WARMED_QUERIES,
)
from generalforecast.runtime.budget import Call, _shapes, note_compiled
from generalforecast.runtime.device import cache_root, cpu_flags
from generalforecast.runtime.programs import (
    Kind,
    Recorded,
    abstract,
    counts_up_to,
    ladder,
    most_rows,
    zeros,
)
from generalforecast.worker.calls import _signature, _stacked, _stacked_shape
from generalforecast.worker.records import Fit, QueryBatch, Unit

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


def _manifests(bucket_minimum: int, slot_bytes: int) -> Path:
    """Where workers record the kinds of call their models compile for, beside JAX's compile cache.

    One directory per version of the code workers trace and of JAX, device
    and padding, since programs recorded under another may not compile, or
    compile alike.
    """
    device = jax.devices()[0]
    identity = "|".join(
        [
            _traced(),
            jax.__version__,
            device.platform,
            device.device_kind,
            *([_cpu_features()] if device.platform == "cpu" else []),
            str(bucket_minimum),
            str(slot_bytes),
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return cache_root() / "programs" / digest


def _cpu_features() -> str:
    """The instruction sets XLA may compile this machine's CPU programs with, which a program shipped from another must match.

    The set `portable_cpu_programs` pinned, or else every flag the CPU lists.
    """
    pinned = re.search(r"xla_cpu_max_isa=(\w+)", os.environ.get("XLA_FLAGS", ""))
    if pinned:
        return f"{platform.machine()} {pinned[1]}"
    return " ".join([platform.machine(), *sorted(cpu_flags())])


@functools.cache
def _traced() -> str:
    """A digest of the code workers trace and measure, in place of this package's version.

    A release that leaves this code alone compiles every program as the one
    before did, so it keeps that release's recorded programs and bundles.
    """
    root = Path(__file__).parents[1]
    files: list[Path] = []
    for package in ("models", "nn", "profiling", "runtime", "worker"):
        files += (root / package).rglob("*.py")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(f"{path.relative_to(root).as_posix()}\0".encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class Ahead:
    """How a worker readies programs before its steps call them."""

    def _measured(
        self: "Worker", model: int, key: tuple, measure: Callable[[], Any]
    ) -> Any:
        """What `measure` returns, kept for later processes under the model's spec and settings."""
        spec = self._specs.get(model)
        if spec is None:
            return measure()
        return self.programs.measured(spec, (*key, repr(self._settings)), measure)

    def warm_start(
        self: "Worker",
        progress: Callable[[int, int], None] | None = None,
        models: Sequence[int] | None = None,
    ) -> None:
        """Compiles the kinds of call earlier processes recorded for the loaded models: the most recently used before any step, the rest ahead.

        Each kind compiles for every power of two of items up to the most a
        call of it took, and for each count it took, from shapes alone, so
        the device stays free. Programs JAX's compile cache holds load from
        it; tracing them is what remains, which steps would otherwise do.
        Each model's `STARTED_PROGRAMS` most recently used programs compile
        here, so a start takes about as long however much was recorded; the
        rest compile ahead in quiet gaps between steps, most recently used
        first. A kind that no longer compiles, as after the model's code
        changed, is skipped. Call it after `allocate`: paged decodes read
        the pool.

        Args:
            progress: Called with the programs compiled so far and the
                programs to compile before any step, after each program and
                each measure, so the caller can show how far along it is.
            models: The loaded models to compile for; every one by default.
        """
        started = time.perf_counter()
        chosen = self._specs.keys() if models is None else models
        planned: list[tuple[int, Recorded, list[tuple[Kind, int]]]] = []
        for model in chosen:
            recorded = self.programs.recorded(self._specs[model])
            programs = [
                (kind, count)
                for kind, counts in recorded.by_use()
                for count in self._buckets(kind.program[0]) or counts_up_to(counts)
            ]
            planned.append((model, recorded, programs))
        total = sum(min(len(programs), STARTED_PROGRAMS) for *_, programs in planned)
        done = 0
        for model, recorded, programs in planned:
            if not self.forecaster(model) and recorded.kinds:
                # What a step measures by compiling, once per model and width,
                # before it decides where a context goes.
                self.decode_profile(model)
                for measure in recorded.measures:
                    self._measure(model, measure)
                    if progress is not None:
                        progress(done, total)
            failed: set[Kind] = set()
            for kind, count in programs[:STARTED_PROGRAMS]:
                if kind not in failed and not self._warm_program(model, kind, count):
                    failed.add(kind)
                done += 1
                if progress is not None:
                    progress(done, total)
            for kind, count in programs[STARTED_PROGRAMS:]:
                if kind not in failed:
                    self.programs.later(
                        lambda model=model, kind=kind, count=count: self._warm_program(
                            model, kind, count
                        )
                    )
        self.programs.warm_seconds += time.perf_counter() - started

    def _warm_program(self: "Worker", model: int, kind: Kind, count: int) -> bool:
        """Compiles a recorded kind of a model's call for `count` items, and runs it on zeros where that is quick; whether it compiled.

        A program's first run sets it up on the device, which would otherwise
        slow the first step that calls it. One reading a large table takes as
        long on zeros as the call it would speed up, so only compiles.
        """
        program = (kind.program[0], model, *kind.program[2:])
        parameters = self._models[model].parameters
        try:
            function = cast(jax.stages.Wrapped, self._program(program))
            items = abstract(kind.item, count)
            shared = self._shared(program, kind, items)
            arguments = (parameters, *shared, *items)
            note_compiled(function, arguments, function.lower(*arguments).compile())
            inputs = zeros((shared, items))
            if inputs is not None and (
                kind.program[0] not in TABLE_STAGES
                or most_rows(kind.item) <= WARM_RUN_ROWS
            ):
                jax.block_until_ready(function(parameters, *inputs[0], *inputs[1]))
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "not compiling a recorded %s call: %s", kind.program[0], error
            )
            return False
        self.programs.warmed += 1
        return True

    def _remember(self: "Worker", model: int, measure: tuple) -> None:
        """Records a measure a model took by compiling, for `warm_start` to take again."""
        if model in self._specs:
            self.programs.remember_measure(self._specs[model], measure)

    def _measure(self: "Worker", model: int, measure: tuple) -> None:
        """Takes a recorded measure again: a width's sizes and layout, or an encode's bytes."""
        try:
            match measure:
                case ("width", int(features)):
                    self.profile(model, features)
                    self.layout(model, features)
                case ("encode", int(features), int(rows), bool(fit)):
                    self.encode_bytes(model, features, rows, fit)
        except Exception as error:  # noqa: BLE001
            logger.warning("not measuring a recorded %s: %s", measure[0], error)

    def _shared(self: "Worker", program: tuple, kind: Kind, items: tuple) -> tuple:
        """A recorded kind's shared arguments for `items`: a paged decode reads this pool."""
        if program[0] != "paged":
            return abstract(kind.shared)
        blocks = items[2]
        pool = self._device.data
        if isinstance(pool, np.ndarray):
            # A CPU's decode reads a copy of its slots; see `DevicePool.source`.
            rows = bucket_size(int(np.prod(blocks.shape)), 1)
            source = jax.ShapeDtypeStruct((rows, pool.shape[1]), np.uint8)
        else:
            source = jax.ShapeDtypeStruct(pool.shape, pool.dtype)
        return (source,)

    def _warm(self: "Worker", fit: Fit) -> None:
        """Compiles a newly fit table's page embeds on a thread, before writes need them.

        A write re-embeds the pages it changes, one or a few at a time; on an
        accelerator each new shape compiles for seconds, which the step that
        meets it would otherwise wait for.
        """
        spec = self._tables[fit.table]
        if len(fit.features) == 0:
            return
        features, targets, _ = self._padded(
            fit.features[:1], fit.targets[:1], spec.page_size
        )
        arguments = (spec.state, features, targets)
        # Tables of one model, task and padded shapes share their programs,
        # so only the first of them compiles; many request contexts are.
        marker = ("embeds", spec.model, spec.num_classes, _signature(arguments))
        if self.programs.first(marker):
            self.programs.later(
                lambda: self._compile_embeds(spec.model, spec.num_classes, arguments)
            )

    def _compile_embeds(
        self: "Worker", model: int, num_classes: int, arguments: tuple
    ) -> None:
        """Queues page embeds to compile for each row bucket up to the page size and the first `WARMED_PAGES` page counts, without running one.

        A page with sources embeds only its own rows, padded to their bucket,
        so a step meets the buckets below the page size too. Compiling from
        shapes alone leaves the device to the step thread, so a warm never
        runs beside a step. Each program compiles in a task of its own, so a
        drop of every program ends those not compiled yet.
        """
        key = ("embed", model, num_classes, _signature(arguments))
        estimate = self._estimate("embed", model, num_classes, arguments)
        size = self._chunks.size(key, self._call_bytes, estimate)
        compiled = self._compiled_stage("embed", model, num_classes, len(arguments))
        parameters = self._models[model].parameters
        state, features, targets = arguments
        page = len(features)
        below = [
            rows for rows in ladder(self._minimum, page.bit_length()) if rows < page
        ]
        counts = sorted(
            {bucket_size(min(items, size), 1) for items in ladder(1, WARMED_PAGES)}
        )
        for rows in [*below, page]:
            sliced = (state, features[:rows], targets[:rows])
            for count in counts:
                stacked = jax.tree.map(
                    lambda value, count=count: _stacked_shape(value, count), sliced
                )
                self.programs.compile_later(compiled, parameters, *stacked)

    def _warmed_queries(self: "Worker") -> list[int]:
        """Padded query counts compiled ahead: `WARMED_QUERIES` buckets above the smallest."""
        return ladder(self._size(self._minimum + 1), WARMED_QUERIES)

    def _warm_query_embeds(self: "Worker", batch: QueryBatch) -> None:
        """Compiles a query embed for each of `_warmed_queries` ahead, once per model, task and shapes."""
        spec = self._tables[batch.table]
        if spec.state is None:
            return
        features = np.asarray(batch.features)
        marker = (
            "queries",
            spec.model,
            spec.num_classes,
            _signature((spec.state, features[:0])),
        )
        if not self.programs.first(marker):
            return
        compiled = self._compiled_stage("embed", spec.model, spec.num_classes, 2)
        for more in self._warmed_queries():
            shape = jax.ShapeDtypeStruct((1, more, *features.shape[1:]), features.dtype)
            self.programs.compile_later(
                compiled,
                self._models[spec.model].parameters,
                _stacked_shape(spec.state, 1, stacked=True),
                shape,
            )

    def _defers(self: "Worker", unit: Unit, arguments: tuple) -> bool:
        """Whether a fused unit waits for its program, readied beside steps.

        On an accelerator a new shape's program compiles for tens of
        seconds, and its first unit's measures take as long again, which
        every other table's steps would wait for. Such a unit's table is
        deferred instead, under its program's key, which `announce` names
        once the program is ready, for the server to plan it again.
        """
        if not self._on_side or self.announce is None:
            return False
        table = self._tables[unit.table]
        key = ("fit_decode", table.model, table.num_classes, _signature(arguments))
        if self._chunks.ran(key):
            return False
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        name = f"{self.programs.drops}:{digest}"
        with self._readying:
            if name in self._unpreparable:
                return False
            start = name not in self._preparing
            self._preparing.add(name)
        self.deferred[unit.table] = name
        if start:
            self.programs.soon(lambda: self._prepare(unit, key, arguments, name))
        return True

    def _prepare(
        self: "Worker", unit: Unit, key: tuple, arguments: tuple, name: str
    ) -> None:
        """Measures a deferred unit and compiles its program for one, from shapes, then announces its key."""
        stage, model, num_classes, _ = key
        program = (stage, model, num_classes, len(arguments))
        try:
            self._fits_device(unit)
            compiled = self._compiled_stage(*program)
            shared = (self._models[model].parameters,)
            items = _shapes(_stacked([arguments]))
            jitted = cast(jax.stages.Wrapped, compiled)
            note_compiled(
                jitted, (*shared, *items), jitted.lower(*shared, *items).compile()
            )
            self._chunks.mark(key, 1)
            self._seen(Call(compiled, shared, items, 1, program))
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "could not ready a program beside steps (%s); the step that "
                "next needs it compiles it",
                error,
            )
            with self._readying:
                self._unpreparable.add(name)
        finally:
            with self._readying:
                self._preparing.discard(name)
            if self.announce is not None:
                self.announce([name])

    def _warm_decodes(
        self: "Worker",
        compiled: Callable[..., Any],
        shared: tuple,
        arguments: tuple,
        queries: int,
    ) -> None:
        """Compiles a paged decode ahead for one context and each larger of `_warmed_queries`."""

        def one(value: Any, queries: int | None = None) -> jax.ShapeDtypeStruct:
            shape = list(np.shape(value))
            shape[0] = 1
            if queries is not None:
                shape[1] = queries
            return jax.ShapeDtypeStruct(
                tuple(shape),
                value.dtype,
                weak_type=bool(getattr(value, "weak_type", False)),
            )

        parameters, source = shared
        states, index, blocks, positions, counts, rows, *reads = arguments
        for more in self._warmed_queries():
            if more > queries:
                self.programs.compile_later(
                    compiled,
                    parameters,
                    source,
                    jax.tree.map(one, states),
                    one(index),
                    one(blocks),
                    one(positions),
                    one(counts),
                    one(rows, more),
                    *[one(read, more) for read in reads],
                )
