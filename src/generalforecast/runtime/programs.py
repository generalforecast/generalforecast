import fcntl
import hashlib
import logging
import os
import pickle
import threading
import time
from collections.abc import Callable, Hashable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np

from generalforecast.runtime.device import DeviceSettings, with_precision

# Table shapes, a model, task and width each, a worker keeps compiled programs
# for while memory is short. JAX keeps every program it compiles, about 200 MiB
# of host memory a shape on a GPU, so a worker past this many, short of memory
# by its budget, drops them all; those still used compile again, which with
# many widths costs far more than the memory, so plenty keeps them.
COMPILED_SHAPES = 32

# Page counts a table's embeds compile for once it is fit, this many buckets
# from one page: a write changes one page or a few.
WARMED_PAGES = 4
# Query counts a table's query embeds and decodes compile for once it has
# decoded, this many buckets above the smallest: steps merging many callers'
# requests reach them.
WARMED_QUERIES = 5
# Seconds a worker goes without a step before compiling ahead. A caller sending
# steps back to back would otherwise meet a compile in each of its next steps.
WARM_QUIET = 0.5
# Kinds of call a model's manifest keeps, the latest used last: each table
# shape compiles about a kind per stage, so this keeps `COMPILED_SHAPES`'s.
RECORDED_KINDS = 8 * COMPILED_SHAPES

logger = logging.getLogger(__name__)


def ladder(first: int, count: int) -> list[int]:
    """`count` padded sizes from `first`, a padded size, each double the last, as buckets are."""
    return [first << step for step in range(count)]


def counts_up_to(counts: Iterable[int]) -> list[int]:
    """Item counts a kind of call is compiled for at start: each power of two up to the most it took, and those it took."""
    taken = set(counts)
    if not taken:
        return []
    return sorted(taken | {count for count in ladder(1, max(taken).bit_length())})


Signature = tuple[Any, tuple[tuple[tuple[int, ...], str, bool], ...]]


def signature(tree: Any, stacked: bool = False) -> Signature:
    """A tree's structure and each leaf's shape, dtype and weak type; with `stacked`, without the leaves' first axis."""
    leaves, structure = jax.tree.flatten(tree)
    return structure, tuple(
        (
            tuple(np.shape(leaf))[1 if stacked else 0 :],
            np.dtype(getattr(leaf, "dtype", type(leaf))).str,
            bool(getattr(leaf, "weak_type", False)),
        )
        for leaf in leaves
    )


def abstract(signature: Signature, count: int | None = None) -> Any:
    """The tree a signature describes, of shapes and dtypes, each leaf stacked `count` times if given."""
    structure, leaves = signature
    lead = () if count is None else (count,)
    return jax.tree.unflatten(
        structure,
        [
            jax.ShapeDtypeStruct((*lead, *shape), np.dtype(dtype), weak_type=weak)
            for shape, dtype, weak in leaves
        ],
    )


@dataclass(frozen=True)
class Kind:
    """A kind of compiled call, as a later process compiles it again.

    Attributes:
        program: The `Programs` key of the function called, with no model:
            its slot holds None, as model ids differ between processes.
        shared: The arguments every item shares beside the model's parameters.
        item: One item's arguments.
    """

    program: tuple
    shared: Signature
    item: Signature


@dataclass
class Recorded:
    """What a model's manifest holds: each kind of call, oldest first, with the item counts it took, and what was measured by compiling.

    A measure is a tuple its worker reads, such as a table width whose
    sizes were measured; the latest `RECORDED_KINDS` are kept.
    """

    kinds: dict[Kind, set[int]]
    measures: dict[tuple, None]


class Manifest:
    """The kinds of call one model compiled for, the item counts of each and what was measured, on disk for later processes.

    A file of its own per model, read and rewritten whole under a lock, so
    workers sharing a machine add to it rather than overwrite each other.
    Only the latest `RECORDED_KINDS` kinds are kept. A file that cannot be
    read, such as one an older version wrote, reads as empty.
    """

    def __init__(self, path: Path) -> None:
        """A manifest kept at `path`."""
        self.path = path

    def read(self) -> Recorded:
        """What is recorded."""
        try:
            with open(self.path, "rb") as file:
                recorded = pickle.load(file)
        except FileNotFoundError:
            return Recorded({}, {})
        except Exception:  # noqa: BLE001
            logger.warning("ignoring an unreadable program manifest %s", self.path)
            return Recorded({}, {})
        return recorded if isinstance(recorded, Recorded) else Recorded({}, {})

    def add(self, kind: Kind, count: int) -> None:
        """Records a kind's call at `count` items as the latest used."""

        def added(recorded: Recorded) -> None:
            counts = recorded.kinds.pop(kind, set())
            counts.add(count)
            recorded.kinds[kind] = counts
            while len(recorded.kinds) > RECORDED_KINDS:
                del recorded.kinds[next(iter(recorded.kinds))]

        self._update(added)

    def add_measure(self, measure: tuple) -> None:
        """Records a measure as the latest taken."""

        def added(recorded: Recorded) -> None:
            recorded.measures.pop(measure, None)
            recorded.measures[measure] = None
            while len(recorded.measures) > RECORDED_KINDS:
                del recorded.measures[next(iter(recorded.measures))]

        self._update(added)

    def _update(self, change: Callable[[Recorded], None]) -> None:
        """Reads, changes and rewrites the file, under its lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            recorded = self.read()
            change(recorded)
            part = self.path.with_suffix(f".{os.getpid()}.part")
            with open(part, "wb") as file:
                pickle.dump(recorded, file)
            os.replace(part, self.path)


class Programs:
    """The programs a worker compiles, until memory runs short, and those it compiles ahead.

    Each function compiles once per key, and JAX keeps a program of it for
    each shape it is called with. Compiles ahead run one at a time on a
    thread of their own, each once no step has run for `quiet` seconds, since
    a compile competes for the host with a step's own, and many new tables at
    once would otherwise wait on compiles for sizes not yet asked.
    """

    def __init__(
        self,
        settings: DeviceSettings | None,
        short_of_memory: Callable[[], bool],
        helpers: Iterable[Callable[..., Any]] = (),
        clock: Callable[[], float] = time.monotonic,
        ahead: bool = True,
        manifests: Path | None = None,
    ) -> None:
        """Sets up with nothing compiled.

        Args:
            settings: Device settings for matrix multiply precision, if any.
            short_of_memory: Whether memory is short, asked past `limit` shapes.
            helpers: Other compiled functions whose programs drop with these.
            clock: Seconds, as `time.monotonic` counts them.
            ahead: Whether to compile ahead at all; without it, each program
                compiles in the step that first calls it.
            manifests: A directory for each model's `Manifest`, which the
                kinds of call it compiles for are recorded in, if any.
        """
        self._settings = settings
        self._short_of_memory = short_of_memory
        self._helpers = tuple(helpers)
        self._clock = clock
        self._ahead = ahead
        # Table shapes kept compiled while memory is short, and seconds
        # without a step before a compile ahead.
        self.limit = COMPILED_SHAPES
        self.quiet = WARM_QUIET
        # Times every program was dropped.
        self.drops = 0
        self._compiled: dict[Hashable, Callable[..., Any]] = {}
        # The table shapes compiled for, and what was compiled ahead, since
        # programs were last dropped.
        self._shapes: set[Hashable] = set()
        self._done: set[Hashable] = set()
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="gf-warm")
        self._pending: list[Future] = []
        self._lock = threading.Lock()
        # Whether a step runs, and when the last ended; compiles ahead wait on them.
        self._steps = threading.Condition()
        self._running = False
        self._idle_since = clock()
        self._manifests = manifests
        # Kinds and counts recorded by this process, which need no rewrite.
        self._recorded: set[tuple[str, Hashable, int]] = set()
        # Programs compiled at start, and their seconds.
        self.warmed = 0
        self.warm_seconds = 0.0

    def compiled(
        self, key: Hashable, build: Callable[[], Callable[..., Any]]
    ) -> Callable[..., Any]:
        """The function `build` makes, compiled at the settings' precision, once per key.

        Args:
            key: What the function is, such as a stage, model and task.
            build: Makes the function to compile.

        Returns:
            The compiled function.
        """
        with self._lock:
            compiled = self._compiled.get(key)
        if compiled is not None:
            return compiled
        function = build()
        if self._settings is not None:
            function = with_precision(function, self._settings.matmul_precision)
        compiled = jax.jit(function)
        with self._lock:
            return self._compiled.setdefault(key, compiled)

    def manifest(self, model: str) -> Manifest | None:
        """Where a model's kinds of call are recorded, under a name of its spec's digest; None without manifests."""
        if self._manifests is None:
            return None
        digest = hashlib.sha256(model.encode()).hexdigest()[:32]
        return Manifest(self._manifests / f"{digest}.pickle")

    def remember(self, model: str, kind: Kind, count: int) -> None:
        """Records that a model's kind of call compiled for `count` items, for later processes to compile at start."""
        manifest = self.manifest(model)
        if manifest is None or (model, kind, count) in self._recorded:
            return
        self._recorded.add((model, kind, count))
        try:
            manifest.add(kind, count)
        except OSError as error:
            logger.warning("cannot record a compiled program: %s", error)

    def remember_measure(self, model: str, measure: tuple) -> None:
        """Records that a model measured something by compiling, which later processes measure at start."""
        manifest = self.manifest(model)
        if manifest is None or (model, measure, 0) in self._recorded:
            return
        self._recorded.add((model, measure, 0))
        try:
            manifest.add_measure(measure)
        except OSError as error:
            logger.warning("cannot record a measure: %s", error)

    def recorded(self, model: str) -> Recorded:
        """What a model compiled for and served in earlier processes."""
        manifest = self.manifest(model)
        return Recorded({}, {}) if manifest is None else manifest.read()

    def add_shape(self, shape: Hashable) -> None:
        """Counts a table shape, dropping every program first if memory is short past `limit`."""
        if (
            shape not in self._shapes
            and len(self._shapes) >= self.limit
            and self._short_of_memory()
        ):
            self.drop()
        self._shapes.add(shape)

    def drop(self) -> None:
        """Drops every compiled program, once compiles ahead finish.

        Only these functions' programs go, not those of other JAX code in the
        process. Each compiles again when next called, and is compiled ahead
        again as it was before.
        """
        self.wait()
        for function in (*self._compiled.values(), *self._helpers):
            cast(Any, function).clear_cache()
        self._compiled.clear()
        self._shapes.clear()
        self._done.clear()
        self.drops += 1

    def first(self, marker: Hashable) -> bool:
        """Whether `marker` is new since programs were last dropped; it is not after this."""
        if marker in self._done:
            return False
        self._done.add(marker)
        return True

    def compile_later(self, function: Callable[..., Any], *arguments: Any) -> None:
        """Compiles a function for these arguments, or their shapes, in a quiet gap."""
        self.later(
            lambda: cast(jax.stages.Wrapped, function).lower(*arguments).compile()
        )

    def later(self, task: Callable[[], Any]) -> None:
        """Runs a task that only saves time, such as compiling ahead, in a quiet gap.

        A task that fails is dropped: the call that needs what it compiles
        compiles it again, and reports the failure. Nothing runs if the
        programs compile nothing ahead.
        """
        if not self._ahead:
            return

        def run() -> None:
            self._wait_quiet()
            try:
                task()
            except Exception:  # noqa: BLE001
                return

        with self._lock:
            self._pending = [future for future in self._pending if not future.done()]
            self._pending.append(self._thread.submit(run))

    def wait(self) -> None:
        """Waits for every task started in the background."""
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            future.result()

    @contextmanager
    def step(self) -> Iterator[None]:
        """Holds tasks off while a step runs, and for `quiet` seconds after."""
        with self._steps:
            self._running = True
        try:
            yield
        finally:
            with self._steps:
                self._running = False
                self._idle_since = self._clock()
                self._steps.notify_all()

    def _wait_quiet(self) -> None:
        """Waits until no step has run for `quiet` seconds."""
        with self._steps:
            while True:
                left = self._idle_since + self.quiet - self._clock()
                if not self._running and left <= 0:
                    return
                self._steps.wait(None if self._running else left)
