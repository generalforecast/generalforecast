# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import fcntl
import hashlib
import importlib
import json
import logging
import os
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

from generalforecast.constants import (
    COMPILED_SHAPES,
    RECORDED_KINDS,
    RECORDED_PROGRAMS,
    RECORDED_SECONDS,
    WARM_QUIET,
)
from generalforecast.runtime.budget import forget_compiled, note_compiled
from generalforecast.runtime.device import DeviceSettings, with_precision
from generalforecast.system import mappings_short

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


# The types a manifest's JSON may name beside this package's own.
_BUILTIN_TYPES = {
    f"builtins:{kind.__qualname__}": kind for kind in (tuple, list, dict, type(None))
}


def _type_name(kind: type) -> str:
    return f"{kind.__module__}:{kind.__qualname__}"


def _named_type(name: str) -> type:
    """A type a manifest names: a builtin container or one of this package's, never another module's."""
    if name in _BUILTIN_TYPES:
        return _BUILTIN_TYPES[name]
    module, _, qualname = name.partition(":")
    if module.split(".")[0] != "generalforecast" or module == "builtins":
        raise ValueError(f"a manifest may not name {name}")
    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    if not isinstance(found, type):
        raise TypeError(f"{name} is not a type")
    return found


def encoded(value: Any) -> Any:
    """A recorded value as JSON: containers, dtypes, shapes, tree structures and this package's dataclasses and named tuples.

    Raises:
        TypeError: For a value of another type.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.bool_ | np.integer | np.floating):
        return value.item()
    if isinstance(value, np.dtype):
        return {"dtype": value.str}
    if isinstance(value, jax.ShapeDtypeStruct):
        return {
            "shape": list(value.shape),
            "dtype": np.dtype(value.dtype).str,
            "weak": bool(value.weak_type),
        }
    if isinstance(value, jax.tree_util.PyTreeDef):
        return {"tree": _encoded_tree(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return {
            "class": _type_name(type(value)),
            "fields": {
                field.name: encoded(getattr(value, field.name)) for field in fields
            },
        }
    if isinstance(value, tuple) and type(value) is not tuple:
        return {
            "class": _type_name(type(value)),
            "items": [encoded(item) for item in value],
        }
    if isinstance(value, tuple):
        return {"tuple": [encoded(item) for item in value]}
    if isinstance(value, list):
        return [encoded(item) for item in value]
    if isinstance(value, set | frozenset):
        return {"set": [encoded(item) for item in value]}
    if isinstance(value, dict):
        return {"dict": [[encoded(key), encoded(item)] for key, item in value.items()]}
    raise TypeError(f"a manifest cannot hold a {type(value).__name__}")


def decoded(value: Any) -> Any:
    """The value `encoded` wrote.

    Raises:
        ValueError: If it names a type outside this package and builtin containers.
    """
    if isinstance(value, list):
        return [decoded(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "shape" in value:
        return jax.ShapeDtypeStruct(
            tuple(value["shape"]), np.dtype(value["dtype"]), weak_type=value["weak"]
        )
    if "dtype" in value:
        return np.dtype(value["dtype"])
    if "tree" in value:
        return _decoded_tree(value["tree"])
    if "tuple" in value:
        return tuple(decoded(item) for item in value["tuple"])
    if "set" in value:
        return {decoded(item) for item in value["set"]}
    if "dict" in value:
        return {decoded(key): decoded(item) for key, item in value["dict"]}
    kind = _named_type(value["class"])
    if "items" in value:
        return kind(*(decoded(item) for item in value["items"]))
    return kind(**{name: decoded(item) for name, item in value["fields"].items()})


def _encoded_tree(tree: Any) -> Any:
    data = tree.node_data()
    if data is None:
        return None
    kind, aux = data
    return [
        _type_name(kind),
        encoded(aux),
        [_encoded_tree(child) for child in tree.children()],
    ]


def _decoded_tree(node: Any) -> Any:
    if node is None:
        return jax.tree.structure(0)
    name, aux, children = node
    return jax.tree_util.PyTreeDef.from_node_data_and_children(
        jax.tree_util.default_registry,
        (_named_type(name), decoded(aux)),
        [_decoded_tree(child) for child in children],
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


def most_rows(signature: Signature) -> int:
    """The longest first axis of a signature's leaves: a table's rows, for a call reading one."""
    return max((shape[0] for shape, _, _ in signature[1] if shape), default=0)


def zeros(tree: Any) -> Any | None:
    """Zeros of each leaf's shape, dtype and weak type, for a tree of shapes; None if a leaf is a value."""
    leaves, structure = jax.tree.flatten(tree)
    if not all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in leaves):
        return None

    def zero(leaf: jax.ShapeDtypeStruct) -> Any:
        if leaf.weak_type:
            # Only a Python scalar makes a weakly typed array.
            weak = jax.lax.full(leaf.shape, np.dtype(leaf.dtype).type(0).item())
            if weak.dtype == leaf.dtype:
                return weak
        return np.zeros(leaf.shape, leaf.dtype)

    return jax.tree.unflatten(structure, [zero(leaf) for leaf in leaves])


def _compile(function: Callable[..., Any], arguments: tuple) -> None:
    """Compiles a function for these arguments, or their shapes, keeping its program's FLOPs."""
    jitted = cast(jax.stages.Wrapped, function)
    note_compiled(function, arguments, jitted.lower(*arguments).compile())


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
    """What a model's manifest holds: each kind of call, least recently used first, with the item counts it took, and what was measured by compiling.

    A measure is a tuple its worker reads, such as a table width whose
    sizes were measured; the latest `RECORDED_KINDS` are kept. `used` holds
    when a worker last used each kind, in seconds since the epoch.
    """

    kinds: dict[Kind, set[int]]
    measures: dict[tuple, None]
    used: dict[Kind, float] = dataclasses.field(default_factory=dict)

    def programs(self, kind: Kind) -> int:
        """Programs a kind compiles at a start: each count it took, and each power of two up to the most."""
        return len(counts_up_to(self.kinds[kind]))

    def prune(self, now: float) -> None:
        """Forgets the least recently used kinds past `RECORDED_PROGRAMS` programs, and those unused for `RECORDED_SECONDS`."""
        for kind in [
            kind
            for kind, used in self.used.items()
            if kind in self.kinds and now - used > RECORDED_SECONDS
        ]:
            del self.kinds[kind]
        total = sum(self.programs(kind) for kind in self.kinds)
        while self.kinds and (
            total > RECORDED_PROGRAMS or len(self.kinds) > RECORDED_KINDS
        ):
            oldest = next(iter(self.kinds))
            total -= self.programs(oldest)
            del self.kinds[oldest]
        self.used = {kind: self.used[kind] for kind in self.kinds if kind in self.used}

    def by_use(self) -> list[tuple[Kind, set[int]]]:
        """The kinds, most recently used first, with their counts."""
        return list(reversed(self.kinds.items()))


class Manifest:
    """The kinds of call one model compiled for, the item counts of each and what was measured, on disk for later processes.

    A file of its own per model, read and rewritten whole under a lock, so
    workers sharing a machine add to it rather than overwrite each other.
    It is bounded as `Recorded.prune` says, by last use, so the programs a
    start compiles stay few however many shapes were ever served. A file
    that cannot be read, such as one an older version wrote, reads as empty.
    """

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        """A manifest kept at `path`, its uses timed by `clock`."""
        self.path = path
        self._clock = clock

    def read(self) -> Recorded:
        """What is recorded."""
        try:
            with open(self.path) as file:
                recorded = decoded(json.load(file))
        except FileNotFoundError:
            return Recorded({}, {})
        except Exception:  # noqa: BLE001
            logger.warning("ignoring an unreadable program manifest %s", self.path)
            return Recorded({}, {})
        return recorded if isinstance(recorded, Recorded) else Recorded({}, {})

    def add(self, kind: Kind, count: int) -> None:
        """Records a kind's call at `count` items as the latest used, and forgets what `Recorded.prune` does."""

        def added(recorded: Recorded) -> None:
            counts = recorded.kinds.pop(kind, set())
            counts.add(count)
            recorded.kinds[kind] = counts
            now = self._clock()
            recorded.used[kind] = now
            recorded.prune(now)

        self._update(added)

    def add_measure(self, measure: tuple) -> None:
        """Records a measure as the latest taken."""

        def added(recorded: Recorded) -> None:
            recorded.measures.pop(measure, None)
            recorded.measures[measure] = None
            while len(recorded.measures) > RECORDED_KINDS:
                del recorded.measures[next(iter(recorded.measures))]

        self._update(added)

    def measured(self) -> dict[Hashable, Any]:
        """The measures kept beside the manifest, by key; empty if none can be read."""
        try:
            with open(self._measures) as file:
                kept = decoded(json.load(file))
        except FileNotFoundError:
            return {}
        except Exception:  # noqa: BLE001
            logger.warning("ignoring unreadable measures %s", self._measures)
            return {}
        return kept if isinstance(kept, dict) else {}

    def keep_measured(self, key: Hashable, value: Any) -> None:
        """Keeps a measure beside the manifest, with those other workers kept."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            kept = self.measured()
            kept[key] = value
            try:
                text = json.dumps(encoded(kept))
            except TypeError:
                logger.warning("not keeping a measure JSON cannot hold: %r", key)
                return
            part = self._measures.with_suffix(f".{os.getpid()}.part")
            part.write_text(text)
            os.replace(part, self._measures)

    @property
    def _measures(self) -> Path:
        return self.path.with_suffix(".measures")

    def _update(self, change: Callable[[Recorded], None]) -> None:
        """Reads, changes and rewrites the file, under its lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            recorded = self.read()
            change(recorded)
            part = self.path.with_suffix(f".{os.getpid()}.part")
            part.write_text(json.dumps(encoded(recorded)))
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
        short_of_mappings: Callable[[], bool] = mappings_short,
        dropped: Callable[[], None] | None = None,
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
            short_of_mappings: Whether the process holds most of the memory
                mappings it may, asked at each new shape.
            dropped: Called once every program is dropped, to forget what
                held for them, such as the counts a kind has programs for.
        """
        self._settings = settings
        self._short_of_memory = short_of_memory
        self._short_of_mappings = short_of_mappings
        self._helpers = tuple(helpers)
        self._dropped = dropped
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
        self._soon = ThreadPoolExecutor(1, thread_name_prefix="gf-soon")
        # Tasks that only save time, which a drop cancels, and tasks steps
        # wait on, which it never does.
        self._later: list[Future] = []
        self._pending: list[Future] = []
        self._lock = threading.Lock()
        # Whether a step runs, and when the last ended; compiles ahead wait on
        # them. A drop moves the generation on, so those queued before it end.
        self._steps = threading.Condition()
        self._running = False
        self._idle_since = clock()
        self._generation = 0
        # The generation of the task this thread runs, which the tasks it
        # queues keep, so a drop ends them too.
        self._running_generation = threading.local()
        # Whether the last drop for memory mappings left them short, so the
        # programs were not what held them.
        self._mappings_futile = False
        self._manifests = manifests
        # Kinds and counts recorded by this process, which need no rewrite.
        self._recorded: set[tuple[str, Hashable, int]] = set()
        # Each model's measures, by spec and key, as read from its manifest.
        self._measures: dict[str, dict[Hashable, Any]] = {}
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
        return Manifest(self._manifests / f"{digest}.json")

    def remember(self, model: str, kind: Kind, count: int) -> None:
        """Records that a model's kind of call ran at `count` items, for later processes to compile at start.

        Recorded once a process, at its first call, so a manifest keeps the
        kinds by when a worker last used them.
        """
        manifest = self.manifest(model)
        if manifest is None or not self._record((model, kind, count)):
            return
        try:
            manifest.add(kind, count)
        except OSError as error:
            logger.warning("cannot record a compiled program: %s", error)

    def remember_measure(self, model: str, measure: tuple) -> None:
        """Records that a model measured something by compiling, which later processes measure at start."""
        manifest = self.manifest(model)
        if manifest is None or not self._record((model, measure, 0)):
            return
        try:
            manifest.add_measure(measure)
        except OSError as error:
            logger.warning("cannot record a measure: %s", error)

    def _record(self, entry: tuple[str, Hashable, int]) -> bool:
        """Whether `entry` is new to this process's records; it is not after this."""
        with self._lock:
            if entry in self._recorded:
                return False
            self._recorded.add(entry)
            return True

    def measured(self, model: str, key: Hashable, measure: Callable[[], Any]) -> Any:
        """What `measure` returns for a model's `key`, kept on disk for later processes.

        Measures follow from the model, its shapes and the device alone, so a
        later process reads them rather than compiling programs to take them
        again. Without manifests, `measure` runs each time.
        """
        manifest = self.manifest(model)
        if manifest is None:
            return measure()
        with self._lock:
            kept = self._measures.get(model)
        if kept is None:
            kept = manifest.measured()
            with self._lock:
                kept = self._measures.setdefault(model, kept)
        if key in kept:
            return kept[key]
        value = measure()
        with self._lock:
            kept[key] = value
        try:
            manifest.keep_measured(key, value)
        except OSError as error:
            logger.warning("cannot keep a measure: %s", error)
        return value

    def recorded(self, model: str) -> Recorded:
        """What a model compiled for and served in earlier processes."""
        manifest = self.manifest(model)
        return Recorded({}, {}) if manifest is None else manifest.read()

    def add_shape(self, shape: Hashable) -> None:
        """Counts a table shape, dropping every program first if memory is short past `limit`, or mappings are.

        Short of mappings, a new shape drops the programs whatever their
        count, since the next compile would fail. A drop that leaves them
        short shows the programs were not what held them, so until mappings
        free up, they drop for mappings only past `limit` shapes, as for
        memory, rather than at every new shape.
        """
        with self._lock:
            if shape in self._shapes:
                return
            crowded = len(self._shapes) >= self.limit
        mappings = self._short_of_mappings()
        if not mappings:
            self._mappings_futile = False
        if (mappings and (crowded or not self._mappings_futile)) or (
            crowded and self._short_of_memory()
        ):
            self.drop()
            if mappings:
                futile = self._short_of_mappings()
                if futile and not self._mappings_futile:
                    logger.warning(
                        "dropping compiled programs left the process short of "
                        "memory mappings, so they drop again only past %d table "
                        "shapes; raise vm.max_map_count to compile more",
                        self.limit,
                    )
                self._mappings_futile = futile
        with self._lock:
            self._shapes.add(shape)

    def drop(self) -> None:
        """Drops every compiled program, once the compiles under way finish.

        Only these functions' programs go, not those of other JAX code in the
        process. Each compiles again when next called, and is compiled ahead
        again as it was before. Compiles ahead not started yet are dropped
        with them rather than compiled for nothing; those steps wait on are
        let finish.
        """
        with self._steps:
            self._generation += 1
            self._steps.notify_all()
        with self._lock:
            later, self._later = self._later, []
        for future in later:
            future.cancel()
        self.wait(later)
        for function in (*self._compiled.values(), *self._helpers):
            cast(Any, function).clear_cache()
        with self._lock:
            self._compiled.clear()
            self._shapes.clear()
            self._done.clear()
        forget_compiled()
        self.drops += 1
        if self._dropped is not None:
            self._dropped()

    def first(self, marker: Hashable) -> bool:
        """Whether `marker` is new since programs were last dropped; it is not after this."""
        with self._lock:
            if marker in self._done:
                return False
            self._done.add(marker)
            return True

    def compile_later(self, function: Callable[..., Any], *arguments: Any) -> None:
        """Compiles a function for these arguments, or their shapes, in a quiet gap."""
        self.later(lambda: _compile(function, arguments))

    def compile_soon(
        self,
        function: Callable[..., Any],
        arguments: tuple,
        done: Callable[[bool], None],
    ) -> None:
        """Compiles a function for these arguments beside steps, then tells `done` whether it compiled.

        For a program steps already wait on, so unlike `later` it does not
        wait for a quiet gap. A compile that fails is logged.
        """

        def compile_it() -> None:
            try:
                _compile(function, arguments)
            except Exception as error:  # noqa: BLE001
                logger.warning("a compile beside steps failed: %s", error)
                done(False)
                return
            done(True)

        self.soon(compile_it)

    def soon(self, task: Callable[[], Any]) -> None:
        """Runs a task on a thread of its own beside steps, without waiting for a quiet gap."""
        with self._lock:
            self._pending = [future for future in self._pending if not future.done()]
            self._pending.append(self._soon.submit(task))

    def later(self, task: Callable[[], Any]) -> None:
        """Runs a task that only saves time, such as compiling ahead, in a quiet gap.

        A task that fails is dropped: the call that needs what it compiles
        compiles it again, and reports the failure. Nothing runs if the
        programs compile nothing ahead, and a task still waiting when every
        program drops never runs.
        """
        if not self._ahead:
            return
        generation = getattr(self._running_generation, "value", None)
        if generation is None:
            with self._steps:
                generation = self._generation

        def run() -> None:
            if not self._wait_quiet(generation):
                return
            self._running_generation.value = generation
            try:
                task()
            except Exception:  # noqa: BLE001
                return
            finally:
                self._running_generation.value = None

        with self._lock:
            self._later = [future for future in self._later if not future.done()]
            self._later.append(self._thread.submit(run))

    def wait(self, tasks: Iterable[Future] | None = None) -> None:
        """Waits for every task started in the background, those they start too, or for `tasks` and those steps wait on.

        A task cancelled before it started is not waited for.
        """
        waited: set[Future] = set()
        chosen = None if tasks is None else list(tasks)
        while True:
            with self._lock:
                pending = [
                    future
                    for future in [
                        *self._pending,
                        *(self._later if chosen is None else chosen),
                    ]
                    if future not in waited
                ]
            if not pending:
                return
            for future in pending:
                waited.add(future)
                if not future.cancelled():
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

    def _wait_quiet(self, generation: int) -> bool:
        """Waits until no step has run for `quiet` seconds; False if programs dropped since `generation`."""
        with self._steps:
            while True:
                if self._generation != generation:
                    return False
                left = self._idle_since + self.quiet - self._clock()
                if not self._running and left <= 0:
                    return True
                self._steps.wait(None if self._running else left)
