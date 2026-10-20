import threading
import time
from collections.abc import Callable, Hashable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, cast

import jax

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


def ladder(first: int, count: int) -> list[int]:
    """`count` padded sizes from `first`, a padded size, each double the last, as buckets are."""
    return [first << step for step in range(count)]


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
    ) -> None:
        """Sets up with nothing compiled.

        Args:
            settings: Device settings for matrix multiply precision, if any.
            short_of_memory: Whether memory is short, asked past `limit` shapes.
            helpers: Other compiled functions whose programs drop with these.
            clock: Seconds, as `time.monotonic` counts them.
            ahead: Whether to compile ahead at all; without it, each program
                compiles in the step that first calls it.
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
        if key not in self._compiled:
            function = build()
            if self._settings is not None:
                function = with_precision(function, self._settings.matmul_precision)
            self._compiled[key] = jax.jit(function)
        return self._compiled[key]

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

    def forget(self, marker: Hashable) -> None:
        """Forgets a marker, as of a table removed."""
        self._done.discard(marker)

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
