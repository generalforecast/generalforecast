# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A worker's compile helper: a child process that compiles programs aside of the worker's steps.

A step that meets a query count its decode has no program for would compile
it, which takes seconds; tracing and lowering hold the worker's GIL, so even
a compile on another thread stalls steps. The helper compiles instead: the
worker sends it a program's key and its arguments' shapes, it rebuilds the
program from the model's catalog spec as the worker built it, compiles it,
and sends back the serialized executable, which the worker loads in about a
tenth of a second and calls directly (`Loaded`).

The helper runs at the lowest scheduling priority, so its compile threads
take only cores the worker leaves idle; it never preallocates device memory,
and keeps its parameters in host memory: it only compiles, so its device
holds no more than its client's context. It reads and writes one request at
a time, length-prefixed pickles on its stdin and stdout. `CompileHelper` is
the worker's side: it starts the helper when first asked, sends it one
request at a time, counts a step needs before counts it may need, starts it
again if it dies, and stops it when the worker closes.
"""

import functools
import heapq
import itertools
import logging
import os
import pickle
import select
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Hashable
from typing import IO, Any, cast

logger = logging.getLogger(__name__)

# Set to `0` to compile every program in the worker's own steps.
HELPER_ENV = "GF_COMPILE_HELPER"
# Seconds a request may take before the helper is taken to be stuck: a
# decode compiles in under 15 s on one core.
REQUEST_SECONDS = 300.0
# Times the helper may die before the worker stops asking it, until the
# programs drop; each start waits longer, from `RESTART_SECONDS`.
RESTARTS = 3
RESTART_SECONDS = 5.0
# Seconds without a request after which the helper exits, giving back its
# host memory; the next request starts it again.
IDLE_SECONDS = 60.0
_LENGTH = struct.Struct("<Q")


def _write(stream: IO[bytes], message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_LENGTH.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exactly(stream: IO[bytes], count: int) -> bytes | None:
    data = b""
    while len(data) < count:
        part = stream.read(count - len(data))
        if not part:
            return None
        data += part
    return data


def _read(stream: IO[bytes]) -> Any:
    """The next message, or None once the stream ends."""
    header = _read_exactly(stream, _LENGTH.size)
    if header is None:
        return None
    payload = _read_exactly(stream, _LENGTH.unpack(header)[0])
    return None if payload is None else pickle.loads(payload)


def helper_environment() -> dict[str, str]:
    """The helper's environment: the worker's, with no device memory taken ahead.

    XLA's flags stay the worker's, so its compiles share the worker's keys
    in JAX's persistent compile cache; its compile threads run at the
    helper's lowest priority, on cores the worker leaves idle.
    """
    environment = dict(os.environ)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Device memory goes back as soon as a compile's autotuning frees it.
    environment["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    return environment


def compiles_aside(spec: str | None) -> bool:
    """Whether the helper can rebuild a model's programs: a catalog model, not custom code."""
    return spec is not None and _catalog_model(spec)


@functools.cache
def _catalog_model(spec: str) -> bool:
    from generalforecast.models import registry

    if spec in registry.IMPLEMENTATIONS:
        return False
    try:
        return not registry._catalog.parse(spec).custom
    except Exception:  # noqa: BLE001
        return False


class CompileHelper:
    """The worker's side of its compile helper.

    `submit` queues a compile, once per token, and returns at once; a thread
    sends the queue to the helper one request at a time, lowest priority
    first, and gives each executable it gets back to the request's `done`,
    or None if the request failed. The helper starts on the first request,
    and starts again, after a pause, if it dies; past `RESTARTS` deaths the
    requests fail until `reset`.
    """

    def __init__(self, setup: dict[str, Any]) -> None:
        """Keeps what the helper is started with: the worker's settings, bucket minimum and slot bytes."""
        self._setup = setup
        self._lock = threading.Condition()
        self._queue: list[tuple[int, int, Hashable, dict, Callable[[Any], None]]] = []
        self._tokens: set[Hashable] = set()
        self._order = itertools.count()
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._deaths = 0
        self.compiled = 0
        self.failed = 0
        self.seconds = 0.0

    def submit(
        self,
        token: Hashable,
        priority: int,
        request: dict[str, Any],
        done: Callable[[Any], None],
    ) -> bool:
        """Queues a compile once per token; False if it was queued before, or the helper is closed."""
        with self._lock:
            if self._closed or token in self._tokens:
                return False
            self._tokens.add(token)
            heapq.heappush(
                self._queue, (priority, next(self._order), token, request, done)
            )
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="gf-compile-helper", daemon=True
                )
                self._thread.start()
            self._lock.notify_all()
        return True

    def reset(self) -> None:
        """Forgets what was asked for and how often the helper died, as programs drop; queued requests fail."""
        with self._lock:
            dropped, self._queue = self._queue, []
            self._tokens.clear()
            self._deaths = 0
        for *_, done in dropped:
            done(None)

    def close(self) -> None:
        """Stops the helper and its thread; queued requests fail."""
        with self._lock:
            self._closed = True
            dropped, self._queue = self._queue, []
            process = self._process
            self._lock.notify_all()
        for *_, done in dropped:
            done(None)
        if process is not None:
            _stop(process)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        while True:
            idle = None
            job = None
            with self._lock:
                if not self._queue and not self._closed:
                    self._lock.wait(IDLE_SECONDS)
                    if not self._queue and not self._closed:
                        # Quiet for a while: the helper's memory goes back.
                        idle, self._process = self._process, None
                if self._closed:
                    return
                if self._queue:
                    *_, request, done = heapq.heappop(self._queue)
                    job = (request, done, self._deaths > RESTARTS)
            if idle is not None:
                _stop(idle)
            if job is None:
                continue
            request, done, given_up = job
            result = None if given_up else self._compile(request)
            try:
                done(result)
            except Exception:
                logger.exception("a compiled program could not be kept")

    def _compile(self, request: dict[str, Any]) -> Any:
        """The executable the helper compiles for a request, or None."""
        try:
            process = self._started()
            assert process.stdin is not None and process.stdout is not None
            started = time.perf_counter()
            _write(process.stdin, request)
            ready, _, _ = select.select([process.stdout], [], [], REQUEST_SECONDS)
            if not ready:
                raise TimeoutError(f"no reply in {REQUEST_SECONDS:.0f} s")
            reply = _read(process.stdout)
            if reply is None:
                raise EOFError("the compile helper exited")
        except (OSError, EOFError, TimeoutError, pickle.UnpicklingError) as error:
            self._died(error)
            return None
        if "error" in reply:
            logger.warning("the compile helper failed a compile: %s", reply["error"])
            self.failed += 1
            return None
        from jax.experimental.serialize_executable import deserialize_and_load

        try:
            loaded = deserialize_and_load(
                reply["executable"], reply["in_tree"], reply["out_tree"]
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("a compiled executable failed to load: %s", error)
            self.failed += 1
            return None
        self.compiled += 1
        self.seconds += time.perf_counter() - started
        return loaded

    def _started(self) -> subprocess.Popen:
        """The running helper, started if it is not."""
        with self._lock:
            process = self._process
            deaths = self._deaths
        if process is not None and process.poll() is None:
            return process
        if deaths:
            time.sleep(RESTART_SECONDS * 2 ** (deaths - 1))
        process = subprocess.Popen(
            [sys.executable, "-m", "generalforecast.worker.helper"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=helper_environment(),
        )
        assert process.stdin is not None and process.stdout is not None
        _write(process.stdin, self._setup)
        with self._lock:
            if self._closed:
                _stop(process)
                raise OSError("the worker closed")
            self._process = process
        ready = _read(process.stdout)
        if ready != "ready":
            raise EOFError("the compile helper did not start")
        return process

    def _died(self, error: BaseException) -> None:
        with self._lock:
            process, self._process = self._process, None
            self._deaths += 1
            closed = self._closed
        if process is not None:
            _stop(process)
        if not closed:
            logger.warning("the compile helper stopped: %s", error)
        self.failed += 1


def _stop(process: subprocess.Popen) -> None:
    """Ends the helper: its stdin closes, then it is killed if it lingers."""
    try:
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()


def main() -> None:
    """Runs the helper: one compile per request, until stdin closes."""
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    except (AttributeError, OSError):
        pass
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Libraries that print would corrupt the replies.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    requests = sys.stdin.buffer
    setup = _read(requests)
    if setup is None:
        return

    import jax
    from jax.experimental.serialize_executable import serialize

    from generalforecast.models import registry
    from generalforecast.worker.runner import Worker

    worker = Worker(
        setup["settings"],
        setup["bucket_minimum"],
        setup["slot_bytes"],
        compile_cache=setup["compile_cache"],
        compile_ahead=False,
    )
    models: dict[str, int] = {}
    host = jax.devices("cpu")[0]
    _write(replies, "ready")
    while (request := _read(requests)) is not None:
        reply: dict[str, Any]
        try:
            spec = request["spec"]
            if spec not in models:
                with jax.default_device(host):
                    worker.add_model(len(models), *registry.load_engine_model(spec))
                models[spec] = len(models)
            key = request["key"]
            function = worker._program((key[0], models[spec], *key[2:]))
            jitted = cast(jax.stages.Wrapped, function)
            compiled = jitted.lower(*request["shapes"]).compile()
            executable, in_tree, out_tree = serialize(compiled)
            reply = {"executable": executable, "in_tree": in_tree, "out_tree": out_tree}
        except Exception as error:  # noqa: BLE001
            reply = {"error": f"{type(error).__name__}: {error}"}
        try:
            _write(replies, reply)
        except OSError:
            # The worker closed its end: nothing waits for the reply.
            return


if __name__ == "__main__":
    main()
