# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The thread that runs a local node's steps, so concurrent callers share them.

A `LocalClient` and the `generalforecast.engine.LocalEngine` over it submit
through one `Driver`: each call hands it the requests to queue and blocks on a
future, and the driver's thread queues every call that has arrived, plans the
node's next step over all of them, runs it on the worker and answers the calls
it completes. A call arriving while a step runs joins the next one, so callers
with different tables share steps whenever the node batches their tables
together, as a server's shard does.
"""

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from generalforecast._core import Node, Step
from generalforecast.native.client import Answer
from generalforecast.worker import Worker


@dataclass(frozen=True)
class Answered:
    """What a call's steps returned.

    Attributes:
        predictions: Each submission's predictions, by its name.
        answers: Each answered submission's epoch and share of the steps'
            time, by its name.
        past_cap: Tables that read a context longer than their model's cap
            while the call was queued.
    """

    predictions: dict[str, np.ndarray]
    answers: dict[str, Answer]
    past_cap: frozenset[int]


@dataclass(eq=False)
class _Call:
    """One caller's submissions, its requests once queued and what they returned."""

    submissions: Sequence[tuple[str, Callable[[], int]]]
    future: Future[Answered] = field(default_factory=Future)
    requests: dict[int, str] = field(default_factory=dict)
    predictions: dict[int, np.ndarray] = field(default_factory=dict)
    answers: dict[int, Answer] = field(default_factory=dict)
    past_cap: set[int] = field(default_factory=set)
    error: BaseException | None = None
    settled: bool = False

    @property
    def complete(self) -> bool:
        return len(self.predictions) == len(self.requests)

    def finish(self) -> None:
        """Answers the caller with its error or what its steps returned."""
        if self.error is not None:
            self.future.set_exception(self.error)
            return
        self.future.set_result(
            Answered(
                {name: self.predictions[r] for r, name in self.requests.items()},
                {
                    name: self.answers[r]
                    for r, name in self.requests.items()
                    if r in self.answers
                },
                frozenset(self.past_cap),
            )
        )


class Driver:
    """Runs a node's steps on one thread, for every caller at once.

    The driver holds `lock` while it touches the node or the worker; anything
    else that does, such as a table's writes, takes it too, and so runs
    between steps. A caller must not hold `lock` while it waits on `answer`,
    since the driver needs it to answer.

    Attributes:
        lock: Guards the node and the worker, which are not safe for
            concurrent use. Reentrant, so a table dropped by the garbage
            collector mid-step cannot deadlock.
        fitting: Each tabular table's model and feature count, which measure
            its fits.
    """

    def __init__(self, node: Node, worker: Worker) -> None:
        """Drives `node`'s steps on `worker`; the thread starts with the first call."""
        self._node = node
        self._worker = worker
        self.lock = threading.RLock()
        self.fitting: dict[int, tuple[int, int]] = {}
        self._arrived = threading.Lock()
        self._queue: list[_Call] = []
        self._thread: threading.Thread | None = None
        self._closed = False
        self._settled: list[_Call] = []

    def answer(self, submissions: Sequence[tuple[str, Callable[[], int]]]) -> Answered:
        """Queues requests, blocks until the steps they plan into answer them.

        Args:
            submissions: Names, and calls that queue a request on the node
                and return its id; the driver makes them, holding `lock`.
                Nothing they queue outlives the call.

        Returns:
            What the steps returned, by name.

        Raises:
            KeyError: If the node refuses a request naming what does not exist.
            ValueError: If the node refuses a request otherwise, naming its table.
            RuntimeError: If the driver is closed.
        """
        call = _Call(submissions)
        if not submissions:
            call.finish()
            return call.future.result()
        with self._arrived:
            if self._closed:
                raise RuntimeError("the client is closed")
            self._queue.append(call)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._drive, name="gf-driver", daemon=True
                )
                self._thread.start()
        return call.future.result()

    def close(self) -> None:
        """Answers what is queued, then waits for the thread to end."""
        with self._arrived:
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _drive(self) -> None:
        """Queues arrived calls and runs steps until closed with nothing to answer."""
        active: dict[int, _Call] = {}
        # Whether the last step ran: the node may plan work of its own after
        # its requests are answered, such as a rebuild, which runs before the
        # driver waits.
        ran = False
        while True:
            with self._arrived:
                if not self._queue and not active and not ran:
                    # Idle: the thread ends, so an idle client holds no thread
                    # and is collected like any other; the next call starts one.
                    self._thread = None
                    return
                arrived, self._queue = self._queue, []
            calls = {*active.values(), *arrived}
            try:
                with self.lock:
                    for call in arrived:
                        self._submit(call, active)
                    ran = self._step(active)
                    # Work the node plans of its own once its requests are
                    # answered runs before they are, so a caller that returns
                    # finds the node at rest.
                    while ran and not active and not self._queue:
                        ran = self._step(active)
            except BaseException as error:  # noqa: BLE001
                ran = False
                # A failure outside any one step: no caller may wait forever.
                for call in calls:
                    self._fail(call, error, active)
            settled, self._settled = self._settled, []
            for call in settled:
                call.finish()

    def _submit(self, call: _Call, active: dict[int, _Call]) -> None:
        """Queues a call's requests, failing the call if the node refuses one."""
        for name, submit in call.submissions:
            try:
                request = submit()
            except (KeyError, ValueError) as error:
                raised = type(error)(f"table {name!r}: {error}")
                self._fail(call, raised, active)
                return
            call.requests[request] = name
            active[request] = call

    def _step(self, active: dict[int, _Call]) -> bool:
        """Runs the node's next step and answers the calls it completes; whether one ran."""
        step = self._next_step()
        if step is None and not active:
            return False
        if step is None:
            # The node refuses what it cannot plan as it plans.
            self._settle(active)
            error = RuntimeError("the node planned no step for queued requests")
            for call in set(active.values()):
                self._fail(call, error, active)
            return False
        spent = sum(self._worker.stats.seconds.values())
        try:
            predictions = self._worker.run(step)
        except BaseException as error:  # noqa: BLE001
            # The worker may hold only part of what the step encoded.
            for movement in self._node.fail(step):
                self._worker.apply(movement)
            if self._worker.lost:
                # Its device pool went with a failed write.
                self._node.forget_cache()
                self._worker.forget_cache()
            ran = {batch.request for batch in step.queries}
            ran |= {forecast.request for forecast in step.forecasts}
            for call in {active[r] for r in ran if r in active}:
                self._fail(call, error, active)
            self._settle(active)
            return True
        for movement in self._node.complete(step):
            self._worker.apply(movement)
        for request, prediction in predictions.items():
            if request in active:
                active[request].predictions[request] = prediction
        for request, answer in _answers(step, self._worker, spent).items():
            if request in active:
                active[request].answers[request] = answer
        self._settle(active)
        return True

    def _settle(self, active: dict[int, _Call]) -> None:
        """Fails refused calls, notes long contexts and finishes complete calls."""
        for request, error, missing in self._node.take_refused():
            if request in active:
                call = active[request]
                raised = KeyError if missing else ValueError
                self._fail(
                    call, raised(f"table {call.requests[request]!r}: {error}"), active
                )
        past_cap = self._node.take_past_cap()
        for call in set(active.values()):
            call.past_cap.update(past_cap)
            if call.complete and not call.settled:
                self._release(call, active)
                call.settled = True
                self._settled.append(call)

    def _next_step(self) -> Step | None:
        """The next step the node plans, once the tables it holds for their fits' rows are measured."""
        while True:
            step = self._node.schedule()
            waiting = self._node.take_unmeasured()
            for table in waiting:
                model, features = self.fitting[table]
                self._node.set_fit_limits(
                    table, *self._worker.fit_rows(model, features)
                )
            if step is not None or not waiting:
                return step

    def _fail(
        self, call: _Call, error: BaseException, active: dict[int, _Call]
    ) -> None:
        """Settles a call with an error, cancelling whatever of it is still queued."""
        if call.settled:
            return
        self._release(call, active)
        call.error = error
        call.settled = True
        self._settled.append(call)

    def _release(self, call: _Call, active: dict[int, _Call]) -> None:
        """Cancels a call's requests on the node, so none runs or fails in a later step."""
        for request in call.requests:
            active.pop(request, None)
            self._node.cancel(request)


def _answers(step: Step, worker: Worker, before: float) -> dict[int, Answer]:
    """Each request's epoch and share of a step: by the queries' units read, and by the forecasts made."""
    spent = sum(worker.stats.seconds.values()) - before
    read = sum(len(batch.units) for batch in step.queries)
    read = (read + len(step.forecasts)) or 1
    answers = {
        batch.request: Answer(batch.epoch, spent * len(batch.units) / read)
        for batch in step.queries
    }
    made: dict[int, tuple[int, int]] = {}
    for forecast in step.forecasts:
        epoch, count = made.get(forecast.request, (forecast.epoch, 0))
        made[forecast.request] = (epoch, count + 1)
    for request, (epoch, count) in made.items():
        answers[request] = Answer(epoch, spent * count / read)
    return answers
