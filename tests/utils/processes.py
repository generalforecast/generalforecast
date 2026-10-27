# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The server processes tests start, each in a process group of its own.

A server's workers share its group, so stopping the group stops them all.
Every group still running when the test process ends, whether its tests
finished, failed, timed out or were terminated, is killed: on interpreter
exit, and on SIGTERM or SIGHUP, as pytest-xdist and CI runners send them.

A process killed outright runs no exit handler, so each server also gets
the read end of a pipe in `GF_PARENT_FD`, whose write end only this process
holds: gf-server and gf-router stop when it closes, however this process
ends. `gf serve` hands the variable and the descriptor on to the gf-server
it becomes.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from typing import Any

# The variable that names the descriptor a server watches for its parent's exit.
PARENT_FD = "GF_PARENT_FD"
# The groups started and not yet stopped, by leader pid, with the write end
# of the pipe each one's server watches.
_STARTED: dict[int, tuple[subprocess.Popen, int]] = {}
_ARMED = False


def start(arguments: list[str], **options: Any) -> subprocess.Popen:
    """Starts a process as the leader of a new process group, watching this one."""
    _arm()
    watched, holder = os.pipe()
    environment = dict(options.pop("env", None) or os.environ)
    environment[PARENT_FD] = str(watched)
    try:
        process = subprocess.Popen(
            arguments,
            start_new_session=True,
            pass_fds=(watched, *options.pop("pass_fds", ())),
            env=environment,
            **options,
        )
    except BaseException:
        os.close(holder)
        raise
    finally:
        os.close(watched)
    _STARTED[process.pid] = (process, holder)
    return process


def stop(
    process: subprocess.Popen, *, sent: int = signal.SIGTERM, timeout: float = 30
) -> int | None:
    """Sends `sent` to the leader, waits up to `timeout`, then kills its group.

    The leader alone gets `sent`, so it shuts its workers down as it would
    under a supervisor; whatever of the group is left after it exits, or
    after `timeout`, is killed.

    Returns:
        The leader's exit code, or None if it had to be killed.
    """
    code: int | None = None
    try:
        if process.poll() is None:
            process.send_signal(sent)
        code = process.wait(timeout)
    except subprocess.TimeoutExpired:
        pass
    finally:
        _kill_group(process)
    return code


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        pass
    _, holder = _STARTED.pop(process.pid, (None, None))
    if holder is not None:
        os.close(holder)


@contextlib.contextmanager
def scoped() -> Iterator[None]:
    """Stops every group started inside the block when it ends, however it ends."""
    before = set(_STARTED)
    try:
        yield
    finally:
        for pid in set(_STARTED) - before:
            _kill_group(_STARTED[pid][0])


def stop_all() -> None:
    """Kills every group still running."""
    for process, _ in list(_STARTED.values()):
        _kill_group(process)


def _arm() -> None:
    """Kills the groups at exit and on a terminating signal, once."""
    global _ARMED
    if _ARMED:
        return
    _ARMED = True
    atexit.register(stop_all)
    for number in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(number) is signal.SIG_DFL:
            signal.signal(number, _terminated)


def _terminated(number: int, _frame: Any) -> None:
    stop_all()
    signal.signal(number, signal.SIG_DFL)
    os.kill(os.getpid(), number)


def children(pid: int) -> list[int]:
    """The processes whose parent is `pid`."""
    listed = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False
    )
    return [int(child) for child in listed.stdout.split()]


def alive(pid: int) -> bool:
    """Whether `pid` is running; an exited process not yet reaped is not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def assert_exited(pids: list[int], seconds: float = 30) -> None:
    """Waits up to `seconds` for every one of `pids` to exit."""
    deadline = time.monotonic() + seconds
    running = list(pids)
    while time.monotonic() < deadline:
        running = [pid for pid in pids if alive(pid)]
        if not running:
            return
        time.sleep(0.1)
    raise AssertionError(f"still running: {running}")
