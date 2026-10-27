# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Process checks the serve tests share, the same on Linux and macOS."""

import os
import subprocess
import time


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
    while time.monotonic() < deadline:
        running = [pid for pid in pids if alive(pid)]
        if not running:
            return
        time.sleep(0.1)
    raise AssertionError(f"still running: {running}")
