# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the operating system lets this process use, the same on Linux and macOS.

Calls only some platforms have, like Linux's CPU affinity, go through here,
so the code that needs them runs everywhere.
"""

import os
from collections.abc import Iterable


def usable_cpus() -> int:
    """The cores this process may run on: its affinity on Linux, every core elsewhere."""
    affinity = getattr(os, "sched_getaffinity", None)
    count = len(affinity(0)) if affinity else os.cpu_count()
    return max(1, count or 1)


def pin_cpus(cpus: Iterable[int]) -> None:
    """Keeps this process on `cpus` where the platform allows it, and does nothing elsewhere."""
    pin = getattr(os, "sched_setaffinity", None)
    if pin is not None:
        pin(0, list(cpus))
