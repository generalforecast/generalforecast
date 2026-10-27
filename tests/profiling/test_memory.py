# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import mmap
import sys
from pathlib import Path

import numpy as np
import pytest

from generalforecast.profiling import memory


def _held(size: int) -> mmap.mmap:
    """Memory the process holds until it is closed, which every platform takes back at once.

    A NumPy array's buffer goes back to its allocator when freed, which on
    macOS may keep it, so the test maps pages of its own.
    """
    mapped = mmap.mmap(-1, size)
    view = np.frombuffer(mapped, dtype=np.uint8)
    view[:] = 1
    del view
    return mapped


@pytest.mark.skipif(
    sys.platform not in ("linux", "darwin"), reason="no resettable peak to read"
)
def test_a_reset_starts_the_host_peak_from_what_the_process_holds() -> None:
    memory.reset()
    held = _held(256 << 20)
    before = memory.reading()["host_peak"]
    held.close()

    memory.reset()

    assert memory.reading()["host_peak"] < before


def test_a_reading_leaves_out_what_the_platform_does_not_report() -> None:
    assert set(memory.reading()) <= {"host_peak", "device_in_use"}


def test_a_refused_reset_leaves_the_host_peak_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(memory, "_libproc", lambda: None)
    monkeypatch.setattr(memory, "_CLEAR_REFS", tmp_path / "missing" / "clear_refs")
    try:
        memory.reset()
        assert "host_peak" not in memory.reading()
    finally:
        monkeypatch.undo()
        memory.reset()
