# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ctrl-C: the workers leave it to the server, which stops them all without a word."""

import os
import pty
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

from generalforecast import protocol
from tests.utils import processes
from tests.utils.processes import assert_exited as _assert_exited
from tests.utils.processes import children as _children

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _stopped():
    """Every process a test here starts is stopped, with its group, when it ends."""
    with processes.scoped():
        yield


def _assert_quiet(stderr):
    """No traceback anywhere, and no warning or error once the stop began."""
    assert "Traceback" not in stderr and "KeyboardInterrupt" not in stderr, stderr
    lines = stderr.splitlines()
    begun = next((i for i, line in enumerate(lines) if "draining" in line), len(lines))
    noisy = [line for line in lines[begun:] if "WARN" in line or "ERROR" in line]
    assert not noisy, stderr


def test_a_worker_ignores_sigint_and_exits_when_its_socket_closes():
    # A socket path holds at most 104 bytes on macOS, which pytest's own
    # temporary directories outgrow; the system's is short enough.
    with (
        tempfile.TemporaryDirectory(prefix="gfsock") as short,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener,
    ):
        path = Path(short) / "worker.sock"
        listener.bind(str(path))
        listener.listen(1)
        worker = processes.start(
            [
                sys.executable,
                "-m",
                "generalforecast.worker_process",
                "--socket",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
        )
        listener.settimeout(300)
        connection, _ = listener.accept()
    with connection:
        header, _ = protocol.read_frame(connection)
        assert header["kind"] == "ready"
        worker.send_signal(signal.SIGINT)
        time.sleep(1)
        assert worker.poll() is None
        connection.shutdown(socket.SHUT_RDWR)
    assert worker.wait(60) == 0
    stdout, stderr = worker.communicate()
    assert stdout == ""
    _assert_quiet(stderr)


def _server(tmp_path, *, stdout):
    from tests.utils.server import KEY, Server, build

    server = Server(build(), tmp_path)
    # A fresh cache: with programs recorded by earlier processes the workers
    # would compile them at start and the terminal would show that progress.
    environment = {
        **os.environ,
        "GF_TEST_KEY": KEY,
        "PYTHONPATH": str(ROOT),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    process = processes.start(
        [str(server.binary), "run", "--config", str(server.config)],
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    return server, process


def _wait_healthy(server, process):
    from pyarrow import flight

    server.client = flight.FlightClient(f"grpc://127.0.0.1:{server.port}")
    deadline = time.monotonic() + 300
    while True:
        try:
            server.action("health")
            return
        except flight.FlightError:
            assert process.poll() is None, process.stderr.read()
            assert time.monotonic() < deadline
            time.sleep(0.2)


def test_ctrl_c_while_serving_stops_the_workers_and_exits_zero(tmp_path):
    server, process = _server(tmp_path, stdout=subprocess.PIPE)
    _wait_healthy(server, process)
    workers = _children(process.pid)
    assert workers
    os.killpg(process.pid, signal.SIGINT)
    stdout, stderr = process.communicate(timeout=120)
    assert process.returncode == 0
    assert stdout == ""
    _assert_quiet(stderr)
    _assert_exited(workers)


def test_ctrl_c_while_the_workers_start_stops_them_and_exits_zero(tmp_path):
    _, process = _server(tmp_path, stdout=subprocess.PIPE)
    deadline = time.monotonic() + 300
    while not (workers := _children(process.pid)):
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    os.killpg(process.pid, signal.SIGINT)
    stdout, stderr = process.communicate(timeout=120)
    assert process.returncode == 0
    assert stdout == ""
    _assert_quiet(stderr)
    _assert_exited(workers)


def _drain(fd):
    """Reads a pty's output as it comes, until the last writer closes it.

    A terminal is read while the program runs; macOS discards what is left
    unread on the pty once its last writer exits, where Linux keeps it.
    """
    shown = bytearray()

    def read():
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                return
            if not chunk:
                return
            shown.extend(chunk)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    return reader, shown


def test_a_terminal_sees_one_stopped_line(tmp_path):
    parent, child = pty.openpty()
    try:
        server, process = _server(tmp_path, stdout=child)
    finally:
        os.close(child)
    reader, shown = _drain(parent)
    _wait_healthy(server, process)
    os.killpg(process.pid, signal.SIGINT)
    assert process.wait(120) == 0
    reader.join(30)
    os.close(parent)
    lines = bytes(shown).decode().replace("\r\n", "\n").splitlines()
    assert len(lines) == 1, lines
    assert lines[0].startswith("\x1b[1;32m     Stopped\x1b[0m after "), lines
    _assert_quiet(process.stderr.read())


def test_a_server_stops_when_the_process_that_started_it_is_killed(tmp_path):
    """No exit handler runs under SIGKILL; the server sees its parent's pipe close."""
    from tests.utils.server import build

    code = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from tests.utils.server import Server
        server = Server(Path({str(build())!r}), Path({str(tmp_path)!r})).start()
        Path({str(tmp_path / "pid")!r}).write_text(str(server.process.pid))
        time.sleep(600)
        """
    )
    log = tmp_path / "parent.log"
    with log.open("w") as written:
        parent = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=written,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
    try:
        deadline = time.monotonic() + 600
        while not (tmp_path / "pid").exists():
            assert parent.poll() is None, log.read_text()
            assert time.monotonic() < deadline
            time.sleep(0.2)
        server = int((tmp_path / "pid").read_text())
        workers = _children(server)
        assert workers
        parent.kill()
        parent.wait(30)
        _assert_exited([server, *workers], seconds=120)
    finally:
        parent.kill()
