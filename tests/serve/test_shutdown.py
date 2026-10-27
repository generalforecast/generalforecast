# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ctrl-C: the workers leave it to the server, which stops them all without a word."""

import os
import pty
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from generalforecast import protocol

ROOT = Path(__file__).resolve().parents[2]


def _children(pid):
    listed = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False
    )
    return [int(child) for child in listed.stdout.split()]


def _assert_exited(pids):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not any(Path(f"/proc/{pid}").exists() for pid in pids):
            return
        time.sleep(0.1)
    raise AssertionError(f"workers still running: {pids}")


def _assert_quiet(stderr):
    """No traceback anywhere, and no warning or error once the stop began."""
    assert "Traceback" not in stderr and "KeyboardInterrupt" not in stderr, stderr
    lines = stderr.splitlines()
    begun = next((i for i, line in enumerate(lines) if "draining" in line), len(lines))
    noisy = [line for line in lines[begun:] if "WARN" in line or "ERROR" in line]
    assert not noisy, stderr


def test_a_worker_ignores_sigint_and_exits_when_its_socket_closes(tmp_path):
    path = tmp_path / "worker.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        listener.listen(1)
        worker = subprocess.Popen(
            [sys.executable, "-m", "generalforecast.worker_process", "--socket", path],
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
    process = subprocess.Popen(
        [str(server.binary), "run", "--config", str(server.config)],
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
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


def test_a_terminal_sees_one_stopped_line(tmp_path):
    parent, child = pty.openpty()
    try:
        server, process = _server(tmp_path, stdout=child)
    finally:
        os.close(child)
    _wait_healthy(server, process)
    os.killpg(process.pid, signal.SIGINT)
    assert process.wait(120) == 0
    shown = b""
    while select.select([parent], [], [], 1)[0]:
        try:
            chunk = os.read(parent, 4096)
        except OSError:
            break
        if not chunk:
            break
        shown += chunk
    os.close(parent)
    lines = shown.decode().replace("\r\n", "\n").splitlines()
    assert len(lines) == 1, lines
    assert lines[0].startswith("\x1b[1;32m     Stopped\x1b[0m after "), lines
    _assert_quiet(process.stderr.read())
