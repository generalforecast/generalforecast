"""The generalforecast command: a server on this machine, and a benchmark of one."""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.utils import legacy_cli as cli

pytestmark = pytest.mark.server


@pytest.fixture(scope="module")
def binary() -> Path:
    from tests.utils.server import build

    return build()


@pytest.mark.slow
def test_serve_keeps_its_configuration_in_the_data_and_its_key_apart(
    binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = []
    monkeypatch.setattr(cli, "binary", lambda name: binary)
    monkeypatch.setattr(os, "execve", lambda *call: started.append(call))
    monkeypatch.delenv("GF_LOG_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    data = tmp_path / "data"
    key = tmp_path / "config" / "generalforecast" / "key"

    cli.main(["serve", "--data", str(data), "--listen", "127.0.0.1:1"])

    (server, arguments, environment) = started[0]
    config = tomllib.loads((data / "generalforecast.toml").read_text())
    assert Path(server).name == "gf-server"
    assert arguments[1:] == ["run", "--config", str(data / "generalforecast.toml")]
    assert config["listen"] == "127.0.0.1:1"
    assert config["tokens"] == [(data / "token").read_text()]
    assert config["workers"][0]["command"][0] == sys.executable
    # The server reads its key from a file only this user reads, not the environment.
    assert config["encryption_key_file"] == str(key)
    assert "GF_LOG_KEY" not in environment
    assert key.stat().st_mode & 0o777 == 0o600
    assert not (data / "key").exists()
    # A second start keeps the same token and key, and the configuration as
    # edited, whatever the flags say.
    first = key.read_text()
    edited = (
        (data / "generalforecast.toml")
        .read_text()
        .replace("127.0.0.1:1", "127.0.0.1:2")
    )
    (data / "generalforecast.toml").write_text(edited)
    cli.main(["serve", "--data", str(data)])
    assert key.read_text() == first
    assert (data / "generalforecast.toml").read_text() == edited


@pytest.fixture
def server(binary: Path, tmp_path: Path) -> Iterator[object]:
    from tests.utils.server import Server

    running = Server(binary, tmp_path).start()
    yield running
    running.stop()


@pytest.mark.slow
def test_bench_measures_a_server_and_leaves_no_table(
    server, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.utils import legacy_client as generalforecast

    cli.main(
        [
            "bench",
            f"--url=grpc://127.0.0.1:{server.port}",
            "--token=secret",
            "--model=tiny",
            "--rows=64",
            "--queries=16",
            "--features=5",
            "--classes=3",
            "--batch=8",
            "--concurrency=2",
        ]
    )

    report = capsys.readouterr().out
    assert "64 rows" in report and "16 queries" in report and "p95" in report
    client = generalforecast.connect(f"grpc://127.0.0.1:{server.port}", token="secret")
    assert client.tables() == []
