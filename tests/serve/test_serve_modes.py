# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`gf serve`'s modes: which server each becomes, and with what arguments."""

import argparse
import os
import sys
from pathlib import Path

import pytest

from generalforecast.deployments.contracts import ServiceError
from generalforecast.serve import listener


class Became(Exception):
    def __init__(self, program, arguments, environment):
        super().__init__(arguments)
        self.program, self.arguments, self.environment = program, arguments, environment


@pytest.fixture
def become(monkeypatch):
    monkeypatch.setattr(listener, "binary", lambda name="gf-server": Path("/x") / name)

    def execve(program, arguments, environment):
        raise Became(program, arguments[1:], environment)

    monkeypatch.setattr(listener.os, "execve", execve)


def parse(*argv):
    parser = argparse.ArgumentParser()
    listener.add_arguments(parser)
    return parser.parse_args(argv)


def became(*argv):
    with pytest.raises(Became) as caught:
        listener.run(parse(*argv))
    return Path(caught.value.program).name, caught.value.arguments


def test_a_config_file_runs_the_server_it_configures(become, tmp_path):
    node, router = tmp_path / "gf.toml", tmp_path / "router.toml"
    node.write_text('listen = "127.0.0.1:8815"\n')
    router.write_text('listen = "0.0.0.0:8815"\n[[nodes]]\nname = "a"\n')

    assert became("--config", str(node)) == (
        "gf-server",
        ["run", "--config", str(node)],
    )
    assert became("--config", str(router)) == (
        "gf-router",
        ["run", "--config", str(router)],
    )
    assert became("--config", str(router), "--check") == (
        "gf-router",
        ["check", "--config", str(router)],
    )


def test_config_files_are_read_before_any_server_starts(become, tmp_path):
    with pytest.raises(ServiceError, match="does not exist"):
        listener.run(parse("--config", str(tmp_path / "missing.toml")))
    (tmp_path / "bad.toml").write_text("listen = \n")
    with pytest.raises(ServiceError, match="not valid TOML"):
        listener.run(parse("--config", str(tmp_path / "bad.toml")))


def test_init_writes_a_node_or_a_router_starter(become, tmp_path):
    path = str(tmp_path / "out.toml")
    assert became("--init", path) == (
        "gf-server",
        [
            "init",
            "--path",
            path,
            "--listen",
            "127.0.0.1:8815",
            "--python",
            sys.executable,
        ],
    )
    assert became(
        "--init",
        path,
        "--node",
        "http://a:8815",
        "--node",
        "http://b:8815",
        "--host",
        "::",
        "--port",
        "9000",
        "--force",
    ) == (
        "gf-router",
        [
            "init",
            "--path",
            path,
            "--listen",
            "[::]:9000",
            "--node",
            "http://a:8815",
            "--node",
            "http://b:8815",
            "--force",
        ],
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["model", "--config", "gf.toml"],
        ["--config", "gf.toml", "--init", "x.toml"],
        ["--config", "gf.toml", "--node", "http://a:8815"],
        ["--init", "x.toml", "--check"],
        ["--force"],
        ["model", "--node", "http://a:8815"],
        ["--node", "http://a:8815", "--upstream", "http://127.0.0.1:8001"],
    ],
)
def test_modes_that_do_not_combine_are_refused(become, argv):
    with pytest.raises(ServiceError):
        listener.run(parse(*argv))


def test_a_router_takes_both_tokens_from_the_environment(become, monkeypatch):
    monkeypatch.delenv("GF_TOKEN", raising=False)
    monkeypatch.setenv("GF_REPLICATION_TOKEN", "r")
    with pytest.raises(ServiceError, match="set GF_TOKEN"):
        listener.run(parse("--node", "http://a:8815"))


def test_the_servers_inside_the_package_come_first(monkeypatch, tmp_path):
    for name in ("gf-server", "gf-router"):
        (tmp_path / name).write_text("")
    monkeypatch.setattr(listener, "BIN", tmp_path)
    assert listener.binary() == tmp_path / "gf-server"
    assert listener.binary("gf-router") == tmp_path / "gf-router"
    monkeypatch.setattr(listener, "BIN", tmp_path / "none")
    monkeypatch.setattr(
        listener.sysconfig, "get_path", lambda _: str(tmp_path / "none")
    )
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ServiceError, match="gf-router is not installed"):
        listener.binary("gf-router")


@pytest.mark.router
def test_nodes_become_a_router_over_them(monkeypatch, tmp_path):
    from tests.utils.server import build_router

    router = build_router()
    monkeypatch.setattr(listener, "BIN", router.parent)
    monkeypatch.setenv("GF_TOKEN", "client-token")
    monkeypatch.setenv("GF_REPLICATION_TOKEN", "replication-token")
    monkeypatch.setattr(listener.tempfile, "mkdtemp", lambda prefix: str(tmp_path))

    def execve(program, arguments, environment):
        raise Became(program, arguments[1:], environment)

    monkeypatch.setattr(listener.os, "execve", execve)
    with pytest.raises(Became) as caught:
        listener.run(
            parse("--node", "http://10.0.0.2:8815", "--node", "http://10.0.0.3:8815")
        )
    config = tmp_path / "router.toml"
    assert Path(caught.value.program) == router
    assert caught.value.arguments == ["run", "--config", str(config)]
    text = config.read_text()
    assert 'listen = "127.0.0.1:8815"' in text
    assert 'token = "client-token"' in text and 'token = "replication-token"' in text
    assert text.count("[[nodes]]") == 2
    assert os.stat(config).st_mode & 0o777 == 0o600
