# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How `gf serve`, `gf admin node` and `gf admin router` start their servers."""

import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import urllib.request
from pathlib import Path

import pytest

from generalforecast import cli
from generalforecast.serve import listener

ROOT = Path(__file__).resolve().parents[2]


def gf(*argv, monkeypatch):
    """Runs `gf argv` in process; returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("NO_COLOR", "1")
    try:
        cli.main(list(argv))
    except SystemExit as error:
        return error.code or 0, out.getvalue(), err.getvalue()
    return 0, out.getvalue(), err.getvalue()


def _fake(directory: Path, name: str, script: str) -> Path:
    """An executable `name` that records its arguments and runs `script`."""
    path = directory / name
    path.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{directory}/{name}.args"\n' + textwrap.dedent(script)
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def private(monkeypatch, tmp_path):
    """Private directories under a temporary directory of the test's own."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def test_a_missing_server_is_reported_before_any_weights(monkeypatch, private):
    def served(*_args, **_kwargs):
        raise AssertionError("weights were checked before the server was found")

    monkeypatch.setattr("generalforecast.serve.models.served", served)
    monkeypatch.setattr(listener, "BIN", private / "absent")
    monkeypatch.setattr(listener.shutil, "which", lambda name: None)
    monkeypatch.setattr(listener.sysconfig, "get_path", lambda name: str(private))
    monkeypatch.delenv("GF_SERVER_BIN", raising=False)
    code, stdout, stderr = gf(
        "--json",
        "serve",
        "general-forecast/tabicl-v2-classifier",
        monkeypatch=monkeypatch,
    )
    assert code == 1 and not stdout
    error = json.loads(stderr)["error"]
    assert error["message"] == "gf-server is not installed with gf"
    assert error["reason"] == "configuration_error"
    assert not list(private.iterdir())


def test_a_failure_before_the_server_starts_removes_its_directory(
    monkeypatch, private, tmp_path
):
    from generalforecast.errors import NotFound

    made = []

    def served(target, directory, **_):
        made.append(directory)
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        (directory / "context.arrow").write_bytes(b"rows")
        raise NotFound(f"no predictor at {target}")

    monkeypatch.setattr("generalforecast.serve.models.served", served)
    monkeypatch.setattr(listener, "binary", lambda name: tmp_path / name)
    predictor = tmp_path / "unlocked"
    predictor.mkdir()
    code, _, stderr = gf("serve", str(predictor), monkeypatch=monkeypatch)
    assert code == 1 and stderr == f"error: no predictor at {predictor}\n"
    assert made and not made[0].exists()


def test_the_directory_goes_once_the_process_it_became_exits(tmp_path):
    """The directory survives `exec` and goes when the server and its children exit."""
    report = tmp_path / "directory"
    code = textwrap.dedent(
        f"""
        import os, sys
        from generalforecast.serve.listener import private_directory
        directory, _ = private_directory()
        (directory / "generalforecast.toml").write_text("token")
        open({str(report)!r}, "w").write(str(directory))
        os.execv("/bin/sh", ["sh", "-c", "sleep 0.5"])
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, timeout=60)
    directory = Path(report.read_text())
    deadline = time.monotonic() + 30
    while directory.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not directory.exists()


def test_the_servers_refuse_where_they_cannot_run(monkeypatch):
    monkeypatch.setattr(listener, "POSIX", False)
    for argv in (("serve",), ("admin", "node", "serve", "--config", "x.toml")):
        code, stdout, stderr = gf("--json", *argv, monkeypatch=monkeypatch)
        assert code == 1 and not stdout
        error = json.loads(stderr)["error"]
        assert error["reason"] == "unsupported_operation"
        assert error["message"] == "the servers run on Linux and macOS only"


def test_node_init_writes_through_gf_server_and_names_the_next_steps(
    monkeypatch, tmp_path
):
    server = _fake(tmp_path, "gf-server", "exit 0\n")
    monkeypatch.setattr(listener, "binary", lambda name: server)
    monkeypatch.delenv("GF_LOG_KEY", raising=False)
    config = tmp_path / "node.toml"
    code, stdout, stderr = gf(
        "admin",
        "node",
        "init",
        "--path",
        str(config),
        "--port",
        "8816",
        "--model",
        "tabicl=general-forecast/tabicl-v2-classifier",
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    arguments = (tmp_path / "gf-server.args").read_text().split("\n")
    assert arguments[:5] == [
        "init",
        "--path",
        str(config),
        "--listen",
        "127.0.0.1:8816",
    ]
    assert arguments[5:9] == ["--data", "gf-data", "--python", sys.executable]
    assert arguments[9:11] == [
        "--model",
        "tabicl=general-forecast/tabicl-v2-classifier",
    ]
    assert stderr == f"       Wrote {config}, which holds the client token\n"
    assert stdout.splitlines() == [
        "next: export GF_LOG_KEY=$(openssl rand -hex 32)",
        f"      gf admin node check --config {config}",
        f"      gf admin node serve --config {config}",
    ]
    config.write_text("")
    code, _, stderr = gf(
        "admin", "node", "init", "--path", str(config), monkeypatch=monkeypatch
    )
    assert code == 1 and stderr.startswith(f"error: `{config}` already exists")


def test_router_check_reports_and_fails_with_the_routers_words(monkeypatch, tmp_path):
    router = _fake(
        tmp_path,
        "gf-router",
        """
        echo "ok     node a: http://127.0.0.1:8816"
        echo "failed node b: http://127.0.0.1:8817: connection refused"
        exit 1
        """,
    )
    monkeypatch.setattr(listener, "binary", lambda name: router)
    config = tmp_path / "router.toml"
    config.write_text("")
    code, stdout, stderr = gf(
        "--json",
        "admin",
        "router",
        "check",
        "--config",
        str(config),
        monkeypatch=monkeypatch,
    )
    assert code == 1 and not stdout
    error = json.loads(stderr)["error"]
    assert error["message"] == "gf-router check failed"
    passing = _fake(tmp_path, "gf-router", 'echo "ok     node a"\n')
    monkeypatch.setattr(listener, "binary", lambda name: passing)
    code, stdout, _ = gf(
        "admin", "router", "check", "--config", str(config), monkeypatch=monkeypatch
    )
    assert code == 0 and stdout == "ok     node a\n"
    assert (tmp_path / "gf-router.args").read_text().split() == [
        "check",
        "--config",
        str(config),
    ]


def test_serving_a_node_or_router_becomes_its_server(monkeypatch, tmp_path):
    monkeypatch.delenv("GF_OFFLINE", raising=False)
    started = []
    monkeypatch.setattr(listener, "binary", lambda name: tmp_path / name)
    monkeypatch.setattr(
        listener.os, "execve", lambda *call: started.append(call) or sys.exit(0)
    )
    config = tmp_path / "cluster.toml"
    code, _, stderr = gf(
        "admin", "node", "serve", "--config", str(config), monkeypatch=monkeypatch
    )
    assert code == 1 and stderr.startswith(f"error: no configuration at `{config}`")
    config.write_text("")
    gf(
        "admin",
        "node",
        "serve",
        "--config",
        str(config),
        "--offline",
        monkeypatch=monkeypatch,
    )
    gf("admin", "router", "serve", "--config", str(config), monkeypatch=monkeypatch)
    (node, node_args, node_env), (router, router_args, router_env) = started
    assert Path(node).name == "gf-server" and Path(router).name == "gf-router"
    assert node_args[1:] == router_args[1:] == ["run", "--config", str(config)]
    assert node_env["HF_HUB_OFFLINE"] == node_env["GF_OFFLINE"] == "1"
    assert "GF_OFFLINE" not in router_env


def test_metrics_are_configured_only_when_asked_for(tmp_path):
    model = {"worker": "tabicl", "reference": "general-forecast/tabicl-v2-classifier"}
    plain = tomllib.loads(
        listener.configuration(tmp_path, "127.0.0.1:8000", 30.0, [model])
    )
    assert "metrics_listen" not in plain
    config = tomllib.loads(
        listener.configuration(
            tmp_path, "127.0.0.1:8000", 30.0, [model], metrics="127.0.0.1:9090"
        )
    )
    # A server setting, not one of the Models API's under [http].
    assert config["metrics_listen"] == "127.0.0.1:9090"
    assert "metrics_listen" not in config["http"]
    assert {
        k: v for k, v in config.items() if k not in ("metrics_listen", "tokens")
    } == {k: v for k, v in plain.items() if k != "tokens"}


@pytest.mark.slow
@pytest.mark.server
def test_a_server_configured_with_metrics_serves_them_there():
    from tests.conformance.test_surfaces import Served, _port
    from tests.utils.server import build

    metrics = f"127.0.0.1:{_port()}"
    served = Served(build(), "general-forecast/tabicl-v2-classifier", metrics=metrics)
    try:
        for path, expected in (("/healthz", "ok"), ("/metrics", "gf_")):
            with urllib.request.urlopen(f"http://{metrics}{path}", timeout=30) as reply:
                assert reply.status == 200
                assert expected in reply.read().decode()
    finally:
        served.stop()


def test_offline_shapes_the_servers_environment(monkeypatch):
    import argparse

    monkeypatch.delenv("GF_OFFLINE", raising=False)
    flags = argparse.Namespace(verbose=False, quiet=False, json=False)
    assert "GF_OFFLINE" not in listener._environment(flags)
    offline = listener._environment(flags, offline=True)
    assert offline["GF_OFFLINE"] == offline["HF_HUB_OFFLINE"] == "1"


@pytest.mark.slow
@pytest.mark.server
def test_a_real_node_configuration_writes_and_checks(tmp_path, monkeypatch):
    from tests.utils.server import build

    server = build()
    monkeypatch.setattr(listener, "binary", lambda name: server)
    monkeypatch.setenv("GF_LOG_KEY", "0" * 64)
    monkeypatch.chdir(tmp_path)
    code, stdout, stderr = gf("admin", "node", "init", monkeypatch=monkeypatch)
    assert code == 0, stderr
    config = tmp_path / "generalforecast.toml"
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert "gf admin node check --config generalforecast.toml" in stdout
    code, stdout, stderr = gf(
        "--json",
        "admin",
        "node",
        "check",
        "--config",
        str(config),
        monkeypatch=monkeypatch,
    )
    assert code == 0, stderr
    assert json.loads(stdout)["checks"]
    assert os.path.isfile(config)


KEYS = {"kind": "introspect", "url": "http://127.0.0.1:9", "token_env": "T"}
PROXIES = ("127.0.0.1/32", "10.0.0.0/8")
MODELS = [{"worker": "tabicl", "reference": "general-forecast/tabicl-v2-classifier"}]


def hosted(directory: Path, models: list[dict]) -> str:
    """gf-server's configuration as a hosted plane writes it for its front."""
    return listener.configuration(
        directory,
        "127.0.0.1:0",
        30.0,
        models,
        keys=KEYS,
        surface="remote",
        upstream="http://127.0.0.1:9",
        trusted_proxies=PROXIES,
        revisions="http://127.0.0.1:9",
        metrics="127.0.0.1:9090",
    )


def test_a_hosted_configuration_names_its_key_store_upstream_and_revisions(tmp_path):
    import tomllib

    plain = tomllib.loads(listener.configuration(tmp_path, "127.0.0.1:0", 30.0, MODELS))
    assert plain["http"]["keys"] == {"kind": "static", "env": listener.API_KEY_ENV}
    assert plain["http"]["surface"] == "self-hosted"
    assert not {"upstream", "trusted_proxies", "revisions"} & plain["http"].keys()
    assert "metrics_listen" not in plain

    config = tomllib.loads(hosted(tmp_path, MODELS))
    assert config["http"]["keys"] == KEYS
    assert config["http"]["surface"] == "remote"
    assert config["http"]["upstream"] == "http://127.0.0.1:9"
    assert config["http"]["trusted_proxies"] == list(PROXIES)
    assert config["http"]["revisions"] == {
        "url": "http://127.0.0.1:9",
        "token_env": "GF_INTROSPECTION_TOKEN",
    }
    assert config["metrics_listen"] == "127.0.0.1:9090"


@pytest.mark.slow
@pytest.mark.server
def test_gf_server_checks_a_hosted_configuration(tmp_path):
    from tests.utils.server import build

    (tmp_path / "models.json").write_text("[]")
    config = tmp_path / "server.toml"
    config.write_text(hosted(tmp_path, []))
    config.chmod(0o600)
    done = subprocess.run(
        [str(build()), "check", "--config", str(config)],
        env=dict(os.environ, GF_LOG_KEY="0" * 64, T="t" * 32),
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_server_is_found_in_its_documented_order(monkeypatch, tmp_path):
    def program(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
        return path

    checkout = tmp_path / "checkout"
    (checkout / "crates" / "gf-server" / "src").mkdir(parents=True)
    source = checkout / "crates" / "gf-server" / "src" / "main.rs"
    source.write_text("")
    (checkout / "Cargo.lock").write_text("")
    staged = program(tmp_path / "bin" / "gf-server")
    on_path = program(tmp_path / "path" / "gf-server")
    monkeypatch.setattr(listener, "BIN", staged.parent)
    monkeypatch.setattr(listener, "_checkout", lambda: checkout)
    monkeypatch.setattr(listener.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.delenv("GF_SERVER_BIN", raising=False)

    os.utime(source, (time.time() - 60, time.time() - 60))
    assert listener.binary() == staged
    # A source edited after the staging makes the staged server stale.
    os.utime(source, None)
    os.utime(staged, (time.time() - 60, time.time() - 60))
    assert listener.binary() == on_path
    explicit = program(tmp_path / "explicit" / "gf-server")
    monkeypatch.setenv("GF_SERVER_BIN", str(explicit))
    assert listener.binary() == explicit
    monkeypatch.setenv("GF_SERVER_BIN", str(tmp_path / "missing"))
    with pytest.raises(Exception, match="GF_SERVER_BIN names no program"):
        listener.binary()
    monkeypatch.delenv("GF_SERVER_BIN")
    monkeypatch.setattr(listener, "_checkout", lambda: None)
    assert listener.binary() == staged
