# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Installed CLI checks; HTTP fixture below proves contracts, not model inference."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from generalforecast.cli import parser, redact, run
from generalforecast.contracts import DevSession, Environment, PatchRequest, Revision
from generalforecast.deployments.boundary import Request, success
from generalforecast.deployments.contracts import ServiceError
from generalforecast.packages import Package, PackageError, PackageLock, PackageSpec
from generalforecast.tabular import read_table, table_digest


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "fraud"
    run(
        [
            "init",
            str(root),
            "--model",
            "tabicl-classifier",
            "--target",
            "y",
            "--context",
            "history.csv",
        ]
    )
    (root / "history.csv").write_text("x,y\n1,false\n2,true\n3,false\n4,true\n")
    Package.load(root).check()
    return root


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    from generalforecast.commands import _CREDENTIAL_VALUES

    _CREDENTIAL_VALUES.clear()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for key in list(os.environ):
        if key.startswith("GF_") and key != "GF_CLI":
            monkeypatch.delenv(key)


def test_offline_checks_are_socket_free_and_strict(package, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline CLI opened a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    first = run(["check", str(package), "--offline"])
    assert run(["check", str(package), "--locked"]) == first
    assert run(["check", str(package), "--frozen"]) == first
    original = (package / "gf.lock").read_bytes()
    (package / "history.csv").write_text("x,y\n8,false\n9,true\n")
    with pytest.raises(ValueError):
        run(["check", str(package), "--frozen"])
    assert (package / "gf.lock").read_bytes() == original
    with pytest.raises(ValueError):
        run(["check", str(package), "--online", "--locked"])


def test_init_refuses_overwrite_and_uses_project_relative_csv(package):
    spec = Package.load(package).spec
    assert isinstance(spec, PackageSpec)
    assert spec.context.source == "history.csv"
    with pytest.raises(ServiceError):
        run(["init", str(package), "--model", "tabicl-classifier"])
    assert (package / "history.csv").is_file()
    assert {path.name for path in package.iterdir()} == {
        "history.csv",
        "gf.yaml",
        "gf.lock",
    }


def test_parser_and_secret_presentation(monkeypatch):
    args = parser().parse_args(
        ["--project", "demo", "push", "--output", "json", "--watch"]
    )
    assert args.project == "demo" and args.output == "json" and args.watch
    monkeypatch.setenv("GF_API_KEY", "private-runtime-value")
    assert redact(
        {"token": "hidden", "nested": ["private-runtime-value"], "value": 1}
    ) == {"token": "[redacted]", "nested": ["[redacted]"], "value": 1}


@pytest.fixture
def http_contract(monkeypatch):
    state = SimpleNamespace(
        revisions={},
        locks={},
        environments={},
        sessions={},
        snapshots={},
        snapshot_kind="dataset",
        calls=[],
        pending=False,
        conflict=False,
        reject=False,
        drop_patch=False,
        stopped=False,
    )
    state.token = "contract-only-key"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            try:
                if self.headers.get("Authorization") != "Bearer " + state.token:
                    raise ServiceError(
                        "Unauthorized", "credential-redaction-canary", 401
                    )
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                state.calls.append((self.command, self.path))
                tail = self.path.removeprefix("/v1alpha1/").removeprefix(
                    "projects/demo/"
                )
                if tail == "snapshots":
                    table = read_table(raw, "arrow")
                    digest = table_digest(table)
                    state.snapshots[digest] = table
                    request = Request.parse(
                        {
                            "api_version": "gf/v1alpha1",
                            "project": "demo",
                            "request_id": self.headers["X-GF-Request-ID"],
                            "resource": {
                                "kind": state.snapshot_kind,
                                "name": "snapshots",
                            },
                            "payload": {},
                        }
                    )
                    result = {"digest": digest, "rows": table.num_rows}
                else:
                    if raw:
                        request = Request.parse(json.loads(raw))
                        body = request.payload
                    else:
                        request = Request.parse(
                            {
                                "api_version": "gf/v1alpha1",
                                "project": "demo",
                                "request_id": self.headers["X-GF-Request-ID"],
                                "resource": {
                                    "kind": "workspace"
                                    if tail == "auth/status"
                                    else "deployment",
                                    "name": "auth"
                                    if tail == "auth/status"
                                    else "fraud",
                                },
                                "payload": {},
                            }
                        )
                        body = {}
                    result = self.dispatch(tail, body)
                encoded = json.dumps(success(request, result)).encode()
                self.send_response(200)
            except ServiceError as error:
                encoded = json.dumps(error.body()).encode()
                self.send_response(error.status)
            except ValueError:
                encoded = (
                    b'{"error":{"code":"InvalidPackage","message":"invalid patch"}}'
                )
                self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def dispatch(self, tail, body):
            if tail == "auth/status":
                return {"authenticated": True}
            if tail == "deployments":
                state.created = True
                return {"name": body["name"]}
            parts = tail.split("/")
            if parts == ["deployments", "fraud"]:
                if not getattr(state, "created", False):
                    raise ServiceError("NotFound", "not found", 404)
                return {"name": "fraud"}
            if parts[2] == "revisions":
                if self.command == "POST":
                    lock = PackageLock.from_dict(body["lock"])
                    assert body["snapshot"] == lock.data_digest
                    revision = Revision(
                        "fraud",
                        len(state.revisions) + 1,
                        lock.package_digest,
                        lock.identity,
                        time.time(),
                        "ready",
                    )
                    state.revisions[revision.number] = revision
                    state.locks[revision.number] = lock
                    return revision.to_dict()
                revision = state.revisions[int(parts[3])]
                return {
                    "revision": revision.to_dict(),
                    "lock": state.locks[revision.number].to_dict(),
                }
            if parts[2] == "environments":
                if len(parts) == 3:
                    env = Environment("fraud", body["name"], None, 0, body["runtime"])
                    state.environments[env.name] = env
                    return env.to_dict()
                if parts[3] not in state.environments:
                    raise ServiceError("NotFound", "not found", 404)
                env = state.environments[parts[3]]
                if len(parts) == 5:
                    if state.conflict or body["expected_generation"] != env.generation:
                        raise ServiceError("Conflict", "stale", 409)
                    env = replace(
                        env,
                        revision=body["revision"],
                        generation=env.generation + 1,
                        status="pending" if state.pending else "ready",
                    )
                    state.environments[env.name] = env
                return env.to_dict()
            if parts[2] == "dev-sessions":
                if len(parts) == 3:
                    session = DevSession(
                        "dev-one",
                        "fraud",
                        body["base_revision"],
                        0,
                        state.locks[body["base_revision"]].package_digest,
                        time.time() + 3600,
                        "ready",
                    )
                    state.sessions[session.id] = session
                    return session.to_dict()
                session = state.sessions[parts[3]]
                if len(parts) == 5 and parts[4] == "patch":
                    if state.reject:
                        raise ServiceError(
                            "UnsupportedCapability", "runtime rejected candidate", 422
                        )
                    request = PatchRequest.from_dict(body)
                    session = replace(
                        session.apply_patch(request, now=time.time()), status="ready"
                    )
                    state.sessions[session.id] = session
                    if state.drop_patch:
                        state.drop_patch = False
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                elif len(parts) == 5 and parts[4] == "stop":
                    session = replace(session, status="stopped")
                    state.sessions[session.id] = session
                    state.stopped = True
                return session.to_dict()
            raise ServiceError("NotFound", "unknown route", 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("GF_URL", state.url)
    monkeypatch.setenv("GF_PROJECT", "demo")
    monkeypatch.setenv("GF_API_KEY", state.token)
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def test_login_stores_only_reference_and_rejects_profile_symlink(
    http_contract, tmp_path, monkeypatch
):
    assert run(["login", "--profile", "work"])["method"] == "configured-key"
    path = tmp_path / "config/gf/profiles.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert http_contract.token not in path.read_text()
    assert json.loads(path.read_text())["work"]["key_env"] == "GF_API_KEY"
    assert run(["auth", "status", "--profile", "work"])["authenticated"]
    path.rename(path.with_suffix(".backup"))
    path.symlink_to(path.with_suffix(".backup"))
    with pytest.raises(ServiceError):
        run(["auth", "status", "--profile", "work"])


def test_push_uploads_full_snapshot_and_awaits_exact_readiness(package, http_contract):
    spec = yaml.safe_load((package / "gf.yaml").read_text())
    spec["context"] = {
        "source": "history.csv",
        "selection": "most_recent",
        "order_by": "x",
        "max_rows": 2,
    }
    (package / "gf.yaml").write_text(yaml.safe_dump(spec))
    result = run(["push", str(package), "--env", "production"])
    assert result["environment"]["status"] == "ready"
    assert next(iter(http_contract.snapshots.values())).num_rows == 4
    assert http_contract.locks[1].context_rows == 2
    http_contract.conflict = True
    with pytest.raises(ServiceError) as error:
        run(["promote", "fraud", "--revision", "1"])
    assert error.value.code == "Conflict"


def test_push_timeout_is_nonzero(package, http_contract):
    http_contract.pending = True
    with pytest.raises(ServiceError) as error:
        run(["push", str(package), "--timeout", "0.02"])
    assert error.value.code == "Timeout"


def test_snapshot_resource_kind_mismatch_fails_closed(package, http_contract):
    http_contract.snapshot_kind = "package"
    with pytest.raises(ServiceError) as error:
        run(["push", str(package)])
    assert error.value.code == "ProtocolError"
    assert not http_contract.revisions


def test_watch_patch_and_invalid_edit_preserve_production(
    package, http_contract, capsys
):
    from generalforecast.package_commands import PackageCommands, _Watcher

    commands = PackageCommands(parser().parse_args(["push", str(package), "--watch"]))
    checked = commands.checked()
    revision, env = commands.push(checked)
    session = DevSession.from_dict(
        commands.call(
            "POST",
            "fraud",
            "/dev-sessions",
            {"base_revision": revision.number, "runtime": "local", "ttl_seconds": 3600},
        )
    )
    watcher = _Watcher(commands, checked, session)
    watcher.tick()
    original = (package / "gf.yaml").read_text()
    (package / "gf.yaml").write_text(original + "outputs: [label]\n")
    http_contract.drop_patch = True
    watcher.tick()
    assert watcher.session.generation == 1
    assert watcher.session.package_digest != revision.package_digest
    assert http_contract.environments[env.name] == env
    assert len(http_contract.revisions) == 1
    (package / "gf.yaml").write_text("model: [invalid\n")
    watcher.tick()
    assert http_contract.sessions[session.id] == watcher.session
    assert (
        json.loads(capsys.readouterr().err.splitlines()[-1])["event"]
        == "patch_rejected"
    )
    (package / "gf.yaml").write_text(original + "outputs: [probabilities]\n")
    http_contract.reject = True
    watcher.tick()
    assert http_contract.sessions[session.id].generation == 1
    assert http_contract.environments[env.name] == env


def test_installed_watch_ctrl_c_stops_only_dev_session(package, http_contract):
    process = subprocess.Popen(
        [
            str(Path(sys.executable).with_name("gf")),
            "push",
            str(package),
            "--watch",
            "--interval",
            "0.05",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        event = json.loads(process.stdout.readline())
        assert event["event"] == "watching"
        process.send_signal(signal.SIGINT)
        out, err = process.communicate(timeout=15)
        assert process.returncode == 130, err
        assert any(
            json.loads(line).get("event") == "stopped" for line in out.splitlines()
        )
        assert http_contract.stopped
        assert http_contract.environments["staging"].status == "ready"
        assert http_contract.token not in out + err
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_watch_resume_never_republishes(package, http_contract):
    result = run(
        [
            "push",
            str(package),
            "--watch",
            "--watch-seconds",
            "0.05",
            "--interval",
            "0.05",
        ]
    )
    session = http_contract.sessions[result["session"]["id"]]
    http_contract.sessions[session.id] = replace(session, status="ready")
    count = len(http_contract.calls)
    run(
        [
            "push",
            str(package),
            "--watch",
            "--resume",
            session.id,
            "--watch-seconds",
            "0.05",
            "--interval",
            "0.05",
        ]
    )
    calls = http_contract.calls[count:]
    assert not any(path.endswith(("/revisions", "/promote")) for _, path in calls)


@pytest.mark.parametrize("action", ["check", "help", "invalid"])
def test_rust_launcher_matches_installed_python(package, action):
    binary = Path(
        os.environ.get("GF_CLI", str(Path(__file__).parents[2] / "target/debug/gf"))
    ).absolute()
    if not binary.is_file():
        if "GF_CLI" in os.environ:
            pytest.fail("explicit GF_CLI binary is missing")
        pytest.skip("build cargo build -p gf-cli --bin gf for launcher parity")
    env = os.environ | {"GF_PYTHON": sys.executable}
    if action == "invalid":
        (package / "gf.yaml").write_text("model: [invalid\n")
    arguments = ["--help"] if action == "help" else ["check", str(package), "--frozen"]
    results = [
        subprocess.run(
            [*command, *arguments],
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        for command in ([sys.executable, "-m", "generalforecast"], [str(binary)])
    ]
    assert all(
        result.returncode == (1 if action == "invalid" else 0) for result in results
    ), [result.stderr for result in results]
    assert results[0].stdout == results[1].stdout
    assert results[0].stderr == results[1].stderr
    assert "Traceback" not in results[0].stderr


def test_cli_argument_errors_do_not_echo_credentials():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "generalforecast",
            "--token",
            "sensitive-unrecognized-argument",
        ],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]["code"] == "UsageError"
    assert "sensitive-unrecognized-argument" not in result.stderr


def test_local_dataset_import_uses_uploaded_arrow_identity(tmp_path, monkeypatch):
    from generalforecast import commands

    source = tmp_path / "data.csv"
    source.write_text("id,x,y\na,1.0,2.0\n")

    from generalforecast import Client

    class ImportClient(Client):
        def upload_snapshot(self, data):
            return {"digest": table_digest(data), "rows": data.num_rows}

        def _call(self, method, path, kind, resource, payload=None, **options):
            assert (method, path, kind, resource) == (
                "POST",
                "/datasets/observations/imports",
                "dataset",
                "observations",
            )
            return payload

    client = ImportClient("http://127.0.0.1:9", project="local", api_key="test-key")
    monkeypatch.setattr(commands, "client_for", lambda _: client)
    result = run(["datasets", "import", "observations", str(source), "--format", "csv"])
    assert result["source"] == "upload://" + result["digest"].replace(":", "-")
    assert result["format"] == "arrow"


@pytest.mark.parametrize(
    "option,value", [("--interval", "0"), ("--watch-seconds", "nan"), ("--ttl", "0")]
)
def test_invalid_watch_options_do_not_publish(package, http_contract, option, value):
    with pytest.raises(ServiceError):
        run(["push", str(package), "--watch", option, value])
    assert http_contract.calls == []


def _watcher(package):
    from generalforecast.package_commands import PackageCommands, _Watcher

    commands = PackageCommands(parser().parse_args(["push", str(package), "--watch"]))
    checked = commands.checked()
    revision, _ = commands.push(checked)
    session = DevSession.from_dict(
        commands.call(
            "POST",
            "fraud",
            "/dev-sessions",
            {
                "base_revision": revision.number,
                "runtime": "local",
                "ttl_seconds": 3600,
            },
        )
    )
    watcher = _Watcher(commands, checked, session)
    watcher.tick()
    return watcher


def test_watch_rechecks_edits_made_during_patch(package, http_contract, monkeypatch):
    watcher = _watcher(package)
    path = package / "gf.yaml"
    original = path.read_text()
    path.write_text(original + "outputs: [label]\n")
    call = watcher.commands.call

    def edit_during_patch(method, deployment, suffix="", payload=None):
        value = call(method, deployment, suffix, payload)
        if suffix.endswith("/patch") and value["generation"] == 1:
            path.write_text(original + "outputs: [probabilities]\n")
        return value

    monkeypatch.setattr(watcher.commands, "call", edit_during_patch)
    watcher.tick()
    watcher.tick()
    assert watcher.session.generation == 2
    assert watcher.package.lock.outputs == ("probabilities",)
    assert len(http_contract.revisions) == 1


@pytest.mark.parametrize("kind", ["symlink-parent", "fifo"])
def test_watch_rejects_nonregular_or_indirect_inputs(package, http_contract, kind):
    watcher = _watcher(package)
    if kind == "symlink-parent":
        (package / "linked").symlink_to(package.parent, target_is_directory=True)
        watcher.commands.args.data = "linked/history.csv"
    else:
        (package / "history.csv").unlink()
        os.mkfifo(package / "history.csv")
    with pytest.raises(PackageError) as error:
        watcher.signature()
    assert error.value.code == "UnsafePath"


def test_watch_recovers_transport_loss_while_waiting_for_patch(
    package, http_contract, monkeypatch
):
    watcher = _watcher(package)
    path = package / "gf.yaml"
    path.write_text(path.read_text() + "outputs: [label]\n")
    ready = watcher.ready
    failed = False

    def interrupted(expected):
        nonlocal failed
        if expected.generation == 1 and not failed:
            failed = True
            raise ServiceError("TransportError", "interrupted", 503)
        return ready(expected)

    monkeypatch.setattr(watcher, "ready", interrupted)
    watcher.tick()
    watcher.tick()
    assert watcher.session.generation == 1
    assert watcher.session.status == "ready"
    assert not http_contract.stopped
    assert watcher.package.lock.outputs == ("label",)
    assert Package.load(package).lock == watcher.package.lock


def test_watch_reconciles_lost_patch_and_failed_reconciliation_read(
    package, http_contract, monkeypatch
):
    watcher = _watcher(package)
    path = package / "gf.yaml"
    path.write_text(path.read_text() + "outputs: [label]\n")
    call = watcher.commands.call
    lost_patch = False
    lost_read = False

    def interrupted(method, deployment, suffix="", payload=None):
        nonlocal lost_patch, lost_read
        if lost_patch and not lost_read and method == "GET":
            lost_read = True
            raise ServiceError("TransportError", "unavailable", 503)
        value = call(method, deployment, suffix, payload)
        if suffix.endswith("/patch") and not lost_patch:
            lost_patch = True
            raise ServiceError("TransportError", "lost receipt", 503)
        return value

    monkeypatch.setattr(watcher.commands, "call", interrupted)
    watcher.tick()
    watcher.tick()
    assert watcher.session.generation == 1
    assert watcher.package.lock.outputs == ("label",)
    assert sum(path.endswith("/patch") for _, path in http_contract.calls) == 1


def test_watch_conflict_never_stops_a_newer_writer(package, http_contract):
    watcher = _watcher(package)
    http_contract.sessions[watcher.session.id] = replace(watcher.session, generation=1)
    with pytest.raises(ServiceError) as error:
        watcher.commands.watch(watcher.package, watcher.session)
    assert error.value.code == "Conflict"
    assert not http_contract.stopped


@pytest.mark.parametrize(
    "field,value",
    [("deployment", "other"), ("name", "other"), ("revision", 2), ("generation", 9)],
)
def test_push_rejects_mismatched_promote_identity(
    package, http_contract, monkeypatch, field, value
):
    from generalforecast.package_commands import PackageCommands

    call = PackageCommands.call

    def wrong_identity(self, method, deployment, suffix="", payload=None):
        result = call(self, method, deployment, suffix, payload)
        if suffix.endswith("/promote"):
            result[field] = value
        return result

    monkeypatch.setattr(PackageCommands, "call", wrong_identity)
    with pytest.raises(ServiceError):
        run(["push", str(package)])


def test_login_reuses_custom_profile_credential_and_redacts_it(
    http_contract, monkeypatch, capsys
):
    from generalforecast.cli import emit

    monkeypatch.setenv("WORK_KEY", http_contract.token)
    monkeypatch.delenv("GF_API_KEY")
    run(["login", "--profile", "work", "--key-env", "WORK_KEY"])
    assert run(["login", "--profile", "work"])["authenticated"]
    emit({"value": http_contract.token, "nested": {http_contract.token: "value"}})
    assert http_contract.token not in capsys.readouterr().out


@pytest.mark.parametrize(
    "key_env,configured,expected",
    [
        ("GF_API_KEY", False, True),
        ("GF_API_KEY", True, True),
        ("WORK_KEY", False, False),
        ("WORK_KEY", True, True),
    ],
)
def test_saved_profile_token_fallback_respects_key_reference(
    http_contract, monkeypatch, tmp_path, key_env, configured, expected
):
    monkeypatch.setenv("GF_TOKEN", http_contract.token)
    monkeypatch.delenv("GF_API_KEY")
    monkeypatch.delenv("WORK_KEY", raising=False)
    profile = "default" if key_env == "GF_API_KEY" else "work"
    config = tmp_path / "config/gf/profiles.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                profile: {
                    "url": os.environ["GF_URL"],
                    "project": "demo",
                    "key_env": key_env,
                }
            }
        )
    )
    config.chmod(0o600)
    if configured:
        monkeypatch.setenv(key_env, http_contract.token)
        monkeypatch.setenv("GF_TOKEN", "wrong-fallback-key")
    if expected:
        assert run(["auth", "status", "--profile", profile])["authenticated"]
        assert http_contract.calls == [("GET", "/v1alpha1/auth/status")]
    else:
        with pytest.raises(ServiceError) as error:
            run(["auth", "status", "--profile", profile])
        assert error.value.code == "ValidationError"
        assert not http_contract.calls


def test_cli_and_sdk_share_profile_schema_location_and_environment(
    http_contract, monkeypatch, tmp_path
):
    from generalforecast.commands import client_for

    monkeypatch.delenv("GF_URL")
    monkeypatch.delenv("GF_PROJECT")
    config = tmp_path / "elsewhere.toml"
    config.touch(mode=0o600)
    config.write_text(
        f'[profiles.prod]\nurl="{http_contract.url}"\nproject="demo"\n'
        'environment="staging"\nruntime="cpu"\n'
    )
    monkeypatch.setenv("GF_CONFIG_PATH", str(config))
    client = client_for(parser().parse_args(["auth", "status", "--profile", "prod"]))
    assert (client.project, client.profile.environment, client.profile.runtime) == (
        "demo",
        "staging",
        "cpu",
    )
    client.close()
    assert run(["auth", "status", "--profile", "prod"])["authenticated"]
    with pytest.raises(ServiceError) as error:
        run(["login", "--profile", "prod"])
    assert error.value.code == "UnsupportedOperation"


def test_cli_empty_default_key_fails_closed_and_login_preserves_profiles(
    http_contract, monkeypatch, tmp_path
):
    config = tmp_path / "config/gf/profiles.json"
    config.parent.mkdir(parents=True)
    config.touch(mode=0o600)
    config.write_text(
        json.dumps({"other": {"url": "https://other.example", "project": "o"}})
    )
    monkeypatch.setenv("GF_TOKEN", http_contract.token)
    monkeypatch.setenv("GF_API_KEY", "")
    with pytest.raises(ServiceError) as error:
        run(["auth", "status"])
    assert error.value.code == "ValidationError" and not http_contract.calls
    monkeypatch.delenv("GF_API_KEY")
    run(["login", "--profile", "work"])
    stored = json.loads(config.read_text())
    assert stored["other"] == {"url": "https://other.example", "project": "o"}
    assert stored["work"]["key_env"] == "GF_TOKEN"
    assert http_contract.token not in config.read_text()
