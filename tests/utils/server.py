# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A gf-server process for tests, and a raw Flight client for it."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import flight

from tests.utils import processes

ROOT = Path(__file__).resolve().parents[2]
TOKEN = b"Bearer secret"
KEY = "ab" * 32
NEAR_A = np.zeros((2, 5), dtype=np.float32)
# The five numeric columns `features=5` makes, as a raw create body names them.
COLUMNS = [{"name": f"f{index}", "kind": "numeric"} for index in range(5)]

# How long a test server may take to answer once started. Its worker first
# compiles the programs this session's servers recorded, which on a machine
# busy with other test workers takes minutes rather than seconds.
STARTUP_SECONDS = float(os.environ.get("GF_TEST_STARTUP_SECONDS", "600"))
# How long a condition a test waits on may take to hold under the same load,
# such as a replica catching up after its node starts again.
SETTLE_SECONDS = float(os.environ.get("GF_TEST_SETTLE_SECONDS", "180"))
# How long a stopped server may take to drain before it is killed.
STOP_SECONDS = 60.0


def _session_cache() -> Path:
    """This process's cache directory for the servers it starts, removed at exit.

    A worker compiles every program the servers before it recorded as it
    starts, and records live in the cache directory. Shared with every run
    on the machine, they grow with every shape any test ever asked for, and
    each start compiles them all; a directory of this session's own keeps a
    start to what its tests asked for.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = Path(tempfile.mkdtemp(prefix="gf-test-cache-"))
        atexit.register(shutil.rmtree, _CACHE, ignore_errors=True)
    return _CACHE


_CACHE: Path | None = None


def server_environment() -> dict[str, str]:
    """The environment test servers run under, with this session's cache directory.

    JAX's compile cache stays the shared one, so programs a worker compiled
    in an earlier run load rather than compile again.
    """
    from generalforecast.runtime.device import cache_root

    shared = os.environ.get("JAX_COMPILATION_CACHE_DIR") or str(cache_root() / "jax")
    return {
        **os.environ,
        "GF_TEST_KEY": KEY,
        "PYTHONPATH": str(ROOT),
        "XDG_CACHE_HOME": str(_session_cache()),
        "JAX_COMPILATION_CACHE_DIR": shared,
    }


def _ready(process: subprocess.Popen, what: str, probe: Callable[[], object]) -> None:
    """Waits until `probe` succeeds, at most `STARTUP_SECONDS`.

    Raises:
        AssertionError: If the process exits first, or the time runs out; the
            process is killed then.
    """
    deadline = time.monotonic() + STARTUP_SECONDS
    while True:
        try:
            probe()
            return
        except Exception as error:
            code = process.poll()
            if code is not None:
                processes.kill(process)
                raise AssertionError(
                    f"{what} exited with {code} as it started"
                ) from error
            if time.monotonic() > deadline:
                processes.stop(process, timeout=STOP_SECONDS)
                raise AssertionError(
                    f"{what} did not answer within {STARTUP_SECONDS:.0f} seconds"
                ) from error
        time.sleep(0.2)


def build() -> Path:
    """Builds the server binary, or skips without cargo."""
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed")
    subprocess.run(["cargo", "build", "-q", "-p", "gf-server"], cwd=ROOT, check=True)
    return ROOT / "target" / "debug" / "gf-server"


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Server:
    """A running gf-server with one worker, and a Flight client for it.

    With `tls`, the server presents that certificate and key; `trust` is what the
    test client trusts, the certificate itself by default. With `client_ca`, the
    server requires client certificates, and the test client presents `client`.
    """

    def __init__(
        self,
        binary: Path,
        directory: Path,
        tls: tuple[Path, Path] | None = None,
        client_ca: Path | None = None,
        trust: Path | None = None,
        client: tuple[Path, Path] | None = None,
        name: str = "node",
        require_lease: bool = False,
        timing_log: bool = False,
        workers: int = 1,
        placement: str = "",
        profiles: bool = False,
    ) -> None:
        self.name = name
        self.tables = directory / "data" / "tables"
        self.port, self.metrics_port = _port(), _port()
        self.config = directory / "generalforecast.toml"
        # With `timing_log`, the server logs each request's timings here.
        self.timings = directory / "timings.jsonl"
        timings = f'timing_log = "{self.timings}"' if timing_log else ""
        # `profiles` also turns on the server's timers.
        self.profiles = directory / "profiles"
        profiling = (
            f'\n[profiling]\ndirectory = "{self.profiles}"\n' if profiles else ""
        )
        worker = f"""
[[workers]]
command = ["{sys.executable}", "-m", "generalforecast.worker_process"]
models = {{ tiny = "tests/tabicl-classifier", tiny-capped = "tests/tabicl-classifier-capped", tiny-regressor = "tests/tabicl-regressor", tiny-forecaster = "tests/timestral-forecaster" }}
"""
        tls_section = ""
        if tls is not None:
            tls_section = f'\n[tls]\ncertificate = "{tls[0]}"\nkey = "{tls[1]}"\n'
            if client_ca is not None:
                tls_section += f'client_ca = "{client_ca}"\n'
        self.config.write_text(
            f"""
listen = "127.0.0.1:{self.port}"
metrics_listen = "127.0.0.1:{self.metrics_port}"
{timings}
data ="{directory / "data"}"
tokens = ["secret", {{ token = "clinic", tables = ["clinic-*"] }}]
replication_tokens = ["replicator", {{ token = "clinic-router", tables = ["clinic-*"] }}]
require_lease = {str(require_lease).lower()}
encryption_key_env = "GF_TEST_KEY"
# Small tiers, so test servers never size themselves by the machine.
device_bytes = {64 << 20}
host_bytes = {256 << 20}
{worker * workers}{placement}
{tls_section}{profiling}"""
        )
        self.binary = binary
        self.tls = tls
        self.trust = (
            trust if trust is not None else (tls[0] if tls is not None else None)
        )
        self.client_identity = client
        self.process: subprocess.Popen | None = None

    @property
    def address(self) -> str:
        scheme = "https" if self.tls is not None else "http"
        return f"{scheme}://localhost:{self.port}"

    def start(self) -> Server:
        """Starts the server, and returns once it answers over Flight.

        It listens only once its workers have compiled the programs they
        recorded, so an answer means the server takes calls at once.
        """
        self.process = processes.start(
            [str(self.binary), "run", "--config", str(self.config)],
            env=server_environment(),
        )
        if self.trust is None:
            self.client = flight.FlightClient(f"grpc://127.0.0.1:{self.port}")
        else:
            options = {"tls_root_certs": self.trust.read_bytes()}
            if self.client_identity is not None:
                options["cert_chain"] = self.client_identity[0].read_bytes()
                options["private_key"] = self.client_identity[1].read_bytes()
            self.client = flight.FlightClient(
                f"grpc+tls://localhost:{self.port}", **options
            )
        _ready(self.process, f"gf-server {self.name}", lambda: self.action("health"))
        return self

    def stop(self) -> int:
        """Stops the server with SIGTERM and returns its exit code."""
        code = 0
        if self.process is not None:
            stopped = processes.stop(self.process, timeout=STOP_SECONDS)
            assert stopped is not None, "the server did not stop, and was killed"
            code, self.process = stopped, None
        return code

    def http(self, path: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.metrics_port}{path}"
            ) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def options(self) -> flight.FlightCallOptions:
        return flight.FlightCallOptions(headers=[(b"authorization", TOKEN)])

    def action(self, kind: str, body: dict | None = None) -> list:
        payload = json.dumps(body or {}).encode()
        results = self.client.do_action(flight.Action(kind, payload), self.options())
        return [json.loads(result.body.to_pybytes() or b"null") for result in results]

    def write(
        self,
        kind: str,
        table: str,
        keys,
        features,
        targets,
        headers: tuple[tuple[bytes, bytes], ...] = (),
    ) -> None:
        columns = {"key": pa.array(np.asarray(keys, dtype=np.uint64))}
        for index in range(features.shape[1]):
            columns[f"f{index}"] = pa.array(features[:, index].astype(np.float32))
        columns["target"] = pa.array(np.asarray(targets, dtype=np.int32))
        batch = pa.RecordBatch.from_pydict(columns)
        descriptor = flight.FlightDescriptor.for_path(kind, table)
        options = flight.FlightCallOptions(
            headers=[(b"authorization", TOKEN), *headers]
        )
        writer, _ = self.client.do_put(descriptor, batch.schema, options)
        writer.write_batch(batch)
        writer.close()

    def predict(self, table: str, queries: np.ndarray) -> np.ndarray:
        batch = pa.RecordBatch.from_pydict(
            {
                f"f{index}": pa.array(queries[:, index])
                for index in range(queries.shape[1])
            }
        )
        descriptor = flight.FlightDescriptor.for_path("predict", table)
        writer, reader = self.client.do_exchange(descriptor, self.options())
        writer.begin(batch.schema)
        writer.write_batch(batch)
        writer.done_writing()
        result = reader.read_all()
        writer.close()
        return np.column_stack([column.to_numpy() for column in result.columns])


class Router:
    """A running gf-router in front of nodes."""

    def __init__(
        self,
        binary: Path,
        directory: Path,
        nodes: list[Server],
        clients: str = '[[clients]]\ntoken = "secret"\n',
        tls: str = "",
        node_tls: str = "",
        replicas: int = 1,
        failover: float | None = None,
        write_timeout: float | None = None,
        name: str = "router",
    ) -> None:
        self.port = _port()
        self.metrics_port = _port()
        self.config = directory / f"{name}.toml"
        # What the router logs, for tests to read.
        self.log = directory / f"{name}.log"
        self.binary = binary
        self.settings = (
            f'listen = "127.0.0.1:{self.port}"\nname = "{name}"\n'
            f'metrics_listen = "127.0.0.1:{self.metrics_port}"\n'
            f"replicas = {replicas}\ncheck_seconds = 0.2\n{clients}\n{tls}\n"
        )
        if failover is not None:
            self.settings = f"failover_seconds = {failover}\n" + self.settings
        if write_timeout is not None:
            self.settings = f"write_timeout_seconds = {write_timeout}\n" + self.settings
        self.node_tls = node_tls
        self.process: subprocess.Popen | None = None
        self._write(nodes)

    def _write(self, nodes: list[Server]) -> None:
        self.nodes = nodes
        sections = "".join(
            f'\n[[nodes]]\nname = "{node.name}"\naddress = "{node.address}"\n'
            f'token = "replicator"\n{self.node_tls}'
            for node in nodes
        )
        self.config.write_text(self.settings + sections)

    def start(self, check: Callable[[], object]) -> Router:
        """Starts the router, and returns once `check`, a call through it, succeeds.

        The router serves only once it has learned its nodes' copies.
        """
        with self.log.open("ab") as log:
            self.process = processes.start(
                [str(self.binary), "run", "--config", str(self.config)], stderr=log
            )
        _ready(self.process, "gf-router", check)
        return self

    def reload(self, nodes: list[Server]) -> None:
        """Rewrites the node list and signals the router to reload it."""
        self._write(nodes)
        assert self.process is not None
        self.process.send_signal(signal.SIGHUP)

    def locate(self, table: str) -> list[str]:
        """The nodes that should hold a table, primary first."""
        client = flight.FlightClient(f"grpc://127.0.0.1:{self.port}")
        options = flight.FlightCallOptions(headers=[(b"authorization", TOKEN)])
        action = flight.Action("locate", json.dumps({"table": table}).encode())
        (result,) = client.do_action(action, options)
        return json.loads(result.body.to_pybytes())["nodes"]

    def stop(self) -> None:
        if self.process is not None:
            processes.stop(self.process, timeout=STOP_SECONDS)
            self.process = None

    def metric(self, line: str) -> bool:
        """Whether the router's metrics hold a line, such as `gf_router_tables 2`."""
        url = f"http://127.0.0.1:{self.metrics_port}/metrics"
        with urllib.request.urlopen(url, timeout=5) as response:
            return line in response.read().decode().splitlines()

    def copies(self, table: str) -> list[Server]:
        """The nodes that should hold a table, primary first."""
        by_name = {node.name: node for node in self.nodes}
        return [by_name[name] for name in self.locate(table)]


def wait_until(condition: Callable[[], bool], timeout: float = SETTLE_SECONDS) -> None:
    """Polls until a condition holds, treating errors as not yet."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if condition():
                return
        except Exception:
            if time.monotonic() > deadline:
                raise
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for a condition")
        time.sleep(0.2)


def build_router() -> Path:
    """Builds the router binary, or skips without cargo."""
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed")
    subprocess.run(["cargo", "build", "-q", "-p", "gf-router"], cwd=ROOT, check=True)
    return ROOT / "target" / "debug" / "gf-router"


def certificates(directory: Path) -> dict[str, Path]:
    """A certificate authority, and certificates it signed for localhost and a client."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.UTC)
    paths: dict[str, Path] = {}

    def save(
        name: str, certificate: x509.Certificate, key: ec.EllipticCurvePrivateKey
    ) -> None:
        paths[f"{name}.pem"] = directory / f"{name}.pem"
        paths[f"{name}.key"] = directory / f"{name}.key"
        paths[f"{name}.pem"].write_bytes(
            certificate.public_bytes(serialization.Encoding.PEM)
        )
        paths[f"{name}.key"].write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )

    def builder(subject: str, issuer: x509.Name, key: ec.EllipticCurvePrivateKey):
        return (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1))
        )

    authority_key = ec.generate_private_key(ec.SECP256R1())
    authority_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "generalforecast test authority")]
    )
    authority = (
        builder("generalforecast test authority", authority_name, authority_key)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(authority_key, hashes.SHA256())
    )
    save("authority", authority, authority_key)
    for name, usage in (
        ("server", ExtendedKeyUsageOID.SERVER_AUTH),
        ("client", ExtendedKeyUsageOID.CLIENT_AUTH),
    ):
        key = ec.generate_private_key(ec.SECP256R1())
        certificate = (
            builder(name, authority_name, key)
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    ]
                ),
                critical=False,
            )
            .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
            .sign(authority_key, hashes.SHA256())
        )
        save(name, certificate, key)
    return paths
