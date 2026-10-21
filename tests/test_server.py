"""The server binary end to end, over Arrow Flight, with a Python worker."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

flight = pytest.importorskip("pyarrow.flight")
pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

from tests.models.tabicl.tiny import classifier
from tests.utils import legacy_client as generalforecast
from tests.utils.server import COLUMNS, NEAR_A, TOKEN, Server, build, wait_until

pytestmark = pytest.mark.server

REPLICATION = b"Bearer replicator"


@pytest.fixture(scope="session")
def binary() -> Path:
    return build()


@pytest.fixture
def server(binary: Path, tmp_path: Path) -> Iterator[Server]:
    running = Server(binary, tmp_path).start()
    yield running
    running.stop()


def _clusters(rows: int = 32) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    centers = np.where(np.arange(rows) % 2 == 0, 0.0, 10.0)[:, None]
    features = (centers + generator.normal(size=(rows, 5))).astype(np.float32)
    return features, (np.arange(rows) % 3).astype(np.int32)


def _create(server: Server, name: str = "rows") -> None:
    server.action(
        "create_table",
        {
            "name": name,
            "model": "tiny",
            "columns": COLUMNS,
            "classes": 3,
            "strategy": {"kind": "exact"},
        },
    )


def _local(features: np.ndarray, targets: np.ndarray) -> generalforecast.Table:
    table = generalforecast.local().create_table(
        "rows", model=classifier(), features=5, classes=3
    )
    table.insert(np.arange(len(features)), features, targets)
    return table


def test_server_predictions_match_local_mode(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)

    remote = server.predict("rows", NEAR_A)

    np.testing.assert_allclose(
        remote, _local(features, targets).predict(NEAR_A), atol=1e-5
    )
    assert server.action("list_tables") == [["rows"]]


def test_concurrent_requests_each_get_their_predictions(server: Server) -> None:
    features, targets = _clusters()
    for name in ("first", "second"):
        _create(server, name)
        server.write("insert", name, np.arange(32), features, targets)
    queries = {"first": NEAR_A, "second": NEAR_A + 10.0}
    expected = {name: server.predict(name, rows) for name, rows in queries.items()}

    # Requests arrive while earlier steps run, so steps overlap in the pipeline.
    names = ["first", "second"] * 12
    with ThreadPoolExecutor(8) as pool:
        results = list(
            pool.map(lambda name: server.predict(name, queries[name]), names)
        )

    for name, result in zip(names, results, strict=True):
        np.testing.assert_allclose(result, expected[name], atol=1e-5)


def test_large_frames_travel_through_shared_memory(server: Server) -> None:
    # The fit carries every row: 50,000 rows of 5 features is about 1.4 MB.
    generator = np.random.default_rng(0)
    features = generator.normal(size=(50_000, 5)).astype(np.float32)
    targets = (features[:, 0] > 0).astype(np.int32)
    server.action(
        "create_table",
        {
            "name": "rows",
            "model": "tiny",
            "columns": COLUMNS,
            "classes": 3,
            "strategy": {"kind": "retrieved", "neighbors": 256},
        },
    )
    server.write("insert", "rows", np.arange(len(features)), features, targets)

    predictions = server.predict("rows", NEAR_A)

    assert predictions.shape == (2, 3)
    assert server.process is not None
    pattern = f"gf-{server.process.pid}-*/*.frame"
    places = [Path(tempfile.gettempdir()), Path("/dev/shm")]
    assert not [path for place in places for path in place.glob(pattern)]


def test_compaction_drops_old_records_and_keeps_the_rows(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)
    # Records the compacted log no longer needs, well past what it adds.
    for shift in (1.0, 2.0, 3.0):
        server.write("update", "rows", np.arange(32), features + shift, targets)
    server.write("update", "rows", np.arange(8), features[:8] + 1.0, targets[:8])
    server.action("delete", {"table": "rows", "keys": list(range(8, 16))})
    before = server.predict("rows", NEAR_A)
    log = next((server.config.parent / "data" / "tables").glob("rows.log"))
    size = log.stat().st_size

    (length,) = server.action("compact", {"table": "rows"})
    server.stop()
    server.start()

    assert log.stat().st_size < size
    assert server.action("describe", {"table": "rows"})[0]["rows"] == 24
    assert length["log_length"] == 8
    np.testing.assert_allclose(server.predict("rows", NEAR_A), before, atol=1e-5)


def test_rows_past_memory_wait_sealed_on_disk(binary: Path, tmp_path: Path) -> None:
    server = Server(binary, tmp_path)
    # No memory for rows: every block goes to the table's row file.
    server.config.write_text("row_memory_bytes = 0\n" + server.config.read_text())
    server.start()
    try:
        features, targets = _clusters(64)
        _create(server)
        server.write("insert", "rows", np.arange(64), features, targets)
        before = server.predict("rows", NEAR_A)
        rows = (server.tables / "rows.rows").read_bytes()

        assert len(rows) > 0
        assert features.astype(np.float32)[0].tobytes() not in rows
        server.stop()
        server.start()
        np.testing.assert_allclose(server.predict("rows", NEAR_A), before, atol=1e-5)
    finally:
        server.stop()


def test_tables_whose_rows_are_wider_than_a_slot_are_served(
    binary: Path, tmp_path: Path
) -> None:
    from generalforecast.runtime.memory import memory_profile

    functions, model = classifier()
    row = memory_profile(functions, model, 5).cache_bytes_per_row
    server = Server(binary, tmp_path)
    server.config.write_text(
        f"slot_bytes = {row // 3 + 1}\n" + server.config.read_text()
    )
    server.start()
    try:
        features, targets = _clusters()
        _create(server)
        server.write("insert", "rows", np.arange(32), features, targets)

        np.testing.assert_allclose(
            server.predict("rows", NEAR_A),
            _local(features, targets).predict(NEAR_A),
            atol=1e-5,
        )
    finally:
        server.stop()


def _exchange(
    server: Server, command: str, batch: pa.RecordBatch, body: dict, token: bytes
) -> pa.Table:
    """An exchange whose descriptor names `command` on `rows` and carries `body`."""
    descriptor = flight.FlightDescriptor.for_path(command, "rows", json.dumps(body))
    options = flight.FlightCallOptions(headers=[(b"authorization", token)])
    writer, reader = server.client.do_exchange(descriptor, options)
    writer.begin(batch.schema)
    writer.write_batch(batch)
    writer.done_writing()
    result = reader.read_all()
    writer.close()
    return result


def test_a_partition_offers_rows_and_predicts_from_rows_it_is_given(
    server: Server,
) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)
    queries = pa.RecordBatch.from_pydict(
        {f"f{index}": pa.array(NEAR_A[:, index]) for index in range(5)}
    )

    with pytest.raises(flight.FlightError, match="replication token"):
        _exchange(server, "candidates", queries, {}, TOKEN)
    offered = _exchange(server, "candidates", queries, {}, REPLICATION)
    keyed = offered.filter(pc.is_valid(offered["key"]))
    asked = offered.filter(pc.is_null(offered["key"]))
    rows = pa.concat_tables([keyed, asked]).drop_columns(["query", "score"])
    given = {"contexts": [list(range(keyed.num_rows))] * len(NEAR_A)}
    predicted = _exchange(
        server,
        "predict_given",
        rows.combine_chunks().to_batches()[0],
        given,
        REPLICATION,
    )

    # An exact table offers every row once, to every query, then the queries.
    assert keyed.num_rows == 32
    assert keyed["query"].null_count == 32
    assert asked.num_rows == len(NEAR_A)
    np.testing.assert_allclose(
        np.column_stack([column.to_numpy() for column in predicted.columns]),
        server.predict("rows", NEAR_A),
        atol=1e-5,
    )
    # Rows given again read the contexts encoded for them before.
    again = _exchange(
        server,
        "predict_given",
        rows.combine_chunks().to_batches()[0],
        given,
        REPLICATION,
    )
    np.testing.assert_allclose(
        np.column_stack([column.to_numpy() for column in again.columns]),
        np.column_stack([column.to_numpy() for column in predicted.columns]),
        atol=1e-6,
    )
    _, text = server.http("/metrics")
    metrics = dict(
        line.split(" ", 1)
        for line in text.splitlines()
        if line and not line.startswith("#")
    )
    assert int(metrics["gf_units_reused_total"]) >= 1
    # Predictions from given rows count toward the partition's own latency.
    assert metrics['gf_table_predict_seconds_count{table="rows"}'] == "3"
    # The tables made for requests are the server's own, never listed.
    assert server.action("list_tables") == [["rows"]]
    # Only a router splits a table into partitions.
    with pytest.raises(pa.ArrowException, match="only through a router"):
        server.action(
            "create_table",
            {
                "name": "split",
                "model": "tiny",
                "columns": COLUMNS,
                "classes": 3,
                "partitions": 2,
            },
        )


def test_requests_need_a_token(server: Server) -> None:
    with pytest.raises(flight.FlightUnauthenticatedError):
        list(server.client.do_action(flight.Action("list_tables", b"{}")))


def test_scoped_tokens_reach_only_their_tables(server: Server) -> None:
    features, targets = _clusters()
    _create(server, "payments")
    clinic = generalforecast.connect(f"grpc://127.0.0.1:{server.port}", token="clinic")
    table = clinic.create_table("clinic-1", model="tiny", features=5, classes=3)
    table.insert(np.arange(32), features, targets)

    assert clinic.tables() == ["clinic-1"]
    assert table.predict(NEAR_A).shape == (2, 3)
    with pytest.raises(PermissionError, match="payments"):
        clinic.create_table("payments-2", model="tiny", features=5, classes=3)
    # Tables outside the scope look as if they did not exist.
    with pytest.raises(KeyError, match="payments"):
        clinic.table("payments")
    with pytest.raises(KeyError, match="payments"):
        clinic.drop_table("payments")


def test_a_write_sent_twice_with_one_id_applies_once(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    headers = ((b"gf-write-id", b"retry-1"),)

    server.write("insert", "rows", np.arange(32), features, targets, headers=headers)
    server.write("insert", "rows", np.arange(32), features, targets, headers=headers)
    with pytest.raises(pa.ArrowException, match="key 0"):
        server.write("insert", "rows", np.arange(32), features, targets)

    assert server.action("describe", {"table": "rows"})[0]["rows"] == 32
    assert server.action("describe", {"table": "rows"})[0]["log_length"] == 2


def test_a_create_its_client_gave_up_on_leaves_the_name_usable(
    server: Server,
) -> None:
    create = {"name": "rows", "model": "tiny", "columns": COLUMNS, "classes": 3}
    hasty = flight.FlightCallOptions(
        headers=[(b"authorization", b"Bearer secret")], timeout=0.001
    )
    action = flight.Action("create_table", json.dumps(create).encode())
    with pytest.raises(pa.ArrowException):
        list(server.client.do_action(action, hasty))

    # The first creation finished or never began; either way the name works.
    try:
        server.action("create_table", create)
    except pa.ArrowException as error:
        assert "exists" in str(error)
    assert server.action("list_tables") == [["rows"]]
    assert server.action("describe", {"table": "rows"})[0]["rows"] == 0


def test_only_replication_tokens_read_logs(server: Server) -> None:
    server.action(
        "create_table",
        {"name": "rows", "model": "tiny", "columns": COLUMNS, "classes": 3},
    )
    ticket = flight.Ticket(json.dumps({"table": "rows", "from": 0}).encode())
    lengths = flight.Action("log_lengths", b"{}")
    replicator = flight.FlightCallOptions(
        headers=[(b"authorization", b"Bearer replicator")]
    )

    with pytest.raises(flight.FlightError, match="replication token"):
        server.client.do_get(ticket, server.options()).read_all()
    with pytest.raises(flight.FlightError, match="replication token"):
        server.action("log_lengths")
    (result,) = server.client.do_action(lengths, replicator)
    term = {"term": 0, "primary": "", "start": 0}
    (listed,) = json.loads(result.body.to_pybytes()).values()
    assert (listed["length"], listed["terms"]) == (1, [term])
    assert len(listed["created"]) == 16
    # Routers forward client writes under replication tokens, so applying log
    # entries also needs the mark only a router's own shipments carry.
    descriptor = flight.FlightDescriptor.for_path("replicate", "rows")
    writer, _ = server.client.do_put(descriptor, pa.schema([]), replicator)
    with pytest.raises(pa.ArrowException, match="shipment"):
        writer.close()
    # A router's token scoped to other tables neither reads nor lists this log.
    scoped = flight.FlightCallOptions(
        headers=[(b"authorization", b"Bearer clinic-router")]
    )
    with pytest.raises(KeyError, match="does not exist"):
        server.client.do_get(ticket, scoped).read_all()
    (result,) = server.client.do_action(lengths, scoped)
    assert json.loads(result.body.to_pybytes()) == {}


def test_writes_are_logged_only_at_the_term_they_were_sent_for(
    server: Server,
) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(16), features[:16], targets[:16])

    with pytest.raises(pa.ArrowException, match="term 0"):
        server.write(
            "insert",
            "rows",
            np.arange(16, 32),
            features[16:],
            targets[16:],
            headers=((b"gf-term", b"1"),),
        )
    server.write(
        "insert",
        "rows",
        np.arange(16, 32),
        features[16:],
        targets[16:],
        headers=((b"gf-term", b"0"),),
    )
    assert server.action("describe", {"table": "rows"})[0]["rows"] == 32


def test_rows_survive_a_restart(binary: Path, tmp_path: Path) -> None:
    features, targets = _clusters()
    server = Server(binary, tmp_path).start()
    for name in ("rows", "damaged"):
        _create(server, name)
        server.write("insert", name, np.arange(32), features, targets)
    before = server.predict("rows", NEAR_A)
    server.stop()
    damaged = server.tables / "damaged.log"
    data = bytearray(damaged.read_bytes())
    data[len(data) // 2] ^= 1
    damaged.write_bytes(bytes(data))

    # A log that does not open is set aside, and the other tables start.
    server.start()
    try:
        after = server.predict("rows", NEAR_A)
        tables = server.action("list_tables")
    finally:
        server.stop()

    np.testing.assert_allclose(after, before, atol=1e-6)
    assert tables == [["rows"]]
    assert (server.tables / "damaged.log.unreadable").exists()


def test_changes_through_the_api_match_local_mode(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)
    server.predict("rows", NEAR_A)
    local = _local(features, targets)
    local.predict(NEAR_A)

    changed = features[[1, 5]] + 1.0
    server.write("update", "rows", [1, 5], changed, [2, 2])
    server.write("upsert", "rows", [5, 40], changed, [0, 1])
    server.action("delete", {"table": "rows", "keys": [2, 7, 99], "missing_ok": True})
    server.action("refit", {"table": "rows"})
    local.update([1, 5], changed, [2, 2])
    local.upsert([5, 40], changed, [0, 1])
    local.delete([2, 7, 99], missing_ok=True)
    local.refit()

    np.testing.assert_allclose(
        server.predict("rows", NEAR_A), local.predict(NEAR_A), atol=1e-5
    )


def test_tables_are_checked_against_their_model_when_created(server: Server) -> None:
    for classes in (0, 99):
        with pytest.raises(pa.ArrowInvalid, match="model"):
            server.action(
                "create_table",
                {
                    "name": "rows",
                    "model": "tiny",
                    "columns": COLUMNS,
                    "classes": classes,
                },
            )
    assert server.action("list_tables")[0] == []


def test_bad_requests_are_refused_and_change_nothing(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)
    before = server.predict("rows", NEAR_A)

    with pytest.raises(pa.ArrowException):
        server.write("insert", "rows", [0], features[:1], targets[:1])  # key 0 exists
    with pytest.raises(pa.ArrowException):
        server.predict(
            "rows", np.zeros((1, 4), dtype=np.float32)
        )  # four features, not five
    with pytest.raises(pa.ArrowException):
        _create(server)  # the name is taken
    with pytest.raises(pa.ArrowException):
        server.predict("missing", NEAR_A)
    # Names are checked before they reach a file path.
    with pytest.raises(pa.ArrowException, match="table names are"):
        server.predict("../rows", NEAR_A)
    with pytest.raises(pa.ArrowException, match="table names are"):
        server.action(
            "create_table",
            {"name": "../outside", "model": "tiny", "columns": COLUMNS, "classes": 3},
        )
    row = {
        "key": pa.array([100], pa.uint64()),
        **{f"f{index}": pa.array([0.0], pa.float32()) for index in range(5)},
        "target": pa.array([0], pa.int32()),
    }
    null_key = {**row, "key": pa.array([None], pa.uint64())}
    null_time = {**row, "time": pa.array([None], pa.int64())}
    descriptor = flight.FlightDescriptor.for_path("insert", "rows")
    for columns in (null_key, null_time):
        batch = pa.RecordBatch.from_pydict(columns)
        writer, _ = server.client.do_put(descriptor, batch.schema, server.options())
        writer.write_batch(batch)
        with pytest.raises(pa.ArrowException, match="nulls"):
            writer.close()
    # A column of Arrow's null type takes no bytes on the wire however many
    # rows it claims, so requests are bounded by what their rows take decoded.
    vast = pa.RecordBatch.from_pydict(
        {f"f{index}": pa.nulls(1 << 28) for index in range(5)}
    )
    descriptor = flight.FlightDescriptor.for_path("predict", "rows")
    writer, reader = server.client.do_exchange(descriptor, server.options())
    with pytest.raises(pa.ArrowException, match="MiB"):
        writer.begin(vast.schema)
        writer.write_batch(vast)
        writer.done_writing()
        reader.read_all()

    np.testing.assert_array_equal(server.predict("rows", NEAR_A), before)


def _certificate(directory: Path) -> tuple[Path, Path]:
    """A self-signed certificate for localhost, and its key."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path, key_path = directory / "server.pem", directory / "server.key"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


def test_tls_serves_encrypted_clients_and_refuses_plaintext(
    binary: Path, tmp_path: Path
) -> None:
    server = Server(binary, tmp_path, tls=_certificate(tmp_path)).start()
    try:
        features, targets = _clusters()
        _create(server)
        server.write("insert", "rows", np.arange(32), features, targets)
        predictions = server.predict("rows", NEAR_A)

        plaintext = flight.FlightClient(f"grpc://127.0.0.1:{server.port}")
        with pytest.raises(pa.ArrowException):
            list(plaintext.do_action(flight.Action("health", b"{}"), server.options()))
    finally:
        server.stop()

    assert predictions.shape == (2, 3)
    # Clients keep tokens off the network in plaintext unless told otherwise.
    with pytest.raises(ValueError, match="unencrypted"):
        generalforecast.connect("grpc://generalforecast.internal:8815", token="secret")


def test_metrics_count_rows_steps_and_encodes(binary: Path, tmp_path: Path) -> None:
    server = Server(binary, tmp_path, timing_log=True, profiles=True).start()
    try:
        _check_metrics(server)
    finally:
        server.stop()


def _check_metrics(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)
    server.predict("rows", NEAR_A)
    queries = pa.RecordBatch.from_pydict(
        {f"f{index}": pa.array(NEAR_A[:, index]) for index in range(5)}
    )
    answer = _exchange(server, "predict", queries, {}, TOKEN)

    status, text = server.http("/metrics")

    assert status == 200
    metrics = dict(
        line.split(" ", 1)
        for line in text.splitlines()
        if line and not line.startswith("#")
    )
    assert metrics["gf_rows_written_total"] == "32"
    assert metrics["gf_predictions_total"] == "4"
    assert metrics["gf_steps_total"] == "2"
    assert (metrics["gf_units_encoded_total"], metrics["gf_units_reused_total"]) == (
        "1",
        "1",
    )
    assert metrics["gf_tables"] == "1"
    assert metrics['gf_table_rows_written_total{table="rows"}'] == "32"
    assert metrics['gf_table_predictions_total{table="rows"}'] == "4"
    assert metrics['gf_table_predict_seconds_count{table="rows"}'] == "2"
    assert server.http("/healthz")[0] == server.http("/readyz")[0] == 200
    # Each step's worker time goes to its requests, and the one fit to its table.
    worker = float(metrics["gf_worker_seconds_total"])
    assert float(metrics['gf_table_gpu_seconds_total{table="rows"}']) == (
        pytest.approx(worker, abs=1e-6)
    )
    assert 0 < float(answer.schema.metadata[b"generalforecast.gpu_seconds"]) < worker
    # The state that answered: the first fit's.
    assert answer.schema.metadata[b"generalforecast.epoch"] == b"0"
    assert metrics['gf_worker_stage_calls_total{stage="fit"}'] == "1"
    assert int(metrics['gf_worker_stage_calls_total{stage="decode"}']) >= 2
    assert metrics['gf_table_refits_total{table="rows"}'] == "1"
    assert metrics['gf_table_refit_rows_total{table="rows"}'] == "32"
    assert float(metrics['gf_table_refit_seconds_total{table="rows"}']) == (
        pytest.approx(float(metrics['gf_worker_stage_seconds_total{stage="fit"}']))
    )
    assert int(metrics['gf_table_index_bytes{table="rows"}']) > 0
    assert metrics['gf_table_rebuild_seconds_total{table="rows"}'] == "0"
    # Timed steps, plans and requests count their parts.
    assert int(metrics['gf_worker_part_calls_total{part="other"}']) >= 2
    assert int(metrics['gf_plan_part_calls_total{part="checks"}']) >= 2
    for part in ("decode", "route", "shard", "encode"):
        assert int(metrics[f"gf_request_{part}_seconds_count"]) >= 3
    # The timing log has a line per request, in order, adding up as well.
    wait_until(lambda: len(server.timings.read_text().splitlines()) == 2)
    lines = [json.loads(line) for line in server.timings.read_text().splitlines()]
    for line in lines:
        assert (line["table"], line["queries"]) == ("rows", len(NEAR_A))
        names = ("arrived", "planned", "sent", "started", "finished")
        times = [line[f"{name}_us"] for name in names]
        assert times == sorted(times)
    assert sum(line["gpu_seconds"] for line in lines) == pytest.approx(worker, abs=1e-6)


def test_untimed_servers_export_the_part_metrics_empty(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)
    server.predict("rows", NEAR_A)

    _, text = server.http("/metrics")

    for family in ("gf_worker_part_calls_total", "gf_plan_part_calls_total"):
        assert f"# TYPE {family} counter" in text
        assert f"{family}{{" not in text
    for part in ("decode", "route", "shard", "encode"):
        assert f"\ngf_request_{part}_seconds_count 0\n" in text


def test_a_worker_that_dies_is_started_again_with_its_tables(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)
    expected = server.predict("rows", NEAR_A)
    assert server.process is not None

    subprocess.run(["pkill", "-KILL", "-P", str(server.process.pid)], check=True)

    wait_until(lambda: server.http("/readyz") == (503, "restarting a worker"))
    wait_until(lambda: server.http("/readyz")[0] == 200)
    np.testing.assert_allclose(server.predict("rows", NEAR_A), expected, atol=1e-6)


def test_sigterm_drains_and_exits_cleanly(server: Server) -> None:
    features, targets = _clusters()
    _create(server)
    server.write("insert", "rows", np.arange(32), features, targets)
    server.predict("rows", NEAR_A)

    assert server.stop() == 0


def test_a_page_written_once_embeds_the_written_row_and_copies_the_rest(
    server: Server,
) -> None:
    features, targets = _clusters(33)
    for name in ("written", "plain"):
        _create(server, name)
        server.write("insert", name, np.arange(32), features[:32], targets[:32])
        server.predict(name, NEAR_A)
        server.write("insert", name, [32], features[32:], targets[32:])
    # The plain table's other rows are written as they were, so its page's
    # earlier entry has no row to copy and the page embeds whole.
    server.write("update", "plain", np.arange(32), features[:32], targets[:32])
    written = server.predict("written", NEAR_A)
    plain = server.predict("plain", NEAR_A)

    np.testing.assert_allclose(written, plain, atol=1e-6)
    _, text = server.http("/metrics")
    metrics = dict(
        line.split(" ", 1)
        for line in text.splitlines()
        if line and not line.startswith("#")
    )
    # The fit computed each page; the next version of the written table's
    # embeds the inserted row alone and copies the other 32.
    assert metrics['gf_table_rows_embedded_total{table="written"}'] == "1"
    assert metrics['gf_table_rows_copied_total{table="written"}'] == "32"
    assert metrics['gf_table_rows_embedded_total{table="plain"}'] == "33"
    assert metrics['gf_table_rows_copied_total{table="plain"}'] == "0"


def test_worker_profiles_need_profiling_and_stay_under_its_directory(
    binary: Path, tmp_path: Path
) -> None:
    body = {"directory": "run-1", "seconds": 0.2}
    for name in ("without", "with"):
        (tmp_path / name).mkdir()
    without = Server(binary, tmp_path / "without").start()
    try:
        with pytest.raises(pa.ArrowException, match=r"\[profiling\]"):
            without.action("profile_workers", body)
    finally:
        without.stop()

    server = Server(binary, tmp_path / "with", profiles=True).start()
    try:
        # A token scoped to some tables may not profile the whole server.
        scoped = flight.FlightCallOptions(
            headers=[(b"authorization", b"Bearer clinic")]
        )
        action = flight.Action("profile_workers", json.dumps(body).encode())
        with pytest.raises(pa.ArrowException, match="unscoped token"):
            list(server.client.do_action(action, scoped))
        for escaping in ["/tmp/run-1", "../run-1"]:
            with pytest.raises(pa.ArrowException, match="relative"):
                server.action("profile_workers", {**body, "directory": escaping})

        (answer,) = server.action("profile_workers", body)

        assert answer["seconds"] == 0.2 and answer["traces"] == []
        (stacks,) = (Path(file) for file in answer["files"])
        assert stacks == server.profiles / "run-1" / "worker-0.txt"
        # The file appears once the window ends.
        wait_until(stacks.exists)
        assert stacks.read_text()
    finally:
        server.stop()
