# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The router in front of nodes: placement, forwarding, quotas, mutual TLS and replicas."""

from __future__ import annotations

import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
flight = pytest.importorskip("pyarrow.flight")

from tests.utils import engine_client as generalforecast
from tests.utils.server import (
    NEAR_A,
    SETTLE_SECONDS,
    Router,
    Server,
    build,
    build_router,
    certificates,
    wait_until,
)

pytestmark = pytest.mark.router


@pytest.fixture(scope="session")
def binaries() -> tuple[Path, Path]:
    return build(), build_router()


def _clusters(rows: int = 32) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    centers = np.where(np.arange(rows) % 2 == 0, 0.0, 10.0)[:, None]
    features = (centers + generator.normal(size=(rows, 5))).astype(np.float32)
    return features, (np.arange(rows) % 3).astype(np.int32)


def _nodes(
    binary: Path,
    directory: Path,
    count: int,
    first: int = 0,
    require_lease: bool = False,
) -> list[Server]:
    nodes = []
    for index in range(first, first + count):
        (directory / f"node-{index}").mkdir()
        server = Server(
            binary,
            directory / f"node-{index}",
            name=f"node-{index}",
            require_lease=require_lease,
        )
        nodes.append(server.start())
    return nodes


@pytest.fixture
def cluster(
    binaries: tuple[Path, Path], tmp_path: Path
) -> Iterator[tuple[Router, list[Server]]]:
    server_binary, _ = binaries
    nodes = _nodes(server_binary, tmp_path, 2)
    clients = (
        '[[clients]]\ntoken = "secret"\n\n'
        '[[clients]]\ntoken = "limited"\nper_second = 0.5\nburst = 2\n\n'
        '[[clients]]\ntoken = "clinic"\ntables = ["clinic-*"]\n'
    )
    router = Router(binaries[1], tmp_path, nodes, clients=clients)
    router.start(lambda: _connect(router).tables())
    yield router, nodes
    router.stop()
    for node in nodes:
        node.stop()


def _connect(router: Router, token: str = "secret") -> generalforecast.RemoteClient:
    return generalforecast.connect(f"grpc://127.0.0.1:{router.port}", token=token)


def _same(actual: np.ndarray, expected: np.ndarray) -> None:
    """Predictions from separately compiled programs, equal to float32 rounding."""
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def _direct(node: Server) -> generalforecast.RemoteClient:
    return generalforecast.connect(f"grpc://127.0.0.1:{node.port}", token="secret")


def test_tables_spread_over_nodes_and_route_consistently(cluster) -> None:
    router, nodes = cluster
    client = _connect(router)
    features, targets = _clusters()
    names = [f"table-{index}" for index in range(8)]
    for name in names:
        table = client.create_table(name, model="tiny", features=5, classes=3)
        table.insert(np.arange(32), features, targets)

    owners = {
        name: index
        for index, node in enumerate(nodes)
        for name in _direct(node).tables()
    }

    assert sorted(owners) == names
    assert set(owners.values()) == {0, 1}
    assert client.tables() == names
    for name in names[:3]:
        through_router = client.table(name).predict(NEAR_A)
        direct = _direct(nodes[owners[name]]).table(name).predict(NEAR_A)
        np.testing.assert_array_equal(through_router, direct)


def test_quotas_limit_each_token(cluster) -> None:
    router, _ = cluster
    limited = _connect(router, token="limited")

    limited.tables()
    limited.tables()
    with pytest.raises(pa.ArrowException, match="quota"):
        limited.tables()
    assert len([_connect(router).tables() for _ in range(5)]) == 5


def test_unknown_tokens_are_refused(cluster) -> None:
    router, _ = cluster

    with pytest.raises(PermissionError):
        _connect(router, token="nobody").tables()


def test_scoped_tokens_reach_only_their_tables(cluster) -> None:
    router, _ = cluster
    _connect(router).create_table("payments", model="tiny", features=5, classes=3)
    clinic = _connect(router, token="clinic")
    clinic.create_table("clinic-1", model="tiny", features=5, classes=3)

    assert clinic.tables() == ["clinic-1"]
    with pytest.raises(KeyError, match="payments"):
        clinic.table("payments")
    with pytest.raises(PermissionError, match="payments"):
        clinic.create_table("payments", model="tiny", features=5, classes=3)


def test_reloading_the_configuration_changes_the_tokens(cluster) -> None:
    router, nodes = cluster

    router.settings = router.settings.replace('token = "clinic"', 'token = "lab"')
    router.reload(nodes)

    wait_until(lambda: _connect(router, token="lab").tables() == [])
    with pytest.raises(PermissionError):
        _connect(router, token="clinic").tables()


def test_clients_cannot_reach_the_calls_routers_make(cluster) -> None:
    router, _ = cluster
    _connect(router).create_table("orders", model="tiny", features=5, classes=3)
    client = flight.FlightClient(f"grpc://127.0.0.1:{router.port}")
    options = flight.FlightCallOptions(headers=[(b"authorization", b"Bearer secret")])

    descriptor = flight.FlightDescriptor.for_path("replicate", "orders")
    writer, _ = client.do_put(descriptor, pa.schema([]), options)
    with pytest.raises(pa.ArrowException, match="unknown command"):
        writer.close()


def test_a_failed_node_affects_only_its_tables(cluster) -> None:
    router, nodes = cluster
    client = _connect(router)
    features, targets = _clusters()
    for index in range(6):
        table = client.create_table(
            f"table-{index}", model="tiny", features=5, classes=3
        )
        table.insert(np.arange(32), features, targets)
    lost = set(_direct(nodes[1]).tables())

    nodes[1].stop()
    wait_until(lambda: router.metric('gf_router_node_up{node="node-1"} 0'))
    assert "the node is down" in router.log.read_text()

    for index in range(6):
        name = f"table-{index}"
        if name in lost:
            with pytest.raises(pa.ArrowException):
                client.table(name).predict(NEAR_A)
        else:
            assert client.table(name).predict(NEAR_A).shape == (2, 3)


def test_mutual_tls_from_clients_to_router_to_nodes(binaries, tmp_path: Path) -> None:
    server_binary, router_binary = binaries
    paths = certificates(tmp_path)
    tls = (paths["server.pem"], paths["server.key"])
    nodes = []
    for index in range(2):
        (tmp_path / f"node-{index}").mkdir()
        node = Server(
            server_binary,
            tmp_path / f"node-{index}",
            name=f"node-{index}",
            tls=tls,
            client_ca=paths["authority.pem"],
            trust=paths["authority.pem"],
            client=(paths["client.pem"], paths["client.key"]),
        )
        nodes.append(node.start())
    router_tls = (
        f'[tls]\ncertificate = "{paths["server.pem"]}"\nkey = "{paths["server.key"]}"\n'
        f'client_ca = "{paths["authority.pem"]}"\n'
    )
    node_tls = (
        f'ca = "{paths["authority.pem"]}"\ncertificate = "{paths["client.pem"]}"\n'
        f'key = "{paths["client.key"]}"\n'
    )
    router = Router(router_binary, tmp_path, nodes, tls=router_tls, node_tls=node_tls)
    authority = paths["authority.pem"].read_bytes()

    def connect(certificate: bool) -> generalforecast.RemoteClient:
        return generalforecast.connect(
            f"grpc+tls://localhost:{router.port}",
            token="secret",
            tls_root_certs=authority,
            client_certificate=paths["client.pem"].read_bytes()
            if certificate
            else None,
            client_key=paths["client.key"].read_bytes() if certificate else None,
        )

    try:
        router.start(lambda: connect(True).tables())
        client = connect(True)
        features, targets = _clusters()
        table = client.create_table("secure", model="tiny", features=5, classes=3)
        table.insert(np.arange(32), features, targets)
        assert table.predict(NEAR_A).shape == (2, 3)
        with pytest.raises(pa.ArrowException):
            connect(False).tables()
    finally:
        router.stop()
        for node in nodes:
            node.stop()


@pytest.fixture
def replicated(
    binaries: tuple[Path, Path], tmp_path: Path
) -> Iterator[tuple[Router, list[Server]]]:
    nodes = _nodes(binaries[0], tmp_path, 3)
    router = Router(binaries[1], tmp_path, nodes, replicas=2)
    router.start(lambda: _connect(router).tables())
    yield router, nodes
    router.stop()
    for node in nodes:
        node.stop()


def _holders(nodes: list[Server]) -> dict[str, list[str]]:
    """The running nodes that hold each table."""
    holders: dict[str, list[str]] = {}
    for node in nodes:
        if node.process is not None:
            for name in _direct(node).tables():
                holders.setdefault(name, []).append(node.name)
    return {name: sorted(names) for name, names in holders.items()}


def test_replicas_hold_every_write(replicated) -> None:
    router, nodes = replicated
    client = _connect(router)
    features, targets = _clusters()
    names = [f"table-{index}" for index in range(4)]
    for name in names:
        table = client.create_table(name, model="tiny", features=5, classes=3)
        table.insert(np.arange(32), features, targets)
        table.update(np.arange(2), features[:2] + 1, targets[:2])
        table.delete(np.array([2]))
        table.refit()

    holders = _holders(nodes)
    by_name = {node.name: node for node in nodes}
    for name in names:
        assert holders[name] == sorted(router.locate(name))
        copies = [_direct(by_name[holder]).table(name) for holder in holders[name]]
        assert [len(copy) for copy in copies] == [31, 31]
        predictions = [copy.predict(NEAR_A) for copy in copies]
        _same(predictions[0], predictions[1])
        _same(client.table(name).predict(NEAR_A), predictions[0])


def test_writes_larger_than_a_message_reach_every_replica(replicated) -> None:
    router, nodes = replicated
    rows = 25_000
    features = np.random.default_rng(0).normal(size=(rows, 50)).astype(np.float32)
    table = _connect(router).create_table("large", model="tiny", features=50, classes=3)

    # About 5 MB in one log entry, past gRPC's default 4 MiB message.
    table.insert(np.arange(rows), features, np.arange(rows) % 3)

    by_name = {node.name: node for node in nodes}
    copies = [_direct(by_name[name]).table("large") for name in router.locate("large")]
    assert [len(copy) for copy in copies] == [rows, rows]


def test_dropped_tables_leave_every_copy_and_stay_gone(replicated) -> None:
    router, nodes = replicated
    client = _connect(router)
    features, targets = _clusters()
    table = client.create_table("orders", model="tiny", features=5, classes=3)
    table.insert(np.arange(32), features, targets)
    by_name = {node.name: node for node in nodes}
    bystander = next(node for node in nodes if node.name not in router.locate("orders"))

    bystander.stop()
    with pytest.raises(pa.ArrowException, match="every node up"):
        client.drop_table("orders")
    bystander.start()
    wait_until(lambda: client.drop_table("orders") is None)

    # Background checks would bring a surviving copy back to the replicas.
    time.sleep(1.0)
    assert "orders" not in _holders(nodes)
    assert client.tables() == []
    recreated = client.create_table("orders", model="tiny", features=5, classes=3)
    assert len(recreated) == 0
    assert all(
        len(_direct(by_name[name]).table("orders")) == 0
        for name in router.locate("orders")
    )


@pytest.fixture
def failing_over(
    binaries: tuple[Path, Path], tmp_path: Path
) -> Iterator[tuple[Router, list[Server]]]:
    nodes = _nodes(binaries[0], tmp_path, 3, require_lease=True)
    # Writes to a node that stopped answering give up sooner than by default.
    router = Router(
        binaries[1], tmp_path, nodes, replicas=3, failover=1.0, write_timeout=5.0
    )
    router.start(lambda: _connect(router).tables())
    yield router, nodes
    router.stop()
    for node in nodes:
        node.stop()


def _orders(router: Router) -> tuple[generalforecast.Table, list[Server]]:
    """A table of 32 rows once the router has leased its nodes, and its copies, primary first."""
    client = _connect(router)
    features, targets = _clusters()
    wait_until(
        lambda: (
            client.create_table("orders", model="tiny", features=5, classes=3)
            is not None
        )
    )
    table = client.table("orders")
    table.insert(np.arange(32), features, targets)
    return table, router.copies("orders")


def test_writes_fail_over_once_the_primary_is_lost(failing_over) -> None:
    router, _ = failing_over
    table, (primary, replica, _) = _orders(router)
    features, targets = _clusters(40)

    primary.stop()
    wait_until(
        lambda: table.insert(np.arange(32, 40), features[32:40], targets[32:40]) is None
    )

    assert router.locate("orders")[0] == replica.name
    assert router.metric("gf_router_promotions_total 1")
    # The old primary comes back behind, not diverged, and catches up.
    primary.start()
    wait_until(lambda: len(_direct(primary).table("orders")) == 40)
    _same(_direct(primary).table("orders").predict(NEAR_A), table.predict(NEAR_A))


def test_a_primary_cut_off_while_running_is_fenced_then_catches_up(
    failing_over,
) -> None:
    router, _ = failing_over
    table, (primary, replica, _) = _orders(router)
    features, targets = _clusters(40)
    assert primary.process is not None

    # Paused, the primary holds its connections but answers nothing, as across
    # a partition; it keeps its state, and writes may wait in its sockets.
    primary.process.send_signal(signal.SIGSTOP)
    try:
        # A write larger than a connection's window stalls partway, and gives
        # up after the write timeout rather than the call's.
        many = np.tile(features, (1_000, 1))
        started = time.monotonic()
        with pytest.raises(pa.ArrowException, match="no progress"):
            table.upsert(np.arange(len(many)), many, np.tile(targets, 1_000))
        assert time.monotonic() - started < 60
        wait_until(
            lambda: (
                table.insert(np.arange(32, 40), features[32:40], targets[32:40]) is None
            ),
            # The write it waits on must first find the new primary.
            timeout=2 * SETTLE_SECONDS,
        )
    finally:
        primary.process.send_signal(signal.SIGCONT)

    assert router.locate("orders")[0] == replica.name
    # Its lease lapsed while it was cut off, so whatever it read on waking took
    # no write of its own: it comes back behind, not diverged, and catches up.
    wait_until(lambda: len(_direct(primary).table("orders")) == 40)
    _same(_direct(primary).table("orders").predict(NEAR_A), table.predict(NEAR_A))


def test_a_second_router_that_fails_over_stands_down(
    binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    nodes = _nodes(binaries[0], tmp_path, 3, require_lease=True)
    routers = [
        Router(
            binaries[1],
            tmp_path,
            nodes,
            replicas=3,
            failover=1.0,
            write_timeout=5.0,
            name=name,
        )
        for name in ("first", "second")
    ]
    first, second = routers
    try:
        first.start(lambda: _connect(first).tables())
        table, (primary, replica, _) = _orders(first)
        second.start(lambda: _connect(second).tables())

        # Nodes renew a lease only for the router holding it, so the second
        # router finds the first's and leaves failing over to it.
        wait_until(lambda: "only one router may set" in second.log.read_text())
        features, targets = _clusters(40)
        primary.stop()
        wait_until(
            lambda: (
                table.insert(np.arange(32, 40), features[32:40], targets[32:40]) is None
            )
        )
        wait_until(lambda: second.locate("orders")[0] == replica.name)
        assert len(_connect(second).table("orders")) == 40

        # Once the first router is gone, the second takes over failing over.
        first.stop()
        wait_until(lambda: "this router takes over" in second.log.read_text())
        primary.start()
        wait_until(lambda: len(_direct(primary).table("orders")) == 40)
        mirror = _connect(second).table("orders")
        replica.stop()
        wait_until(
            lambda: (
                mirror.upsert(np.arange(32, 40), features[32:40], targets[32:40])
                is None
            )
        )
        assert second.locate("orders")[0] != replica.name
    finally:
        for router in routers:
            router.stop()
        for node in nodes:
            node.stop()


def test_a_replica_missing_acknowledged_writes_is_never_promoted(
    failing_over,
) -> None:
    router, _ = failing_over
    table, (primary, behind, holder) = _orders(router)
    features, targets = _clusters(44)
    behind.stop()
    table.insert(np.arange(32, 40), features[32:40], targets[32:40])

    # Only the replica that missed the write is up, so even a router that
    # never saw the write cannot tell it is behind, and promotes nothing.
    holder.stop()
    primary.stop()
    behind.start()
    router.stop()
    router.start(lambda: _connect(router).tables())
    time.sleep(3.0)
    with pytest.raises(pa.ArrowException, match="unavailable"):
        table.insert(np.arange(40, 44), features[40:44], targets[40:44])

    holder.start()
    wait_until(
        lambda: table.insert(np.arange(40, 44), features[40:44], targets[40:44]) is None
    )
    assert router.locate("orders")[0] == holder.name
    assert len(table) == 44


def test_only_unacknowledged_writes_are_set_aside(failing_over) -> None:
    router, _ = failing_over
    table, (primary, *replicas) = _orders(router)
    features, targets = _clusters(40)
    for replica in replicas:
        replica.stop()
    with pytest.raises(pa.ArrowException, match="not acknowledged"):
        table.insert(np.arange(32, 40), features[32:40], targets[32:40])

    # A restarted router never saw that write, so it promotes a replica; the
    # old primary's copy then diverges, and is set aside and copied again.
    primary.stop()
    for replica in replicas:
        replica.start()
    router.stop()
    router.start(lambda: _connect(router).tables())
    wait_until(
        lambda: table.insert(np.arange(32, 36), features[32:36], targets[32:36]) is None
    )
    primary.start()
    wait_until(lambda: len(_direct(primary).table("orders")) == 36)
    # Its first write named it primary under term 1.
    assert (primary.tables / "orders.log.diverged-1").exists()


def test_a_write_no_replica_took_does_not_hold_up_failover(
    binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    nodes = _nodes(binaries[0], tmp_path, 2, require_lease=True)
    router = Router(binaries[1], tmp_path, nodes, replicas=2, failover=1.0)
    try:
        router.start(lambda: _connect(router).tables())
        table, (primary, replica) = _orders(router)
        features, targets = _clusters(40)
        replica.stop()
        with pytest.raises(pa.ArrowException, match="not acknowledged"):
            table.insert(np.arange(32, 40), features[32:40], targets[32:40])

        primary.stop()
        replica.start()

        wait_until(lambda: router.locate("orders")[0] == replica.name)
    finally:
        router.stop()
        for node in nodes:
            node.stop()


def test_a_primary_leaving_the_configuration_holds_writes_until_it_returns(
    replicated,
) -> None:
    router, nodes = replicated
    client = _connect(router)
    features, targets = _clusters(40)
    table = client.create_table("orders", model="tiny", features=5, classes=3)
    table.insert(np.arange(32), features[:32], targets[:32])
    primary = next(node for node in nodes if node.name == router.locate("orders")[0])

    def refused() -> bool:
        try:
            table.upsert(np.arange(32, 40), features[32:40], targets[32:40])
        except pa.ArrowException as error:
            return "left the configuration" in str(error)
        return False

    # Its replica holds every acknowledged write, but writes there under the
    # same term could merge with ones the primary took unacknowledged.
    router.reload([node for node in nodes if node is not primary])
    wait_until(refused)
    router.reload(nodes)
    wait_until(
        lambda: table.upsert(np.arange(32, 40), features[32:40], targets[32:40]) is None
    )
    assert len(table) == 40
    assert router.locate("orders")[0] == primary.name


def test_replicas_catch_up_and_serve_when_the_primary_fails(replicated) -> None:
    router, nodes = replicated
    client = _connect(router)
    features, targets = _clusters(44)
    table = client.create_table("orders", model="tiny", features=5, classes=3)
    table.insert(np.arange(32), features[:32], targets[:32])
    by_name = {node.name: node for node in nodes}
    primary, replica = (by_name[name] for name in router.locate("orders"))

    # With its replica down, a write reaches the primary but is not acknowledged.
    replica.stop()
    with pytest.raises(pa.ArrowException, match="not acknowledged"):
        table.insert(np.arange(32, 40), features[32:40], targets[32:40])
    replica.start()
    wait_until(lambda: len(_direct(replica).table("orders")) == 40)

    primary.stop()
    expected = _direct(replica).table("orders").predict(NEAR_A)
    np.testing.assert_array_equal(client.table("orders").predict(NEAR_A), expected)
    # Without a router that fails tables over, writes wait for the primary.
    with pytest.raises(pa.ArrowException, match="unavailable"):
        table.insert(np.arange(40, 44), features[40:44], targets[40:44])


def test_adding_a_node_moves_the_tables_it_now_serves(binaries, tmp_path) -> None:
    nodes = _nodes(binaries[0], tmp_path, 2)
    router = Router(binaries[1], tmp_path, nodes)
    try:
        router.start(lambda: _connect(router).tables())
        client = _connect(router)
        features, targets = _clusters()
        names = [f"table-{index}" for index in range(8)]
        for name in names:
            table = client.create_table(name, model="tiny", features=5, classes=3)
            table.insert(np.arange(32), features, targets)
        expected = {name: client.table(name).predict(NEAR_A) for name in names}

        nodes += _nodes(binaries[0], tmp_path, 1, first=2)
        router.reload(nodes)
        # A table hands its writes to the new node once it has caught up, and
        # its old copy goes once the new one holds every write.
        wait_until(
            lambda: (
                any(router.locate(name) == ["node-2"] for name in names)
                and _holders(nodes) == {name: router.locate(name) for name in names}
            )
        )
        assert not router.metric("gf_router_moves_total 0")
        for name in names:
            _same(client.table(name).predict(NEAR_A), expected[name])
    finally:
        router.stop()
        for node in nodes:
            node.stop()


def test_a_replica_down_during_compaction_catches_up_from_the_snapshot(
    replicated,
) -> None:
    router, nodes = replicated
    client = _connect(router)
    features, targets = _clusters(48)
    table = client.create_table("orders", model="tiny", features=5, classes=3)
    table.insert(np.arange(48), features, targets)
    by_name = {node.name: node for node in nodes}
    primary, replica = (by_name[name] for name in router.locate("orders"))

    replica.stop()
    with pytest.raises(pa.ArrowException, match="not acknowledged"):
        table.delete(np.arange(40, 48))
    table.compact()
    replica.start()

    wait_until(lambda: len(_direct(replica).table("orders")) == 40)
    _same(
        _direct(replica).table("orders").predict(NEAR_A),
        _direct(primary).table("orders").predict(NEAR_A),
    )


@pytest.mark.parametrize(
    "strategy",
    [
        generalforecast.Strategy.exact(),
        generalforecast.Strategy.subsample(16, 0),
        generalforecast.Strategy.neighbors(8),
        generalforecast.Strategy.retrieved(4, core=8, kind="stratified"),
        generalforecast.Strategy.adaptive(),
    ],
    ids=["exact", "subsample", "neighbors", "retrieved", "adaptive"],
)
def test_a_table_split_into_partitions_serves_as_one(
    cluster: tuple[Router, list[Server]], strategy: generalforecast.Strategy
) -> None:
    router, _ = cluster
    client = _connect(router)
    features, targets = _clusters()
    whole = client.create_table(
        "whole", model="tiny", features=5, classes=3, strategy=strategy, page_size=4
    )
    adaptive = strategy.spec() == generalforecast.Strategy.adaptive().spec()
    split = client.create_table(
        "split",
        model="tiny",
        features=5,
        classes=3,
        strategy=strategy,
        page_size=4,
        partitions=4,
        # An adaptive strategy stays within the model's cap, so the clamp
        # that a gathered context otherwise refuses has nothing to cut.
        past_cap="clamp" if adaptive else "warn",
    )
    for table in (whole, split):
        table.insert(np.arange(32), features, targets)
    if adaptive:
        # The tiny model has no cap: 32 rows over the partitions read whole.
        described = split.describe()
        assert described.strategy == strategy
        assert described.resolved_strategy == generalforecast.Strategy.exact()
        # Under the clamp, only an adaptive strategy gathers.
        with pytest.raises(ValueError, match="clamp"):
            split.set_strategy(generalforecast.Strategy.neighbors(4))
        assert split.describe().strategy == strategy
    # Every partition holds some of the rows, and none holds them all.
    held = [
        client._action("describe", {"table": f"split--part-{index}-of-4"})[0]["rows"]
        for index in range(4)
    ]

    predicted = split.predict(NEAR_A)

    assert predicted.shape == (len(NEAR_A), 3)
    np.testing.assert_allclose(predicted.sum(axis=1), 1.0, atol=1e-5)
    if adaptive or strategy.spec() == generalforecast.Strategy.exact().spec():
        # Every row, gathered from every partition, as the whole table holds them.
        np.testing.assert_allclose(predicted, whole.predict(NEAR_A), atol=1e-4)
    assert split.name == "split"
    assert sum(held) == 32 and max(held) < 32, held
    assert sorted(client.tables()) == ["split", "whole"]
    assert len(split) == 32
    split.delete([0, 1])
    assert len(split) == 30
    client.drop_table("split")
    assert client.tables() == ["whole"]
    # Made again with more classes over the same rows, it predicts them all,
    # from tables of its own rather than the first one's.
    again = client.create_table(
        "split",
        model="tiny",
        features=5,
        classes=4,
        strategy=strategy,
        page_size=4,
        partitions=4,
    )
    again.insert(np.arange(32), features, targets)
    assert again.predict(NEAR_A).shape == (len(NEAR_A), 4)


def test_a_partitioned_tables_cutoffs_read_rows_from_before_them(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, nodes = cluster
    client = _connect(router)
    features, targets = _clusters()
    tables = {}
    for name, partitions in (("whole", 1), ("split", 4)):
        tables[name] = client.create_table(
            name, model="tiny", features=5, classes=3, partitions=partitions
        )
        tables[name].insert(np.arange(32), features, targets, times=np.arange(32))

    # Two cutoffs in one request: each gathers its own rows from before it.
    split = tables["split"].predict(NEAR_A, before=[16, 24])

    for query, cutoff in enumerate((16, 24)):
        whole = tables["whole"].predict(NEAR_A[query : query + 1], before=cutoff)
        np.testing.assert_allclose(split[query : query + 1], whole, atol=1e-4)
    with pytest.raises(ValueError, match="precedes every row"):
        tables["split"].predict(NEAR_A, before=0)

    # The answer carries the worker time of both cutoffs' predictions, and no
    # epoch, as no one state answered, with cutoffs or without.
    spent = _gpu_seconds(nodes, "split")
    tables["split"].predict(NEAR_A, before=[16, 24])
    answer = tables["split"].last_answer
    assert answer is not None and answer.epoch is None
    assert answer.gpu_seconds == pytest.approx(
        _gpu_seconds(nodes, "split") - spent, abs=1e-6
    )
    tables["split"].predict(NEAR_A)
    split, whole = tables["split"].last_answer, tables["whole"].last_answer
    assert split is not None and split.epoch is None
    assert whole is not None and whole.epoch is not None
    with pytest.raises(ValueError, match="gathered for each prediction"):
        tables["split"].contexts(NEAR_A)
    with pytest.raises(ValueError, match="has no regions"):
        tables["split"].region_churn(NEAR_A)
    assert np.isnan(tables["whole"].region_churn(NEAR_A)).all()
    # Each partition would cut its context to the model's length on its own.
    with pytest.raises(ValueError, match="one partition"):
        client.create_table(
            "clamped",
            model="tiny",
            features=5,
            classes=3,
            partitions=4,
            past_cap="clamp",
        )
    with pytest.raises(ValueError, match="one partition"):
        client.create_table(
            "regioned", model="tiny", features=5, classes=3, partitions=4, region_rows=8
        )
    with pytest.raises(ValueError, match="one partition"):
        client.create_table(
            "recent",
            model="tiny",
            features=5,
            classes=3,
            partitions=4,
            strategy=generalforecast.Strategy.retrieved(4, last=16),
        )


def test_a_partitioned_tables_retrieval_columns_choose_its_rows(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    client = _connect(router)
    features, targets = _clusters()
    columns = {**{f"f{index}": "numeric" for index in range(5)}, "r": "retrieval"}
    rows = np.column_stack([features, np.arange(32)])
    split = client.create_table(
        "placed",
        model="tiny",
        columns=columns,
        classes=3,
        strategy=generalforecast.Strategy.neighbors(4),
        partitions=4,
    )
    split.insert(np.arange(32), rows, targets)

    # Each query reads the 4 rows nearest its retrieval value, whatever its
    # features, as a table of those rows alone reads them.
    for value, nearest in ((0.4, [0, 1, 2, 3]), (27.6, [26, 27, 28, 29])):
        alone = client.create_table(
            f"alone{nearest[0]}", model="tiny", columns=columns, classes=3
        )
        alone.insert(nearest, rows[nearest], targets[nearest])
        query = np.column_stack([features[:1], [value]])
        np.testing.assert_allclose(
            split.predict(query), alone.predict(query), atol=1e-4
        )
    assert split.columns[-1] == generalforecast.Column("r", "retrieval")


def test_a_partitioned_tables_rows_join_contexts_once_labeled(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    client = _connect(router)
    features, targets = _clusters()
    # Rows at A itself, of a class A's other rows do not hold, without labels.
    at_a = np.zeros((8, 5), np.float32)
    split = client.create_table(
        "late",
        model="tiny",
        features=5,
        classes=3,
        strategy=generalforecast.Strategy.neighbors(4),
        partitions=4,
    )
    split.insert(np.arange(32), features, targets)
    split.insert(np.arange(32, 40), at_a, np.full(8, -1, np.int32))
    before = split.predict(NEAR_A)

    split.label(np.arange(32, 40), np.full(8, 2, np.int32))

    assert not np.allclose(split.predict(NEAR_A), before, atol=1e-3)


def test_a_partitioned_table_refuses_given_rows(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    split = _connect(router).create_table(
        "split", model="tiny", features=5, classes=3, partitions=4
    )
    split.insert(np.arange(32), *_clusters())

    with pytest.raises(ValueError, match="table of one partition"):
        split.predict_given(NEAR_A, [[1, 2], [3, 4]])


def test_a_routed_tables_partitions_answer_their_own_queries(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    client = _connect(router)
    # Even rows sit near 0 in f0 and odd rows near 10, with targets of their own.
    features, _ = _clusters()
    targets = np.where(np.arange(32) % 2 == 0, np.arange(32) % 3, 2 - np.arange(32) % 3)
    targets = targets.astype(np.int32)
    route = {"column": "f0", "kind": "ordered", "bounds": [5.0]}
    routed = client.create_table(
        "routed", model="tiny", features=5, classes=3, partitions=2, route_by=route
    )
    low = client.create_table("low", model="tiny", features=5, classes=3)
    high = client.create_table("high", model="tiny", features=5, classes=3)
    routed.insert(np.arange(32), features, targets)
    low.insert(np.arange(0, 32, 2), features[::2], targets[::2])
    high.insert(np.arange(1, 32, 2), features[1::2], targets[1::2])

    # Each partition holds the rows on its side of the bound, and the table
    # describes its route.
    held = [
        client._action("describe", {"table": f"routed--part-{index}-of-2"})[0]["rows"]
        for index in range(2)
    ]
    assert held == [16, 16]
    assert client._action("describe", {"table": "routed"})[0]["route_by"] == route
    # Each query is answered by the partition on its side, from its rows
    # alone, and the answers come back in the queries' order.
    near_high = NEAR_A + 10
    queries = np.concatenate([NEAR_A[:1], near_high, NEAR_A[1:]])
    predicted = routed.predict(queries)
    expected = np.concatenate(
        [low.predict(NEAR_A[:1]), high.predict(near_high), low.predict(NEAR_A[1:])]
    )
    np.testing.assert_allclose(predicted, expected, atol=1e-4)
    assert routed.last_answer is not None and routed.last_answer.epoch is None
    assert not np.allclose(low.predict(NEAR_A), high.predict(NEAR_A), atol=1e-3)
    # A query with a cutoff reads the partition's rows before it, as a whole
    # table's would.
    np.testing.assert_allclose(
        routed.predict(near_high, before=[1, 1]),
        high.predict(near_high, before=1),
        atol=1e-4,
    )
    # A delete carries keys alone, so it reaches whichever partition holds
    # them, and a key no partition holds is refused as one table refuses it.
    routed.delete([0, 1])
    assert len(routed) == 30
    with pytest.raises(KeyError, match="do not exist"):
        routed.delete([0, 2])
    assert len(routed) == 29
    routed.delete([0, 1], missing_ok=True)
    with pytest.raises(ValueError, match="through update"):
        routed.label([3], [0])
    # A routed table's partitions answer alone, so each may have regions.
    client.create_table(
        "regioned",
        model="tiny",
        features=5,
        classes=3,
        partitions=2,
        route_by=route,
        region_rows=8,
    )
    with pytest.raises(ValueError, match="cut 2 partitions, not 4"):
        client.create_table(
            "uncut", model="tiny", features=5, classes=3, partitions=4, route_by=route
        )
    with pytest.raises(ValueError, match="is not a column"):
        client.create_table(
            "unrouted",
            model="tiny",
            features=5,
            classes=3,
            partitions=2,
            route_by={"column": "f9"},
        )


def test_a_hashed_route_spreads_rows_and_answers_in_query_order(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    client = _connect(router)
    routed = client.create_table(
        "hashed",
        model="tiny",
        features=5,
        classes=3,
        partitions=4,
        route_by={"column": "f0"},
    )
    routed.insert(np.arange(32), *_clusters())

    held = [
        client._action("describe", {"table": f"hashed--part-{index}-of-4"})[0]["rows"]
        for index in range(4)
    ]
    assert sum(held) == 32 and max(held) < 32, held
    assert len(routed) == 32
    queries = np.random.default_rng(1).normal(size=(6, 5)).astype(np.float32)
    predicted = routed.predict(queries)
    assert predicted.shape == (6, 3)
    for index in range(6):
        # Each query alone is answered as it is among the others.
        alone = routed.predict(queries[index : index + 1])
        np.testing.assert_allclose(predicted[index : index + 1], alone, atol=1e-5)


def test_a_routed_rows_route_value_cannot_change(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    client = _connect(router)
    features, targets = _clusters()
    route = {"column": "f0", "kind": "ordered", "bounds": [5.0]}
    routed = client.create_table(
        "fixed", model="tiny", features=5, classes=3, partitions=2, route_by=route
    )
    routed.insert(np.arange(32), features, targets)

    # Row 0 sits below the bound. An update that moves it over the bound
    # reaches the partition without the row, which says why it is refused.
    moved = features[:1] + 10
    with pytest.raises(KeyError, match="cannot change"):
        routed.update([0], moved, targets[:1])
    routed.update([0], features[:1], [2])
    assert len(routed) == 32
    # The router does not know which partition holds a key, so an insert
    # with a new value is not refused: it makes a second row for the key,
    # which a delete of the key then removes with the first.
    routed.insert([0], moved, targets[:1])
    assert len(routed) == 33
    routed.delete([0])
    assert len(routed) == 31


def test_a_routed_delete_retried_with_its_id_is_answered_as_it_was(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    client = _connect(router)
    routed = client.create_table(
        "retried",
        model="tiny",
        features=5,
        classes=3,
        partitions=2,
        route_by={"column": "f0"},
    )
    routed.insert(np.arange(32), *_clusters())
    batch = pa.RecordBatch.from_pydict({"key": generalforecast.keys_of([0, 1]).arrow()})
    descriptor = flight.FlightDescriptor.for_path("delete", "retried")
    options = client._write_options("retry-1")

    def delete() -> None:
        writer, _ = client._flight.do_put(descriptor, batch.schema, options)
        writer.write_batch(batch)
        writer.close()

    delete()
    assert len(routed) == 30
    # Each partition answers the retry as it answered the first delete, so the
    # keys count as held although no partition holds them any more.
    delete()
    assert len(routed) == 30
    with pytest.raises(KeyError, match="do not exist"):
        routed.delete([0, 1])


def test_a_routed_table_needs_a_number_in_its_column_for_every_row(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    client = _connect(router)
    routed = client.create_table(
        "numbered",
        model="tiny",
        features=5,
        classes=3,
        partitions=2,
        route_by={"column": "f0"},
    )
    features, targets = _clusters()
    routed.insert(np.arange(32), features, targets)
    queries = NEAR_A.copy()
    queries[1, 0] = np.nan
    with pytest.raises(ValueError, match="row 1 has none"):
        routed.predict(queries)
    with pytest.raises(ValueError, match="row 0 has none"):
        routed.insert([32], queries[1:2], targets[:1])
    assert len(routed) == 32


def test_a_routed_table_of_one_partition_is_one_table(
    cluster: tuple[Router, list[Server]],
) -> None:
    router, _ = cluster
    client = _connect(router)
    features, targets = _clusters()
    one = client.create_table(
        "one",
        model="tiny",
        features=5,
        classes=3,
        partitions=1,
        route_by={"column": "f0"},
    )
    plain = client.create_table("unsplit", model="tiny", features=5, classes=3)
    one.insert(np.arange(32), features, targets)
    plain.insert(np.arange(32), features, targets)
    described = client._action("describe", {"table": "one"})[0]
    assert described["route_by"] == {"column": "f0", "kind": "hashed"}
    assert "partitions" not in described
    assert len(one) == 32
    _same(one.predict(NEAR_A), plain.predict(NEAR_A))
    # An ordered route's bounds must cut as many partitions as the table has.
    for partitions, bounds in [(1, [5.0]), (2, [1.0, 2.0, 3.0])]:
        with pytest.raises(
            ValueError, match=f"cut {len(bounds) + 1} partitions, not {partitions}"
        ):
            client.create_table(
                "miscut",
                model="tiny",
                features=5,
                classes=3,
                partitions=partitions,
                route_by={"column": "f0", "kind": "ordered", "bounds": bounds},
            )


def test_a_series_tables_given_groups_pass_through_the_router(
    cluster: tuple[Router, list[Server]],
) -> None:

    settings: dict[str, Any] = {
        "series": "item",
        "time": "t",
        "target": "units",
        "step": 10,
        "group": generalforecast.GroupStrategy.given(2),
        "horizon": 5,
    }
    router, _ = cluster
    routed = _connect(router).create_series_table(
        "given", model="tiny-forecaster", **settings
    )
    local = generalforecast.local().create_series_table(
        "given", model="tests/timestral-forecaster", **settings
    )
    generator = np.random.default_rng(0)
    times = list(range(100, 200, 10))
    frame = {
        "item": ["shoes"] * 10 + ["socks"] * 10,
        "t": times + times,
        "units": generator.normal(size=20).astype(np.float32),
    }
    routed.insert(frame)
    local.insert(frame)

    # The forecast exchange's group column reaches the node unchanged.
    np.testing.assert_allclose(
        routed.forecast_given(["shoes"], [["shoes", "socks"]], before=200),
        local.forecast_given(["shoes"], [["shoes", "socks"]], before=200),
        rtol=1e-4,
        atol=1e-4,
    )
    with pytest.raises(ValueError, match="takes each series' group"):
        routed.forecast(["shoes"], before=200)


def _gpu_seconds(nodes: list[Server], table: str) -> float:
    """Worker time the nodes spent on a table's partitions."""
    total = 0.0
    for node in nodes:
        _, text = node.http("/metrics")
        for line in text.splitlines():
            if line.startswith(f'gf_table_gpu_seconds_total{{table="{table}'):
                total += float(line.split()[1])
    return total
