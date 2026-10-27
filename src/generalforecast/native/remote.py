# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Flight table client, for a `gf-server`, a `gf-router` or this process's engine."""

import asyncio
import contextlib
import functools
import ipaddress
import json
import re
import threading
import urllib.parse
import uuid
import warnings
from collections.abc import Callable, Coroutine, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Concatenate, Self, override

import numpy as np

from generalforecast._core import DEFAULT_FIT_ROWS, GroupStrategy, Strategy
from generalforecast.columns import (
    Column,
    arrow_columns,
    columns_of,
    prepare,
)
from generalforecast.keys import Key, Keys, key_of_spec, keys_of, spec_of_key
from generalforecast.native.handles import (
    _REBUILD_WAIT,
    Answer,
    Definition,
    Described,
    DuplicateKey,
    PastCap,
    Rebuild,
    RebuildWait,
    RefitCutoffs,
    RetrievalCategoricals,
    Roles,
    SeriesDefinition,
    SeriesTable,
    Summary,
    Table,
    Times,
    WriteKind,
    _bucket,
    _check_classes,
    _check_times,
    _duration,
    _given,
    _levels,
    _rebuild_spec,
    _series_definition_of,
    _times,
)


@contextlib.contextmanager
def _errors() -> Iterator[None]:
    """Maps refused tokens onto `PermissionError`.

    pyarrow raises the rest by their gRPC status already: `NOT_FOUND` as a
    `KeyError`, and `INVALID_ARGUMENT`, `FAILED_PRECONDITION` and, as servers
    mark it for Arrow clients, `ALREADY_EXISTS` as a `ValueError`.
    """
    from pyarrow import flight

    try:
        yield
    except (flight.FlightUnauthenticatedError, flight.FlightUnauthorizedError) as error:
        raise PermissionError(str(error)) from error


# How a server refuses a write adding a key it holds: gf-core's
# `TableError::DuplicateKey`, which leaves the table unchanged.
_DUPLICATE = re.compile(r"row key .+? already exists")


@contextlib.contextmanager
def _duplicates() -> Iterator[None]:
    """Raises a write refused for a key the table holds as `DuplicateKey`."""
    try:
        yield
    except ValueError as error:
        found = _DUPLICATE.search(str(error))
        if isinstance(error, DuplicateKey) or found is None:
            raise
        raise DuplicateKey(found.group(0)) from error


# Bytes of rows per Flight message, well under the servers' 64 MiB limit.
_BATCH_BYTES = 16 << 20


def _retry_once(call: Callable[[], Any]) -> Any:
    """Calls once more if the server was unavailable, as writes with ids allow."""
    from pyarrow import flight

    try:
        return call()
    except flight.FlightUnavailableError:
        return call()


def _series_key_column(digests: np.ndarray, given: np.ndarray) -> Any:
    """Each observation's series key as the server keeps it, dictionary-encoded.

    The server reads each series' key once by its digest, so the column
    carries each distinct key's JSON once and an index per observation.
    """
    import pyarrow as pa

    _, first, indices = np.unique(digests, return_index=True, return_inverse=True)
    specs = [json.dumps(spec_of_key(key)) for key in given[first].tolist()]
    return pa.DictionaryArray.from_arrays(
        pa.array(indices.astype(np.int32), type=pa.int32()),
        pa.array(specs, type=pa.string()),
    )


def _split(batch: Any) -> list[Any]:
    """A record batch as batches of at most `_BATCH_BYTES`, for one message each."""
    row_bytes = max(1, batch.nbytes // max(1, batch.num_rows))
    rows = max(1, _BATCH_BYTES // row_bytes)
    starts = range(0, max(batch.num_rows, 1), rows)
    return [batch.slice(start, rows) for start in starts]


def _put(
    client: "RemoteClient",
    kind: str,
    name: str,
    batch: Any,
    naming: contextlib.AbstractContextManager[None],
    missing_ok: bool = False,
) -> None:
    """Sends one write of a table's batch under an id of its own, retried once.

    `naming` re-raises the server's errors naming keys as the caller gave
    them; a write refused for a key the table holds raises `DuplicateKey`.
    """
    from pyarrow import flight

    descriptor = flight.FlightDescriptor.for_path(kind, client._wire(name))
    options = client._write_options(uuid.uuid4().hex, missing_ok)

    def put() -> None:
        writer, _ = client._flight.do_put(descriptor, batch.schema, options)
        for part in _split(batch):
            writer.write_batch(part)
        writer.close()

    with _duplicates(), naming, _errors():
        _retry_once(put)


def _answer_of(result: Any) -> Answer:
    """What answered a table's predictions or forecasts, from their reply's metadata."""
    metadata = result.schema.metadata or {}
    epoch = metadata.get(b"generalforecast.epoch")
    return Answer(
        epoch=None if epoch is None else int(epoch),
        gpu_seconds=float(metadata.get(b"generalforecast.gpu_seconds", 0.0)),
    )


class RemoteTable(Table):
    """A table on a server."""

    def __init__(self, client: "RemoteClient", definition: Definition) -> None:
        """Wraps a table on the client's server."""
        super().__init__(definition)
        self._client = client

    def _write(
        self,
        kind: WriteKind,
        keys: Iterable[Key],
        features: Any,
        targets: Any,
        times: Any,
    ) -> None:
        import pyarrow as pa

        wide = keys_of(keys)
        if not len(wide):
            return
        prepared, _ = prepare(features, self.columns)
        columns = {"key": wide.arrow()}
        columns.update(arrow_columns(prepared, self.columns))
        columns["target"] = pa.array(self._targets(targets))
        if (stamps := _times(times, len(wide), self.times)) is not None:
            columns["time"] = pa.array(stamps)
        self._put(kind, wide, columns)

    def _put(
        self, kind: str, wide: Any, columns: dict[str, Any], missing_ok: bool = False
    ) -> None:
        """Sends one write of these columns; see `_put`."""
        import pyarrow as pa

        batch = pa.RecordBatch.from_pydict(columns)
        _put(self._client, kind, self.name, batch, wide.naming(), missing_ok)

    @override
    def insert(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._write("insert", keys, features, targets, times)

    @override
    def update(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._write("update", keys, features, targets, times)

    @override
    def upsert(
        self, keys: Iterable[Key], features: Any, targets: Any, times: Any = None
    ) -> None:
        self._write("upsert", keys, features, targets, times)

    @override
    def label(self, keys: Iterable[Key], targets: Any) -> None:
        import pyarrow as pa

        wide = keys_of(keys)
        if not len(wide):
            return
        columns = {"key": wide.arrow(), "target": pa.array(self._targets(targets))}
        self._put("label", wide, columns)

    @override
    def delete(self, keys: Iterable[Key], missing_ok: bool = False) -> None:
        wide = keys_of(keys)
        if not len(wide):
            return
        self._put("delete", wide, {"key": wide.arrow()}, missing_ok)

    @override
    def truncate(self) -> None:
        self._client._action("truncate", {"table": self.name})

    @override
    def predict(
        self,
        queries: Any,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        return self._client._predict(
            self,
            queries,
            before,
            _levels(self.name, self.definition.quantile_levels, quantiles),
        )

    @override
    def predict_given(
        self,
        queries: Any,
        contexts: Sequence[Iterable[Key]],
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        levels = _levels(self.name, self.definition.quantile_levels, quantiles)
        return self._client._predict(self, queries, None, levels, given=contexts)

    @override
    def contexts(self, queries: Any, before: Any = None) -> list[np.ndarray]:
        return self._client._contexts(self, queries, before)

    @override
    def region_churn(self, queries: Any) -> np.ndarray:
        return self._client._region_churn(self, queries)

    @override
    def vocabularies(self) -> dict[str, list[str]]:
        return self._client._describe(self.name).vocabularies

    @override
    @property
    def labels(self) -> list[str] | None:
        return self._client._describe(self.name, words=False).labels

    @override
    def describe(self) -> Described:
        return self._client._describe(self.name)

    @override
    def refit(self) -> None:
        self._client._action("refit", {"table": self.name})

    @override
    def set_strategy(self, strategy: Strategy) -> None:
        self._client._action(
            "set_strategy",
            {"table": self.name, "strategy": json.loads(strategy.spec())},
        )

    @override
    def compact(self) -> None:
        self._client._action("compact", {"table": self.name})

    @override
    def __len__(self) -> int:
        return self._client._describe(self.name, words=False).rows


class RemoteSeriesTable(SeriesTable):
    """A table of series on a server."""

    def __init__(self, client: "RemoteClient", definition: SeriesDefinition) -> None:
        """Wraps a table on the client's server."""
        super().__init__(definition)
        self._client = client

    def _put(self, kind: str, batch: Any, missing_ok: bool = False) -> None:
        """Sends one write of a batch; see `_put`."""
        _put(self._client, kind, self.name, batch, self._naming(), missing_ok)

    def _change(self, kind: str, rows: Any) -> None:
        import pyarrow as pa

        stamps, keys, given = self._placed(rows)
        if not len(keys):
            return
        prepared, _ = prepare(rows, (*self.columns, Column(self.target)))
        columns: dict[str, Any] = {"key": Keys(keys, keys[:, 1]).arrow()}
        columns.update(arrow_columns(prepared[:-1], self.columns))
        columns["target"] = pa.array(
            np.ascontiguousarray(prepared[-1], dtype=np.float32), type=pa.float32()
        )
        columns["time"] = pa.array(stamps)
        columns["series_key"] = _series_key_column(keys[:, 1], given)
        self._put(kind, pa.RecordBatch.from_pydict(columns))

    @override
    def insert(self, rows: Any) -> None:
        self._change("insert", rows)

    @override
    def upsert(self, rows: Any) -> None:
        self._change("upsert", rows)

    @override
    def delete(self, rows: Any, missing_ok: bool = False) -> None:
        import pyarrow as pa

        _, keys, _ = self._placed(rows)
        if not len(keys):
            return
        batch = pa.RecordBatch.from_pydict({"key": Keys(keys, keys[:, 1]).arrow()})
        self._put("delete", batch, missing_ok)

    @override
    def truncate(self) -> None:
        self._client._action("truncate", {"table": self.name})

    @override
    def known_series(self) -> list[Any]:
        """The keys of the server's series as given, in the order it first saw them.

        The server keeps each series' key with its observations and replays
        them with its log, so a client that opens the table later gets them
        too; a truncated table holds none.
        """
        known = self._client._action("known_series", {"table": self.name})[0]
        keys = []
        for named in known:
            key = key_of_spec(named["key"])
            self._names.setdefault(int(named["digest"]), key)
            keys.append(key)
        return keys

    @override
    def forecast(
        self,
        series: Iterable[Key],
        horizon: int | None = None,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        return self._forecast(series, None, horizon, before, quantiles)

    @override
    def forecast_given(
        self,
        series: Iterable[Key],
        groups: Sequence[Iterable[Key]],
        horizon: int | None = None,
        before: Any = None,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        return self._forecast(series, groups, horizon, before, quantiles)

    def _forecast(
        self,
        series: Iterable[Key],
        groups: Sequence[Iterable[Key]] | None,
        horizon: int | None,
        before: Any,
        quantiles: Sequence[float] | None,
    ) -> np.ndarray:
        """Forecasts series through the server's forecast exchange, each series' group in a `group` column if given."""
        import pyarrow as pa
        from pyarrow import flight

        horizon = self._horizon(horizon)
        levels = _levels(self.name, self.quantile_levels, quantiles)
        digests = self._keys(series)
        if not len(digests):
            return self._empty(horizon, len(levels or self.quantile_levels))
        columns: dict[str, Any] = {"series": pa.array(digests, type=pa.uint64())}
        if (cutoffs := _times(before, len(digests), self.times)) is not None:
            columns["before"] = pa.array(cutoffs)
        if groups is not None:
            members = [self._keys(group) for group in groups]
            if len(members) != len(digests):
                raise ValueError(
                    f"expected {len(digests)} groups, one per series, got {len(members)}"
                )
            offsets = np.concatenate(
                [[0], np.cumsum([len(group) for group in members])]
            )
            values = np.concatenate(members) if members else np.zeros(0, np.uint64)
            columns["group"] = pa.ListArray.from_arrays(
                pa.array(offsets, pa.int32()), pa.array(values, type=pa.uint64())
            )
        batch = pa.RecordBatch.from_pydict(columns)
        headers = [*self._client._headers, (b"gf-horizon", str(horizon).encode())]
        if levels is not None:
            listed = ",".join(repr(level) for level in levels)
            headers.append((b"gf-quantiles", listed.encode()))
        options = flight.FlightCallOptions(headers=headers)
        with self._naming():
            result = self._client._exchanged("forecast", self.name, batch, options)
        self.last_answer = _answer_of(result)
        # The server interpolates the levels asked, so the long batch's
        # quantile columns are the forecast's last axis.
        values = np.column_stack(
            [
                result.column(name).to_numpy()
                for name in result.schema.names
                if name.startswith("q")
            ]
        )
        return values.reshape(len(digests), horizon, -1).astype(np.float32)

    @override
    def vocabularies(self) -> dict[str, list[str]]:
        return dict(self._client._reply(self.name).get("vocabularies", {}))

    @override
    def __len__(self) -> int:
        return int(self._client._reply(self.name, words=False)["rows"])


class RemoteClient:
    """A client for a `gf-server` or a `gf-router`."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        tls_root_certs: bytes | None = None,
        client_certificate: bytes | None = None,
        client_key: bytes | None = None,
        server_name: str | None = None,
        allow_plaintext: bool = False,
    ) -> None:
        """Connects to a server.

        Args:
            url: The server, such as `grpc+tls://generalforecast.internal:8815`.
            token: A bearer token the server accepts.
            tls_root_certs: PEM certificates to trust, for private certificate authorities.
            client_certificate: A PEM client certificate, for servers that require one.
            client_key: The client certificate's PEM private key.
            server_name: The name to verify the server's certificate against, when
                it differs from the URL's host, such as a cluster reached by address.
            allow_plaintext: Send the token without TLS to another machine, on a
                network you trust; otherwise only `grpc+tls://` URLs and this
                machine take a token.

        Raises:
            ValueError: If the token would cross the network in plaintext, or a
                client certificate comes without its key or a key without one.
        """
        from pyarrow import flight

        # How to connect again, as unpickling does; the token travels with it.
        self._connection = {
            "url": url,
            "token": token,
            "tls_root_certs": tls_root_certs,
            "client_certificate": client_certificate,
            "client_key": client_key,
            "server_name": server_name,
            "allow_plaintext": allow_plaintext,
        }
        if (client_certificate is None) != (client_key is None):
            raise ValueError(
                "a client certificate needs its key, and a key its certificate"
            )

        if token is not None and not allow_plaintext and _plaintext(url):
            raise ValueError(
                f"connecting to {url} would send the token unencrypted; use "
                "grpc+tls://, or pass allow_plaintext=True on a network you trust"
            )

        options: dict[str, Any] = {}
        if tls_root_certs is not None:
            options["tls_root_certs"] = tls_root_certs
        if server_name is not None:
            options["override_hostname"] = server_name
        if client_certificate is not None and client_key is not None:
            options["cert_chain"], options["private_key"] = (
                client_certificate,
                client_key,
            )
        if url.startswith("grpc+unix://"):
            # gRPC sends a socket's path as the authority, which tonic refuses.
            options["generic_options"] = [("grpc.default_authority", "localhost")]
        self._flight = flight.FlightClient(url, **options)
        self._headers: list[tuple[bytes, bytes]] = (
            [] if token is None else [(b"authorization", f"Bearer {token}".encode())]
        )
        self._options = flight.FlightCallOptions(headers=self._headers)
        # Tables as last described, by name, so predictions skip describing them.
        self._known: dict[str, RemoteTable] = {}
        self._known_series: dict[str, RemoteSeriesTable] = {}
        self._lock = threading.Lock()
        # Prefixes this client's table names on the server, which other
        # clients share.
        self._namespace = ""

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickles as the settings to connect again with, token and keys included."""
        warnings.warn(
            "pickling a RemoteClient includes its token and any client key; keep "
            "the pickle as secret as the token",
            stacklevel=2,
        )
        return functools.partial(RemoteClient, **self._connection), ()

    def __repr__(self) -> str:
        """The client's endpoint, or the table's name."""
        return f"<RemoteClient {self._connection['url']}>"

    def close(self) -> None:
        """Closes the connection; tables on the server stay as they are."""
        self._flight.close()

    def __enter__(self) -> Self:
        """The client, closed on leaving the block."""
        return self

    def __exit__(self, *_: object) -> None:
        """Closes the client."""
        self.close()

    def _write_options(self, write_id: str, missing_ok: bool = False) -> Any:
        """Call options carrying a write's id, so a retry of it applies once."""
        from pyarrow import flight

        headers = [*self._headers, (b"gf-write-id", write_id.encode())]
        if missing_ok:
            headers.append((b"gf-missing-ok", b"1"))
        return flight.FlightCallOptions(headers=headers)

    def models(self) -> list[str]:
        """The names of the models the server serves, sorted."""
        return self._action("list_models")[0]

    def _action(self, kind: str, body: dict[str, Any] | None = None) -> list[Any]:
        from pyarrow import flight

        body = dict(body or {})
        if "table" in body:
            body["table"] = self._wire(body["table"])
        if kind == "create_table":
            body["name"] = self._wire(body["name"])
        payload = json.dumps(body).encode()
        with _errors():
            results = list(
                self._flight.do_action(flight.Action(kind, payload), self._options)
            )
        return [json.loads(result.body.to_pybytes() or b"null") for result in results]

    def _create_table(
        self,
        name: str,
        *,
        model: str,
        features: int | None = None,
        columns: Any = None,
        classes: int | None = None,
        strategy: Strategy | None = None,
        page_size: int = 128,
        time_bucket: Any = None,
        refit_after: float | None = None,
        refit_shift: float | None = None,
        times: Times = "integer",
        fit_rows: int = DEFAULT_FIT_ROWS,
        past_cap: PastCap = "warn",
        region_rows: int | None = None,
        rebuild: Rebuild = "always",
        rebuild_wait: RebuildWait = _REBUILD_WAIT,
        refit_cutoffs: RefitCutoffs = "each",
        scale_rows: int | None = None,
        summary: Summary = "table",
        retrieval_categoricals: RetrievalCategoricals = "codes",
        partitions: int = 1,
        route_by: Mapping[str, Any] | None = None,
        roles: Roles | None = None,
    ) -> RemoteTable:
        """Adds an empty table on the server.

        Args:
            name: The table's name: letters, digits, '_' or '-'.
            model: A model name from the server's configuration.
            features: Number of numeric feature columns, named `f0 ..`; or give
                `columns`.
            columns: The feature columns by name, each with its kind: `numeric`,
                `categorical` (strings, or codes `0 ..`), `boolean`, `datetime`
                or `retrieval`; see `gf.engine.Table`. Names alone make
                numeric columns. Or a frame whose column types give the kinds;
                see `generalforecast.columns.infer`.
            classes: Number of classes, for a classifier's table. A regressor's
                table takes none: the model decides what a table predicts.
            strategy: How each query's context is chosen; exact by default.
            page_size: Rows per page.
            time_bucket: A window of time that pages never straddle, in the
                units of row times, or a timedelta for a datetime table; with
                cutoffs, pages wholly before one keep their caches as later rows
                arrive.
            refit_after: Refit the table's state on its own once rows inserted,
                updated or deleted since the last fit reach this fraction of the
                rows it was fit on. A query with a cutoff reads a state fit only
                on rows before it; with this set, a state fit before an earlier
                cutoff also serves later ones, with their caches, until the rows
                between reach this fraction, and without it each cutoff reads a
                state fit on all the rows before it.
            refit_shift: Refit once a column's mean moves by more than this many
                of its standard deviations at the last fit.
            times: What row times and cutoffs are: `integer`, in a unit of your
                choosing, or `datetime`, for dates and times.
            fit_rows: The most rows the table's state is fit on. A table holding
                more fits on a fixed sample of this many and embeds each page
                only once a query reads it, so its fits stay bounded. A wide
                table whose state on this many rows would not fit a worker's
                call budget fits on a sample of the rows it takes, the
                description's `effective_fit_rows`. A fit on more rows than
                one call embeds with its state also embeds each page only
                once a query reads it.
            past_cap: What a context larger than the model was trained on does:
                `warn` reads it whole and the server logs it once; `clamp` cuts
                it to that length.
            region_rows: Live rows per region whose queries share a context;
                see `gf.engine.Table`.
            rebuild: When a region's context is built again; see
                `gf.engine.Table`.
            rebuild_wait: How long a context due a rebuild is read stale; see
                `gf.engine.Table`.
            refit_cutoffs: When a query's cutoff inside the table's history
                calls for a new state; see `gf.engine.Table`.
            scale_rows: Rows the retrieval scaling stays fixed after, once fit
                on at least that many; see `gf.engine.Table`.
            summary: What the model's state, preprocessing and column
                summary, is fit on: `table` fits it once on the table's rows,
                so row representations are cached per page; `context` fits the
                unit's own state on each context's rows, as the model's own
                predictor does, and embeds that context's queries against it.
                Only models whose state pools over rows, such as TabICL,
                TabPFN and Kumo Tabular, take `context`. Such a table's units
                depend on their rows alone, so refit policies and cutoffs
                refit nothing.
            retrieval_categoricals: How categorical columns place rows for
                retrieval; see `gf.engine.Table`.
            partitions: Tables to split the table into, through a router, which
                spreads them over its nodes. Rows go to partitions by their
                keys, and each query's context is gathered from them all. Such
                a table takes classes as integers, and queries without cutoffs.
            route_by: A numeric column whose value sends each row and query
                to one partition, which answers the query from its own rows:
                `{"column": name}` hashes the value, and `{"column": name,
                "kind": "ordered", "bounds": [...]}` cuts its range at the
                bounds, one fewer than the partitions. Such a table may have
                regions, and labels rows through `update`. The value places
                a row, so it cannot change for a key once the row is written.
                Delete the row and insert it again with the new value.
            roles: The frame columns the rows carry besides the feature
                columns, and a classifier's class values, which the server
                keeps and describes so `gf.engine.Table` opens the table again.

        Returns:
            The table.
        """
        made = columns_of(features, columns)
        _check_times(times)
        _check_classes(classes)
        spec = json.loads((strategy or Strategy.exact()).spec())
        self._action(
            "create_table",
            {
                "name": name,
                "model": self._model_ref(model),
                "columns": [
                    {"name": column.name, "kind": column.kind} for column in made
                ],
                # The server stores a regressor's table as 0 classes.
                "classes": classes or 0,
                "strategy": spec,
                "page_size": page_size,
                "time_bucket": _bucket(time_bucket, times),
                "times": times,
                "refit_after": refit_after,
                "refit_shift": refit_shift,
                "fit_rows": fit_rows,
                "past_cap": past_cap,
                "refit_cutoffs": refit_cutoffs,
                **({"scale_rows": scale_rows} if scale_rows is not None else {}),
                "retrieval_categoricals": retrieval_categoricals,
                **(
                    {
                        "regions": {
                            "rows": region_rows,
                            "rebuild": _rebuild_spec(rebuild),
                            **(
                                {"wait": asdict(rebuild_wait)}
                                if rebuild_wait != _REBUILD_WAIT
                                else {}
                            ),
                        }
                    }
                    if region_rows is not None
                    else {}
                ),
                "summary": summary,
                # Only a router splits a table; servers refuse the field.
                **({"partitions": partitions} if partitions != 1 else {}),
                **({"route_by": dict(route_by)} if route_by is not None else {}),
                **({"roles": roles.spec()} if roles is not None else {}),
            },
        )
        return self._remember(self._describe(name))

    def _create_series_table(
        self,
        name: str,
        *,
        model: str,
        series: str,
        time: str,
        target: str,
        step: Any,
        columns: Any = None,
        known: Iterable[str] = (),
        group: GroupStrategy | None = None,
        horizon: int | None = None,
        times: Times = "integer",
    ) -> RemoteSeriesTable:
        """Creates a table of series on the server; see `gf.engine.SeriesTable`.

        `model` names a forecaster the server's configuration serves.
        """
        _check_times(times)
        fixed = (series, time, target)
        made = () if columns is None else columns_of(None, columns)
        made = tuple(column for column in made if column.name not in fixed)
        known = list(known)
        if horizon is not None and horizon < 1:
            raise ValueError(f"a horizon is at least one step, not {horizon}")
        group = group or GroupStrategy.alone()
        self._action(
            "create_table",
            {
                "name": name,
                "model": self._model_ref(model),
                "columns": [
                    {"name": column.name, "kind": column.kind} for column in made
                ],
                "classes": 0,
                "times": times,
                "series": {
                    "series": series,
                    "time": time,
                    "target": target,
                    "step": _duration(step, times, "step"),
                    "known": known,
                    "group": json.loads(group.spec()),
                    **({"horizon": horizon} if horizon is not None else {}),
                },
            },
        )
        return self.series_table(name)

    def _reply(self, name: str, words: bool = True) -> dict[str, Any]:
        """A table's description as the server sends it."""
        body = {"table": name, "words": words}
        reply = self._action("describe", body)[0]
        return {**reply, "name": reply["name"].removeprefix(self._namespace)}

    def _describe(self, name: str, words: bool = True) -> Described:
        """A table as the server describes it: its definition, rows and vocabularies.

        Without `words`, the vocabularies stay on the server, which may hold
        many words per column; a count of rows or the labels need none.

        Raises:
            TypeError: If the table holds series; `series_table` gives it.
        """
        reply = self._reply(name, words)
        if reply.get("series"):
            raise TypeError(f"table {name!r} holds series; use series_table")
        return Described.of(reply)

    def table(self, name: str) -> RemoteTable:
        """A table on the server, by name, as it is now.

        Raises:
            KeyError: If there is no such table.
            TypeError: If the table holds series; `series_table` gives it.
        """
        return self._remember(self._describe(name))

    def _remember(self, described: Described) -> RemoteTable:
        """The table a description describes, remembered for predictions.

        The table remembered keeps its last answer while its definition holds;
        one the server holds under another definition, as when it was dropped
        and made again, takes its place.
        """
        fresh = RemoteTable(self, described)
        with self._lock:
            known = self._known.get(fresh.name)
            if known is None or known.definition != fresh.definition:
                known = self._known[fresh.name] = fresh
        return known

    def series_table(self, name: str) -> RemoteSeriesTable:
        """A table of series on the server, by name, as it is now.

        The table remembered keeps the series keys written through it while
        its definition holds; one made again under another takes its place.

        Raises:
            KeyError: If there is no such table.
            TypeError: If the table holds rows, not series; `table` gives it.
        """
        reply = self._reply(name)
        if not reply.get("series"):
            raise TypeError(f"table {name!r} holds rows, not series; use table")
        definition = _series_definition_of(reply)
        with self._lock:
            table = self._known_series.get(name)
            if table is None or table.definition != definition:
                table = RemoteSeriesTable(self, definition)
                self._known_series[name] = table
        return table

    def _cached(self, name: str) -> RemoteTable:
        """A table as last described, described now if it has not been."""
        with self._lock:
            known = self._known.get(name)
        return known if known is not None else self.table(name)

    def drop_table(self, name: str) -> None:
        """Removes a table, its rows, its log and its caches, on every node."""
        with self._lock:
            self._known.pop(name, None)
            self._known_series.pop(name, None)
        self._action("drop_table", {"table": name})

    def tables(self) -> list[str]:
        """Every table's name on the server that this client's token reaches, sorted."""
        names = self._action("list_tables")[0]
        prefix = self._namespace
        return [name[len(prefix) :] for name in names if name.startswith(prefix)]

    def _model_ref(self, model: str) -> str:
        """The name the server knows a table's model by."""
        if not isinstance(model, str):
            raise TypeError(
                "a server serves the models its configuration names; give a model name"
            )
        return model

    def _wire(self, name: str) -> str:
        """The server's name for one of this client's tables."""
        return self._namespace + name

    def predict(
        self, queries: Mapping[str, Any], before: Mapping[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        """Predicts queries for many tables at once, so the server batches them.

        Tables are described once and remembered, so a call is one exchange per
        table. One that fails as missing or refused is described again, and
        tried once more if the table changed, as one dropped and created anew
        with other columns would have; a refusal of the queries themselves
        is raised as it is.
        """
        cutoffs = before or {}

        def predict(name: str, rows: Any) -> np.ndarray:
            return self._predict_known(name, rows, cutoffs.get(name))

        if len(queries) == 1:
            ((name, rows),) = queries.items()
            return {name: predict(name, rows)}
        with ThreadPoolExecutor(max(1, min(len(queries), 16))) as pool:
            futures = {
                name: pool.submit(predict, name, rows) for name, rows in queries.items()
            }
        return {name: future.result() for name, future in futures.items()}

    def _predict_known(
        self,
        name: str,
        rows: Any,
        before: Any,
        levels: list[float] | None = None,
    ) -> np.ndarray:
        """One table's predictions, through its remembered description.

        If the server refuses them as missing or invalid, the table is
        described again, and they are tried once more only if it changed.
        """
        known = self._cached(name)
        try:
            return self._predict(known, rows, before, levels)
        except (KeyError, ValueError):
            fresh = self.table(name)
            if fresh.definition == known.definition:
                raise
            return self._predict(fresh, rows, before, levels)

    def _predict(
        self,
        table: RemoteTable,
        queries: Any,
        before: Any,
        levels: list[float] | None = None,
        given: Sequence[Iterable[Key]] | None = None,
    ) -> np.ndarray:
        """One table's predictions, from one exchange with the server."""
        from pyarrow import flight

        options = self._options
        if levels is not None:
            listed = ",".join(repr(level) for level in levels)
            headers = [*self._headers, (b"gf-quantiles", listed.encode())]
            options = flight.FlightCallOptions(headers=headers)
        result = self._exchange("predict", table, queries, before, options, given)
        if result is None:
            return (
                table._empty()
                if levels is None
                else np.empty((0, len(levels)), np.float32)
            )
        table.last_answer = _answer_of(result)
        return np.column_stack([column.to_numpy() for column in result.columns])

    def _contexts(
        self, table: RemoteTable, queries: Any, before: Any
    ) -> list[np.ndarray]:
        """One table's context keys per query, from one exchange with the server."""
        result = self._exchange("contexts", table, queries, before, self._options)
        if result is None:
            return []
        return [
            np.frombuffer(b"".join(keys), dtype="<u8").reshape(-1, 2)
            for keys in result.column("keys").to_pylist()
        ]

    def _region_churn(self, table: RemoteTable, queries: Any) -> np.ndarray:
        """One table's region churn per query, from one exchange with the server."""
        result = self._exchange("region_churn", table, queries, None, self._options)
        if result is None:
            return np.empty(0)
        return result.column("churn").to_numpy(zero_copy_only=False)

    def _exchange(
        self,
        command: str,
        table: RemoteTable,
        queries: Any,
        before: Any,
        options: Any,
        given: Sequence[Iterable[Key]] | None = None,
    ) -> Any:
        """The server's answer to queries sent with `command`, or None for none."""
        import pyarrow as pa

        prepared, rows = prepare(queries, table.columns)
        if rows == 0:
            return None
        columns = arrow_columns(prepared, table.columns)
        if (cutoffs := _times(before, rows, table.times)) is not None:
            columns["before"] = pa.array(cutoffs)
        if given is not None:
            keys = _given(given, rows)
            offsets = np.concatenate(
                [[0], np.cumsum([len(context) for context in keys])]
            )
            values = np.concatenate(keys).astype("<u8").tobytes() if keys else b""
            columns["context"] = pa.ListArray.from_arrays(
                pa.array(offsets, pa.int32()),
                pa.FixedSizeBinaryArray.from_buffers(
                    pa.binary(16), int(offsets[-1]), [None, pa.py_buffer(values)]
                ),
            )
        batch = pa.RecordBatch.from_pydict(columns)
        return self._exchanged(command, table.name, batch, options)

    def _exchanged(self, command: str, name: str, batch: Any, options: Any) -> Any:
        """The server's whole reply to one table's batch sent with `command`."""
        from pyarrow import flight

        descriptor = flight.FlightDescriptor.for_path(command, self._wire(name))
        with _errors():
            writer, reader = self._flight.do_exchange(descriptor, options)
            try:
                writer.begin(batch.schema)
                for part in _split(batch):
                    writer.write_batch(part)
                writer.done_writing()
                result = reader.read_all()
            except BaseException:
                # The stream failed already; its first error is the one to raise.
                with contextlib.suppress(Exception):
                    writer.close()
                raise
            writer.close()
        return result


class _Awaited[**P, R]:
    """A coroutine method that runs the wrapped table's method of its name off the event loop.

    Each takes its synchronous twin's arguments and docstring, so the two
    cannot drift apart.
    """

    def __init__(self, method: Callable[Concatenate[Any, P], R]) -> None:
        """Wraps `method`, which the wrapped table has under the same name."""
        self.__doc__ = method.__doc__
        self._name: str = getattr(method, "__name__", "")

    def __set_name__(self, owner: type, name: str) -> None:
        """Checks the method keeps its twin's name."""
        if name != self._name:
            raise TypeError(f"{owner.__name__}.{name} wraps {self._name!r}")

    def __get__(
        self, instance: Any, owner: type | None = None
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        """The coroutine method of a wrapping table."""
        name = self._name

        async def run(*arguments: P.args, **options: P.kwargs) -> R:
            method = getattr(instance._table, name)
            return await instance._client._run(
                functools.partial(method, *arguments, **options)
            )

        run.__doc__ = self.__doc__
        return run


class AsyncRemoteTable:
    """A table on a server, with coroutines; see `Table` for each call.

    Each coroutine runs `RemoteTable`'s method of its name on the client's
    threads, with its arguments.
    """

    def __init__(self, client: "AsyncRemoteClient", table: RemoteTable) -> None:
        """Wraps a table of the client's synchronous twin."""
        self._client, self._table = client, table
        self.definition = table.definition
        self.name, self.columns = table.name, table.columns
        self.classes, self.outputs = table.classes, table.outputs
        self.times, self.model, self.strategy = table.times, table.model, table.strategy

    def __repr__(self) -> str:
        """The client's endpoint, or the table's name."""
        return repr(self._table).replace("RemoteTable", "AsyncRemoteTable", 1)

    @property
    def features(self) -> int:
        """Number of feature columns."""
        return len(self.columns)

    @property
    def last_answer(self) -> Answer | None:
        """What answered the table's latest predictions; see `Table.last_answer`."""
        return self._table.last_answer

    describe = _Awaited(RemoteTable.describe)
    insert = _Awaited(RemoteTable.insert)
    update = _Awaited(RemoteTable.update)
    upsert = _Awaited(RemoteTable.upsert)
    label = _Awaited(RemoteTable.label)
    delete = _Awaited(RemoteTable.delete)
    truncate = _Awaited(RemoteTable.truncate)
    predict = _Awaited(RemoteTable.predict)
    predict_given = _Awaited(RemoteTable.predict_given)
    contexts = _Awaited(RemoteTable.contexts)
    region_churn = _Awaited(RemoteTable.region_churn)
    refit = _Awaited(RemoteTable.refit)
    set_strategy = _Awaited(RemoteTable.set_strategy)
    compact = _Awaited(RemoteTable.compact)
    vocabularies = _Awaited(RemoteTable.vocabularies)

    async def labels(self) -> list[str] | None:
        """A classifier's labels in code order; see `Table.labels`."""
        return await self._client._run(lambda: self._table.labels)

    async def count(self) -> int:
        """Number of live rows."""
        return await self._client._run(len, self._table)


class AsyncRemoteSeriesTable:
    """A table of series on a server, with coroutines; see `SeriesTable` for each call.

    Each coroutine runs `RemoteSeriesTable`'s method of its name on the
    client's threads, with its arguments.
    """

    def __init__(self, client: "AsyncRemoteClient", table: RemoteSeriesTable) -> None:
        """Wraps a table of the client's synchronous twin."""
        self._client, self._table = client, table
        self.definition = table.definition
        self.name, self.series, self.time = table.name, table.series, table.time
        self.target, self.columns, self.known = table.target, table.columns, table.known
        self.step, self.times, self.model = table.step, table.times, table.model
        self.group, self.horizon = table.group, table.horizon
        self.quantile_levels = table.quantile_levels

    def __repr__(self) -> str:
        """The client's endpoint, or the table's name."""
        return repr(self._table).replace(
            "RemoteSeriesTable", "AsyncRemoteSeriesTable", 1
        )

    @property
    def last_answer(self) -> Answer | None:
        """What answered the table's latest forecasts; see `SeriesTable.last_answer`."""
        return self._table.last_answer

    insert = _Awaited(RemoteSeriesTable.insert)
    upsert = _Awaited(RemoteSeriesTable.upsert)
    delete = _Awaited(RemoteSeriesTable.delete)
    truncate = _Awaited(RemoteSeriesTable.truncate)
    forecast = _Awaited(RemoteSeriesTable.forecast)
    forecast_given = _Awaited(RemoteSeriesTable.forecast_given)
    vocabularies = _Awaited(RemoteSeriesTable.vocabularies)
    known_series = _Awaited(RemoteSeriesTable.known_series)

    async def count(self) -> int:
        """Number of observations."""
        return await self._client._run(len, self._table)


class AsyncRemoteClient:
    """A client for a server whose calls are coroutines.

    Each call runs its synchronous twin on a thread, at most `concurrency` at
    once, so many predictions can be in flight from one event loop.
    """

    def __init__(self, client: RemoteClient, concurrency: int = 16) -> None:
        """Wraps a synchronous client."""
        self._client = client
        self._pool = ThreadPoolExecutor(
            max(1, concurrency), thread_name_prefix="generalforecast"
        )

    async def _run(self, call: Callable[..., Any], *arguments: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, lambda: call(*arguments))

    async def close(self) -> None:
        """Closes the connection and the threads behind the calls."""
        await self._run(self._client.close)
        self._pool.shutdown(wait=False)

    async def __aenter__(self) -> Self:
        """The client, closed on leaving the block."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Closes the client."""
        await self.close()

    async def _create_table(self, name: str, **settings: Any) -> AsyncRemoteTable:
        """Adds an empty table on the server; see `gf.engine.Table`."""
        table = await self._run(lambda: self._client._create_table(name, **settings))
        return AsyncRemoteTable(self, table)

    async def _create_series_table(
        self, name: str, **settings: Any
    ) -> AsyncRemoteSeriesTable:
        """Creates a table of series; see `gf.engine.SeriesTable`."""
        table = await self._run(
            lambda: self._client._create_series_table(name, **settings)
        )
        return AsyncRemoteSeriesTable(self, table)

    async def series_table(self, name: str) -> AsyncRemoteSeriesTable:
        """A table of series by name; see `RemoteClient.series_table`."""
        return AsyncRemoteSeriesTable(
            self, await self._run(self._client.series_table, name)
        )

    async def table(self, name: str) -> AsyncRemoteTable:
        """A table on the server, by name."""
        return AsyncRemoteTable(self, await self._run(self._client.table, name))

    async def drop_table(self, name: str) -> None:
        """Removes a table, its rows, its log and its caches, on every node."""
        await self._run(self._client.drop_table, name)

    async def tables(self) -> list[str]:
        """Every table's name on the server that this client's token reaches, sorted."""
        return await self._run(self._client.tables)

    async def models(self) -> list[str]:
        """The names of the models the server serves, sorted."""
        return await self._run(self._client.models)

    async def predict(
        self, queries: Mapping[str, Any], before: Mapping[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        """Predicts queries for many tables at once, so the server batches them.

        Tables are described once and remembered, as the synchronous client does.
        """
        names = list(queries)
        cutoffs = before or {}
        results = await asyncio.gather(
            *(
                self._run(
                    self._client._predict_known, name, queries[name], cutoffs.get(name)
                )
                for name in names
            )
        )
        return dict(zip(names, results, strict=True))

    def __repr__(self) -> str:
        """The client's endpoint, or the table's name."""
        return f"<AsyncRemoteClient {self._client._connection['url']}>"


def _plaintext(url: str) -> bool:
    """Whether a URL reaches another machine without TLS."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("grpc", "grpc+tcp"):
        return False
    host = parts.hostname or ""
    if host == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def connect(
    url: str,
    *,
    token: str | None = None,
    tls_root_certs: bytes | None = None,
    client_certificate: bytes | None = None,
    client_key: bytes | None = None,
    server_name: str | None = None,
    allow_plaintext: bool = False,
) -> RemoteClient:
    """A client for a `gf-server` or `gf-router`; see `RemoteClient`."""
    return RemoteClient(
        url,
        token,
        tls_root_certs,
        client_certificate,
        client_key,
        server_name,
        allow_plaintext,
    )


def connect_async(
    url: str,
    *,
    token: str | None = None,
    tls_root_certs: bytes | None = None,
    client_certificate: bytes | None = None,
    client_key: bytes | None = None,
    server_name: str | None = None,
    allow_plaintext: bool = False,
    concurrency: int = 16,
) -> AsyncRemoteClient:
    """A client with coroutines for a server; see `AsyncRemoteClient` and `connect`."""
    client = RemoteClient(
        url,
        token,
        tls_root_certs,
        client_certificate,
        client_key,
        server_name,
        allow_plaintext,
    )
    return AsyncRemoteClient(client, concurrency)
