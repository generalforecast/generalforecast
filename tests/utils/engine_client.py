# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The engine's Flight client under the names server, router and engine tests use.

The public SDK is `gf.engine.Table`; these tests drive the engine's keyed
tables directly, so this shim keeps the former names (`Strategy`,
`GroupStrategy`, `create_table`) on the internal clients.
"""

from typing import Any, cast

from generalforecast import worker
from generalforecast._core import GroupStrategy, Strategy
from generalforecast.columns import Column
from generalforecast.keys import keys_of
from generalforecast.native.handles import RebuildWait, SeriesTable, Table
from generalforecast.native.local_client import LocalClient as _LocalClient
from generalforecast.native.remote import AsyncRemoteClient as _AsyncRemoteClient
from generalforecast.native.remote import (
    AsyncRemoteSeriesTable,
    AsyncRemoteTable,
    RemoteSeriesTable,
    RemoteTable,
)
from generalforecast.native.remote import RemoteClient as _RemoteClient
from generalforecast.native.remote import connect as _connect
from generalforecast.native.remote import connect_async as _connect_async


class RemoteClient(_RemoteClient):
    """The Flight client with the former `create_table` names."""

    def create_table(self, name: str, **settings: Any) -> RemoteTable:
        """Creates a keyed table; see `RemoteClient._create_table`."""
        return self._create_table(name, **settings)

    def create_series_table(self, name: str, **settings: Any) -> RemoteSeriesTable:
        """Creates a table of series; see `RemoteClient._create_series_table`."""
        return self._create_series_table(name, **settings)


class LocalClient(_LocalClient):
    """This process's engine with the former `create_table` names."""

    def create_table(self, name: str, **settings: Any) -> Any:
        """Creates a keyed table; see `LocalClient._create_table`."""
        return self._create_table(name, **settings)

    def create_series_table(self, name: str, **settings: Any) -> Any:
        """Creates a table of series; see `LocalClient._create_series_table`."""
        return self._create_series_table(name, **settings)


class AsyncRemoteClient(_AsyncRemoteClient):
    """The asynchronous Flight client with the former `create_table` names."""

    async def create_table(self, name: str, **settings: Any) -> AsyncRemoteTable:
        """Creates a keyed table."""
        return await self._create_table(name, **settings)

    async def create_series_table(
        self, name: str, **settings: Any
    ) -> AsyncRemoteSeriesTable:
        """Creates a table of series."""
        return await self._create_series_table(name, **settings)


def connect(*args: Any, **options: Any) -> RemoteClient:
    """Connects to a server; see `generalforecast.native.remote.connect`."""
    client = _connect(*args, **options)
    client.__class__ = RemoteClient
    return cast(RemoteClient, client)


def connect_async(*args: Any, **options: Any) -> AsyncRemoteClient:
    """Connects asynchronously; see `generalforecast.native.remote.connect_async`."""
    client = _connect_async(*args, **options)
    client.__class__ = AsyncRemoteClient
    return cast(AsyncRemoteClient, client)


def local(**options: Any) -> LocalClient:
    """This process's engine as a client; see `generalforecast.native.local_client.local`."""
    return LocalClient(**options)


__all__ = [
    "AsyncRemoteClient",
    "AsyncRemoteSeriesTable",
    "AsyncRemoteTable",
    "Column",
    "GroupStrategy",
    "LocalClient",
    "RebuildWait",
    "RemoteClient",
    "RemoteSeriesTable",
    "RemoteTable",
    "SeriesTable",
    "Strategy",
    "Table",
    "connect",
    "connect_async",
    "keys_of",
    "local",
    "worker",
]
