# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Flight client, under the names server, router and CLI tests use."""

from generalforecast import Column, GroupStrategy, RebuildWait, Strategy, Table, local
from generalforecast.keys import keys_of
from generalforecast.native.remote import (
    AsyncRemoteClient,
    AsyncRemoteSeriesTable,
    AsyncRemoteTable,
    RemoteClient,
    RemoteSeriesTable,
    RemoteTable,
    connect,
    connect_async,
)

__all__ = [
    "AsyncRemoteClient",
    "AsyncRemoteSeriesTable",
    "AsyncRemoteTable",
    "Column",
    "GroupStrategy",
    "RebuildWait",
    "RemoteClient",
    "RemoteSeriesTable",
    "RemoteTable",
    "Strategy",
    "Table",
    "connect",
    "connect_async",
    "keys_of",
    "local",
]
