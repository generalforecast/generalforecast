# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A standalone Models API server for catalog models.

`gf serve` runs gf-server, which answers the hosted Models API routes with
the same request and response envelopes, so the SDK works against both. It
holds no accounts, revisions or database; put auth beyond a single bearer
key and versioning in front of it.
"""

from generalforecast.serve.listener import main, serve

__all__ = ["main", "serve"]
