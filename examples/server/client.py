# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Canonical scoped client; provision a normal key using docs/deployments.md first.

Set GF_URL, GF_PROJECT and GF_TOKEN in the environment, then run this file.
The source-only native server/router protocols are not a public SDK.
"""

import os

from generalforecast import Client

client = Client(
    os.environ["GF_URL"],
    project=os.environ.get("GF_PROJECT", "local"),
    token=os.environ["GF_TOKEN"],
)
print(client.models.list())
