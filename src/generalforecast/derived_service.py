# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Derived models the CLI resolves through the user's own connection.

A hosted plane keeps a project's fine-tunes, `<project>/<name>@ft<version>`,
and answers what one is made of on the project's own route,
`GET /v1/projects/{project}/fine-tunes/{name}/versions/{version}`, to the
project's keys. The CLI installs `ServiceResolver` when nothing else
resolves derived models, so `gf check` and `gf push` lock a predictor on a
fine-tune with the same profile, key and project every other command uses.
Each model is asked for once per process.

The weights stay on the plane, which runs the fine-tune; this resolver
never loads them.
"""

from collections.abc import Callable
from typing import Any

from generalforecast.models.catalog import ModelRefError, parse
from generalforecast.models.derived import Derived


def _client(project: str) -> Any:
    from generalforecast.client import Client

    return Client(project=project)


class ServiceResolver:
    """Resolves fine-tunes through the hosted plane the user's connection names."""

    def __init__(self, connect: Callable[[str], Any] = _client) -> None:
        """Connects to a project with `connect`; the CLI's connection by default."""
        self.connect = connect
        self._found: dict[str, Derived] = {}

    def resolve(self, model: str) -> Derived:
        """What a fine-tune version is made of, as its project's plane says.

        Raises:
            ModelRefError: `UnknownModel` when the plane has no such version
                the connection may read.
        """
        if model in self._found:
            return self._found[model]
        from generalforecast import _operations as ops
        from generalforecast.service.contracts import ServiceError

        ref = parse(model)
        project, name = ref.name.split("/", 1)

        def read() -> Any:
            return (
                yield ops.Call(
                    "GET", f"/fine-tunes/{name}/versions/{ref.version}", "model", name
                )
            )

        try:
            found = self.connect(project)._run(read())
        except ServiceError as error:
            if error.status in (401, 403, 404):
                raise ModelRefError(
                    "UnknownModel",
                    f"{model} is not a fine-tune this connection may read: {error.message}",
                ) from None
            raise
        made = Derived(
            model, found["base"], found["base_digest"], found["digest"], found["bags"]
        )
        self._found[model] = made
        return made

    def weights(self, made: Derived, base: Any) -> Any:
        """Refused: a fine-tune runs on the plane that keeps its weights."""
        raise ModelRefError(
            "UnsupportedCapability",
            f"{made.model} runs on the hosted plane that keeps it; push a predictor on it "
            "and predict there",
        )
