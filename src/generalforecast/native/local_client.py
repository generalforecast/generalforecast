# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`LocalClient`: the table client on this process's engine."""

import functools
import uuid
from typing import Any, Literal, Self, override

from generalforecast import engine
from generalforecast.models.catalog import ModelRefError, entries, parse
from generalforecast.native.remote import RemoteClient, RemoteSeriesTable, RemoteTable

LocalTable = RemoteTable
LocalSeriesTable = RemoteSeriesTable

# gf-server's memory settings, by `LocalClient`'s argument, at their defaults.
_TIERS: dict[str, Any] = {
    "device_bytes": None,
    "host_bytes": None,
    "slot_bytes": 1 << 20,
    "bucket_minimum": 8,
    "disk_bytes": 0,
    "step_bytes": None,
    "eviction": "gdsf",
}


class LocalClient(RemoteClient):
    """Tables on this process's engine: gf-server's shards and workers.

    It is `RemoteClient` on the engine's private Flight endpoint, so tables
    are gf-server's, scheduled by its shards and run by its worker
    processes. Clients with the default memory share the process's engine,
    and so its loaded models and steps, with every other surface; each
    client sees only its own tables. Safe to share between threads.
    """

    def __init__(
        self,
        device_bytes: int | None = None,
        host_bytes: int | None = None,
        slot_bytes: int = 1 << 20,
        bucket_minimum: int = 8,
        disk_bytes: int = 0,
        step_bytes: int | None = None,
        eviction: Literal["gdsf", "lru"] = "gdsf",
        allow_downloads: bool = True,
    ) -> None:
        """Connects to the process's engine, or starts one with other memory.

        Args:
            device_bytes: Accelerator memory for each worker's cache; by
                default measured from the device.
            host_bytes: Host memory for each worker's spilled cache; by
                default measured from the machine.
            slot_bytes: Bytes of each cache slot.
            bucket_minimum: The fewest rows a compiled shape pads to.
            disk_bytes: Bytes of each worker's disk tier; 0 for none.
            step_bytes: Most bytes a step's inputs take; by default unbounded.
            eviction: Which cached entry a full tier gives up first.
            allow_downloads: Whether a built-in model's missing pinned files
                may download as it loads.
        """
        given: dict[str, int | str | None] = {
            "device_bytes": device_bytes,
            "host_bytes": host_bytes,
            "slot_bytes": slot_bytes,
            "bucket_minimum": bucket_minimum,
            "disk_bytes": disk_bytes,
            "step_bytes": step_bytes,
            "eviction": eviction,
        }
        self._tiers: dict[str, int | str] = {
            name: value
            for name, value in given.items()
            if value is not None and value != _TIERS[name]
        }
        self._allow_downloads = allow_downloads
        self._engine = (
            engine.LocalEngine(allow_downloads=allow_downloads, tiers=self._tiers)
            if self._tiers
            else engine.local(allow_downloads=allow_downloads)
        )
        location, token = self._engine.serve()
        super().__init__(location, token)
        self._namespace = f"local-{uuid.uuid4().hex[:16]}-"

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickles as the settings to make a new, empty client with."""
        return functools.partial(
            LocalClient, allow_downloads=self._allow_downloads, **self._tiers
        ), ()

    def __repr__(self) -> str:  # noqa: D105
        return f"<LocalClient: {len(self.tables())} tables>"

    def close(self) -> None:
        """Drops this client's tables, and stops an engine it started."""
        for name in self.tables():
            self.drop_table(name)
        super().close()
        if self._tiers:
            self._engine.close()

    def __enter__(self) -> Self:  # noqa: D105
        return self

    def models(self) -> list[str]:
        """The model specs tables can name, sorted.

        These are the built-in models, which load on first use, and any other
        spec loaded so far.
        """
        built_in = {entry.id for entry in entries()}
        return sorted(
            {_catalog_name(name) for name in built_in | set(super().models())}
        )

    @override
    def _model_ref(self, model: str) -> str:
        """Loads a model spec in the engine's workers: the name they serve it by."""
        if not isinstance(model, str):
            raise TypeError(
                "LocalClient runs models in the engine's workers; give a model "
                "spec, such as 'general-forecast/tabicl-v2-classifier'"
            )
        return self._engine.load(parse(model))


def local(**options: Any) -> LocalClient:
    """A client of this process's engine; see `LocalClient`."""
    return LocalClient(**options)


def _catalog_name(model: str) -> str:
    """A model's catalog name, as tables name it; other specs as given."""
    try:
        return str(parse(model))
    except ModelRefError:
        return model
