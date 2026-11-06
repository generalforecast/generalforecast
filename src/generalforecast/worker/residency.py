# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which models' parameters a worker's device holds: a pinned core, and the rest by least recent use.

A worker serves every model it was started with and every model a server
asks it to load since, derived models included, but its device holds only
some of their parameters at once. Models started pinned (`PINNED_ENV`, or
every model a worker is started with when it names none) stay on the device
for the worker's life. The rest share `capacity` bytes of the device: before
one is placed, the models used least recently that no step is using are
moved off it until it fits, and a model that cannot fit even then is
refused with `DeviceMemoryRefused`, which a server answers as a device that
cannot hold it now rather than as a model that is broken.

Parameters are held leaf by leaf, as buffers, and a leaf two models hold
(the same array) is one buffer: a derived model whose base is loaded holds
only its own weights beside the base's, so a LoRA fine-tune costs its
adapters, not another copy of its base. A buffer stays on the device while
any model on it holds it, so moving a base off the device frees nothing
while a fine-tune of it is still there, and is never chosen for that.

A model moved off the device keeps its parameters in host memory, from which
it comes back by a copy, as bit for bit the same arrays: contexts kept for
its tables stay valid, and its compiled programs are kept. Host copies take
at most `host_capacity` bytes; past it, the copies of the models used least
recently that can load again (`reload`) are dropped, and such a model loads
again when next used.

A step leases the models it uses (`lease`): a leased model is never moved off
the device, and a model that needs room another step's lease holds waits
for it, for at most `wait` seconds.

Every count is under one lock, which waiting gives up.
"""

import itertools
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

from generalforecast.worker.admission import DeviceMemoryRefused

logger = logging.getLogger(__name__)

# The server's model names a worker keeps on its device for its life,
# comma-separated; unset, every model it is started with.
PINNED_ENV = "GF_PINNED_MODELS"
# Device bytes the models that are not pinned share, as bytes or with a
# KiB, MiB or GiB suffix; unset, `DEFAULT_CACHE_SHARE` of what the pinned
# models leave.
CACHE_ENV = "GF_MODEL_CACHE_BYTES"
# Host bytes the parameters of models moved off the device may take; unset,
# `DEFAULT_HOST_SHARE` of the worker's host memory.
HOST_ENV = "GF_MODEL_HOST_BYTES"
# The share of the device, beside the pinned models, models not pinned take by default.
DEFAULT_CACHE_SHARE = 8
# The share of the host the parameters of models off the device take by default.
DEFAULT_HOST_SHARE = 8
# The most a model waits for steps using the models it needs moved off the device.
WAIT_SECONDS = 30.0

_UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


def parse_bytes(text: str) -> int:
    """Bytes from `text`: a count, or one with a K, M, G or T suffix (`KiB`, `MB` and `GiB` alike), powers of 1024.

    Raises:
        ValueError: For text that is not such a count.
    """
    found = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?\s*", text, flags=re.IGNORECASE
    )
    if found is None:
        raise ValueError(f"{text!r} is not a number of bytes, such as 4GiB")
    return int(float(found[1]) * _UNITS[found[2].lower()])


def cache_bytes(device: int, pinned: int, named: str | None = None) -> int:
    """Device bytes the models not pinned share beside `pinned` bytes of pinned ones, on an accelerator a worker counts `device` bytes of.

    `named` is `CACHE_ENV`'s value, at most half of what the pinned models
    leave, so contexts and calls keep the rest; unset, `DEFAULT_CACHE_SHARE`
    of it.
    """
    left = max(0, device - pinned)
    if named is None:
        return left // DEFAULT_CACHE_SHARE
    return min(parse_bytes(named), left // 2)


def _held(leaf: Any) -> bool:
    """Whether a parameter leaf is an array the residency holds; other leaves stay as they are."""
    return isinstance(leaf, jax.Array)


def _bytes(leaf: Any) -> int:
    return math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize


@dataclass
class _Buffer:
    """One parameter array: on the device, in host memory, or both."""

    nbytes: int
    device: Any = None
    host: Any = None
    # The models holding it.
    users: set[int] = field(default_factory=set)


@dataclass
class _Static:
    """A parameter leaf the residency leaves as it is, such as a number."""

    value: Any


@dataclass
class _Entry:
    """A model's parameters as buffers, by leaf."""

    treedef: Any
    leaves: list[int | _Static]
    pinned: bool
    reload: Callable[[], Any] | None
    resident: bool = True
    leases: int = 0
    # Whether its own buffers were dropped from host memory, to load again.
    unloaded: bool = False
    # Whether it was placed on the device once: placing it again restores it.
    placed: bool = False
    # Its parameters, while on the device.
    tree: Any = None


@dataclass(frozen=True)
class ResidencyReport:
    """What a worker's device holds of its models, and what moved, since it started.

    Attributes:
        resident: Models on the device, pinned ones included.
        pinned: Models pinned to it.
        models: Models the worker serves, on the device or not.
        pinned_bytes: Device bytes the pinned models' parameters take.
        cached_bytes: Device bytes the other models' parameters take.
        capacity_bytes: Device bytes those may take, or None for no bound.
        host_bytes: Host bytes models off the device take.
        evictions: Models moved off the device.
        restores: Models put back on it.
        reloads: Models loaded again after their host copy was dropped.
        refusals: Models refused for want of device room.
    """

    resident: int
    pinned: int
    models: int
    pinned_bytes: int
    cached_bytes: int
    capacity_bytes: int | None
    host_bytes: int
    evictions: int
    restores: int
    reloads: int
    refusals: int

    def frame(self) -> dict[str, Any]:
        """The report as a reply header's `models` field."""
        found = {
            "resident": self.resident,
            "pinned": self.pinned,
            "models": self.models,
            "pinned_bytes": self.pinned_bytes,
            "cached_bytes": self.cached_bytes,
            "host_bytes": self.host_bytes,
            "evictions": self.evictions,
            "restores": self.restores,
            "reloads": self.reloads,
            "refusals": self.refusals,
        }
        if self.capacity_bytes is not None:
            found["capacity_bytes"] = self.capacity_bytes
        return found


class Residency:
    """The parameters of a worker's models, on its device or off it, within a byte budget."""

    def __init__(
        self,
        capacity: int | None = None,
        host_capacity: int | None = None,
        wait: float = WAIT_SECONDS,
    ) -> None:
        """Holds no models yet.

        Args:
            capacity: Device bytes models not pinned may take; None for no bound.
            host_capacity: Host bytes models off the device may take; None for no bound.
            wait: The most seconds room waits for leases to end.
        """
        self.capacity = capacity
        self.host_capacity = host_capacity
        self.wait = wait
        self._lock = threading.Condition(threading.RLock())
        self._buffers: dict[int, _Buffer] = {}
        self._entries: dict[int, _Entry] = {}
        # Models by use, least recent first.
        self._order: OrderedDict[int, None] = OrderedDict()
        # Each buffer by the id of an array it holds, on the device or off it,
        # so a leaf another model holds is found as the same buffer. Only the
        # arrays a buffer keeps are named here, and `_find` reads an id only
        # as the very array its buffer keeps: an id names an object only
        # while it lives, and a freed array's id is soon another array's.
        self._ids: dict[int, int] = {}
        self._keys = itertools.count()
        # An accelerator's parameters off it are host memory; a CPU's device is the host.
        self._accelerator = jax.default_backend() != "cpu"
        self.evictions = self.restores = self.reloads = self.refusals = 0

    # Adding and removing models.

    def add(
        self,
        model: int,
        parameters: Any,
        *,
        pinned: bool = False,
        reload: Callable[[], Any] | None = None,
    ) -> None:
        """Holds a model's parameters, placing them on the device.

        Leaves already on the device stay there; leaves in host memory (a
        CPU's arrays on an accelerator) go to the device once room is made.
        A leaf another model holds is shared, not copied.

        Args:
            model: The worker's id for the model.
            parameters: Its parameters, configured for the device.
            pinned: Whether it stays on the device for the worker's life.
            reload: Loads its parameters again once their host copy is
                dropped; without it, the copy is never dropped.

        Raises:
            DeviceMemoryRefused: If it would not fit beside the pinned models
                with every other model moved off the device.
        """
        with self._lock:
            if model in self._entries:
                raise ValueError(f"model {model} is already held")
            created: list[int] = []
            entry = self._entry(parameters, pinned, reload, created)
            self._entries[model] = entry
            for key in self._keys_of(entry):
                self._buffers[key].users.add(model)
            self._order[model] = None
            entry.resident = False
            try:
                self._place(model)
            except BaseException:
                self._forget(model)
                raise

    def remove(self, model: int) -> None:
        """Forgets a model, freeing the buffers no other model holds."""
        with self._lock:
            self._forget(model)
            self._lock.notify_all()

    def _entry(
        self,
        parameters: Any,
        pinned: bool,
        reload: Callable[[], Any] | None,
        created: list[int],
    ) -> _Entry:
        """A model's parameters as buffers: those other models hold found by identity, the rest new."""
        leaves, treedef = jax.tree.flatten(parameters)
        refs: list[int | _Static] = []
        # This tree's leaves by id, which hold while `parameters` keeps them:
        # a leaf it holds twice is one buffer, whether or not one is kept.
        seen: dict[int, int] = {}
        for leaf in leaves:
            if not _held(leaf) and not isinstance(leaf, np.ndarray):
                refs.append(_Static(leaf))
                continue
            key = seen.get(id(leaf))
            if key is None:
                key = self._find(leaf)
            if key is None:
                if not _held(leaf):
                    # A NumPy leaf is left where its model put it, as before.
                    refs.append(_Static(leaf))
                    continue
                key = next(self._keys)
                buffer = _Buffer(_bytes(leaf))
                if self._on_host(leaf):
                    # The buffer keeps this copy, not `leaf`, which its
                    # loader frees: `leaf`'s id is not the buffer's to name.
                    buffer.host = np.asarray(leaf)
                    self._ids[id(buffer.host)] = key
                else:
                    buffer.device = leaf
                    self._ids[id(leaf)] = key
                self._buffers[key] = buffer
                created.append(key)
            seen[id(leaf)] = key
            refs.append(key)
        return _Entry(treedef, refs, pinned, reload)

    def _find(self, leaf: Any) -> int | None:
        """The buffer that keeps this very array, on the device or off it; None if none does.

        An id is read only as the array its buffer keeps now: an id the
        index still names after its array was freed or replaced may be a
        new array's, of another model and shape, which must not run as it.
        """
        key = self._ids.get(id(leaf))
        if key is None:
            return None
        buffer = self._buffers.get(key)
        if buffer is None or (buffer.device is not leaf and buffer.host is not leaf):
            del self._ids[id(leaf)]
            return None
        return key

    def _on_host(self, leaf: jax.Array) -> bool:
        """Whether an array is in host memory, off an accelerator."""
        return self._accelerator and all(
            device.platform == "cpu" for device in leaf.devices()
        )

    def _forget(self, model: int) -> None:
        entry = self._entries.pop(model, None)
        self._order.pop(model, None)
        if entry is None:
            return
        for key in set(self._keys_of(entry)):
            buffer = self._buffers.get(key)
            if buffer is None:
                continue
            buffer.users.discard(model)
            if not buffer.users:
                self._drop(key)

    def _drop(self, key: int) -> None:
        """Frees a buffer every model gave up."""
        buffer = self._buffers.pop(key)
        for array in (buffer.device, buffer.host):
            if array is not None and self._ids.get(id(array)) == key:
                del self._ids[id(array)]

    @staticmethod
    def _keys_of(entry: _Entry) -> list[int]:
        return [ref for ref in entry.leaves if isinstance(ref, int) and ref >= 0]

    # What models' parameters are.

    def parameters(self, model: int) -> Any:
        """A model's parameters: on the device while it is there, else its host copy.

        A step leases its models first (`lease`), so it reads them on the
        device; what reads a model's shapes alone, as a measure, reads
        either. A model whose host copy was dropped loads again.
        """
        with self._lock:
            entry = self._entries[model]
            if entry.resident:
                if entry.tree is None:
                    entry.tree = self._tree(entry)
                return entry.tree
            if entry.unloaded:
                self._reload(model)
            return self._tree(self._entries[model])

    def _tree(self, entry: _Entry) -> Any:
        leaves = []
        for ref in entry.leaves:
            if isinstance(ref, _Static):
                leaves.append(ref.value)
                continue
            buffer = self._buffers[ref]
            leaves.append(buffer.device if buffer.device is not None else buffer.host)
        return jax.tree.unflatten(entry.treedef, leaves)

    def held(self, model: int) -> bool:
        """Whether a model's parameters are held, on the device or off it."""
        with self._lock:
            return model in self._entries

    def resident(self, model: int) -> bool:
        """Whether a model's parameters are on the device."""
        with self._lock:
            return self._entries[model].resident

    def pinned(self, model: int) -> bool:
        """Whether a model stays on the device for the worker's life."""
        with self._lock:
            return self._entries[model].pinned

    def order(self) -> list[int]:
        """The models by use, least recent first."""
        with self._lock:
            return list(self._order)

    # Bytes.

    def _pinned_keys(self) -> set[int]:
        return {
            key
            for entry in self._entries.values()
            if entry.pinned
            for key in self._keys_of(entry)
        }

    def pinned_bytes(self) -> int:
        """Device bytes the pinned models' parameters take."""
        with self._lock:
            return sum(self._buffers[key].nbytes for key in self._pinned_keys())

    def cached_bytes(self) -> int:
        """Device bytes the parameters of models not pinned take."""
        with self._lock:
            pinned = self._pinned_keys()
            return sum(
                buffer.nbytes
                for key, buffer in self._buffers.items()
                if buffer.device is not None and key not in pinned
            )

    def device_bytes(self) -> int:
        """Device bytes every model's parameters take, each buffer once."""
        with self._lock:
            return sum(
                buffer.nbytes
                for buffer in self._buffers.values()
                if buffer.device is not None
            )

    def host_bytes(self) -> int:
        """Host bytes the parameters of models off the device take."""
        with self._lock:
            return sum(
                buffer.nbytes
                for buffer in self._buffers.values()
                if buffer.host is not None and buffer.device is None
            )

    def model_bytes(self, model: int) -> int:
        """Bytes a model's parameters take, each buffer once, shared ones included."""
        with self._lock:
            keys = set(self._keys_of(self._entries[model]))
            return sum(self._buffers[key].nbytes for key in keys)

    def own_bytes(self, model: int) -> int:
        """Bytes of a model's parameters no other model holds: what holding it adds."""
        with self._lock:
            keys = set(self._keys_of(self._entries[model]))
            return sum(
                self._buffers[key].nbytes
                for key in keys
                if self._buffers[key].users == {model}
            )

    def report(self) -> ResidencyReport:
        """What the device holds, and what moved since the worker started."""
        with self._lock:
            entries = self._entries.values()
            return ResidencyReport(
                resident=sum(entry.resident for entry in entries),
                pinned=sum(entry.pinned for entry in entries),
                models=len(self._entries),
                pinned_bytes=self.pinned_bytes(),
                cached_bytes=self.cached_bytes(),
                capacity_bytes=self.capacity,
                host_bytes=self.host_bytes(),
                evictions=self.evictions,
                restores=self.restores,
                reloads=self.reloads,
                refusals=self.refusals,
            )

    # Moving models on and off the device.

    @contextmanager
    def lease(self, models: Iterable[int]) -> Iterator[None]:
        """Holds `models` on the device until the block ends, putting back any moved off it.

        Raises:
            DeviceMemoryRefused: If one would not fit with every model not
                pinned, leased or among them moved off the device, after
                waiting for leases to end.
        """
        chosen = list(dict.fromkeys(models))
        with self._lock:
            chosen = [model for model in chosen if model in self._entries]
            for model in chosen:
                self._entries[model].leases += 1
            try:
                for model in chosen:
                    self._order.move_to_end(model)
                    if not self._entries[model].resident:
                        self._place(model, keep=set(chosen))
            except BaseException:
                self._release(chosen)
                raise
        try:
            yield
        finally:
            with self._lock:
                self._release(chosen)

    def _release(self, models: list[int]) -> None:
        for model in models:
            entry = self._entries.get(model)
            if entry is not None:
                entry.leases -= 1
        self._lock.notify_all()

    def leases(self, model: int) -> int:
        """Steps using a model now."""
        with self._lock:
            return self._entries[model].leases

    def shrink(self) -> None:
        """Moves models off the device, least recently used first, until those not pinned fit `capacity`.

        A model no room can be made for is moved off the device too: it
        comes back when a step needs it, if it fits then.
        """
        with self._lock:
            while self.capacity is not None and self.cached_bytes() > self.capacity:
                chosen = self._evictable(set())
                if chosen is None:
                    break
                self._evict(chosen)

    def spill(self, free: Callable[[], int | None], needed: int | None = None) -> int:
        """Moves models no step uses off the device, least recently used first, until `free` says it has `needed` bytes free.

        For a call that needs more of the device than it has free: the
        models not pinned hold their share of the device whether a step uses
        them or not, and one moved off comes back by a copy when a step next
        needs it. With `needed` None, as for a call that ran out of memory
        by how much is not known, every such model moves off. Pinned and
        leased models stay; nothing waits.

        Returns:
            The models moved off the device.
        """
        moved = 0
        with self._lock:
            while True:
                if needed is not None:
                    left = free()
                    if left is None or left >= needed:
                        break
                chosen = self._evictable(set())
                if chosen is None:
                    break
                self._evict(chosen)
                moved += 1
        return moved

    def evict(self, model: int) -> None:
        """Moves a model not pinned off the device, waiting for steps using it.

        Raises:
            ValueError: For a pinned model.
            DeviceMemoryRefused: If steps still use it after waiting.
        """
        with self._lock:
            entry = self._entries[model]
            if entry.pinned:
                raise ValueError(f"model {model} is pinned to the device")
            if not entry.resident:
                return
            deadline = time.monotonic() + self.wait
            while entry.leases:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise DeviceMemoryRefused(
                        f"model {model} is in use by a step", 0, 0
                    )
                self._lock.wait(left)
            self._evict(model)

    def _place(self, model: int, keep: set[int] | None = None) -> None:
        """Puts a model's parameters on the device, making room for them first."""
        entry = self._entries[model]
        if entry.unloaded:
            self._reload(model)
            entry = self._entries[model]
        keep = {model} | (keep or set())
        if not entry.pinned:
            self._make_room(model, keep)
        restored = False
        for key in set(self._keys_of(entry)):
            buffer = self._buffers[key]
            if buffer.device is None:
                buffer.device = jax.device_put(buffer.host)
                self._ids[id(buffer.device)] = key
                # The host copy goes once nothing off the device holds it.
                if all(
                    self._entries[user].resident or user == model
                    for user in buffer.users
                ):
                    self._ids.pop(id(buffer.host), None)
                    buffer.host = None
                restored = True
        entry.resident = True
        entry.tree = None
        if restored and entry.placed:
            self.restores += 1
        entry.placed = True

    def _needed(self, model: int) -> int:
        """Device bytes not pinned that placing a model adds."""
        pinned = self._pinned_keys()
        keys = set(self._keys_of(self._entries[model]))
        return sum(
            self._buffers[key].nbytes
            for key in keys
            if self._buffers[key].device is None and key not in pinned
        )

    def _frees(self, model: int) -> int:
        """Device bytes moving a model off the device frees."""
        pinned = self._pinned_keys()
        freed = 0
        for key in set(self._keys_of(self._entries[model])):
            buffer = self._buffers[key]
            if buffer.device is None or key in pinned:
                continue
            if all(
                user == model or not self._entries[user].resident
                for user in buffer.users
            ):
                freed += buffer.nbytes
        return freed

    def _floor(self, keep: set[int]) -> int:
        """Device bytes not pinned that the models in `keep` hold, which no eviction frees."""
        pinned = self._pinned_keys()
        return sum(
            buffer.nbytes
            for key, buffer in self._buffers.items()
            if buffer.device is not None
            and key not in pinned
            and not keep.isdisjoint(buffer.users)
        )

    def _evictable(self, keep: set[int]) -> int | None:
        """The model used least recently whose moving off the device frees bytes, and that no step uses."""
        for model in self._order:
            entry = self._entries[model]
            if (
                entry.resident
                and not entry.pinned
                and not entry.leases
                and model not in keep
                and self._frees(model) > 0
            ):
                return model
        return None

    def _waitable(self, keep: set[int]) -> bool:
        """Whether a model a step uses would free bytes once the step ends."""
        return any(
            entry.resident
            and not entry.pinned
            and entry.leases
            and model not in keep
            and self._frees(model) > 0
            for model, entry in self._entries.items()
        )

    def _make_room(self, model: int, keep: set[int]) -> None:
        """Moves models off the device, least recently used first, until `model` fits `capacity`."""
        if self.capacity is None:
            return
        needed = self._needed(model)
        deadline = time.monotonic() + self.wait
        while self.cached_bytes() + needed > self.capacity:
            # What would stay with every other model moved off: a model that
            # would not fit even then moves none off for nothing.
            floor = self._floor(keep) + needed
            chosen = self._evictable(keep) if floor <= self.capacity else None
            if chosen is not None:
                self._evict(chosen)
                continue
            left = deadline - time.monotonic()
            if floor <= self.capacity and self._waitable(keep) and left > 0:
                self._lock.wait(left)
                needed = self._needed(model)
                continue
            self.refusals += 1
            room = max(0, self.capacity - self.cached_bytes())
            raise DeviceMemoryRefused(
                f"its parameters take {needed >> 20} MiB of the device beside "
                f"the models it shares them with, but the models that are not "
                f"pinned have {room >> 20} MiB of their {self.capacity >> 20} MiB "
                "left with every model no step is using moved off it",
                needed,
                room,
            )

    def _evict(self, model: int) -> None:
        """Moves a model off the device, keeping a host copy of what no model on it holds."""
        entry = self._entries[model]
        pinned = self._pinned_keys()
        for key in set(self._keys_of(entry)):
            buffer = self._buffers[key]
            if buffer.device is None or key in pinned:
                continue
            if any(
                user != model and self._entries[user].resident for user in buffer.users
            ):
                continue
            if buffer.host is None:
                buffer.host = np.asarray(jax.device_get(buffer.device))
                self._ids[id(buffer.host)] = key
            self._ids.pop(id(buffer.device), None)
            buffer.device = None
        entry.resident = False
        entry.tree = None
        self.evictions += 1
        logger.info("model %d moved off the device", model)
        self._trim_host()

    def _trim_host(self) -> None:
        """Drops the host copies of models that load again, least recently used first, past `host_capacity`."""
        if self.host_capacity is None:
            return
        for model in list(self._order):
            if self.host_bytes() <= self.host_capacity:
                return
            entry = self._entries[model]
            if entry.resident or entry.unloaded or entry.reload is None:
                continue
            if entry.leases:
                continue
            for key in set(self._keys_of(entry)):
                buffer = self._buffers.get(key)
                if buffer is not None and buffer.users == {model}:
                    buffer.users.clear()
                    self._drop(key)
            # A dropped buffer reads as -1 until the model loads again.
            entry.leaves = [
                ref if isinstance(ref, _Static) or ref in self._buffers else -1
                for ref in entry.leaves
            ]
            entry.unloaded = True

    def _reload(self, model: int) -> None:
        """Loads a model whose host copy was dropped again, sharing what other models still hold."""
        entry = self._entries[model]
        assert entry.reload is not None
        parameters = entry.reload()
        created: list[int] = []
        fresh = self._entry(parameters, entry.pinned, entry.reload, created)
        if fresh.treedef != entry.treedef:
            for key in created:
                self._drop(key)
            raise ValueError(f"model {model} loaded again with other parameters")
        for key in self._keys_of(fresh):
            self._buffers[key].users.add(model)
        fresh.leases = entry.leases
        fresh.resident = False
        fresh.placed = True
        self._entries[model] = fresh
        self.reloads += 1
