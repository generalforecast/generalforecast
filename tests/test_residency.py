# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A worker's device holds a pinned core of models and the rest by least recent use."""

from __future__ import annotations

import queue
import threading
import time
import weakref
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast import protocol
from generalforecast._core import CacheKey
from generalforecast.models import catalog, derived, registry
from generalforecast.models.conformance import table as conformance_table
from generalforecast.models.tabicl.model import tabicl_functions
from generalforecast.runtime.budget import MemoryBudget
from generalforecast.worker import Worker
from generalforecast.worker.admission import Admission, DeviceMemoryRefused
from generalforecast.worker.residency import Residency, parse_bytes
from tests.models.conformance.cases import FEATURES
from tests.models.tabicl.tiny import tiny_tabicl

KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30


def _weights(kib: int, value: float = 1.0) -> dict[str, jax.Array]:
    """Parameters of `kib` KiB of float32."""
    return {"w": jnp.full((kib * KIB // 4,), value, jnp.float32)}


def test_bytes_read_with_their_units() -> None:
    assert parse_bytes("4GiB") == 4 * GIB
    assert parse_bytes("512 MiB") == 512 * MIB
    assert parse_bytes("1.5G") == 3 * GIB // 2
    assert parse_bytes("123") == 123
    with pytest.raises(ValueError, match="number of bytes"):
        parse_bytes("lots")


def test_the_model_used_least_recently_moves_off_the_device_first() -> None:
    residency = Residency(capacity=2 * 64 * KIB)
    residency.add(0, _weights(64, 0.0))
    residency.add(1, _weights(64, 1.0))
    with residency.lease([0]):
        pass

    # Model 1 was used least recently, model 0 leased since.
    residency.add(2, _weights(64, 2.0))

    assert [residency.resident(model) for model in (0, 1, 2)] == [True, False, True]
    assert residency.cached_bytes() == 2 * 64 * KIB
    assert residency.report().evictions == 1

    # Put back as it was, bit for bit, moving off the one used least since.
    with residency.lease([1]):
        restored = residency.parameters(1)["w"]
        assert isinstance(restored, jax.Array)
        np.testing.assert_array_equal(np.asarray(restored), 1.0)
    assert [residency.resident(model) for model in (0, 1, 2)] == [False, True, True]
    assert residency.report().restores == 1
    # Off the device, a model reads as its host copy, which shapes alone need.
    assert isinstance(residency.parameters(0)["w"], np.ndarray)


def test_a_calls_room_moves_the_models_no_step_uses_off_the_device_least_recent_first() -> (
    None
):
    capacity = 3 * 64 * KIB
    residency = Residency(capacity=capacity)
    residency.add(9, _weights(256), pinned=True)
    for model in range(3):
        residency.add(model, _weights(64, float(model)))

    # The device has 100 KiB free beside what the models hold.
    def free() -> int:
        return 100 * KIB + capacity - residency.cached_bytes()

    with residency.lease([0]):
        # Room for a call of 150 KiB: model 1, the least recent no step
        # uses, moves off; model 0's step keeps it.
        assert residency.spill(free, 150 * KIB) == 1
        assert [residency.resident(model) for model in range(3)] == [True, False, True]
        # Room the device has already moves nothing.
        assert residency.spill(free, 100 * KIB) == 0
        # For a call that ran out of memory, every model no step uses.
        assert residency.spill(free) == 1
        assert [residency.resident(model) for model in range(3)] == [True, False, False]
        assert residency.spill(free) == 0
    assert residency.resident(9)
    assert residency.report().evictions == 2
    # A model moved off comes back as it was when a step needs it.
    with residency.lease([1]):
        np.testing.assert_array_equal(np.asarray(residency.parameters(1)["w"]), 1.0)


def test_a_worker_on_an_accelerator_makes_room_for_a_call_from_its_model_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generalforecast.runtime import budget as runtime_budget

    worker = Worker.__new__(Worker)
    worker.residency = Residency(capacity=2 * 64 * KIB)
    worker.residency.add(0, _weights(64))
    worker.residency.add(1, _weights(64))
    monkeypatch.setattr(runtime_budget, "free_device_bytes", lambda: None)

    # On a CPU, whose device is the host's memory, nothing moves.
    worker.budget = MemoryBudget(GIB, GIB, cpu=True)
    assert not worker._room_for_call(None)
    assert worker.residency.resident(0) and worker.residency.resident(1)

    worker.budget = MemoryBudget(GIB, GIB, cpu=False)
    assert worker._room_for_call(None)
    assert not worker.residency.resident(0) and not worker.residency.resident(1)
    assert not worker._room_for_call(None)


def test_a_pinned_model_is_never_moved_off_the_device_nor_counted_against_the_rest() -> (
    None
):
    residency = Residency(capacity=64 * KIB)
    residency.add(0, _weights(256), pinned=True)
    residency.add(1, _weights(64))
    residency.add(2, _weights(64))

    assert residency.resident(0)
    assert residency.pinned_bytes() == 256 * KIB
    assert residency.cached_bytes() == 64 * KIB
    assert not residency.resident(1)
    with pytest.raises(ValueError, match="pinned"):
        residency.evict(0)
    residency.shrink()
    assert residency.resident(0)


def test_room_waits_for_the_step_using_the_model_it_moves() -> None:
    residency = Residency(capacity=64 * KIB, wait=10.0)
    residency.add(0, _weights(64))
    leased, release = threading.Event(), threading.Event()

    def step() -> None:
        with residency.lease([0]):
            leased.set()
            assert release.wait(10)

    stepping = threading.Thread(target=step)
    stepping.start()
    assert leased.wait(10)
    added: queue.Queue[BaseException | None] = queue.Queue()

    def add() -> None:
        try:
            residency.add(1, _weights(64))
            added.put(None)
        except BaseException as error:  # noqa: BLE001
            added.put(error)

    adding = threading.Thread(target=add)
    adding.start()
    time.sleep(0.2)
    # Model 0 stays on the device while its step runs.
    assert added.empty()
    assert residency.resident(0)

    release.set()
    stepping.join(10)
    assert added.get(timeout=10) is None
    assert not residency.resident(0)
    assert residency.resident(1)


def test_a_model_no_room_can_be_made_for_is_refused_as_the_devices() -> None:
    residency = Residency(capacity=64 * KIB, wait=0.1)
    residency.add(0, _weights(64))

    with residency.lease([0]), pytest.raises(DeviceMemoryRefused) as refused:
        residency.add(1, _weights(32))
    with pytest.raises(DeviceMemoryRefused):
        residency.add(2, _weights(128))

    assert "MiB" in str(refused.value)
    assert not residency.held(1) and not residency.held(2)
    assert residency.report().refusals == 2
    assert residency.resident(0)


def test_a_dropped_host_copy_loads_again_when_next_used() -> None:
    loads: list[None] = []

    def reload() -> Any:
        loads.append(None)
        return _weights(64, 3.0)

    residency = Residency(capacity=64 * KIB, host_capacity=0)
    residency.add(0, _weights(64, 3.0), reload=reload)
    residency.add(1, _weights(64))
    # Model 0 is off the device, its host copy past the host's bound dropped.
    assert not residency.resident(0)
    assert residency.host_bytes() == 0

    with residency.lease([0]):
        np.testing.assert_array_equal(np.asarray(residency.parameters(0)["w"]), 3.0)
    assert loads == [None]
    assert residency.report().reloads == 1


def test_a_leaf_two_models_hold_is_one_buffer() -> None:
    base = _weights(256)
    residency = Residency(capacity=64 * KIB)
    residency.add(0, base, pinned=True)
    fine_tune = {"base": base["w"], "a": jnp.ones((16, 64), jnp.float32)}

    residency.add(1, fine_tune)

    assert residency.own_bytes(1) == 16 * 64 * 4
    assert residency.model_bytes(1) == 256 * KIB + 16 * 64 * 4
    assert residency.device_bytes() == 256 * KIB + 16 * 64 * 4
    assert residency.parameters(1)["base"] is base["w"]


def test_admission_charges_the_models_the_device_holds() -> None:
    # A Mitra-2 slice decoded in its fused program, as admission's own
    # tests size it on an A10G, fits the 7,452 MiB beside the pool; with
    # 5 GiB of models taken from the same budget it does not.
    from tests.test_transient_admission import A10G_ROOM, _unit
    from tests.test_transient_admission import _worker as _admitting

    worker, _ = _admitting()
    pool = worker._pool_bytes
    worker.budget = MemoryBudget(pool + A10G_ROOM, 64 * GIB, cpu=False)
    Admission._fits_device(worker, _unit(5120), fused=True)
    worker.residency = SimpleNamespace(
        capacity=None,
        shrink=lambda: None,
        pinned_bytes=lambda: 4 * GIB,
        cached_bytes=lambda: 0,
    )

    Worker.hold_models(worker, 1 * GIB)

    assert worker.budget.device == pool + A10G_ROOM - 5 * GIB
    with pytest.raises(DeviceMemoryRefused):
        Admission._fits_device(worker, _unit(5120), fused=True)


# Fine-tunes as a hosted plane resolves them: low-rank updates of a tiny
# TabICL's matrices, applied in the program as a LoRA fine-tune's are.
BASE = "general-forecast/tabicl-v2-classifier"
RANK = 2


def _adapted(tree: Any) -> list[int]:
    """The leaves of a base a fine-tune adapts: its matrices wider than the rank both ways."""
    return [
        index
        for index, leaf in enumerate(jax.tree.leaves(tree))
        if getattr(leaf, "ndim", 0) == 2 and min(leaf.shape) > RANK
    ]


def _combine(shared: Any, own: dict[str, tuple[Any, Any]]) -> Any:
    """A bag's parameters: each adapted matrix `W + B @ A`, as `lora.combine` gives them."""
    leaves, treedef = jax.tree.flatten(shared)
    for index, (a, b) in own.items():
        weight = leaves[int(index)]
        leaves[int(index)] = (weight.astype(jnp.float32) + b @ a).astype(weight.dtype)
    return jax.tree.unflatten(treedef, leaves)


def _own(base: Any, rank: int = RANK, seed: int = 7) -> dict[str, tuple[Any, Any]]:
    """One bag's adapters of `rank` over every adapted matrix of `base`, stacked over the bag."""
    leaves = jax.tree.leaves(base)
    keys = jax.random.split(jax.random.key(seed), 2 * len(leaves))
    own = {}
    for index in _adapted(base):
        outputs, inputs = leaves[index].shape
        a = jax.random.normal(keys[2 * index], (1, rank, inputs)) / inputs
        b = jax.random.normal(keys[2 * index + 1], (1, outputs, rank)) / rank
        own[str(index)] = (a, b)
    return own


class _Plane:
    """A resolver keeping one LoRA fine-tune of the tiny base."""

    def __init__(self, own: Any) -> None:
        self.own = own

    def resolve(self, model: str) -> derived.Derived:
        return derived.Derived(
            model, BASE, catalog.ENTRIES[BASE].artifact.digest, "sha256:" + "0" * 64
        )

    def weights(self, made: derived.Derived, base: Any) -> derived.Weights:
        return derived.Weights(self.own, _combine)


class _Planes:
    """A resolver keeping LoRA fine-tunes of the tiny base by spec, each its own rank."""

    def __init__(self, own: dict[str, Any]) -> None:
        self.own = own

    def resolve(self, model: str) -> derived.Derived:
        return derived.Derived(
            model, BASE, catalog.ENTRIES[BASE].artifact.digest, "sha256:" + "0" * 64
        )

    def weights(self, made: derived.Derived, base: Any) -> derived.Weights:
        # Fresh arrays each load, as a plane's are read from their files, so
        # nothing but the residency keeps them once a load is done.
        own = jax.tree.map(jnp.array, self.own[made.model])
        return derived.Weights(own, _combine)


@pytest.fixture
def plane() -> Any:
    found = _Plane(_own(tiny_tabicl(regression=False)))
    derived.use(found)
    yield found
    derived.use(None)


def _predict(worker: Worker, model: int) -> np.ndarray:
    """A tiny classifier's predictions for one table of `model`, fused with its queries."""
    data = conformance_table(False, rows=48, queries=6)
    if not worker.allocated:
        worker.allocate(256, 0, 0)
    table = 10 + model
    worker.add_table(table, model, data.num_classes, FEATURES)
    key = CacheKey(table, 0, 0, "context", 1)
    rows = np.arange(48, dtype=np.uint32)
    unit = protocol.Unit(
        table,
        key,
        rows,
        "fuse",
        [],
        np.asarray(data.features),
        np.asarray(data.targets),
    )
    queries = np.asarray(data.queries)
    batch = protocol.QueryBatch(model, table, 0, queries, [0] * len(queries))
    predictions, failed = worker.run_apart(protocol.Step([], [], [], [unit], [batch]))
    assert failed == {}
    return predictions[model]


def test_a_lora_fine_tune_runs_over_its_loaded_base_as_a_delta(plane: Any) -> None:
    spec = "acme/credit@ft1"
    base = tiny_tabicl(regression=False)
    # As a worker loaded fine-tunes before: over a copy of the base its own.
    merged = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    copy = jax.tree.map(jnp.array, base)
    made = catalog.resolve_derived(catalog.parse(spec))
    merged.add_model(0, tabicl_functions, base)
    merged.add_model(1, *registry._derive(made, tabicl_functions, copy, None, None))
    # Now: over the base the worker serves, by reference.
    shared = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    shared.add_model(0, tabicl_functions, base)
    bases = {BASE: (tabicl_functions, shared.residency.parameters(0))}
    shared.residency.capacity = sum(a.nbytes + b.nbytes for a, b in plane.own.values())
    shared.add_model(1, *registry.load_engine_model(spec, bases=bases), pinned=False)

    expected = _predict(merged, 1)
    found = _predict(shared, 1)

    np.testing.assert_allclose(found, expected, rtol=1e-6, atol=1e-7)
    # The adapters change what the base predicts.
    assert not np.allclose(found, _predict(shared, 0), atol=1e-4)
    adapters = sum(a.nbytes + b.nbytes for a, b in plane.own.values())
    base_bytes = shared.residency.model_bytes(0)
    # The fine-tune adds its adapters alone, within a bound of just them.
    assert shared.residency.own_bytes(1) == adapters
    assert shared.parameter_bytes() == base_bytes + adapters
    # Before, the base twice.
    assert merged.parameter_bytes() == 2 * base_bytes + adapters
    assert adapters < base_bytes / 4


def test_a_model_moved_off_the_device_predicts_as_it_did() -> None:
    worker = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    parameters = tiny_tabicl(regression=False)
    worker.add_model(0, tabicl_functions, parameters, pinned=False)
    before = _predict(worker, 0)
    worker.residency.evict(0)
    assert not worker.residency.resident(0)

    after = _predict(worker, 0)

    np.testing.assert_array_equal(after, before)
    assert worker.residency.resident(0)


def test_a_step_whose_model_cannot_fit_fails_its_table_as_the_devices() -> None:
    worker = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    worker.residency.wait = 0.1
    worker.add_model(0, tabicl_functions, tiny_tabicl(regression=False))
    worker.add_model(1, tabicl_functions, tiny_tabicl(regression=False), pinned=False)
    pinned = _predict(worker, 0)
    worker.residency.evict(1)
    worker.residency.capacity = 0
    data = conformance_table(False, rows=48, queries=6)
    worker.add_table(11, 1, data.num_classes, FEATURES)
    rows = np.arange(48, dtype=np.uint32)
    features, targets = np.asarray(data.features), np.asarray(data.targets)
    queries = np.asarray(data.queries)
    step = protocol.Step(
        [],
        [],
        [],
        [
            protocol.Unit(
                10,
                CacheKey(10, 0, 1, "context", 1),
                rows,
                "fuse",
                [],
                features,
                targets,
            ),
            protocol.Unit(
                11,
                CacheKey(11, 0, 1, "context", 1),
                rows,
                "fuse",
                [],
                features,
                targets,
            ),
        ],
        [
            protocol.QueryBatch(0, 10, 0, queries, [0] * len(queries)),
            protocol.QueryBatch(1, 11, 1, queries, [1] * len(queries)),
        ],
    )

    predictions, failed = worker.run_apart(step)

    # The pinned model's table still answers.
    np.testing.assert_array_equal(predictions[0], pinned)
    assert list(failed) == [11]
    assert failed[11].startswith("DeviceMemoryRefused: ")
    assert worker.failure_kinds == {11: "device_memory"}
    frame, _ = protocol.predictions(predictions, failed, kinds=worker.failure_kinds)
    assert frame["failures"][0]["kind"] == "device_memory"


def test_a_load_that_cannot_fit_is_reported_as_the_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generalforecast import worker_process

    def load(spec: str, bases: Any = None) -> tuple[Any, Any]:
        return tabicl_functions, tiny_tabicl(regression=False)

    monkeypatch.setattr(registry, "load_engine_model", load)
    worker = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    worker.add_model(0, tabicl_functions, tiny_tabicl(regression=False))
    worker.residency.capacity = 1024
    worker.residency.wait = 0.1
    names = {"served": 0}
    sent: queue.Queue[dict[str, Any]] = queue.Queue()
    loads = worker_process._Loads(worker, names, sent.put, lambda *_: None)
    loads.ask("tuned", "tiny")
    loads.start()

    reported = sent.get(timeout=30)

    assert reported["failure"].startswith("DeviceMemoryRefused: ")
    assert reported["models"]["refusals"] == 1
    assert reported["models"]["resident"] == 1
    assert "tuned" not in names


def test_a_worker_loads_a_fine_tune_over_the_base_it_serves(plane: Any) -> None:
    from generalforecast import worker_process

    worker = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
    worker.add_model(0, tabicl_functions, tiny_tabicl(regression=False))
    base_bytes = worker.parameter_bytes()
    names, bases = {BASE: 0}, {BASE: 0}
    sent: queue.Queue[dict[str, Any]] = queue.Queue()
    loads = worker_process._Loads(worker, names, sent.put, lambda *_: None, bases)
    loads.ask("acme/credit@ft1", "acme/credit@ft1")
    loads.start()

    reported = sent.get(timeout=60)

    assert "failure" not in reported, reported
    adapters = sum(a.nbytes + b.nbytes for a, b in plane.own.values())
    assert worker.residency.own_bytes(names["acme/credit@ft1"]) == adapters
    assert worker.parameter_bytes() == base_bytes + adapters
    assert reported["models"]["models"] == 2
    # Not pinned, it shares the room models not pinned have, by its adapters.
    assert not worker.residency.pinned(1)
    assert reported["models"]["cached_bytes"] == adapters


def _as_on_an_accelerator(
    residency: Residency, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holds parameters as a worker on an accelerator does, on a CPU.

    A load there stages its parameters in host memory; the residency keeps
    its own copy of each, puts that on the device as a copy, and the staged
    arrays are freed once the load is done. A CPU's `device_put` aliases the
    host copy instead, which would keep the staged arrays alive.
    """
    residency._accelerator = True
    put = jax.device_put
    monkeypatch.setattr(
        jax, "device_put", lambda x, *args, **kwargs: put(np.array(x), *args, **kwargs)
    )


def test_an_array_freed_after_its_load_never_names_another_models_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    residency = Residency()
    _as_on_an_accelerator(residency, monkeypatch)

    staged: list[tuple[int, weakref.ref[jax.Array]]] = []

    def load(model: int, weight: jax.Array) -> None:
        # Its staged array is freed as this returns: its id is soon another's.
        staged.append((id(weight), weakref.ref(weight)))
        residency.add(model, {"w": weight})

    load(0, jnp.zeros((128, 128)))
    first, alive = staged[0]
    assert alive() is None
    # A new array where the first load's staged one was, as CPython soon
    # reuses a freed object's memory: another model's, of another shape.
    # Where this allocator does not reuse it, the loads after still might.
    made = []
    for _ in range(2_000):
        made.append(jnp.ones((3, 5)))
        if id(made[-1]) == first:
            break
    reused = made.pop()
    del made
    load(1, reused)
    del reused

    for model in range(2, 64):
        width = 8 + model
        load(model, jnp.full((width, width), float(model)))

    assert residency.parameters(0)["w"].shape == (128, 128)
    assert residency.parameters(1)["w"].shape == (3, 5)
    np.testing.assert_array_equal(np.asarray(residency.parameters(1)["w"]), 1.0)
    for model in range(2, 64):
        width = 8 + model
        weight = residency.parameters(model)["w"]
        assert weight.shape == (width, width), model
        np.testing.assert_array_equal(np.asarray(weight), float(model))
    assert residency.device_bytes() == 4 * (128**2 + 3 * 5) + sum(
        4 * (8 + m) ** 2 for m in range(2, 64)
    )


def test_fine_tunes_of_other_ranks_loaded_in_turn_each_run_their_own_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generalforecast import worker_process

    base = tiny_tabicl(regression=False)
    own = {
        f"acme/credit@ft{index}": _own(base, rank=1 + index % 4, seed=7 + index)
        for index in range(1, 9)
    }
    derived.use(_Planes(own))
    try:
        worker = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
        worker.add_model(0, tabicl_functions, base)
        _as_on_an_accelerator(worker.residency, monkeypatch)
        names, bases = {BASE: 0}, {BASE: 0}
        sent: queue.Queue[dict[str, Any]] = queue.Queue()
        loads = worker_process._Loads(worker, names, sent.put, lambda *_: None, bases)
        for spec in own:
            loads.ask(spec, spec)
            loads.start()
            reported = sent.get(timeout=60)
            assert "failure" not in reported, reported

        for spec, adapters in own.items():
            # Each over its own copy of the base, as before residency.
            alone = Worker(slot_bytes=4096, compile_cache=False, compile_ahead=False)
            made = catalog.resolve_derived(catalog.parse(spec))
            copy = jax.tree.map(jnp.array, base)
            alone.add_model(0, tabicl_functions, base)
            alone.add_model(
                1, *registry._derive(made, tabicl_functions, copy, None, None)
            )
            model = names[spec]
            assert worker.residency.own_bytes(model) == sum(
                a.nbytes + b.nbytes for a, b in adapters.values()
            ), spec
            np.testing.assert_allclose(
                _predict(worker, model), _predict(alone, 1), rtol=1e-6, atol=1e-7
            )
    finally:
        derived.use(None)
