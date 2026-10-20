"""Local mode: the node, the worker and the client API in one process."""

from __future__ import annotations

import dataclasses
import datetime
import threading
import time as clock
import warnings
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import generalforecast
from generalforecast import protocol
from generalforecast.columns import MAX_WORDS, Column, datetime_parts
from generalforecast.models import registry
from generalforecast.models.kumo.model import kumo_functions
from generalforecast.models.tabicl.model import tabicl_functions
from generalforecast.predict import predict
from generalforecast.runtime.device import device_settings
from generalforecast.runtime.memory import NoMemoryAnalysis, memory_profile
from generalforecast.runtime.paging import DevicePool, read_context
from generalforecast.worker import Worker
from tests.models.conformance.cases import CASES, Case, table
from tests.models.kumo.tiny import tiny_kumo
from tests.models.tabicl.tiny import classifier, tiny_tabicl
from tests.models.timestral.tiny import tiny_timestral

PAGE = 8
NEAR_A = np.zeros((2, 5), dtype=np.float32)
NEAR_B = np.full((2, 5), 10.0, dtype=np.float32)


def _clusters(rows: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Rows around 0 (even rows) and around 10 (odd rows), with three classes."""
    generator = np.random.default_rng(0)
    centers = np.where(np.arange(rows) % 2 == 0, 0.0, 10.0)[:, None]
    features = (centers + generator.normal(size=(rows, 5))).astype(np.float32)
    return features, (np.arange(rows) % 3).astype(np.int32)


def _table(
    client: generalforecast.LocalClient, name: str = "t", strategy=None, model=None
) -> generalforecast.Table:
    return client.create_table(
        name,
        model=model or (tabicl_functions, tiny_tabicl(regression=False)),
        features=5,
        classes=3,
        strategy=strategy or generalforecast.Strategy.exact(),
        page_size=PAGE,
    )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_local_predictions_match_predict(case: Case) -> None:
    functions, model = case.functions(), case.build(0)
    data = table(case.regression, rows=40)
    client = generalforecast.local()
    rows = client.create_table(
        "rows",
        model=(functions, model),
        features=5,
        classes=data.num_classes or None,
        page_size=PAGE,
    )
    rows.insert(np.arange(40), np.asarray(data.features), np.asarray(data.targets))

    actual = rows.predict(np.asarray(data.queries))

    expected = predict(functions, model, data.features, data.targets, data.queries)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_repeated_queries_reuse_the_encoded_context() -> None:
    client = generalforecast.local()
    rows = _table(client)
    rows.insert(np.arange(32), *_clusters())

    first = rows.predict(NEAR_A)
    second = rows.predict(NEAR_A)

    stats = client.stats
    assert (stats.encoded, stats.reused) == (1, 1)
    assert (stats.rows_encoded, stats.rows_reused) == (32, 32)
    assert (stats.steps, stats.queries) == (2, 2 * len(NEAR_A))
    assert stats.seconds.keys() == stats.calls.keys()
    np.testing.assert_array_equal(first, second)


def test_queries_near_one_cluster_share_an_encode() -> None:
    client = generalforecast.local()
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(16))
    rows.insert(np.arange(32), *_clusters())

    probabilities = rows.predict(np.concatenate([NEAR_A, NEAR_B]))

    assert probabilities.shape == (4, 3)
    assert (client.stats.encoded, client.stats.rows_encoded) == (2, 32)


def test_evicted_contexts_come_back_without_reencoding() -> None:
    model = tiny_tabicl(regression=False)
    unit = memory_profile(tabicl_functions, model, 5).context_bytes(16)
    client = generalforecast.local(
        device_bytes=unit, host_bytes=4 * unit, slot_bytes=unit // 2
    )
    rows = _table(
        client,
        strategy=generalforecast.Strategy.neighbors(16),
        model=(tabicl_functions, model),
    )
    rows.insert(np.arange(32), *_clusters())

    first = rows.predict(NEAR_A)
    rows.predict(NEAR_B)
    again = rows.predict(NEAR_A)

    assert (client.stats.encoded, client.stats.promoted) == (2, 1)
    np.testing.assert_array_equal(first, again)


def test_moves_between_tiers_compile_once_per_part_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An accelerator's pool, whose moves run compiled programs, on this CPU.
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    pool = DevicePool(4, 160)
    compiles: list[float] = []
    listening = [True]
    # Other tests' clients may still compile ahead on threads of their own.
    mine = threading.get_ident()

    def compiled(event: str, duration_secs: float, **_: str | int) -> None:
        if (
            listening[0]
            and threading.get_ident() == mine
            and event == "/jax/core/compile/backend_compile_duration"
        ):
            compiles.append(duration_secs)

    jax.clear_caches()
    jax.monitoring.register_event_duration_secs_listener(compiled)
    for count, first in [
        *((count, count) for count in range(1, 10)),
        (70, 10),
        (130, 20),
    ]:
        slots = list(range(first, first + count))
        data = np.arange(count * 4, dtype=np.uint8).reshape(count, 4)
        pool.write(slots, data)
        np.testing.assert_array_equal(pool.read_host(slots), data)
    # The pool's last slots: the padding past its end is dropped.
    pool.write([156], np.full((1, 4), 7, np.uint8))
    pool.write([157, 158, 159], np.full((3, 4), 9, np.uint8))
    moved = pool.read_host(list(range(156, 160)))
    listening[0] = False

    np.testing.assert_array_equal(moved[:, 0], [7, 9, 9, 9])
    assert pool.read_host([]).shape == (0, 4)
    # Parts of 64 slots, and last parts padded to 1, 2, 4, 8 or 16 of them:
    # a read and a write program for each of 6 sizes.
    assert len(compiles) == 12


def test_contexts_spilled_to_disk_come_back_without_reencoding(tmp_path: Path) -> None:
    model = tiny_tabicl(regression=False)
    unit = memory_profile(tabicl_functions, model, 5).context_bytes(16)
    slot = unit // 4
    # One context on the device; one context and the tables' pages on the host.
    client = generalforecast.local(
        device_bytes=unit,
        host_bytes=unit + 6 * slot,
        slot_bytes=slot,
        disk_bytes=4 * unit,
        disk_path=tmp_path / "cache.slots",
    )
    tables = []
    for index in range(3):
        features, targets = _clusters(16)
        table = _table(client, name=f"t{index}", model=(tabicl_functions, model))
        table.insert(np.arange(16), features + index, targets)
        tables.append(table)

    first = tables[0].predict(NEAR_A)
    for table in tables[1:]:
        table.predict(NEAR_A)
    again = tables[0].predict(NEAR_A)

    assert client.stats.encoded == 3
    assert client.stats.spilled >= 1 and client.stats.promoted >= 1
    np.testing.assert_array_equal(first, again)


def test_updates_reencode_only_the_contexts_holding_the_row() -> None:
    client = generalforecast.local()
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(16))
    features, targets = _clusters()
    rows.insert(np.arange(32), features, targets)
    before = rows.predict(np.concatenate([NEAR_A, NEAR_B]))

    rows.update([0], features[[0]] + 0.5, targets[[0]])
    after = rows.predict(np.concatenate([NEAR_A, NEAR_B]))

    assert (client.stats.encoded, client.stats.reused) == (3, 1)
    assert not np.allclose(after[:2], before[:2])
    np.testing.assert_array_equal(after[2:], before[2:])


@pytest.mark.parametrize("strategy", ["exact", "neighbors"])
def test_rows_wider_than_a_slot_span_slots(strategy: str) -> None:
    model = tiny_tabicl(regression=False)
    row = memory_profile(tabicl_functions, model, 5).cache_bytes_per_row
    features, targets = _clusters(64)
    chosen = (
        generalforecast.Strategy.exact()
        if strategy == "exact"
        else generalforecast.Strategy.neighbors(8)
    )
    spread = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32) * 5
    queries = np.concatenate([NEAR_A, NEAR_B, spread])
    predictions = []
    # Odd slot sizes, so the bytes of a padding block land misaligned.
    for slot_bytes in (1 << 20, row // 3 + 1, 257):
        client = generalforecast.local(slot_bytes=slot_bytes)
        rows = _table(client, strategy=chosen, model=(tabicl_functions, model))
        rows.insert(np.arange(64), features, targets)
        predictions.append(rows.predict(queries))
        np.testing.assert_allclose(rows.predict(queries), predictions[-1])
        assert client.stats.reused > 0

    for predicted in predictions[1:]:
        np.testing.assert_allclose(predicted, predictions[0], rtol=1e-5, atol=1e-6)


def test_pools_never_pass_the_nodes_slots() -> None:
    client = generalforecast.local(
        device_bytes=64 << 10, host_bytes=256 << 10, slot_bytes=4096
    )
    features, targets = _clusters()
    _table(client).insert(np.arange(32), features, targets)

    client.table("t").predict(NEAR_A)

    # On a CPU the device pool is the tier's slots, committed as they are written.
    assert 0 < client._worker._device.data.shape[0] <= 16
    assert client._worker._host.data.shape == (64, 4096)
    with pytest.raises(ValueError, match="past the pool"):
        client._worker._device.write([16], jnp.zeros((1, 4096), jnp.uint8))


def test_contexts_past_the_models_length_warn_once_or_are_clamped() -> None:
    model = tiny_tabicl(regression=False)
    # The tiny model as if it were trained on contexts of 8 rows.
    functions = tabicl_functions._replace(
        properties=lambda parameters: dataclasses.replace(
            tabicl_functions.properties(parameters), context_rows=8
        )
    )
    client = generalforecast.local()
    tables = {
        past_cap: client.create_table(
            past_cap,
            model=(functions, model),
            features=5,
            classes=3,
            page_size=PAGE,
            past_cap=past_cap,
        )
        for past_cap in ("warn", "clamp")
    }
    for handle in tables.values():
        handle.insert(np.arange(32), *_clusters())

    with pytest.warns(UserWarning, match="'warn' read a context of more than 8 rows"):
        tables["warn"].predict(NEAR_A)
    encoded = client.stats.rows_encoded
    tables["clamp"].predict(NEAR_A)

    assert client.stats.rows_encoded - encoded == 8
    described = tables["clamp"].describe()
    assert (described.past_cap, described.context_rows) == ("clamp", 8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tables["warn"].predict(NEAR_B)


def test_contexts_whose_calls_would_not_fit_the_device_are_refused() -> None:
    model = tiny_tabicl(regression=False)
    client = generalforecast.local()
    worker = client._worker
    features, targets = _clusters(256)
    whole = _table(client, name="whole", model=(tabicl_functions, model))
    whole.insert(np.arange(256), features, targets)
    nearest = _table(
        client,
        name="nearest",
        model=(tabicl_functions, model),
        strategy=generalforecast.Strategy.neighbors(16),
    )
    nearest.insert(np.arange(256), features, targets)
    (loaded,) = worker._models
    size = worker._size(256)
    row = memory_profile(tabicl_functions, model, 5).cache_bytes_per_row
    decode = worker.decode_profile(loaded).working_bytes(row, size)
    encode = worker.encode_bytes(loaded, 5, size)
    assert encode is not None
    # Beside its pool, a device that holds a context of 256 rows as the model
    # caches it and the decode of the fewest queries against it, but not the
    # context's encode.
    held = max(size * row, decode)
    assert encode > held
    room = (held + encode) // 2
    worker.budget = dataclasses.replace(worker.budget, device=worker._pool_bytes + room)

    # The exact context is refused before its encode, which would run out of
    # memory; each query's nearest rows read fine.
    with pytest.raises(ValueError, match="encoding it"):
        whole.predict(NEAR_A)
    assert client.stats.encoded == 0
    assert nearest.predict(NEAR_A).shape == (2, 3)


def test_representations_are_reused_until_their_page_changes() -> None:
    client = generalforecast.local()
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(16))
    features, targets = _clusters()
    rows.insert(np.arange(32), features, targets)

    rows.predict(NEAR_A)
    rows.predict(NEAR_B)
    from_the_fit = client.stats.rows_embedded
    rows.update([0], features[[0]], (targets[[0]] + 1) % 3)
    rows.predict(NEAR_A)

    assert from_the_fit == 0
    assert 0 < client.stats.rows_embedded <= PAGE


def test_writes_after_a_fit_reembed_pages_without_compiling() -> None:
    jax.config.update("jax_enable_compilation_cache", False)
    compiled: list[str] = []

    def compile_event(event: str, duration_secs: float, **details: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(details.get("fun_name", ""))

    jax.monitoring.register_event_duration_secs_listener(compile_event)
    try:
        client = generalforecast.local()
        rows = _table(client)
        features, targets = _clusters()
        rows.insert(np.arange(24), features[:24], targets[:24])
        rows.predict(NEAR_A)
        client._worker.wait_warm()
        compiled.clear()
        # Rows that fill and split pages, then a change to rows of two pages.
        rows.insert(np.arange(24, 32), features[24:], targets[24:])
        rows.predict(NEAR_A)
        rows.update([0, 1], features[[1, 0]], targets[[0, 1]])
        rows.predict(NEAR_A)
    finally:
        jax.monitoring.unregister_event_duration_listener(compile_event)
        jax.config.update("jax_enable_compilation_cache", True)

    assert client.stats.rows_embedded > 0
    assert "jit(embed)" not in compiled


def test_appends_that_change_a_querys_neighbors_need_no_new_decode() -> None:
    client = generalforecast.local()
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(32))
    features, targets = _clusters(96)
    rows.insert(np.arange(48), features[:48], targets[:48])
    rows.predict(NEAR_A)
    programs = client._worker.programs

    def decodes() -> set:
        done = programs._done
        return {m for m in done if isinstance(m, tuple) and m[0] == "decodes"}

    before = decodes()
    # As a backtest appends a few rows before each cutoff: pages fill and
    # split, so the query's nearest rows span other slots, but no more rows.
    for start in range(48, 96, 4):
        end = start + 4
        rows.insert(np.arange(start, end), features[start:end], targets[start:end])
        rows.predict(NEAR_A)

    # A paged decode's shape follows its rows, not the slots they span, so an
    # accelerator compiles, measures and warms no new one.
    assert before and decodes() == before


def test_steps_of_more_queries_after_a_decode_compile_no_decode() -> None:
    jax.config.update("jax_enable_compilation_cache", False)
    compiled: list[str] = []

    def compile_event(event: str, duration_secs: float, **details: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(details.get("fun_name", ""))

    jax.monitoring.register_event_duration_secs_listener(compile_event)
    try:
        client = generalforecast.local()
        rows = _table(client)
        features, targets = _clusters()
        rows.insert(np.arange(32), features, targets)
        rows.predict(NEAR_A)
        client._worker.wait_warm()
        compiled.clear()
        # As a step merging many callers' requests would decode.
        rows.predict(np.repeat(NEAR_A, 10, axis=0))
    finally:
        jax.monitoring.unregister_event_duration_listener(compile_event)
        jax.config.update("jax_enable_compilation_cache", True)

    assert "jit(decode)" not in compiled
    assert "jit(embed)" not in compiled


def test_a_new_number_of_queries_in_a_bucket_compiles_nothing() -> None:
    jax.config.update("jax_enable_compilation_cache", False)
    compiled: list[str] = []

    def compile_event(event: str, duration_secs: float, **details: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(details.get("fun_name", ""))

    jax.monitoring.register_event_duration_secs_listener(compile_event)
    try:
        client = generalforecast.local()
        rows = _table(client)
        features, targets = _clusters()
        rows.insert(np.arange(32), features, targets)
        rows.predict(np.repeat(NEAR_A, 9, axis=0))
        client._worker.wait_warm()
        compiled.clear()
        # Backtests ask a few more or fewer queries at each cutoff.
        rows.predict(np.repeat(NEAR_A, 11, axis=0))
    finally:
        jax.monitoring.unregister_event_duration_listener(compile_event)
        jax.config.update("jax_enable_compilation_cache", True)

    assert compiled == []


def test_a_new_number_of_tables_in_a_step_compiles_nothing_per_state_leaf() -> None:
    jax.config.update("jax_enable_compilation_cache", False)
    compiled: list[str] = []

    def compile_event(event: str, duration_secs: float, **details: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(details.get("fun_name", ""))

    jax.monitoring.register_event_duration_secs_listener(compile_event)
    try:
        client = generalforecast.local()
        model = (tabicl_functions, tiny_tabicl(regression=False))
        tables = [_table(client, name, model=model) for name in "abc"]
        features, targets = _clusters()
        for rows in tables:
            rows.insert(np.arange(32), features, targets)
            rows.predict(NEAR_A)
        client._worker.wait_warm()
        compiled.clear()
        # One step for three tables stacks their states and queries into
        # one call of each stage, for a number of items not seen before.
        client.predict({rows.name: NEAR_A for rows in tables})
    finally:
        jax.monitoring.unregister_event_duration_listener(compile_event)
        jax.config.update("jax_enable_compilation_cache", True)

    # Each stage's program for the new count, with the stack of its inputs
    # and the item taken from its outputs, and the stack of the decode's
    # query rows: nothing per leaf of a table's state.
    assert len(compiled) <= 8
    assert compiled.count("jit(stack)") <= 1


@pytest.mark.parametrize("kind", ["random", "stratified", "balanced", "diverse"])
def test_a_retrieved_context_of_every_row_predicts_as_the_exact_one(
    kind: Literal["random", "stratified", "balanced", "diverse"],
) -> None:
    client = generalforecast.local()
    features, targets = _clusters()
    exact = _table(client, name="exact")
    retrieved = _table(
        client,
        name="retrieved",
        strategy=generalforecast.Strategy.retrieved(16, core=16, kind=kind),
    )
    for handle in (exact, retrieved):
        handle.insert(np.arange(32), features, targets)

    queries = np.concatenate([NEAR_A, NEAR_B])
    np.testing.assert_allclose(
        retrieved.predict(queries), exact.predict(queries), rtol=1e-5, atol=1e-5
    )
    with pytest.raises(ValueError, match="core or neighbors"):
        generalforecast.Strategy.retrieved(0)
    with pytest.raises(ValueError, match="core kind"):
        generalforecast.Strategy.subsample(8, 0, kind=cast(Any, "nearest"))


def test_contexts_are_the_rows_predictions_read_and_encode_nothing() -> None:
    client = generalforecast.local()
    table = _table(client, strategy=generalforecast.Strategy.neighbors(4))
    features, targets = _clusters()
    table.insert(np.arange(32), features, targets)

    contexts = table.contexts(np.concatenate([NEAR_A, NEAR_B]))

    assert client.stats.encoded == 0
    # Rows near A are the even keys, near B the odd ones.
    assert [set(keys[:, 0] % 2) for keys in contexts] == [{0}, {0}, {1}, {1}]
    assert all(keys.shape == (4, 2) for keys in contexts)


def test_retrieval_columns_place_rows_and_the_model_never_reads_them() -> None:
    client = generalforecast.local()
    features, targets = _clusters()
    columns = {f"f{index}": "numeric" for index in range(5)}
    tables = [
        client.create_table(
            name,
            model=(tabicl_functions, tiny_tabicl(regression=False)),
            columns={**columns, "r": "retrieval"},
            classes=3,
            strategy=strategy,
        )
        for name, strategy in (
            ("near", generalforecast.Strategy.neighbors(4)),
            ("all", generalforecast.Strategy.exact()),
        )
    ]
    # Even rows sit near A in features but near 10 in the retrieval column.
    placed = np.where(np.arange(32) % 2 == 0, 10.0, 0.0).astype(np.float32)
    for made in tables:
        made.insert(np.arange(32), np.column_stack([features, placed]), targets)
    queries = np.column_stack([np.concatenate([NEAR_A, NEAR_A]), [10, 10, 0, 0]])

    contexts = tables[0].contexts(queries)

    assert [set(keys[:, 0] % 2) for keys in contexts] == [{0}, {0}, {1}, {1}]
    plain = _table(client, "plain")
    plain.insert(np.arange(32), features, targets)
    np.testing.assert_allclose(
        tables[1].predict(queries[:2]), plain.predict(NEAR_A), rtol=1e-5
    )
    assert tables[0].describe().columns[-1] == Column("r", "retrieval")
    with pytest.raises(ValueError, match=r"retrieval columns \['r'\] too"):
        tables[0].predict(NEAR_A)
    with pytest.raises(ValueError, match="after every other column"):
        client.create_table(
            "first",
            model=(tabicl_functions, tiny_tabicl(regression=False)),
            columns={"r": "retrieval", **columns},
            classes=3,
        )


def test_target_encoding_places_categories_by_their_rows_targets() -> None:
    client = generalforecast.local()
    # Categories 0 and 9 hold class 0 alone; the others class 1 or 2.
    codes = np.arange(32) % 10
    targets = np.where(np.isin(codes, [0, 9]), 0, 1 + codes % 2).astype(np.int32)
    rows = np.column_stack([np.zeros(32), codes]).astype(np.float32)
    contexts = {}
    for encoding in ("codes", "target"):
        table = client.create_table(
            encoding,
            model=(tabicl_functions, tiny_tabicl(regression=False)),
            columns={"f": "numeric", "c": "categorical"},
            classes=3,
            strategy=generalforecast.Strategy.neighbors(7),
            retrieval_categoricals=encoding,
        )
        table.insert(np.arange(32), rows, targets)
        contexts[encoding] = set(table.contexts(rows[:1])[0][:, 0] % 10)

    assert contexts == {"codes": {0, 1}, "target": {0, 9}}


@pytest.mark.parametrize(
    "strategy",
    [
        generalforecast.Strategy.exact(),
        generalforecast.Strategy.subsample(10, kind="stratified"),
        generalforecast.Strategy.retrieved(6, core=4, last=20),
        generalforecast.Strategy.neighbors(8),
    ],
    ids=["exact", "stratified", "recent", "neighbors"],
)
@pytest.mark.parametrize("before", [None, [50, 80, 101]], ids=["all", "cutoffs"])
def test_rows_wait_out_of_contexts_until_labeled(
    strategy: generalforecast.Strategy, before: list[int] | None
) -> None:
    client = generalforecast.local()
    generator = np.random.default_rng(0)
    features, targets = _clusters(52)
    times = generator.integers(0, 100, 52)
    queries = generator.normal(size=(3, 5)).astype(np.float32) * 3
    names = np.array(["cat", "dog", "fish"])[targets]
    waiting = _table(client, "waiting", strategy=strategy)
    # More rows wait for labels than have them.
    waiting.insert(np.arange(52), features, [*names[:20], *[None] * 32], times)
    arriving = _table(client, "arriving", strategy=strategy)
    arriving.insert(np.arange(20), features[:20], names[:20], times[:20])

    def same() -> None:
        for got, want in zip(
            waiting.contexts(queries, before),
            arriving.contexts(queries, before),
            strict=True,
        ):
            np.testing.assert_array_equal(np.sort(got[:, 0]), np.sort(want[:, 0]))
        np.testing.assert_allclose(
            waiting.predict(queries, before),
            arriving.predict(queries, before),
            rtol=1e-4,
            atol=1e-5,
        )

    # Unlabeled, the rows are as if not yet inserted; labeled, as if inserted
    # then, with their labels.
    same()
    waiting.label(np.arange(20, 52), names[20:])
    arriving.insert(np.arange(20, 52), features[20:], names[20:], times[20:])
    same()


def test_a_model_with_a_target_mask_reads_unlabeled_rows() -> None:
    from generalforecast.models.limix.model import limix_functions
    from tests.models.limix.tiny import tiny_limix

    client = generalforecast.local()
    features, targets = _clusters(40)
    table = client.create_table(
        "masked",
        model=(limix_functions, tiny_limix(regression=False)),
        features=5,
        classes=3,
        strategy=generalforecast.Strategy.neighbors(24),
    )
    targets[32:] = -1
    table.insert(np.arange(40), features, targets)

    contexts = table.contexts(NEAR_A)

    assert all((keys[:, 0] >= 32).any() for keys in contexts)
    assert np.isfinite(table.predict(NEAR_A)).all()


def test_rows_without_labels_are_refused_and_take_nulls() -> None:
    import pyarrow as pa

    client = generalforecast.local()
    features, _ = _clusters()
    table = _table(client, "unlabeled")
    # Nulls are labels not known yet, in an Arrow array too.
    table.insert(np.arange(32), features, pa.array([None] * 32, pa.int32()))
    with pytest.raises(ValueError, match="none labeled"):
        table.predict(NEAR_A)
    with pytest.raises(ValueError, match="NaN for rows not labeled yet"):
        client.create_table(
            "values",
            model=(tabicl_functions, tiny_tabicl(regression=True)),
            features=5,
        ).insert([0], features[:1], [np.inf])


def test_given_rows_predict_as_a_table_of_those_rows() -> None:
    client = generalforecast.local()
    features, targets = _clusters()
    model = (tabicl_functions, tiny_tabicl(regression=False))
    tables = {
        summary: client.create_table(
            summary, model=model, features=5, classes=3, summary=summary
        )
        for summary in ("table", "context")
    }
    for made in tables.values():
        made.insert(np.arange(32), features, targets)
    chosen = [[1, 3, 5, 7], [0, 2, 4, 6, 8]]

    # Summarizing each context, a query's state is its rows' alone.
    given = tables["context"].predict_given(NEAR_A, chosen)
    for query, keys in enumerate(chosen):
        alone = client.create_table(
            f"alone{query}", model=model, features=5, classes=3, summary="context"
        )
        alone.insert(keys, features[keys], targets[keys])
        np.testing.assert_allclose(
            given[query], alone.predict(NEAR_A[query : query + 1])[0], rtol=1e-5
        )
    # With one state for the table, every row given is the exact context.
    every = [list(range(32))] * len(NEAR_A)
    np.testing.assert_allclose(
        tables["table"].predict_given(NEAR_A, every),
        tables["table"].predict(NEAR_A),
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="2 contexts"):
        tables["table"].predict_given(NEAR_A, chosen[:1])
    with pytest.raises(KeyError, match="99"):
        tables["table"].predict_given(NEAR_A, [[99], [1]])


def test_a_regions_queries_share_a_context_rebuilt_once_they_are_answered() -> None:
    client = generalforecast.local()
    table = client.create_table(
        "t",
        model=(tabicl_functions, tiny_tabicl(regression=False)),
        features=5,
        classes=3,
        strategy=generalforecast.Strategy.neighbors(8),
        page_size=PAGE,
        region_rows=16,
        rebuild=0.25,
    )
    features, targets = _clusters()
    table.insert(np.arange(32), features, targets)
    table.predict(np.concatenate([NEAR_A, NEAR_A + 0.01]))
    encoded = client.stats.encoded
    # Rows at A, among the region's nearest, past a quarter of its context.
    table.insert(np.arange(32, 36), np.zeros((4, 5), np.float32), targets[:4])
    churn = table.region_churn(np.concatenate([NEAR_A, NEAR_B]))

    stale = table.predict(NEAR_A)
    rebuilt = client.stats.encoded - encoded
    fresh = table.predict(NEAR_A)

    # Four queries shared one context; the stale one answered, then the
    # rebuild ran in a step of its own, and later queries read it.
    assert encoded == 1 and rebuilt == 1
    assert not np.allclose(stale, fresh)
    stats = table.describe().region_stats
    assert stats is not None
    assert stats["regions"] == 2
    assert (stats["misses"], stats["stale"], stats["hits"]) == (4, 2, 2)
    assert (stats["rebuilds"], stats["rebuilt_rows"]) == (1, 8)
    # Four of the context's 8 rows changed in A's region; B's has no context.
    np.testing.assert_array_equal(churn, [0.5, 0.5, np.nan, np.nan])
    np.testing.assert_array_equal(table.region_churn(NEAR_A), [0.0, 0.0])


@pytest.mark.parametrize("nearest", [False, True], ids=["exact", "neighbors"])
def test_requests_to_one_table_in_a_step_decode_together(nearest: bool) -> None:
    client = generalforecast.local()
    strategy = generalforecast.Strategy.neighbors(8) if nearest else None
    rows = _table(client, strategy=strategy)
    features, targets = _clusters()
    rows.insert(np.arange(32), features, targets)
    # Queries far apart, so with neighbors each reads a context of its own.
    queries = np.concatenate([NEAR_A, NEAR_B])
    apart = [rows.predict(queries[[query]]) for query in range(len(queries))]
    node, worker = client._node, client._worker
    table = client._tables["t"]._id
    requests = [node.submit(table, queries[[q]], None) for q in range(len(queries))]
    step = node.schedule()
    assert step is not None
    decodes = worker.stats.calls["decode"]
    together = worker.run(step)

    # Every request's queries decode in one call, each against its own
    # context, as they would apart.
    if nearest:
        assert len({unit for batch in step.queries for unit in batch.units}) > 1
    assert worker.stats.calls["decode"] == decodes + 1
    for request, predicted in zip(requests, apart, strict=True):
        np.testing.assert_allclose(together[request], predicted, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tight", [False, True], ids=["roomy", "tight"])
def test_queries_with_contexts_of_their_own_decode_apart(tight: bool) -> None:
    client = generalforecast.local()
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(8))
    features, targets = _clusters(64)
    rows.insert(np.arange(64), features, targets)
    queries = np.random.default_rng(2).normal(size=(16, 5)).astype(np.float32) * 6
    alone = np.concatenate([rows.predict(queries[[query]]) for query in range(16)])
    worker = client._worker
    largest = max(len(context.slots) for context in worker._contexts.values())
    if tight:
        # A call budget that holds one context and its copy, as a wide
        # model's context over a long page budget does: beside a pool larger
        # than the device, a call takes a sixteenth of it.
        device = 32 * worker._slot_bytes * largest
        worker.budget = dataclasses.replace(worker.budget, device=device)

    together = rows.predict(queries)

    np.testing.assert_allclose(together, alone, rtol=1e-5, atol=1e-6)


def test_decodings_pad_only_to_their_own_size() -> None:
    client = generalforecast.local()
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(8))
    features, targets = _clusters(64)
    rows.insert(np.arange(64), features, targets)
    spread = np.random.default_rng(2).normal(size=(16, 5)).astype(np.float32) * 6
    # Many queries reading one context, beside queries that each read their own.
    queries = np.concatenate([np.repeat(NEAR_A[:1], 32, axis=0), spread])
    alone = np.concatenate([rows.predict(queries[[query]]) for query in range(48)])
    decodes = client.stats.calls["decode"]

    together = rows.predict(queries)

    # The large decoding calls apart from the small ones, so they pad to
    # their own size, as `alike` groups them.
    assert client.stats.calls["decode"] - decodes > 1
    np.testing.assert_allclose(together, alone, rtol=1e-5, atol=1e-6)


def test_a_step_stores_each_context_as_it_is_encoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = memory_profile(tabicl_functions, tiny_tabicl(regression=False), 5)
    unit = profile.context_bytes(PAGE)
    client = generalforecast.local(device_bytes=16 * unit, slot_bytes=unit)
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(8))
    features, targets = _clusters(128)
    rows.insert(np.arange(128), features, targets)
    worker = client._worker
    (loaded,) = worker._models
    # One context a call, so a step's contexts come from many calls: beside
    # the pool, room for a context's encode, or for the context, its copy in
    # the decode and the decode's working memory, and a call takes half that.
    row = profile.cache_bytes_per_row
    decode = worker.decode_profile(loaded).working_bytes(row, PAGE)
    encode = worker.encode_bytes(loaded, 5, PAGE)
    assert encode is not None
    device = worker._pool_bytes + max(encode, 2 * unit + decode)
    worker.budget = dataclasses.replace(worker.budget, device=device)
    assert worker._call_bytes < encode
    store = worker._store
    alive: list[int] = []

    def counting(
        key: Any,
        slots: Any,
        layout: Any,
        rows: int,
        cache: Any,
        own: Any = None,
        state: Any = None,
    ) -> None:
        # Contexts on the device now, counted by this one's largest parts.
        leaves = jax.tree.leaves(cache)
        part = max(leaves, key=lambda leaf: leaf.nbytes)

        def like(array: Any) -> bool:
            return array.shape == part.shape and array.dtype == part.dtype

        held = sum(like(array) for array in jax.live_arrays())
        alive.append(held // sum(like(leaf) for leaf in leaves))
        store(key, slots, layout, rows, cache, own, state)

    monkeypatch.setattr(worker, "_store", counting)
    queries = np.random.default_rng(0).normal(size=(64, 5)).astype(np.float32) * 5
    rows.predict(queries)

    # Each context went to its slots as its call finished, so the device
    # never held the step's contexts all at once.
    assert len(alive) >= 12
    assert max(alive) <= 2, alive


def test_contexts_whose_encode_cannot_be_measured_are_checked_by_their_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = generalforecast.local()
    table = _table(client)
    features, targets = _clusters(64)
    table.insert(np.arange(64), features, targets)

    def unmeasured(*args: Any, **kwargs: Any) -> int:
        raise NoMemoryAnalysis("no analysis")

    # Where XLA gives no measure, the decode alone decides.
    monkeypatch.setattr(generalforecast.worker, "encode_bytes", unmeasured)
    assert table.predict(NEAR_A).shape == (2, 3)


def test_contexts_whose_encode_cannot_be_compiled_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = generalforecast.local()
    table = _table(client)
    features, targets = _clusters(64)
    table.insert(np.arange(64), features, targets)

    def failing(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("RESOURCE_EXHAUSTED: Failed to get configs")

    # An encode XLA cannot compile to measure would fail the same way when run.
    monkeypatch.setattr(generalforecast.worker, "encode_bytes", failing)
    with pytest.raises(ValueError, match="could not compile the encode"):
        table.predict(NEAR_A)
    assert client.stats.encoded == 0


@pytest.mark.parametrize("short", [True, False], ids=["short", "plenty"])
def test_a_worker_drops_its_programs_past_its_table_shapes_when_memory_is_short(
    short: bool,
) -> None:
    client = generalforecast.local()
    worker = client._worker
    worker.programs.limit = 2
    # Memory is short when less than a quarter of the host's is free.
    host = (1 << 62) if short else 0
    worker.budget = dataclasses.replace(worker.budget, host=host)
    tables = []
    for width in (3, 4, 5):
        table = client.create_table(
            f"w{width}", model=classifier(), features=width, classes=3
        )
        features, targets = _clusters()
        table.insert(np.arange(32), features[:, :width], targets)
        tables.append((table, width, table.predict(NEAR_A[:, :width])))

    # A third shape dropped the first two's programs when memory was short;
    # they compile again. With plenty free, recompiling costs more than it saves.
    assert worker.programs.drops == (1 if short else 0)
    for table, width, predicted in tables:
        np.testing.assert_allclose(
            table.predict(NEAR_A[:, :width]), predicted, rtol=1e-5, atol=1e-6
        )


def test_evicting_representations_changes_no_prediction() -> None:
    model = tiny_tabicl(regression=False)
    page = memory_profile(tabicl_functions, model, 5).row_bytes * PAGE
    ample = generalforecast.local()
    tight = generalforecast.local(host_bytes=2 * page, slot_bytes=page)
    features, targets = _clusters(64)
    queries = np.concatenate([NEAR_A, NEAR_B])
    predictions = []
    for client in (ample, tight):
        rows = _table(
            client,
            strategy=generalforecast.Strategy.neighbors(16),
            model=(tabicl_functions, model),
        )
        rows.insert(np.arange(64), features, targets)
        rows.predict(queries)
        rows.update([0, 1], features[[0, 1]], (targets[[0, 1]] + 1) % 3)
        predictions.append(rows.predict(queries))

    # Fit and embed compile to different shapes, so they agree to rounding.
    np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-6)
    assert tight.stats.rows_embedded > ample.stats.rows_embedded


def test_deleted_rows_leave_the_cache_at_once() -> None:
    client = generalforecast.local()
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(16))
    rows.insert(np.arange(32), *_clusters())
    rows.predict(np.concatenate([NEAR_A, NEAR_B]))
    cached = client.stats.dropped

    rows.delete([0])

    assert client.stats.dropped == cached + 1


def test_changes_then_a_refit_match_a_fresh_table() -> None:
    client = generalforecast.local()
    rows = _table(client)
    features, targets = _clusters()
    rows.insert(np.arange(32), features, targets)
    rows.predict(NEAR_A)

    features[[1, 5]] += 1.0
    targets[[1, 5]] = 2
    rows.update([1, 5], features[[1, 5]], targets[[1, 5]])
    rows.delete([2, 7])
    rows.refit()

    kept = [row for row in range(32) if row not in (2, 7)]
    fresh = _table(client, name="fresh")
    fresh.insert(np.asarray(kept), features[kept], targets[kept])
    np.testing.assert_allclose(
        rows.predict(NEAR_A), fresh.predict(NEAR_A), rtol=1e-5, atol=1e-5
    )
    assert client.stats.dropped == 1


def test_a_drifted_table_refits_on_its_own() -> None:
    client = generalforecast.local()
    model = (tabicl_functions, tiny_tabicl(regression=False))
    rows = client.create_table(
        "rows", model=model, features=5, classes=3, page_size=PAGE, refit_shift=1.0
    )
    features, targets = _clusters()
    rows.insert(np.arange(32), features, targets)
    rows.predict(NEAR_A)

    shifted = features + 50.0
    rows.insert(np.arange(32, 64), shifted, targets)

    fresh = _table(client, name="fresh")
    fresh.insert(
        np.arange(64), np.concatenate([features, shifted]), np.tile(targets, 2)
    )
    np.testing.assert_allclose(
        rows.predict(NEAR_A), fresh.predict(NEAR_A), rtol=1e-5, atol=1e-5
    )


def test_categorical_columns_reach_the_model() -> None:
    client = generalforecast.local()
    model = tiny_kumo(regression=False)
    data = table(regression=False, rows=32)
    coded = client.create_table(
        "coded",
        model=(kumo_functions, model),
        columns={
            "a": "categorical",
            "b": "numeric",
            "c": "categorical",
            "d": "numeric",
            "e": "numeric",
        },
        classes=data.num_classes,
    )
    # Categorical columns hold codes: whole numbers from 0.
    features, queries = (
        np.asarray(data.features).copy(),
        np.asarray(data.queries).copy(),
    )
    for rows in (features, queries):
        rows[:, [0, 2]] = np.abs(np.round(rows[:, [0, 2]]))
    coded.insert(np.arange(32), features, np.asarray(data.targets))

    mask = jnp.asarray([True, False, True, False, False])
    state, rows = kumo_functions.fit(
        model, jnp.asarray(features), data.targets, data.num_classes, categorical=mask
    )
    cache = kumo_functions.encode(model, state, rows, data.targets)
    outputs = kumo_functions.decode(
        model, state, cache, kumo_functions.embed(model, state, jnp.asarray(queries))
    )
    expected = kumo_functions.postprocess(model, state, outputs)
    np.testing.assert_allclose(coded.predict(queries), expected, rtol=1e-5, atol=1e-5)
    # A table without categorical columns keeps a different state, yet decodes
    # in the same step.
    plain = client.create_table(
        "plain", model=(kumo_functions, model), features=5, classes=data.num_classes
    )
    plain.insert(np.arange(32), features, np.asarray(data.targets))
    together = client.predict({"coded": queries, "plain": queries})
    np.testing.assert_allclose(together["coded"], expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="codes"):
        coded.insert([99], np.array([[0.5, 0, 0, 0, 0]], np.float32), [0])
    with pytest.raises(ValueError, match="kind"):
        client.create_table(
            "bad", model=(kumo_functions, model), columns={"a": "text"}, classes=3
        )
    with pytest.raises(ValueError, match="kinds are"):
        Column("a", "text")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="fixed column"):
        client.create_table(
            "bad", model=(kumo_functions, model), columns={"key": "numeric"}, classes=3
        )


def test_column_kinds_encode_frames_as_the_model_reads_them() -> None:
    client = generalforecast.local()
    model = (tabicl_functions, tiny_tabicl(regression=False))
    kinds = {
        "store": "categorical",
        "amount": "numeric",
        "paid": "boolean",
        "when": "datetime",
        "count": "numeric",
    }
    typed = client.create_table("typed", model=model, columns=kinds, classes=3)
    # A datetime column becomes five model features.
    plain = client.create_table("plain", model=model, features=9, classes=3)
    features, targets = _clusters()
    stores = np.array(["north", "south", "east"])[np.arange(32) % 3]
    days = np.array(["1970-01-02", "1970-01-03"])[np.arange(32) % 2]
    frame = {
        "when": days.astype("datetime64[ms]"),
        "store": stores,
        "amount": features[:, 1],
        "paid": features[:, 2] > 0,
        "count": features[:, 4],
        "extra": np.zeros(32),
    }
    second = np.arange(32) % 2
    # 1970-01-02 and -03 at midnight: days since 1970, hour, day of the week
    # (a Friday and a Saturday), day of the month, month.
    when = np.column_stack(
        [1 + second, 0 * second, 4 + second, 2 + second, 1 + 0 * second]
    )
    coded = np.column_stack(
        [np.arange(32) % 3, features[:, 1], features[:, 2] > 0, when, features[:, 4]]
    ).astype(np.float32)

    typed.insert(np.arange(32), frame, targets)
    plain.insert(np.arange(32), coded, targets)
    # A table made like the frame takes the same kinds from its types.
    like = {name: frame[name] for name in kinds}
    inferred = client.create_table("inferred", model=model, columns=like, classes=3)

    assert inferred.columns == typed.columns
    assert typed.features == 5
    assert typed.vocabularies() == {"store": ["north", "south", "east"]}
    # An unseen category, a null and an unparsed time are missing values.
    query = {
        "store": ["south", "west"],
        "amount": [0.0, 0.0],
        "paid": [True, None],
        "when": ["1970-01-02T12:00:00", None],
        "count": [0.0, 0.0],
    }
    missing = [np.nan] * 5
    codes = np.array(
        [[1, 0, 1, 1.5, 12, 4, 2, 1, 0], [np.nan, 0, np.nan, *missing, 0]], np.float32
    )
    expected = plain.predict(codes)
    np.testing.assert_allclose(typed.predict(query), expected, rtol=1e-6, atol=1e-6)
    # A frame with unnamed columns, as many as the table's, is read in order;
    # one whose names differ from the table's is refused.
    unnamed = dict(enumerate(query.values()))
    np.testing.assert_allclose(typed.predict(unnamed), expected, rtol=1e-6, atol=1e-6)
    renamed = dict(zip("abcde", query.values(), strict=True))
    for wrong in (renamed, {"store": ["south"], "amount": [0.0]}):
        with pytest.raises(ValueError, match="lack the columns"):
            typed.predict(wrong)
    with pytest.raises(ValueError, match="strings"):
        plain.insert([99], np.array([["x", *[0] * 8]], dtype=object), [0])
    # Categories stay short and few enough for float32 codes, as on a server.
    with pytest.raises(ValueError, match="at most 1024 bytes"):
        typed.insert([98, 99], {**query, "store": ["x" * 1025, "y"]}, [0, 0])
    with pytest.raises(ValueError, match=f"below {MAX_WORDS}"):
        typed.insert([98, 99], {**query, "store": [float(MAX_WORDS), 0.0]}, [0, 0])
    assert typed.predict({name: [] for name in kinds}).shape == (0, 3)


def test_times_split_into_the_features_the_server_gives_them() -> None:
    # The same times as the server's columns test, with the same features.
    times = np.array(
        ["2026-10-13T15:30:45.250", "1969-12-31T23:59:59.999", "2024-02-29", "NaT"],
        dtype="datetime64[ms]",
    )
    parts = datetime_parts(times)

    millis = 1_791_905_445_250
    assert parts[0, 0] == np.float32(millis / 86_400_000)
    np.testing.assert_allclose(parts[0, 1], 15 + 30 / 60 + 45.25 / 3600, atol=1e-5)
    np.testing.assert_array_equal(parts[0, 2:], [1, 13, 10])
    assert parts[1, 1] == np.float32(86_399_999 / 3_600_000)
    np.testing.assert_array_equal(parts[1, 2:], [2, 31, 12])
    np.testing.assert_array_equal(parts[2, 3:], [29, 2])
    assert np.isnan(parts[3]).all()


def test_datetime_columns_take_dates_and_rows_keep_their_kinds() -> None:
    model = (tabicl_functions, tiny_tabicl(regression=False))
    kinds = {"paid_at": "datetime", "store": "categorical", "amount": "numeric"}
    table = generalforecast.local().create_table(
        "dated", model=model, columns=kinds, classes=3
    )

    # Epoch seconds read as days would be 86,400 times too late, so plain
    # numbers are refused rather than guessed at.
    with pytest.raises(ValueError, match="plain numbers"):
        table.insert(
            [0], {"paid_at": [1_700_000_000], "store": ["a"], "amount": [1.0]}, [0]
        )
    # Rows as a list keep their numbers; byte strings read as text.
    table.insert([0], [[np.datetime64("2026-10-01"), "north", 1.5]], [0])
    table.insert(
        [1],
        {"paid_at": ["2026-10-02"], "store": np.array([b"south"]), "amount": [2.0]},
        [1],
    )

    assert table.vocabularies() == {"store": ["north", "south"]}
    assert len(table) == 2


def test_tables_take_times_of_the_kind_they_declare() -> None:
    model = classifier()
    features, targets = _clusters(2)
    client = generalforecast.local()
    counted = client.create_table("counted", model=model, features=5, classes=3)
    with pytest.raises(ValueError, match="times='datetime'"):
        counted.insert(
            [0], features[:1], targets[:1], times=np.datetime64("2026-10-01")
        )
    with pytest.raises(TypeError, match="timedelta"):
        client.create_table(
            "wrong",
            model=model,
            features=5,
            classes=3,
            times="datetime",
            time_bucket=3600,
        )

    dated = client.create_table(
        "dated",
        model=model,
        features=5,
        classes=3,
        times="datetime",
        time_bucket=datetime.timedelta(days=1),
    )
    with pytest.raises(ValueError, match="plain numbers"):
        dated.insert([0], features[:1], targets[:1], times=[1_700_000_000])
    stamps = np.array(["2026-10-01T08:00", "2026-10-03T08:00"], dtype="datetime64[m]")
    dated.insert([0, 1], features, targets, times=stamps)

    # Dates are microseconds since 1970, as an integer table given them reads.
    micros = client.create_table("micros", model=model, features=5, classes=3)
    micros.insert(
        [0, 1], features, targets, times=stamps.astype("datetime64[us]").astype(int)
    )
    cutoff = np.datetime64("2026-10-02", "us")
    np.testing.assert_allclose(
        dated.predict(NEAR_A, before=cutoff),
        micros.predict(NEAR_A, before=int(cutoff.astype(int))),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="precedes every row"):
        dated.predict(NEAR_A, before="2026-09-30")


def test_small_tables_run_together_in_one_compiled_call() -> None:
    traces = []

    def traced(stage: str) -> Any:
        def call(*arguments: Any) -> Any:
            traces.append(stage)
            return getattr(tabicl_functions, stage)(*arguments)

        return call

    functions = tabicl_functions._replace(
        embed=traced("embed"), decode=traced("decode")
    )
    client = generalforecast.local()
    model = (functions, tiny_tabicl(regression=False))
    tables = {}
    for index in range(6):
        features, targets = _clusters(17 + index)
        tables[f"clinic-{index}"] = _table(client, name=f"clinic-{index}", model=model)
        tables[f"clinic-{index}"].insert(np.arange(len(features)), features, targets)

    together = client.predict({name: NEAR_A for name in tables})

    # Every table pads to 32 rows, so each stage runs once for all six.
    assert client.stats.calls == {"fit": 1, "embed": 1, "encode": 1, "decode": 1}
    # Warming compiles larger sizes once the worker is idle; let it finish.
    client._worker.wait_warm()
    traces.clear()
    client.predict({name: NEAR_B for name in list(tables)[:5]})
    # Five tables pad to the same batch of eight as six, so nothing compiles again.
    assert traces == []
    for name, handle in tables.items():
        np.testing.assert_allclose(
            together[name], handle.predict(NEAR_A), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("dense", [False, True], ids=["paged", "dense"])
def test_a_decode_that_runs_out_of_memory_splits_its_queries(
    monkeypatch: pytest.MonkeyPatch, dense: bool
) -> None:
    features, targets = _clusters()
    queries = np.random.default_rng(1).normal(size=(40, 5)).astype(np.float32)
    plain = _table(generalforecast.local())
    plain.insert(np.arange(32), features, targets)
    expected = plain.predict(queries)

    if dense:
        model = tiny_tabicl(regression=False)
        unit = memory_profile(tabicl_functions, model, 5).context_bytes(32)
        # No room on the device: the context is transient, decoded from a dense array.
        client = generalforecast.local(device_bytes=unit // 2, slot_bytes=unit // 2)
    else:
        client = generalforecast.local()
    worker = client._worker
    sizes: list[int] = []

    def refusing(compiled: Any, rows: int) -> Any:
        """`compiled`, failing for more than 16 queries in its `rows`th argument."""

        def call(parameters: Any, *arguments: Any) -> Any:
            padded = arguments[rows].shape[1]
            sizes.append(padded)
            # The device holds a decode of 16 queries against the context, not more.
            if padded > 16:
                raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory")
            return compiled(parameters, *arguments)

        return call

    if dense:
        stage = worker._compiled_stage
        monkeypatch.setattr(
            worker,
            "_compiled_stage",
            lambda name, *rest: (
                refusing(stage(name, *rest), 2)
                if name == "decode"
                else stage(name, *rest)
            ),
        )
    else:
        paged = worker._compiled_paged
        monkeypatch.setattr(
            worker, "_compiled_paged", lambda *settings: refusing(paged(*settings), 6)
        )
    rows = _table(client)
    rows.insert(np.arange(32), features, targets)
    rows.predict(queries[:1])
    sizes.clear()

    split = rows.predict(queries)
    first = sizes.index(16)
    sizes.clear()
    again = rows.predict(queries)

    # The call of 40 failed, and halves until they fit went through; later
    # calls start at that size.
    assert first > 0
    assert set(sizes) == {16}
    assert client.stats.transient > 0 if dense else client.stats.transient == 0
    np.testing.assert_allclose(split, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(again, expected, rtol=1e-5, atol=1e-6)


def _as_frame(step: Any) -> protocol.Step:
    """A step from the bindings as a server's worker receives it."""

    def copy(kind: Any, item: Any) -> Any:
        return kind(**{f.name: getattr(item, f.name) for f in dataclasses.fields(kind)})

    return protocol.Step(
        movements=[copy(protocol.Movement, item) for item in step.movements],
        fits=[copy(protocol.Fit, item) for item in step.fits],
        pages=[copy(protocol.Page, item) for item in step.pages],
        units=[copy(protocol.Unit, item) for item in step.units],
        queries=[copy(protocol.QueryBatch, item) for item in step.queries],
    )


def test_a_failing_table_fails_only_its_own_part_of_a_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = (tabicl_functions, tiny_tabicl(regression=False))
    features, targets = _clusters()
    alone = _table(generalforecast.local(), name="good", model=model)
    alone.insert(np.arange(len(features)), features, targets)
    client = generalforecast.local()
    tables = {name: _table(client, name=name, model=model) for name in ("bad", "good")}
    for handle in tables.values():
        handle.insert(np.arange(len(features)), features, targets)
    bad = client._tables["bad"]._id
    node, worker = client._node, client._worker
    requests = {
        node.submit(client._tables[name]._id, NEAR_A, None): name for name in tables
    }
    step = node.schedule()
    compute = worker._compute

    def fragile(part: Any) -> dict[int, np.ndarray]:
        predictions = compute(part)
        if not part.queries or any(batch.table == bad for batch in part.queries):
            raise ValueError("bad rows")
        return predictions

    monkeypatch.setattr(worker, "_compute", fragile)
    predictions, failed = worker.run_apart(_as_frame(step))

    assert failed == {bad: "ValueError: bad rows"}
    (good,) = [request for request, name in requests.items() if name == "good"]
    assert list(predictions) == [good]
    np.testing.assert_allclose(
        predictions[good], alone.predict(NEAR_A), rtol=1e-5, atol=1e-5
    )
    # Only the good table's part counts, once, though the whole step ran first.
    assert worker.stats.encoded == 1
    assert worker.stats.calls == {"fit": 1, "embed": 1, "encode": 1, "decode": 1}
    # A step with no table to blame raises.
    with pytest.raises(ValueError, match="bad rows"):
        worker.run_apart(protocol.Step([], [], [], [], []))


def test_units_must_read_rows_of_the_steps_pages() -> None:
    client = generalforecast.local()
    rows = _table(client)
    rows.insert(np.arange(32), *_clusters())
    rows.delete([5])
    table = client._tables["t"]._id
    client._node.submit(table, NEAR_A, None)
    step = _as_frame(client._node.schedule())
    (unit,) = step.units
    # The deleted row is in no page; it would otherwise read its neighbor's.
    stray = dataclasses.replace(unit, rows=np.union1d(unit.rows, 5).astype(np.uint32))

    with pytest.raises(RuntimeError, match=f"table {table}"):
        client._worker.run(dataclasses.replace(step, units=[stray]))


def test_contexts_of_different_sizes_decode_in_one_call() -> None:
    client = generalforecast.local()
    model = (tabicl_functions, tiny_tabicl(regression=False))
    tables = {}
    for name, rows in (("small", 20), ("large", 70)):
        features, targets = _clusters(rows)
        tables[name] = _table(client, name=name, model=model)
        tables[name].insert(np.arange(rows), features, targets)

    together = client.predict({name: NEAR_A for name in tables})

    # Fits and encodes compile per padded size; decode reads every context's slots at once.
    assert client.stats.calls["decode"] == 1
    for name, handle in tables.items():
        np.testing.assert_allclose(
            together[name], handle.predict(NEAR_A), rtol=1e-5, atol=1e-5
        )


def test_a_decode_holds_its_context_once_however_large_its_slots() -> None:
    client = generalforecast.local()
    rows = _table(client, strategy=generalforecast.Strategy.neighbors(16))
    features, targets = _clusters(64)
    rows.insert(np.arange(64), features, targets)
    queries = np.concatenate([NEAR_A, NEAR_B])
    rows.predict(queries)
    layout = client._worker.layout(0, 5)
    pool = jax.ShapeDtypeStruct((1024, layout.slot_bytes), jnp.uint8)
    blocks = jax.ShapeDtypeStruct((16, layout.slots_per_row), jnp.int32)
    positions = jax.ShapeDtypeStruct((64,), jnp.int32)
    count = jax.ShapeDtypeStruct((), jnp.int32)

    compiled = (
        jax.jit(lambda *values: read_context(layout, *values))
        .lower(pool, blocks, positions, count)
        .compile()
    )
    analysis = compiled.memory_analysis()

    # Rows come straight from the pool, so the decode holds the cache once:
    # not the pool, nor the slots' bytes whole, nor a reordered copy.
    assert analysis is not None
    cache = 64 * layout.row_bytes
    held = analysis.temp_size_in_bytes + analysis.output_size_in_bytes
    assert held <= 2 * cache + (1 << 12), (held, cache)
    assert held < 16 * layout.slot_bytes // 8
    np.testing.assert_allclose(
        rows.predict(queries), client.predict({"t": queries})["t"], rtol=1e-6
    )


def test_decoding_from_slots_matches_decoding_dense_caches() -> None:
    model = tiny_tabicl(regression=False)
    unit = memory_profile(tabicl_functions, model, 5).context_bytes(16)
    paged = generalforecast.local()
    # No room on the device: every context is transient, decoded from dense arrays.
    dense = generalforecast.local(device_bytes=unit // 2, slot_bytes=unit // 2)
    features, targets = _clusters(64)
    queries = np.concatenate([NEAR_A, NEAR_B, NEAR_A + 0.5])
    predictions = []
    for client in (paged, dense):
        rows = _table(
            client,
            strategy=generalforecast.Strategy.neighbors(16),
            model=(tabicl_functions, model),
        )
        rows.insert(np.arange(64), features, targets)
        predictions.append(rows.predict(queries))

    assert dense.stats.transient > 0 and paged.stats.transient == 0
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-5, atol=1e-5)


def test_contexts_with_no_room_are_encoded_only_as_their_decode_needs_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_tabicl(regression=False)
    unit = memory_profile(tabicl_functions, model, 5).context_bytes(16)
    # No room on the device: every context is transient.
    client = generalforecast.local(device_bytes=unit // 2, slot_bytes=unit // 2)
    rows = _table(
        client,
        strategy=generalforecast.Strategy.neighbors(16),
        model=(tabicl_functions, model),
    )
    features, targets = _clusters(128)
    rows.insert(np.arange(128), features, targets)
    worker = client._worker
    # One context a call, so a step's contexts come from many calls: a call
    # takes at most an eighth of a device with room for a context, its copy
    # in the decode and the decode's working memory.
    worker.budget = dataclasses.replace(worker.budget, device=5 * unit)
    call = worker._call
    alive: list[int] = []

    def counting(stage: str, jobs: Any, consume: Any = None) -> Any:
        if stage == "decode" and jobs:
            # Contexts on the device now, counted by this one's largest parts.
            leaves = jax.tree.leaves(jobs[0][1][1])
            part = max(leaves, key=lambda leaf: leaf.nbytes)

            def like(array: Any) -> bool:
                return array.shape == part.shape and array.dtype == part.dtype

            held = sum(like(array) for array in jax.live_arrays())
            alive.append(held // sum(like(leaf) for leaf in leaves))
        return call(stage, jobs, consume)

    monkeypatch.setattr(worker, "_call", counting)
    queries = np.random.default_rng(0).normal(size=(32, 5)).astype(np.float32) * 5
    rows.predict(queries)

    # The step never held every context it had no room for at once.
    assert client.stats.transient >= 8
    assert len(alive) >= 8
    assert max(alive) <= 2, alive


def test_dense_contexts_are_encoded_a_call_budget_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_tabicl(regression=False)
    unit = memory_profile(tabicl_functions, model, 5).context_bytes(16)
    paged = generalforecast.local()
    dense = generalforecast.local(device_bytes=unit // 2, slot_bytes=unit // 2)
    worker = dense._worker
    # Less than two contexts, so each is decoded before the next is encoded:
    # a call takes at most an eighth of a device with room for a context, its
    # copy in the decode and the decode's working memory.
    worker.budget = dataclasses.replace(worker.budget, device=5 * unit)
    events: list[str] = []
    call = worker._call

    def calling(stage: str, jobs: Any, consume: Any = None) -> Any:
        if stage in ("encode", "decode") and jobs:
            events.append(stage)
        return call(stage, jobs, consume)

    monkeypatch.setattr(worker, "_call", calling)
    features, targets = _clusters(64)
    queries = np.concatenate([NEAR_A, NEAR_B, NEAR_A + 0.5])
    predictions = []
    for client in (paged, dense):
        rows = _table(
            client,
            strategy=generalforecast.Strategy.neighbors(16),
            model=(tabicl_functions, model),
        )
        rows.insert(np.arange(64), features, targets)
        predictions.append(rows.predict(queries))

    assert events.count("encode") > 1
    assert events == ["encode", "decode"] * events.count("encode")
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("budget", ["device", "step"])
def test_requests_past_a_steps_budget_run_in_later_steps(budget: str) -> None:
    model = tiny_tabicl(regression=False)
    unit = memory_profile(tabicl_functions, model, 5).context_bytes(32)
    # Room for one table's context, on the device or in a step, so a step
    # takes one table.
    if budget == "device":
        client = generalforecast.local(device_bytes=unit, slot_bytes=unit)
    else:
        client = generalforecast.local(slot_bytes=unit, step_bytes=unit)
    alone = generalforecast.local()
    names = ["first", "second", "third"]
    for target in (client, alone):
        for index, name in enumerate(names):
            features, targets = _clusters(20 + index)
            table = _table(target, name=name, model=(tabicl_functions, model))
            table.insert(np.arange(len(features)), features, targets)

    together = client.predict({name: NEAR_A for name in names})

    assert client.stats.calls["decode"] == 3
    for name in names:
        np.testing.assert_allclose(
            together[name], alone.table(name).predict(NEAR_A), rtol=1e-5, atol=1e-5
        )


def test_a_table_past_its_fit_rows_fits_on_a_sample_and_embeds_pages_it_reads() -> None:
    client = generalforecast.local()
    table = client.create_table(
        "t",
        model=(tabicl_functions, tiny_tabicl(regression=False)),
        features=5,
        classes=3,
        strategy=generalforecast.Strategy.neighbors(8),
        page_size=8,
        fit_rows=8,
    )
    table.insert(np.arange(32), *_clusters())

    predictions = table.predict(NEAR_A)

    assert predictions.shape == (len(NEAR_A), 3)
    assert np.allclose(predictions.sum(axis=1), 1, atol=1e-5)
    # Only the pages the queries read are embedded, not the whole table.
    assert 0 < client.stats.rows_embedded < 32
    assert table.definition.fit_rows == 8


def test_a_table_that_summarizes_each_context_fits_units_and_reuses_them() -> None:
    from generalforecast.models.limix.model import limix_functions
    from tests.models.limix.tiny import tiny_limix

    client = generalforecast.local()
    table = client.create_table(
        "t",
        model=(kumo_functions, tiny_kumo(regression=False)),
        features=5,
        classes=3,
        strategy=generalforecast.Strategy.neighbors(8),
        page_size=8,
        summary="context",
    )
    table.insert(np.arange(32), *_clusters())

    first = table.predict(NEAR_A)
    second = table.predict(NEAR_A)

    np.testing.assert_allclose(second, first, rtol=1e-6, atol=1e-6)
    # Both queries read one unit, encoded once and then read from its slots.
    assert (client.stats.encoded, client.stats.reused) == (1, 1)
    # No table fit and no pages: each unit is fit on its own rows.
    assert client.stats.rows_embedded == 0 and "fit" not in client.stats.calls
    assert table.describe().summary == "context"
    with pytest.raises(ValueError, match="no column summary"):
        client.create_table(
            "limix",
            model=(limix_functions, tiny_limix(regression=False)),
            features=5,
            classes=3,
            summary="context",
        )


def test_cutoffs_limit_contexts_and_states_to_earlier_rows() -> None:
    model = tiny_tabicl(regression=False)
    client = generalforecast.local()
    features, targets = _clusters()
    table = _table(client, model=(tabicl_functions, model))
    table.insert(np.arange(32), features, targets, times=np.arange(32))
    rows, labels = jnp.asarray(features), jnp.asarray(targets)

    def expected(context: int) -> np.ndarray:
        """A state fit on the rows before 16, and a context of the first `context` rows."""
        state, fitted = tabicl_functions.fit(model, rows[:16], labels[:16], 3)
        later = tabicl_functions.embed(model, state, rows[16:], labels[16:])
        representations = jnp.concatenate([fitted, later])[:context]
        cache = tabicl_functions.encode(model, state, representations, labels[:context])
        queries = tabicl_functions.embed(model, state, jnp.asarray(NEAR_A))
        outputs = tabicl_functions.decode(model, state, cache, queries)
        return np.asarray(tabicl_functions.postprocess(model, state, outputs))

    # As a table of the rows before 16 alone predicts: no later row reaches
    # the state, as none reaches the context.
    np.testing.assert_allclose(
        table.predict(NEAR_A, before=16), expected(16), rtol=1e-5, atol=1e-5
    )
    # A request's queries share a state fit before its earliest cutoff.
    mixed = table.predict(NEAR_A, before=[16, 32])
    np.testing.assert_allclose(mixed[1], expected(32)[1], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="precedes every row"):
        table.predict(NEAR_A, before=0)


def test_a_failed_step_leaves_nothing_behind(monkeypatch) -> None:
    features, targets = _clusters()
    tables = [_table(generalforecast.local()) for _ in range(2)]
    for each in tables:
        each.insert(np.arange(32), features, targets)
    store = Worker._store

    def failing(*arguments) -> None:
        monkeypatch.setattr(Worker, "_store", store)
        raise RuntimeError("out of memory")

    monkeypatch.setattr(Worker, "_store", failing)
    with pytest.raises(RuntimeError, match="out of memory"):
        tables[0].predict(NEAR_A)

    np.testing.assert_allclose(tables[0].predict(NEAR_A), tables[1].predict(NEAR_A))


def test_a_write_that_loses_the_device_pool_forgets_every_context(monkeypatch) -> None:
    from generalforecast.runtime import paging

    features, targets = _clusters()
    client = generalforecast.local()
    # An accelerator's pool, which a donated write can take with it.
    client._worker._device._on_host = False
    tables = [_table(client, name) for name in ("first", "second")]
    for each in tables:
        each.insert(np.arange(32), features, targets)
    expected = tables[0].predict(NEAR_A)
    write = paging._write

    def failing(pool, slots, data):
        # As a donated write that ran out of memory leaves the pool.
        monkeypatch.setattr(paging, "_write", write)
        pool.delete()
        raise RuntimeError("out of memory")

    monkeypatch.setattr(paging, "_write", failing)
    with pytest.raises(paging.PoolLost):
        tables[1].predict(NEAR_A)

    # The first table's cached context went with the pool, so it is encoded again.
    encoded = client.stats.encoded
    np.testing.assert_allclose(tables[0].predict(NEAR_A), expected, atol=1e-6)
    assert client.stats.encoded > encoded


def test_regressors_pick_quantiles_between_the_models_own() -> None:
    features, _ = _clusters()
    table = generalforecast.local().create_table(
        "values", model=(tabicl_functions, tiny_tabicl(regression=True)), features=5
    )
    table.insert(np.arange(32), features, features.sum(axis=1))
    every = table.predict(NEAR_A)
    columns = every.shape[1]

    picked = table.predict(NEAR_A, quantiles=[0.5, 0.0015, 0.0005])

    # Level 0.5 is a column; 0.0015 lies halfway between the first two; and
    # 0.0005 lies past the first, which it takes.
    assert columns == 999
    np.testing.assert_array_equal(picked[:, 0], every[:, 499])
    np.testing.assert_allclose(picked[:, 1], (every[:, 0] + every[:, 1]) / 2, rtol=1e-6)
    np.testing.assert_array_equal(picked[:, 2], every[:, 0])
    assert table.predict(NEAR_A[:0], quantiles=[0.5]).shape == (0, 1)
    assert table.describe().model is None
    with pytest.raises(ValueError, match="predicts classes"):
        _table(generalforecast.local()).predict(NEAR_A, quantiles=[0.5])
    for strategy in (
        generalforecast.Strategy.subsample(8, seed=3),
        generalforecast.Strategy.retrieved(8, core=4, seed=1, kind="diverse"),
        generalforecast.Strategy.retrieved(8, among=32, last=1024),
        generalforecast.Strategy.retrieved(4, core=4, span=3600),
    ):
        assert eval(repr(strategy), {"Strategy": generalforecast.Strategy}) == strategy
    assert (
        generalforecast.Classifier().get_params()["model"]
        == "general-forecast/tabicl-v2-classifier"
    )
    assert (
        generalforecast.Regressor().get_params()["model"]
        == "general-forecast/tabicl-v2-regressor"
    )


def test_string_labels_code_in_order_and_stay_apart_from_codes() -> None:
    client = generalforecast.local()
    features, targets = _clusters()
    names = np.array(["cat", "dog", "fish"])
    labelled, coded = _table(client, "labelled"), _table(client, "coded")

    labelled.insert(np.arange(32), features, names[targets])
    coded.insert(np.arange(32), features, targets)

    np.testing.assert_allclose(labelled.predict(NEAR_A), coded.predict(NEAR_A))
    assert labelled.labels == ["cat", "dog", "fish"]
    assert coded.labels is None
    with pytest.raises(ValueError, match="labels"):
        labelled.insert([99], features[:1], [0])
    with pytest.raises(ValueError, match="codes"):
        coded.insert([99], features[:1], ["cat"])
    with pytest.raises(ValueError, match="'eel' would be another"):
        labelled.upsert([1, 99], features[:2], ["cat", "eel"])
    assert len(labelled) == 32


def _int8(client_bytes: int = 1 << 30) -> generalforecast.LocalClient:
    """A local client storing caches as int8, as GF_CACHE_DTYPE=int8 does."""
    exact = device_settings("cpu", exact=True)
    return generalforecast.local(
        settings=dataclasses.replace(exact, cache_dtype=jnp.dtype("int8")),
        device_bytes=client_bytes,
    )


def test_models_that_read_caches_as_encoded_get_them_back_in_float32() -> None:
    import importlib.util

    path = Path(__file__).parents[1] / "examples" / "custom_model.py"
    spec = importlib.util.spec_from_file_location("custom_model", path)
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    features, targets = _clusters()
    predictions = []
    for client in (generalforecast.local(), _int8()):
        table = client.create_table(
            "kernel", model=example.kernel(), features=5, classes=3
        )
        table.insert(np.arange(32), features, targets)
        predictions.append(table.predict(NEAR_A))

    # Read raw, int8 keys would be coordinates hundreds of times too large.
    np.testing.assert_allclose(predictions[1], predictions[0], atol=2e-2)


def test_time_buckets_keep_later_rows_off_earlier_pages() -> None:
    features, targets = _clusters(64)
    embedded, copied = {}, {}
    for time_bucket in (None, 8):
        client = generalforecast.local()
        table = client.create_table(
            "rows",
            model=(tabicl_functions, tiny_tabicl(regression=False)),
            features=5,
            classes=3,
            page_size=PAGE,
            time_bucket=time_bucket,
            # A state fit on the first rows serves the later ones too.
            refit_after=1.0,
        )
        table.insert(np.arange(56), features[:56], targets[:56], times=np.arange(56))
        table.predict(NEAR_A)
        table.insert(
            np.arange(56, 64), features[56:], targets[56:], times=np.arange(56, 64)
        )
        table.predict(NEAR_A)
        embedded[time_bucket] = client.stats.rows_embedded
        copied[time_bucket] = client.stats.rows_copied

    # Eight rows a window fill one page, so the later rows embed one page of
    # their own; without buckets they land on earlier pages, whose other rows
    # are copied from the entries those pages had.
    assert embedded[8] == PAGE and copied[8] == 0
    assert embedded[None] == PAGE and copied[None] > 0


def test_tables_naming_a_model_share_its_weights(monkeypatch) -> None:
    loads = []
    load = registry.load
    monkeypatch.setattr(registry, "load", lambda spec: loads.append(spec) or load(spec))
    client = generalforecast.local()

    for name in ("first", "second"):
        client.create_table(
            name, model="tests.models.tabicl.tiny:classifier", features=5, classes=3
        )

    assert len(loads) == 1


def test_a_failed_predict_leaves_nothing_for_the_next() -> None:
    client = generalforecast.local()
    features, targets = _clusters()
    for name in ("first", "second"):
        _table(client, name).insert(np.arange(32), features, targets, np.arange(32))
    expected = client.table("first").predict(NEAR_A)

    with pytest.raises(ValueError):
        client.predict({"first": NEAR_A, "second": NEAR_A[:, :4]})
    assert client._node.schedule() is None
    with pytest.raises(ValueError, match="precedes every row"):
        client.predict({"first": NEAR_A, "second": NEAR_A}, before={"second": 0})

    np.testing.assert_allclose(client.predict({"first": NEAR_A})["first"], expected)


def test_threads_take_turns_on_a_local_client() -> None:
    from concurrent.futures import ThreadPoolExecutor

    client = generalforecast.local()
    features, targets = _clusters()
    table = _table(client)
    table.insert(np.arange(32), features, targets)

    with ThreadPoolExecutor(4) as pool:
        predictions = list(pool.map(lambda _: table.predict(NEAR_A), range(8)))

    for each in predictions[1:]:
        np.testing.assert_array_equal(each, predictions[0])


def test_keys_labels_and_rows_are_checked_rather_than_converted() -> None:
    table = _table(generalforecast.local())
    features, targets = _clusters(2)

    cases: list[tuple[Any, Any, Any]] = [
        ([0.5, 2], targets, features),
        ([0, 1], [0.5, 1], features),
        ([0, 1], targets, features[0]),
    ]
    for keys, labels, rows in cases:
        with pytest.raises(ValueError):
            table.insert(keys, rows, labels)
    assert len(table) == 0
    table.insert([0, 1], features, targets)
    with pytest.raises(ValueError, match="2 columns, but the table has 5"):
        table.insert([2], features[:1, :2], targets[:1])
    with pytest.raises(KeyError, match="key 7"):
        table.delete([7])
    with pytest.raises(KeyError, match="key 7"):
        table.update([7], features[:1], targets[:1])
    with pytest.raises(ValueError, match="table 't': the table has no rows"):
        generalforecast.local().create_table(
            "t", model=classifier(), features=5, classes=3
        ).predict(features)


def test_tables_and_strategies_are_checked_when_created() -> None:
    client = generalforecast.local()

    with pytest.raises(ValueError, match="feature"):
        client.create_table("flat", model=classifier(), features=0, classes=3)
    with pytest.raises(ValueError, match="2 classes"):
        client.create_table("one", model=classifier(), features=3, classes=1)
    # The model says what a table predicts: classes only for a classifier.
    with pytest.raises(ValueError, match="give the table its classes"):
        client.create_table("none", model=classifier(), features=3)
    regressor = (tabicl_functions, tiny_tabicl(regression=True))
    with pytest.raises(ValueError, match="without classes"):
        client.create_table("values", model=regressor, features=3, classes=2)
    assert client.create_table("values", model=regressor, features=3).classes is None
    client.drop_table("values")
    with pytest.raises(ValueError, match="core or neighbors"):
        generalforecast.Strategy.neighbors(0)
    with pytest.raises(ValueError, match="core's rows must be positive"):
        generalforecast.Strategy.subsample(0)
    assert client.tables() == []


def test_clients_take_generators_predict_nothing_and_close() -> None:
    features, targets = _clusters()
    with generalforecast.local(disk_bytes=1 << 20, slot_bytes=1 << 16) as client:
        table = _table(client, model=classifier())
        table.insert((key for key in range(32)), features, targets)
        temporary = client._temporary

        assert len(table) == 32
        assert table.predict(features[:0]).shape == (0, 3)
        assert "general-forecast/tabicl-v2-classifier" in client.models()
        with pytest.raises(KeyError, match="missing"):
            client.predict({"missing": NEAR_A})
        table.predict(NEAR_A)
        device = client._worker._device
        assert device._host is not None
    assert temporary is not None and not temporary.exists()
    assert client.tables() == []
    # The device pool's memory goes with the client, not with the collector.
    assert device._host is None and device._data is None


def test_tables_are_looked_up_by_name_and_names_are_unique() -> None:
    client = generalforecast.local()
    rows = _table(client)

    assert client.table("t") is rows
    with pytest.raises(ValueError, match="exists"):
        _table(client)


def test_dropped_tables_free_their_name_and_caches() -> None:
    client = generalforecast.local()
    rows = _table(client)
    features, targets = _clusters()
    rows.insert(np.arange(32), features, targets)
    rows.predict(NEAR_A)
    dropped = client.stats.dropped

    client.drop_table("t")

    assert client.tables() == []
    assert client.stats.dropped > dropped
    # The old handle names the table it lost, even once the name is taken again.
    assert len(_table(client)) == 0
    with pytest.raises(KeyError, match="'t' does not exist"):
        rows.insert([99], features[:1], targets[:1])
    with pytest.raises(KeyError, match="'t' does not exist"):
        rows.predict(NEAR_A)
    with pytest.raises(KeyError):
        client.table("missing")


def test_built_in_models_load_from_a_local_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GF_CHECKPOINTS", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="tabicl-v2-classifier"):
        registry.load("general-forecast/tabicl-v2-classifier")

    checkpoints = Path(__file__).resolve().parents[1] / "checkpoints"
    if not (checkpoints / "tabicl-v2-classifier").is_dir():
        pytest.skip("local checkpoints are not downloaded")
    monkeypatch.setenv("GF_CHECKPOINTS", str(checkpoints))
    functions, model = registry.load("general-forecast/tabicl-v2-classifier")
    assert functions is tabicl_functions
    assert not functions.properties(model).regression


def test_a_client_sizes_its_tiers_from_its_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "generalforecast.runtime.budget.device_memory", lambda: 64 << 20
    )

    client = generalforecast.local(slot_bytes=1 << 20)

    # A quarter of the device's memory, and as much again on the host.
    assert client._worker._slots[:2] == (16, 16)


def _observations(times: list[int], series: str, seed: int) -> dict[str, Any]:
    """A series' observations with one covariate, as a mapping of columns."""
    generator = np.random.default_rng(seed)
    return {
        "item": [series] * len(times),
        "t": times,
        "units": generator.normal(size=len(times)).astype(np.float32),
        "price": np.linspace(1.0, 2.0, len(times), dtype=np.float32),
    }


def _series_forecast(
    functions: Any, model: Any, rows: dict[str, Any], cutoff: int, horizon: int
) -> np.ndarray:
    """A series' forecast as the model gives it on a group laid by hand.

    The target row and its known covariate row, the context left-padded to
    whole patches, and the covariate's future from the cutoff on.
    """
    from generalforecast.models.series import KNOWN, TARGET, SeriesGroup

    patch = functions.properties(model).patch
    times = np.asarray(rows["t"])
    units, price = np.asarray(rows["units"]), np.asarray(rows["price"])
    grid = np.arange(times.min(), cutoff, 10)
    values = np.zeros((2, len(grid)), np.float32)
    observed = np.zeros((2, len(grid)), bool)
    future = np.zeros((2, horizon), np.float32)
    future_observed = np.zeros((2, horizon), bool)
    for time, unit, cost in zip(times, units, price, strict=True):
        if time < cutoff:
            at = (time - grid[0]) // 10
            if np.isfinite(unit):
                values[0, at], observed[0, at] = unit, True
            values[1, at], observed[1, at] = cost, True
        elif time < cutoff + 10 * horizon:
            at = (time - cutoff) // 10
            future[1, at], future_observed[1, at] = cost, True
    pad = -len(grid) % patch
    group = SeriesGroup(
        jnp.asarray(np.pad(values, ((0, 0), (pad, 0)))),
        jnp.asarray(np.pad(observed, ((0, 0), (pad, 0)))),
        jnp.asarray(np.array([TARGET, KNOWN], np.int32)),
        jnp.asarray(future),
        jnp.asarray(future_observed),
    )
    return np.asarray(functions.forecast(model, group, horizon))[0]


def test_series_tables_forecast_as_the_model_does_on_the_group() -> None:
    from generalforecast.models.timestral.model import timestral_functions

    model = tiny_timestral()
    client = generalforecast.local()
    sales = client.create_series_table(
        "sales",
        model=(timestral_functions, model),
        series="item",
        time="t",
        target="units",
        step=10,
        columns={"price": "numeric"},
        known=["price"],
        horizon=5,
    )
    first = _observations(list(range(100, 200, 10)), "shoes", seed=1)
    second = _observations(list(range(120, 200, 10)), "socks", seed=2)
    sales.insert(first)
    sales.insert(second)
    # Prices known ahead, with no units yet.
    ahead = _observations(list(range(200, 250, 10)), "shoes", seed=3)
    ahead["units"] = [None] * 5
    sales.insert(ahead)
    assert len(sales) == 10 + 8 + 5
    assert client.series_table("sales") is sales
    assert sales.quantile_levels == timestral_functions.properties(model).quantiles

    forecasts = sales.forecast(["shoes", "socks"], before=200)

    merged = {key: [*first[key], *ahead[key]] for key in first}
    merged["units"] = [*first["units"], *[np.nan] * 5]
    expected = np.stack(
        [
            _series_forecast(timestral_functions, model, merged, 200, 5),
            _series_forecast(timestral_functions, model, second, 200, 5),
        ]
    )
    assert forecasts.shape == (2, 5, 9)
    np.testing.assert_allclose(forecasts, expected, rtol=1e-4, atol=1e-4)
    assert sales.last_answer is not None
    assert sales.last_answer.epoch == 0 and sales.last_answer.gpu_seconds > 0

    # Levels the model predicts come back as they are; one between two lies between.
    median = sales.forecast(["shoes"], horizon=3, before=200, quantiles=[0.5, 0.15])
    shorter = _series_forecast(timestral_functions, model, merged, 200, 3)
    np.testing.assert_allclose(median[0, :, 0], shorter[:, 4], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        median[0, :, 1], (shorter[:, 0] + shorter[:, 1]) / 2, rtol=1e-4, atol=1e-4
    )
    # Without a cutoff, each series is forecast from after its latest time.
    latest = sales.forecast(["socks"], horizon=2)
    assert latest.shape == (1, 2, 9)
    assert sales.forecast([], horizon=2).shape == (0, 2, 9)

    # Replacing and removing observations changes what the context holds.
    changed = _observations([190], "socks", seed=4)
    sales.upsert(changed)
    sales.delete({"item": ["socks"], "t": [120]})
    assert len(sales) == 22
    moved = sales.forecast(["socks"], horizon=2, before=200)
    assert not np.allclose(moved, latest)


def test_series_tables_forecast_the_groups_given_them() -> None:
    from generalforecast.models.timestral.model import timestral_functions
    from tests.models.timestral.tiny import tiny_timestral

    model = (timestral_functions, tiny_timestral())
    client = generalforecast.local()

    def table(name: str, group: generalforecast.GroupStrategy, items: list[str]) -> Any:
        made = client.create_series_table(
            name,
            model=model,
            series="item",
            time="t",
            target="units",
            step=10,
            group=group,
            horizon=5,
        )
        for item in items:
            rows = _observations(list(range(100, 200, 10)), item, seeds[item])
            del rows["price"]
            made.insert(rows)
        return made

    seeds = {"shoes": 1, "socks": 2, "boots": 3}
    given = table(
        "given", generalforecast.GroupStrategy.given(3), ["shoes", "socks", "boots"]
    )
    pair = table("pair", generalforecast.GroupStrategy.random(2, 0), ["shoes", "socks"])
    alone = table("alone", generalforecast.GroupStrategy.alone(), ["boots"])

    # A group named is the group forecast: the pair's only group, and a
    # series alone, forecast as tables drawing those groups forecast them.
    found = given.forecast_given(["shoes"], [["shoes", "socks"]], before=200)
    np.testing.assert_allclose(found, pair.forecast(["shoes"], before=200), rtol=1e-5)
    found = given.forecast_given(["boots"], [["boots"]], before=200)
    np.testing.assert_allclose(found, alone.forecast(["boots"], before=200), rtol=1e-5)
    # Two forecasts naming one group, the series among its members or not,
    # share it; a series the table never saw is forecast through its group.
    both = given.forecast_given(
        ["shoes", "socks"], [["socks"], ["socks", "shoes"]], before=200
    )
    assert both.shape == (2, 5, 9)
    cold = given.forecast_given(["new"], [["shoes", "socks"]], before=200)
    assert cold.shape == (1, 5, 9)

    with pytest.raises(ValueError, match="takes each series' group"):
        given.forecast(["shoes"], before=200)
    with pytest.raises(ValueError, match="draws its own groups"):
        pair.forecast_given(["shoes"], [["shoes", "socks"]], before=200)
    with pytest.raises(ValueError, match="2 groups"):
        given.forecast_given(["shoes", "socks"], [["shoes"]], before=200)
    with pytest.raises(ValueError, match="past the table's 3"):
        given.forecast_given(
            ["shoes"], [["shoes", "socks", "boots", "new"]], before=200
        )
    with pytest.raises(KeyError, match="does not exist"):
        given.forecast_given(["shoes"], [["shoes", "nowhere"]], before=200)


def test_series_tables_refuse_groups_past_the_models_rows() -> None:
    from generalforecast.models.timesfm3.model import timesfm_functions
    from tests.models.timesfm3.tiny import tiny_timesfm

    # The tiny TimesFM reads 32 rows per group: with one covariate, 16 series
    # of two rows each fit, 32 do not, and the table says so at creation
    # rather than forecasting a smaller group than asked.
    client = generalforecast.local()
    client.create_series_table(
        "fits",
        model=(timesfm_functions, tiny_timesfm()),
        series="item",
        time="t",
        target="units",
        step=10,
        columns={"price": "numeric"},
        known=["price"],
        group=generalforecast.GroupStrategy.random(16, 0),
    )
    # Alone, a series with 16 covariates is 17 rows and fits; a pair does not,
    # and the trace of such a group says why in the planner's words.
    wide = {f"c{index}": "numeric" for index in range(16)}
    client.create_series_table(
        "alone",
        model=(timesfm_functions, tiny_timesfm()),
        series="item",
        time="t",
        target="units",
        step=10,
        columns=wide,
    )
    with pytest.raises(ValueError, match="per group"):
        client.create_series_table(
            "pair",
            model=(timesfm_functions, tiny_timesfm()),
            series="item",
            time="t",
            target="units",
            step=10,
            columns=wide,
            group=generalforecast.GroupStrategy.random(2, 0),
        )
    with pytest.raises(ValueError, match="past what the model serves per group"):
        client.create_series_table(
            "wide",
            model=(timesfm_functions, tiny_timesfm()),
            series="item",
            time="t",
            target="units",
            step=10,
            columns={"price": "numeric"},
            known=["price"],
            group=generalforecast.GroupStrategy.random(32, 0),
        )


def test_series_tables_refuse_what_they_cannot_hold() -> None:
    from generalforecast.models.timestral.model import timestral_functions

    forecaster: Any = (timestral_functions, tiny_timestral())
    tabular: Any = (tabicl_functions, tiny_tabicl(regression=True))
    client = generalforecast.local()
    with pytest.raises(TypeError, match="forecaster"):
        client.create_table("rows", model=forecaster, features=2)
    with pytest.raises(TypeError, match="not a forecaster"):
        client.create_series_table(
            "rows",
            model=tabular,
            series="item",
            time="t",
            target="units",
            step=10,
        )
    with pytest.raises(TypeError, match="step is a timedelta"):
        client.create_series_table(
            "daily",
            model=forecaster,
            series="item",
            time="t",
            target="units",
            step=1,
            times="datetime",
        )
    with pytest.raises(ValueError, match="known columns"):
        client.create_series_table(
            "sales",
            model=forecaster,
            series="item",
            time="t",
            target="units",
            step=10,
            known=["price"],
        )
    sales = client.create_series_table(
        "sales", model=forecaster, series="item", time="t", target="units", step=10
    )
    rows = _table(client)
    with pytest.raises(TypeError, match="holds series"):
        client.table("sales")
    with pytest.raises(TypeError, match="holds rows"):
        client.series_table("t")
    with pytest.raises(TypeError, match="holds series"):
        client.predict({"sales": NEAR_A})

    sales.insert({"item": ["shoes", "shoes"], "t": [100, 110], "units": [1.0, 2.0]})
    with pytest.raises(ValueError, match="series 'shoes'"):
        sales.insert({"item": ["shoes"], "t": [125], "units": [3.0]})
    # A cutoff off the grid starts the forecast at the next point on it.
    np.testing.assert_array_equal(
        sales.forecast(["shoes"], horizon=2, before=125),
        sales.forecast(["shoes"], horizon=2, before=130),
    )
    with pytest.raises(ValueError, match="without a horizon"):
        sales.forecast(["shoes"])
    with pytest.raises(ValueError, match="strictly between"):
        sales.forecast(["shoes"], horizon=2, quantiles=[1.5])
    # Alone, a series the table never saw, or one with nothing observed
    # before its cutoff, has no context to forecast from.
    with pytest.raises(ValueError, match="holds no series"):
        sales.forecast(["boots"], horizon=2, before=130)
    with pytest.raises(ValueError, match="observation before cutoff 100"):
        sales.forecast(["shoes"], horizon=2, before=100)
    assert len(rows) == 0

    daily = client.create_series_table(
        "daily",
        model=forecaster,
        series="item",
        time="day",
        target="units",
        step=datetime.timedelta(days=1),
        times="datetime",
        horizon=2,
    )
    days = np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]")
    daily.insert({"item": ["a"] * 8, "day": days, "units": np.arange(8.0)})
    assert daily.forecast(["a"], before=np.datetime64("2024-01-09")).shape == (1, 2, 9)


def test_a_page_written_once_embeds_the_written_row_and_copies_the_rest() -> None:
    written, plain = generalforecast.local(), generalforecast.local()
    model = (tabicl_functions, tiny_tabicl(regression=False))
    tables = {
        client: client.create_table(
            "rows", model=model, features=5, classes=3, page_size=32
        )
        for client in (written, plain)
    }
    features, targets = _clusters()
    changed = np.where(np.arange(32) == 0, (targets + 1) % 3, targets)
    for rows in tables.values():
        rows.insert(np.arange(32), features, targets)
        rows.predict(NEAR_A)
    embedded = {client: client.stats.rows_embedded for client in tables}
    tables[written].update([0], features[[0]], changed[[0]])
    # The same change, with every other row written as it was: the page's
    # earlier entry then has no row to copy, so the page embeds whole.
    tables[plain].update(np.arange(32), features, changed)
    predictions = {client: rows.predict(NEAR_A) for client, rows in tables.items()}

    assert written.stats.rows_embedded - embedded[written] == 1
    assert written.stats.rows_copied == 31
    assert plain.stats.rows_embedded - embedded[plain] == 32
    assert plain.stats.rows_copied == 0
    np.testing.assert_allclose(predictions[written], predictions[plain], atol=1e-6)


def test_a_page_written_once_embeds_its_row_with_a_warmed_program() -> None:
    jax.config.update("jax_enable_compilation_cache", False)
    compiled: list[str] = []

    def compile_event(event: str, duration_secs: float, **details: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(details.get("fun_name", ""))

    jax.monitoring.register_event_duration_secs_listener(compile_event)
    try:
        client = generalforecast.local()
        rows = client.create_table(
            "rows",
            model=(tabicl_functions, tiny_tabicl(regression=False)),
            features=5,
            classes=3,
            page_size=32,
        )
        features, targets = _clusters()
        rows.insert(np.arange(32), features, targets)
        rows.predict(NEAR_A)
        client._worker.wait_warm()
        compiled.clear()
        # One row of the page changes, so the page embeds that row alone,
        # padded to the smallest bucket, which the warm compiled too.
        rows.update([0], features[[0]], (targets[[0]] + 1) % 3)
        rows.predict(NEAR_A)
    finally:
        jax.monitoring.unregister_event_duration_listener(compile_event)
        jax.config.update("jax_enable_compilation_cache", True)

    assert client.stats.rows_copied == 31
    assert "jit(embed)" not in compiled


def test_a_steps_parts_add_up_to_its_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    from generalforecast import worker_process

    client = generalforecast.local()
    rows = _table(client)
    rows.insert(np.arange(32), *_clusters())
    node, worker = client._node, client._worker
    worker.timers = True
    node.submit(client._tables["t"]._id, NEAR_A, None)
    step = _as_frame(node.schedule())
    monkeypatch.setattr(protocol, "request", lambda header, buffers: step)
    before = worker.stats.snapshot()

    header, buffers = worker_process._answer(worker, {}, {"kind": "step"}, [])()

    parts = {name: spent["seconds"] for name, spent in header["parts"].items()}
    stages = sum(spent["seconds"] for spent in header["stages"].values())
    # A step that fits, encodes and decodes runs every part once at least;
    # a call's handing to the device is `compile` for a shape its program
    # had not seen and `dispatch` otherwise, and the decode's query arrays
    # go to the device as `dispatch` either way.
    assert set(parts) - {"compile"} == {
        "frame",
        "move",
        "pad",
        "stack",
        "dispatch",
        "wait",
        "unstack",
        "fetch",
        "store",
        "other",
        "reply",
    }
    assert all(seconds >= 0 for seconds in parts.values())
    assert abs(header["seconds"] - sum(parts.values())) < 1e-3
    # Compiled calls wait on the device; the parts outside them do not.
    assert parts["wait"] <= stages
    assert header["parts"]["frame"]["calls"] == header["parts"]["reply"]["calls"] == 1
    assert len(buffers) == 1
    # The worker's own counts grew by the same parts.
    assert worker.stats.parts_since(before) == {
        name: (spent["calls"], spent["seconds"])
        for name, spent in header["parts"].items()
    }


def test_a_failed_steps_time_counts_as_other(monkeypatch: pytest.MonkeyPatch) -> None:
    from generalforecast import worker_process

    client = generalforecast.local()
    rows = _table(client)
    rows.insert(np.arange(32), *_clusters())
    node, worker = client._node, client._worker
    worker.timers = True
    node.submit(client._tables["t"]._id, NEAR_A, None)
    step = _as_frame(node.schedule())
    monkeypatch.setattr(protocol, "request", lambda header, buffers: step)
    compute = worker._compute

    def failing(step: protocol.Step) -> dict[int, np.ndarray]:
        # Parts run, and are rolled back, before the step fails.
        with worker.stats.timed("pad"):
            clock.sleep(0.02)
        monkeypatch.setattr(worker, "_compute", compute)
        raise RuntimeError("the device went away")

    monkeypatch.setattr(worker, "_compute", failing)
    before = worker.stats.snapshot()

    header, _ = worker_process._answer(worker, {}, {"kind": "step"}, [])()

    # The attempt's parts went with its counts, and its time is `other`.
    assert header["failures"] and not header["results"]
    assert set(header["parts"]) == {"frame", "move", "other", "reply"}
    assert header["parts"]["other"]["calls"] == 1
    assert header["parts"]["other"]["seconds"] >= 0.02
    assert (
        abs(
            header["seconds"]
            - sum(spent["seconds"] for spent in header["parts"].values())
        )
        < 1e-3
    )
    assert "pad" not in worker.stats.parts_since(before)


def test_an_untimed_step_reports_no_parts_and_predicts_as_a_timed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generalforecast import worker_process

    client = generalforecast.local()
    rows = _table(client)
    rows.insert(np.arange(32), *_clusters())
    node, worker = client._node, client._worker
    node.submit(client._tables["t"]._id, NEAR_A, None)
    step = _as_frame(node.schedule())
    monkeypatch.setattr(protocol, "request", lambda header, buffers: step)
    assert not worker.timers

    untimed, untimed_buffers = worker_process._answer(
        worker, {}, {"kind": "step"}, []
    )()
    worker.timers = True
    timed, timed_buffers = worker_process._answer(worker, {}, {"kind": "step"}, [])()

    # The reply keeps its parts field, empty, and its stages and seconds.
    assert untimed["parts"] == {} and timed["parts"]
    assert untimed["stages"] and untimed["seconds"] > 0
    assert untimed["results"] == timed["results"]
    for left, right in zip(untimed_buffers, timed_buffers, strict=True):
        np.testing.assert_array_equal(
            np.frombuffer(left, np.float32), np.frombuffer(right, np.float32)
        )


def _stacks(path: Path) -> dict[tuple[str, ...], int]:
    """A collapsed stacks file, each stack's frames to its samples."""
    found: dict[tuple[str, ...], int] = {}
    for line in path.read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        found[tuple(stack.split(";"))] = int(count)
    return found


@pytest.mark.slow
def test_a_record_request_samples_the_steps_stacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from generalforecast import worker_process

    client = generalforecast.local()
    rows = _table(client)
    rows.insert(np.arange(32), *_clusters())
    node, worker = client._node, client._worker
    node.submit(client._tables["t"]._id, NEAR_A, None)
    step = _as_frame(node.schedule())
    request = protocol.request
    monkeypatch.setattr(
        protocol,
        "request",
        lambda header, buffers: (
            step if header["kind"] == "step" else request(header, buffers)
        ),
    )
    # The file's directory is made before the record answers.
    path = tmp_path / "profile" / "stacks.txt"
    header = {"kind": "record", "seconds": 0.5, "path": str(path)}
    for _ in range(3):
        worker_process._answer(worker, {}, {"kind": "step"}, [])()

    answered, _ = worker_process._answer(worker, {}, header, [])()
    assert answered == {"kind": "ok"}
    assert path.parent.is_dir()
    # One record at a time.
    answered, _ = worker_process._answer(worker, {}, header, [])()
    assert answered["kind"] == "error" and "already" in answered["message"]
    # Steps run on this thread while the sampler runs beside them.
    deadline = clock.perf_counter() + 10
    while not path.exists() and clock.perf_counter() < deadline:
        worker_process._answer(worker, {}, {"kind": "step"}, [])()

    stacks = _stacks(path)
    this = threading.current_thread().name
    mine = {stack: n for stack, n in stacks.items() if stack[0] == this}
    assert sum(mine.values()) >= 50, "fewer than a sample per 10 ms"
    stepping = {
        stack: n
        for stack, n in mine.items()
        if any(
            frame.startswith("run_apart (generalforecast/worker.py:") for frame in stack
        )
    }
    assert sum(stepping.values()) > sum(mine.values()) / 2
    # The other threads sample too, under their own names.
    assert {stack[0] for stack in stacks} > {this}
    assert "gf-record" not in {stack[0] for stack in stacks}


@pytest.mark.slow
def test_a_record_request_with_a_trace_writes_the_trace_and_its_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from generalforecast import worker_process
    from generalforecast.profiling import trace

    client = generalforecast.local()
    rows = _table(client)
    rows.insert(np.arange(32), *_clusters())
    node, worker = client._node, client._worker
    worker.timers = True
    node.submit(client._tables["t"]._id, NEAR_A, None)
    step = _as_frame(node.schedule())
    request = protocol.request
    monkeypatch.setattr(
        protocol,
        "request",
        lambda header, buffers: (
            step if header["kind"] == "step" else request(header, buffers)
        ),
    )
    directory = tmp_path / "worker-0"
    header = {
        "kind": "record",
        "seconds": 0.5,
        "path": str(directory / "stacks.txt"),
        "trace": str(directory),
    }
    worker_process._answer(worker, {}, {"kind": "step"}, [])()
    answered, _ = worker_process._answer(worker, {}, header, [])()
    assert answered == {"kind": "ok"}
    # One record at a time, with or without a trace.
    answered, _ = worker_process._answer(worker, {}, header, [])()
    assert answered["kind"] == "error"
    answered, _ = worker_process._answer(
        worker, {}, {"kind": "record", "seconds": 0.1, "path": str(tmp_path / "x")}, []
    )()
    assert answered["kind"] == "error"

    ran = 0
    deadline = clock.perf_counter() + 10
    while not (directory / trace.STEPS_FILE).exists():
        assert clock.perf_counter() < deadline
        worker_process._answer(worker, {}, {"kind": "step"}, [])()
        ran += 1
    assert (directory / "stacks.txt").exists()
    assert list(directory.glob(trace.TRACE_GLOB))

    summary = trace.summarize(directory)
    steps = summary.steps.to_pylist()
    assert 1 <= len(steps) <= ran
    assert [row["step"] for row in steps] == list(range(len(steps)))
    for row in steps:
        assert row["kernels"] > 0
        assert 0 < row["device_s"] <= row["trace_s"] <= row["wall_s"]
        assert row["device_s"] + row["idle_s"] == pytest.approx(row["trace_s"])
        assert row["gap_after"]
        # Its parts add up to its wall, as a reply's do.
        parts = sum(value for name, value in row.items() if name.endswith("_s"))
        parts -= sum(row[name] for name in trace._STEP_COLUMNS if name.endswith("_s"))
        assert parts == pytest.approx(row["wall_s"], rel=0.05)
    assert "wait_s" in summary.steps.column_names
    # Every program the step ran is in the tables, with the stage's runs.
    programs = {row["program"]: row for row in summary.programs.to_pylist()}
    assert programs["jit_decode"]["runs"] >= len(steps)
    assert summary.kernels.num_rows > 0
    # Steps run past the window as they did before it, and a new record may follow.
    worker_process._answer(worker, {}, {"kind": "step"}, [])()
    assert worker_process._tracing is None
    assert not worker_process._recording


@pytest.mark.slow
def test_sampling_costs_a_step_loop_a_few_percent_at_most(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Wall-clock, so it runs with the slow tests, and with a bound loose
    # enough for a shared machine.
    from generalforecast import worker_process

    client = generalforecast.local()
    rows = _table(client)
    rows.insert(np.arange(32), *_clusters())
    node, worker = client._node, client._worker
    node.submit(client._tables["t"]._id, NEAR_A, None)
    step = _as_frame(node.schedule())
    monkeypatch.setattr(protocol, "request", lambda header, buffers: step)

    def steps(seconds: float) -> int:
        deadline = clock.perf_counter() + seconds
        count = 0
        while clock.perf_counter() < deadline:
            worker_process._answer(worker, {}, {"kind": "step"}, [])()
            count += 1
        return count

    steps(0.5)
    plain, sampled = [], []
    for round in range(3):
        plain.append(steps(1.0))
        sampler = threading.Thread(
            target=worker_process._sampled, args=(1.0, str(tmp_path / f"{round}"))
        )
        sampler.start()
        sampled.append(steps(1.0))
        sampler.join()

    overhead = 1 - max(sampled) / max(plain)
    print(f"sampling costs {overhead:.1%} of {max(plain)} steps a second")
    assert overhead < 0.15
