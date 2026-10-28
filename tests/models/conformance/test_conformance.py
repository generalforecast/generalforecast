# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Contract checks every model must pass, run on small random instances.

These are the guarantees the engine relies on: what each call depends on, that
padding and batching are invisible, and how precision and memory behave.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast import protocol
from generalforecast._core import CacheKey, bucket_size
from generalforecast.models.conformance import Table, check, predict, table
from generalforecast.models.conformance import close as _close
from generalforecast.models.conformance import padded as _padded
from generalforecast.models.contract import TabularFunctions
from generalforecast.nn.attention import AttentionKernel
from generalforecast.runtime import paging
from generalforecast.runtime.decoding import (
    Context,
    Decoding,
    paged_decode,
    paged_inputs,
    paged_shape,
)
from generalforecast.runtime.device import (
    DeviceSettings,
    configure,
    device_settings,
    load_cache,
    store_cache,
    to_cache_dtype,
)
from generalforecast.runtime.memory import fit_profile, memory_profile, tree_bytes
from generalforecast.runtime.paging import cache_layout, pack, pack_into, read_context
from generalforecast.worker import Worker
from tests.models.conformance.cases import CASES, FEATURES, RUNTIME_CASES, Case
from tests.utils import node_harness
from tests.utils.predict import predict as gf_predict

FLOAT32, FLOAT16, BFLOAT16, INT8 = (
    jnp.dtype(name) for name in ("float32", "float16", "bfloat16", "int8")
)
REFERENCE = DeviceSettings(FLOAT32, FLOAT32, AttentionKernel())
BUDGETED = {"tabicl-values", "limix-classes", "ensemble-values"}
runtime = pytest.mark.parametrize(
    "case", RUNTIME_CASES, ids=[case.name for case in RUNTIME_CASES]
)


@pytest.fixture(params=CASES, ids=[case.name for case in CASES])
def case(request: pytest.FixtureRequest) -> Case:
    return request.param


@pytest.fixture
def functions(case: Case) -> TabularFunctions:
    return case.functions()


@pytest.fixture
def model(case: Case) -> Any:
    return case.build(0)


def test_models_pass_the_conformance_check(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    check(functions, model)

    assert functions.properties(model).regression == case.regression
    # Another instance of the same shapes shares the properties, so its tables
    # share programs.
    assert functions.properties(case.build(1)) == functions.properties(model)


def test_the_class_count_comes_from_the_caller_within_the_models_slots(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    data = table(case.regression)
    slots = functions.properties(model).class_slots
    invalid = [2] if case.regression else [0, slots + 1]

    for num_classes in invalid:
        with pytest.raises(ValueError, match="class"):
            functions.fit(model, data.features, data.targets, num_classes)
    if case.regression:
        return
    # Four classes, two of them never seen in the context.
    labels = jnp.arange(data.features.shape[0], dtype=jnp.int32) % 2
    probabilities = predict(
        functions, model, data._replace(targets=labels, num_classes=4)
    )
    assert probabilities.shape == (data.queries.shape[0], 4)
    _close(probabilities.sum(axis=-1), jnp.ones(data.queries.shape[0]))


def test_tables_in_one_bucket_share_one_compile(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    traces: list[object] = []
    num_classes = table(case.regression).num_classes

    def pipeline(model: Any, arrays: tuple[jax.Array, ...], num_rows: jax.Array):
        traces.append(num_rows)
        padded = Table(*arrays, num_classes=num_classes)
        return predict(functions, model, padded, num_rows=num_rows)

    compiled = jax.jit(pipeline)
    for rows, seed in ((20, 0), (27, 1)):
        data = table(case.regression, rows=rows, queries=6, seed=seed)
        padded = _padded(data, context_size=32, query_size=8, seed=seed)
        result = compiled(model, tuple(padded[:3]), rows)
        _close(result[:6], predict(functions, model, data))

    assert len(traces) == 1


def test_parameters_set_the_compute_dtype(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    half = configure(model, DeviceSettings(FLOAT16, FLOAT16, AttentionKernel()))
    data = table(case.regression)

    state, context_rows = functions.fit(
        half, data.features, data.targets, data.num_classes
    )
    cache = functions.encode(half, state, context_rows, data.targets)
    outputs = functions.decode(
        half, state, cache, functions.embed(half, state, data.queries)
    )

    assert context_rows.dtype == outputs.dtype == jnp.float16
    floating = [
        leaf
        for leaf in jax.tree.leaves(cache)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    assert floating and all(leaf.dtype == jnp.float16 for leaf in floating)
    assert functions.postprocess(half, state, outputs).dtype == jnp.float32


def test_device_settings_barely_move_predictions(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    """Reduced precision caches, GPU settings and attention chunks stay near float32.

    Drift is the largest change in probability, or in medians per target
    deviation.
    """
    data = table(case.regression, rows=40, queries=12)

    def run(settings: DeviceSettings) -> jax.Array:
        configured = configure(model, settings)
        state, context_rows = functions.fit(
            configured, data.features, data.targets, data.num_classes
        )
        cache = store_cache(
            functions.encode(configured, state, context_rows, data.targets), settings
        )
        query_rows = functions.embed(configured, state, data.queries)
        outputs = functions.decode(configured, state, cache, query_rows)
        return functions.postprocess(configured, state, outputs)

    full = run(REFERENCE)

    def drift(settings: DeviceSettings) -> float:
        reduced = run(settings)
        if not case.regression:
            return float(jnp.max(jnp.abs(reduced - full)))
        middle = full.shape[-1] // 2
        change = jnp.abs(reduced[:, middle] - full[:, middle])
        return float(jnp.max(change) / jnp.std(data.targets))

    caches = [
        (f"{dtype} cache", DeviceSettings(FLOAT32, dtype, AttentionKernel()))
        for dtype in (FLOAT16, BFLOAT16, INT8)
    ]
    # Chunks only reorder sums.
    chunks = [
        ("query chunks", DeviceSettings(FLOAT32, FLOAT32, AttentionKernel("xla", 8))),
        (
            "block chunks",
            DeviceSettings(FLOAT32, FLOAT32, AttentionKernel(block_chunk=16)),
        ),
    ]
    bounds = [
        *((name, settings, case.cache_drift) for name, settings in caches),
        ("gpu", device_settings("gpu"), 3e-2),
        *((name, settings, 1e-5) for name, settings in chunks),
    ]

    moved = {name: drift(settings) for name, settings, _ in bounds}

    assert {
        name: moved[name] for name, _, bound in bounds if moved[name] >= bound
    } == {}


# The node sizes such a fit from a model's state alone: one whose state
# summarizes columns, one whose does not, and an ensemble's stacked states,
# whose categorical columns can be crowded.
@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.name in BUDGETED],
    ids=lambda case: case.name,
)
def test_a_fit_past_the_call_budget_predicts_as_one_pass(
    case: Case,
    functions: TabularFunctions,
    model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rows short of their padded size, a column of category codes and a
    # constant one, so padding, masks and dropped columns all run.
    rows, padded = 500, 512
    data = table(case.regression, rows=rows, queries=6)
    features, queries = np.array(data.features), np.array(data.queries)
    features[:, 0] = np.arange(rows) % 4
    queries[:, 0] = np.arange(len(queries)) % 4
    features[:, 1] = queries[:, 1] = 5.0
    mask = jnp.asarray([True] + [False] * (FEATURES - 1))
    # A call budget past the state's, but short of a fit that embeds every
    # row: the table fits its state alone on every row, its column summary a
    # few rows a step, and embeds pages as plans read them.
    profile = fit_profile(functions, configure(model, REFERENCE), FEATURES)
    budget = int(profile.fused.bytes(padded))
    assert profile.state_rows(budget) >= padded
    monkeypatch.setattr(Worker, "_call_bytes", property(lambda _: budget))
    client = node_harness.local(settings=REFERENCE, compile_cache=False)
    columns = {"code": "categorical"} | {
        f"f{index}": "numeric" for index in range(1, FEATURES)
    }
    served = client.create_table(
        "rows",
        model=(functions, model),
        columns=columns,
        classes=data.num_classes or None,
    )
    served.insert(np.arange(rows), features, np.asarray(data.targets))

    actual = served.predict(queries)

    described = served.describe()
    assert rows <= (described.effective_fit_rows or 0) <= described.fit_rows
    assert client.stats.rows_embedded == rows
    state, context_rows = functions.fit(
        model,
        jnp.asarray(features),
        data.targets,
        data.num_classes,
        categorical=mask,
    )
    cache = functions.encode(model, state, context_rows, data.targets)
    query_rows = functions.embed(model, state, jnp.asarray(queries))
    outputs = functions.decode(model, state, cache, query_rows)
    _close(actual, functions.postprocess(model, state, outputs))


def test_memory_profile_matches_real_outputs(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    profile = memory_profile(functions, model, num_features=FEATURES)

    # Called with row counts, as the worker calls models, and a categorical
    # mask, which some states keep more for.
    mask = jnp.zeros(FEATURES, bool)
    for rows in (16, 40):
        data = table(case.regression, rows=rows)
        state, context_rows = functions.fit(
            model,
            data.features,
            data.targets,
            data.num_classes,
            num_rows=rows,
            categorical=mask,
        )
        cache = functions.encode(
            model, state, context_rows, data.targets, num_rows=rows
        )
        assert profile.categorical_state_bytes == tree_bytes(state)
        assert profile.row_bytes * rows == tree_bytes(context_rows)
        assert profile.context_bytes(rows) == tree_bytes(cache)
    plain, _ = functions.fit(
        model, data.features, data.targets, data.num_classes, num_rows=40
    )
    assert profile.state_bytes == tree_bytes(plain)
    # Parameter shapes alone give the same profile.
    abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), model)
    assert memory_profile(functions, abstract, FEATURES) == profile


@runtime
def test_caches_pack_into_slots_and_back(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    data = _padded(
        table(case.regression, rows=20, queries=6), context_size=32, query_size=8
    )
    state, rows = functions.fit(
        model, data.features, data.targets, data.num_classes, num_rows=20
    )
    encoded = functions.encode(model, state, rows, data.targets, num_rows=20)
    for dtype in (FLOAT32, FLOAT16, INT8):
        cache = to_cache_dtype(encoded, dtype)
        layout = cache_layout(
            functions, model, FEATURES, slot_bytes=4096, cache_dtype=dtype
        )

        slots = pack(layout, cache)

        profile = memory_profile(functions, model, FEATURES, cache_dtype=dtype)
        assert layout.row_bytes == profile.cache_bytes_per_row
        assert slots.shape == (layout.slots(32), 4096)
        assert slots.dtype == jnp.uint8
        # A decode reads the rows back, a block at a time or row by row.
        blocks = jnp.arange(slots.shape[0]).reshape(-1, layout.slots_per_row)
        expected = layout.treedef.flatten_up_to(cache)
        for in_order in (True, False):
            restored = layout.treedef.flatten_up_to(
                read_context(
                    layout, slots, blocks, jnp.arange(32), jnp.array(32), None, in_order
                )
            )
            for leaf in layout.row_leaves:
                np.testing.assert_array_equal(
                    restored[leaf.index], expected[leaf.index]
                )


@runtime
def test_a_cache_packs_into_the_pool_as_pack_packs_it_in_parts(
    case: Case, functions: TabularFunctions, model: Any, monkeypatch: Any
) -> None:
    data = _padded(
        table(case.regression, rows=20, queries=6), context_size=32, query_size=8
    )
    state, rows = functions.fit(
        model, data.features, data.targets, data.num_classes, num_rows=20
    )
    cache = to_cache_dtype(
        functions.encode(model, state, rows, data.targets, num_rows=20), FLOAT16
    )
    layout = cache_layout(
        functions, model, FEATURES, slot_bytes=4096, cache_dtype=FLOAT16
    )
    count = layout.packed_slots(cache)
    slots = list(range(count + 3, 3, -1))[:count]
    expected = (
        jnp.zeros((count + 5, 4096), jnp.uint8)
        .at[jnp.array(slots)]
        .set(pack(layout, cache))
    )
    # Three blocks a part: parts of every size, the last padded.
    monkeypatch.setattr(paging, "PACK_BYTES", 3 * layout.block_bytes)

    pool = pack_into(jnp.zeros((count + 5, 4096), jnp.uint8), slots, layout, cache)

    np.testing.assert_array_equal(pool, expected)


@runtime
def test_a_paged_decode_predicts_as_one_of_the_whole_cache(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    data = _padded(
        table(case.regression, rows=20, queries=6), context_size=32, query_size=8
    )
    state, rows = functions.fit(
        model, data.features, data.targets, data.num_classes, num_rows=20
    )
    cache = to_cache_dtype(
        functions.encode(model, state, rows, data.targets, num_rows=20), FLOAT16
    )
    layout = cache_layout(
        functions, model, FEATURES, slot_bytes=4096, cache_dtype=FLOAT16
    )
    slots = pack(layout, cache)
    blocks = jnp.arange(slots.shape[0]).reshape(-1, layout.slots_per_row)
    queries = functions.embed(model, state, data.queries)
    stored = functions.properties(model).stored_cache

    decode = paged_decode(functions, stored, layout, masked=False, in_order=True)
    paged = decode(
        model,
        slots,
        jax.tree.map(lambda values: values[None], state),
        jnp.zeros(1, jnp.int32),
        blocks[None],
        jnp.arange(32)[None],
        jnp.array([20]),
        queries[None],
    )[0]

    whole = read_context(layout, slots, blocks, jnp.arange(32), jnp.array(20))
    whole = whole if stored else load_cache(whole)
    np.testing.assert_array_equal(
        paged,
        functions.postprocess(
            model, state, functions.decode(model, state, whole, queries)
        ),
    )


@runtime
def test_a_decoding_in_a_masked_call_predicts_as_in_an_unmasked_one(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    """A decoding of one context, decoded beside one of two, gets the runtime's all-True mask.

    `paged_inputs` gives it every row of the call's padded context, past its
    own 20, so only its length bounds what it reads.
    """
    data = _padded(
        table(case.regression, rows=20, queries=6), context_size=32, query_size=8
    )
    state, rows = functions.fit(
        model, data.features, data.targets, data.num_classes, num_rows=20
    )
    cache = functions.encode(model, state, rows, data.targets, num_rows=20)
    layout = cache_layout(functions, model, FEATURES, slot_bytes=4096)
    slots = pack(layout, cache)
    context = Context(layout, 20, list(range(slots.shape[0])))
    queries = functions.embed(model, state, data.queries)[:6]
    batch = SimpleNamespace(table=0)
    alone = Decoding(batch, list(range(6)), [context], queries)
    # The same context twice, each query reading one of them.
    joined = Decoding(
        batch, list(range(6)), [context, context], queries, [0] * 3 + [1] * 3
    )
    stored = functions.properties(model).stored_cache

    def decoded(decodings: list[Decoding]) -> tuple[np.ndarray, tuple]:
        shape = paged_shape(decodings, layout, bucket_size)
        _, inputs = paged_inputs(decodings, layout, shape)
        decode = paged_decode(functions, stored, layout, shape.masked, shape.in_order)
        states = jax.tree.map(lambda values: values[None], state)
        return np.asarray(decode(model, slots, states, *inputs)), inputs

    together, inputs = decoded([alone, joined])
    expected, _ = decoded([alone])

    # The mask reads past the decoding's 20 rows, as the runtime builds it.
    counts, reads = inputs[3], inputs[-1]
    assert counts[0] == 20 and reads.shape[-1] > 20 and reads[0].all()
    _close(together[0, :6], expected[0, :6])
    _close(together[1, :6], expected[0, :6])


@runtime
def test_a_unit_that_summarizes_its_own_rows_predicts_as_those_rows_alone(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    data = table(case.regression, rows=48, queries=6)
    worker = Worker(REFERENCE, slot_bytes=4096, compile_cache=False)
    worker.add_model(0, functions, model)
    worker.allocate(256, 0, 0)
    if not functions.properties(model).column_summary:
        with pytest.raises(ValueError, match="no column summary"):
            worker.add_table(0, 0, data.num_classes, FEATURES, summary="context")
        return
    worker.add_table(0, 0, data.num_classes, FEATURES, summary="context")
    # A context of 20 of the table's 48 rows, every class among them.
    rows = np.arange(0, 40, 2, dtype=np.uint32)
    features = np.asarray(data.features)[rows]
    targets = np.asarray(data.targets)[rows]
    state_slots = -(-worker.profile(0, FEATURES).state_bytes // 4096)
    slots = list(range(worker.layout(0, FEATURES).slots(32) + state_slots))
    key = CacheKey(0, 0, 0, "context", 1)

    def predicted(action: protocol.Action) -> np.ndarray:
        fit = (None, None) if action == "reuse" else (features, targets)
        unit = protocol.Unit(0, key, rows, action, slots, *fit)
        queries = np.asarray(data.queries)
        batch = protocol.QueryBatch(0, 0, 0, queries, [0] * len(queries))
        return worker.run(protocol.Step([], [], [], [unit], [batch]))[0]

    context = (jnp.asarray(features), jnp.asarray(targets))
    expected = gf_predict(functions, model, *context, data.queries)

    # Encoded and kept, read back from its slots, and encoded for one step.
    _close(predicted("encode"), expected)
    _close(predicted("reuse"), expected)
    _close(predicted("transient"), expected)


@runtime
def test_a_fused_unit_predicts_as_its_rows_fit_and_encoded_apart(
    case: Case, functions: TabularFunctions, model: Any
) -> None:
    data = table(case.regression, rows=48, queries=6)
    worker = Worker(REFERENCE, slot_bytes=4096, compile_cache=False)
    worker.add_model(0, functions, model)
    worker.allocate(256, 0, 0)
    worker.add_table(0, 0, data.num_classes, FEATURES)
    rows = np.arange(48, dtype=np.uint32)
    features = np.asarray(data.features)
    targets = np.asarray(data.targets)
    key = CacheKey(0, 0, 0, "context", 1)
    unit = protocol.Unit(0, key, rows, "fuse", [], features, targets)
    queries = np.asarray(data.queries)
    batch = protocol.QueryBatch(0, 0, 0, queries, [0] * len(queries))
    predicted = worker.run(protocol.Step([], [], [], [unit], [batch]))[0]
    expected = gf_predict(
        functions, model, jnp.asarray(features), jnp.asarray(targets), data.queries
    )
    _close(predicted, expected)
