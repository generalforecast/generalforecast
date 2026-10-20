from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast import protocol
from generalforecast._core import CacheKey
from generalforecast.models.tabicl.model import TabICL, tabicl_functions
from generalforecast.nn.attention import AttentionKernel
from generalforecast.runtime.budget import Call, Chunks, measured_bytes
from generalforecast.runtime.device import (
    FLOAT32,
    DeviceSettings,
    configure,
    device_settings,
)
from generalforecast.runtime.memory import (
    MEASURED_ROWS,
    MemoryProfile,
    _encode_parts,
    _traced,
    decode_profile,
    encode_profile,
    encode_widths,
    memory_profile,
    tree_bytes,
)
from generalforecast.worker import Worker
from tests.models.conformance.cases import CASES
from tests.models.tabicl.tiny import tiny_tabicl

CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"
TASKS = pytest.mark.parametrize("regression", [False, True], ids=["classes", "values"])
FEATURES = 5


def _nbytes(tree: object) -> int:
    """Bytes held by the arrays in `tree`, counted independently of `tree_bytes`."""
    return sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(tree))


def _run(
    model: TabICL, rows: int, features: int = FEATURES
) -> tuple[object, jax.Array, object]:
    """Real state, context rows and cache for a random table."""
    generator = np.random.default_rng(rows)
    table = jnp.asarray(generator.normal(size=(rows, features)), jnp.float32)
    if model.regression:
        targets, num_classes = table.sum(axis=1), 0
    else:
        targets, num_classes = jnp.arange(rows, dtype=jnp.int32) % 3, 3
    state, context_rows = tabicl_functions.fit(model, table, targets, num_classes)
    cache = tabicl_functions.encode(model, state, context_rows, targets)
    return state, context_rows, cache


def test_tree_bytes_counts_arrays_and_shapes() -> None:
    tree = {
        "array": jnp.zeros((3, 4), jnp.float32),
        "shape": jax.ShapeDtypeStruct((2, 5), jnp.int8),
        "nothing": None,
    }

    assert tree_bytes(tree) == 3 * 4 * 4 + 2 * 5


def test_context_bytes_grow_linearly() -> None:
    profile = MemoryProfile(
        state_bytes=100, row_bytes=8, cache_bytes_per_row=16, cache_fixed_bytes=4
    )

    assert profile.context_bytes(0) == 4
    assert profile.context_bytes(10) == 164


def test_state_bytes_grow_with_features_not_rows() -> None:
    model = tiny_tabicl(regression=False)

    narrow = memory_profile(tabicl_functions, model, num_features=4)
    wide = memory_profile(tabicl_functions, model, num_features=8)

    assert wide.state_bytes > narrow.state_bytes
    assert _nbytes(_run(model, rows=16, features=4)[0]) == narrow.state_bytes
    assert _nbytes(_run(model, rows=64, features=4)[0]) == narrow.state_bytes


def test_profile_counts_the_stored_cache_dtype() -> None:
    model = tiny_tabicl(regression=False)

    computed = memory_profile(tabicl_functions, model, num_features=FEATURES)
    stored = memory_profile(
        tabicl_functions, model, FEATURES, cache_dtype=jnp.dtype("float16")
    )

    assert stored.cache_bytes_per_row * 2 == computed.cache_bytes_per_row
    assert stored.row_bytes == computed.row_bytes
    assert stored.state_bytes == computed.state_bytes


@pytest.mark.parametrize("name", ["mitra-values", "limix-classes"])
def test_a_decodes_estimate_covers_what_xla_measures(name: str) -> None:
    case = next(case for case in CASES if case.name == name)
    functions, model = case.functions(), case.build(0)

    def decode(model: object, state: object, cache: object, rows: jax.Array) -> object:
        return functions.postprocess(
            model, state, functions.decode(model, state, cache, rows)
        )

    profile = decode_profile(functions, model, decode, queries=8)
    for width in (3, 30):
        per_row = memory_profile(functions, model, width).cache_bytes_per_row
        for rows in (1 << 10, 1 << 14):
            state, _, cache = _traced(functions, model, width, rows, None)
            queries = jax.ShapeDtypeStruct((8, width), jnp.float32)
            embedded = jax.eval_shape(functions.embed, model, state, queries)
            compiled = jax.jit(decode).lower(model, state, cache, embedded).compile()
            analysis = compiled.memory_analysis()
            assert analysis is not None
            measured = analysis.temp_size_in_bytes + analysis.output_size_in_bytes

            # Measured once per model, the estimate holds at other widths and
            # sizes, within its margin.
            estimate = profile.working_bytes(per_row, rows)
            assert measured <= estimate <= 1.3 * measured, (width, rows)


@pytest.mark.parametrize("name", ["kumo-classes", "limix-classes"])
def test_an_encodes_estimate_covers_what_xla_measures(name: str) -> None:
    case = next(case for case in CASES if case.name == name)
    functions = case.functions()
    model = configure(case.build(0), device_settings("cpu"))

    for width in (3, 128):
        # Measured around the table's width: both models' temporaries grow
        # faster than their caches with width.
        profile = encode_profile(functions, model, widths=encode_widths(width))
        per_row = memory_profile(functions, model, width).cache_bytes_per_row
        for rows in (MEASURED_ROWS, 4 * MEASURED_ROWS):
            held, temporaries, outputs = _encode_parts(
                functions, model, False, width, rows
            )
            measured = held + max(temporaries, outputs)

            # LimiX chunks its attention at a size that moves with width, so
            # past the sizes measured a wide table's estimate runs high.
            estimate = profile.working_bytes(per_row, rows)
            assert measured <= estimate <= 1.6 * measured, (width, rows)


@pytest.mark.slow
def test_a_large_fit_keeps_every_call_in_the_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_tabicl(regression=False)
    rows, page, budget = 8192, 128, 4 << 20
    monkeypatch.setattr(Worker, "_call_bytes", property(lambda _: budget))
    settings = DeviceSettings(FLOAT32, FLOAT32, AttentionKernel())
    worker = Worker(settings, compile_cache=False)
    worker.add_model(0, tabicl_functions, model)
    worker.allocate(0, 0, 0)
    worker.add_table(0, 0, 3, FEATURES, page_size=page)
    state_rows, fused_rows = worker.fit_rows(0, FEATURES)
    # Past the rows one call embeds with the state, but not past those the
    # state alone is fit on.
    assert fused_rows < rows <= state_rows
    # Every call that runs, with the budget it ran under.
    ran: list[tuple[Call, int]] = []
    run = Chunks.run

    def recorded(chunks, stage, key, count, budget, estimate, prepare, consume):
        prepared: dict[tuple[int, int], Call] = {}

        def preparing(first: int, stop: int, padded: int) -> Call:
            prepared[first, stop] = prepare(first, stop, padded)
            return prepared[first, stop]

        def consuming(first: int, stop: int, outputs: object) -> None:
            ran.append((prepared[first, stop], budget))
            consume(first, stop, outputs)

        run(chunks, stage, key, count, budget, estimate, preparing, consuming)

    monkeypatch.setattr(Chunks, "run", recorded)
    generator = np.random.default_rng(0)
    features = generator.normal(size=(rows, FEATURES)).astype(np.float32)
    targets = (np.arange(rows) % 3).astype(np.int32)
    ids = np.arange(rows, dtype=np.uint32)
    # The state alone, then its pages embedded as a plan reads them.
    fit = protocol.Fit(0, 0, ids, features, targets, [])
    pages = [
        protocol.Page(
            0,
            CacheKey(0, 0, 0, "rows", index),
            ids[start : start + page],
            "transient",
            [],
            features[start : start + page],
            targets[start : start + page],
        )
        for index, start in enumerate(range(0, rows, page))
    ]

    worker.run(protocol.Step([], [fit], pages, [], []))

    # XLA's measure of each call is within the budget, which a one-pass fit
    # far exceeds.
    assert worker.stats.calls["fit"] == 1 and worker.stats.calls["embed"] > 1
    for call, limit in ran:
        assert measured_bytes(call) * call.padded <= limit
    one_pass = (
        jax.jit(lambda m, x, y: tabicl_functions.fit(m, x, y, 3))
        .lower(model, features, targets)
        .compile()
        .memory_analysis()
    )
    assert one_pass is not None
    assert one_pass.temp_size_in_bytes > 4 * budget


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_fits_at_their_limits_stay_in_the_call_budget(
    case, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = 8 << 20
    monkeypatch.setattr(Worker, "_call_bytes", property(lambda _: budget))
    settings = DeviceSettings(FLOAT32, FLOAT32, AttentionKernel())
    worker = Worker(settings, compile_cache=False)
    worker.add_model(0, case.functions(), case.build(0))
    classes = 0 if case.regression else 3
    worker.add_table(0, 0, classes, FEATURES)
    state_rows, fused_rows = worker.fit_rows(0, FEATURES)
    # A fit before its limits are measured takes fewer rows than either.
    assert worker.fit_bound(0, FEATURES) <= min(state_rows, fused_rows)

    # The worker's calls for a table's fit, at each limit: a padded size.
    for stage, rows in (("fit", fused_rows), ("state", state_rows)):
        call = worker._compiled_stage(stage, 0, classes, 4)
        target = jnp.float32 if case.regression else jnp.int32
        inputs = (
            jax.ShapeDtypeStruct((1, rows, FEATURES), jnp.float32),
            jax.ShapeDtypeStruct((1, rows), target),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1, FEATURES), jnp.bool_),
        )
        parameters = worker._models[0].parameters
        analysis = jax.jit(call).lower(parameters, *inputs).compile().memory_analysis()
        assert analysis is not None
        taken = analysis.temp_size_in_bytes + analysis.output_size_in_bytes
        assert taken + tree_bytes(inputs) <= budget, stage
