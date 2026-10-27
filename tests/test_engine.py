# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The local engine: gf-server's shards in this process, models loaded in the workers."""

import gc
import subprocess
import sys
import threading

import numpy as np
import pytest

from generalforecast import _core
from generalforecast.engine import Labeled, LocalEngine, _Models
from generalforecast.models.catalog import ModelRefError, parse
from tests.utils.engine import tabular_request

REF = parse("general-forecast/tabicl-v2-classifier")
TINY = {"general-forecast/tabicl-v2-classifier": "tests/tabicl-classifier"}
CALLERS = 8


@pytest.fixture(scope="module")
def engine():
    engine = LocalEngine(models=TINY, cpu_threads=2)
    yield engine
    engine.close()


def _request(seed: int, queries: int = 4):
    """Rows around 0 and 10 with three classes, shifted by `seed`."""
    generator = np.random.default_rng(seed)
    centers = np.where(np.arange(32) % 2 == 0, 0.0, 10.0)[:, None]
    context = (centers + generator.normal(size=(32, 5))).astype(np.float32)
    labels = [("a", "b", "c")[index % 3] for index in range(32)]
    return tabular_request(
        "general-forecast/tabicl-v2-classifier",
        context,
        labels,
        generator.normal(size=(queries, 5)).astype(np.float32),
    )


def test_concurrent_callers_get_what_each_would_alone(engine):
    requests = [_request(seed) for seed in range(CALLERS)]
    alone = [engine.tabular(REF, request) for request in requests]
    results: list[np.ndarray | None] = [None] * CALLERS

    def call(index: int) -> None:
        results[index] = engine.tabular(REF, requests[index])

    threads = [threading.Thread(target=call, args=(i,)) for i in range(CALLERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result, want in zip(results, alone, strict=True):
        assert result is not None and result.shape == (4, 3)
        np.testing.assert_allclose(result, want, rtol=1e-5, atol=1e-6)


def test_a_repeated_context_answers_alike(engine):
    request = _request(100)
    first = engine.tabular(REF, request)
    np.testing.assert_allclose(engine.tabular(REF, request), first, rtol=1e-6)
    np.testing.assert_allclose(first.sum(axis=1), 1, atol=1e-5)


@pytest.mark.parametrize(
    ("model", "labels"),
    [
        (
            "general-forecast/tabicl-v2-classifier",
            [("a", "b", "c")[row % 3] for row in range(300)],
        ),
        ("general-forecast/tabicl-v2-regressor", list(np.linspace(0, 1, 300))),
    ],
    ids=["classifier", "regressor"],
)
def test_a_fused_first_answer_is_within_float32_of_its_repeats(model, labels):
    """A single-use context's first call fuses its fit, encode and decode; repeats do not.

    The two order the same float32 arithmetic differently, so they answer
    within 1e-6 of each other (about 3e-7 seen), not bitwise alike.
    """
    models = {**TINY, "general-forecast/tabicl-v2-regressor": "tests/tabicl-regressor"}
    engine = LocalEngine(models=models, cpu_threads=2)
    try:
        generator = np.random.default_rng(0)
        request = tabular_request(
            model,
            generator.normal(size=(300, 5)).astype(np.float32),
            labels,
            generator.normal(size=(6, 5)).astype(np.float32),
        )
        first = engine.tabular(parse(model), request)
        for _ in range(2):
            repeat = engine.tabular(parse(model), request)
            assert float(np.max(np.abs(repeat - first))) <= 1e-6
    finally:
        engine.close()


def test_a_digest_pinned_reference_runs_its_spec(engine):
    digest = REF.entry.artifact.digest
    assert digest is not None
    pinned = parse(f"general-forecast/tabicl-v2-classifier@{digest}")
    request = _request(300)
    np.testing.assert_allclose(
        engine.tabular(pinned, request), engine.tabular(REF, request), rtol=1e-5
    )


def test_a_bound_context_answers_as_its_request_does(engine):
    request = _request(200)
    context = Labeled(
        request.task, request.context, request.targets, tuple(request.classes)
    )
    bound = engine.bind(REF, context)
    np.testing.assert_allclose(
        engine.predict(bound, request.queries),
        engine.tabular(REF, request),
        rtol=1e-5,
        atol=1e-6,
    )
    table = bound.table
    del bound
    gc.collect()
    with pytest.raises(_core.RequestError):
        engine._shared.shards.predict(table, request.queries)


def test_a_refused_model_never_reaches_the_workers(engine):
    with pytest.raises(ValueError, match="unknown model"):
        engine.gated("remote", False).tabular(
            parse("tests/tabicl-classifier"), _request(0)
        )


def test_the_engine_process_imports_no_model_code():
    script = (
        "import sys\n"
        "from generalforecast.engine import LocalEngine\n"
        "from generalforecast.models.catalog import parse\n"
        "from tests.test_engine import TINY, _request\n"
        "engine = LocalEngine(models=TINY, cpu_threads=8)\n"
        "assert engine.tabular(parse('general-forecast/tabicl-v2-classifier'), _request(0)).shape == (4, 3)\n"
        "engine.close()\n"
        "loaded = [m for m in sys.modules if m.startswith(('jax', 'generalforecast.models.registry', 'generalforecast.models.tabicl', 'generalforecast.worker'))]\n"
        "assert not loaded, loaded\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-2000:]


def test_a_model_loading_holds_up_neither_loaded_models_nor_other_loads():
    """Loading one model takes no lock a loaded model's callers or another load wait on."""
    slow_started, slow_released = threading.Event(), threading.Event()
    added: list[str] = []

    def add(name: str, spec: str) -> None:
        if name == "slow":
            slow_started.set()
            assert slow_released.wait(10)
        added.append(name)

    models = _Models(add)
    models.ensure("loaded", "loaded")
    loading = threading.Thread(target=models.ensure, args=("slow", "slow"))
    loading.start()
    try:
        assert slow_started.wait(10)
        done = threading.Event()

        def callers() -> None:
            models.ensure("loaded", "loaded")
            models.ensure("other", "other")
            done.set()

        threading.Thread(target=callers).start()
        assert done.wait(5), "a loaded model or another load waited on the slow one"
    finally:
        slow_released.set()
        loading.join()
    assert added == ["loaded", "other", "slow"]
    models.ensure("slow", "slow")
    assert added.count("slow") == 1


def test_callers_of_one_model_wait_for_its_single_load():
    started, release = threading.Event(), threading.Event()
    added: list[str] = []

    def add(name: str, spec: str) -> None:
        started.set()
        assert release.wait(10)
        added.append(name)

    models = _Models(add)
    first = threading.Thread(target=models.ensure, args=("m", "m"))
    second = threading.Thread(target=models.ensure, args=("m", "m"))
    first.start()
    assert started.wait(10)
    second.start()
    release.set()
    first.join()
    second.join()
    assert added == ["m"]


def test_a_load_the_workers_refuse_is_tried_again():
    attempts: list[str] = []

    def add(name: str, spec: str) -> None:
        attempts.append(name)
        if len(attempts) == 1:
            raise _core.RequestError("UnknownModel", "no such model", None)

    models = _Models(add)
    with pytest.raises(ModelRefError, match="no such model"):
        models.ensure("m", "m")
    models.ensure("m", "m")
    assert attempts == ["m", "m"]
