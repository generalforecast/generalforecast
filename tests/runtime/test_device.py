# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.nn.attention import (
    AttentionCache,
    AttentionKernel,
    AttentionParameters,
)
from generalforecast.runtime.device import (
    DeviceSettings,
    available_memory,
    cast_floating,
    configure,
    default_settings,
    device_settings,
    host_memory,
    report_fallback,
    store_cache,
    to_cache_dtype,
    with_precision,
    without_command_buffers,
)
from tests.models.tabicl.tiny import tiny_tabicl

FLOAT32, FLOAT16, BFLOAT16, INT8 = (
    jnp.dtype(name) for name in ("float32", "float16", "bfloat16", "int8")
)
BLOCK = 1 << 18


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        (
            "cpu",
            DeviceSettings(FLOAT32, FLOAT32, AttentionKernel("xla", 256, None, BLOCK)),
        ),
        (
            "gpu",
            DeviceSettings(
                FLOAT32, FLOAT16, AttentionKernel("cudnn", 256, FLOAT16, BLOCK)
            ),
        ),
        (
            "tpu",
            DeviceSettings(
                FLOAT32, BFLOAT16, AttentionKernel("xla", 1024, None, BLOCK)
            ),
        ),
    ],
)
def test_device_settings_per_platform(platform: str, expected: DeviceSettings) -> None:
    assert device_settings(platform) == expected


@pytest.mark.parametrize(("platform", "chunk"), [("gpu", 256), ("tpu", 1024)])
def test_exact_settings_keep_float32_with_full_precision_matmuls(
    platform: str, chunk: int
) -> None:
    exact = device_settings(platform, exact=True)

    assert exact == DeviceSettings(
        FLOAT32,
        FLOAT32,
        AttentionKernel("xla", chunk, block_chunk=BLOCK),
        matmul_precision="highest",
    )


def test_gf_precision_picks_the_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = jax.default_backend()
    monkeypatch.setenv("GF_PRECISION", "exact")
    exact = default_settings()
    monkeypatch.delenv("GF_PRECISION")
    fast = default_settings()
    monkeypatch.setenv("GF_PRECISION", "sloppy")

    assert exact == device_settings(backend, exact=True)
    assert fast == device_settings(backend)
    with pytest.raises(ValueError, match="GF_PRECISION"):
        default_settings()


def test_matmul_precision_is_traced_into_programs() -> None:
    def product(left: jax.Array, right: jax.Array) -> jax.Array:
        return left @ right

    ones = jnp.ones((4, 4))
    traced = str(jax.make_jaxpr(with_precision(product, "highest"))(ones, ones))

    assert "HIGHEST" in traced


def test_int8_caches_keep_a_scale_per_row_and_head() -> None:
    generator = np.random.default_rng(0)
    key = jnp.asarray(generator.normal(size=(1, 16, 2, 8)), jnp.float32)
    cache = AttentionCache(key=key, value=2 * key, length=jnp.asarray([12]))

    stored = to_cache_dtype(cache, INT8)

    assert stored.key.dtype == INT8 and stored.key_scale is not None
    assert stored.key_scale.shape == (1, 16, 2)
    restored = stored.key.astype(jnp.float32) * stored.key_scale[..., None]
    np.testing.assert_allclose(restored, key, atol=float(jnp.max(jnp.abs(key))) / 127)
    np.testing.assert_array_equal(stored.length, cache.length)


def test_gf_cache_dtype_overrides_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GF_CACHE_DTYPE", "int8")
    assert default_settings().cache_dtype == INT8
    monkeypatch.setenv("GF_CACHE_DTYPE", "float8")
    with pytest.raises(ValueError, match="GF_CACHE_DTYPE"):
        default_settings()


def test_device_settings_are_hashable_for_this_machine() -> None:
    settings = device_settings(jax.default_backend())

    assert hash(settings) == hash(device_settings(jax.default_backend()))


def test_device_settings_refuse_unknown_platforms() -> None:
    with pytest.raises(ValueError, match="platform"):
        device_settings("abacus")


def test_cast_floating_leaves_integers_and_none_alone() -> None:
    tree = {
        "weight": jnp.ones((2, 3), jnp.float32),
        "length": jnp.asarray([4], jnp.int32),
        "bias": None,
    }

    cast = cast_floating(tree, jnp.bfloat16)

    assert cast["weight"].dtype == jnp.bfloat16
    assert cast["length"].dtype == jnp.int32
    assert cast["bias"] is None


def test_configure_casts_parameters_and_sets_every_kernel() -> None:
    settings = DeviceSettings(FLOAT16, FLOAT16, AttentionKernel("xla", 16))

    model = configure(tiny_tabicl(regression=False), settings)

    attentions = [
        node
        for node in jax.tree.leaves(
            model, is_leaf=lambda node: isinstance(node, AttentionParameters)
        )
        if isinstance(node, AttentionParameters)
    ]
    # Two ICL blocks, two row blocks, and compress and expand in two column blocks.
    assert len(attentions) == 8
    assert all(attention.kernel == settings.kernel for attention in attentions)
    assert all(leaf.dtype == jnp.float16 for leaf in jax.tree.leaves(model))


def test_store_cache_casts_keys_and_values_but_not_lengths() -> None:
    cache = AttentionCache(
        key=jnp.ones((1, 4, 2, 8), jnp.float32),
        value=jnp.ones((1, 4, 2, 8), jnp.float32),
        length=jnp.asarray([3], jnp.int32),
    )

    stored = store_cache(cache, device_settings("gpu"))

    assert stored.key.dtype == stored.value.dtype == jnp.float16
    assert stored.length is not None
    assert stored.length.dtype == jnp.int32


def test_workers_keep_compiled_programs_on_disk() -> None:
    from generalforecast.worker import Worker

    Worker()

    assert jax.config.jax_compilation_cache_dir


def test_a_worker_starts_without_process_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As on macOS, which has no CPU affinity.
    from generalforecast import system
    from generalforecast.worker import Worker

    monkeypatch.delattr(system.os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(system.os, "sched_setaffinity", raising=False)
    monkeypatch.setattr(system.os, "cpu_count", lambda: 6)

    assert system.usable_cpus() == 6
    system.pin_cpus([0])
    Worker().close()


def test_command_buffers_are_off_unless_xla_flags_set_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/x")
    without_command_buffers()
    assert os.environ["XLA_FLAGS"] == (
        "--xla_dump_to=/tmp/x --xla_gpu_enable_command_buffer="
    )
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=FUSION")
    without_command_buffers()
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer=FUSION"


def test_available_memory_is_what_a_container_or_machine_has_free(
    tmp_path: Path,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1000 kB\nMemAvailable: 600 kB\n")
    cgroup = tmp_path / "cgroup"
    (cgroup / "memory").mkdir(parents=True)

    if sys.platform == "darwin":
        # macOS has no meminfo, and counts its free and inactive pages instead.
        free = available_memory(cgroup, tmp_path / "missing")
        assert free is not None and 0 < free <= host_memory()
    else:
        assert available_memory(cgroup, tmp_path / "missing") is None
    assert available_memory(cgroup, meminfo) == 600 * 1024
    (cgroup / "memory" / "memory.limit_in_bytes").write_text(f"{512 << 20}\n")
    (cgroup / "memory" / "memory.usage_in_bytes").write_text(f"{192 << 20}\n")
    assert available_memory(cgroup, meminfo) == 320 << 20
    (cgroup / "memory.max").write_text(f"{256 << 20}\n")
    (cgroup / "memory.current").write_text(f"{64 << 20}\n")
    assert available_memory(cgroup, meminfo) == 192 << 20


def test_host_memory_is_a_containers_limit(tmp_path: Path) -> None:
    machine = host_memory(tmp_path)
    (tmp_path / "memory.max").write_text("max\n")
    assert host_memory(tmp_path) == machine
    (tmp_path / "memory.max").write_text(f"{64 << 30}\n")
    assert host_memory(tmp_path) == min(machine, 64 << 30)
    (tmp_path / "memory.max").write_text(f"{1 << 20}\n")
    assert host_memory(tmp_path) == 1 << 20


@pytest.mark.parametrize(
    ("machine", "flags", "xla", "expected"),
    [
        (
            "x86_64",
            ["avx", "avx2", "fma", "f16c", "avx512f"],
            "",
            "--xla_cpu_max_isa=AVX2",
        ),
        ("x86_64", ["avx", "sse4_2"], "", ""),
        ("arm64", [], "", ""),
        (
            "x86_64",
            ["avx", "avx2", "fma", "f16c"],
            "--xla_cpu_max_isa=SSE4_2",
            "--xla_cpu_max_isa=SSE4_2",
        ),
    ],
)
def test_cpu_programs_pin_avx2_where_the_cpu_has_it(
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    flags: list[str],
    xla: str,
    expected: str,
) -> None:
    from generalforecast.runtime import device

    monkeypatch.setattr(device.platform, "machine", lambda: machine)
    monkeypatch.setattr(device, "cpu_flags", lambda: flags)
    monkeypatch.setenv("XLA_FLAGS", xla)

    device.portable_cpu_programs()

    assert os.environ["XLA_FLAGS"] == expected


_COMPILED_TWICE = """
import sys, threading
import jax, numpy as np
from generalforecast.runtime import device

jax.config.update("jax_compilation_cache_dir", sys.argv[1])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
device.count_compile_cache()
doubled = jax.jit(lambda rows: rows * 2)
doubled(np.ones(4, np.float32))
first = device.compile_cache_counts()
doubled(np.ones(4, np.float32))
called = device.compile_cache_counts()
jax.clear_caches()
doubled(np.ones(4, np.float32))
loaded = device.compile_cache_counts()
compiles = [
    threading.Thread(
        target=lambda n=n: doubled.lower(jax.ShapeDtypeStruct((n,), np.float32)).compile()
    )
    for n in (5, 6, 7)
]
for compile in compiles:
    compile.start()
for compile in compiles:
    compile.join()
print(first, called, loaded, device.compile_cache_counts())
"""


def test_programs_the_persistent_compile_cache_holds_count_as_hits_and_others_as_misses(
    tmp_path: Path,
) -> None:
    # In a process of its own, since JAX sets its persistent cache up once.
    found = subprocess.run(
        [sys.executable, "-c", _COMPILED_TWICE, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    ).stdout.split(")")
    first, called, loaded, threaded = (
        tuple(int(count) for count in part.strip(" (\n").split(", "))
        for part in found[:4]
    )
    hits, misses = first
    assert hits == 0 and misses >= 1
    # A call of a compiled program is no compile.
    assert called == first
    # The same program again, once JAX's own caches are cleared, loads from disk.
    assert loaded == (misses, misses)
    # Compiles on threads at once each count once, for their own lookups.
    assert threaded == (misses, misses + 3)


@pytest.fixture
def cpu_backend(monkeypatch):
    """A process whose JAX backend is the CPU and that has not warned yet."""
    from generalforecast.runtime import device

    monkeypatch.setattr(device.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(device, "_reported", False)
    monkeypatch.setattr(device._advice, "held", [])
    return device


def test_report_fallback_warns_once_about_a_visible_gpu(cpu_backend, tmp_path):
    import io

    gpu = tmp_path / "nvidia0"
    gpu.touch()
    err = io.StringIO()
    assert report_fallback(err, devices=[str(gpu)])
    lines = err.getvalue().splitlines()
    assert (
        lines[0]
        == "warning: an NVIDIA GPU is present, but generalforecast runs on the CPU"
    )
    assert (
        lines[1]
        == '  hint: install the cuda extra to use it: pip install "generalforecast[cuda]"'
    )
    assert not report_fallback(err, devices=[str(gpu)])
    assert len(err.getvalue().splitlines()) == 2


def test_report_fallback_stays_quiet_without_a_gpu(cpu_backend, tmp_path):
    import io

    err = io.StringIO()
    assert not report_fallback(err, devices=[str(tmp_path / "nvidia0")])
    assert err.getvalue() == ""


def test_report_fallback_stays_quiet_on_a_gpu(cpu_backend, monkeypatch, tmp_path):
    import io

    monkeypatch.setattr(cpu_backend.jax, "default_backend", lambda: "gpu")
    gpu = tmp_path / "nvidia0"
    gpu.touch()
    err = io.StringIO()
    assert not report_fallback(err, devices=[str(gpu)])
    assert err.getvalue() == ""


def test_report_fallback_replaces_jax_advice(cpu_backend, tmp_path, caplog):
    import io
    import logging

    gpu = tmp_path / "nvidia0"
    gpu.touch()
    report_fallback(io.StringIO(), devices=[str(gpu)])
    logger = logging.getLogger("jax._src.xla_bridge")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        logger.warning(
            "An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib "
            "is not installed. Falling back to cpu."
        )
        logger.warning(
            "CUDA backend failed to initialize: no device. Falling back to cpu."
        )
        logger.warning(
            "A Google TPU may be present on this machine, but either a TPU-enabled "
            "jaxlib or libtpu is not installed. Falling back to cpu."
        )
        logger.warning("unrelated advice")
    assert [record.getMessage() for record in caplog.records] == [
        (
            "A Google TPU may be present on this machine, but either a TPU-enabled "
            "jaxlib or libtpu is not installed. Falling back to cpu."
        ),
        "unrelated advice",
    ]
    assert len(cpu_backend._advice.held) == 2


def test_report_fallback_gives_the_cuda_failure_as_the_hint(cpu_backend, tmp_path):
    import io

    gpu = tmp_path / "nvidia0"
    gpu.touch()
    cpu_backend._advice.held.append("CUDA backend failed to initialize: no device.")
    err = io.StringIO()
    assert report_fallback(err, devices=[str(gpu)])
    assert err.getvalue().splitlines()[1] == (
        "  hint: CUDA failed to start. CUDA backend failed to initialize: no device."
    )


def test_report_fallback_respects_a_chosen_platform(cpu_backend, monkeypatch, tmp_path):
    import io

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    gpu = tmp_path / "nvidia0"
    gpu.touch()
    err = io.StringIO()
    assert not report_fallback(err, devices=[str(gpu)])
    assert err.getvalue() == ""


def test_report_fallback_reports_a_cuda_failure_on_a_chosen_platform(
    cpu_backend, monkeypatch, tmp_path
):
    import io

    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    gpu = tmp_path / "nvidia0"
    gpu.touch()
    cpu_backend._advice.held.append("CUDA backend failed to initialize: no device.")
    err = io.StringIO()
    assert report_fallback(err, devices=[str(gpu)])
    assert err.getvalue().splitlines()[1] == (
        "  hint: CUDA failed to start. CUDA backend failed to initialize: no device."
    )


def test_report_fallback_reports_a_cuda_failure_without_a_device_file(
    cpu_backend, tmp_path
):
    import io

    cpu_backend._advice.held.append("CUDA backend failed to initialize: no device.")
    err = io.StringIO()
    assert report_fallback(err, devices=[str(tmp_path / "nvidia0")])
    assert err.getvalue().splitlines() == [
        "warning: CUDA failed to start, so generalforecast runs on the CPU",
        "  hint: CUDA backend failed to initialize: no device.",
    ]
    assert not report_fallback(err, devices=[str(tmp_path / "nvidia0")])
