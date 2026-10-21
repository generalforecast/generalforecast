import ctypes
import dataclasses
import functools
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from generalforecast.nn.attention import (
    AttentionCache,
    AttentionKernel,
    AttentionParameters,
)

FLOAT32 = jnp.dtype("float32")
FLOAT16 = jnp.dtype("float16")
BFLOAT16 = jnp.dtype("bfloat16")
INT8 = jnp.dtype("int8")
_CACHE_DTYPES = {
    "float32": FLOAT32,
    "float16": FLOAT16,
    "bfloat16": BFLOAT16,
    "int8": INT8,
}


@dataclass(frozen=True)
class DeviceSettings:
    """How models run and what the runtime stores on one kind of device.

    Attributes:
        compute_dtype: Dtype of parameters and activations.
        cache_dtype: Dtype the runtime stores context caches in.
        kernel: How attention runs, including its own dtype.
        matmul_precision: JAX's precision for float32 matrix multiplies:
            `default` lets GPUs use TF32, `highest` keeps float32.
    """

    compute_dtype: jnp.dtype
    cache_dtype: jnp.dtype
    kernel: AttentionKernel
    matmul_precision: str = "default"


# Measured on the TabICL v2 checkpoints on an NVIDIA L4, against float32
# (MagicTelescope probabilities; California medians in target deviations):
#
#   context rows             1,024           8,192
#   float16 attention     0.022 / 0.056   0.027 / 0.054
#   bfloat16 attention    0.14  / 0.21    0.15  / 0.53
#   float16 cache        0.0007 / 0.003  0.0010 / 0.004
#   fast (below)          0.016 / 0.043   0.026 / 0.054, 2x and 5x faster
#
# An int8 cache, opt-in through GF_CACHE_DTYPE, moved them by 0.0026 / 0.027 on
# a CPU at 1,024 rows, and takes 13 KiB a row against 24 KiB in float16.
#
# On TabPFN classifiers, on a CPU at 350 rows, a float16 cache moved
# probabilities by 0.0016 on TabPFN-3.5 and 0.023 on TabPFN-3, and an int8
# cache by 0.0081 and 0.18. TabPFN-3's cached keys quantize poorly. Int8 also
# quantizes the retrieval decoder's keys, which TabPFN keeps exact; without
# them TabPFN-3.5 moved by 0.0045.
#
# QASSMax sharpens attention by log n, which amplifies rounding in the scores,
# so bfloat16 attention is out. Fast mode takes float16 cuDNN attention and
# float16 caches for their speed; exact mode keeps float32 throughout. TPUs have
# no fast float16, so fast mode stores bfloat16 caches there until measured.
# Attention that falls back to XLA, as masked or int8 attention does, chunks
# its queries; cuDNN's flash attention never holds every score, so it does not.
# Attention blocks take at most this many positions per step, so large tables
# fit in memory; smaller ones run in one step.
_BLOCK_CHUNK = 1 << 18
_FAST = {
    "cpu": DeviceSettings(
        FLOAT32, FLOAT32, AttentionKernel("xla", 256, block_chunk=_BLOCK_CHUNK)
    ),
    "gpu": DeviceSettings(
        FLOAT32,
        FLOAT16,
        AttentionKernel("cudnn", 256, dtype=FLOAT16, block_chunk=_BLOCK_CHUNK),
    ),
    "tpu": DeviceSettings(
        FLOAT32, BFLOAT16, AttentionKernel("xla", 1024, block_chunk=_BLOCK_CHUNK)
    ),
}
_CHUNKS = {"cpu": 256, "gpu": 256, "tpu": 1024}


def device_settings(platform: str, exact: bool = False) -> DeviceSettings:
    """Settings for a JAX platform, such as `jax.default_backend()`.

    Args:
        platform: `cpu`, `gpu` or `tpu`.
        exact: Whether to keep float32 throughout, matrix multiplies included,
            instead of the platform's fast settings.

    Returns:
        The platform's settings.
    """
    if platform not in _FAST:
        raise ValueError(f"no device settings for platform {platform!r}")
    if not exact:
        return _FAST[platform]
    kernel = AttentionKernel("xla", _CHUNKS[platform], block_chunk=_BLOCK_CHUNK)
    return DeviceSettings(FLOAT32, FLOAT32, kernel, matmul_precision="highest")


def default_settings() -> DeviceSettings:
    """This machine's settings, from the environment.

    `GF_PRECISION` names the mode, `fast` (default) or `exact`, and
    `GF_CACHE_DTYPE` overrides the dtype caches are stored in: `float32`,
    `float16`, `bfloat16`, or `int8` with a scale per row and head.
    """
    mode = os.environ.get("GF_PRECISION", "fast")
    if mode not in ("fast", "exact"):
        raise ValueError(f"GF_PRECISION must be fast or exact, not {mode!r}")
    settings = device_settings(jax.default_backend(), exact=mode == "exact")
    cache = os.environ.get("GF_CACHE_DTYPE")
    if cache is None:
        return settings
    if cache not in _CACHE_DTYPES:
        raise ValueError(
            f"GF_CACHE_DTYPE must be one of {sorted(_CACHE_DTYPES)}, not {cache!r}"
        )
    return dataclasses.replace(settings, cache_dtype=_CACHE_DTYPES[cache])


def host_memory(cgroup: Path = Path("/sys/fs/cgroup")) -> int:
    """Bytes of memory this process may use: the machine's, or its container's limit.

    A container sees the whole machine's memory, so the limit comes from its
    cgroup: `memory.max` under cgroup v2, `memory/memory.limit_in_bytes` under v1.

    Args:
        cgroup: Where cgroups are mounted.
    """
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    for limit in (cgroup / "memory.max", cgroup / "memory" / "memory.limit_in_bytes"):
        try:
            text = limit.read_text().strip()
        except OSError:
            continue
        if text.isdigit():
            memory = min(memory, int(text))
    return memory


def available_memory(
    cgroup: Path = Path("/sys/fs/cgroup"), meminfo: Path = Path("/proc/meminfo")
) -> int | None:
    """Bytes of memory still free to this process, or None where it cannot tell.

    In a container, its limit less what the container uses, every process in
    it counted: `memory.current` under cgroup v2, `memory/memory.usage_in_bytes`
    under v1, which counts its page cache too. Outside one, the machine's
    available memory: `MemAvailable` on Linux, free and inactive pages on macOS.

    Args:
        cgroup: Where cgroups are mounted.
        meminfo: The kernel's memory summary.
    """
    for used in (
        cgroup / "memory.current",
        cgroup / "memory" / "memory.usage_in_bytes",
    ):
        try:
            text = used.read_text().strip()
        except OSError:
            continue
        if text.isdigit():
            return max(0, host_memory(cgroup) - int(text))
    try:
        lines = meminfo.read_text().splitlines()
    except OSError:
        return _darwin_available() if sys.platform == "darwin" else None
    for line in lines:
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


# `host_statistics64`'s flavor for `vm_statistics64`, and that struct's size in
# 32-bit words.
_HOST_VM_INFO64 = 4
_VM_INFO64_WORDS = 38


@functools.cache
def _mach_host() -> tuple[ctypes.CDLL, int]:
    """The system library and this host's Mach port, taken once."""
    system = ctypes.CDLL(None)
    system.mach_host_self.restype = ctypes.c_uint
    return system, system.mach_host_self()


def _darwin_available() -> int | None:
    """Free and inactive memory on macOS, which reclaims inactive pages first."""
    try:
        system, host = _mach_host()
        count = ctypes.c_uint(_VM_INFO64_WORDS)
        info = (ctypes.c_uint * _VM_INFO64_WORDS)()
        failed = system.host_statistics64(
            host, _HOST_VM_INFO64, info, ctypes.byref(count)
        )
    except (OSError, AttributeError):
        return None
    if failed:
        return None
    # `vm_statistics64` starts with its free, active and inactive page counts.
    return (info[0] + info[2]) * os.sysconf("SC_PAGE_SIZE")


def without_command_buffers() -> None:
    """Turns off XLA's command buffers (CUDA graphs) unless `XLA_FLAGS` sets them.

    With them, steps vectorized over many tables failed on GPUs with
    `cudaErrorStreamCaptureInvalidated` while XLA captured a command buffer.
    It takes effect only before JAX starts its first backend.
    """
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_enable_command_buffer" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_enable_command_buffer=".strip()


def device_memory() -> int:
    """Bytes of memory a worker counts as its device's.

    An accelerator's own memory, or on a CPU, where the device is host memory
    too, a quarter of the host's; `MemoryBudget` divides it.
    """
    stats = jax.devices()[0].memory_stats() or {}
    return int(stats.get("bytes_limit", host_memory() // 4))


def use_compile_cache() -> None:
    """Keeps compiled programs on disk, so a later process loads them instead of compiling.

    In `$XDG_CACHE_HOME/generalforecast/jax`, `~/.cache/generalforecast/jax` by default, and every
    program however quick it was to compile. The cache has no size limit: with
    one, JAX's bookkeeping made a cold start about four times slower. A cache
    JAX already has, such as one `JAX_COMPILATION_CACHE_DIR` names, is kept,
    and keeps every program unless `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS`
    says otherwise.
    """
    if not jax.config.jax_compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", str(cache_root() / "jax"))
    # Workers compile the programs earlier ones recorded as they start, so a
    # quick program is worth keeping too; JAX keeps only those over a second.
    if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def cache_root() -> Path:
    """Where generalforecast keeps what later processes reuse: `$XDG_CACHE_HOME/generalforecast`, `~/.cache/generalforecast` by default."""
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return Path(root) / "generalforecast"


def with_precision(function: Callable[..., Any], precision: str) -> Callable[..., Any]:
    """`function`, traced with JAX's float32 matrix multiplies at `precision`."""

    @functools.wraps(function)
    def traced(*arguments: Any) -> Any:
        with jax.default_matmul_precision(precision):
            return function(*arguments)

    return traced


def cast_floating(tree: Any, dtype: jnp.dtype) -> Any:
    """Casts the floating-point arrays in `tree` to `dtype`, leaving the rest.

    Args:
        tree: Pytree of arrays.
        dtype: Target dtype.

    Returns:
        The tree with floating-point arrays cast.
    """

    def cast(leaf: jax.Array) -> jax.Array:
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(dtype)
        return leaf

    return jax.tree.map(cast, tree)


def configure(model: Any, settings: DeviceSettings) -> Any:
    """Prepares model parameters for a device: their dtype and attention kernels.

    Args:
        model: Model parameters.
        settings: Device settings.

    Returns:
        Parameters in the compute dtype, with every attention using the kernel.
    """

    def set_kernel(node: Any) -> Any:
        if isinstance(node, AttentionParameters):
            return dataclasses.replace(node, kernel=settings.kernel)
        return node

    model = jax.tree.map(
        set_kernel, model, is_leaf=lambda node: isinstance(node, AttentionParameters)
    )
    return cast_floating(model, settings.compute_dtype)


def with_summary_chunk(model: Any, cells: int | None) -> Any:
    """Model parameters whose fits read at most `cells` cells a step for their column summary.

    Args:
        model: Model parameters, which may be traced.
        cells: Cells a step reads (see `AttentionKernel.summary_chunk`), or
            None to read every cell at once.

    Returns:
        The parameters with every attention's kernel set so.
    """

    def set_chunk(node: Any) -> Any:
        if isinstance(node, AttentionParameters):
            kernel = dataclasses.replace(node.kernel, summary_chunk=cells)
            return dataclasses.replace(node, kernel=kernel)
        return node

    return jax.tree.map(
        set_chunk, model, is_leaf=lambda node: isinstance(node, AttentionParameters)
    )


def to_cache_dtype(cache: Any, dtype: jnp.dtype) -> Any:
    """Converts a context cache to a storage dtype.

    Floating dtypes cast. `int8` quantizes each row and head's keys and values
    symmetrically, keeping one float32 scale per row and head beside them.

    Args:
        cache: Cache from a model's `encode`.
        dtype: The storage dtype.

    Returns:
        The stored cache. Lengths stay integers.
    """
    if dtype != INT8:
        return cast_floating(cache, dtype)

    def quantize(values: jax.Array) -> tuple[jax.Array, jax.Array]:
        values = values.astype(FLOAT32)
        scale = jnp.max(jnp.abs(values), axis=-1) / 127
        scale = jnp.where(scale > 0, scale, 1.0)
        return jnp.round(values / scale[..., None]).astype(INT8), scale

    def store(layer: Any) -> Any:
        if not isinstance(layer, AttentionCache):
            return layer
        key, key_scale = quantize(layer.key)
        value, value_scale = quantize(layer.value)
        return layer._replace(
            key=key, value=value, key_scale=key_scale, value_scale=value_scale
        )

    return jax.tree.map(
        store, cache, is_leaf=lambda node: isinstance(node, AttentionCache)
    )


@functools.partial(jax.jit, static_argnums=1)
def store_cache(cache: Any, settings: DeviceSettings) -> Any:
    """Converts a context cache to the dtype the runtime stores it in."""
    return to_cache_dtype(cache, settings.cache_dtype)


def load_cache(cache: Any) -> Any:
    """A stored context cache back in float32, its scales applied and dropped.

    For models whose `decode` reads caches as `encode` returned them; see
    `TabularProperties.stored_cache`.
    """

    def load(layer: Any) -> Any:
        if not isinstance(layer, AttentionCache):
            return layer
        key, value = layer.key.astype(FLOAT32), layer.value.astype(FLOAT32)
        if layer.key_scale is not None:
            key = key * layer.key_scale[..., None]
        if layer.value_scale is not None:
            value = value * layer.value_scale[..., None]
        return layer._replace(key=key, value=value, key_scale=None, value_scale=None)

    return jax.tree.map(
        load, cache, is_leaf=lambda node: isinstance(node, AttentionCache)
    )
