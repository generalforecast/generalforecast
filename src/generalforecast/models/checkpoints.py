# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint directories, read and written the same way for every model.

A directory holds `config.json`, naming the architecture and its settings, and
`model.safetensors`, with arrays under PyTorch parameter names. Checkpoints
published as PyTorch files are read without PyTorch to convert them.
"""

import collections
import io
import json
import math
import os
import pickle
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from filelock import FileLock
from huggingface_hub import hf_hub_download
from jax import Array
from safetensors.numpy import load_file, save_file

from generalforecast.models.catalog import short_name
from generalforecast.nn.attention import AttentionParameters
from generalforecast.nn.linear import LinearParameters
from generalforecast.nn.retrieval import RetrievalParameters


class Published(NamedTuple):
    """A checkpoint in a Hugging Face repository, at a pinned revision.

    Attributes:
        repository: The repository.
        revision: The pinned revision.
        filename: The one file a converted checkpoint is read from, or None
            when its directory holds the files.
    """

    repository: str
    revision: str
    filename: str | None = None


def local_checkpoint(name: str) -> Path | None:
    """A built-in checkpoint's directory under `GF_CHECKPOINTS`, or None if unset.

    Raises:
        FileNotFoundError: If `GF_CHECKPOINTS` is set but holds no such directory,
            as on a machine without internet that was given the wrong one.
    """
    local = os.environ.get("GF_CHECKPOINTS")
    if not local:
        return None
    path = Path(local) / name
    if not path.is_dir():
        raise FileNotFoundError(f"GF_CHECKPOINTS has no {name!r}: {path}")
    return path


def converted_checkpoint(
    name: str, revision: str, convert: Callable[[Path], None]
) -> Path:
    """A checkpoint directory converted from published files, converting it if missing.

    With `GF_CHECKPOINTS` set, the directory must already be there, as on
    machines without internet. Otherwise it is kept under
    `~/.cache/generalforecast/checkpoints/<name>/<revision>`, so a new pinned revision
    converts again instead of finding the old one's directory.

    Args:
        name: The checkpoint's name, such as `limix-2-classifier`.
        revision: The pinned revision of the published files.
        convert: Downloads the published files and writes the checkpoint
            directory at the path it is given.

    Returns:
        A directory with `config.json` and `model.safetensors`.
    """
    if (local := local_checkpoint(name)) is not None:
        return local
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = cache / "generalforecast" / "checkpoints" / name / revision
    if directory.is_dir():
        return directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    # One process converts; the others wait, then find its directory.
    with FileLock(directory.parent / f"{revision}.lock"):
        if directory.is_dir():
            return directory
        # Written aside and renamed, so an interrupted conversion leaves nothing.
        with tempfile.TemporaryDirectory(dir=directory.parent) as scratch:
            convert(Path(scratch) / revision)
            os.replace(Path(scratch) / revision, directory)
    return directory


def converted_file(
    id: str,
    published: Published,
    convert: Callable[[Path, str], tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> Path:
    """A checkpoint directory converted from one published file, converting it if missing.

    See `converted_checkpoint` for where the directory is kept.

    Args:
        id: The checkpoint's canonical id.
        published: Where the file is.
        convert: Reads the downloaded file for the checkpoint of a short
            name, as `convert(path, name)`: its architecture, settings and
            arrays keyed by PyTorch parameter name.

    Returns:
        A directory with `config.json` and `model.safetensors`.
    """
    if published.filename is None:
        raise ValueError(f"{id} is published as a directory, not one file")
    name, filename = short_name(id), published.filename

    def write(directory: Path) -> None:
        path = hf_hub_download(
            published.repository, filename, revision=published.revision
        )
        save_state(directory, *convert(Path(path), name))

    return converted_checkpoint(name, published.revision, write)


def without(state: Mapping[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    """A state dict without the parameters whose names start with any of `prefixes`."""
    return {key: value for key, value in state.items() if not key.startswith(prefixes)}


def check_settings(
    model: str,
    settings: Mapping[str, Any],
    supported: Mapping[str, tuple[Any, ...]],
    known: tuple[str, ...] | None = None,
    prefix: str = "",
) -> None:
    """Raises for a setting off the values a loader has been checked with.

    Args:
        model: The model's name, as messages give it.
        settings: The checkpoint's settings, or a section of them.
        supported: Each checked setting's supported values; a missing
            setting reads as None.
        known: The other settings the loader reads, of any value, when every
            setting must be known; None takes any others.
        prefix: What messages put before a setting's name, such as its section.

    Raises:
        NotImplementedError: Naming the unsupported or unknown setting.
    """
    for key, values in supported.items():
        if settings.get(key) not in values:
            raise NotImplementedError(
                f"{model} {prefix}{key}={settings.get(key)!r} is not supported"
            )
    if known is None:
        return
    unknown = sorted(set(settings) - set(supported) - set(known))
    if unknown:
        raise NotImplementedError(
            f"{model} {prefix}settings {unknown} are not supported"
        )


class MissingParameter(KeyError):
    """A parameter a model reads that its checkpoint lacks."""


class Settings(dict[str, Any]):
    """A checkpoint's settings, which name a key `config.json` lacks as such."""

    def __missing__(self, key: str) -> Any:  # noqa: D105
        raise ValueError(f"the checkpoint's config.json lacks the setting {key!r}")


class _Read(dict[str, Array]):
    """A state dict that remembers which parameters were read."""

    def __init__(self, state: Mapping[str, Array]) -> None:
        super().__init__(state)
        self.read: set[object] = set()

    def __getitem__(self, key: str) -> Array:
        self.read.add(key)
        if key not in self:
            raise MissingParameter(key)
        return super().__getitem__(key)

    def get(self, key: object, default: Any = None, /) -> Any:
        self.read.add(key)
        return super().get(key, default)


def read_linear(
    state: Mapping[str, Array], prefix: str, bias: bool | None = True
) -> LinearParameters:
    """Linear parameters stored under `prefix`, with a bias if the layer has one.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        prefix: The layer's name.
        bias: Whether the layer has a bias; without one, none is read. None
            reads one if the state has it, for releases that differ.

    Returns:
        The weight and bias.
    """
    if bias is None:
        stored = state.get(f"{prefix}.bias")
    else:
        stored = state[f"{prefix}.bias"] if bias else None
    return LinearParameters(weight=state[f"{prefix}.weight"], bias=stored)


def read_all[T](
    state: Mapping[str, Array],
    build: Callable[[dict[str, Array]], T],
    buffers: tuple[str, ...] = (),
) -> T:
    """Builds a model from a state dict that must hold exactly its parameters.

    Args:
        state: Arrays keyed by PyTorch parameter name.
        build: Reads the parameters it needs from the state dict.
        buffers: Endings of the names of buffers the model derives from its
            settings instead, such as RoPE frequencies, which need no reading.

    Returns:
        What `build` returns.

    Raises:
        ValueError: If a parameter is missing, or one is left unread, as in a
            checkpoint of a variant the loader does not know, or if a head count
            does not fit its attention's weights.
    """
    read = _Read(state)
    try:
        model = build(read)
    except MissingParameter as error:
        raise ValueError(f"the checkpoint lacks {error.args[0]!r}") from error
    unread = sorted(key for key in set(read) - read.read if not key.endswith(buffers))
    if unread:
        shown = ", ".join(unread[:5]) + (
            f" and {len(unread) - 5} more" if len(unread) > 5 else ""
        )
        raise ValueError(
            f"the checkpoint has parameters the model does not read: {shown}"
        )
    check_heads(model)
    return model


def check_heads(model: Any) -> None:
    """Raises if a head count in a model does not fit its attention's weights.

    Head counts come from a checkpoint's settings and widths from its weights,
    so a mismatch would otherwise surface at the first call as a bare reshape.

    Args:
        model: Model parameters, as a loader builds them.

    Raises:
        ValueError: Naming the attention whose heads do not divide its widths.
    """
    kinds = (AttentionParameters, RetrievalParameters)
    found, _ = jax.tree_util.tree_flatten_with_path(
        model, is_leaf=lambda node: isinstance(node, kinds)
    )
    for path, node in found:
        if not isinstance(node, kinds):
            continue
        name = jax.tree_util.keystr(path).lstrip(".") or "the attention"
        heads, width = node.num_heads, node.query.weight.shape[0]
        if heads < 1 or width % heads:
            raise ValueError(
                f"{name} has {heads} heads, which do not divide its query width {width}"
            )
        size = width // heads
        if isinstance(node, RetrievalParameters):
            projections = {"key": node.key.weight.shape[0]}
            fits = {"key": projections["key"] == width}
        else:
            projections = {
                "key": node.key.weight.shape[0],
                "value": node.value.weight.shape[0],
            }
            # Key and value heads have the query heads' size, and query heads share
            # them in equal groups.
            fits = {
                kind: kv % size == 0 and kv >= size and heads % (kv // size) == 0
                for kind, kv in projections.items()
            }
        for kind, fit in fits.items():
            if not fit:
                raise ValueError(
                    f"{name} has {heads} heads of {size}, which do not fit its "
                    f"{kind} width {projections[kind]}"
                )


def load_state(
    directory: Path, architecture: str
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Reads a checkpoint directory.

    Args:
        directory: Directory with `config.json` and `model.safetensors`.
        architecture: The architecture name `config.json` must have.

    Returns:
        Arrays keyed by PyTorch parameter name, and the model's settings,
        which raise `ValueError` for a setting they lack.

    Raises:
        ValueError: If the checkpoint is of another architecture.
    """
    metadata = json.loads((directory / "config.json").read_text())
    if metadata["architecture_name"] != architecture:
        raise ValueError(
            f"not a {architecture} checkpoint: {metadata['architecture_name']!r}"
        )
    state = load_file(directory / "model.safetensors")
    arrays = {key: jnp.asarray(value) for key, value in state.items()}
    return arrays, Settings(metadata["config"])


def save_state(
    directory: Path,
    architecture: str,
    settings: Mapping[str, Any],
    state: Mapping[str, np.ndarray],
) -> None:
    """Writes a checkpoint directory, as from a PyTorch state dict converted to NumPy.

    Args:
        directory: Directory to write, made if missing.
        architecture: The architecture name.
        settings: The model's settings.
        state: Arrays keyed by PyTorch parameter name.
    """
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"architecture_name": architecture, "config": dict(settings)}
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    arrays = {key: np.ascontiguousarray(value) for key, value in state.items()}
    save_file(arrays, directory / "model.safetensors")


_STORAGE_DTYPES = {
    "torch.FloatStorage": np.dtype(np.float32),
    "torch.DoubleStorage": np.dtype(np.float64),
    "torch.HalfStorage": np.dtype(np.float16),
    "torch.BFloat16Storage": np.dtype(jnp.bfloat16),
    "torch.LongStorage": np.dtype(np.int64),
    "torch.IntStorage": np.dtype(np.int32),
    "torch.ShortStorage": np.dtype(np.int16),
    "torch.CharStorage": np.dtype(np.int8),
    "torch.ByteStorage": np.dtype(np.uint8),
    "torch.BoolStorage": np.dtype(np.bool_),
}


class _Storage(NamedTuple):
    dtype: np.dtype


def _rebuild_tensor(
    storage: np.ndarray,
    offset: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    *_: object,
) -> np.ndarray:
    # A damaged file could name a view outside its storage, which as_strided
    # would read. With no negative step, the first element is the lowest.
    if offset < 0 or min((*shape, *stride), default=0) < 0:
        raise ValueError("a tensor in the checkpoint has a negative size or step")
    last = offset + sum(
        (size - 1) * step for size, step in zip(shape, stride, strict=True)
    )
    if math.prod(shape) and max(offset, last) >= storage.size:
        raise ValueError("a tensor in the checkpoint reaches past its storage")
    strides = tuple(step * storage.itemsize for step in stride)
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape, strides)
    return np.ascontiguousarray(view)


class _WeightsUnpickler(pickle.Unpickler):
    """Rebuilds tensors and plain values, and refuses any other name."""

    def __init__(self, data: bytes, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(io.BytesIO(data))
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module: str, name: str) -> Any:
        qualified = f"{module}.{name}"
        if qualified == "torch._utils._rebuild_tensor_v2":
            return _rebuild_tensor
        if qualified == "collections.OrderedDict":
            return collections.OrderedDict
        if qualified in _STORAGE_DTYPES:
            return _Storage(_STORAGE_DTYPES[qualified])
        raise pickle.UnpicklingError(f"checkpoint refers to {qualified}, not a tensor")

    def persistent_load(self, pid: Any) -> np.ndarray:
        kind, storage, key, _location, count = pid
        if kind != "storage" or not isinstance(storage, _Storage):
            raise pickle.UnpicklingError(f"unexpected storage reference {pid!r}")
        data = self._archive.read(f"{self._prefix}/data/{key}")
        return np.frombuffer(data, dtype=storage.dtype.newbyteorder("<"), count=count)


def read_torch_checkpoint(path: str | Path) -> Any:
    """Reads a `torch.save` checkpoint without PyTorch.

    Only tensors and plain values are rebuilt, so no code from the file runs.

    Args:
        path: Path to the checkpoint.

    Returns:
        The stored object, with tensors as NumPy arrays.

    Raises:
        ValueError: If the file is not a little-endian PyTorch checkpoint.
        pickle.UnpicklingError: If it holds anything but tensors and plain values.
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickled = next((name for name in names if name.endswith("/data.pkl")), None)
        if pickled is None:
            raise ValueError(f"{path} is not a PyTorch checkpoint")
        prefix = pickled.removesuffix("/data.pkl")
        byteorder = f"{prefix}/byteorder"
        if byteorder in names and archive.read(byteorder) != b"little":
            raise ValueError("only little-endian checkpoints are supported")
        return _WeightsUnpickler(archive.read(pickled), archive, prefix).load()
