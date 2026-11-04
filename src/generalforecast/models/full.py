# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full fine-tuning: the named components a model trains, and how gf serves them.

Full fine-tuning trains some of a model's weights outright, by component:

- `column_encoder`: what embeds each cell and relates it to its column's
  other cells across rows;
- `row_encoder`: what turns a row's cell embeddings into one row
  representation;
- `in_context_transformer`: what embeds the context's targets and attends
  from every row to the context;
- `prediction_head`: what turns a query's representation into class logits
  or quantiles.

`MAPS` says, per model family, which parameter paths make up each
component, as prefixes of `jax.tree_util.keystr` paths. Every
floating-point leaf of a mapped family is in exactly one component or in
its `FIXED` entry, the values the model was built with that are not
weights; a trainer trains only the leaves a component lists, never every
floating-point value, and leaves everything else as it was loaded.

Stored, a full fine-tune is one object in format `full/1`: an `.npz` read
without pickles, holding a JSON `meta` (format, bags, the trained
components and the paths of their leaves) and, for the i-th path, `d{i}`:
the trained value less the base's, in float32, stacked over bags. A delta
of an unchanged leaf is zeros, which the archive's deflate stores in next
to nothing. gf serves it as a derived model of kind `full`: `unpack` adds
each delta to the base once, and `combine` puts the results in place of
the base's leaves, so every full fine-tune of a base shares its compiled
programs and its copy of the leaves no component trains.
"""

import io
import json
import zipfile
from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.models.catalog import ModelRefError
from generalforecast.models.derived import Weights

FORMAT = "full/1"
COMPONENTS = (
    "column_encoder",
    "row_encoder",
    "in_context_transformer",
    "prediction_head",
)

# TabICL and TabPFN: a column stage, a row stage and an ICL stage whose
# decoder (and TabICL's optional head) predicts.
_STAGED = {
    "column_encoder": (".columns",),
    "row_encoder": (".rows",),
    "in_context_transformer": (
        ".icl.blocks",
        ".icl.target_encoder",
        ".icl.output_norm",
    ),
    "prediction_head": (".icl.decoder", ".icl.head"),
}
# Each component's parameter paths, by model family: the parameters' type
# as `module.qualname`. A family missing here cannot be fully fine-tuned.
MAPS: dict[str, dict[str, tuple[str, ...]]] = {
    "generalforecast.models.tabicl.model.TabICL": _STAGED,
    "generalforecast.models.tabpfn.model.TabPFN": _STAGED,
    # Kumo Tabular's row embedding interleaves attention across rows within
    # each column (its induced blocks) and across a row's cells (its row
    # blocks): the cell embedding, context labels and column blocks embed
    # cells, and the readout tokens, row blocks and the projection to the
    # ICL width make rows.
    "generalforecast.models.kumo.model.KumoTabular": {
        "column_encoder": (
            ".rows.cells",
            ".rows.target_encoder",
            ".rows.columns",
        ),
        "row_encoder": (
            ".rows.readout",
            ".rows.rows",
            ".rows.output_norm",
            ".project",
        ),
        "in_context_transformer": (
            ".icl.blocks",
            ".icl.target_encoder",
            ".icl.output_norm",
        ),
        "prediction_head": (".icl.decoder", ".icl.head"),
    },
    # Mitra embeds each cell with one projection, then each layer attends
    # across rows (in context, with the target tokens) and within each row.
    "generalforecast.models.mitra.model.Mitra": {
        "column_encoder": (".cells",),
        "row_encoder": (".layers[*].within_row",),
        "in_context_transformer": (".targets", ".layers[*].across_rows"),
        "prediction_head": (".final_norm", ".decoder"),
    },
}
# Floating-point leaves a family holds that are not weights: fixed values
# training must leave as loaded. Kumo Tabular's Fourier frequencies are
# the sampled bands of its cell embedding, kept as the checkpoint has them.
FIXED: dict[str, tuple[str, ...]] = {
    "generalforecast.models.kumo.model.KumoTabular": (
        ".rows.cells.frequencies",
        ".rows.cells.categorical_frequencies",
    ),
}


def family(parameters: Any) -> str:
    """The parameters' model family, as `MAPS` names it."""
    kind = type(parameters)
    return f"{kind.__module__}.{kind.__qualname__}"


def _matches(path: str, prefix: str) -> bool:
    """Whether a leaf's path is under `prefix`, whose `[*]` matches any index."""
    pattern = prefix.split("[*]")
    if len(pattern) == 1:
        return path == prefix or path.startswith((prefix + ".", prefix + "["))
    head, rest = pattern[0], "[*]".join(pattern[1:])
    if not path.startswith(head + "["):
        return False
    close = path.find("]", len(head))
    if close < 0:
        return False
    return _matches(path[close + 1 :], rest) if rest else True


def _floating(leaf: Any) -> bool:
    return hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)


def _leaves(parameters: Any) -> list[tuple[str, Any]]:
    found, _ = jax.tree_util.tree_flatten_with_path(parameters)
    return [(jax.tree_util.keystr(path), leaf) for path, leaf in found]


def component_of(parameters: Any) -> dict[str, str | None]:
    """Each floating-point leaf's component, by path; None for a fixed value.

    A fixed value is one whatever component its path is under.

    Raises:
        ValueError: For a family `MAPS` does not map, or a leaf in no
            component and not fixed, or in more than one.
    """
    name = family(parameters)
    if name not in MAPS:
        raise ValueError(f"{name} has no component map for full fine-tuning")
    components, fixed = MAPS[name], FIXED.get(name, ())
    found: dict[str, str | None] = {}
    for path, leaf in _leaves(parameters):
        if not _floating(leaf):
            continue
        if any(_matches(path, prefix) for prefix in fixed):
            found[path] = None
            continue
        owners = [
            component
            for component, prefixes in components.items()
            if any(_matches(path, prefix) for prefix in prefixes)
        ]
        if len(owners) != 1:
            raise ValueError(
                f"{name}'s leaf {path} is in {len(owners)} components, not one"
            )
        found[path] = owners[0]
    return found


def trainable(parameters: Any, components: Iterable[str]) -> dict[str, Any]:
    """The leaves full fine-tuning trains for `components`, by path.

    Raises:
        ValueError: For a component gf does not name, or parameters
            `component_of` refuses.
    """
    wanted = set(components)
    unknown = sorted(wanted - set(COMPONENTS))
    if unknown:
        raise ValueError(f"no component {', '.join(unknown)}")
    owners = component_of(parameters)
    return {
        path: leaf for path, leaf in _leaves(parameters) if owners.get(path) in wanted
    }


def combine(shared: Any, own: Mapping[str, Any]) -> Any:
    """The base's parameters with the leaves `own` holds, by path, in place."""

    def placed(path: Any, leaf: Any) -> Any:
        value = own.get(jax.tree_util.keystr(path))
        return leaf if value is None else value.astype(leaf.dtype)

    return jax.tree_util.tree_map_with_path(placed, shared)


def pack(
    base: Any, bags: list[Mapping[str, Any]], *, components: Iterable[str]
) -> bytes:
    """Stores bags of trained leaves over `base` as one `full/1` object.

    Each bag maps the same paths, the leaves of `components`, to their
    trained values; they are stored as their difference from the base's.

    Raises:
        ValueError: For no bags, bags over other paths than the components'
            leaves, or values whose shapes are not the base's.
    """
    if not bags:
        raise ValueError("a fine-tune has at least one bag")
    components = sorted(set(components))
    leaves = trainable(base, components)
    paths = sorted(leaves)
    if any(sorted(bag) != paths for bag in bags):
        raise ValueError(f"every bag holds the leaves of {', '.join(components)}")
    arrays: dict[str, np.ndarray] = {}
    for index, path in enumerate(paths):
        start = np.asarray(leaves[path], np.float32)
        deltas = []
        for bag in bags:
            value = np.asarray(bag[path], np.float32)
            if value.shape != start.shape:
                raise ValueError(f"{path} is {value.shape}, not {start.shape}")
            deltas.append(value - start)
        arrays[f"d{index}"] = np.stack(deltas)
    meta = {
        "format": FORMAT,
        "bags": len(bags),
        "components": components,
        "paths": paths,
    }
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    sink = io.BytesIO()
    # The fastest deflate: trained deltas are mostly noise, which no level
    # shrinks more than a few percent further, while zero deltas shrink
    # alike at any level. Higher levels took 4.5 times as long.
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for key, array in arrays.items():
            with archive.open(f"{key}.npy", "w") as member:
                np.lib.format.write_array(member, array, allow_pickle=False)
    return sink.getvalue()


def _refused(message: str) -> ModelRefError:
    return ModelRefError("ModelMismatch", message)


def unpack(body: bytes, base: Any, bags: int) -> Weights:
    """A stored `full/1` object as gf runs it over `base`: leaves stacked by bag.

    Raises:
        ModelRefError: `ModelMismatch` for another format or bag count, or
            leaves the base's components do not hold or whose shapes differ.
    """
    try:
        stored = np.load(io.BytesIO(body), allow_pickle=False)
        meta = json.loads(bytes(stored["meta"]).decode())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        raise _refused("the fine-tune's weights are not a full/1 object") from None
    if meta.get("format") != FORMAT:
        raise _refused(f"weights in format {meta.get('format')!r}, not {FORMAT}")
    if meta.get("bags") != bags:
        raise _refused(f"weights stack {meta.get('bags')} bags, not {bags}")
    try:
        leaves = trainable(base, meta["components"])
    except ValueError as refused:
        raise _refused(str(refused)) from None
    if sorted(meta["paths"]) != sorted(leaves):
        raise _refused("the weights do not hold the leaves of their components")
    own: dict[str, Any] = {}
    for index, path in enumerate(meta["paths"]):
        leaf = leaves[path]
        delta = stored[f"d{index}"]
        if delta.shape != (bags, *leaf.shape):
            raise _refused(f"the weights of {path} do not fit the base's")
        value = jnp.asarray(leaf, jnp.float32)[None] + jnp.asarray(delta)
        own[path] = value.astype(leaf.dtype)
    return Weights(own, combine)
