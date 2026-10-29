# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Finetuning a tabular model on a dataset of its own, with low-rank adapters.

`finetune` trains a LoRA adapter for the linear layers of a classifier or
regressor: each chosen weight `W` becomes `W + (alpha / rank) * B @ A`, with
`A` and `B` trained and `W` kept as the checkpoint has it. Training is
episodic, as the models were pretrained: each step draws a context and query
rows from the dataset, runs the contract calls the engine makes (see
`generalforecast.models.conformance.predict`) and scores the queries, by log
loss for a classifier and by pinball loss over its k/1000 quantiles for a
regressor.

An adapter merges into the weights at load, so a finetuned model runs through
the same contract calls and compiled programs as its base model:
`gf.Classifier(model=tuned.model)` takes it, and a custom model's
`module:attribute` can return `load(directory).model` for tables and workers.
An adapter directory holds `adapter.json`, naming the base model and its pinned
revision, and `adapter.safetensors`, with each layer's `A` and `B` under its
path in the parameters.
"""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from safetensors.numpy import load_file, save_file

from generalforecast.models.conformance import Table, predict
from generalforecast.models.contract import TabularFunctions
from generalforecast.models.outputs import REGRESSION_LEVELS, REGRESSION_STEP
from generalforecast.nn.linear import LinearParameters

FORMAT = 1
CONFIG = "adapter.json"
WEIGHTS = "adapter.safetensors"

Model = tuple[TabularFunctions, Any]


class LoRA(NamedTuple):
    """One linear layer's adapter: `delta = B @ A`, in the weight's (output, input) layout."""

    a: Array
    b: Array


@dataclass(frozen=True)
class Adapter:
    """Low-rank adapters for a model's linear layers, by their path in its parameters.

    Attributes:
        layers: Each adapted layer's `A` and `B`, keyed as
            `jax.tree_util.keystr` spells the layer's path, such as
            `.icl.blocks[0].attention.query`.
        rank: The adapters' rank.
        alpha: The scale numerator; each delta is scaled by `alpha / rank`.
        base: The canonical id of the catalog model it was trained on, or
            None for a model passed as its functions and parameters.
        revision: The base checkpoint's pinned revision, if it has one.
    """

    layers: dict[str, LoRA]
    rank: int
    alpha: float
    base: str | None = None
    revision: str | None = None

    @property
    def scale(self) -> float:
        """What each `B @ A` is multiplied by before it is added to its weight."""
        return self.alpha / self.rank

    def merge(self, parameters: Any) -> Any:
        """Parameters with each adapted weight replaced by `W + scale * B @ A`.

        The merged weights keep the base weights' dtype, so a merged model
        compiles as its base does.

        Raises:
            ValueError: If an adapted layer is not in the parameters, or its
                shape differs from the adapter's.
        """
        found = set()

        def merged(path: Any, node: Any) -> Any:
            if not isinstance(node, LinearParameters):
                return node
            name = jax.tree_util.keystr(path)
            lora = self.layers.get(name)
            if lora is None:
                return node
            found.add(name)
            shape = (lora.b.shape[0], lora.a.shape[1])
            if node.weight.shape != shape:
                raise ValueError(
                    f"the adapter's {name} is {shape}, the model's {node.weight.shape}"
                )
            delta = self.scale * (lora.b.astype(jnp.float32) @ lora.a.astype(jnp.float32))
            weight = (node.weight.astype(jnp.float32) + delta).astype(node.weight.dtype)
            return node._replace(weight=weight)

        result = jax.tree_util.tree_map_with_path(
            merged, parameters, is_leaf=_is_linear
        )
        missing = sorted(set(self.layers) - found)
        if missing:
            raise ValueError(
                f"the model has no layer {missing[0]} the adapter adapts"
                + (f", nor {len(missing) - 1} more" if len(missing) > 1 else "")
            )
        return result

    def save(self, directory: str | Path) -> Path:
        """Writes the adapter as `adapter.json` and `adapter.safetensors` in `directory`.

        Returns:
            The directory, made if missing.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "format": FORMAT,
            "base": self.base,
            "revision": self.revision,
            "rank": self.rank,
            "alpha": self.alpha,
            "layers": sorted(self.layers),
        }
        arrays = {}
        for name, lora in self.layers.items():
            arrays[f"{name}.a"] = np.asarray(lora.a, np.float32)
            arrays[f"{name}.b"] = np.asarray(lora.b, np.float32)
        save_file(arrays, directory / WEIGHTS)
        (directory / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
        return directory

    @classmethod
    def read(cls, directory: str | Path) -> "Adapter":
        """Reads an adapter directory `save` wrote.

        Raises:
            ValueError: If the directory holds another format, or a layer
                `adapter.json` names has no weights.
        """
        directory = Path(directory)
        config = json.loads((directory / CONFIG).read_text())
        if config.get("format") != FORMAT:
            raise ValueError(
                f"{directory} holds adapter format {config.get('format')!r}, "
                f"not {FORMAT}"
            )
        arrays = load_file(directory / WEIGHTS)
        layers = {}
        for name in config["layers"]:
            if f"{name}.a" not in arrays or f"{name}.b" not in arrays:
                raise ValueError(f"{directory} has no weights for {name}")
            layers[name] = LoRA(
                jnp.asarray(arrays[f"{name}.a"]), jnp.asarray(arrays[f"{name}.b"])
            )
        return cls(
            layers,
            int(config["rank"]),
            float(config["alpha"]),
            config.get("base"),
            config.get("revision"),
        )


@dataclass(frozen=True)
class Finetuned:
    """A model with its adapter merged in, as `finetune` and `load` return it.

    Attributes:
        model: The merged model's contract calls and parameters, which
            `gf.Classifier`, `gf.Regressor` and a custom model take.
        adapter: The adapter merged into it.
        losses: The training loss of each step, if it was trained here.
    """

    model: Model
    adapter: Adapter
    losses: tuple[float, ...] = field(default=())

    def save(self, directory: str | Path) -> Path:
        """Writes the adapter, as `Adapter.save` does; `load` reads it back."""
        return self.adapter.save(directory)


def _is_linear(node: Any) -> bool:
    return isinstance(node, LinearParameters)


def linear_layers(parameters: Any) -> dict[str, tuple[int, int]]:
    """Each linear layer's path in the parameters and its weight's (output, input) shape."""
    found, _ = jax.tree_util.tree_flatten_with_path(parameters, is_leaf=_is_linear)
    return {
        jax.tree_util.keystr(path): tuple(node.weight.shape)
        for path, node in found
        if isinstance(node, LinearParameters)
    }


def chosen_layers(
    parameters: Any, rank: int, layers: Sequence[str] | None = None
) -> dict[str, tuple[int, int]]:
    """The linear layers an adapter of `rank` adapts.

    Args:
        parameters: The model's parameters.
        rank: The adapters' rank. A layer narrower than it gains nothing
            from a low-rank delta, so it is left as it is.
        layers: Substrings of the paths to adapt, such as `.icl.`; every
            linear layer wide enough if None.

    Raises:
        ValueError: If no layer is chosen.
    """
    chosen = {
        name: shape
        for name, shape in linear_layers(parameters).items()
        if min(shape) > rank
        and (layers is None or any(part in name for part in layers))
    }
    if not chosen:
        raise ValueError(
            f"no linear layer is wider than rank {rank}"
            + ("" if layers is None else f" among those matching {list(layers)}")
        )
    return chosen


def initial_adapter(
    shapes: dict[str, tuple[int, int]], rank: int, seed: int
) -> dict[str, LoRA]:
    """LoRA's initialization: `A` random, `B` zero, so the model starts as its base."""
    keys = jax.random.split(jax.random.key(seed), len(shapes))
    return {
        name: LoRA(
            jax.random.normal(key, (rank, inputs), jnp.float32) / math.sqrt(inputs),
            jnp.zeros((outputs, rank), jnp.float32),
        )
        for key, (name, (outputs, inputs)) in zip(keys, shapes.items(), strict=True)
    }


def episode_loss(
    functions: TabularFunctions,
    parameters: Any,
    episode: Table,
    query_targets: Array,
    regression: bool,
) -> Array:
    """The mean loss of an episode's query rows, from the engine's own contract calls.

    Args:
        functions: The model's contract calls.
        parameters: Its parameters, adapters merged.
        episode: Context rows, their targets and the query rows.
        query_targets: The query rows' classes, or their values scaled to
            unit spread.
        regression: Whether to score quantiles rather than probabilities.

    Returns:
        Log loss for a classifier; for a regressor, pinball loss averaged
        over its k/1000 quantiles, in units of the targets' spread.
    """
    predictions = predict(functions, parameters, episode)
    if regression:
        levels = REGRESSION_STEP * jnp.arange(1, REGRESSION_LEVELS + 1)
        errors = query_targets[:, None] - predictions
        return jnp.mean(jnp.maximum(levels * errors, (levels - 1) * errors))
    chosen = jnp.take_along_axis(predictions, query_targets[:, None], axis=1)
    return -jnp.mean(jnp.log(jnp.clip(chosen, 1e-7, 1.0)))


class _Adam(NamedTuple):
    step: Array
    first: Any
    second: Any


def _adamw(
    learning_rate: float, weight_decay: float, b1: float = 0.9, b2: float = 0.999
) -> tuple[Callable[[Any], _Adam], Callable[[Any, Any, _Adam], tuple[Any, _Adam]]]:
    """AdamW's start and update, over any tree of float32 arrays."""

    def start(values: Any) -> _Adam:
        zeros = jax.tree.map(jnp.zeros_like, values)
        return _Adam(jnp.zeros((), jnp.int32), zeros, zeros)

    def update(values: Any, gradients: Any, state: _Adam) -> tuple[Any, _Adam]:
        step = state.step + 1
        first = jax.tree.map(
            lambda m, g: b1 * m + (1 - b1) * g, state.first, gradients
        )
        second = jax.tree.map(
            lambda v, g: b2 * v + (1 - b2) * g * g, state.second, gradients
        )
        correction = jnp.sqrt(1 - b2**step) / (1 - b1**step)

        def moved(value: Array, m: Array, v: Array) -> Array:
            direction = correction * m / (jnp.sqrt(v) + 1e-8)
            return value - learning_rate * (direction + weight_decay * value)

        return jax.tree.map(moved, values, first, second), _Adam(step, first, second)

    return start, update


def _base(model: str | Model) -> tuple[Model, str | None, str | None]:
    """A model to finetune, with its canonical id and pinned revision if it is a catalog model."""
    if not isinstance(model, str):
        functions, parameters = model
        if not isinstance(functions, TabularFunctions):
            raise ValueError("finetuning takes a classifier or regressor, not a forecaster")
        return (functions, parameters), None, None
    from generalforecast.models import catalog, registry

    ref = catalog.parse(model)
    if ref.custom or ref.variant:
        raise ValueError(
            f"finetuning takes a catalog model's id with no codes or estimators: {model}"
        )
    loaded = registry.load_tabular(ref.name)
    return loaded, ref.name, ref.entry.artifact.revision


def finetune(
    model: str | Model,
    X: Any,
    y: Any,
    *,
    steps: int = 200,
    rank: int = 8,
    alpha: float | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    context_rows: int | None = None,
    query_rows: int | None = None,
    layers: Sequence[str] | None = None,
    seed: int = 0,
    progress: Callable[[int, float], None] | None = None,
) -> Finetuned:
    """Trains a LoRA adapter on a labeled dataset and returns the model with it merged.

    Each step draws `context_rows` and `query_rows` apart from the dataset at
    random and scores the queries in context of the rest, so the model learns
    the dataset's structure while still reading its context at predict time.
    Fit the finetuned model on the context it will serve, as its base.

    Args:
        model: A classifier's or regressor's canonical catalog id, such as
            `general-forecast/tabicl-v2-classifier`, or its contract calls
            and parameters.
        X: Numeric rows, `(rows, features)`, with NaN for missing values.
        y: A label per row for a classifier, which keeps them in sorted order
            as `gf.Classifier`'s `classes_` does; a value per row for a
            regressor.
        steps: Optimizer steps.
        rank: The adapters' rank.
        alpha: The scale numerator, `2 * rank` if None.
        learning_rate: AdamW's learning rate.
        weight_decay: AdamW's decoupled weight decay, on the adapters alone.
        context_rows: Context rows per episode; the rows past `query_rows`,
            at most 1024, if None.
        query_rows: Query rows per episode; a quarter of the rows, at most
            256, if None.
        layers: Substrings of the paths of the linear layers to adapt, as
            `linear_layers` lists them; every layer wider than `rank` if None.
        seed: Seed for the adapters and the episodes.
        progress: Called after each step with its index and loss.

    Returns:
        The finetuned model, its adapter and each step's loss.

    Raises:
        ValueError: If the dataset is too small for an episode, the model is
            not a plain classifier or regressor, or no layer is chosen.
    """
    (functions, parameters), base, revision = _base(model)
    regression = functions.properties(parameters).regression
    features = np.asarray(X, np.float32)
    targets = np.asarray(y)
    if features.ndim != 2 or targets.shape != (features.shape[0],):
        raise ValueError(
            f"expected rows (rows, features) and a target per row, got "
            f"{features.shape} and {targets.shape}"
        )
    total = features.shape[0]
    if regression:
        values = targets.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("finetuning needs a finite target in every row")
        spread = float(np.std(values)) or 1.0
        targets, num_classes = values, 0
    else:
        classes, codes = np.unique(targets, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("finetuning a classifier needs at least two classes")
        spread, targets, num_classes = 1.0, codes.astype(np.int32), len(classes)
    queries = query_rows or max(1, min(256, total // 4))
    context = context_rows or min(1024, total - queries)
    if context < 2 or queries < 1 or context + queries > total:
        raise ValueError(
            f"an episode of {context} context rows and {queries} queries needs "
            f"more than the {total} rows given"
        )

    shapes = chosen_layers(parameters, rank, layers)
    alpha = 2.0 * rank if alpha is None else alpha
    adapters = initial_adapter(shapes, rank, seed)
    start, update = _adamw(learning_rate, weight_decay)

    # A regressor is scored in units of the targets' spread, so one learning
    # rate suits targets of any scale.
    scored = _scaled(functions, spread) if regression else functions

    def loss(
        adapters: dict[str, LoRA], parameters: Any, rows: Array, labels: Array
    ) -> Array:
        merged = Adapter(adapters, rank, alpha).merge(parameters)
        episode = Table(rows[:context], labels[:context], rows[context:], num_classes)
        queried = labels[context:] / spread if regression else labels[context:]
        return episode_loss(scored, merged, episode, queried, regression)

    @jax.jit
    def step(
        adapters: dict[str, LoRA], state: _Adam, parameters: Any, rows: Array, labels: Array
    ) -> tuple[dict[str, LoRA], _Adam, Array]:
        value, gradients = jax.value_and_grad(loss)(adapters, parameters, rows, labels)
        adapters, state = update(adapters, gradients, state)
        return adapters, state, value

    generator = np.random.default_rng(seed)
    state = start(adapters)
    losses = []
    for index in range(steps):
        chosen = generator.permutation(total)[: context + queries]
        adapters, state, value = step(
            adapters,
            state,
            parameters,
            jnp.asarray(features[chosen]),
            jnp.asarray(targets[chosen]),
        )
        losses.append(float(value))
        if progress is not None:
            progress(index, losses[-1])
    adapter = Adapter(adapters, rank, alpha, base, revision)
    return Finetuned(
        (functions, adapter.merge(parameters)), adapter, tuple(losses)
    )


def _scaled(functions: TabularFunctions, spread: float) -> TabularFunctions:
    """A regressor's calls with predictions divided by `spread`, as its scaled targets are."""
    postprocess = functions.postprocess

    def scaled(model: Any, state: Any, outputs: Array) -> Array:
        return postprocess(model, state, outputs) / spread

    return functions._replace(postprocess=scaled)


def load(directory: str | Path, model: Model | None = None) -> Finetuned:
    """Loads an adapter directory and merges it into its base model.

    Args:
        directory: A directory `Finetuned.save` wrote.
        model: The base model's contract calls and parameters, for an adapter
            trained on a model passed that way; the catalog model the adapter
            names if None.

    Returns:
        The finetuned model and its adapter.

    Raises:
        ValueError: If the adapter names no catalog model and none is given,
            its base model's pinned revision differs from the one it was
            trained on, or its layers do not fit the model.
    """
    adapter = Adapter.read(directory)
    if model is None:
        if adapter.base is None:
            raise ValueError(
                f"the adapter in {directory} names no catalog model; pass its "
                "base model to load"
            )
        model, _, revision = _base(adapter.base)
        if adapter.revision is not None and revision != adapter.revision:
            raise ValueError(
                f"the adapter in {directory} was trained on {adapter.base} at "
                f"revision {adapter.revision}, which is now pinned at {revision}"
            )
    functions, parameters = model
    return Finetuned((functions, adapter.merge(parameters)), adapter)
