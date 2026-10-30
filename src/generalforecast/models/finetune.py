# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Finetuning a classifier or regressor on a labeled table: full finetuning, LoRA, or a mix.

What trains is a tuple of parameter groups, as PyTorch's `param_groups` and
PEFT's `LoraConfig` divide a model. The first group whose `match` prefixes a
parameter's path (as `jax.tree_util.keystr` spells it, such as
`.icl.blocks[0].attention.query.weight`) takes it, and its `mode` says how:

- `full`: the parameter itself trains.
- `lora`: each linear layer the group matches keeps its weight as the base has
  it, and a low-rank adapter beside it trains: `W + (alpha / rank) * B @ A`,
  `A` random and `B` zero, so training starts at the base model. A linear
  layer no wider than the rank both ways gains nothing from one and stays as
  it is.
- `frozen`: nothing trains.

A parameter no group matches is frozen. `full()` and `lora()` build the usual
groups: everything in full, or LoRA on the linear layers with the decoder head
optionally in full (PEFT's `modules_to_save`).

Training is episodic, as the models were pretrained: each step draws context
and query rows from the table on the device, runs the contract calls the
engine makes (`generalforecast.models.conformance.predict`) and scores the
queries, by log loss for a classifier and by pinball loss over its k/1000
quantiles for a regressor. AdamW with linear warmup, cosine decay to a tenth
of the peak, and gradient clipping; each group scales the learning rate and
sets its own weight decay. `steps_per_call` steps run as one compiled program.
A step whose loss is not finite ends training at the weights before it.

A finetuned model's changes (`Delta`: the trained parameters of `full` groups
and the adapters of `lora` groups) merge into the base weights at load, so it
runs through the same contract calls and compiled programs as its base:
`gf.Classifier(model=tuned.model)` takes it. A delta directory holds
`finetune.json`, naming the base model, its pinned revision and the groups,
and `delta.safetensors`.
"""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple

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
CONFIG = "finetune.json"
WEIGHTS = "delta.safetensors"

Model = tuple[TabularFunctions, Any]
Mode = Literal["full", "lora", "frozen"]


@dataclass(frozen=True)
class Group:
    """Parameters whose path starts with `match`, and how they train.

    Attributes:
        match: A path prefix; `""` matches every parameter.
        mode: `full`, `lora` or `frozen`.
        rank: The adapters' rank, for `lora`.
        alpha: The adapters' scale numerator, for `lora`; each delta is
            `alpha / rank` times `B @ A`.
        learning_rate_scale: What the step's learning rate is multiplied by.
        weight_decay: AdamW's decoupled weight decay: toward zero for a `full`
            parameter, toward a zero delta (the base model) for an adapter.
    """

    match: str = ""
    mode: Mode = "full"
    rank: int = 16
    alpha: float = 32.0
    learning_rate_scale: float = 1.0
    weight_decay: float = 0.0


def full(**options: Any) -> tuple[Group, ...]:
    """Every parameter in full, with `Group`'s other options."""
    return (Group("", "full", **options),)


def lora(
    rank: int = 16,
    alpha: float | None = None,
    layers: Sequence[str] = ("",),
    head: str | None = None,
    **options: Any,
) -> tuple[Group, ...]:
    """LoRA on the linear layers under `layers`, and the parameters under `head` in full.

    Args:
        rank: The adapters' rank.
        alpha: The scale numerator, `2 * rank` if None.
        layers: Path prefixes whose linear layers take adapters.
        head: A path prefix trained in full, such as `.icl.decoder.`; none if None.
        options: `Group`'s other options, for every group.
    """
    alpha = 2.0 * rank if alpha is None else alpha
    groups = [] if head is None else [Group(head, "full", **options)]
    groups += [Group(prefix, "lora", rank, alpha, **options) for prefix in layers]
    return (*groups, Group("", "frozen"))


def _is_linear(node: Any) -> bool:
    return isinstance(node, LinearParameters)


def _floating(leaf: Any) -> bool:
    return isinstance(leaf, jax.Array | np.ndarray) and jnp.issubdtype(
        leaf.dtype, jnp.floating
    )


def _group(path: str, groups: Sequence[Group]) -> Group | None:
    return next((group for group in groups if path.startswith(group.match)), None)


class Plan(NamedTuple):
    """Which parameters train, by path: `full` leaves and `lora` layers, each with its group."""

    full: dict[str, Group]
    lora: dict[str, Group]


def plan(parameters: Any, groups: Sequence[Group]) -> Plan:
    """The parameters `groups` train in the model's parameters.

    Raises:
        ValueError: If nothing trains, or a `lora` group matches no linear
            layer wider than its rank.
    """
    layers, _ = jax.tree_util.tree_flatten_with_path(parameters, is_leaf=_is_linear)
    adapted: dict[str, Group] = {}
    for path, node in layers:
        if not isinstance(node, LinearParameters):
            continue
        name = jax.tree_util.keystr(path)
        group = _group(f"{name}.weight", groups)
        if (
            group is not None
            and group.mode == "lora"
            and min(node.weight.shape) > group.rank
        ):
            adapted[name] = group
    leaves, _ = jax.tree_util.tree_flatten_with_path(parameters)
    trained = {}
    for path, leaf in leaves:
        name = jax.tree_util.keystr(path)
        group = _group(name, groups)
        if group is not None and group.mode == "full" and _floating(leaf):
            trained[name] = group
    for group in groups:
        if group.mode == "lora" and group not in adapted.values():
            raise ValueError(
                f"the lora group matching {group.match!r} adapts no linear layer "
                f"wider than rank {group.rank}"
            )
    if not trained and not adapted:
        raise ValueError(
            "no parameter trains: every group is frozen or matches nothing"
        )
    return Plan(trained, adapted)


@dataclass(frozen=True)
class Delta:
    """What finetuning changed: trained `full` parameters and `lora` adapters, by path.

    Attributes:
        full: Each trained parameter's new value.
        lora: Each adapted layer's `(A, B)`, in its weight's (output, input) layout.
        groups: The groups it was trained with.
        base: The canonical id of the catalog model it was trained on, or None
            for a model passed as its functions and parameters.
        revision: The base checkpoint's pinned revision, if it has one.
    """

    full: dict[str, Array]
    lora: dict[str, tuple[Array, Array]]
    groups: tuple[Group, ...]
    base: str | None = None
    revision: str | None = None

    def _scale(self, name: str) -> float:
        group = _group(f"{name}.weight", self.groups)
        assert group is not None and group.mode == "lora"
        return group.alpha / group.rank

    def merge(self, parameters: Any) -> Any:
        """The parameters with the delta applied, each in the base's dtype.

        Raises:
            ValueError: If a changed parameter is not in the parameters, or
                its shape differs from the delta's.
        """
        found: set[str] = set()

        def replaced(path: Any, leaf: Any) -> Any:
            name = jax.tree_util.keystr(path)
            value = self.full.get(name)
            if value is None:
                return leaf
            if value.shape != leaf.shape:
                raise ValueError(
                    f"the delta's {name} is {value.shape}, the model's {leaf.shape}"
                )
            found.add(name)
            return value.astype(leaf.dtype)

        def adapted(path: Any, node: Any) -> Any:
            if not isinstance(node, LinearParameters):
                return node
            name = jax.tree_util.keystr(path)
            if name not in self.lora:
                return node
            a, b = self.lora[name]
            shape = (b.shape[0], a.shape[1])
            if node.weight.shape != shape:
                raise ValueError(
                    f"the delta's {name} adapter is {shape}, the model's {node.weight.shape}"
                )
            found.add(name)
            delta = self._scale(name) * (b.astype(jnp.float32) @ a.astype(jnp.float32))
            weight = node.weight.astype(jnp.float32) + delta
            return node._replace(weight=weight.astype(node.weight.dtype))

        result = jax.tree_util.tree_map_with_path(replaced, parameters)
        result = jax.tree_util.tree_map_with_path(adapted, result, is_leaf=_is_linear)
        missing = sorted((set(self.full) | set(self.lora)) - found)
        if missing:
            raise ValueError(
                f"the model has no {missing[0]}, which the delta changes"
                + (f", nor {len(missing) - 1} more" if len(missing) > 1 else "")
            )
        return result

    def save(self, directory: str | Path) -> Path:
        """Writes `finetune.json` and `delta.safetensors` in `directory`, made if missing."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {f"full{name}": np.asarray(value) for name, value in self.full.items()}
        for name, (a, b) in self.lora.items():
            arrays[f"lora{name}.a"] = np.asarray(a, np.float32)
            arrays[f"lora{name}.b"] = np.asarray(b, np.float32)
        save_file(arrays, directory / WEIGHTS)
        config = {
            "format": FORMAT,
            "base": self.base,
            "revision": self.revision,
            "groups": [asdict(group) for group in self.groups],
            "full": sorted(self.full),
            "lora": sorted(self.lora),
        }
        (directory / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
        return directory

    @classmethod
    def read(cls, directory: str | Path) -> "Delta":
        """Reads a directory `save` wrote.

        Raises:
            ValueError: If it holds another format, or lacks a changed parameter's arrays.
        """
        directory = Path(directory)
        config = json.loads((directory / CONFIG).read_text())
        if config.get("format") != FORMAT:
            raise ValueError(
                f"{directory} holds finetune format {config.get('format')!r}, not {FORMAT}"
            )
        arrays = load_file(directory / WEIGHTS)

        def array(key: str) -> Array:
            if key not in arrays:
                raise ValueError(f"{directory} has no array {key}")
            return jnp.asarray(arrays[key])

        return cls(
            {name: array(f"full{name}") for name in config["full"]},
            {
                name: (array(f"lora{name}.a"), array(f"lora{name}.b"))
                for name in config["lora"]
            },
            tuple(Group(**group) for group in config["groups"]),
            config.get("base"),
            config.get("revision"),
        )


@dataclass(frozen=True)
class Finetuned:
    """A model with a delta merged in, as `finetune` and `load` return it.

    Attributes:
        model: The merged model's contract calls and parameters, which
            `gf.Classifier`, `gf.Regressor` and a custom model take.
        delta: What finetuning changed.
        losses: Each step's training loss, if it was trained here.
        stopped: Why training ended before its steps, if it did.
    """

    model: Model
    delta: Delta
    losses: tuple[float, ...] = field(default=())
    stopped: str | None = None

    def save(self, directory: str | Path) -> Path:
        """Writes the delta, as `Delta.save`; `load` reads it back."""
        return self.delta.save(directory)


def episode_loss(
    functions: TabularFunctions,
    parameters: Any,
    episode: Table,
    query_targets: Array,
    spread: float,
) -> Array:
    """An episode's mean query loss, through the contract calls the engine makes.

    Log loss for a classifier; for a regressor, pinball loss averaged over its
    k/1000 quantiles, divided by `spread`, the training targets' spread.
    """
    predictions = predict(functions, parameters, episode)
    if episode.num_classes == 0:
        levels = REGRESSION_STEP * jnp.arange(1, REGRESSION_LEVELS + 1)
        errors = (query_targets[:, None] - predictions) / spread
        return jnp.mean(jnp.maximum(levels * errors, (levels - 1) * errors))
    chosen = jnp.take_along_axis(predictions, query_targets[:, None], axis=1)
    return -jnp.mean(jnp.log(jnp.clip(chosen, 1e-7, 1.0)))


class _Adam(NamedTuple):
    first: Any
    second: Any


def _schedule(step: Array, peak: float, warmup: int, total: int) -> Array:
    """Linear warmup to `peak`, then cosine decay to a tenth of it at `total`."""
    rising = peak * (step + 1) / max(warmup, 1)
    progress = jnp.clip((step - warmup) / max(total - warmup, 1), 0.0, 1.0)
    falling = peak * (0.1 + 0.45 * (1 + jnp.cos(jnp.pi * progress)))
    return jnp.where(step < warmup, rising, falling)


def _base(model: str | Model) -> tuple[Model, str | None, str | None]:
    """A model to finetune, with its canonical id and pinned revision if it is a catalog model."""
    if not isinstance(model, str):
        functions, parameters = model
        if not isinstance(functions, TabularFunctions):
            raise ValueError(  # noqa: TRY004
                "finetuning takes a classifier or regressor, not a forecaster"
            )
        return (functions, parameters), None, None
    from generalforecast.models import catalog, registry

    ref = catalog.parse(model)
    if ref.custom or ref.variant:
        raise ValueError(
            f"finetuning takes a catalog model's id with no codes or estimators: {model}"
        )
    return registry.load_tabular(ref.name), ref.name, ref.entry.artifact.revision


def finetune(
    model: str | Model,
    X: Any,
    y: Any,
    *,
    trainable: Sequence[Group] = full(),
    steps: int = 300,
    learning_rate: float = 1e-5,
    warmup: int = 20,
    clip_norm: float = 1.0,
    context_rows: int | None = None,
    query_rows: int | None = None,
    episodes_per_step: int = 1,
    steps_per_call: int = 25,
    seed: int = 0,
    progress: Callable[[int, float], None] | None = None,
) -> Finetuned:
    """Finetunes a model on a labeled table and returns it with the delta merged.

    Each step draws `context_rows` and `query_rows` apart from the table at
    random and scores the queries in context of the rest, so the weights learn
    the table while the model still reads its context at predict time. Fit the
    finetuned model on the context it will serve, as its base.

    Args:
        model: A classifier's or regressor's canonical catalog id, such as
            `general-forecast/tabicl-v2-classifier`, or its contract calls and
            parameters.
        X: Numeric rows, `(rows, features)`, with NaN for missing values.
        y: A label per row for a classifier, kept in sorted order as
            `gf.Classifier`'s `classes_` keeps them; a value per row for a
            regressor.
        trainable: Parameter groups: `full()` (the default), `lora()`, or groups
            of one's own.
        steps: Optimizer steps.
        learning_rate: The peak learning rate, which groups scale.
        warmup: Steps of linear warmup before cosine decay.
        clip_norm: The global gradient norm above which gradients are scaled down.
        context_rows: Context rows per episode; the rows past `query_rows`, at
            most 1024, if None.
        query_rows: Query rows per episode; a quarter of the rows, at most 256,
            if None.
        episodes_per_step: Episodes averaged per step.
        steps_per_call: Steps compiled into one program and run per call.
        seed: Seeds the adapters and the episodes.
        progress: Called after each step with its index and loss.

    Returns:
        The finetuned model, its delta, each step's loss, and why training
        stopped early if it did.

    Raises:
        ValueError: If the table is too small for an episode, the model is not
            a plain classifier or regressor, or the groups train nothing.
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

    groups = tuple(trainable)
    chosen = plan(parameters, groups)
    leaves = {
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(parameters)[0]
    }
    nodes = {
        jax.tree_util.keystr(path): node
        for path, node in jax.tree_util.tree_flatten_with_path(
            parameters, is_leaf=_is_linear
        )[0]
        if isinstance(node, LinearParameters)
    }
    keys = iter(jax.random.split(jax.random.key(seed), len(chosen.lora) + 1))
    trained = {
        "full": {
            name: jnp.array(leaves[name], jnp.float32, copy=True)
            for name in chosen.full
        },
        "lora": {},
    }
    for name, group in chosen.lora.items():
        outputs, inputs = nodes[name].weight.shape
        trained["lora"][name] = (
            jax.random.normal(next(keys), (group.rank, inputs), jnp.float32)
            / math.sqrt(inputs),
            jnp.zeros((outputs, group.rank), jnp.float32),
        )
    episode_key = next(keys)
    rate_scale = {
        "full": {name: g.learning_rate_scale for name, g in chosen.full.items()},
        "lora": {name: (g.learning_rate_scale,) * 2 for name, g in chosen.lora.items()},
    }
    decay = {
        "full": {name: g.weight_decay for name, g in chosen.full.items()},
        "lora": {name: (g.weight_decay,) * 2 for name, g in chosen.lora.items()},
    }

    def delta(values: Any) -> Delta:
        return Delta(values["full"], values["lora"], groups, base, revision)

    data_features = jnp.asarray(features)
    data_targets = jnp.asarray(targets)
    size = context + queries

    def loss(values: Any, key: Array) -> Array:
        merged = delta(values).merge(parameters)

        def one(key: Array) -> Array:
            rows = jax.random.choice(key, total, (size,), replace=False)
            rows_x, rows_y = data_features[rows], data_targets[rows]
            episode = Table(
                rows_x[:context], rows_y[:context], rows_x[context:], num_classes
            )
            return episode_loss(functions, merged, episode, rows_y[context:], spread)

        return jnp.mean(jax.lax.map(one, jax.random.split(key, episodes_per_step)))

    def steps_program(values: Any, state: _Adam, start: Array, keys: Array) -> tuple:
        def step(carry: tuple, inputs: tuple) -> tuple:
            values, state = carry
            index, key = inputs
            value, gradients = jax.value_and_grad(loss)(values, key)
            norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(gradients)))
            gradients = jax.tree.map(
                lambda g: g * jnp.minimum(1.0, clip_norm / (norm + 1e-12)), gradients
            )
            first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, state.first, gradients)
            second = jax.tree.map(
                lambda v, g: 0.999 * v + 0.001 * g * g, state.second, gradients
            )
            count = index + 1
            correction = jnp.sqrt(1 - 0.999**count) / (1 - 0.9**count)
            rate = _schedule(index, learning_rate, warmup, steps)
            values = jax.tree.map(
                lambda w, m, v, s, d: (
                    w - rate * s * (correction * m / (jnp.sqrt(v) + 1e-8) + d * w)
                ),
                values,
                first,
                second,
                rate_scale,
                decay,
            )
            return (values, _Adam(first, second)), value

        indices = start + jnp.arange(keys.shape[0])
        (values, state), losses = jax.lax.scan(step, (values, state), (indices, keys))
        return values, state, losses

    run = jax.jit(steps_program)
    zeros = jax.tree.map(jnp.zeros_like, trained)
    state = _Adam(zeros, zeros)
    losses: list[float] = []
    stopped = None
    done = 0
    while done < steps:
        count = min(steps_per_call, steps - done)
        episode_key, key = jax.random.split(episode_key)
        updated, new_state, chunk = run(
            trained, state, jnp.asarray(done), jax.random.split(key, count)
        )
        chunk = np.asarray(chunk)
        finite = np.isfinite(chunk)
        if not finite.all():
            first_bad = int(np.argmin(finite))
            losses.extend(chunk[:first_bad].tolist())
            stopped = f"non-finite loss at step {done + first_bad}"
            break
        trained, state = updated, new_state
        for offset, value in enumerate(chunk.tolist()):
            losses.append(value)
            if progress is not None:
                progress(done + offset, value)
        done += count
    result = delta(trained)
    return Finetuned(
        (functions, result.merge(parameters)), result, tuple(losses), stopped
    )


def load(directory: str | Path, model: Model | None = None) -> Finetuned:
    """Loads a delta directory and merges it into its base model.

    Args:
        directory: A directory `Finetuned.save` wrote.
        model: The base model's contract calls and parameters, for a delta
            trained on a model passed that way; the catalog model it names if None.

    Raises:
        ValueError: If the delta names no catalog model and none is given, its
            base's pinned revision has moved since it was trained, or it does
            not fit the model.
    """
    delta = Delta.read(directory)
    if model is None:
        if delta.base is None:
            raise ValueError(
                f"the delta in {directory} names no catalog model; pass its base model to load"
            )
        model, _, revision = _base(delta.base)
        if delta.revision is not None and revision != delta.revision:
            raise ValueError(
                f"the delta in {directory} was trained on {delta.base} at revision "
                f"{delta.revision}, which is now pinned at {revision}"
            )
    functions, parameters = model
    return Finetuned((functions, delta.merge(parameters)), delta)
