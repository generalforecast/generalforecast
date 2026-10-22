# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A kernel classifier written from scratch against generalforecast's tabular contract.

The classifier weights each query's neighbors' labels by their distance. In
practice, one would use a built-in model; this shows the calls the engine runs
every model through: fit, embed, encode, decode and postprocess, and checks the
model against what the engine relies on with `generalforecast.models.conformance.check`.

It requires scikit-learn (uv sync --extra sklearn).
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.datasets import load_wine

import generalforecast
from generalforecast.models.conformance import check
from generalforecast.models.contract import TabularFunctions, TabularProperties
from generalforecast.nn.attention import AttentionCache

CLASSES = 10


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Kernel:
    """Parameters: the kernel's temperature, and the width rows are padded to."""

    temperature: Array
    width: int = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class State:
    """Each column's center and scale, fit on the table, and its class count."""

    center: Array
    scale: Array
    num_classes: int = field(metadata={"static": True})


def fit(model, features, targets, num_classes, num_rows=None, categorical=None):
    """Fits the state on the table's real rows, and embeds them."""
    rows = features.shape[0] if num_rows is None else num_rows
    valid = jnp.arange(features.shape[0]) < rows
    present = valid[:, None] & ~jnp.isnan(features)
    count = jnp.maximum(present.sum(axis=0), 1)
    center = jnp.where(present, features, 0).sum(axis=0) / count
    spread = jnp.where(present, (features - center) ** 2, 0).sum(axis=0) / count
    state = State(center, jnp.where(spread > 0, jnp.sqrt(spread), 1.0), num_classes)
    return state, embed(model, state, features)


def embed(model, state, features, targets=None):
    """Standardizes rows, fills missing values with the center, pads to the width."""
    scaled = jnp.nan_to_num((features - state.center) / state.scale)
    return jnp.pad(scaled, ((0, 0), (0, model.width - features.shape[1])))


def encode(model, state, rows, targets, num_rows=None):
    """Keeps the rows as keys and their one-hot labels as values."""
    labels = jax.nn.one_hot(targets, CLASSES, dtype=rows.dtype)
    length = jnp.asarray(rows.shape[0] if num_rows is None else num_rows, jnp.int32)
    return AttentionCache(key=rows[:, None, :], value=labels[:, None, :], length=length)


def decode(model, state, cache, rows):
    """Averages the context's labels, weighted by each row's distance to the query.

    Keys past the cache's length are padding, and when the engine sets a mask,
    each query reads only the keys it allows. The engine may store caches in a
    reduced precision; this model leaves `stored_cache` unset, so they come
    back in float32, as `encode` made them.
    """
    keys, labels = cache.key[:, 0], cache.value[:, 0]
    distances = jnp.sum((rows[:, None, :] - keys[None, :, :]) ** 2, axis=-1)
    # A cache without a length has only real keys.
    length = keys.shape[0] if cache.length is None else cache.length
    allowed = (jnp.arange(keys.shape[0]) < length)[None, :]
    if cache.mask is not None:
        allowed = allowed & cache.mask
    # The dtype's least value rather than -inf, so a padded query that reads
    # no key gives finite outputs, which the engine discards.
    least = jnp.finfo(distances.dtype).min
    logits = jnp.where(allowed, -distances / model.temperature, least)
    return jax.nn.softmax(logits, axis=-1) @ labels


def postprocess(model, state, outputs):
    """Keeps the task's classes."""
    probabilities = outputs[:, : state.num_classes]
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def properties(model) -> TabularProperties:
    """What the engine needs to know before running the model."""
    return TabularProperties(
        regression=False,
        class_slots=CLASSES,
        num_outputs=CLASSES,
    )


kernel_functions = TabularFunctions(fit, embed, encode, decode, postprocess, properties)


def kernel() -> tuple[TabularFunctions, Kernel]:
    """The model as a spec names it: `custom_model:kernel` from this directory."""
    return kernel_functions, Kernel(temperature=jnp.asarray(2.0), width=32)


if __name__ == "__main__":
    # Raises if the model breaks a guarantee the engine relies on.
    check(*kernel())

    X, y = load_wine(return_X_y=True)
    X = X.astype(np.float32)
    order = np.random.default_rng(0).permutation(len(X))
    train, test = order[:140], order[140:]

    # A server would name it in its configuration as "custom_model:kernel".
    client = generalforecast.local()
    table = client.create_table("wine", model=kernel(), features=13, classes=3)
    table.insert(train, X[train], y[train])
    predicted = table.predict(X[test]).argmax(axis=1)
    print("kernel model accuracy:", np.mean(predicted == y[test]))
