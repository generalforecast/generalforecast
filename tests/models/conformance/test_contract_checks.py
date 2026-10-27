# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Each contract check catches the model that breaks what it checks.

Every model here is the kernel classifier of `custom_model.py` with one call
changed the way a custom model could get it wrong.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generalforecast.models import custom
from generalforecast.models.catalog import ModelRefError
from generalforecast.models.conformance import check
from generalforecast.models.contract import TabularFunctions
from tests.models.conformance import custom_model as kernel

PARAMETERS = kernel.Kernel(temperature=jnp.asarray(2.0), width=32)


def _claiming(**claims: Any) -> Callable[[Any], Any]:
    """The kernel's properties, claiming more than it does."""
    return lambda model: dataclasses.replace(kernel.properties(model), **claims)


def _fit_without_categorical(model, features, targets, num_classes, num_rows=None):
    return kernel.fit(model, features, targets, num_classes, num_rows)


def _fit_without_keywords(model, features, targets, num_classes):
    return kernel.fit(model, features, targets, num_classes)


def _decode_masks_without_lengths(model, state, cache, rows):
    """Reads the keys the mask allows, past the length when a mask is set."""
    if cache.mask is None:
        return kernel.decode(model, state, cache, rows)
    keys, labels = cache.key[:, 0], cache.value[:, 0]
    distances = jnp.sum((rows[:, None, :] - keys[None, :, :]) ** 2, axis=-1)
    least = jnp.finfo(distances.dtype).min
    logits = jnp.where(cache.mask, -distances / model.temperature, least)
    return jax.nn.softmax(logits, axis=-1) @ labels


def _postprocess_three_classes(model, state, outputs):
    probabilities = outputs[:, :3]
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _postprocess_unnormalized(model, state, outputs):
    """Leaves the share of unlabeled rows out, so probabilities fall short of 1."""
    return outputs[:, : state.num_classes]


def _reading_another_cache(model, state, read, rows):
    return 2 * kernel.decode(model, state, read(rows, lambda cache: cache), rows)


def _crowded_array(state) -> Any:
    return np.int64(0)


BROKEN: dict[str, dict[str, Any]] = {
    "calls": {"fit": _fit_without_categorical},
    "masks": {"decode": _decode_masks_without_lengths},
    "class counts": {"postprocess": _postprocess_three_classes},
    # The kernel's decode reads raw int8 keys, without their scales.
    "stored caches": {"properties": _claiming(stored_cache=True)},
    "unlabeled context": {
        "properties": _claiming(unlabeled_context=True),
        "postprocess": _postprocess_unnormalized,
    },
    "decode reading": {"decode_reading": _reading_another_cache},
    "crowded categories": {"crowded_categories": _crowded_array},
}


def test_the_kernel_classifier_passes_every_check() -> None:
    check(kernel.kernel_functions, PARAMETERS)


@pytest.mark.parametrize("name", list(BROKEN))
def test_each_check_catches_the_model_that_breaks_it(name: str) -> None:
    functions = kernel.kernel_functions._replace(**BROKEN[name])

    with pytest.raises(AssertionError, match=f"^{name}: "):
        check(functions, PARAMETERS)


@pytest.fixture
def module() -> Iterator[types.ModuleType]:
    """A module custom specs name, holding broken kernels."""
    held = types.ModuleType("broken_kernels")
    sys.modules[held.__name__] = held
    yield held
    del sys.modules[held.__name__]
    custom._capability.cache_clear()


def _attribute(functions: TabularFunctions) -> Callable[[], tuple[Any, Any]]:
    return lambda: (functions, PARAMETERS)


@pytest.mark.parametrize(
    ("changes", "name", "reason"),
    [
        ({"fit": _fit_without_keywords}, "calls", "TypeError"),
        ({"fit": _fit_without_categorical}, "calls", "categorical"),
        (BROKEN["stored caches"], "stored caches", "int8 cache"),
    ],
)
def test_checked_custom_models_fail_to_load_as_invalid(
    module: types.ModuleType, changes: dict[str, Any], name: str, reason: str
) -> None:
    vars(module)["model"] = _attribute(kernel.kernel_functions._replace(**changes))

    with pytest.raises(ModelRefError) as caught:
        custom.checked("broken_kernels:model")

    assert caught.value.code == "InvalidModel"
    assert f"fails the contract's `{name}` check: " in str(caught.value)
    assert reason in str(caught.value)
