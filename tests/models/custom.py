# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Custom models as a user's `module:attribute` spec names them: pure Python, no weights.

`tests.models.custom:functions` is the kernel classifier of
`tests/models/conformance/custom_model.py`. `broken` is the same model with a
decode that reads the padding past a context's rows, which the contract's
checks catch; `weights` is no model at all. `forecaster` is a tiny Chronos-2
with random weights at the levels it unrolls and two no evenly spaced grid
implies.
"""

from __future__ import annotations

from typing import Any


def functions() -> tuple[Any, Any]:
    """The kernel classifier, as the engine's workers load it."""
    from tests.models.conformance.custom_model import kernel

    return kernel()


def broken() -> tuple[Any, Any]:
    """The kernel classifier reading padded keys as context rows."""
    import jax
    import jax.numpy as jnp

    from tests.models.conformance.custom_model import kernel

    def decode(model, state, cache, rows):
        keys, labels = cache.key[:, 0], cache.value[:, 0]
        distances = jnp.sum((rows[:, None, :] - keys[None, :, :]) ** 2, axis=-1)
        return jax.nn.softmax(-distances / model.temperature, axis=-1) @ labels

    kernel_functions, parameters = kernel()
    return kernel_functions._replace(decode=decode), parameters


weights = {"temperature": 2.0}

LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def forecaster() -> tuple[Any, Any]:
    """A tiny Chronos-2 forecasting at `LEVELS`."""
    from generalforecast.models.chronos2.model import chronos2_functions
    from tests.models.chronos2.tiny import chronos2_from_state, tiny_checkpoint

    return chronos2_functions, chronos2_from_state(*tiny_checkpoint(quantiles=LEVELS))
