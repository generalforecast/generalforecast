# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a hosted plane's trainer builds on: a catalog model's calls and parameters.

A plane fine-tunes a catalog model with its own training code. gf gives it
the model as the engine runs it, so what it trains is what gf serves, and
the parameter types it adapts. It reaches this module as
`generalforecast.hosting.training`; `VERSION` changes with any change to
what this module offers.

gf holds no training code: how a trainer samples, scores and updates is the
plane's, and what it publishes comes back as a derived model's weights
(`generalforecast.hosting.DerivedWeights`).

Exports:
    VERSION: this surface's version.
    load: a catalog model's contract calls and parameters.
    TabularFunctions: a tabular model's contract calls.
    LinearParameters: a linear layer's weight, in the (output, input)
        layout, and bias: what low-rank and full fine-tuning adapt.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from generalforecast.models.contract import TabularFunctions
    from generalforecast.nn.linear import LinearParameters

VERSION = 1


def __getattr__(name: str) -> Any:
    """The parameter types, imported on first use: `hosting` stays free of JAX."""
    if name == "TabularFunctions":
        from generalforecast.models.contract import TabularFunctions

        return TabularFunctions
    if name == "LinearParameters":
        from generalforecast.nn.linear import LinearParameters

        return LinearParameters
    raise AttributeError(name)


def load(model: str, *, allow_downloads: bool = True) -> "tuple[TabularFunctions, Any]":
    """A catalog tabular model's contract calls and verified parameters.

    The model loads as hosted predictors load it, under the predictor
    surface's gates, so a trainer only fine-tunes what gf serves.

    Raises:
        ValueError: If `model` is not a catalog tabular model qualified for
            hosted predictors.
    """
    from generalforecast.models import registry
    from generalforecast.models.contract import TabularFunctions

    functions, parameters = registry.load(
        model, surface="predictor", allow_downloads=allow_downloads
    )
    if not isinstance(functions, TabularFunctions):
        raise ValueError(f"{model} is not a tabular model")  # noqa: TRY004
    return functions, parameters


__all__ = ["VERSION", "LinearParameters", "TabularFunctions", "load"]
