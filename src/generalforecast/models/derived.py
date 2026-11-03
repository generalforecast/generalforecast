# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Derived models: weights a hosted plane keeps over a catalog base model.

A derived model is named `<owner>/<name>@ft<version>`, such as
`acme/credit-default@ft3`. gf knows only what one is made of: the catalog
base it runs on, pinned by digest, the digest of its own weights, and how
many bags of weights it averages. The hosted plane that keeps them installs
a resolver, and gf asks it what a name means and, to run one, for its
weights; without one, no derived model resolves. How the weights were made
is the plane's alone.

A plane installs its resolver in process with `use`. Engine workers are
processes of their own, so they find it by `GF_DERIVED_RESOLVER`, a
`module:attribute` naming the resolver, which they import the first time a
derived model is looked up.

To run a derived model, gf loads its catalog base once and asks the
resolver for `weights`: each bag's own part, stacked along a leading axis,
and how one part combines with the base's parameters into a bag's. gf runs
them as one `Bagged` model, so every derived model on a base shares that
base's copy. Weights are arguments to compiled programs, never constants
in them, so a program compiled for one derived model serves every other of
the same shapes and leaks no weights through the compile cache.

This module is standard library alone, as the catalog is.
"""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

# Names the resolver engine workers import: `module:attribute`.
RESOLVER = "GF_DERIVED_RESOLVER"


@dataclass(frozen=True)
class Derived:
    """What a derived model is made of.

    Attributes:
        model: Its id, `<owner>/<name>@ft<version>`.
        base: The catalog id of the base model it runs on.
        base_digest: The base artifact's digest, `sha256:...`, which the
            weights were made against and are only valid with.
        digest: The digest of its own weights, `sha256:...`.
        bags: How many sets of weights it averages; each counts as one
            estimator against the serving policy.
    """

    model: str
    base: str
    base_digest: str
    digest: str
    bags: int = 1


@dataclass(frozen=True)
class Weights:
    """A derived model's weights, as its resolver hands them over.

    Attributes:
        own: Each bag's own part of its parameters, as a pytree whose every
            leaf stacks the bags along its leading axis.
        combine: Gives one bag's parameters from the base model's parameters
            and that bag's part. The same function for every derived model
            of a kind lets them share compiled programs.
    """

    own: Any
    combine: Callable[[Any, Any], Any]


class Resolver(Protocol):
    """What a hosted plane answers for the derived models it keeps."""

    def resolve(self, model: str) -> Derived:
        """What `model`, `<owner>/<name>@ft<version>`, is made of.

        Raises:
            catalog.ModelRefError: `UnknownModel` for a model it does not
                keep, or one the caller may not use.
        """
        ...

    def weights(self, made: Derived, base: Any) -> Weights:
        """The weights of `made`, verified against `made.digest`.

        Args:
            made: What `resolve` said the model is made of.
            base: The base model's loaded parameters, which the weights
                combine with.
        """
        ...


_resolver: Resolver | None = None


def use(resolver: Resolver | None) -> None:
    """Installs the resolver derived models are looked up with; None removes it."""
    global _resolver
    _resolver = resolver


def resolver() -> Resolver | None:
    """The installed resolver, if any; else the one `GF_DERIVED_RESOLVER` names.

    Raises:
        ValueError: If `GF_DERIVED_RESOLVER` is not `module:attribute`.
    """
    global _resolver
    if _resolver is None and (named := os.environ.get(RESOLVER)):
        module, colon, attribute = named.partition(":")
        if not (module and colon and attribute):
            raise ValueError(f"{RESOLVER} names a resolver as module:attribute")
        _resolver = getattr(importlib.import_module(module), attribute)
    return _resolver
