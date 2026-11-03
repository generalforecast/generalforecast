# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Derived models: weights a hosted plane keeps over a catalog base model.

A derived model is named `<owner>/<name>@ft<version>`, such as
`acme/credit-default@ft3`. gf knows only what one is made of: the catalog
base it runs on, pinned by digest, the digest of its own weights, and how
many bags of weights it averages. The hosted plane that keeps them installs
a resolver with `use`, and gf asks it what a name means; without one, no
derived model resolves. How the weights were made is the plane's alone.

This module is standard library alone, as the catalog is.
"""

from dataclasses import dataclass
from typing import Protocol


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


class Resolver(Protocol):
    """What a hosted plane answers for the derived models it keeps."""

    def resolve(self, model: str) -> Derived:
        """What `model`, `<owner>/<name>@ft<version>`, is made of.

        Raises:
            catalog.ModelRefError: `UnknownModel` for a model it does not
                keep, or one the caller may not use.
        """
        ...


_resolver: Resolver | None = None


def use(resolver: Resolver | None) -> None:
    """Installs the resolver derived models are looked up with; None removes it."""
    global _resolver
    _resolver = resolver


def resolver() -> Resolver | None:
    """The installed resolver, if any."""
    return _resolver
