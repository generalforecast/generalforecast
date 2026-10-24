# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sklearn's estimator bases, or stand-ins when the `sklearn` extra is absent.

The estimators subclass these at import time, so importing `generalforecast`
never needs sklearn; their constructors call `require_sklearn`, which raises
the install hint when it is missing.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from generalforecast.errors import InvalidInput, boundary

HINT = 'pip install "generalforecast[sklearn]"'

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.multiclass import type_of_target
    from sklearn.utils.validation import validate_data

    MISSING: ImportError | None = None
else:
    try:
        from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.multiclass import type_of_target
        from sklearn.utils.validation import validate_data

        MISSING: ImportError | None = None
    except ImportError as error:  # pragma: no cover - exercised without the extra
        MISSING = error

        class BaseEstimator:  # type: ignore[no-redef]
            """A stand-in for sklearn's base; constructors refuse to run."""

            def get_params(self, deep: bool = True) -> dict[str, Any]:
                """Returns the constructor arguments, as sklearn does."""
                return {
                    key: value
                    for key, value in vars(self).items()
                    if not key.startswith("_") and not key.endswith("_")
                }

            def set_params(self, **params: Any) -> Any:
                """Sets constructor arguments, as sklearn does."""
                for key, value in params.items():
                    setattr(self, key, value)
                return self

            def __getstate__(self) -> dict[str, Any]:
                """Pickles the instance's attributes."""
                return dict(vars(self))

        class ClassifierMixin:  # type: ignore[no-redef]
            """A stand-in for sklearn's classifier mixin."""

        class RegressorMixin:  # type: ignore[no-redef]
            """A stand-in for sklearn's regressor mixin."""

        class NotFittedError(ValueError, AttributeError):  # type: ignore[no-redef]
            """Raised when an estimator is used before it is fit."""

        def type_of_target(y: Any) -> str:  # type: ignore[misc]
            """Unavailable without sklearn."""
            require_sklearn()
            raise AssertionError  # pragma: no cover

        def validate_data(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            """Unavailable without sklearn."""
            require_sklearn()
            raise AssertionError  # pragma: no cover


def require_sklearn() -> None:
    """Raises `InvalidInput` with the install hint when sklearn is missing."""
    if MISSING is not None:
        raise InvalidInput(
            "the estimators need scikit-learn, which is not installed",
            hint=HINT,
        ) from None


def public(method: Callable[..., Any]) -> Callable[..., Any]:
    """Raises every failure inside a public estimator method as a `gf.Error`."""

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            with boundary():
                return method(*args, **kwargs)
        except InvalidInput as error:
            cause = error.__cause__
            if isinstance(cause, NotFittedError):
                raise cause from None
            raise

    return wrapped


__all__ = [
    "HINT",
    "BaseEstimator",
    "ClassifierMixin",
    "NotFittedError",
    "RegressorMixin",
    "public",
    "require_sklearn",
    "type_of_target",
    "validate_data",
]
