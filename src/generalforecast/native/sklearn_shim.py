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
from typing import TYPE_CHECKING, Any, Self

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


class _Refused(InvalidInput):
    """Input an estimator refuses: a `gf.Error` that reads as the builtin error it came from.

    sklearn's tools and checks look for a `ValueError` or `TypeError` by its
    message, word for word; the fields carry the error as `gf` renders it.
    """

    _raised: str

    @classmethod
    def of(cls, error: InvalidInput, cause: Exception) -> Self:
        """The error `error` is, read as `cause` reads."""
        found = cls(
            error.message,
            code=error.code,
            reason=error.reason,
            param=error.param,
            because=error.because,
            hint=error.hint,
            limit=error.limit,
            actual=error.actual,
            request_id=error.request_id,
            path=error.path,
            line=error.line,
            column=error.column,
        )
        found._raised = str(cause)
        return found

    def __str__(self) -> str:
        """The builtin error's own message."""
        return self._raised


class EstimatorInputError(_Refused, ValueError):
    """Input an estimator refuses with a `ValueError`, as sklearn's validation does."""


class EstimatorTypeError(_Refused, TypeError):
    """Input of a type an estimator refuses with a `TypeError`, as sklearn's validation does."""


# Each builtin error sklearn's tools expect, and what an estimator raises for it.
_REFUSED: tuple[tuple[type[Exception], type[_Refused]], ...] = (
    (ValueError, EstimatorInputError),
    (TypeError, EstimatorTypeError),
)


def public(method: Callable[..., Any]) -> Callable[..., Any]:
    """Raises every failure inside a public estimator method as a `gf.Error`.

    A `NotFittedError` stays one, and input refused with a `ValueError` or
    `TypeError`, as sklearn's validation refuses it, raises an
    `EstimatorInputError` or `EstimatorTypeError`, so sklearn's tools and
    checks still see the errors they look for.
    """

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            with boundary():
                return method(*args, **kwargs)
        except InvalidInput as error:
            cause = error.__cause__
            if isinstance(cause, NotFittedError):
                raise cause from None
            for builtin, refused in _REFUSED:
                if isinstance(cause, builtin) and not isinstance(error, builtin):
                    raise refused.of(error, cause) from cause
            raise

    return wrapped


__all__ = [
    "HINT",
    "BaseEstimator",
    "ClassifierMixin",
    "EstimatorInputError",
    "EstimatorTypeError",
    "NotFittedError",
    "RegressorMixin",
    "public",
    "require_sklearn",
    "type_of_target",
    "validate_data",
]
