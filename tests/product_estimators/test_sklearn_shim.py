# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Estimator errors as both `gf.Error` and the builtin errors sklearn's tools look for."""

import numpy as np
import pytest
from sklearn.utils.validation import check_array, check_scalar

from generalforecast.errors import InvalidInput
from generalforecast.native.sklearn_shim import NotFittedError, public


@public
def _validating(rows: object) -> None:
    check_array(rows)


@public
def _checking(value: object) -> None:
    check_scalar(value, "horizon", int)


@public
def _raising(error: Exception) -> None:
    raise error


def test_input_sklearn_refuses_with_a_value_error_reads_as_one() -> None:
    with pytest.raises(ValueError, match=r"^Expected 2D array") as caught:
        _validating(np.arange(3.0))

    assert isinstance(caught.value, InvalidInput)
    assert caught.value.message.startswith("expected 2D array")


def test_input_sklearn_refuses_with_a_type_error_reads_as_one() -> None:
    with pytest.raises(TypeError, match=r"^horizon must be an instance of") as caught:
        _checking(1.5)

    assert isinstance(caught.value, InvalidInput)


def test_a_not_fitted_error_stays_one() -> None:
    with pytest.raises(NotFittedError):
        _raising(NotFittedError("call fit first"))


def test_an_error_the_engine_did_not_raise_stays_itself() -> None:
    with pytest.raises(KeyError) as caught:
        _raising(KeyError("table 'x' does not exist"))

    assert not isinstance(caught.value, InvalidInput)
