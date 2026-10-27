# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Estimator errors as both `gf.Error` and the builtin errors sklearn's tools look for."""

import pytest

from generalforecast.errors import Error, InvalidInput
from generalforecast.native.sklearn_shim import NotFittedError, public


@public
def _raising(error: Exception) -> None:
    raise error


@pytest.mark.parametrize("builtin", [ValueError, TypeError])
def test_refused_input_reads_as_the_builtin_error_it_came_from(builtin) -> None:
    with pytest.raises(builtin, match=r"^X has 4 features, but") as caught:
        _raising(builtin("X has 4 features, but Classifier is expecting 5"))

    assert isinstance(caught.value, InvalidInput)
    assert caught.value.message == "x has 4 features, but Classifier is expecting 5"


def test_a_not_fitted_error_stays_one_and_others_stay_gf_errors() -> None:
    with pytest.raises(NotFittedError):
        _raising(NotFittedError("call fit first"))
    with pytest.raises(Error) as caught:
        _raising(KeyError("table 'x' does not exist"))
    assert not isinstance(caught.value, ValueError | TypeError)
