# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The test suite's small models, registered as engine implementations.

Workers import this module through `GF_WORKER_MODULES`; `tests/conftest.py`
imports it for in-process engines. Each loader imports its family on first use.
"""

from __future__ import annotations

from typing import Any

from generalforecast.models import registry


def _tabicl_classifier() -> tuple[Any, Any]:
    from tests.models.tabicl.tiny import classifier

    return classifier()


def _tabicl_regressor() -> tuple[Any, Any]:
    from tests.models.tabicl.tiny import regressor

    return regressor()


def _kumo_classifier() -> tuple[Any, Any]:
    from tests.models.kumo.tiny import classifier

    return classifier()


def _kumo_regressor() -> tuple[Any, Any]:
    from tests.models.kumo.tiny import regressor

    return regressor()


def _limix_classifier() -> tuple[Any, Any]:
    from tests.models.limix.tiny import classifier

    return classifier()


def _limix_regressor() -> tuple[Any, Any]:
    from tests.models.limix.tiny import regressor

    return regressor()


def _tabpfn_classifier() -> tuple[Any, Any]:
    from tests.models.tabpfn.tiny import classifier

    return classifier()


def _tabpfn_regressor() -> tuple[Any, Any]:
    from tests.models.tabpfn.tiny import regressor

    return regressor()


def _timestral_forecaster() -> tuple[Any, Any]:
    from tests.models.timestral.tiny import forecaster

    return forecaster()


def _timesfm3_forecaster() -> tuple[Any, Any]:
    from tests.models.timesfm3.tiny import forecaster

    return forecaster()


def _chronos2_forecaster() -> tuple[Any, Any]:
    from tests.models.chronos2.tiny import forecaster

    return forecaster()


def _chronos2_catalog_forecaster() -> tuple[Any, Any]:
    from tests.models.chronos2.tiny import catalog_forecaster

    return catalog_forecaster()


TINY_MODELS = {
    "tests/tabicl-classifier": _tabicl_classifier,
    "tests/tabicl-regressor": _tabicl_regressor,
    "tests/kumo-classifier": _kumo_classifier,
    "tests/kumo-regressor": _kumo_regressor,
    "tests/limix-classifier": _limix_classifier,
    "tests/limix-regressor": _limix_regressor,
    "tests/tabpfn-classifier": _tabpfn_classifier,
    "tests/tabpfn-regressor": _tabpfn_regressor,
    "tests/timestral-forecaster": _timestral_forecaster,
    "tests/timesfm3-forecaster": _timesfm3_forecaster,
    "tests/chronos2-forecaster": _chronos2_forecaster,
    "tests/chronos2-catalog-forecaster": _chronos2_catalog_forecaster,
}

for _id, _loader in TINY_MODELS.items():
    registry.register(_id, _loader)
