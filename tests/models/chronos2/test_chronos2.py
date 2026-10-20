"""Chronos-2's loader checks."""

from __future__ import annotations

import pytest

from generalforecast.models.chronos2.checkpoint import (
    check_supported,
    chronos2_from_state,
)
from tests.models.chronos2.tiny import tiny_checkpoint


def test_unsupported_settings_raise() -> None:
    state, settings = tiny_checkpoint()
    forecasting = settings["chronos_config"]

    with pytest.raises(NotImplementedError, match="feed_forward_proj='gated-gelu'"):
        check_supported({**settings, "feed_forward_proj": "gated-gelu"})
    with pytest.raises(NotImplementedError, match="use_arcsinh=False"):
        check_supported(
            {**settings, "chronos_config": {**forecasting, "use_arcsinh": False}}
        )
    with pytest.raises(NotImplementedError, match="different sizes"):
        check_supported(
            {**settings, "chronos_config": {**forecasting, "output_patch_size": 8}}
        )
    with pytest.raises(NotImplementedError, match="n_kv_heads"):
        check_supported({**settings, "n_kv_heads": 1})
    with pytest.raises(ValueError, match="parameters the model does not read"):
        chronos2_from_state({**state, "extra.weight": state["shared.weight"]}, settings)
