# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`as_of` on the wire: RFC 3339 text or epoch seconds in, RFC 3339 UTC out."""

import datetime

import pandas as pd
import pytest

from generalforecast import as_of
from generalforecast.contracts import ContextVersion, context_version
from generalforecast.service.contracts import ServiceError

MOMENT = 1793088000.0  # 2026-10-27T08:00:00Z


@pytest.mark.parametrize(
    "given",
    [
        "2026-10-27T08:00:00Z",
        "2026-10-27T08:00:00",
        "2026-10-27T10:00:00+02:00",
        MOMENT,
        int(MOMENT),
        datetime.datetime(2026, 10, 27, 8, tzinfo=datetime.UTC),
        # A naive time is UTC.
        datetime.datetime(2026, 10, 27, 8, tzinfo=datetime.UTC).replace(tzinfo=None),
        pd.Timestamp("2026-10-27 08:00", tz="UTC"),
    ],
)
def test_every_accepted_form_is_one_instant(given):
    assert as_of.parse(given) == MOMENT
    assert as_of.wire(given) == "2026-10-27T08:00:00Z"


def test_a_date_is_its_midnight_in_utc_and_fractions_are_kept():
    assert as_of.wire(datetime.date(2026, 10, 27)) == "2026-10-27T00:00:00Z"
    assert as_of.format(MOMENT + 0.25) == "2026-10-27T08:00:00.250000Z"
    assert as_of.format(None) is None and as_of.parse(None) is None


@pytest.mark.parametrize(
    "given",
    ["yesterday", "", "2026-10-27", True, float("nan"), float("inf"), -1, [MOMENT]],
)
def test_anything_else_is_a_validation_error_naming_as_of(given):
    with pytest.raises(ServiceError) as refused:
        as_of.parse(given)
    assert (refused.value.code, refused.value.param, refused.value.status) == (
        "ValidationError",
        "as_of",
        422,
    )


def test_a_prediction_context_has_one_shape():
    live = ContextVersion("live", 7, 6, MOMENT, name="events")
    assert live.to_dict() == {
        "source": "live",
        "name": "events",
        "version": 7,
        "digest": None,
        "rows": 6,
        "as_of": "2026-10-27T08:00:00Z",
    }
    assert context_version(live.to_dict()) == live
    snapshot = ContextVersion("snapshot", rows=3, digest="sha256:" + "a" * 64)
    assert ContextVersion("snapshot", "sha256:" + "a" * 64, 3) == snapshot
    # gf-server's earlier form, epoch seconds included, reads as the same value.
    assert context_version(
        {"kind": "live", "name": "events", "rows": 6, "as_of": MOMENT}
    ) == ContextVersion("live", None, 6, MOMENT, name="events")


def test_requests_read_as_the_fixture_gf_server_shares_says():
    import json
    from pathlib import Path

    fixture = Path(__file__).resolve().parents[2] / "crates/gf-server/tests/fixtures"
    cases = json.loads((fixture / "as_of.json").read_text())["cases"]
    for given, expected in cases:
        if expected is None:
            with pytest.raises(ServiceError):
                as_of.micros(given)
        else:
            assert as_of.micros(given) == expected, given
            # Written back, it reads as the same microsecond.
            assert as_of.micros(as_of.wire(given)) == expected, given
