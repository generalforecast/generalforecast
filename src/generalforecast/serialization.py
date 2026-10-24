# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Portable canonical JSON shared by content-addressed model and predictor identities."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Encode finite portable values as sorted compact UTF-8 JSON."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            if not all(type(key) is str for key in item):
                raise ValueError("JSON keys must be strings")
            return {key: normalize(child) for key, child in item.items()}
        if type(item) in (tuple, list):
            return [normalize(child) for child in item]
        if item is not None and type(item) not in (str, bool, int, float):
            raise ValueError("expected portable JSON")
        return item

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    """Hash canonical structured content with its algorithm prefix."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()
