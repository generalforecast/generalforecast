# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Strict portable values used by the authored and generated predictor formats."""

import hashlib
import re
from collections.abc import Set as AbstractSet
from typing import Any

from generalforecast.serialization import canonical_json


class PredictorError(ValueError):
    """A predictor cannot be compiled without an explicit user correction."""

    def __init__(self, code: str, message: str):
        """Keeps machine-readable failure codes separate from safe diagnostics."""
        self.code = code
        self.message = message
        super().__init__(message)


def ensure(condition: Any, message: str, code: str = "InvalidPredictor") -> None:
    """Raises a stable error without interpolating untrusted values."""
    if not condition:
        raise PredictorError(code, message)


def fields(
    value: Any, required: set[str], optional: AbstractSet[str] = frozenset()
) -> dict:
    """Rejects missing and unknown keys, including credentials and runtime placement."""
    ensure(type(value) is dict, "expected an object")
    ensure(required <= value.keys(), "required fields are missing")
    ensure(value.keys() <= required | optional, "unknown fields are not supported")
    return value


def text(value: Any) -> str:
    """Checks a bounded literal, never an environment expansion or secret value."""
    ensure(type(value) is str and 0 < len(value) <= 512, "expected bounded text")
    ensure(not any(ord(c) < 32 for c in value), "control characters are not allowed")
    ensure(
        not any(c in value for c in ("${", "{{", "://", "?", "#")),
        "use logical references, not URLs or interpolation",
    )
    ensure(
        not re.search(
            r"(?i)(gf_(?:live|test)_|hf_|sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{12,}|(?:AKIA|ASIA)[A-Z0-9]{16}",
            value,
        ),
        "embedded credentials are forbidden",
    )
    return value


def column(value: Any) -> str:
    """Checks a column name or a class label: any printable text a CSV header or cell holds.

    Unlike `text`, which guards references a service resolves, a column or
    label is data: `is_vip?`, `item #` and `{{x}}` are fine names.
    """
    ensure(
        type(value) is str and 0 < len(value) <= 512,
        "column names and labels are 1 to 512 characters",
    )
    ensure(not any(ord(c) < 32 for c in value), "control characters are not allowed")
    return value


# The one rule for the names people choose: predictors, live contexts and
# environments. `gf.yaml`, `gf.lock`, the SDK and every service route check it.
NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")
# Opaque identifiers the platform issues or operators configure, such as
# projects, runtimes and secret names, which may also hold capitals and `_`.
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}")


def name(value: Any) -> str:
    """Checks a chosen name: lowercase letters, digits and hyphens, starting with a letter."""
    ensure(
        type(value) is str and NAME.fullmatch(value) is not None,
        "names use lowercase letters, digits and hyphens, starting with a letter",
    )
    return value


def identifier(value: Any) -> str:
    """Checks a route-safe opaque identifier, such as a project or a secret's name."""
    ensure(
        type(value) is str and IDENTIFIER.fullmatch(value) is not None,
        "invalid resource name",
    )
    return value


def names(value: Any) -> tuple[str, ...]:
    """Copies unique ordered column names into an immutable tuple."""
    ensure(type(value) is list, "expected a list of columns")
    result = tuple(column(item) for item in value)
    ensure(len(set(result)) == len(result), "columns must be unique")
    return result


def integer(value: Any, low: int = 1, high: int = 10**9) -> int:
    """Rejects booleans, coercion and out-of-range resource counts."""
    ensure(
        type(value) is int and low <= value <= high,
        "integer is outside supported bounds",
    )
    return value


def canonical(value: Any) -> bytes:
    """Encodes sorted compact UTF-8 JSON; floats must be finite."""
    try:
        return canonical_json(value)
    except (ValueError, UnicodeError):
        raise PredictorError("InvalidPredictor", "value is not portable JSON") from None


def digest(value: Any) -> str:
    """Hashes canonical structured content with its algorithm prefix."""
    return bytes_digest(canonical(value))


def bytes_digest(value: bytes) -> str:
    """Hashes raw bytes independently of their container format."""
    return "sha256:" + hashlib.sha256(value).hexdigest()


def hash_value(value: Any) -> str:
    """Checks a full content address, never a mutable alias."""
    ensure(
        type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value),
        "expected a full sha256 digest",
    )
    return value
