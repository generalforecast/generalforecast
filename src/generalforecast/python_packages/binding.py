# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Content-bound extension for the canonical Package/SDK/API, not a new registry."""

from dataclasses import dataclass

from generalforecast.packages.validation import (
    bytes_digest,
    canonical,
    ensure,
    fields,
    hash_value,
)
from generalforecast.python_packages.compiler import PythonRevision


@dataclass(frozen=True)
class PythonBinding:
    """Lock extension plus archive content address for existing artifact upload paths.

    The shared package lock must include to_dict() under its own digest. The
    canonical server fetches only this project's archive blob and calls verify
    before selecting an operator-configured executor. This is admission evidence,
    not isolation, authentication, successful inference or provider readiness.
    """

    lock: bytes
    source: str

    @classmethod
    def from_revision(cls, revision: PythonRevision) -> "PythonBinding":
        """Construct a source-bound, immutable extension without loading code."""
        manifest = revision.inspect()
        return cls(revision.lock, manifest["archive"])

    def to_dict(self) -> dict:
        """Return the exact shared-lock extension fields, without source bytes."""
        import json

        hash_value(self.source)
        return {"kind": "python", "lock": json.loads(self.lock), "source": self.source}

    @classmethod
    def from_dict(cls, value: object) -> "PythonBinding":
        """Parse an untrusted lock extension; source verification is still required."""
        value = fields(value, {"kind", "lock", "source"})
        ensure(value["kind"] == "python", "expected Python lock extension")
        raw = canonical(value["lock"])
        ensure(len(raw) <= 262144, "Python lock exceeds byte bound")
        return cls(raw, hash_value(value["source"]))

    def verify(self, archive: bytes) -> PythonRevision:
        """Verify the complete source bytes before any executor may load a predictor."""
        ensure(
            bytes_digest(archive) == self.source,
            "Python source differs from package binding",
            "LockMismatch",
        )
        result = PythonRevision(self.lock, archive)
        manifest = result.inspect()
        ensure(
            manifest["archive"] == self.source,
            "Python archive differs from lock",
            "LockMismatch",
        )
        return result
