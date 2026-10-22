# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from generalforecast.packages.validation import PackageError
from generalforecast.python_packages.binding import PythonBinding
from generalforecast.python_packages.tables import from_data, to_data


def test_canonical_binding_roundtrip_and_data_path(revision, query):
    binding = PythonBinding.from_revision(revision)
    assert binding.to_dict()["kind"] == "python"
    received = PythonBinding.from_dict(binding.to_dict())
    assert received.verify(revision.archive) == revision
    assert from_data(to_data(query), query.schema) == query
    with pytest.raises(PackageError, match="differs"):
        received.verify(revision.archive + b"tampered")
    with pytest.raises(PackageError):
        PythonBinding.from_dict({**binding.to_dict(), "project": "other"})


def test_sdk_data_checks(revision):
    schema = revision.spec.inputs
    with pytest.raises(PackageError, match="order/names"):
        from_data({"other": [1.0]}, schema)
    with pytest.raises(PackageError, match="physical types"):
        from_data({"x": [1]}, schema)
    with pytest.raises(ValueError, match="infinite"):
        from_data({"x": [float("inf")]}, schema)
