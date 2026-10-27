# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One canonical JSON encoding for content identity, shared by artifacts and locks."""

import hashlib
import json
from types import MappingProxyType

import pytest

from generalforecast.predictors.validation import PredictorError, canonical, digest
from generalforecast.serialization import canonical_json, content_digest
from generalforecast.service.catalog_artifacts import artifact


@pytest.mark.parametrize("filename", ["model.safetensors", "échantillon.safetensors"])
def test_artifact_and_package_content_identity_share_one_encoding(filename):
    files = {filename: "a" * 64, "config.json": "b" * 64}
    result = artifact("organization/model", "c" * 40, "test", files)
    assert result["digest"] == digest(files) == content_digest(files)
    if filename.isascii():
        previous = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        assert result["digest"] == "sha256:" + hashlib.sha256(previous).hexdigest()


def test_canonical_values_include_immutable_mappings_and_tuples():
    value = MappingProxyType({"z": (1, True, None), "a": "é"})
    assert (
        canonical_json(value)
        == canonical(value)
        == b'{"a":"\xc3\xa9","z":[1,true,null]}'
    )


@pytest.mark.parametrize(
    "value", [{1: "invalid"}, {"x": object()}, float("inf"), float("nan")]
)
def test_invalid_values_fail_in_the_shared_encoder_and_at_package_boundary(value):
    with pytest.raises(ValueError):
        canonical_json(value)
    with pytest.raises(PredictorError) as caught:
        canonical(value)
    assert caught.value.code == "InvalidPredictor"
