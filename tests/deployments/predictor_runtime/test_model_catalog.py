# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Canonical model catalog: one entry per family and frozen v1 pin identities."""

import dataclasses

import pytest

from generalforecast.predictors.models import CatalogResolver
from generalforecast.predictors.validation import PredictorError, digest

# ModelPin.to_dict() at 3a7e9d0, before the shared catalog.
GOLDEN = {
    "general-forecast/chronos-2": {
        "cpu": 2,
        "digest": "sha256:f29896cb84d2c1b74cec30fe21c2f811bf50af92e381bf41f4093057a1bcd3a4",
        "files": {
            "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
            "NOTICE": "f244df7a5307a500b53a6675eb4058359509fbf332f37aee9bb490f9829cf3f5",
            "config.json": "4c685ee692bbe94cdf775b8f6d9fa46498078ea1303bec8c2689f39436e60986",
            "model.safetensors": "6d353b01a3a939d324ea3abfb8f11cfba6e817b57db9b30a23b0def70528a087",
        },
        "future_inputs": True,
        "input_types": ["int64", "float64"],
        "max_classes": None,
        "max_context_rows": 100_000,
        "max_features": 8,
        "max_horizon": 32,
        "memory": "8Gi",
        "missing_values": True,
        "name": "chronos-2",
        "outputs": ["value", "quantiles"],
        "past_inputs": True,
        "quantile_step": None,
        "quantiles": [
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ],
        "repository": "general-forecast/chronos-2",
        "revision": "a287b88beaeef49cf436e6735d9f61e13dfd50f8",
        "tasks": ["forecast"],
    },
    "general-forecast/tabicl-v2-classifier": {
        "cpu": 2,
        "digest": "sha256:b141070296a7981994afbf7c832e8bf9aa85657ed54baa8f231fcbcbca752b4d",
        "files": {
            "LICENSE": "4f90c5fa25a2b7883d6235c7f0b11f942a37859b9f89ac36157cc485329913b4",
            "config.json": "7ec2941cc537535e1bd8f37d33fc97cea97953343d69d765a261ee6a1605d937",
            "model.safetensors": "58f5fc1f276744f721974079c19d2f67efacc39083c54483fe3feba52297d253",
        },
        "future_inputs": False,
        "input_types": ["boolean", "int64", "float64", "string", "date", "timestamp"],
        "max_classes": 10,
        "max_context_rows": 100_000,
        "max_features": 32,
        "max_horizon": None,
        "memory": "8Gi",
        "missing_values": True,
        "name": "tabicl-v2-classifier",
        "outputs": ["label", "probabilities"],
        "past_inputs": False,
        "quantile_step": None,
        "quantiles": [],
        "repository": "general-forecast/tabicl-v2-classifier",
        "revision": "5fb6a4c6f24bd5fb3623f97e3c62186e12e28e5b",
        "tasks": ["classification"],
    },
    "general-forecast/tabicl-v2-regressor": {
        "cpu": 2,
        "digest": "sha256:1c75c9897514647ecf588e630c3a39e206ae05c7bb4f40138685ea13dae7778e",
        "files": {
            "LICENSE": "4f90c5fa25a2b7883d6235c7f0b11f942a37859b9f89ac36157cc485329913b4",
            "config.json": "eacc28b62075709af08bc5e7678720e160cbc27881eeeda8b7549245d3695b79",
            "model.safetensors": "b2d384f7ee3e79f971224dddb0fefbe15ba05c49ecae5030028d50a184b9b518",
        },
        "future_inputs": False,
        "input_types": ["boolean", "int64", "float64", "string", "date", "timestamp"],
        "max_classes": None,
        "max_context_rows": 100_000,
        "max_features": 32,
        "max_horizon": None,
        "memory": "8Gi",
        "missing_values": True,
        "name": "tabicl-v2-regressor",
        "outputs": ["value", "quantiles"],
        "past_inputs": False,
        "quantile_step": 0.001,
        "quantiles": [],
        "repository": "general-forecast/tabicl-v2-regressor",
        "revision": "1466a562be5d04a25ab612940d80c2c2f901c797",
        "tasks": ["regression"],
    },
}

# Model identities: artifact, variant and capability, never the serving policy.
DIGESTS = {
    "general-forecast/tabicl-v2-classifier": "sha256:493d15b4dd095d62f6c9d7415a323e0d5022211113e39e73c6879776d4e10437",
    "general-forecast/tabicl-v2-regressor": "sha256:5b381e1e102418647b8b3c7fe8561337e10446a30170147493febfc60fcfd652",
    "general-forecast/chronos-2": "sha256:8f24f588cf5e03b2c0c3154ea53dc672130dde9294dfa49726627f25dfdd36a6",
}

OLD_SPELLINGS = (
    "tabicl-classifier",
    "tabicl-v2-classifier",
    "tabicl-regressor",
    "amazon/chronos-2",
    "chronos-2",
    "catalog/chronos-2",
)


def test_pins_keep_artifact_bytes_and_separate_policy_from_identity():
    for name, expected in GOLDEN.items():
        pin = CatalogResolver().resolve(name)
        assert pin.to_dict() == {**expected, "codes": None, "estimators": None}
        expected_digest = expected["digest"]
        assert isinstance(expected_digest, str)
        assert pin.resolved.endswith(expected_digest)
        assert digest(pin.identity()) == DIGESTS[name]
        assert CatalogResolver().resolve(pin.resolved) == pin
        relaxed = dataclasses.replace(pin, cpu=8, memory="32Gi", max_context_rows=9)
        assert digest(relaxed.identity()) == DIGESTS[name]
    for spelling in OLD_SPELLINGS:
        with pytest.raises(PredictorError):
            CatalogResolver().resolve(spelling)


OFFLINE = """
import runpy, socket, sys

class Native:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"jax", "jaxlib", "safetensors", "huggingface_hub", "torch"} or (
            name == "generalforecast._core"
        ):
            raise ImportError("gf check imported " + name)

def offline(*args, **kwargs):
    raise OSError("gf check opened a connection")

class Offline(socket.socket):
    connect = connect_ex = sendto = offline

sys.meta_path.insert(0, Native())
socket.socket = Offline
socket.create_connection = socket.getaddrinfo = offline
sys.argv = ["gf", "--json", "check"]
runpy.run_module("generalforecast", run_name="__main__")
"""


def test_gf_check_is_offline_without_native_imports(tmp_path):
    import json
    import subprocess
    import sys

    (tmp_path / "gf.yaml").write_text(
        "name: fraud\nmodel: general-forecast/tabicl-v2-classifier\ntarget: y\ncontext: {snapshot: history.csv}\n"
    )
    (tmp_path / "history.csv").write_text("id,x,y\na,1,false\nb,2,true\n")
    done = subprocess.run(
        [sys.executable, "-c", OFFLINE],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    lock = json.loads(done.stdout)
    from generalforecast.models.catalog import parse

    pinned = parse("general-forecast/tabicl-v2-classifier").entry.artifact.digest
    assert lock["model_digest"] == pinned
