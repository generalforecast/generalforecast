import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from generalforecast.deployments import catalog_artifacts
from generalforecast.deployments.catalog import CatalogService
from generalforecast.deployments.catalog_artifacts import preload, verify
from generalforecast.deployments.contracts import ServiceError
from generalforecast.models import artifacts, catalog
from generalforecast.models.artifacts import ArtifactError

ENTRY = catalog.ENTRIES["tabicl-v2-classifier"]


@pytest.mark.parametrize("online", [False, True])
def test_checkpoint_access_is_explicitly_configured(online, monkeypatch):
    import huggingface_hub

    calls = []

    def download(*args, local_files_only, **kwargs):
        calls.append((args, {**kwargs, "local_files_only": local_files_only}))
        if local_files_only:
            raise LocalEntryNotFoundError("not cached")
        raise OSError("network down")

    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(ArtifactError) as caught:
        artifacts.checkpoint(ENTRY, surface="remote", allow_downloads=online)
    assert caught.value.code == "ModelUnavailable"
    assert [call[1]["local_files_only"] for call in calls] == [True, False][
        : 1 + online
    ]
    assert len(calls[0][1]["revision"]) == 40
    assert calls[0][1]["allow_patterns"] == list(ENTRY.artifact.files)


def test_content_verification_rejects_substitution_and_missing_license(tmp_path):
    files = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in ["config.json", "model.safetensors", "LICENSE"]
    }
    for name in files:
        (tmp_path / name).write_text(name)
    verify(tmp_path, files)
    (tmp_path / "model.safetensors").write_text("substituted weights")
    with pytest.raises(ServiceError) as caught:
        verify(tmp_path, files)
    assert caught.value.code == "ArtifactMismatch"
    (tmp_path / "model.safetensors").write_text("model.safetensors")
    (tmp_path / "LICENSE").unlink()
    with pytest.raises(ServiceError) as caught:
        verify(tmp_path, files)
    assert caught.value.code == "ModelUnavailable"


def test_preload_verifies_every_remote_qualified_model(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        artifacts,
        "checkpoint",
        lambda entry, **kwargs: loaded.append((entry.name, kwargs)),
    )
    preload()
    assert loaded == [
        (name, {"surface": "remote", "allow_downloads": True})
        for name in ("tabicl-v2-classifier", "tabicl-v2-regressor", "chronos-2")
    ]


def test_preload_rejects_corrupted_download_before_fetching_another_pin(
    monkeypatch, tmp_path
):
    import huggingface_hub

    calls = []
    for filename in ENTRY.artifact.files:
        (tmp_path / filename).write_bytes(b"corrupted")

    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(ArtifactError) as caught:
        preload()
    assert caught.value.code == "ArtifactMismatch"
    assert len(calls) == 1


def test_download_failures_are_typed_and_redacted(monkeypatch):
    import huggingface_hub

    def refuse(*_args, **_kwargs):
        raise RuntimeError("private-signed-url")

    monkeypatch.delenv("GF_CHECKPOINTS", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    with pytest.raises(ServiceError) as caught:
        CatalogService(allow_downloads=True)._predict(
            catalog.parse("tabicl-classifier"), [[0]], [0], [[1]], 2
        )
    assert (caught.value.code, caught.value.status) == ("ModelUnavailable", 503)
    assert "private-signed-url" not in str(caught.value)


@pytest.mark.parametrize(
    "kind",
    ["shape", "finite", "normalization", "negative", "quantile-order", "exception"],
)
def test_native_output_validation(kind, monkeypatch):
    regression = kind == "quantile-order"
    functions = SimpleNamespace(properties=lambda _: SimpleNamespace(num_outputs=999))
    monkeypatch.setattr(
        "generalforecast.models.registry.load_ref", lambda *_, **__: (functions, None)
    )
    monkeypatch.setattr("generalforecast.models.inference.fit", lambda *_: (None, None))

    def run(*_):
        if kind == "exception":
            raise RuntimeError("private-context")
        return {
            "shape": np.ones((2, 2)),
            "finite": np.array([[np.nan, 0.5]]),
            "normalization": np.array([[0.2, 0.2]]),
            "negative": np.array([[-0.1, 1.1]]),
            "quantile-order": -np.arange(999).reshape(1, 999),
        }[kind]

    monkeypatch.setattr(
        "generalforecast.models.inference._tabular", lambda _: (None, run)
    )
    model = "tabicl-regressor" if regression else "tabicl-classifier"
    with pytest.raises(ServiceError) as caught:
        CatalogService()._predict(
            catalog.parse(model), [[0], [1]], [0, 1], [[2]], 0 if regression else 2
        )
    assert caught.value.code == (
        "InferenceFailed" if kind == "exception" else "ModelOutputError"
    )
    assert "private-context" not in str(caught.value)


def test_catalog_artifacts_module_has_no_private_loaders():
    assert not hasattr(catalog_artifacts, "_model")
