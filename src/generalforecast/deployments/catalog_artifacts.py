"""Content verification shared by curated Models API adapters."""

from dataclasses import asdict
from pathlib import Path

from generalforecast.deployments.boundary import Artifact, JsonObject
from generalforecast.deployments.contracts import ServiceError
from generalforecast.serialization import content_digest


def artifact(
    repository: str, revision: str, license: str, files: dict[str, str]
) -> JsonObject:
    """Identifies the canonical sorted compact JSON map of file SHA-256 hashes."""
    digest = content_digest(files)
    return {
        **asdict(Artifact(repository, revision, digest, license)),
        "digest_format": "sha256-of-sorted-compact-json-file-sha256-map",
        "files": dict(files),
    }


def verify(directory: Path, files: dict[str, str]) -> None:
    """Refuses missing or substituted config, weights and license evidence."""
    from generalforecast.models.artifacts import ArtifactError
    from generalforecast.models.artifacts import verify as verified

    try:
        verified(directory, files)
    except ArtifactError as error:
        message = (
            "curated checkpoint content does not match"
            if error.code == "ArtifactMismatch"
            else "curated checkpoint files are not readable"
        )
        raise ServiceError(error.code, message, 503) from None
    except OSError:
        raise ServiceError(
            "ModelUnavailable", "curated checkpoint files are not readable", 503
        ) from None


def preload() -> None:
    """Downloads, converts and verifies every remote-qualified model for an offline image."""
    from generalforecast.models import artifacts, catalog

    for entry in catalog.entries():
        if entry.gates.allows("remote"):
            artifacts.checkpoint(entry, surface="remote", allow_downloads=True)


if __name__ == "__main__":
    preload()
