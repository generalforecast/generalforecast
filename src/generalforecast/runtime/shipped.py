# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compiled programs a release ships for a model, so a fresh machine's first call starts warm.

A release builds, on each platform, what a worker keeps for later processes:
JAX's compiled programs and the model's recorded kinds of call and measures, as JSON.
It signs each bundle, and a worker that finds none of its own installs one
only if a key in `PUBLIC_KEYS` signed it, since both hold code it runs.
"""

import hashlib
import io
import logging
import os
import re
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

logger = logging.getLogger(__name__)

REPOSITORY = "general-forecast/compiled"
# Raw Ed25519 public keys the release signs bundles with; none trusts no bundle.
PUBLIC_KEYS: tuple[bytes, ...] = (
    bytes.fromhex("ae8e074fa03af526b8c351b40da67d3087da640b0043c00b66db5d1e91414ea8"),
)

Fetch = Callable[[str], tuple[Path, Path] | None]


def name(identity: str, model: str) -> str:
    """A bundle's path in the repository: its manifests' directory, then its model spec's digest."""
    return f"{identity}/{_digest(model)}.tar"


def _digest(model: str) -> str:
    return hashlib.sha256(model.encode()).hexdigest()[:32]


def _allowed(member: str, identity: str) -> bool:
    return bool(
        re.fullmatch(r"jax/[\w.-]+", member)
        or re.fullmatch(
            rf"programs/{re.escape(identity)}/[0-9a-f]{{32}}\.(json|measures)", member
        )
    )


def bundle(
    root: Path, identity: str, model: str, key: Ed25519PrivateKey, out: Path
) -> Path:
    """Writes and signs a model's bundle from a cache root one process filled; returns the bundle."""
    return _bundle(root, identity, _digest(model), key, out)


def _bundle(
    root: Path, identity: str, digest: str, key: Ed25519PrivateKey, out: Path
) -> Path:
    members: Iterable[Path] = [
        *sorted((root / "jax").glob("*")),
        *(
            root / "programs" / identity / f"{digest}{suffix}"
            for suffix in (".json", ".measures")
        ),
    ]
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        for path in members:
            member = path.relative_to(root).as_posix()
            if path.is_file() and _allowed(member, identity):
                tar.add(path, member, recursive=False)
    target = out / identity / f"{digest}.tar"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data.getvalue())
    target.with_suffix(".tar.sig").write_bytes(key.sign(data.getvalue()))
    return target


def _hub(path: str) -> tuple[Path, Path] | None:
    from huggingface_hub import hf_hub_download

    try:
        bundle = Path(hf_hub_download(REPOSITORY, path))
        return bundle, Path(hf_hub_download(REPOSITORY, f"{path}.sig"))
    except Exception:  # noqa: BLE001
        # A platform without a bundle, or no network: the worker compiles.
        return None


def install(
    root: Path,
    identity: str,
    model: str,
    keys: Iterable[bytes] | None = None,
    fetch: Fetch | None = None,
) -> bool:
    """Installs a model's signed bundle into a cache root, keeping files already there; whether it did.

    `keys` are `PUBLIC_KEYS` and `fetch` downloads from `REPOSITORY` if not
    given. `GF_PRECOMPILED=0` turns it off, so the worker compiles.
    """
    keys = tuple(PUBLIC_KEYS if keys is None else keys)
    if not keys or os.environ.get("GF_PRECOMPILED") == "0":
        return False
    fetched = (fetch or _hub)(name(identity, model))
    if fetched is None:
        return False
    data, signature = (path.read_bytes() for path in fetched)
    if not any(_signed(key, signature, data) for key in keys):
        logger.warning("ignoring compiled programs whose signature does not verify")
        return False
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        members = tar.getmembers()
        if not all(m.isfile() and _allowed(m.name, identity) for m in members):
            logger.warning("ignoring compiled programs with unexpected files")
            return False
        for member in members:
            target = root / member.name
            if target.exists():
                continue
            source = tar.extractfile(member)
            assert source is not None
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(f"{target.name}.{os.getpid()}.part")
            part.write_bytes(source.read())
            part.replace(target)
    return True


def _signed(key: bytes, signature: bytes, data: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(signature, data)
    except (InvalidSignature, ValueError):
        return False
    return True


# Table widths a bundle compiles for. Each takes about 3 s on its first use
# instead of about 20 s; those in `STARTED` compile as the model loads, for
# a first call of under 1 s, but each adds about 3 s to every load.
WIDTHS = (4, 8, 10, 16, 20, 30, 32, 50, 64)
STARTED = (10,)


def warm(task: str, widths: Iterable[int]) -> None:
    """Fits and predicts a released model through its estimator, at 100 and 1,000 rows of each width."""
    import numpy as np

    from generalforecast import Classifier, Regressor

    rng = np.random.default_rng(0)
    estimator = (Classifier if task == "classifier" else Regressor)(download=True)
    for rows in (100, 1000):
        for width in widths:
            data = rng.normal(size=(rows + rows // 10, width))
            targets = data.sum(1)
            if task == "classifier":
                targets = (targets > 0).astype(int)
            estimator.fit(data[:rows], targets[:rows])
            estimator.predict(data[rows:])


def build(
    out: Path,
    key: Ed25519PrivateKey,
    widths: Iterable[int] = WIDTHS,
    started: Iterable[int] = STARTED,
) -> list[Path]:
    """Compiles each released model for `widths` in an empty cache, records only `started` for loads to compile, and signs a bundle for each; returns the bundles."""
    import subprocess
    import sys
    import tempfile

    bundles = []
    # A cache each, so a bundle holds only its own model's programs.
    for task in ("classifier", "regressor"):
        root = Path(tempfile.mkdtemp())
        environment = {
            **os.environ,
            "XDG_CACHE_HOME": str(root),
            "GF_PRECOMPILED": "0",
        }

        def run(
            chosen: Iterable[int], task: str = task, environment: dict = environment
        ) -> None:
            code = (
                "from generalforecast.runtime.shipped import warm; "
                f"warm({task!r}, {list(chosen)})"
            )
            subprocess.run([sys.executable, "-c", code], env=environment, check=True)

        for width in widths:
            # A process each: one compiling every width's programs aborted its
            # worker on an 8-core, 31 GiB machine.
            run([width])
        programs = root / "generalforecast" / "programs"
        for manifest in programs.glob("*/*.json"):
            manifest.unlink()
        run(started)
        bundles += [
            _bundle(
                root / "generalforecast", manifest.parent.name, manifest.stem, key, out
            )
            for manifest in sorted(programs.glob("*/*.json"))
        ]
    return bundles


# Widths `measure` measures each model at by default: every padded width to
# 512 columns, which `padded_width` maps a table's width to.
MEASURED_WIDTHS = tuple(range(1, 513))
# Class counts a classifier's tables are checked at by default.
MEASURED_CLASSES = tuple(range(2, 11))


def measure(
    out: Path,
    models: Iterable[str],
    widths: Iterable[int] = MEASURED_WIDTHS,
    classes: Iterable[int] = MEASURED_CLASSES,
    bucket_minimum: int = 8,
    slot_bytes: int = 1 << 20,
) -> list[Path]:
    """Measures each model as a worker of gf-server would, at each padded width, and writes the measures to ship.

    The measures are what a worker takes by tracing and compiling the model
    before a table's first steps: its cache layout and sizes, its fit,
    encode and decode, and its check of each task. They follow from the
    model, the code that traces it, JAX and the device, so they are written
    under the `_manifests` identity of this machine's programs, as
    `<out>/<identity>/<digest>.measures`, which a worker given `out` as
    `GF_MEASURES` reads in place of measuring.

    Returns:
        The measures files written.
    """
    import shutil
    import tempfile

    from generalforecast.constants import MEASURED_ROWS
    from generalforecast.models import registry
    from generalforecast.runtime.device import cache_root, default_settings
    from generalforecast.runtime.programs import ladder
    from generalforecast.worker import Worker, _manifests
    from generalforecast.worker.ahead import SHIPPED_MEASURES_ENV

    environment = {SHIPPED_MEASURES_ENV: None, "XDG_CACHE_HOME": tempfile.mkdtemp()}
    previous = {name: os.environ.get(name) for name in environment}
    try:
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        worker = Worker(default_settings(), bucket_minimum, slot_bytes)
        identity = _manifests(bucket_minimum, slot_bytes).name
        written = []
        for model, spec in enumerate(models):
            worker.add_model(model, *registry.load_engine_model(spec), spec=spec)
            properties = worker.properties(model)
            tasks = [0] if properties.regression else list(classes)
            worker.decode_profile(model)
            small = [
                rows for rows in ladder(bucket_minimum, 32) if rows < MEASURED_ROWS
            ]
            for width in sorted({worker.width(model, width) for width in widths}):
                worker.layout(model, width)
                worker.profile(model, width)
                worker.fit_rows(model, width)
                for task in tasks:
                    worker.check(model, task, width)
                for fit in (False, True) if properties.column_summary else (False,):
                    worker.encode_profile(model, width, fit)
                    for rows in small:
                        worker.encode_bytes(model, width, rows, fit)
            kept = worker.programs.manifest(spec)
            assert kept is not None
            measures = kept.path.with_suffix(".measures")
            target = out / identity / measures.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(measures, target)
            written.append(target)
        worker.close()
        shutil.rmtree(cache_root(), ignore_errors=True)
        return written
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def built(models: int = 2) -> bool:
    """Whether `REPOSITORY` holds a bundle of each released model for this machine's programs, as gf-server's default padding lays them out."""
    from huggingface_hub import HfApi

    from generalforecast.runtime.device import portable_cpu_programs

    portable_cpu_programs()
    from generalforecast.worker import _manifests

    identity = _manifests(bucket_minimum=8, slot_bytes=1 << 20).name
    files = HfApi().list_repo_files(REPOSITORY)
    return (
        sum(f.startswith(f"{identity}/") and f.endswith(".tar") for f in files)
        >= models
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["measures"]:
        # python -m generalforecast.runtime.shipped measures OUT SPEC... [--widths N,...]
        arguments = sys.argv[2:]
        chosen = MEASURED_WIDTHS
        if "--widths" in arguments:
            at = arguments.index("--widths")
            chosen = tuple(int(width) for width in arguments[at + 1].split(","))
            del arguments[at : at + 2]
        for written in measure(Path(arguments[0]), arguments[1:], chosen):
            print(written)
        sys.exit()

    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    if built():
        print("this release's programs are already built")
        sys.exit()
    signer = load_pem_private_key(os.environ["GF_COMPILED_SIGNING_KEY"].encode(), None)
    assert isinstance(signer, Ed25519PrivateKey)
    for written in build(Path(sys.argv[1]), signer):
        print(written)
