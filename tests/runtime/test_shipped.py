# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import tarfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from generalforecast.runtime import shipped

IDENTITY = "ec5390d8e4be91b0"
MODEL = "general-forecast/tabicl-v2-classifier"


def _raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def _filled(root: Path) -> None:
    digest = hashlib.sha256(MODEL.encode()).hexdigest()[:32]
    (root / "jax").mkdir(parents=True)
    (root / "jax" / "jit_fit-abc-cache").write_bytes(b"program")
    programs = root / "programs" / IDENTITY
    programs.mkdir(parents=True)
    (programs / f"{digest}.json").write_bytes(b"kinds")
    (programs / f"{digest}.measures").write_bytes(b"measures")
    (programs / f"{digest}.lock").write_bytes(b"")


def _fetch(out: Path):
    return lambda path: (out / path, out / f"{path}.sig")


def test_a_signed_bundle_installs_what_one_process_kept(tmp_path) -> None:
    key = Ed25519PrivateKey.generate()
    _filled(tmp_path / "built")
    shipped.bundle(tmp_path / "built", IDENTITY, MODEL, key, tmp_path / "out")

    fresh = tmp_path / "fresh"
    assert shipped.install(
        fresh, IDENTITY, MODEL, [_raw(key)], _fetch(tmp_path / "out")
    )

    kept = sorted(
        p.relative_to(fresh).as_posix() for p in fresh.rglob("*") if p.is_file()
    )
    built = sorted(
        p.relative_to(tmp_path / "built").as_posix()
        for p in (tmp_path / "built").rglob("*")
        if p.is_file() and p.suffix != ".lock"
    )
    assert kept == built
    assert (fresh / "jax" / "jit_fit-abc-cache").read_bytes() == b"program"


def test_bundles_that_do_not_verify_install_nothing(tmp_path) -> None:
    key = Ed25519PrivateKey.generate()
    _filled(tmp_path / "built")
    out = tmp_path / "out"
    bundle = shipped.bundle(tmp_path / "built", IDENTITY, MODEL, key, out)

    assert not shipped.install(tmp_path / "a", IDENTITY, MODEL, [], _fetch(out))
    other = _raw(Ed25519PrivateKey.generate())
    assert not shipped.install(tmp_path / "b", IDENTITY, MODEL, [other], _fetch(out))
    bundle.write_bytes(bundle.read_bytes() + b"\0")
    assert not shipped.install(
        tmp_path / "c", IDENTITY, MODEL, [_raw(key)], _fetch(out)
    )
    assert not any((tmp_path / name).exists() for name in "abc")


def test_a_signed_bundle_with_a_path_outside_the_cache_installs_nothing(
    tmp_path,
) -> None:
    key = Ed25519PrivateKey.generate()
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        member = tarfile.TarInfo("jax/../../escaped")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    out = tmp_path / "out"
    path = out / shipped.name(IDENTITY, MODEL)
    path.parent.mkdir(parents=True)
    path.write_bytes(data.getvalue())
    path.with_suffix(".tar.sig").write_bytes(key.sign(data.getvalue()))

    assert not shipped.install(
        tmp_path / "root", IDENTITY, MODEL, [_raw(key)], _fetch(out)
    )
    assert not (tmp_path / "escaped").exists()


def test_gf_precompiled_0_installs_nothing(tmp_path, monkeypatch) -> None:
    key = Ed25519PrivateKey.generate()
    _filled(tmp_path / "built")
    shipped.bundle(tmp_path / "built", IDENTITY, MODEL, key, tmp_path / "out")
    monkeypatch.setenv("GF_PRECOMPILED", "0")

    assert not shipped.install(
        tmp_path / "fresh", IDENTITY, MODEL, [_raw(key)], _fetch(tmp_path / "out")
    )
    assert not (tmp_path / "fresh").exists()
