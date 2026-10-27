# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The test suite on an NVIDIA L4 on Modal, with the TabICL v2 checkpoints.

Run with `uvx --from modal==1.6.2 modal run .github/gpu.py` (CI installs the
SDK by hash from .github/modal-requirements.txt). Only files git
tracks are uploaded, so a local worktree, state file, key or build output never
reaches the image. The base is a pinned Rust image, and uv installs by hash. Checkpoints come from `checkpoints/` when it has them, or
else from Hugging Face with `HF_TOKEN`; without either, the tests that need
them skip.
"""

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ["tabicl-v2-classifier", "tabicl-v2-regressor"]
# Pinned by digest; rustup turns it into rust-toolchain.toml's toolchain.
RUST_BASE = (
    "rust:1.97-slim-bookworm@sha256:"
    "2775a09d208ff0d7c1f50490c45b62db929e87ba1dcbc3f2132ac71a704bcdd3"
)
UV = (
    "uv==0.8.0 --hash=sha256:"
    "fb939490ce24d5f88ddebf2ceaf30c613b2bd27f2e67ae62ec0960baa5f8412d"
)
TRACKED = frozenset(
    subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
) - {""}


def untracked(path: Path) -> bool:
    """Excludes every file git does not track; Modal passes paths relative to ROOT."""
    return path.as_posix() not in TRACKED


image = (
    modal.Image.from_registry(RUST_BASE, add_python="3.12")
    .apt_install("build-essential", "pkg-config", "git", "procps")
    .run_commands(
        f"echo '{UV}' > /tmp/uv.txt",
        "python -m pip install --no-cache-dir --require-hashes -r /tmp/uv.txt",
    )
    .add_local_dir(ROOT, "/gf", copy=True, ignore=untracked)
    .run_commands(
        # Modal's package mirror does not serve the lockfile's URLs.
        "cd /gf && env -u UV_INDEX_URL uv sync --locked --extra cuda",
        "cd /gf && cargo build --release --locked -p gf-server -p gf-router",
    )
)
for name in CHECKPOINTS:
    local = ROOT / "checkpoints" / name
    if local.exists():
        image = image.add_local_dir(local, f"/gf/checkpoints/{name}", copy=True)
    elif token := os.environ.get("HF_TOKEN"):
        image = image.run_commands(
            "cd /gf && env -u HF_ENDPOINT uv run --no-sync hf download"
            f" general-forecast/{name} --local-dir checkpoints/{name}",
            secrets=[modal.Secret.from_dict({"HF_TOKEN": token})],
        )

app = modal.App("gf-gpu")


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=7200)
def test() -> int:
    """Runs the tests on the GPU and returns pytest's exit status."""
    # Server tests start several worker processes on the one GPU.
    environment = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    command = ["uv", "run", "--no-sync", "pytest", "-q", "-rs"]
    return subprocess.run(command, cwd="/gf", env=environment, check=False).returncode


@app.local_entrypoint()
def main() -> None:
    """Fails when the tests fail, so CI does."""
    if (status := test.remote()) != 0:
        raise SystemExit(status)
