# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import csv
import random
from pathlib import Path

import pytest

MODEL = "general-forecast/tabicl-v2-classifier"
REGRESSOR = "general-forecast/tabicl-v2-regressor"
FORECASTER = "general-forecast/chronos-2"


def write_context(root: Path, rows: int = 200, *, empty_every: int = 10) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    draw = random.Random(1)
    path = root / "context.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["customer_id", "spend", "plan", "signups", "signed_up", "churned", "note"]
        )
        for index in range(rows):
            writer.writerow(
                [
                    1000 + index,
                    round(draw.random() * 100, 2),
                    draw.choice(["free", "pro", "team"]),
                    draw.randint(0, 9),
                    f"2025-{draw.randint(1, 9):02d}-{draw.randint(10, 28)}",
                    "" if empty_every and index % empty_every == 0 else index % 2,
                    f"n{index}",
                ]
            )
    return path


def write_manifest(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "gf.yaml"
    path.write_text(text)
    return path


@pytest.fixture
def churn(tmp_path: Path) -> Path:
    root = tmp_path / "churn"
    write_context(root)
    write_manifest(
        root,
        f"model: {MODEL}\ntarget: churned\nid: customer_id\n"
        "inputs:\n  exclude: [note]\ncontext:\n  snapshot: context.csv\n",
    )
    return root
