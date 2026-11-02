# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Write tests/durations.json, which `--shard` balances CI's parts by.

Each test job on main records its tests' seconds (`--record-durations`) and
uploads them as a `durations-*` artifact. Download a run's artifacts and
merge them:

    gh run download RUN --pattern 'durations-*' --dir durations
    python scripts/durations.py durations/*/*.json

A test recorded in several files keeps its longest time; tests the files do
not name keep the time tests/durations.json already gives them, unless
`--fresh` drops them.
"""

import argparse
import json
from pathlib import Path

DURATIONS = Path(__file__).resolve().parents[1] / "tests" / "durations.json"


def merge(recorded: list[dict[str, float]], kept: dict[str, float]) -> dict[str, float]:
    """The seconds `recorded` gives each test, over what `kept` gave it."""
    seconds = dict(kept)
    fresh: dict[str, float] = {}
    for record in recorded:
        for test, value in record.items():
            fresh[test] = max(fresh.get(test, 0.0), value)
    seconds.update(fresh)
    return {test: round(value, 2) for test, value in sorted(seconds.items())}


def main() -> None:
    """Merge the recorded files into tests/durations.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path, help="recorded durations")
    parser.add_argument(
        "--fresh", action="store_true", help="keep only these files' tests"
    )
    args = parser.parse_args()
    kept = (
        {}
        if args.fresh or not DURATIONS.exists()
        else json.loads(DURATIONS.read_text())
    )
    recorded = [json.loads(path.read_text()) for path in args.files]
    DURATIONS.write_text(json.dumps(merge(recorded, kept), indent=0) + "\n")


if __name__ == "__main__":
    main()
