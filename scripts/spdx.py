# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Check and add SPDX headers to tracked source files."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

COMMENT_PREFIXES = {
    ".py": "# ",
    ".pyi": "# ",
    ".sh": "# ",
    ".tf": "# ",
    ".hcl": "# ",
    ".rs": "// ",
    ".js": "// ",
    ".mjs": "// ",
}
CSS_HEADER = (
    b"/* SPDX-FileCopyrightText: General Forecast, Inc. */",
    b"/* SPDX-License-Identifier: Apache-2.0 */",
)
EXTENSIONS = {*COMMENT_PREFIXES, ".css"}
REPOSITORY = Path(__file__).resolve().parents[1]


def tracked_files() -> list[Path]:
    """Return tracked files with supported source extensions."""
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPOSITORY).split(
        b"\0"
    )
    files = [REPOSITORY / os.fsdecode(path) for path in paths if path]
    brand_assets = {
        receipt_path.parent / name
        for receipt_path in files
        if receipt_path.name == "brand.json"
        for name in json.loads(receipt_path.read_text())["files"]
    }
    return [
        path
        for path in files
        if path not in brand_assets
        and path.suffix in EXTENSIONS
        and path.name != ".terraform.lock.hcl"
    ]


def header_for(path: Path, newline: bytes) -> bytes:
    """Return the two comment lines for a source file."""
    if path.suffix == ".css":
        lines = CSS_HEADER
    else:
        prefix = COMMENT_PREFIXES[path.suffix].encode()
        lines = (
            prefix + b"SPDX-FileCopyrightText: General Forecast, Inc.",
            prefix + b"SPDX-License-Identifier: Apache-2.0",
        )
    return newline.join(lines) + newline


def with_header(path: Path, content: bytes) -> bytes:
    """Insert a missing header without changing existing content bytes."""
    if any(b"SPDX-License-Identifier:" in line for line in content.splitlines()[:5]):
        return content

    match = re.search(rb"\r\n|\r|\n", content)
    newline = match.group() if match else b"\n"
    header = header_for(path, newline)
    shebang = b""
    body = content
    if content.startswith(b"#!"):
        shebang_ending = re.search(rb"\r\n|\r|\n", content)
        if shebang_ending is None:
            shebang = content
            body = b""
        else:
            shebang = content[: shebang_ending.end()]
            body = content[shebang_ending.end() :]
        if not shebang.endswith((b"\n", b"\r")):
            shebang += newline

    separator = newline if content else b""
    return shebang + header + separator + body


def main() -> int:
    """Check source headers or add missing headers in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    offending = []
    for path in tracked_files():
        content = path.read_bytes()
        updated = with_header(path, content)
        if updated != content:
            if args.check:
                offending.append(path.relative_to(REPOSITORY))
            else:
                path.write_bytes(updated)

    if args.check:
        for path in offending:
            print(path)
        return int(bool(offending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
