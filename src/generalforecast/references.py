# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a predictor reference names: a local directory, a catalog model, a served URL or a hosted name."""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path
from typing import Any

import pyarrow.compute as _compute

from generalforecast.errors import (
    AmbiguousReference,
    InvalidInput,
    NotFound,
)
from generalforecast.native import models as registry
from generalforecast.predictors.validation import NAME

pc: Any = _compute


def resolve_reference(ref: Any, *, env: str | None = None) -> tuple[str, Any]:
    """What a reference names: `("local", path)`, `("catalog", id)`, `("served", (base, name))` or `("hosted", name)`.

    An environment names a hosted predictor, so a bare name with one is hosted
    even where a directory of that name holds a `gf.yaml`.

    Raises:
        AmbiguousReference: When a bare name is both a directory with
            `gf.yaml` and so a hosted name, and no environment says which.
    """
    if isinstance(ref, os.PathLike):
        return "local", _local_root(os.fspath(ref))
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidInput(
            "a predictor reference must be a path, a name, a model id or a URL"
        )
    text = ref.strip()
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 1 or parsed.query or parsed.fragment:
            raise InvalidInput(
                f"`{text}` names no predictor",
                hint="use http://host:port/<predictor>",
            )
        base = f"{parsed.scheme}://{parsed.netloc}"
        return "served", (base, parts[0])
    if registry.is_catalog_id(text) and registry.known_org(text):
        return "catalog", registry.canonical(text)
    if registry.is_bare_name(text):
        registry.canonical(text)  # a bare model name: rejected with the canonical id
    has_separator = "/" in text or os.sep in text or text.startswith(".")
    is_dir = (Path(text) / "gf.yaml").is_file()
    if has_separator:
        return "local", _local_root(text)
    if is_dir and env is None:
        raise AmbiguousReference(
            f"`{text}` is both a directory with gf.yaml and a hosted predictor name",
            hint=f"pass ./{text} for the local predictor, or env= for the hosted one",
        )
    if not NAME.fullmatch(text):
        raise InvalidInput(
            f"`{text}` is not a predictor name, path, model id or URL",
            hint="names are lowercase letters, digits and dashes",
        )
    return "hosted", text


def _local_root(path: str) -> Path:
    root = Path(path).expanduser()
    if root.is_file() and root.name == "gf.yaml":
        root = root.parent
    if not root.is_dir():
        raise NotFound(f"no directory at `{path}`")
    if not (root / "gf.yaml").is_file():
        raise NotFound(
            f"no gf.yaml in `{path}`", hint="run `gf init` there to write one"
        )
    return root
