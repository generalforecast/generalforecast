# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The predictor lock, `gf.lock`: reading, writing and diffing."""

from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from generalforecast.errors import InvalidInput
from generalforecast.predictors import files
from generalforecast.predictors.resolution import (
    ContextInfo,
    Resolution,
    SelectionInfo,
)
from generalforecast.predictors.spec import PredictorSpec, load
from generalforecast.predictors.validation import PredictorError

LOCK = "gf.lock"
LOCK_VERSION = 1


def document(resolution: Resolution) -> dict[str, Any]:
    """The lock document a resolution writes."""
    predictor: dict[str, Any] = {
        "name": resolution.name,
        "model": resolution.model,
        "model_digest": resolution.model_digest,
        "task": resolution.task,
        "target": resolution.target,
        "target_type": resolution.target_type,
    }
    if resolution.classes is not None:
        predictor["classes"] = list(resolution.classes)
    if resolution.id is not None:
        predictor["id"] = resolution.id
        if resolution.id_type is not None:
            predictor["id_type"] = resolution.id_type
    if resolution.time is not None:
        predictor["time"] = resolution.time
    if resolution.series:
        predictor["series"] = list(resolution.series)
    if resolution.task == "forecast":
        predictor["frequency"] = resolution.frequency
        predictor["horizon"] = resolution.horizon
        predictor["quantiles"] = list(resolution.quantiles)
    roles = resolution.spec.roles if resolution.spec is not None else {}
    predictor["inputs"] = [
        {"name": name, "type": kind}
        | ({"role": roles[name]} if roles.get(name) in ("past", "future") else {})
        for name, kind in resolution.inputs
    ]
    context: dict[str, Any] = {"kind": resolution.context.kind}
    for key in ("digest", "rows", "query_rows"):
        if getattr(resolution.context, key) is not None:
            context[key] = getattr(resolution.context, key)
    snapshot = getattr(getattr(resolution.spec, "context", None), "snapshot", None)
    if context["kind"] == "snapshot" and snapshot is not None:
        context["path"] = str(snapshot)
    live = getattr(getattr(resolution.spec, "context", None), "live", None)
    if context["kind"] == "live" and live is not None:
        context["name"] = live
    predictor["context"] = context
    selection: dict[str, Any] = {
        "core": resolution.selection.core,
        "neighbors": resolution.selection.neighbors,
        "kind": resolution.selection.kind,
    }
    if resolution.selection.recent is not None:
        selection["recent"] = resolution.selection.recent
    predictor["selection"] = selection
    return {
        "lock_version": LOCK_VERSION,
        "predictor": predictor,
        "model": {"files": dict(resolution.files)},
    }


def read_lock(root: str | Path) -> dict[str, Any] | None:
    """The previous `gf.lock` document, or None when there is none to compare with.

    A lock of another version is treated as absent; `gf check` rewrites it.

    Raises:
        InvalidInput: If `gf.lock` exists but is not readable YAML.
    """
    try:
        data = files.read(Path(root), LOCK)
    except PredictorError as error:
        if error.code == "MissingFile":
            return None
        raise InvalidInput(
            message="cannot read gf.lock", param="lock", hint=str(error)
        ) from None
    try:
        value = files.parse_yaml(data)
    except PredictorError:
        raise InvalidInput(
            message="gf.lock is not valid YAML",
            param="lock",
            hint="delete it and run `gf check` to write it again",
        ) from None
    if (
        value.get("lock_version") != LOCK_VERSION
        or type(value.get("predictor")) is not dict
    ):
        return None
    return value


def write_lock(resolution: Resolution, root: str | Path) -> Path:
    """Writes `gf.lock` for a resolution.

    Returns:
        The written path.
    """
    root = Path(root)
    text = yaml.safe_dump(
        document(resolution), sort_keys=False, default_flow_style=False, width=88
    )
    try:
        files.atomic_write(root, LOCK, text.encode())
    except PredictorError as error:
        raise InvalidInput(
            message="cannot write gf.lock", param="lock", hint=str(error)
        ) from None
    return root / LOCK


def changes(
    previous: dict[str, Any] | None, current: dict[str, Any]
) -> tuple[tuple[str, str, Any, Any], ...]:
    """The `(op, key, old, new)` lines between two lock documents.

    `op` is `+` for an added key, `-` for a removed one and `~` for a changed
    value. Inputs diff by name: `("+", "input", None, "spend")`, `("-", "input",
    "note", None)` and `("~", "input", "spend int", "spend float")`. Other keys
    are dotted paths under `predictor`, such as `selection.core`, with raw
    values; the first lock has no previous document and reports no changes.
    """
    if previous is None:
        return ()
    before = _flatten(previous.get("predictor", {}))
    after = _flatten(current["predictor"])
    result: list[tuple[str, str, Any, Any]] = []
    for key in list(before) + [key for key in after if key not in before]:
        if key == "inputs":
            continue
        old, new = before.get(key, _ABSENT), after.get(key, _ABSENT)
        if old is _ABSENT:
            result.append(("+", key, None, new))
        elif new is _ABSENT:
            result.append(("-", key, old, None))
        elif old != new:
            result.append(("~", key, old, new))
    old_inputs = {
        item["name"]: item.get("type")
        for item in previous.get("predictor", {}).get("inputs", [])
        if type(item) is dict and "name" in item
    }
    new_inputs = {item["name"]: item["type"] for item in current["predictor"]["inputs"]}
    for name in list(old_inputs) + [
        name for name in new_inputs if name not in old_inputs
    ]:
        if name not in new_inputs:
            result.append(("-", "input", name, None))
        elif name not in old_inputs:
            result.append(("+", "input", None, name))
        elif old_inputs[name] != new_inputs[name]:
            result.append(
                (
                    "~",
                    "input",
                    f"{name} {old_inputs[name]}",
                    f"{name} {new_inputs[name]}",
                )
            )
    old_files = previous.get("model", {}).get("files")
    if old_files is not None and old_files != current["model"]["files"]:
        result.append(("~", "model.files", old_files, current["model"]["files"]))
    return tuple(result)


_ABSENT = object()


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if type(item) is dict and key != "inputs":
            flat.update(_flatten(item, path))
        else:
            flat[path] = item
    return flat


def load_lock(root: str | Path, spec: PredictorSpec | None = None) -> Resolution:
    """The resolution a `gf.lock` records, without touching the context.

    This is what `gf check --frozen` and `gf serve` read.

    Raises:
        InvalidInput: If there is no lock, or it is not a version 1 lock, or
            it does not describe the manifest's name, model and target.
    """
    root = Path(root)
    value = read_lock(root)
    if value is None:
        raise InvalidInput(
            message=f"no gf.lock in {root}"
            if not (root / LOCK).exists()
            else "gf.lock is not a version 1 lock",
            param="lock",
            hint="run `gf check` to write it",
        )
    if spec is None:
        spec = load(root)
    predictor = value["predictor"]
    for key in ("name", "model", "target"):
        if predictor.get(key) != getattr(spec, key):
            raise InvalidInput(
                message=f"gf.lock was written for `{key}: {predictor.get(key)}`, "
                f"and gf.yaml now says `{key}: {getattr(spec, key)}`",
                param="lock",
                hint="run `gf check` to update it",
            )
    try:
        inputs = tuple((item["name"], item.get("type")) for item in predictor["inputs"])
        context = predictor["context"]
        selection = predictor["selection"]
        return Resolution(
            name=predictor["name"],
            model=predictor["model"],
            model_digest=predictor.get("model_digest"),
            task=predictor["task"],
            target=predictor["target"],
            target_type=predictor.get("target_type"),
            classes=tuple(predictor["classes"]) if "classes" in predictor else None,
            id=predictor.get("id"),
            id_type=predictor.get("id_type"),
            time=predictor.get("time"),
            series=tuple(predictor.get("series", ())),
            inputs=inputs,
            inferred=MappingProxyType({}),
            context=ContextInfo(
                context["kind"],
                context.get("rows"),
                context.get("query_rows"),
                context.get("digest"),
            ),
            selection=SelectionInfo(
                selection.get("core"),
                selection["neighbors"],
                selection["kind"],
                selection.get("recent"),
            ),
            frequency=predictor.get("frequency"),
            horizon=predictor.get("horizon"),
            quantiles=tuple(predictor.get("quantiles", ())),
            files=MappingProxyType(dict(value.get("model", {}).get("files", {}))),
            spec=spec,
        )
    except (KeyError, TypeError):
        raise InvalidInput(
            message="gf.lock is missing fields",
            param="lock",
            hint="run `gf check` to write it again",
        ) from None


__all__ = [
    "LOCK",
    "LOCK_VERSION",
    "changes",
    "document",
    "load_lock",
    "read_lock",
    "write_lock",
]
