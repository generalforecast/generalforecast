# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The predictor lock, `gf.lock`: its one format, reading, writing, diffing and digest.

`gf check` resolves `gf.yaml` against its context and the model catalog and
writes the result here. `gf check --frozen`, `gf.Predictor`, `gf serve`,
`gf push` and hosted serving all read this document, and nothing else, to
know what a predictor is. It has one format:

    lock_version: 1               # 2 when the predictor runs a derived model
    digest: sha256:...            # of the document without this key or checked_with
    checked_with: {gf, rules}     # the gf version that wrote it and its LOCK_RULES
    manifest: sha256:...          # of the normalized gf.yaml it resolves
    predictor:
      name, model, model_digest, task, target, target_type
      derived: {base, base_digest, bags, kind}   # a derived model's catalog base and
                                  # bag count, in lock_version 2 alone; then
                                  # model is `<owner>/<name>@ft<version>` and
                                  # model_digest pins its weights
      classes                     # a classifier's labels, when the context has them
      id, id_type                 # when the manifest names an id
      time, time_type             # when it names a time
      series, series_types        # a forecaster's series columns and their kinds
      related                     # a forecaster's declared groups: [] is one group,
                                  # else the series columns naming each group
      frequency, horizon          # a forecaster's
      quantiles                   # a regressor's or forecaster's output levels
      outputs                     # what predictions carry
      inputs: [{name, type, role}]    # role is input, past or future
      context: {kind, digest?, rows?, query_rows?, path?, name?, max_rows?,
                until?, holdout?: {fraction, seed}}
      selection: {strategy, core, neighbors, kind, recent, seed}
    resources: {accelerator, replicas_min, replicas_max, idle_seconds}   # when declared
    model:
      files: {path: sha256}

`digest` is the one identity of a predictor revision: hosted serving dedupes
revisions by it, gf-server fingerprints bound tables by it under `gf serve`
and hosted serving alike, and `contracts.PredictorLock.digest` is it.

`checked_with` says which gf wrote the lock: its package version, and
`rules`, the revision of what `gf check` writes for a given `gf.yaml`, rows
and catalog. Two builds of one package version can write different locks
for the same inputs, so `rules`, not the version, is what tells an
out-of-date gf from a lock edited by hand: a plane whose gf finds a pushed
lock other than what it would write, from a lock of other rules, asks for
the gf that writes its rules rather than for `gf check` again. It only
describes the writer, so `digest` leaves it out: the same predictor checked
by two gf versions is one revision. A lock without it was written before
gf recorded it, under rules 1.
"""

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
from generalforecast.predictors.validation import (
    PredictorError,
    digest,
    ensure,
    fields,
    hash_value,
    name,
    text,
)
from generalforecast.predictors.validation import column as column_name

LOCK = "gf.lock"
LOCK_VERSION = 1
# The revision of what `gf check` writes for the same gf.yaml, rows and
# catalog. Bump it with any change that makes `gf check` write a lock other
# than before for the same inputs (a model's context row cap, the engine's
# adaptive selection, a default the lock records), so a plane on the new
# rules tells a customer on an older gf to upgrade instead of to run
# `gf check`, which would write the same lock again.
#   1: what gf check wrote through 064a957, and any lock without
#      `checked_with` (gf between #76 and this record wrote rules 2 without
#      saying so; where its locks agree they are admitted all the same).
#   2: TabICL v2's context row cap of 100,000 (gf #76), which resolves a
#      selection of 60,001 to 100,000 rows as exact where it was retrieved.
LOCK_RULES = 2
# The rules of a lock that does not record them.
UNRECORDED_RULES = 1
# The version of a lock whose predictor runs a derived model; locks of
# catalog models stay at LOCK_VERSION, so none of them changes.
DERIVED_LOCK_VERSION = 2
LOCK_VERSIONS = (LOCK_VERSION, DERIVED_LOCK_VERSION)
TASKS = ("classification", "regression", "forecast")
CONTEXT_KINDS = ("snapshot", "request", "live")
ROLES = ("input", "past", "future")
PREDICTOR_FIELDS = frozenset(
    {
        "name",
        "model",
        "model_digest",
        "task",
        "target",
        "target_type",
        "outputs",
        "inputs",
        "context",
        "selection",
    }
)
OPTIONAL_FIELDS = frozenset(
    {
        "classes",
        "id",
        "id_type",
        "time",
        "time_type",
        "series",
        "series_types",
        "related",
        "frequency",
        "horizon",
        "quantiles",
        "derived",
    }
)
CONTEXT_FIELDS = frozenset(
    {"digest", "rows", "query_rows", "path", "name", "max_rows", "until", "holdout"}
)
RESOURCE_FIELDS = frozenset(
    {"accelerator", "replicas_min", "replicas_max", "idle_seconds"}
)


def lock_digest(document: dict[str, Any]) -> str:
    """The lock's one digest: of its canonical JSON without `digest` and `checked_with`."""
    return digest(
        {
            key: value
            for key, value in document.items()
            if key not in ("digest", "checked_with")
        }
    )


def checked_with() -> dict[str, Any]:
    """The `checked_with` this gf writes: its package version and `LOCK_RULES`."""
    import generalforecast

    return {"gf": generalforecast.__version__, "rules": LOCK_RULES}


def rules_of(document: dict[str, Any]) -> int:
    """The rules a lock was written under; `UNRECORDED_RULES` when it does not say."""
    recorded = document.get("checked_with")
    if type(recorded) is dict and type(recorded.get("rules")) is int:
        return recorded["rules"]
    return UNRECORDED_RULES


def other_rules(document: dict[str, Any]) -> str | None:
    """Why a lock written under other rules than this gf's differs, or None.

    A lock `gf check` wrote under older rules is out of date however often
    that gf runs it again, and one written under newer rules is past what
    this gf writes; either way the fix is the gf that writes this gf's
    rules, which the sentence names.
    """
    rules = rules_of(document)
    if rules == LOCK_RULES:
        return None
    current = checked_with()
    recorded = document.get("checked_with")
    writer = (
        f"gf {recorded['gf']} (lock rules {rules})"
        if type(recorded) is dict and type(recorded.get("gf")) is str
        else f"a gf from before locks recorded it (lock rules {rules})"
    )
    here = f"gf {current['gf']} (lock rules {LOCK_RULES})"
    if rules < LOCK_RULES:
        return (
            f"this predictor was checked with {writer}; this service runs {here}, "
            "whose gf check writes it differently; upgrade the gf SDK "
            "(`pip install -U generalforecast`) and run `gf check` again"
        )
    return (
        f"this predictor was checked with {writer}, newer than {here}, "
        "which this service runs; check it with the gf this service runs, "
        f"or push it once the service runs lock rules {rules}"
    )


def manifest_digest(spec: PredictorSpec) -> str:
    """The digest of the normalized manifest, which a lock records to notice edits."""
    return digest(spec.to_dict())


def document(resolution: Resolution) -> dict[str, Any]:
    """The lock document a resolution writes, its digest included.

    The document of a model without a published artifact digest is not a
    valid lock; `write_lock` and `validate` refuse it.
    """
    predictor: dict[str, Any] = {
        "name": resolution.name,
        "model": resolution.model,
        "model_digest": resolution.model_digest,
        "task": resolution.task,
        "target": resolution.target,
        "target_type": resolution.target_type,
    }
    if resolution.derived is not None:
        predictor["derived"] = dict(resolution.derived)
    if resolution.classes is not None:
        predictor["classes"] = list(resolution.classes)
    if resolution.id is not None:
        predictor["id"] = resolution.id
        predictor["id_type"] = resolution.id_type
    if resolution.time is not None:
        predictor["time"] = resolution.time
        predictor["time_type"] = resolution.time_type
    if resolution.series:
        predictor["series"] = list(resolution.series)
        predictor["series_types"] = list(resolution.series_types)
    if resolution.task == "forecast":
        predictor["frequency"] = resolution.frequency
        predictor["horizon"] = resolution.horizon
        if resolution.related is not None:
            predictor["related"] = list(resolution.related)
    if resolution.task != "classification":
        predictor["quantiles"] = list(resolution.quantiles)
    predictor["outputs"] = list(resolution.outputs)
    predictor["inputs"] = [
        {"name": column, "type": kind, "role": role}
        for (column, kind), role in zip(
            resolution.inputs, resolution.input_roles, strict=True
        )
    ]
    context: dict[str, Any] = {"kind": resolution.context.kind}
    for key in ("digest", "rows", "query_rows"):
        if getattr(resolution.context, key) is not None:
            context[key] = getattr(resolution.context, key)
    spec = resolution.spec
    if spec is not None:
        if context["kind"] == "snapshot":
            context["path"] = str(spec.context.snapshot)
            if spec.context.until is not None:
                context["until"] = spec.context.until
            if spec.context.holdout is not None:
                context["holdout"] = {
                    "fraction": spec.context.holdout.fraction,
                    "seed": spec.context.holdout.seed,
                }
        elif context["kind"] == "live":
            context["name"] = spec.context.live
        elif spec.context.max_rows is not None:
            context["max_rows"] = spec.context.max_rows
    predictor["context"] = context
    predictor["selection"] = resolution.selection.to_dict()
    version = LOCK_VERSION if resolution.derived is None else DERIVED_LOCK_VERSION
    value: dict[str, Any] = {
        "lock_version": version,
        "checked_with": checked_with(),
        "manifest": resolution.manifest,
        "predictor": predictor,
    }
    resources = resolution.resources
    if resources:
        value["resources"] = dict(resources)
    value["model"] = {"files": dict(resolution.files)}
    return {"lock_version": version, "digest": lock_digest(value), **value}


def validate(value: Any) -> dict[str, Any]:
    """Checks a lock document field by field, its digest included.

    Returns:
        The document.

    Raises:
        PredictorError: `UnsupportedLockVersion` for a document of another
            format, with the hint to run `gf check`; `LockMismatch` for a digest that does
            not match the content; `ValidationError` for a field out of shape.
    """
    ensure(type(value) is dict, "lock must be a document")
    version = value.get("lock_version")
    ensure(
        type(version) is int and version in LOCK_VERSIONS,
        "gf.lock is not in the current format; run `gf check` to write it again",
        "UnsupportedLockVersion",
    )
    fields(
        value,
        {"lock_version", "digest", "manifest", "predictor", "model"},
        {"resources", "checked_with"},
    )
    if "checked_with" in value:
        recorded = fields(value["checked_with"], {"gf", "rules"})
        text(recorded["gf"])
        ensure(
            type(recorded["rules"]) is int and recorded["rules"] >= 1,
            "checked_with.rules is a whole number from 1",
        )
    hash_value(value["digest"])
    ensure(
        hash_value(value["manifest"]),
        "invalid manifest digest",
    )
    ensure(
        lock_digest(value) == value["digest"],
        "gf.lock digest does not match its content; run `gf check` to write it again",
        "LockMismatch",
    )
    predictor = fields(value["predictor"], set(PREDICTOR_FIELDS), OPTIONAL_FIELDS)
    name(predictor["name"])
    text(predictor["model"])
    ensure(
        ("derived" in predictor) == (version == DERIVED_LOCK_VERSION),
        f"a lock with a derived model is lock_version {DERIVED_LOCK_VERSION}, "
        f"and only it has `derived`",
        "ValidationError",
    )
    if "derived" in predictor:
        _derived(predictor["model"], predictor["derived"])
    else:
        ensure(
            predictor["model"].startswith("general-forecast/"),
            f"model `{predictor['model']}` is not a catalog id; use general-forecast/<name>",
            "ValidationError",
        )
    ensure(
        predictor["model_digest"] is not None,
        f"{predictor['model']} has no published artifact digest, so a lock cannot pin it",
        "ModelResolutionRequired",
    )
    hash_value(predictor["model_digest"])
    ensure(predictor["task"] in TASKS, "invalid predictor task")
    column_name(predictor["target"])
    _kind(predictor["target_type"])
    task = predictor["task"]
    for key in ("id", "time"):
        if key in predictor:
            column_name(predictor[key])
            ensure(
                f"{key}_type" in predictor,
                f"a lock with an {key} records its {key}_type",
            )
            _kind(predictor[f"{key}_type"])
    if "classes" in predictor:
        classes = predictor["classes"]
        ensure(
            task == "classification"
            and type(classes) is list
            and len(classes) >= 2
            and all(type(label) in (str, int, bool) for label in classes)
            and len(set(map(repr, classes))) == len(classes),
            "a classifier's classes are distinct string, integer or boolean labels",
        )
    if "series" in predictor:
        ensure(
            type(predictor["series"]) is list
            and type(predictor.get("series_types")) is list
            and len(predictor["series"]) == len(predictor["series_types"]),
            "series and series_types are aligned lists",
        )
        for column, kind in zip(
            predictor["series"], predictor["series_types"], strict=True
        ):
            column_name(column)
            _kind(kind)
    if task == "forecast":
        ensure(
            "frequency" in predictor and "horizon" in predictor and "time" in predictor,
            "a forecaster's lock has time, frequency and horizon",
        )
        related = predictor.get("related", [])
        ensure(
            type(related) is list
            and set(related) <= set(predictor.get("series", []))
            and len(set(related)) == len(related),
            "`related` names series columns",
        )
    else:
        ensure(
            not {"frequency", "horizon", "series", "related"} & predictor.keys(),
            "only a forecaster's lock has frequency, horizon, series and related",
        )
    ensure(
        ("quantiles" in predictor) == (task != "classification"),
        "a regressor's or forecaster's lock, and only theirs, has quantiles",
    )
    if "quantiles" in predictor:
        ensure(
            type(predictor["quantiles"]) is list
            and all(
                type(q) in (int, float) and 0 < q < 1 for q in predictor["quantiles"]
            ),
            "quantiles are levels between 0 and 1",
        )
    outputs = predictor["outputs"]
    allowed = (
        ("label", "probabilities")
        if task == "classification"
        else ("value", "quantiles")
    )
    ensure(
        type(outputs) is list
        and bool(outputs)
        and all(type(o) is str and o in allowed for o in outputs)
        and len(set(outputs)) == len(outputs),
        f"a {task} lock's outputs are distinct names among {', '.join(allowed)}",
    )
    inputs = predictor["inputs"]
    ensure(type(inputs) is list, "lock inputs must list {name, type, role} columns")
    ensure(
        len({column.get("name") for column in inputs if type(column) is dict})
        == len(inputs),
        "lock inputs name each column once",
    )
    for column in inputs:
        fields(column, {"name", "type", "role"})
        column_name(column["name"])
        if column["type"] is not None:
            _kind(column["type"])
        ensure(
            column["role"] == "input"
            if task != "forecast"
            else column["role"] in ("past", "future"),
            "a tabular input's role is input; a forecaster's is past or future",
        )
    context = fields(predictor["context"], {"kind"}, CONTEXT_FIELDS)
    ensure(context["kind"] in CONTEXT_KINDS, "invalid context kind")
    if context["kind"] == "snapshot":
        hash_value(context.get("digest"))
        ensure(
            type(context.get("rows")) is int and context["rows"] > 0,
            "a snapshot lock counts its context rows",
        )
        files.relative(context.get("path"))
        if "holdout" in context:
            holdout = fields(context["holdout"], {"fraction", "seed"})
            ensure(
                type(holdout["fraction"]) in (int, float)
                and 0 < holdout["fraction"] < 1
                and type(holdout["seed"]) is int
                and 0 <= holdout["seed"] < 2**32,
                "invalid holdout fraction or seed",
            )
        if "until" in context:
            ensure(_iso(context["until"]), "`until` is an ISO date or timestamp")
        ensure(
            not ("until" in context and "holdout" in context),
            "`until` and `holdout` cannot both be set",
        )
    else:
        ensure(
            not {"digest", "path", "until", "holdout"} & context.keys(),
            "only a snapshot lock has a digest, path, until or holdout",
        )
    if context["kind"] == "live":
        name(context.get("name"))
    if "max_rows" in context:
        ensure(
            context["kind"] == "request"
            and type(context["max_rows"]) is int
            and context["max_rows"] > 0,
            "max_rows bounds a request context",
        )
    try:
        SelectionInfo.from_dict(predictor["selection"])
    except InvalidInput as error:
        raise PredictorError("ValidationError", error.message) from None
    if "resources" in value:
        fields(value["resources"], set(), RESOURCE_FIELDS)
    model = fields(value["model"], {"files"})
    ensure(
        type(model["files"]) is dict
        and all(type(path) is str for path in model["files"]),
        "model files map paths to digests",
    )
    return value


def _derived(model: str, value: Any) -> None:
    """Checks a derived model's id and its block: base, base digest, bags, kind."""
    from generalforecast.models.catalog import ModelRefError, parse
    from generalforecast.models.derived import KINDS, LEGACY_KINDS

    try:
        ref = parse(model)
    except ModelRefError as error:
        raise PredictorError("ValidationError", str(error)) from None
    ensure(
        ref.derived and not ref.variant and ref.digest is None,
        f"model `{model}` is not a derived id, <owner>/<name>@ft<version>",
        "ValidationError",
    )
    fields(value, {"base", "base_digest", "bags", "kind"})
    text(value["base"])
    ensure(
        value["base"].startswith("general-forecast/"),
        "a derived model's base is a catalog id",
        "ValidationError",
    )
    hash_value(value["base_digest"])
    ensure(
        type(value["bags"]) is int and 1 <= value["bags"] <= 64,
        "a derived model has 1 to 64 bags",
        "ValidationError",
    )
    # A lock an earlier gf wrote says `adapter` for an input adapter.
    ensure(
        value["kind"] in KINDS or value["kind"] in LEGACY_KINDS,
        "a derived model's kind is lora, input_adapter or full",
        "ValidationError",
    )


def _iso(value: Any) -> bool:
    import datetime as dt

    if type(value) is not str:
        return False
    try:
        dt.datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _kind(value: Any) -> None:
    from generalforecast.predictors.tables import PHYSICAL

    ensure(value is None or value in PHYSICAL, "invalid column kind")


def read_lock(root: str | Path) -> dict[str, Any] | None:
    """The previous `gf.lock` document, or None when there is no current lock to compare with.

    `gf check` rewrites a lock of another format.

    Raises:
        InvalidInput: If `gf.lock` exists but is not readable YAML.
    """
    data = _read(Path(root))
    if data is None:
        return None
    value = _parsed(data)
    if (
        value.get("lock_version") not in LOCK_VERSIONS
        or type(value.get("predictor")) is not dict
    ):
        return None
    return value


def _read(root: Path) -> bytes | None:
    try:
        return files.read(root, LOCK)
    except PredictorError as error:
        if error.code == "MissingFile":
            return None
        raise InvalidInput(
            message="cannot read gf.lock", param="lock", hint=str(error)
        ) from None


def _parsed(data: bytes) -> dict[str, Any]:
    try:
        return files.parse_yaml(data)
    except PredictorError:
        raise InvalidInput(
            message="gf.lock is not valid YAML",
            param="lock",
            hint="delete it and run `gf check` to write it again",
        ) from None


def lock_document(root: str | Path) -> dict[str, Any]:
    """The validated `gf.lock` in a predictor directory.

    Raises:
        InvalidInput: For no lock, a lock of another version (with the hint
            to run `gf check`), or a lock out of shape.
    """
    root = Path(root)
    data = _read(root)
    if data is None:
        raise InvalidInput(
            message=f"no gf.lock in {root}",
            param="lock",
            hint="run `gf check` to write it",
        )
    try:
        return validate(_parsed(data))
    except PredictorError as error:
        raise InvalidInput(
            message=error.message,
            param="lock",
            hint="run `gf check` to write it again",
        ) from None


def write_lock(resolution: Resolution, root: str | Path) -> Path:
    """Writes `gf.lock` for a resolution.

    Returns:
        The written path.

    Raises:
        InvalidInput: For a resolution no lock can pin, such as a model with
            no published artifact digest, or an unwritable directory.
    """
    root = Path(root)
    value = document(resolution)
    try:
        validate(value)
    except PredictorError as error:
        raise InvalidInput(
            message=error.message,
            param="model" if error.code == "ModelResolutionRequired" else "lock",
            hint="pick a model `gf model list` shows as pinned"
            if error.code == "ModelResolutionRequired"
            else None,
        ) from None
    text = yaml.safe_dump(value, sort_keys=False, default_flow_style=False, width=88)
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
    are dotted paths under `predictor` (or `resources.` and `manifest`), such
    as `selection.core`, with raw values; the first lock has no previous
    document and reports no changes.
    """
    if previous is None:
        return ()
    before = _flatten(_comparable(previous))
    after = _flatten(_comparable(current))
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
        item["name"]: _shown_input(item)
        for item in previous.get("predictor", {}).get("inputs", [])
        if type(item) is dict and "name" in item
    }
    new_inputs = {
        item["name"]: _shown_input(item) for item in current["predictor"]["inputs"]
    }
    for column in list(old_inputs) + [c for c in new_inputs if c not in old_inputs]:
        if column not in new_inputs:
            result.append(("-", "input", column, None))
        elif column not in old_inputs:
            result.append(("+", "input", None, column))
        elif old_inputs[column] != new_inputs[column]:
            result.append(
                (
                    "~",
                    "input",
                    f"{column} {old_inputs[column]}",
                    f"{column} {new_inputs[column]}",
                )
            )
    old_files = previous.get("model", {}).get("files")
    if old_files is not None and old_files != current["model"]["files"]:
        result.append(("~", "model.files", old_files, current["model"]["files"]))
    return tuple(result)


def _shown_input(item: dict[str, Any]) -> str:
    role = item.get("role")
    kind = item.get("type")
    return f"{kind} {role}" if role in ("past", "future") else str(kind)


def _comparable(value: dict[str, Any]) -> dict[str, Any]:
    """The keys a diff walks: the predictor's, with resources and the manifest beside them."""
    out = dict(value.get("predictor", {}))
    if "resources" in value:
        out["resources"] = value["resources"]
    if value.get("manifest") is not None:
        out["manifest"] = value["manifest"]
    return out


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

    A library read of what a lock records, for tools that describe one; no
    entry point predicts from it. `gf check --frozen` and `--locked`,
    `gf.Predictor`, `gf serve` and `gf push` hold a directory's lock to
    `resolution.verified`, which also resolves the context. The lock must
    describe the manifest beside it exactly: it records the digest of the
    normalized `gf.yaml`, so any edit that could change a prediction
    (inputs, types, split, selection and its seed, frequency, quantiles,
    resources) makes it stale.

    Raises:
        InvalidInput: If there is no lock, it is not in the current format, or it
            was written for another manifest.
    """
    root = Path(root)
    value = lock_document(root)
    if spec is None:
        spec = load(root)
    if value["manifest"] != manifest_digest(spec):
        raise InvalidInput(
            message="gf.lock was written for another gf.yaml",
            param="lock",
            because="gf.yaml changed since `gf check` wrote gf.lock",
            hint="run `gf check` to update it",
        )
    return resolution_of(value, spec)


def resolution_of(value: dict[str, Any], spec: PredictorSpec | None) -> Resolution:
    """The `Resolution` a validated lock document records."""
    predictor = value["predictor"]
    context = predictor["context"]
    return Resolution(
        name=predictor["name"],
        model=predictor["model"],
        model_digest=predictor["model_digest"],
        derived=MappingProxyType(dict(predictor["derived"]))
        if "derived" in predictor
        else None,
        task=predictor["task"],
        target=predictor["target"],
        target_type=predictor.get("target_type"),
        classes=tuple(predictor["classes"]) if "classes" in predictor else None,
        id=predictor.get("id"),
        id_type=predictor.get("id_type"),
        time=predictor.get("time"),
        time_type=predictor.get("time_type"),
        series=tuple(predictor.get("series", ())),
        series_types=tuple(predictor.get("series_types", ())),
        related=tuple(predictor["related"]) if "related" in predictor else None,
        inputs=tuple((item["name"], item["type"]) for item in predictor["inputs"]),
        input_roles=tuple(item["role"] for item in predictor["inputs"]),
        inferred=MappingProxyType({}),
        context=ContextInfo(
            context["kind"],
            context.get("rows"),
            context.get("query_rows"),
            context.get("digest"),
        ),
        selection=SelectionInfo.from_dict(predictor["selection"]),
        frequency=predictor.get("frequency"),
        horizon=predictor.get("horizon"),
        quantiles=tuple(predictor.get("quantiles", ())),
        outputs=tuple(predictor["outputs"]),
        resources=MappingProxyType(dict(value.get("resources", {}))),
        manifest=value["manifest"],
        files=MappingProxyType(dict(value["model"]["files"])),
        spec=spec,
    )


__all__ = [
    "DERIVED_LOCK_VERSION",
    "LOCK",
    "LOCK_RULES",
    "LOCK_VERSION",
    "LOCK_VERSIONS",
    "changes",
    "checked_with",
    "document",
    "load_lock",
    "lock_digest",
    "lock_document",
    "manifest_digest",
    "other_rules",
    "read_lock",
    "resolution_of",
    "rules_of",
    "validate",
    "write_lock",
]
