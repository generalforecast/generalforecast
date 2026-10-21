"""Request-scoped package inputs shared by the hosted API and `gf serve`."""

import json
import tempfile
from pathlib import Path

import pyarrow as pa

from generalforecast.deployments.contracts import ServiceError, require
from generalforecast.packages import Package, PackageSpec, PythonLock
from generalforecast.packages.validation import PackageError
from generalforecast.tabular import coerce, to_arrow


def request_table(data, lock, *, max_rows):
    """Keeps other input types strict; ISO strings are the JSON temporal encoding."""
    table = to_arrow(data, max_rows=max_rows)
    if isinstance(lock, PythonLock):
        return table
    try:
        return coerce(
            table,
            {
                column.name: column.type
                for column in lock.schema.columns
                if column.name in table.column_names
                and column.type in {"date", "timestamp"}
                and pa.types.is_string(table[column.name].type)
            },
        )
    except PackageError:
        raise ServiceError(
            "ValidationError", "invalid ISO temporal input", 422
        ) from None


def compile_request(authored, context):
    """Uses the standard offline compiler; writes no data or executable files."""
    PackageSpec.from_dict(authored)
    with tempfile.TemporaryDirectory(prefix="gf-request-") as root:
        (Path(root) / "gf.yaml").write_text(json.dumps(authored))
        package = Package.load(root).check(data=context, write=False, offline=True)
    assert package.lock is not None
    return package.lock


def bind_request(lock, payload, context):
    """Binds explicit per-call roles only when no physical schema was locked."""
    authored = lock.spec.to_dict()
    target = payload.get("target", lock.spec.target)
    require(type(target) is str, "request target is required")
    require(
        lock.spec.target is None or target == lock.spec.target,
        "target differs from lock",
    )
    authored["target"] = (
        {"column": target, "type": lock.spec.target_type}
        if lock.spec.target_type is not None
        else target
    )
    if "features" in payload:
        features = payload["features"]
        require(type(features) is list, "features must be a list")
        inputs = authored.get("inputs", {})
        require(not inputs.get("exclude"), "features differ from lock")
        require(
            not inputs.get("include") or features == inputs["include"],
            "features differ from lock",
        )
        authored["inputs"] = {**inputs, "include": features}
        authored["inputs"].pop("exclude", None)
    return compile_request(authored, context)
