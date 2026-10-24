# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""api/openapi.yaml as data: its schemas, its examples and a validator for them.

The validator covers the JSON Schema the spec uses (refs, allOf/oneOf/anyOf,
types, const, enum, properties, additionalProperties, items, patterns and
bounds) so the conformance tests need nothing beyond PyYAML.
"""

import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "api" / "openapi.yaml"
PREFIX = "#/components/"
SERVED_TAGS = frozenset({"predictors", "models", "health", "engine"})


def load() -> dict[str, Any]:
    return yaml.safe_load(SPEC.read_text())


class SchemaError(AssertionError):
    pass


class Validator:
    """Checks JSON values against the spec's schemas."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    def deref(self, ref: str) -> Any:
        assert ref.startswith(PREFIX), ref
        node: Any = self.spec["components"]
        for part in ref[len(PREFIX) :].split("/"):
            node = node[part]
        return node

    def schema(self, name: str) -> dict[str, Any]:
        return self.spec["components"]["schemas"][name]

    def merge(self, schema: dict[str, Any]) -> dict[str, Any]:
        """One flat object schema for an `allOf`, so a closed part accepts the others' keys."""
        if "$ref" in schema:
            return self.merge(self.deref(schema["$ref"]))
        if "allOf" not in schema:
            return schema
        merged: dict[str, Any] = {key: v for key, v in schema.items() if key != "allOf"}
        properties: dict[str, Any] = {}
        required: list[str] = []
        closed = schema.get("additionalProperties") is False
        for part in schema["allOf"]:
            flat = self.merge(part)
            properties.update(flat.get("properties", {}))
            required.extend(flat.get("required", []))
            closed = closed or flat.get("additionalProperties") is False
            for key, value in flat.items():
                if key not in ("properties", "required", "additionalProperties"):
                    merged.setdefault(key, value)
        merged["properties"] = {**properties, **merged.get("properties", {})}
        merged["required"] = required + merged.get("required", [])
        if closed:
            merged["additionalProperties"] = False
        return merged

    def validate(self, schema: dict[str, Any], value: Any, path: str = "$") -> None:
        if "$ref" in schema:
            self.validate(self.deref(schema["$ref"]), value, path)
            return
        if "allOf" in schema:
            self.validate(self.merge(schema), value, path)
            return
        if "oneOf" in schema:
            matches = [part for part in schema["oneOf"] if self.fits(part, value, path)]
            if len(matches) != 1:
                raise SchemaError(f"{path}: {len(matches)} of oneOf matched {value!r}")
        if "anyOf" in schema and not any(
            self.fits(part, value, path) for part in schema["anyOf"]
        ):
            raise SchemaError(f"{path}: no anyOf matched {value!r}")
        if "const" in schema and value != schema["const"]:
            raise SchemaError(f"{path}: expected {schema['const']!r}, got {value!r}")
        if "enum" in schema and value not in schema["enum"]:
            raise SchemaError(f"{path}: {value!r} is not one of {schema['enum']}")
        kind = schema.get("type")
        if kind is not None:
            kinds = kind if isinstance(kind, list) else [kind]
            if not any(self.has_type(member, value) for member in kinds):
                raise SchemaError(f"{path}: expected {kind}, got {value!r}")
        if isinstance(value, str):
            self.check_string(schema, value, path)
        elif isinstance(value, bool):
            pass
        elif isinstance(value, (int, float)):
            self.check_number(schema, value, path)
        elif isinstance(value, list):
            self.check_array(schema, value, path)
        elif isinstance(value, dict):
            self.check_object(schema, value, path)

    def fits(self, schema: dict[str, Any], value: Any, path: str) -> bool:
        try:
            self.validate(schema, value, path)
        except SchemaError:
            return False
        return True

    @staticmethod
    def has_type(kind: str, value: Any) -> bool:
        if kind == "null":
            return value is None
        if kind == "boolean":
            return type(value) is bool
        if kind == "integer":
            return type(value) is int
        if kind == "number":
            return type(value) in (int, float) and math.isfinite(value)
        if kind == "string":
            return type(value) is str
        if kind == "array":
            return type(value) is list
        if kind == "object":
            return type(value) is dict
        raise SchemaError(f"unknown type {kind}")

    @staticmethod
    def check_string(schema: dict[str, Any], value: str, path: str) -> None:
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise SchemaError(f"{path}: {value!r} does not match {schema['pattern']}")
        if len(value) < schema.get("minLength", 0):
            raise SchemaError(f"{path}: shorter than {schema['minLength']}")
        if len(value) > schema.get("maxLength", math.inf):
            raise SchemaError(f"{path}: longer than {schema['maxLength']}")

    @staticmethod
    def check_number(schema: dict[str, Any], value: float, path: str) -> None:
        if value < schema.get("minimum", -math.inf):
            raise SchemaError(f"{path}: {value} below {schema['minimum']}")
        if value > schema.get("maximum", math.inf):
            raise SchemaError(f"{path}: {value} above {schema['maximum']}")
        if value <= schema.get("exclusiveMinimum", -math.inf):
            raise SchemaError(f"{path}: {value} not above {schema['exclusiveMinimum']}")
        if value >= schema.get("exclusiveMaximum", math.inf):
            raise SchemaError(f"{path}: {value} not below {schema['exclusiveMaximum']}")

    def check_array(self, schema: dict[str, Any], value: list, path: str) -> None:
        if len(value) < schema.get("minItems", 0):
            raise SchemaError(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for index, item in enumerate(value):
                self.validate(schema["items"], item, f"{path}[{index}]")

    def check_object(self, schema: dict[str, Any], value: dict, path: str) -> None:
        properties = self.properties(schema)
        required = self.required(schema)
        for key in required:
            if key not in value:
                raise SchemaError(f"{path}: missing {key}")
        if len(value) < schema.get("minProperties", 0):
            raise SchemaError(f"{path}: fewer than {schema['minProperties']} keys")
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                self.validate(properties[key], item, f"{path}.{key}")
            elif extra is False:
                raise SchemaError(f"{path}.{key}: unknown key")
            elif isinstance(extra, dict):
                self.validate(extra, item, f"{path}.{key}")

    def properties(self, schema: dict[str, Any]) -> dict[str, Any]:
        found: dict[str, Any] = {}
        if "$ref" in schema:
            return self.properties(self.deref(schema["$ref"]))
        for part in schema.get("allOf", []):
            found.update(self.properties(part))
        found.update(schema.get("properties", {}))
        return found

    def required(self, schema: dict[str, Any]) -> list[str]:
        found: list[str] = []
        if "$ref" in schema:
            return self.required(self.deref(schema["$ref"]))
        for part in schema.get("allOf", []):
            found.extend(self.required(part))
        found.extend(schema.get("required", []))
        return found


@dataclass(frozen=True)
class Example:
    """One example of the spec, with the schema it must satisfy."""

    operation: str
    tags: tuple[str, ...]
    method: str
    path: str
    section: str
    status: int | None
    name: str
    schema: dict[str, Any]
    value: Any
    invalid: bool = False

    @property
    def id(self) -> str:
        where = self.section if self.status is None else str(self.status)
        return f"{self.operation}:{where}:{self.name}"


def examples(spec: dict[str, Any]) -> list[Example]:
    validator = Validator(spec)
    found: list[Example] = []
    for path, methods in spec["paths"].items():
        for method, operation in methods.items():
            tags = tuple(operation.get("tags", []))
            body = operation.get("requestBody")
            if body is not None:
                media = body["content"]["application/json"]
                for name, example in media.get("examples", {}).items():
                    found.append(
                        Example(
                            operation["operationId"],
                            tags,
                            method.upper(),
                            path,
                            "request",
                            None,
                            name,
                            media["schema"],
                            example["value"],
                            bool(example.get("x-gf-invalid", False)),
                        )
                    )
            for status, response in operation["responses"].items():
                if "$ref" in response:
                    response = validator.deref(response["$ref"])
                media = response["content"]["application/json"]
                for name, example in media.get("examples", {}).items():
                    found.append(
                        Example(
                            operation["operationId"],
                            tags,
                            method.upper(),
                            path,
                            "response",
                            int(status),
                            name,
                            media["schema"],
                            example["value"],
                        )
                    )
    return found


def pairs(found: list[Example]) -> list[tuple[Example, Example]]:
    """Request examples with the response example of the same name."""
    result = []
    for request in found:
        if request.section != "request":
            continue
        replies = [
            reply
            for reply in found
            if reply.section == "response"
            and reply.operation == request.operation
            and reply.name == request.name
        ]
        assert len(replies) == 1, f"{request.id} needs exactly one reply example"
        result.append((request, replies[0]))
    return result


class Server:
    """A running gf serve named by GF_CONFORMANCE_URL, for replaying the spec."""

    def __init__(self) -> None:
        self.url = os.environ["GF_CONFORMANCE_URL"].rstrip("/")
        self.token = os.environ.get("GF_CONFORMANCE_TOKEN")
        self.project = os.environ.get("GF_CONFORMANCE_PROJECT", "local")
        self.counter = 0

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["X-GF-Request-ID"] = body["request_id"]
            data = json.dumps(body).encode()
        request = urllib.request.Request(
            self.url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def call(
        self, method: str, path: str, kind: str, name: str, payload: Any = None
    ) -> tuple[int, Any]:
        """One enveloped call on a `/v1/projects/{project}` route."""
        self.counter += 1
        request_id = f"r-conformance-{self.counter}"
        body = None
        if method != "GET":
            body = {
                "api_version": "gf/v1",
                "project": self.project,
                "request_id": request_id,
                "resource": {"kind": kind, "name": name},
                "payload": payload or {},
            }
        return self.request(method, f"/v1/projects/{self.project}" + path, body)

    def rewrite(self, value: Any) -> Any:
        """The example with its project replaced by this server's."""
        if isinstance(value, dict):
            return {
                key: self.project if key == "project" else self.rewrite(item)
                for key, item in value.items()
            }
        return value
