# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Emit the Python and TypeScript types for the API from api/openapi.yaml.

The spec is the source of truth. This script needs only the standard library
and PyYAML, writes `src/generalforecast/_api.py` and
`clients/typescript/src/types.ts`, and emits code that `ruff format` leaves
unchanged. `python api/generate.py --check` exits 1 when either file differs
from what the spec says.
"""

from __future__ import annotations

import argparse
import keyword
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "api" / "openapi.yaml"
PYTHON = ROOT / "src" / "generalforecast" / "_api.py"
TYPESCRIPT = ROOT / "clients" / "typescript" / "src" / "types.ts"
WIDTH = 88
ENUM_CLASSES = {"ErrorCode"}
TS_NAMES = {"Error": "ApiError"}
ENVELOPE = "Envelope"

Json = dict[str, Any]


# -- the spec -----------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """One property of a record schema."""

    key: str
    name: str
    schema: Json
    required: bool


@dataclass(frozen=True)
class Record:
    """An object schema with named properties, flattened across `allOf`."""

    name: str
    description: str
    fields: tuple[Field, ...]
    extends: tuple[str, ...]
    extra: Json | bool
    defaults: dict[str, Any]


@dataclass(frozen=True)
class Alias:
    """A schema that is not a record: a scalar, an enum, an array or a mapping."""

    name: str
    description: str
    schema: Json


@dataclass(frozen=True)
class Operation:
    """One route of the spec."""

    name: str
    method: str
    path: str
    request: str | None
    reply: str


class Spec:
    """The parsed spec, with `$ref`s resolved by name."""

    def __init__(self, document: Json) -> None:
        """Index the component schemas and the operations."""
        self.document = document
        self.schemas: dict[str, Json] = document["components"]["schemas"]
        self.definitions: dict[str, Record | Alias] = {}
        for name, schema in self.schemas.items():
            self.definitions[name] = self._define(name, schema)
        self.operations = tuple(self._operations())

    @staticmethod
    def ref_name(schema: Json) -> str | None:
        """The component name a `$ref` points at, if the schema is one."""
        ref = schema.get("$ref")
        if ref is None:
            return None
        prefix = "#/components/schemas/"
        if not ref.startswith(prefix):
            raise ValueError(f"unsupported $ref {ref}")
        return ref[len(prefix) :]

    def resolve(self, schema: Json) -> Json:
        """Follow `$ref`s until a concrete schema."""
        while (name := self.ref_name(schema)) is not None:
            schema = self.schemas[name]
        return schema

    def _define(self, name: str, schema: Json) -> Record | Alias:
        if "allOf" in schema:
            return self._composite(name, schema)
        if schema.get("type") == "object" and (
            "properties" in schema or schema.get("additionalProperties") is False
        ):
            return self._record(name, schema, (), schema.get("properties", {}), schema)
        return Alias(name, schema.get("description", ""), schema)

    def _composite(self, name: str, schema: Json) -> Record:
        extends: list[str] = []
        properties: dict[str, Json] = {}
        required: list[str] = []
        inline: Json = {}
        for part in schema["allOf"]:
            ref = self.ref_name(part)
            if ref is not None:
                extends.append(ref)
                base = self.definitions[ref]
                if not isinstance(base, Record):
                    raise ValueError(f"{name} extends non-record {ref}")
                for field in base.fields:
                    properties[field.key] = field.schema
                    if field.required:
                        required.append(field.key)
            else:
                inline = part
                properties.update(part.get("properties", {}))
                required.extend(part.get("required", []))
        merged = {**inline, "required": required}
        return self._record(name, schema, tuple(extends), properties, merged)

    def _record(
        self,
        name: str,
        schema: Json,
        extends: tuple[str, ...],
        properties: dict[str, Json],
        source: Json,
    ) -> Record:
        required = set(source.get("required", []))
        fields = tuple(
            Field(key, python_name(key), value, key in required)
            for key, value in properties.items()
        )
        defaults = {
            field.key: field.schema["default"]
            for field in fields
            if "default" in field.schema and not field.required
        }
        extra = source.get("additionalProperties", False)
        return Record(
            name, schema.get("description", ""), fields, extends, extra, defaults
        )

    def _operations(self):
        for path, methods in self.document["paths"].items():
            for method, operation in methods.items():
                request = None
                body = operation.get("requestBody")
                if body is not None:
                    request = self.ref_name(
                        body["content"]["application/json"]["schema"]
                    )
                success = next(
                    response
                    for status, response in operation["responses"].items()
                    if status.startswith("2")
                )
                reply = self.ref_name(success["content"]["application/json"]["schema"])
                if reply is None:
                    raise ValueError(f"{path} {method} reply must be a $ref")
                yield Operation(
                    operation["operationId"], method.upper(), path, request, reply
                )

    def ordered(self) -> list[Record | Alias]:
        """Definitions in dependency order, spec order breaking ties."""
        done: dict[str, Record | Alias] = {}
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in done:
                return
            if name in visiting:
                raise ValueError(f"schema cycle through {name}")
            visiting.add(name)
            for dependency in self.dependencies(name):
                visit(dependency)
            visiting.discard(name)
            done[name] = self.definitions[name]

        for name in self.definitions:
            visit(name)
        return list(done.values())

    def dependencies(self, name: str) -> list[str]:
        """The component names a definition refers to."""
        found: list[str] = []

        def walk(schema: Json) -> None:
            ref = self.ref_name(schema)
            if ref is not None:
                if ref not in found:
                    found.append(ref)
                return
            for key in ("items", "additionalProperties"):
                if isinstance(schema.get(key), dict):
                    walk(schema[key])
            for key in ("oneOf", "anyOf", "allOf"):
                for part in schema.get(key, []):
                    walk(part)
            for part in schema.get("properties", {}).values():
                walk(part)

        definition = self.definitions[name]
        if isinstance(definition, Record):
            found.extend(definition.extends)
            for field in definition.fields:
                walk(field.schema)
            if isinstance(definition.extra, dict):
                walk(definition.extra)
        else:
            walk(definition.schema)
        return found


def python_name(key: str) -> str:
    """A Python attribute name for a JSON key."""
    name = re.sub(r"[^0-9A-Za-z_]", "_", key)
    if keyword.iskeyword(name):
        name += "_"
    return name


# -- a black-compatible layout -------------------------------------------------


class Node:
    """An expression that lays itself out on one line or exploded."""

    def flat(self) -> str:
        """The one-line form."""
        raise NotImplementedError

    def render(self, indent: int, suffix: str = "") -> list[str]:
        """Lines at `indent`; `suffix` follows the closing line."""
        flat = self.flat()
        if indent + len(flat) + len(suffix) <= WIDTH:
            return [" " * indent + flat + suffix]
        return self.exploded(indent, suffix)

    def exploded(self, indent: int, suffix: str) -> list[str]:
        """Lines when the one-line form does not fit."""
        raise NotImplementedError


class Atom(Node):
    """Source text that cannot be split."""

    def __init__(self, text: str) -> None:
        """Keep the text."""
        self.text = text

    def flat(self) -> str:
        """The text."""
        return self.text

    def exploded(self, indent: int, suffix: str) -> list[str]:
        """An atom has no shorter form."""
        return [" " * indent + self.text + suffix]


class Bracketed(Node):
    """`head(a, b)`, `head[a, b]`, `(a, b)` or `{a: b}` with magic trailing commas."""

    def __init__(
        self, head: str, items: list[Node], open: str = "(", close: str = ")"
    ) -> None:
        """Items are laid out between `open` and `close` after `head`."""
        self.head, self.items, self.open, self.close = head, items, open, close

    def flat(self) -> str:
        """One line, with the comma a one-element tuple needs."""
        body = ", ".join(item.flat() for item in self.items)
        if self.open == "(" and not self.head and len(self.items) == 1:
            body += ","
        return f"{self.head}{self.open}{body}{self.close}"

    def exploded(self, indent: int, suffix: str) -> list[str]:
        """One item per line with a trailing comma, as black keeps it."""
        lines = [" " * indent + self.head + self.open]
        for item in self.items:
            lines.extend(item.render(indent + 4, ","))
        lines.append(" " * indent + self.close + suffix)
        return lines


class Pair(Node):
    """A `key: value` entry of a dict literal."""

    def __init__(self, key: str, value: Node) -> None:
        """Key is source text."""
        self.key, self.value = key, value

    def flat(self) -> str:
        """One line."""
        return f"{self.key}: {self.value.flat()}"

    def exploded(self, indent: int, suffix: str) -> list[str]:
        """The value explodes after the key."""
        lines = self.value.render(indent, suffix)
        lines[0] = " " * indent + self.key + ": " + lines[0].lstrip()
        return lines


def literal(value: Any) -> Node:
    """A Python literal for a JSON value, as black formats it."""
    if isinstance(value, bool) or value is None:
        return Atom(repr(value))
    if isinstance(value, (int, float)):
        return Atom(repr(value))
    if isinstance(value, str):
        return Atom(quote(value))
    if isinstance(value, list):
        return Bracketed("", [literal(item) for item in value], "[", "]")
    if isinstance(value, dict):
        return Bracketed(
            "",
            [Pair(quote(key), literal(item)) for key, item in value.items()],
            "{",
            "}",
        )
    raise TypeError(f"no literal for {value!r}")


def quote(text: str) -> str:
    """A double-quoted string literal."""
    if '"' in text and "'" not in text:
        return "'" + text.replace("\\", "\\\\") + "'"
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# -- Python --------------------------------------------------------------------

RUNTIME = '''
"""The HTTP API's request and reply types, generated from api/openapi.yaml.

Do not edit: run `python api/generate.py`. Every schema is a frozen dataclass
(or a type alias for scalars, arrays and mappings) with `from_payload`, which
checks the JSON shape and names the offending field in `PayloadError.param`,
and `to_payload`, which writes the JSON back without the optional fields left
as None. `ROUTES` lists every route with its request and reply types.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Literal

VERSION = "gf/v1"


class PayloadError(ValueError):
    """A payload that does not match the spec; `param` names the field."""

    def __init__(self, message: str, param: str) -> None:
        """Keep the message and the dotted path of the field."""
        super().__init__(f"{param}: {message}" if param else message)
        self.message, self.param = message, param


Decoder = Callable[[Any, str], Any]


def _at(param: str, key: str) -> str:
    return f"{param}.{key}" if param else key


def _text(value: Any, param: str) -> str:
    if type(value) is not str:
        raise PayloadError("expected a string", param)
    return value


def _integer(value: Any, param: str) -> int:
    if type(value) is not int:
        raise PayloadError("expected an integer", param)
    return value


def _number(value: Any, param: str) -> float:
    if type(value) not in (int, float):
        raise PayloadError("expected a number", param)
    return value


def _boolean(value: Any, param: str) -> bool:
    if type(value) is not bool:
        raise PayloadError("expected a boolean", param)
    return value


def _null(value: Any, param: str) -> None:
    if value is not None:
        raise PayloadError("expected null", param)


def _any(value: Any, param: str) -> Any:
    return value


def _pattern(expression: str) -> Decoder:
    compiled = re.compile(expression)

    def decode(value: Any, param: str) -> str:
        text = _text(value, param)
        if compiled.search(text) is None:
            raise PayloadError(f"expected a string matching {expression}", param)
        return text

    return decode


def _literal(*options: Any) -> Decoder:
    def decode(value: Any, param: str) -> Any:
        if type(value) is bool and bool not in {type(option) for option in options}:
            raise PayloadError(f"expected one of {list(options)}", param)
        if value not in options:
            raise PayloadError(f"expected one of {list(options)}", param)
        return value

    return decode


def _items(item: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> tuple:
        if type(value) is not list:
            raise PayloadError("expected an array", param)
        return tuple(item(element, f"{param}[{i}]") for i, element in enumerate(value))

    return decode


def _mapping(item: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> dict:
        if type(value) is not dict:
            raise PayloadError("expected an object", param)
        return {key: item(element, _at(param, key)) for key, element in value.items()}

    return decode


def _union(*options: Decoder) -> Decoder:
    def decode(value: Any, param: str) -> Any:
        errors = []
        for option in options:
            try:
                return option(value, param)
            except PayloadError as error:
                errors.append(error.message)
        raise PayloadError(" or ".join(errors), param)

    return decode


def _record(cls: type[Payload]) -> Decoder:
    return cls.from_payload


def _enum(cls: type[StrEnum]) -> Decoder:
    def decode(value: Any, param: str) -> StrEnum:
        try:
            return cls(_text(value, param))
        except ValueError:
            raise PayloadError(
                f"expected one of {[m.value for m in cls]}", param
            ) from None

    return decode


def _encode(value: Any) -> Any:
    if isinstance(value, Payload):
        return value.to_payload()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Field:
    """How one JSON key of a record maps to a dataclass field."""

    key: str
    name: str
    required: bool
    decode: Decoder


@dataclass(frozen=True, kw_only=True)
class Payload:
    """A record schema: JSON in through `from_payload`, out through `to_payload`."""

    FIELDS: ClassVar[tuple[Field, ...]] = ()
    DEFAULTS: ClassVar[dict[str, Any]] = {}
    EXTRA: ClassVar[bool] = False

    @classmethod
    def from_payload(cls, value: Any, param: str = ""):
        """Decode a JSON object, naming the first field that does not fit."""
        if type(value) is not dict:
            raise PayloadError("expected an object", param)
        known = {spec.key for spec in cls.FIELDS}
        kwargs: dict[str, Any] = {}
        for spec in cls.FIELDS:
            if spec.key in value:
                kwargs[spec.name] = spec.decode(value[spec.key], _at(param, spec.key))
            elif spec.required:
                raise PayloadError("required", _at(param, spec.key))
        unknown = [key for key in value if key not in known]
        if cls.EXTRA:
            kwargs["extra"] = {key: value[key] for key in unknown}
        elif unknown:
            raise PayloadError("unknown field", _at(param, unknown[0]))
        return cls(**kwargs)

    def to_payload(self) -> dict[str, Any]:
        """The JSON object, without optional fields left as None."""
        payload: dict[str, Any] = {}
        for spec in self.FIELDS:
            value = getattr(self, spec.name)
            if value is None and not spec.required:
                continue
            payload[spec.key] = _encode(value)
        if self.EXTRA:
            payload.update(_encode(self.__dict__["extra"]))
        return payload


@dataclass(frozen=True)
class Route:
    """A route: its request envelope type (None for GET) and its reply type."""

    name: str
    method: str
    path: str
    request: type[Payload] | None
    reply: type[Payload]
'''


class PythonEmitter:
    """Render the spec as `_api.py`."""

    def __init__(self, spec: Spec) -> None:
        """Keep the spec."""
        self.spec = spec

    def annotation(self, schema: Json) -> str:
        """The type annotation for a schema."""
        ref = self.spec.ref_name(schema)
        if ref is not None:
            return ref
        if "oneOf" in schema or "anyOf" in schema:
            parts = schema.get("oneOf") or schema["anyOf"]
            return " | ".join(self.annotation(part) for part in parts)
        if "const" in schema:
            return f"Literal[{quote(schema['const']) if isinstance(schema['const'], str) else schema['const']}]"
        if "enum" in schema:
            return "Literal[" + ", ".join(quote(v) for v in schema["enum"]) + "]"
        kind = schema.get("type")
        if isinstance(kind, list):
            return " | ".join(
                self.annotation({**schema, "type": member}) for member in kind
            )
        if kind == "string":
            return "str"
        if kind == "integer":
            return "int"
        if kind == "number":
            return "float"
        if kind == "boolean":
            return "bool"
        if kind == "null":
            return "None"
        if kind == "array":
            return f"tuple[{self.annotation(schema.get('items', {}))}, ...]"
        if kind == "object":
            extra = schema.get("additionalProperties", True)
            if "properties" in schema:
                raise ValueError(
                    "inline object schemas with properties are not supported"
                )
            inner = "Any" if isinstance(extra, bool) else self.annotation(extra)
            return f"dict[str, {inner}]"
        return "Any"

    def decoder(self, schema: Json) -> Node:
        """The decoder expression for a schema."""
        ref = self.spec.ref_name(schema)
        if ref is not None:
            definition = self.spec.definitions[ref]
            if isinstance(definition, Record):
                return Bracketed("_record", [Atom(ref)])
            if ref in ENUM_CLASSES:
                return Bracketed("_enum", [Atom(ref)])
            return Atom(f"_decode_{ref}")
        if "oneOf" in schema or "anyOf" in schema:
            parts = schema.get("oneOf") or schema["anyOf"]
            return Bracketed("_union", [self.decoder(part) for part in parts])
        if "const" in schema:
            return Bracketed("_literal", [literal(schema["const"])])
        if "enum" in schema:
            return Bracketed("_literal", [literal(v) for v in schema["enum"]])
        kind = schema.get("type")
        if isinstance(kind, list):
            return Bracketed(
                "_union",
                [self.decoder({**schema, "type": member}) for member in kind],
            )
        if kind == "string":
            if "pattern" in schema:
                return Bracketed("_pattern", [Atom(quote(schema["pattern"]))])
            return Atom("_text")
        if kind == "integer":
            return Atom("_integer")
        if kind == "number":
            return Atom("_number")
        if kind == "boolean":
            return Atom("_boolean")
        if kind == "null":
            return Atom("_null")
        if kind == "array":
            return Bracketed("_items", [self.decoder(schema.get("items", {}))])
        if kind == "object":
            extra = schema.get("additionalProperties", True)
            inner = Atom("_any") if isinstance(extra, bool) else self.decoder(extra)
            return Bracketed("_mapping", [inner])
        return Atom("_any")

    def docstring(self, text: str, fallback: str, indent: int) -> list[str]:
        """A docstring whose first line is one sentence ending in a period."""
        text = " ".join(text.split())
        first, _, rest = text.partition(". ")
        if first:
            first = first[0].upper() + first[1:].rstrip(".") + "."
        pad = " " * indent
        if not first or indent + len(first) + 6 > WIDTH:
            first, rest = fallback, text
        if not rest:
            return [f'{pad}"""{first}"""']
        lines = [f'{pad}"""{first}', ""]
        current = pad
        for word in rest.split():
            if len(current) + len(word) + 1 > WIDTH and current.strip():
                lines.append(current.rstrip())
                current = pad
            current += word + " "
        lines.append(current.rstrip())
        lines.append(f'{pad}"""')
        return lines

    def alias(self, definition: Alias) -> list[str]:
        """A type alias and its decoder, or the error-code enum."""
        schema, name = definition.schema, definition.name
        if name in ENUM_CLASSES:
            lines = [f"class {name}(StrEnum):"]
            lines.extend(
                self.docstring(definition.description, f"The {name} values.", 4)
            )
            lines.append("")
            for value in schema["enum"]:
                lines.append(f"    {value} = {quote(value)}")
            return lines
        if "enum" in schema:
            options = [literal(value) for value in schema["enum"]]
            lines = assign(f"{name} = ", Bracketed("Literal", options, "[", "]"), 0)
        else:
            lines = [f"{name} = {self.annotation(schema)}"]
        lines.extend(assign(f"_decode_{name}: Decoder = ", self.decoder(schema), 0))
        return lines

    def record(self, definition: Record) -> list[str]:
        """A frozen keyword-only dataclass with its field table."""
        lines = ["@dataclass(frozen=True, kw_only=True)"]
        lines.append(f"class {definition.name}(Payload):")
        lines.extend(
            self.docstring(definition.description, f"The {definition.name} schema.", 4)
        )
        lines.append("")
        required = [f for f in definition.fields if f.required]
        optional = [f for f in definition.fields if not f.required]
        for field in required + optional:
            annotation = self.annotation(field.schema)
            if "const" in field.schema:
                default = f" = {literal(field.schema['const']).flat()}"
            elif field.required:
                default = ""
            else:
                if "None" not in annotation.split(" | "):
                    annotation += " | None"
                default = " = None"
            line = f"    {field.name}: {annotation}{default}"
            if len(line) <= WIDTH:
                lines.append(line)
            else:
                lines.extend(
                    [f"    {field.name}: (", f"        {annotation}", f"    ){default}"]
                )
        if definition.extra is not False:
            lines.append("    extra: dict[str, Any] = field(default_factory=dict)")
        if definition.fields or definition.extra is not False:
            lines.append("")
        table = Bracketed(
            "",
            [
                Bracketed(
                    "Field",
                    [
                        Atom(quote(f.key)),
                        Atom(quote(f.name)),
                        Atom(repr(f.required)),
                        self.decoder(f.schema),
                    ],
                )
                for f in definition.fields
            ],
        )
        lines.extend(assign("    FIELDS: ClassVar[tuple[Field, ...]] = ", table, 4))
        if definition.defaults:
            lines.extend(
                assign(
                    "    DEFAULTS: ClassVar[dict[str, Any]] = ",
                    literal(definition.defaults),
                    4,
                )
            )
        if definition.extra is not False:
            lines.append("    EXTRA: ClassVar[bool] = True")
        return lines

    def routes(self) -> list[str]:
        """The `ROUTES` table."""
        table = Bracketed(
            "",
            [
                Bracketed(
                    "Route",
                    [
                        Atom(quote(op.name)),
                        Atom(quote(op.method)),
                        Atom(quote(op.path)),
                        Atom(op.request or "None"),
                        Atom(op.reply),
                    ],
                )
                for op in self.spec.operations
            ],
        )
        return assign("ROUTES: tuple[Route, ...] = ", table, 0)

    def render(self) -> str:
        """The whole module."""
        blocks: list[list[str]] = []
        for definition in self.spec.ordered():
            if isinstance(definition, Record):
                blocks.append(self.record(definition))
            else:
                blocks.append(self.alias(definition))
        blocks.append(self.routes())
        body = RUNTIME.lstrip("\n")
        if any(
            isinstance(d, Record) and d.extra is not False for d in self.spec.ordered()
        ):
            body = body.replace(
                "from dataclasses import dataclass\n",
                "from dataclasses import dataclass, field\n",
            )
        out = body
        for block in blocks:
            out += "\n\n" + "\n".join(block) + "\n"
        return out


def assign(head: str, value: Node, indent: int) -> list[str]:
    """`head value` laid out as black would, `head` already indented."""
    flat = value.flat()
    if len(head) + len(flat) <= WIDTH:
        return [head + flat]
    lines = value.exploded(indent, "")
    lines[0] = head + lines[0].lstrip()
    return lines


# -- TypeScript ----------------------------------------------------------------


class TypeScriptEmitter:
    """Render the spec as `types.ts`."""

    def __init__(self, spec: Spec) -> None:
        """Keep the spec."""
        self.spec = spec

    def annotation(self, schema: Json) -> str:
        """The TypeScript type for a schema."""
        ref = self.spec.ref_name(schema)
        if ref is not None:
            return TS_NAMES.get(ref, ref)
        if "oneOf" in schema or "anyOf" in schema:
            parts = schema.get("oneOf") or schema["anyOf"]
            return " | ".join(self.annotation(part) for part in parts)
        if "const" in schema:
            return ts_literal(schema["const"])
        if "enum" in schema:
            return " | ".join(ts_literal(v) for v in schema["enum"])
        kind = schema.get("type")
        if isinstance(kind, list):
            return " | ".join(
                self.annotation({**schema, "type": member}) for member in kind
            )
        if kind == "string":
            return "string"
        if kind in ("integer", "number"):
            return "number"
        if kind == "boolean":
            return "boolean"
        if kind == "null":
            return "null"
        if kind == "array":
            inner = self.annotation(schema.get("items", {}))
            return f"Array<{inner}>" if " | " in inner else f"{inner}[]"
        if kind == "object":
            extra = schema.get("additionalProperties", True)
            inner = "unknown" if extra is True else self.annotation(extra)
            return f"Record<string, {inner}>"
        return "unknown"

    def comment(self, text: str) -> list[str]:
        """A JSDoc comment for a description, if there is one."""
        text = " ".join(text.split())
        if not text:
            return []
        if len(text) <= WIDTH - 8:
            return [f"/** {text} */"]
        lines = ["/**"]
        current = " *"
        for word in text.split():
            if len(current) + len(word) + 1 > WIDTH:
                lines.append(current)
                current = " *"
            current += " " + word
        lines.append(current)
        lines.append(" */")
        return lines

    def render(self) -> str:
        """The whole module."""
        out = [
            "// The HTTP API's request and reply types, generated from",
            "// api/openapi.yaml by api/generate.py. Do not edit.",
            "",
            'export const VERSION = "gf/v1";',
        ]
        for definition in self.spec.ordered():
            out.append("")
            out.extend(self.comment(definition.description))
            if isinstance(definition, Record):
                extends = (
                    " extends "
                    + ", ".join(TS_NAMES.get(b, b) for b in definition.extends)
                    if definition.extends
                    else ""
                )
                name = TS_NAMES.get(definition.name, definition.name)
                out.append(f"export interface {name}{extends} {{")
                inherited = {
                    field.key
                    for base in definition.extends
                    if isinstance(record := self.spec.definitions[base], Record)
                    for field in record.fields
                }
                for field in definition.fields:
                    if field.key in inherited:
                        continue
                    optional = "" if field.required else "?"
                    out.append(
                        f"  {ts_key(field.key)}{optional}: {self.annotation(field.schema)};"
                    )
                if definition.extra is not False:
                    inner = (
                        "unknown"
                        if definition.extra is True
                        else self.annotation(definition.extra)
                    )
                    out.append(f"  [key: string]: {inner};")
                out.append("}")
            elif definition.name in ENUM_CLASSES:
                values = ", ".join(ts_literal(v) for v in definition.schema["enum"])
                constant = re.sub(r"(?<!^)(?=[A-Z])", "_", definition.name).upper()
                out.append(f"export const {constant}S = [{values}] as const;")
                out.append(
                    f"export type {definition.name} = (typeof {constant}S)[number];"
                )
            else:
                name = TS_NAMES.get(definition.name, definition.name)
                out.append(
                    f"export type {name} = {self.annotation(definition.schema)};"
                )
        out.append("")
        out.append("export const ROUTES = {")
        for op in self.spec.operations:
            request = TS_NAMES.get(op.request or "null", op.request or "null")
            out.append(
                f'  "{op.name}": {{ method: "{op.method}", path: "{op.path}",'
                f' request: "{request}", reply: "{TS_NAMES.get(op.reply, op.reply)}" }},'
            )
        out.append("} as const;")
        out.append("")
        out.append("export type RouteName = keyof typeof ROUTES;")
        return "\n".join(out) + "\n"


def ts_literal(value: Any) -> str:
    """A TypeScript literal type for a JSON scalar."""
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def ts_key(key: str) -> str:
    """A property name, quoted when it is not an identifier."""
    return key if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", key) else f'"{key}"'


# -- entry point ----------------------------------------------------------------


def render(spec_path: Path = SPEC) -> dict[Path, str]:
    """The generated files, by path."""
    spec = Spec(yaml.safe_load(spec_path.read_text()))
    return {
        PYTHON: PythonEmitter(spec).render(),
        TYPESCRIPT: TypeScriptEmitter(spec).render(),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the generated files, or with --check report which differ."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 on any difference")
    args = parser.parse_args(argv)
    stale = []
    for path, text in render().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    for path in stale:
        print(
            f"{path.relative_to(ROOT)} differs from api/openapi.yaml", file=sys.stderr
        )
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
