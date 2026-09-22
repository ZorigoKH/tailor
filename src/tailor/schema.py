"""A small JSON Schema subset, so a model's output can be checked and its errors fed back.

Models return JSON that is *almost* right: a missing key, a string where a list belongs, a
sentence wrapped around the object. Two things fix most of it - parse tolerantly, then
validate strictly and hand the validation errors back for one more try (see
:func:`tailor.prompts.repair_message`).

Supported: ``type`` (including a list of types), ``properties``, ``required``, ``items``,
``enum``, ``minItems``, ``maxItems``, ``minimum``, ``maximum`` and
``additionalProperties: false``. That is enough to pin every shape this package asks for.
"""

from __future__ import annotations

import json
import re
from typing import Any

TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


class SchemaError(ValueError):
    """The model's reply was not usable JSON, or did not match the schema."""


def _type_ok(value: Any, name: str) -> bool:
    if name in {"integer", "number"} and isinstance(value, bool):
        return False  # in Python bool is an int; in JSON it is not a number
    return isinstance(value, TYPES[name])


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable problems; empty means the value fits the schema."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        for name in names:
            if name not in TYPES:
                raise SchemaError(f"unsupported schema type {name!r}")
        if not any(_type_ok(value, name) for name in names):
            got = "null" if value is None else type(value).__name__
            return [f"{path}: expected {' or '.join(names)}, got {got}"]
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(option) for option in schema["enum"])
        errors.append(f"{path}: {value!r} is not one of [{allowed}]")
    if isinstance(value, dict):
        properties: dict[str, dict] = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected key {key!r}")
        for key, sub in properties.items():
            if key in value:
                errors += validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} item(s), got {len(value)}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: allows at most {schema['maxItems']} item(s), got {len(value)}")
        if "items" in schema:
            for i, item in enumerate(value):
                errors += validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    return errors


def loads(text: str) -> Any:
    """Parse the JSON value in ``text``, tolerating code fences and prose around it."""
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise SchemaError("the reply contained no JSON")
    close = {"{": "}", "[": "]"}[cleaned[start]]
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(cleaned)):
        char = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                if char != close:
                    raise SchemaError("mismatched brackets in the reply's JSON")
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise SchemaError(f"invalid JSON: {exc}") from exc
    raise SchemaError("the reply's JSON was cut off")


def parse(text: str, schema: dict) -> Any:
    """Parse and validate in one step; raises :class:`SchemaError` with every problem."""
    value = loads(text)
    errors = validate(value, schema)
    if errors:
        raise SchemaError("; ".join(errors))
    return value
