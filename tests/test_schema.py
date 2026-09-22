"""The schema subset, tolerant parsing, and the one-retry repair loop."""

import pytest

from tailor.ask import ask_json
from tailor.prompts import Message, system
from tailor.schema import SchemaError, loads, parse, validate

PERSON = {
    "type": "object",
    "required": ["name", "tags"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "age": {"type": ["integer", "null"], "minimum": 0},
        "kind": {"type": "string", "enum": ["must", "nice"]},
        "tags": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}},
    },
}


def test_validate_reports_every_problem_with_a_path():
    bad = {"age": -1, "kind": "maybe", "tags": [], "extra": 1}
    errors = validate(bad, PERSON)
    assert "$: missing required key 'name'" in errors
    assert "$.age: -1 is below the minimum 0" in errors
    assert any(e.startswith("$.kind: 'maybe' is not one of") for e in errors)
    assert "$.tags: needs at least 1 item(s), got 0" in errors
    assert "$: unexpected key 'extra'" in errors


def test_validate_accepts_a_good_value_and_nullable_types():
    assert validate({"name": "z", "age": None, "tags": ["a"]}, PERSON) == []
    assert validate({"name": "z", "age": 3, "tags": ["a", "b"]}, PERSON) == []
    assert validate({"name": "z", "age": True, "tags": ["a"]}, PERSON) == [
        "$.age: expected integer or null, got bool"
    ]
    assert validate({"name": 1, "tags": ["a"]}, PERSON) == ["$.name: expected string, got int"]


def test_nested_paths_and_item_checks():
    schema = {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "number"}}}}
    assert validate([{"n": 1}, {"n": "x"}], schema) == ["$[1].n: expected number, got str"]


def test_loads_tolerates_fences_and_prose():
    assert loads('```json\n{"a": 1}\n```') == {"a": 1}
    assert loads('Sure! Here it is: {"a": {"b": [1, 2]}} Hope that helps.') == {"a": {"b": [1, 2]}}
    assert loads('{"s": "a } inside a string"}') == {"s": "a } inside a string"}


def test_loads_explains_failures():
    with pytest.raises(SchemaError, match="no JSON"):
        loads("nothing here")
    with pytest.raises(SchemaError, match="cut off"):
        loads('{"a": [1, 2')


def test_parse_combines_both_steps():
    with pytest.raises(SchemaError, match="missing required key 'tags'"):
        parse('{"name": "z"}', PERSON)


def test_ask_json_repairs_once_with_the_errors_in_the_conversation(scripted):
    provider = scripted('{"name": "z"}', '{"name": "z", "tags": ["ok"]}')
    out = ask_json(provider, [system("judge", schema=PERSON), Message("user", "go")], PERSON)
    assert out == {"name": "z", "tags": ["ok"]}
    assert len(provider.calls) == 2
    second = provider.calls[1]
    assert second[-2].role == "assistant" and second[-2].content == '{"name": "z"}'
    assert "missing required key 'tags'" in second[-1].content


def test_ask_json_gives_up_after_the_retries(scripted):
    provider = scripted("nope", "still nope")
    with pytest.raises(SchemaError, match="never matched the schema"):
        ask_json(provider, [system("judge", schema=PERSON), Message("user", "go")], PERSON)
    assert len(provider.calls) == 2
