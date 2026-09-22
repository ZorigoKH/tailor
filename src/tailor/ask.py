"""One model call, validated - and one second chance.

Models return nearly-right JSON often enough that a repair step is worth more than a bigger
prompt: parse, validate, and if it does not fit, hand the validation errors straight back
with the bad reply still in the conversation. One retry catches almost everything; two
mostly means the schema is wrong.
"""

from __future__ import annotations

from .prompts import Message, repair_message
from .schema import SchemaError, parse


def ask_json(
    provider,
    messages: list[Message],
    schema: dict,
    *,
    retries: int = 1,
    max_tokens: int = 1600,
) -> dict:
    """Call ``provider`` until its reply parses and validates against ``schema``."""
    conversation = list(messages)
    problem = ""
    for _ in range(retries + 1):
        completion = provider.complete(conversation, json_schema=schema, max_tokens=max_tokens)
        try:
            return parse(completion.text, schema)
        except SchemaError as exc:
            problem = str(exc)
            conversation = [
                *conversation,
                Message("assistant", completion.text[:4000]),
                repair_message(problem),
            ]
    raise SchemaError(f"the model's reply never matched the schema: {problem}")
