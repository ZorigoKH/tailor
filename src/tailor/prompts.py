"""The prompts, and the wire format both providers speak.

Every request carries a task tag on its first line (``TASK: judge``) and delimited sections
(``<<REQUIREMENT>> ... <</REQUIREMENT>>``). A hosted model reads them as instructions; the
offline stand-in parses them with regexes. One format, two readers, so the tests exercise
the same path a real model takes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .corpus import Evidence
from .retrieval import Hit


@dataclass(frozen=True)
class Message:
    role: str  # "system" or "user"
    content: str


def block(name: str, body: str, **attributes: object) -> str:
    """One delimited section, e.g. ``<<EVIDENCE id=terran.md:4-6>> ... <</EVIDENCE>>``."""
    attrs = "".join(f" {key}={value}" for key, value in attributes.items())
    return f"<<{name}{attrs}>>\n{body.strip()}\n<</{name}>>"


def section(text: str, name: str) -> str:
    """The body of the first ``name`` section, or "" when there is none."""
    found = re.search(rf"<<{name}[^>]*>>\n(.*?)\n<</{name}>>", text, re.DOTALL)
    return found.group(1).strip() if found else ""


def sections(text: str, name: str) -> list[tuple[dict[str, str], str]]:
    """Every ``name`` section as (attributes, body)."""
    out = []
    for found in re.finditer(rf"<<{name}([^>]*)>>\n(.*?)\n<</{name}>>", text, re.DOTALL):
        attributes = dict(re.findall(r"(\w+)=(\S+)", found.group(1)))
        out.append((attributes, found.group(2).strip()))
    return out


def task_of(messages: list[Message]) -> str:
    """The task tag of a request, read from the system message."""
    for message in messages:
        found = re.match(r"TASK:\s*(\S+)", message.content)
        if found:
            return found.group(1)
    raise ValueError("no TASK tag in the prompt")


# -- schemas ---------------------------------------------------------------------------
REQUIREMENT_SCHEMA = {
    "type": "object",
    "required": ["text", "kind"],
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string"},
        "kind": {"type": "string", "enum": ["must", "nice"]},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
}

POSTING_SCHEMA = {
    "type": "object",
    "required": ["title", "company", "locations", "requirements"],
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "company": {"type": "string"},
        "team": {"type": ["string", "null"]},
        "locations": {"type": "array", "items": {"type": "string"}},
        "employment_type": {"type": ["string", "null"]},
        "deadline": {"type": ["string", "null"]},
        "requirements": {"type": "array", "minItems": 1, "items": REQUIREMENT_SCHEMA},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}

JUDGE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "citations", "quote", "why"],
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["covered", "partial", "missing"]},
        "citations": {"type": "array", "items": {"type": "string"}},
        "quote": {"type": "string"},
        "why": {"type": "string"},
    },
}

DRAFT_SCHEMA = {
    "type": "object",
    "required": ["paragraphs"],
    "additionalProperties": False,
    "properties": {
        "paragraphs": {"type": "array", "minItems": 2, "maxItems": 6, "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
}

AGENT_SCHEMA = {
    "type": "object",
    "required": ["thought", "tool", "args"],
    "additionalProperties": False,
    "properties": {
        "thought": {"type": "string"},
        "tool": {"type": "string"},
        "args": {"type": "object"},
    },
}

GROUNDING_RULES = (
    "Use only the evidence given to you. Never add a fact, number, employer or date that is "
    "not in it.",
    "Quote verbatim. Every quote you return must appear word for word in the evidence you "
    "cite, because it is checked against the source and dropped if it does not.",
    'If the evidence does not support an answer, say so: verdict "missing", empty citations.',
)


def system(
    task: str, *, schema: dict | None = None, rules: tuple[str, ...] = GROUNDING_RULES
) -> Message:
    lines = [f"TASK: {task}", ""]
    lines += [f"- {rule}" for rule in rules]
    if schema is not None:
        lines += ["", "Reply with JSON only, matching this schema:", json.dumps(schema)]
    return Message("system", "\n".join(lines))


def evidence_blocks(items: list[Evidence] | list[Hit]) -> str:
    pieces = [item.evidence if isinstance(item, Hit) else item for item in items]
    return "\n".join(block("EVIDENCE", piece.quotable, id=piece.id) for piece in pieces)


# -- requests --------------------------------------------------------------------------
def posting_messages(text: str) -> list[Message]:
    user = "\n".join(
        [
            "Pull the structured facts out of this job posting.",
            'Split "what we are looking for" into one requirement per line; mark a '
            'requirement "nice" when the posting calls it preferred or nice to have, '
            'otherwise "must".',
            "deadline is an ISO date (YYYY-MM-DD) or null.",
            "",
            block("POSTING", text[:20000]),
        ]
    )
    return [system("posting", schema=POSTING_SCHEMA, rules=()), Message("user", user)]


def judge_messages(requirement: str, hits: list[Hit]) -> list[Message]:
    user = "\n".join(
        [
            "Does the evidence show this person meets the requirement?",
            "Cite the evidence ids you used and quote the words that show it.",
            "",
            block("REQUIREMENT", requirement),
            evidence_blocks(hits),
        ]
    )
    return [system("judge", schema=JUDGE_SCHEMA), Message("user", user)]


def draft_messages(posting: dict, cited: list[Evidence], *, paragraphs: int = 4) -> list[Message]:
    user = "\n".join(
        [
            f"Draft a cover letter of at most {paragraphs} paragraphs for this role, "
            "in plain first-person prose, no greeting and no sign-off.",
            "Each paragraph must rest on the evidence below; name the role and the company "
            "once, and claim nothing else that the evidence does not say.",
            "",
            block("ROLE", f"{posting.get('title', '')} at {posting.get('company', '')}"),
            block(
                "REQUIREMENTS", "\n".join(f"- {r['text']}" for r in posting.get("requirements", []))
            ),
            evidence_blocks(cited),
        ]
    )
    return [system("draft", schema=DRAFT_SCHEMA), Message("user", user)]


def agent_messages(question: str, tools: dict[str, dict], state: str) -> list[Message]:
    catalog = "\n".join(
        f"- {name}: {spec['description']} args={json.dumps(spec['args'])}"
        for name, spec in tools.items()
    )
    user = "\n".join(
        [
            "Answer the question using the tools. One tool call per reply.",
            "Search before you answer, read what you find, and call finish once you can "
            "answer with quotes from the evidence.",
            "",
            block("QUESTION", question),
            block("TOOLS", catalog),
            block("STATE", state or "(nothing yet)"),
        ]
    )
    return [system("agent", schema=AGENT_SCHEMA), Message("user", user)]


def repair_message(errors: str) -> Message:
    """Hand the validation errors back and ask for the same object, fixed."""
    return Message(
        "user",
        "Your last reply did not fit the schema:\n"
        f"{errors}\n"
        "Send the corrected JSON object only, with no other text.",
    )
