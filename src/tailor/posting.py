"""Reading a job posting into fields you can check off.

The posting is the only untrusted text in the pipeline, so nothing it says is executed and
everything it produces is validated: one schema-constrained call, then :mod:`tailor.schema`
decides whether the result is usable. A posting is a plain dict, so a run can be saved,
reread and diffed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import prompts
from .ask import ask_json


def read_posting(source: str) -> str:
    """Posting text from a file path, ``-`` for standard input, or the text itself."""
    if source == "-":
        return sys.stdin.read()
    path = Path(source)
    if len(source) < 4096 and path.exists():
        return path.read_text(encoding="utf-8")
    return source


def extract_posting(provider, text: str, *, retries: int = 1) -> dict:
    """Title, company, locations, deadline and one entry per requirement."""
    if not text.strip():
        raise ValueError("the posting is empty")
    posting = ask_json(
        provider,
        prompts.posting_messages(text),
        prompts.POSTING_SCHEMA,
        retries=retries,
        max_tokens=2400,
    )
    posting.setdefault("notes", [])
    posting.setdefault("team", None)
    posting.setdefault("employment_type", None)
    posting.setdefault("deadline", None)
    for requirement in posting["requirements"]:
        requirement.setdefault("skills", [])
    return posting


def must_haves(posting: dict) -> list[dict]:
    return [r for r in posting["requirements"] if r["kind"] == "must"]


def save_posting(posting: dict, path: str | os.PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(posting, indent=2) + "\n", encoding="utf-8")
    return out


def summarize(posting: dict) -> str:
    """One line for the terminal."""
    where = ", ".join(posting.get("locations") or []) or "location not stated"
    parts = [f"{posting.get('title', '?')} at {posting.get('company', '?')}", where]
    if posting.get("team"):
        parts.append(str(posting["team"]))
    if posting.get("deadline"):
        parts.append(f"apply by {posting['deadline']}")
    musts = len(must_haves(posting))
    parts.append(f"{len(posting['requirements'])} requirements ({musts} must)")
    return " · ".join(parts)
