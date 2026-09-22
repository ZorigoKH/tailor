"""Drafting from evidence only, and flagging whatever the evidence does not cover.

The model is handed the role, the requirements and the verified evidence - nothing else, not
even the rest of the corpus - and asked for prose. Then every number and every name in what
comes back is checked against that evidence: anything unaccounted for is listed as a flag,
which is the human's cue to fix it or cut it. A draft with flags is not blocked, it is
labelled; the person sending it decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts
from .ask import ask_json
from .corpus import Evidence
from .match import Coverage
from .verify import unsupported_terms


@dataclass
class Draft:
    paragraphs: list[str]
    citations: list[str]
    flags: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list, repr=False)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def clean(self) -> bool:
        return not self.flags


def allowed_terms(posting: dict) -> list[str]:
    """Names the draft may use because the posting itself supplies them."""
    terms = [str(posting.get(key) or "") for key in ("title", "company", "team", "employment_type")]
    terms += [str(location) for location in posting.get("locations") or []]
    return [term for term in terms if term]


def draft_letter(
    provider,
    posting: dict,
    coverage: Coverage,
    pieces: dict[str, Evidence],
    *,
    paragraphs: int = 4,
    limit: int = 6,
    tracer=None,
) -> Draft:
    """Write the body paragraphs of a cover letter from verified evidence."""
    cited = coverage.cited(pieces, limit=limit)
    if not cited:
        raise ValueError(
            "no verified evidence to write from: run match first, or add material to the corpus"
        )
    reply = ask_json(
        provider,
        prompts.draft_messages(posting, cited, paragraphs=paragraphs),
        prompts.DRAFT_SCHEMA,
        max_tokens=1800,
    )
    texts = [piece.quotable for piece in cited]
    body = "\n\n".join(reply["paragraphs"])
    flags = unsupported_terms(body, texts, allow=allowed_terms(posting))
    draft = Draft(
        paragraphs=[p.strip() for p in reply["paragraphs"]],
        citations=reply.get("citations") or [piece.id for piece in cited],
        flags=flags,
        evidence=[piece.id for piece in cited],
    )
    if tracer is not None:
        tracer.event(
            "draft",
            paragraphs=len(draft.paragraphs),
            evidence=draft.evidence,
            flags=flags,
            words=len(body.split()),
        )
    return draft
