"""The part that does not trust the model.

Two checks, both deterministic, both cheap:

* :func:`check_quote` - a quote has to appear word for word in the evidence it cites.
  Whitespace and curly quotes are normalized, and an ellipsis splits a quote into parts that
  must each appear, in order. Anything else is not a quote, and the claim resting on it is
  marked unsupported instead of being shown as fact.
* :func:`unsupported_terms` - every number and every name in a draft has to appear in the
  evidence the draft was written from, or in the allow list (the role, the company, the
  city). What is left is exactly what a person should read before sending anything.

Neither check asks a model whether the model was right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUMBER = re.compile(r"(?<![\w$])\$?\d[\d,]*(?:\.\d+)?%?\+?")
NAME = re.compile(r"\b[A-Z][\w&.'’-]*(?:\s+[A-Z][\w&.'’-]*)*\b")

# Words that start sentences or stand in for names; never worth flagging on their own.
COMMON = frozenset(
    """a an the this that these those i we my our it they he she you your there their in on at
    for and but or if so as to of with without before after now right every each most some then
    when while since because both either neither dear sincerely thank thanks yours best regards
    hello hi please""".split()
)

TRANSLATE = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
}


def normalize(text: str) -> str:
    """Casefolded text with collapsed whitespace and straightened punctuation."""
    for fancy, plain in TRANSLATE.items():
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass
class QuoteCheck:
    ok: bool
    reason: str

    def __bool__(self) -> bool:
        return self.ok


def check_quote(quote: str, citations: list[str], sources: dict[str, str]) -> QuoteCheck:
    """Is ``quote`` really in one of the cited sources?"""
    if not quote.strip():
        return QuoteCheck(False, "no quote given")
    if not citations:
        return QuoteCheck(False, "no evidence cited")
    unknown = [c for c in citations if c not in sources]
    if unknown:
        return QuoteCheck(False, f"cites evidence that does not exist: {', '.join(unknown)}")
    parts = [normalize(part) for part in re.split(r"\.{3}|…", quote) if part.strip()]
    for citation in citations:
        haystack = normalize(sources[citation])
        position, ok = 0, True
        for part in parts:
            found = haystack.find(part, position)
            if found < 0:
                ok = False
                break
            position = found + len(part)
        if ok:
            return QuoteCheck(True, f"quote verified against {citation}")
    return QuoteCheck(False, "the quote is not in the cited evidence")


def unsupported_terms(
    draft: str, evidence: list[str], allow: list[str] | tuple[str, ...] = ()
) -> list[str]:
    """Numbers and names in ``draft`` that no evidence (or allowed term) backs up."""
    haystack = normalize(" ".join(list(evidence) + list(allow)))
    flagged: set[str] = set()

    for match in NUMBER.finditer(draft):
        token = match.group(0)
        if normalize(token.strip("$%+,")) not in haystack:
            flagged.add(token)

    for sentence in re.split(r"(?<=[.!?])\s+", draft):
        for match in NAME.finditer(sentence):
            words = match.group(0).split()
            while words and words[0].casefold() in COMMON:
                words.pop(0)  # "At Initech" -> "Initech"
            while words and words[-1].casefold() in COMMON:
                words.pop()  # "Initech I" -> "Initech"
            if not words:
                continue
            name = " ".join(words)
            if len(words) == 1:
                if name.casefold() in COMMON or len(name) < 3:
                    continue
                if sentence.startswith(name):
                    continue  # a capital at the start of a sentence says nothing
            if normalize(name) not in haystack:
                flagged.add(name)
    return sorted(flagged)
