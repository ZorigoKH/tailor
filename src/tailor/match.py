"""Requirement by requirement: what the evidence shows, and what it does not.

For each requirement the index retrieves the most relevant pieces of your corpus, the model
says covered / partial / missing and quotes the words it relied on, and then
:func:`tailor.verify.check_quote` decides whether that quote is real. A verdict whose quote
is not in the cited evidence becomes ``unsupported`` - not a pass, not a silent failure, a
flag with a reason.

The gaps are the useful half: they are the requirements to either learn or leave out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts
from .ask import ask_json
from .corpus import Evidence
from .retrieval import Hit, Index
from .verify import QuoteCheck, check_quote

VERDICTS = ("covered", "partial", "missing", "unsupported")


@dataclass
class Judgment:
    """One requirement, one verdict, and the evidence behind it."""

    requirement: str
    kind: str
    verdict: str
    citations: list[str]
    quote: str
    why: str
    check: QuoteCheck
    hits: list[Hit] = field(default_factory=list, repr=False)

    @property
    def flagged(self) -> bool:
        """True when the model claimed support that the quote check could not confirm."""
        return self.verdict in {"covered", "partial"} and not self.check.ok

    @property
    def final(self) -> str:
        return "unsupported" if self.flagged else self.verdict


@dataclass
class Coverage:
    """Every judgment for one posting, plus the counts worth printing."""

    posting: dict
    judgments: list[Judgment]

    def counts(self) -> dict[str, int]:
        return {
            verdict: sum(1 for j in self.judgments if j.final == verdict) for verdict in VERDICTS
        }

    @property
    def rate(self) -> float:
        """Share of must-have requirements that are covered, with a verified quote."""
        musts = [j for j in self.judgments if j.kind == "must"]
        if not musts:
            return 0.0
        return sum(1 for j in musts if j.final == "covered") / len(musts)

    @property
    def gaps(self) -> list[Judgment]:
        return [j for j in self.judgments if j.final in {"missing", "unsupported"}]

    @property
    def flagged(self) -> list[Judgment]:
        return [j for j in self.judgments if j.flagged]

    def cited(self, pieces: dict[str, Evidence], *, limit: int = 6) -> list[Evidence]:
        """The strongest verified evidence, best first, for the drafting step."""
        order = {"covered": 0, "partial": 1}
        out: list[Evidence] = []
        for judgment in sorted(self.judgments, key=lambda j: order.get(j.final, 9)):
            if judgment.final not in order or not judgment.check.ok:
                continue
            for citation in judgment.citations:
                piece = pieces.get(citation)
                if piece is not None and piece not in out:
                    out.append(piece)
        return out[:limit]

    def rows(self) -> list[tuple[str, str, str, str]]:
        return [
            (
                j.final,
                j.kind,
                j.requirement,
                ", ".join(j.citations) if j.check.ok else j.check.reason,
            )
            for j in self.judgments
        ]


def judge_requirement(provider, index: Index, requirement: dict, *, k: int = 6) -> Judgment:
    """Retrieve, judge, then verify the quote against the cited evidence."""
    hits = index.search(requirement["text"], k=k)
    if not hits:
        return Judgment(
            requirement=requirement["text"],
            kind=requirement["kind"],
            verdict="missing",
            citations=[],
            quote="",
            why="nothing in the corpus matched this requirement",
            check=QuoteCheck(True, "no claim to check"),
        )
    reply = ask_json(
        provider, prompts.judge_messages(requirement["text"], hits), prompts.JUDGE_SCHEMA
    )
    sources = {hit.id: hit.evidence.quotable for hit in hits}
    verdict = reply["verdict"]
    citations = [c for c in reply["citations"] if c]
    check = (
        QuoteCheck(True, "no claim to check")
        if verdict == "missing"
        else check_quote(reply["quote"], citations, sources)
    )
    return Judgment(
        requirement=requirement["text"],
        kind=requirement["kind"],
        verdict=verdict,
        citations=citations,
        quote=reply["quote"],
        why=reply["why"],
        check=check,
        hits=hits,
    )


def match_posting(provider, index: Index, posting: dict, *, k: int = 6, tracer=None) -> Coverage:
    """Judge every requirement in a posting."""
    judgments = []
    for requirement in posting["requirements"]:
        judgment = judge_requirement(provider, index, requirement, k=k)
        if tracer is not None:
            tracer.event(
                "judgment",
                requirement=judgment.requirement[:120],
                requirement_kind=judgment.kind,
                verdict=judgment.final,
                citations=judgment.citations,
                verified=judgment.check.ok,
                reason=judgment.check.reason,
            )
        judgments.append(judgment)
    return Coverage(posting=posting, judgments=judgments)
