"""The eval harness: the same questions every time, scored without a model in the loop.

Five things are measured over a folder of postings with hand-written labels:

* **fields** - did the extractor get the title, company, locations, deadline and requirement
  count right?
* **verdicts** - does the pipeline agree with the label on requirements a person judged?
* **citations** - of the verdicts that claimed support, how many had a quote that verified?
* **refusals** - on requirements the corpus genuinely does not cover, does it say missing?
* **recall@k** - did retrieval surface the evidence the label points at?

Every score is computed by comparison, never by asking a model to grade itself, so a run is
a regression test: with the offline provider it is reproducible and free, and with a hosted
model the same numbers say whether the model is worth its cost.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Evidence
from .draft import draft_letter
from .match import match_posting
from .posting import extract_posting
from .retrieval import Index
from .verify import normalize

METRICS = ("fields", "verdicts", "citations", "refusals", "recall_at_k")


@dataclass
class Score:
    """A hit rate, kept as a fraction so it can be reported and compared."""

    right: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.right += bool(ok)
        self.total += 1

    @property
    def rate(self) -> float:
        return self.right / self.total if self.total else 0.0

    def __str__(self) -> str:
        return f"{self.right}/{self.total} ({100 * self.rate:.0f}%)"


@dataclass
class EvalReport:
    provider: str
    model: str
    scores: dict[str, Score]
    cases: list[dict] = field(default_factory=list)
    draft_flags: int = 0

    def metrics(self) -> dict[str, float]:
        return {name: round(score.rate, 4) for name, score in self.scores.items()}

    def table(self) -> str:
        lines = [f"provider {self.provider} · model {self.model}", ""]
        labels = {
            "fields": "posting fields extracted",
            "verdicts": "verdicts agreeing with labels",
            "citations": "claims whose quote verified",
            "refusals": "gaps correctly called missing",
            "recall_at_k": "labelled evidence retrieved",
        }
        for name, score in self.scores.items():
            lines.append(f"  {labels.get(name, name):<34}{str(score):>14}")
        lines.append(f"  {'unsupported terms in drafts':<34}{self.draft_flags:>14}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "metrics": self.metrics(),
            "counts": {name: [score.right, score.total] for name, score in self.scores.items()},
            "draft_flags": self.draft_flags,
            "cases": self.cases,
        }


def load_labels(path: str | os.PathLike) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "postings" not in data:
        raise ValueError(f"{path} has no 'postings' list")
    return data


def _field_checks(posting: dict, expected: dict) -> list[tuple[str, bool]]:
    checks = []
    for key in ("title", "company", "deadline"):
        if key in expected:
            got, want = posting.get(key), expected[key]
            same = normalize(str(got or "")) == normalize(str(want or ""))
            checks.append((key, same))
    if "locations" in expected:
        got = {normalize(x) for x in posting.get("locations") or []}
        want = {normalize(x) for x in expected["locations"]}
        checks.append(("locations", got == want))
    if "requirements" in expected:
        checks.append(
            ("requirements", len(posting.get("requirements", [])) == expected["requirements"])
        )
    return checks


def run_evals(
    provider,
    index: Index,
    pieces: dict[str, Evidence],
    *,
    postings_dir: str | os.PathLike,
    labels: dict,
    k: int = 6,
    draft: bool = True,
) -> EvalReport:
    """Run every labelled posting through the pipeline and score what comes out."""
    scores = {name: Score() for name in METRICS}
    report = EvalReport(
        provider=getattr(provider, "name", "?"),
        model=getattr(provider, "model", "?"),
        scores=scores,
    )
    folder = Path(postings_dir)
    for entry in labels["postings"]:
        path = folder / entry["file"]
        posting = extract_posting(provider, path.read_text(encoding="utf-8"))
        case: dict = {"file": entry["file"], "fields": {}, "requirements": []}
        for key, ok in _field_checks(posting, entry.get("fields", {})):
            scores["fields"].add(ok)
            case["fields"][key] = ok

        coverage = match_posting(provider, index, posting, k=k)
        found = {normalize(j.requirement): j for j in coverage.judgments}
        for label in entry.get("requirements", []):
            prefix = normalize(label["starts_with"])
            judgment = next((j for text, j in found.items() if text.startswith(prefix)), None)
            if judgment is None:
                scores["verdicts"].add(False)
                case["requirements"].append({"starts_with": label["starts_with"], "found": False})
                continue
            agrees = judgment.final == label["verdict"]
            scores["verdicts"].add(agrees)
            if label["verdict"] == "missing":
                scores["refusals"].add(judgment.final == "missing")
            if label.get("evidence_from"):
                sources = {hit.evidence.source for hit in judgment.hits}
                scores["recall_at_k"].add(bool(sources & set(label["evidence_from"])))
            case["requirements"].append(
                {
                    "starts_with": label["starts_with"],
                    "found": True,
                    "want": label["verdict"],
                    "got": judgment.final,
                    "verified": judgment.check.ok,
                }
            )
        for judgment in coverage.judgments:
            if judgment.verdict in {"covered", "partial"}:
                scores["citations"].add(judgment.check.ok)

        case["coverage"] = coverage.counts()
        case["must_have_rate"] = round(coverage.rate, 3)
        if draft and any(j.final == "covered" and j.check.ok for j in coverage.judgments):
            letter = draft_letter(provider, posting, coverage, pieces)
            report.draft_flags += len(letter.flags)
            case["draft_flags"] = letter.flags
        report.cases.append(case)
    return report
