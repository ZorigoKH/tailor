"""Posting -> match -> draft, with the verification in between doing its job."""

import json

import pytest

from tailor import (
    Tracer,
    draft_letter,
    extract_posting,
    judge_requirement,
    match_posting,
    read_posting,
)
from tailor.posting import must_haves, summarize
from tailor.schema import SchemaError


def test_extract_posting_fills_defaults(offline, ai_posting):
    posting = extract_posting(offline, ai_posting)
    assert posting["company"] == "Northwind Consulting" and len(posting["requirements"]) == 7
    assert len(must_haves(posting)) == 5
    assert summarize(posting).startswith(
        "AI Engineering Intern (Summer 2027) at Northwind Consulting"
    )
    assert "apply by 2026-10-09" in summarize(posting)
    with pytest.raises(ValueError, match="empty"):
        extract_posting(offline, "   ")


def test_extract_posting_rejects_a_reply_that_never_fits(scripted):
    with pytest.raises(SchemaError, match="never matched"):
        extract_posting(scripted("{}", '{"title": "x"}'), "Some posting text", retries=1)


def test_read_posting_takes_files_stdin_or_text(tmp_path, monkeypatch):
    path = tmp_path / "p.txt"
    path.write_text("from a file", encoding="utf-8")
    assert read_posting(str(path)) == "from a file"
    assert read_posting("raw text") == "raw text"
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("from stdin"))
    assert read_posting("-") == "from stdin"


def test_judgment_is_verified_against_the_source(offline, index):
    judgment = judge_requirement(offline, index, {"text": "Python with pandas", "kind": "must"})
    assert judgment.final == "covered" and judgment.check.ok
    assert judgment.quote and judgment.citations == ["talentsnipe.md:8-8"]


def test_an_invented_quote_becomes_unsupported(scripted, index):
    fake = json.dumps(
        {
            "verdict": "covered",
            "citations": ["talentsnipe.md:8-8"],
            "quote": "led a team of forty engineers",
            "why": "it says so",
        }
    )
    judgment = judge_requirement(
        scripted(fake), index, {"text": "Python with pandas", "kind": "must"}
    )
    assert judgment.verdict == "covered" and judgment.flagged and judgment.final == "unsupported"
    assert judgment.check.reason == "the quote is not in the cited evidence"

    ghost = json.dumps(
        {"verdict": "partial", "citations": ["ghost.md:1-1"], "quote": "x", "why": ""}
    )
    judgment = judge_requirement(
        scripted(ghost), index, {"text": "Python with pandas", "kind": "must"}
    )
    assert judgment.final == "unsupported" and "does not exist" in judgment.check.reason


def test_missing_needs_no_quote(scripted, index):
    reply = json.dumps({"verdict": "missing", "citations": [], "quote": "", "why": "nothing"})
    judgment = judge_requirement(scripted(reply), index, {"text": "Python", "kind": "nice"})
    assert judgment.final == "missing" and judgment.check.ok and not judgment.flagged


def test_coverage_counts_rate_gaps_and_trace(offline, index, ai_posting):
    tracer = Tracer()
    posting = extract_posting(offline, ai_posting)
    coverage = match_posting(offline, index, posting, tracer=tracer)
    counts = coverage.counts()
    assert sum(counts.values()) == 7 and counts["unsupported"] == 0
    assert 0 < coverage.rate <= 1
    assert all(j.final in {"missing", "unsupported"} for j in coverage.gaps)
    assert [e["kind"] for e in tracer.events].count("judgment") == 7
    assert tracer.summary()["calls"] == 0  # the provider was not wrapped, so no model events
    rows = coverage.rows()
    assert len(rows) == 7 and rows[0][0] == "covered"


def test_cited_evidence_is_ordered_best_first_and_verified_only(offline, index, lookup, ai_posting):
    coverage = match_posting(offline, index, extract_posting(offline, ai_posting))
    cited = coverage.cited(lookup)
    assert cited and len(cited) <= 6
    finals = {j.final for j in coverage.judgments if j.citations and j.citations[0] == cited[0].id}
    assert "covered" in finals


def test_draft_flags_what_the_evidence_does_not_back(offline, scripted, index, lookup, ai_posting):
    posting = extract_posting(offline, ai_posting)
    coverage = match_posting(offline, index, posting)
    tracer = Tracer()
    letter = draft_letter(offline, posting, coverage, lookup, tracer=tracer)
    assert letter.clean and len(letter.paragraphs) >= 2 and letter.evidence
    assert "Northwind Consulting" in letter.text
    assert tracer.events[-1]["kind"] == "draft"

    invented = json.dumps(
        {"paragraphs": ["I am applying to Northwind Consulting.", "At Initech I led 40 people."]}
    )
    letter = draft_letter(scripted(invented), posting, coverage, lookup)
    assert letter.flags == ["40", "Initech"] and not letter.clean


def test_draft_needs_verified_evidence(scripted, index, lookup, ai_posting, offline):
    posting = extract_posting(offline, ai_posting)
    nothing = json.dumps({"verdict": "missing", "citations": [], "quote": "", "why": "no"})
    coverage = match_posting(scripted(*[nothing] * 7), index, posting)
    with pytest.raises(ValueError, match="no verified evidence"):
        draft_letter(offline, posting, coverage, lookup)
