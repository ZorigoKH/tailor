"""Chunking your material into quotable pieces with stable ids."""

import pytest

from tailor import chunk_markdown, load_corpus

DOC = """---
title: Acme
tags: engineering, payments
---

# Backend engineer, 2025

Built the billing service that every customer invoice goes through, end to end.

- Cut invoice latency in half by batching the ledger writes.
- Added tests.
- Wrote the on-call runbook that the team still uses for every incident today.

## Earlier

Short line.
Another short line that goes with it.
"""


def test_front_matter_headings_bullets_and_line_spans():
    pieces = chunk_markdown(DOC, "acme.md")
    assert [p.id for p in pieces] == [
        "acme.md:8-8",
        "acme.md:10-11",
        "acme.md:12-12",
        "acme.md:16-17",
    ]
    assert pieces[0].title == "Acme - Backend engineer, 2025"
    assert pieces[0].tags == ("engineering", "payments")
    assert pieces[0].text.startswith("Built the billing service")
    # the short bullet was merged into its neighbour, and bullets lost their markers
    assert (
        pieces[1].text == "Cut invoice latency in half by batching the ledger writes. Added tests."
    )
    assert pieces[2].text.startswith("Wrote the on-call runbook")
    assert pieces[3].title == "Acme - Earlier"
    assert pieces[3].lines == (16, 17)


def test_quotable_text_carries_the_heading():
    piece = chunk_markdown(DOC, "acme.md")[2]
    assert piece.quotable.startswith("Acme - Backend engineer, 2025. Wrote the on-call runbook")
    assert piece.words == 13


def test_plain_text_without_front_matter_uses_the_file_name():
    pieces = chunk_markdown(
        "Just one paragraph of ordinary text that is long enough.", "notes-2026.txt"
    )
    assert pieces[0].title == "notes 2026" and pieces[0].id == "notes-2026.txt:1-1"


def test_the_bundled_corpus_loads_with_unique_ids(pieces):
    ids = [p.id for p in pieces]
    assert len(ids) == len(set(ids)) and len(ids) >= 30
    assert {p.source for p in pieces} >= {"terran.md", "ey.md", "tailor.md", "education.md"}


def test_errors_name_the_problem(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .md or .txt files"):
        load_corpus(tmp_path)
    (tmp_path / "empty.md").write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no readable text"):
        load_corpus(tmp_path)
