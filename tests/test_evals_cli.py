"""The eval harness and the command line, all offline."""

import json

import pytest
from conftest import LABELS, POSTINGS

import tailor.cli as cli
from tailor import run_evals
from tailor.evals import load_labels

# The numbers the README quotes for the offline stand-in. Change the rules, change these.
OFFLINE_SCORES = {
    "fields": (15, 15),
    "verdicts": (11, 19),
    "citations": (12, 12),
    "refusals": (3, 4),
    "recall_at_k": (15, 15),
}


def test_offline_eval_scores_are_pinned(offline, index, lookup):
    report = run_evals(offline, index, lookup, postings_dir=POSTINGS, labels=load_labels(LABELS))
    got = {name: (score.right, score.total) for name, score in report.scores.items()}
    assert got == OFFLINE_SCORES
    assert report.draft_flags == 0
    assert report.metrics()["citations"] == 1.0 and report.metrics()["fields"] == 1.0
    data = report.to_json()
    assert data["counts"]["verdicts"] == [11, 19] and len(data["cases"]) == 3
    assert "posting fields extracted" in report.table()


def test_labels_must_have_postings(tmp_path):
    bad = tmp_path / "labels.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'postings'"):
        load_labels(bad)


def test_index_command(capsys):
    assert cli.main(["index", "--corpus", "evidence", "--search", "SOC-CMM"]) == 0
    out = capsys.readouterr().out
    assert "pieces from 10 files" in out and "ey.md" in out and "BM25 + embeddings" in out


def test_posting_and_match_commands(capsys, tmp_path):
    posting = str(POSTINGS / "ai-engineering-intern.txt")
    assert cli.main(["posting", posting, "--json", str(tmp_path / "p.json")]) == 0
    assert "(must) Proficiency in Python [python]" in capsys.readouterr().out
    saved = json.loads((tmp_path / "p.json").read_text())
    assert saved["deadline"] == "2026-10-09"

    out_json = tmp_path / "m.json"
    assert (
        cli.main(
            [
                "match",
                posting,
                "--corpus",
                "evidence",
                "--json",
                str(out_json),
                "--trace",
                str(tmp_path),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "must-have coverage" in text and "gaps to close or leave out" in text
    result = json.loads(out_json.read_text())
    assert len(result["judgments"]) == 7 and result["run"]["calls"] == 8
    traces = list(tmp_path.glob("match-*.jsonl"))
    assert len(traces) == 1
    kinds = [json.loads(line)["kind"] for line in traces[0].read_text().splitlines()]
    assert kinds.count("model") == 8 and kinds.count("judgment") == 7


def test_draft_ask_and_eval_commands(capsys, tmp_path):
    posting = str(POSTINGS / "backend-intern.txt")
    assert cli.main(["draft", posting, "--out", str(tmp_path / "d.txt")]) == 0
    out = capsys.readouterr().out
    assert "traces to your corpus" in out and (tmp_path / "d.txt").read_text().startswith(
        "I am applying"
    )

    assert cli.main(["ask", "What shows I can test code?"]) == 0
    assert "quote verified against" in capsys.readouterr().out

    assert cli.main(["eval", "--json", str(tmp_path / "e.json")]) == 0
    assert "claims whose quote verified" in capsys.readouterr().out
    assert cli.main(["eval", "--no-draft", "--fail-under", "0.99"]) == 1
    assert "below the floor" in capsys.readouterr().err


def test_errors_are_one_line(capsys):
    assert cli.main(["index", "--corpus", "no-such-folder"]) == 1
    assert capsys.readouterr().err.startswith("tailor: no .md or .txt files")
    assert cli.main(["posting", "   "]) == 1
    assert capsys.readouterr().err == "tailor: the posting is empty\n"
