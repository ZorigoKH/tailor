"""BM25, hashed vectors and rank fusion, each checked against its own arithmetic."""

import math

import numpy as np
import pytest

from tailor import Bm25, Index, hashing_embed, rrf, tokenize
from tailor.corpus import chunk_markdown
from tailor.retrieval import stem


def test_tokenize_stems_and_keeps_compound_tokens():
    assert tokenize("Testing the tests we tested") == ["test", "test", "test"]
    assert tokenize("SOC-CMM, Next.js and CI/CD") == ["soc-cmm", "next.js", "ci", "cd"]
    assert stem("policies") == "policy" and stem("services") == stem("service") == "service"
    assert stem("APIs".lower()) == "api" and stem("orchestration") == "orchestrate"
    assert stem("class") == "class" and stem("its") == "its"


def test_bm25_matches_the_formula_by_hand():
    docs = [["apple", "pie"], ["apple", "apple", "tart", "crust"], ["bread"]]
    scores = Bm25(docs, k1=1.5, b=0.75).scores(["apple"])
    n, df = 3, 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    avg = (2 + 4 + 1) / 3

    def expected(f, length):
        return idf * f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * length / avg))

    np.testing.assert_allclose(scores, [expected(1, 2), expected(2, 4), 0.0])
    assert scores[1] > scores[0]  # two mentions beat one, even in a longer document


def test_rrf_sums_reciprocal_ranks():
    fused = rrf([["a", "b", "c"], ["b", "a"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert sorted(fused, key=fused.get, reverse=True)[-1] == "c"


def test_hashing_vectors_are_deterministic_and_unit_length():
    a, b = hashing_embed(["alpha beta"]), hashing_embed(["alpha beta", "gamma"])
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(np.linalg.norm(b, axis=1), [1.0, 1.0], rtol=1e-6)
    assert float(b[0] @ b[1]) == pytest.approx(0.0, abs=0.05)  # no shared words, ~no similarity
    assert np.all(hashing_embed([""]) == 0)


def _index(embedder=None):
    doc = (
        "# Notes\n\n"
        "Python and pandas pipelines that clean recruitment data from more than one source.\n\n"
        "Volleyball on Tuesdays and Thursdays with the varsity team, all season long.\n\n"
        "Wrote the SOC-CMM audit workpapers for the bank engagement and the closing report.\n"
    )
    return Index(chunk_markdown(doc, "n.md"), embedder=embedder)


def test_keyword_search_and_found_by():
    hits = _index().search("pandas pipeline")
    assert hits[0].id == "n.md:3-3" and hits[0].found_by == "keyword"
    assert len(hits) == 1  # BM25 only returns pieces that share a term
    assert _index().search("nothing relevant here") == []


def test_hybrid_search_fuses_both_rankings():
    index = _index(embedder=hashing_embed)
    assert index.has_vectors
    hits = index.search("SOC-CMM workpapers", k=2)
    assert hits[0].id == "n.md:7-7" and hits[0].found_by == "both"
    assert [h.id for h in index.search("pandas", k=1)] == ["n.md:3-3"]


def test_index_rejects_bad_inputs():
    with pytest.raises(ValueError, match="empty"):
        Index([])
    with pytest.raises(ValueError, match="wrong number of vectors"):
        _index(embedder=lambda texts: np.zeros((1, 4)))


def test_bundled_corpus_retrieves_the_obvious_piece(index):
    top = index.search("row-level security policies", k=1)[0]
    assert top.evidence.source == "terran.md"
    top = index.search("SOC-CMM framework bank", k=1)[0]
    assert top.evidence.source == "ey.md"
