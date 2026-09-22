"""Hybrid retrieval: BM25 for the words, embeddings for the meaning, fused by rank.

Keyword search is the one that finds "SOC-CMM" and "Newey-West". Embeddings are the one
that finds "shipped to production" when the evidence says "launched". Neither is reliable
alone, so both run and the two rankings are combined with reciprocal rank fusion (Cormack,
Clarke and Buettcher, 2009), which needs no score calibration between them: an item scores
the sum of 1/(60 + rank) over the lists it appears in.

The embedder is optional. With no API key, :func:`hashing_embed` gives deterministic
offline vectors - weak, but enough to keep the pipeline and its tests honest.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .corpus import Evidence

RRF_K = 60
POOL = 24  # how deep each ranking goes before fusion

STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have how i if in into is it its me
    my of on or our that the their then there these they this to us was we were what when which
    who will with you your""".split()
)

TOKEN = re.compile(r"[a-z0-9]+(?:[-_.+][a-z0-9]+)*")
MIN_SIMILARITY = 0.1  # below this, a hashed-vector match is collision noise


def stem(token: str) -> str:
    """A light stemmer: enough to make tests, tested and testing one word.

    Plurals first (Porter's step 1a), then one derivational suffix. Crude, but applied to
    queries and documents alike, which is all a retriever needs.
    """
    if not token.isalpha() or len(token) <= 3:
        return token
    if token.endswith("sses"):
        token = token[:-2]
    elif token.endswith("ies"):
        token = token[:-3] + "y"
    elif token.endswith("ss"):
        pass
    elif token.endswith("s"):
        token = token[:-1]
    for suffix, replacement in (("ations", "ate"), ("ation", "ate"), ("ing", ""), ("ed", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, stemmed tokens, keeping things like ``soc-cmm`` and ``next.js`` whole."""
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class Bm25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, documents: list[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.documents = documents
        self.lengths = np.array([len(d) for d in documents], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(documents) else 0.0
        self.frequencies: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for tokens in documents:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.frequencies.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        n = len(documents)
        self.idf = {
            token: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for token, df in document_frequency.items()
        }

    def scores(self, query: Iterable[str]) -> np.ndarray:
        out = np.zeros(len(self.documents))
        for token in query:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for i, counts in enumerate(self.frequencies):
                f = counts.get(token, 0)
                if not f:
                    continue
                norm = 1 - self.b + self.b * (self.lengths[i] / (self.avg_length or 1.0))
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


def hashing_embed(texts: list[str], dim: int = 16384) -> np.ndarray:
    """Deterministic bag-of-words vectors: no model, no network, same answer every run."""
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token, count in _counts(tokenize(text)).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 else -1.0
            matrix[row, index] += sign * (1 + math.log(count))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion: sum 1/(k + rank) across rankings, rank counted from 1."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass
class Hit:
    """One retrieved piece of evidence, and which retriever found it."""

    evidence: Evidence
    score: float
    found_by: str  # "keyword", "semantic" or "both"

    @property
    def id(self) -> str:
        return self.evidence.id


class Index:
    """Searchable corpus. Embeddings are optional; without them this is BM25 alone."""

    def __init__(
        self, pieces: list[Evidence], *, embedder: Callable[[list[str]], np.ndarray] | None = None
    ):
        if not pieces:
            raise ValueError("the corpus is empty")
        self.pieces = pieces
        self.ids = [piece.id for piece in pieces]
        self.bm25 = Bm25([tokenize(f"{p.title} {p.text}") for p in pieces])
        self.embedder = embedder
        self.vectors: np.ndarray | None = None
        if embedder is not None:
            self.vectors = np.asarray(
                embedder([f"{p.title}. {p.text}" for p in pieces]), dtype=float
            )
            if self.vectors.shape[0] != len(pieces):
                raise ValueError("the embedder returned the wrong number of vectors")

    @property
    def has_vectors(self) -> bool:
        return self.vectors is not None

    def _keyword(self, query: str, pool: int) -> list[str]:
        scores = self.bm25.scores(tokenize(query))
        order = np.argsort(-scores)[:pool]
        return [self.ids[i] for i in order if scores[i] > 0]

    def _semantic(self, query: str, pool: int) -> list[str]:
        if self.vectors is None or self.embedder is None:
            return []
        vector = np.asarray(self.embedder([query]), dtype=float)[0]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        similarity = self.vectors @ (vector / norm)
        order = np.argsort(-similarity)[:pool]
        return [self.ids[i] for i in order if similarity[i] >= MIN_SIMILARITY]

    def search(self, query: str, *, k: int = 6, pool: int = POOL) -> list[Hit]:
        """Top ``k`` pieces for ``query``, fusing the keyword and semantic rankings."""
        keyword = self._keyword(query, pool)
        semantic = self._semantic(query, pool)
        rankings = [r for r in (keyword, semantic) if r]
        if not rankings:
            return []
        fused = rrf(rankings)
        pieces = {piece.id: piece for piece in self.pieces}
        hits = []
        for piece_id, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]:
            in_keyword, in_semantic = piece_id in keyword, piece_id in semantic
            found_by = (
                "both" if in_keyword and in_semantic else "keyword" if in_keyword else "semantic"
            )
            hits.append(Hit(evidence=pieces[piece_id], score=round(score, 6), found_by=found_by))
        return hits
