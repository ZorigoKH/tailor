from pathlib import Path

import pytest

from tailor import Index, Offline, by_id, hashing_embed, load_corpus
from tailor.providers import Completion, Usage

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
POSTINGS = ROOT / "evals" / "postings"
LABELS = ROOT / "evals" / "labels.json"


class Scripted:
    """A provider that replies with the given texts, in order, whatever it is asked."""

    name = "scripted"
    model = "scripted"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list] = []

    def complete(self, messages, *, json_schema=None, temperature=0.0, max_tokens=1200):
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("the script ran out of replies")
        text = self.replies.pop(0)
        return Completion(text, Usage(10, 5), 0.0, self.model)

    def embed(self, texts):
        return hashing_embed(texts)


@pytest.fixture(scope="session")
def pieces():
    return load_corpus(EVIDENCE)


@pytest.fixture(scope="session")
def lookup(pieces):
    return by_id(pieces)


@pytest.fixture(scope="session")
def index(pieces):
    return Index(pieces, embedder=hashing_embed)


@pytest.fixture
def offline():
    return Offline()


@pytest.fixture(scope="session")
def ai_posting():
    return (POSTINGS / "ai-engineering-intern.txt").read_text(encoding="utf-8")


@pytest.fixture
def scripted():
    return Scripted
