"""tailor: match a job posting against evidence from your own corpus, and draft only what
the evidence supports.

The pipeline is deliberately boring in the middle and strict at the edges: a posting is read
into validated fields, requirements are answered from retrieved evidence, and every quote is
checked word for word against the source it cites before anything is shown as a fact.
"""

from .corpus import Evidence, by_id, chunk_markdown, load_corpus
from .retrieval import Bm25, Hit, Index, hashing_embed, rrf, tokenize
from .schema import SchemaError, validate

__version__ = "0.1.0"
__all__ = [
    "Bm25",
    "Evidence",
    "Hit",
    "Index",
    "SchemaError",
    "by_id",
    "chunk_markdown",
    "hashing_embed",
    "load_corpus",
    "rrf",
    "tokenize",
    "validate",
]
