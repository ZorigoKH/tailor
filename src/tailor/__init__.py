"""tailor: match a job posting against evidence from your own corpus, and draft only what
the evidence supports.

The pipeline is deliberately boring in the middle and strict at the edges: a posting is read
into validated fields, requirements are answered from retrieved evidence, and every quote is
checked word for word against the source it cites before anything is shown as a fact.
"""

from .agent import Agent, Answer, Step
from .corpus import Evidence, by_id, chunk_markdown, load_corpus
from .draft import Draft, draft_letter
from .evals import EvalReport, Score, run_evals
from .match import Coverage, Judgment, judge_requirement, match_posting
from .posting import extract_posting, read_posting, summarize
from .providers import ModelError, Offline, OpenAICompatible, Usage, make
from .retrieval import Bm25, Hit, Index, hashing_embed, rrf, tokenize
from .schema import SchemaError, validate
from .trace import Tracer
from .verify import QuoteCheck, check_quote, unsupported_terms

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "Answer",
    "Bm25",
    "Coverage",
    "Draft",
    "EvalReport",
    "Evidence",
    "Hit",
    "Index",
    "Judgment",
    "ModelError",
    "Offline",
    "OpenAICompatible",
    "QuoteCheck",
    "SchemaError",
    "Score",
    "Step",
    "Tracer",
    "Usage",
    "by_id",
    "check_quote",
    "chunk_markdown",
    "draft_letter",
    "extract_posting",
    "hashing_embed",
    "judge_requirement",
    "load_corpus",
    "make",
    "match_posting",
    "read_posting",
    "rrf",
    "run_evals",
    "summarize",
    "tokenize",
    "unsupported_terms",
    "validate",
]
