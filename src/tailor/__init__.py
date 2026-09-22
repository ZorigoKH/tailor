"""tailor: match a job posting against evidence from your own corpus, and draft only what
the evidence supports.

The pipeline is deliberately boring in the middle and strict at the edges: a posting is read
into validated fields, requirements are answered from retrieved evidence, and every quote is
checked word for word against the source it cites before anything is shown as a fact.
"""

__version__ = "0.1.0"
