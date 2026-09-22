---
title: tailor
tags: open source, python, llm, rag, retrieval, agents, evals, guardrails
---

# tailor, a grounded copilot for job applications

Built tailor, a retrieval-augmented pipeline that reads a job posting into validated structured requirements, retrieves evidence from my own corpus with hybrid BM25 and embedding search fused by reciprocal rank, and asks a language model for a covered, partial or missing verdict on each requirement with the words that support it.

Every quote the model returns is verified against the evidence it cites before anything is reported, so a claim the source does not support is flagged as unsupported instead of passing as a fact, and a draft is scanned for numbers and names that no evidence backs up.

An agent loop with four tools, a step budget and a written state answers open questions about the corpus, and the model adapter speaks the OpenAI-compatible API so the same code runs against a hosted model or a local one.

An eval harness scores posting extraction, verdict agreement against hand-written labels, citation validity, refusals on requirements the corpus does not cover, and retrieval recall, and a rule-based offline provider makes every one of those runs reproducible in continuous integration (CI) with no API key.
