# tailor

[![tests](https://github.com/ZorigoKH/tailor/actions/workflows/ci.yml/badge.svg)](https://github.com/ZorigoKH/tailor/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**A job-application copilot that cannot make things up about you.**

Give it a posting and a folder of your own material (resume bullets, project notes). It
reads the posting into requirements, finds the evidence for each one, and says *covered*,
*partial* or *missing*, quoting the words that show it. Then it checks every quote against
the source, word for word. A claim the evidence does not support is flagged, not shown as a
fact. Drafts are written from verified evidence only, and every number and name in them is
checked the same way.

I built it because I was writing a dozen cover letters and wanted help that would tell me
what I *can't* claim as readily as what I can.

```
$ tailor match posting.txt

AI Engineering Intern (Summer 2027) at Northwind Consulting · New York, NY · apply by 2026-10-09 · 7 requirements (5 must)
must-have coverage 40%  ·  covered 3  partial 3  missing 1  unsupported 0

 + [must] Proficiency in Python
     talentsnipe.md:8-8: “Built a Python pipeline with pandas that cleans and standardizes more than 5,000 recruitment rec”
 + [must] Experience building APIs or services
     terran.md:12-12: “Pushed authorization into the database itself with 76 row-level security policies, guarded by an”
 ~ [must] Strong engineering practices: testing, code review, version control and CI/CD
     unbundle.md:12-12: “57 tests, and every number quoted in the README is recomputed by the test suite, so the document”
 - [nice] Nice to have: hands-on experience with agentic patterns such as tool use and orchestrati
     too few of the requirement's words appear in any evidence

gaps to close or leave out:
 - Nice to have: hands-on experience with agentic patterns such as tool use and orchestration
```

## Quick start

```bash
pip install "tailor @ git+https://github.com/ZorigoKH/tailor.git"

tailor index --search "row-level security"      # what is in the corpus, and what a query finds
tailor match posting.txt                        # every requirement, with verified quotes
tailor draft posting.txt --out draft.txt        # a letter from the verified evidence only
tailor ask "What shows I can ship a service?"   # the agent, with tools and a step budget
tailor eval                                     # score the pipeline against labelled postings
```

With no API key, every command runs against the **offline provider**, a rule-based stand-in
that returns the same JSON shapes a model would. To use a real model, set a key or a base URL;
the adapter speaks the OpenAI-compatible API, so one setting covers most hosts:

```bash
export OPENAI_API_KEY=sk-...                                   # OpenAI
export TAILOR_BASE_URL=http://localhost:11434/v1 TAILOR_MODEL=llama3.1   # Ollama, local and free
export TAILOR_BASE_URL=https://api.groq.com/openai/v1 TAILOR_API_KEY=gsk_... TAILOR_MODEL=llama-3.3-70b-versatile
tailor match posting.txt --provider openai --trace runs/      # and write a trace of the run
```

Your material goes in `evidence/` as markdown: one file per employer or project, headings for
roles, a paragraph or bullet per claim. The `evidence/` folder in this repo is mine.

## How it works

```
  posting.txt ──► extract ──► requirements[]           JSON schema, one repair retry
                                   │
  evidence/*.md ─► chunk ─► index ─┤   BM25 + embeddings, fused by reciprocal rank
                                   ▼
                          judge each requirement       model: verdict + citation + quote
                                   │
                            verify the quote           deterministic: is it really there?
                                   │
              ┌────────────────────┼─────────────────────┐
           coverage             draft                   ask
        verdicts + gaps   evidence-only prose      agent: 4 tools, step budget
                          numbers and names        state the model rereads
                          checked against it       final answer verified
```

**Retrieval.** Every piece of evidence has a stable id (`terran.md:12-12`) and carries the
heading it sits under. Search runs BM25 for the words and embeddings for the meaning, and
fuses the two rankings with reciprocal rank fusion, which needs no calibration between
scores. Without an embedding model the vectors are deterministic hashed bags of words: weak,
but enough to keep the fusion path exercised offline.

**Structured output.** Every model call has a schema. Replies are parsed tolerantly (fences,
prose around the JSON), validated strictly, and if they do not fit, the validation errors go
back to the model with the bad reply for one more try. Two retries usually means the schema
is wrong, not the model.

**Verification.** The model returns a quote with each verdict. `verify.check_quote` normalizes
whitespace and quote marks and checks the quote is a substring of the cited evidence (an
ellipsis splits it into parts that must appear in order). A verdict with a quote that fails is
reported as `unsupported`, with the reason. Drafts get a second check: every number and every
name must appear in the evidence the draft was written from, or in the posting itself. What
fails is listed for a person to fix or cut. Neither check asks a model whether the model was
right.

**The agent.** `tailor ask` runs a loop with four tools: `search_evidence`, `read_evidence`,
`note` and `finish`. It has a step budget, and its state is a short transcript the model
rereads each step; when the transcript passes its character budget the oldest observations
are dropped and the notes carry what mattered. A finished answer must cite evidence that
exists, and any quoted span in it is verified word for word.

**Observability.** `--trace DIR` writes one JSON line per model call (task, model, tokens,
seconds) and per tool call, plus the verdicts, so a run can be replayed and explained.

## Evals

`tailor eval` runs three labelled postings through the whole pipeline and scores the result
by comparison with hand-written labels. Nothing is graded by a model.

| Metric | What it asks | Offline stand-in |
|---|---|---:|
| fields | title, company, locations, deadline, requirement count extracted correctly | 15/15 |
| citations | of the verdicts that claimed support, how many quotes verified | 12/12 |
| recall@k | did retrieval surface the evidence the label points at | 15/15 |
| refusals | on requirements the corpus does not cover, did it say missing | 3/4 |
| verdicts | agreement with the labelled verdict | 11/19 |
| draft flags | numbers or names in drafts that no evidence backs | 0 |

The stand-in judges by word overlap, so its verdict agreement is low and its one false
positive is telling: it called "healthcare claims experience" covered because the word
*claim* appears in a note about verifying claims. That is what the harness is for. The
deterministic parts, extraction, citation checking and retrieval, are what CI enforces; the
verdicts are what a real model improves, and the same command measures it:

```bash
tailor eval --provider openai --json runs/eval-gpt.json
```

The offline numbers are pinned by `tests/test_evals_cli.py`, so a change to the rules that
moves them fails the build.

## Limits

- Word-for-word quotes are a strict standard. A model that paraphrases well will be marked
  unsupported more often than a careless one that copies; that is the intended trade.
- The offline provider parses postings with regexes written for a common layout (a
  "what we're looking for" section with bullets). On unusual layouts it will miss
  requirements; a real model will not.
- Hashed vectors are not semantic. They exist so the hybrid path runs offline; use an
  embedding model (`--embed model --embed-model text-embedding-3-small`) for real recall.
- The name check flags capitalized words it cannot find, so a draft that mentions a city or
  a product the posting does not name will be flagged. Add it to the posting or the corpus.

## Development

```bash
git clone https://github.com/ZorigoKH/tailor.git && cd tailor
pip install -e ".[dev]"
pytest                    # 56 tests, offline, under a second
ruff check . && ruff format --check .
```

## Layout

```
src/tailor/
  corpus.py     evidence files -> quotable pieces with stable ids
  retrieval.py  BM25, hashed vectors, reciprocal rank fusion
  prompts.py    the prompts and the wire format both providers speak
  schema.py     a small JSON Schema subset, tolerant parsing
  ask.py        one validated model call, one repair retry
  providers.py  the OpenAI-compatible adapter and the offline stand-in
  posting.py    posting -> structured requirements
  match.py      requirement -> verdict, quote, verification
  verify.py     the checks that do not trust the model
  draft.py      evidence-only drafting and its guardrail
  agent.py      the tool loop
  evals.py      the harness
  trace.py      run traces
  cli.py        tailor index | posting | match | draft | ask | eval
evidence/       my material, as an example corpus
evals/          three labelled postings
tests/          56 tests, all offline
```

MIT. © 2026 Zorigtbaatar Khasbaatar.
