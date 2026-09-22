"""Command line: ``tailor match posting.txt``, ``tailor draft``, ``tailor ask``, ``tailor eval``.

With no API key every command runs against the offline provider, so the pipeline can be
exercised end to end before anyone spends a cent. Point it at a hosted model with
``--provider openai`` (or ``TAILOR_BASE_URL`` for Ollama and friends).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .corpus import Evidence, by_id, load_corpus
from .draft import draft_letter
from .evals import load_labels, run_evals
from .match import Coverage, match_posting
from .posting import extract_posting, read_posting, save_posting, summarize
from .providers import ModelError, OpenAICompatible, make
from .retrieval import Index, hashing_embed
from .schema import SchemaError
from .trace import Tracer

MARK = {"covered": "+", "partial": "~", "missing": "-", "unsupported": "!"}


def _provider(args: argparse.Namespace):
    return make(
        args.provider, model=args.model, base_url=args.base_url, embed_model=args.embed_model
    )


def _embedder(args: argparse.Namespace, provider):
    if args.embed == "none":
        return None
    if args.embed == "hashing":
        return hashing_embed
    if args.embed == "model":
        return provider.embed
    hosted = isinstance(getattr(provider, "provider", provider), OpenAICompatible)
    return provider.embed if hosted and args.embed_model else hashing_embed


def _index(args: argparse.Namespace, provider) -> tuple[Index, dict[str, Evidence]]:
    pieces = load_corpus(args.corpus)
    return Index(pieces, embedder=_embedder(args, provider)), by_id(pieces)


def _print_coverage(coverage: Coverage) -> None:
    counts = coverage.counts()
    print(f"\n{summarize(coverage.posting)}")
    print(
        f"must-have coverage {100 * coverage.rate:.0f}%  ·  "
        + "  ".join(f"{name} {counts[name]}" for name in counts)
        + "\n"
    )
    for judgment in coverage.judgments:
        head = f" {MARK[judgment.final]} [{judgment.kind}] {judgment.requirement[:88]}"
        print(head)
        if judgment.final == "unsupported":
            print(f"     ! {judgment.check.reason}")
        elif judgment.citations:
            print(f"     {', '.join(judgment.citations)}: “{judgment.quote[:96]}”")
        else:
            print(f"     {judgment.why[:96]}")
    if coverage.gaps:
        print("\ngaps to close or leave out:")
        for judgment in coverage.gaps:
            print(f" - {judgment.requirement[:96]}")


def cmd_index(args: argparse.Namespace) -> int:
    provider = _provider(args)
    index, pieces = _index(args, provider)
    sources: dict[str, int] = {}
    for piece in index.pieces:
        sources[piece.source] = sources.get(piece.source, 0) + 1
    words = sum(piece.words for piece in index.pieces)
    print(f"{len(index.pieces)} pieces from {len(sources)} files, {words} words")
    print(f"retrieval: BM25 + {'embeddings' if index.has_vectors else 'no embeddings'}")
    for source, count in sorted(sources.items()):
        print(f"  {source:<28}{count:>4} pieces")
    if args.search:
        print(f"\ntop matches for {args.search!r}:")
        for hit in index.search(args.search, k=args.k):
            print(f"  {hit.score:.4f} [{hit.found_by:<8}] {hit.id:<22} {hit.evidence.text[:70]}")
    return 0


def cmd_posting(args: argparse.Namespace) -> int:
    provider = _provider(args)
    posting = extract_posting(provider, read_posting(args.source))
    print(summarize(posting))
    for requirement in posting["requirements"]:
        skills = f" [{', '.join(requirement['skills'])}]" if requirement["skills"] else ""
        print(f"  ({requirement['kind']}) {requirement['text']}{skills}")
    if args.json:
        print(f"\nwritten to {save_posting(posting, args.json)}")
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    tracer = Tracer.to(args.trace, "match")
    provider = tracer.wrap(_provider(args))
    index, pieces = _index(args, provider)
    posting = extract_posting(provider, read_posting(args.source))
    coverage = match_posting(provider, index, posting, k=args.k, tracer=tracer)
    _print_coverage(coverage)
    if coverage.flagged:
        print("\nthe model claimed support these quotes did not back up:")
        for judgment in coverage.flagged:
            print(f" ! {judgment.requirement[:70]}: {judgment.check.reason}")
    _print_run(tracer, args)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "posting": posting,
                    "coverage": coverage.counts(),
                    "must_have_rate": coverage.rate,
                    "judgments": [
                        {
                            "requirement": j.requirement,
                            "kind": j.kind,
                            "verdict": j.final,
                            "citations": j.citations,
                            "quote": j.quote,
                            "verified": j.check.ok,
                            "reason": j.check.reason,
                        }
                        for j in coverage.judgments
                    ],
                    "run": tracer.summary(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"written to {args.json}")
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    tracer = Tracer.to(args.trace, "draft")
    provider = tracer.wrap(_provider(args))
    index, pieces = _index(args, provider)
    posting = extract_posting(provider, read_posting(args.source))
    coverage = match_posting(provider, index, posting, k=args.k, tracer=tracer)
    letter = draft_letter(
        provider, posting, coverage, pieces, paragraphs=args.paragraphs, tracer=tracer
    )
    print(f"{summarize(posting)}\n")
    print(letter.text)
    print(f"\nwritten from: {', '.join(letter.evidence)}")
    if letter.flags:
        print("\nnothing in your corpus backs these up - fix them or cut them:")
        for flag in letter.flags:
            print(f" ! {flag}")
    else:
        print("\nevery number and name in the draft traces to your corpus.")
    if coverage.gaps:
        print(f"\n{len(coverage.gaps)} requirement(s) the draft stays quiet about, by design.")
    _print_run(tracer, args)
    if args.out:
        Path(args.out).write_text(letter.text + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    tracer = Tracer.to(args.trace, "ask")
    provider = tracer.wrap(_provider(args))
    index, _ = _index(args, provider)
    answer = Agent(provider, index, max_steps=args.max_steps, k=args.k, tracer=tracer).ask(
        args.question
    )
    for step in answer.steps:
        shown = ", ".join(f"{key}={value!r}" for key, value in step.args.items())
        print(f"  {step.tool}({shown[:80]})")
    print()
    print(answer.text or "(no answer)")
    if answer.citations:
        print(f"\ncited: {', '.join(answer.citations)}")
    print(f"check: {answer.check.reason}")
    for note in answer.notes:
        print(f"note: {note}")
    _print_run(tracer, args)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    tracer = Tracer.to(args.trace, "eval")
    provider = tracer.wrap(_provider(args))
    index, pieces = _index(args, provider)
    labels = load_labels(args.labels)
    report = run_evals(
        provider,
        index,
        pieces,
        postings_dir=args.postings,
        labels=labels,
        k=args.k,
        draft=not args.no_draft,
    )
    print(report.table())
    _print_run(tracer, args)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"written to {args.json}")
    floor = args.fail_under
    if floor and min(report.metrics().values()) < floor:
        print(f"\nbelow the floor of {floor}", file=sys.stderr)
        return 1
    return 0


def _print_run(tracer: Tracer, args: argparse.Namespace) -> None:
    summary = tracer.summary()
    print(
        f"\n{summary['calls']} model calls · {summary['tokens']} tokens · "
        f"{summary['seconds']}s · {summary['tools']} tool calls"
    )
    if tracer.path is not None:
        print(f"trace: {tracer.path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tailor",
        description="Match a job posting against evidence from your own corpus, and draft "
        "only what the evidence supports.",
    )
    parser.add_argument("--version", action="version", version=f"tailor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--corpus", default="evidence", help="folder of your material")
        command.add_argument(
            "--provider",
            default="auto",
            choices=["auto", "offline", "openai"],
            help="auto uses a hosted model when a key is set, offline otherwise",
        )
        command.add_argument("--model", help="model name (default: $TAILOR_MODEL or gpt-4o-mini)")
        command.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. Ollama")
        command.add_argument("--embed-model", help="embedding model, if the endpoint serves one")
        command.add_argument(
            "--embed",
            default="auto",
            choices=["auto", "model", "hashing", "none"],
            help="how to vectorize the corpus (default: hashing unless --embed-model is set)",
        )
        command.add_argument("--k", type=int, default=6, help="pieces retrieved per query")
        command.add_argument("--trace", metavar="DIR", help="write a JSONL trace of the run")

    index = sub.add_parser("index", help="show what is in the corpus, and try a search")
    index.add_argument("--search", metavar="QUERY", help="show the top matches for a query")
    common(index)
    index.set_defaults(func=cmd_index)

    posting = sub.add_parser("posting", help="read a posting into structured fields")
    posting.add_argument("source", help="file, - for stdin, or the text itself")
    posting.add_argument("--json", metavar="PATH", help="also write the fields as JSON")
    common(posting)
    posting.set_defaults(func=cmd_posting)

    match = sub.add_parser("match", help="judge every requirement against your evidence")
    match.add_argument("source", help="file, - for stdin, or the text itself")
    match.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    common(match)
    match.set_defaults(func=cmd_match)

    draft = sub.add_parser("draft", help="draft a letter from the verified evidence")
    draft.add_argument("source", help="file, - for stdin, or the text itself")
    draft.add_argument("--paragraphs", type=int, default=4, help="at most this many paragraphs")
    draft.add_argument("--out", metavar="PATH", help="write the draft to a file")
    common(draft)
    draft.set_defaults(func=cmd_draft)

    ask = sub.add_parser("ask", help="ask the agent an open question about your corpus")
    ask.add_argument("question")
    ask.add_argument("--max-steps", type=int, default=6, help="tool call budget")
    common(ask)
    ask.set_defaults(func=cmd_ask)

    evaluate = sub.add_parser("eval", help="score the pipeline against the labelled postings")
    evaluate.add_argument("--labels", default="evals/labels.json")
    evaluate.add_argument("--postings", default="evals/postings")
    evaluate.add_argument("--json", metavar="PATH", help="write the scores as JSON")
    evaluate.add_argument("--no-draft", action="store_true", help="skip the drafting check")
    evaluate.add_argument("--fail-under", type=float, metavar="RATE", help="exit 1 below this rate")
    common(evaluate)
    evaluate.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ModelError, SchemaError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"tailor: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
