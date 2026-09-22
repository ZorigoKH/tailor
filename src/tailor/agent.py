"""The agent: a bounded loop, four tools, and a state you can read.

``tailor match`` is a fixed pipeline because the question is fixed. An open question ("what
in my corpus shows I can ship a service?") is not, so this loop lets the model choose: search
the corpus, read a piece in full, write a note, or finish. Three things keep it honest:

* **A budget.** At most ``max_steps`` tool calls; running out ends the loop with whatever was
  found, never with an invented answer.
* **A written state.** Each step appends one line to a transcript the model rereads - the
  notes are the memory, and old observations are dropped once the transcript passes its
  character budget, so context stays small and deliberate rather than merely long.
* **A check at the end.** A finished answer must cite evidence that exists, and any quoted
  span in it is verified word for word. Unverified answers say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import prompts
from .ask import ask_json
from .retrieval import Index
from .schema import SchemaError
from .verify import QuoteCheck, check_quote

STATE_BUDGET = 6000  # characters of transcript the model rereads

TOOLS: dict[str, dict] = {
    "search_evidence": {
        "description": "find pieces of the corpus that match a query",
        "args": {"query": "string", "k": "integer, optional"},
    },
    "read_evidence": {
        "description": "read one piece in full, by id",
        "args": {"id": "string, an id from a search result"},
    },
    "note": {
        "description": "write down a finding so it survives later steps",
        "args": {"text": "string"},
    },
    "finish": {
        "description": "answer the question, citing the evidence ids you used",
        "args": {"answer": "string", "citations": "array of evidence ids"},
    },
}


@dataclass
class Step:
    tool: str
    args: dict
    observation: str
    thought: str = ""


@dataclass
class Answer:
    text: str
    citations: list[str]
    steps: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    check: QuoteCheck = field(default_factory=lambda: QuoteCheck(False, "no answer"))

    @property
    def verified(self) -> bool:
        return self.check.ok


class Agent:
    """Answers open questions about the corpus with tools, a budget and a final check."""

    def __init__(self, provider, index: Index, *, max_steps: int = 6, k: int = 5, tracer=None):
        self.provider = provider
        self.index = index
        self.max_steps = max_steps
        self.k = k
        self.tracer = tracer
        self.pieces = {piece.id: piece for piece in index.pieces}

    # -- tools --------------------------------------------------------------------------
    def _search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: search_evidence needs a query"
        k = int(args.get("k") or self.k)
        hits = self.index.search(query, k=max(1, min(k, 10)))
        if not hits:
            return "no matches"
        return "\n".join(f"- {hit.id}: {hit.evidence.title} ({hit.found_by})" for hit in hits)

    def _read(self, args: dict) -> str:
        piece = self.pieces.get(str(args.get("id", "")).strip())
        if piece is None:
            return f"error: no evidence with id {args.get('id')!r}"
        return f'"""{piece.quotable}"""'

    def _note(self, args: dict, notes: list[str]) -> str:
        text = str(args.get("text", "")).strip()
        if not text:
            return "error: note needs text"
        notes.append(text)
        return "noted"

    # -- loop ---------------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        transcript: list[str] = []
        steps: list[Step] = []
        notes: list[str] = []

        for _ in range(self.max_steps):
            state = self._state(transcript, notes)
            try:
                reply = ask_json(
                    self.provider,
                    prompts.agent_messages(question, TOOLS, state),
                    prompts.AGENT_SCHEMA,
                )
            except SchemaError as exc:
                return Answer(
                    text="",
                    citations=[],
                    steps=steps,
                    notes=notes,
                    check=QuoteCheck(False, str(exc)),
                )
            tool, args, thought = reply["tool"], reply.get("args") or {}, reply.get("thought", "")

            if tool == "finish":
                answer = str(args.get("answer", "")).strip()
                citations = [str(c) for c in (args.get("citations") or []) if c]
                check = self._check(answer, citations)
                steps.append(Step(tool, args, "finished", thought))
                self._trace(tool, args, "finished", thought)
                return Answer(
                    text=answer, citations=citations, steps=steps, notes=notes, check=check
                )

            if tool == "search_evidence":
                observation = self._search(args)
            elif tool == "read_evidence":
                observation = self._read(args)
            elif tool == "note":
                observation = self._note(args, notes)
            else:
                observation = f"error: no tool named {tool!r}; pick one of {', '.join(TOOLS)}"

            steps.append(Step(tool, args, observation, thought))
            self._trace(tool, args, observation, thought)
            transcript.append(self._render(tool, args, observation))

        return Answer(
            text="",
            citations=[],
            steps=steps,
            notes=notes,
            check=QuoteCheck(False, f"ran out of steps after {self.max_steps} tool calls"),
        )

    # -- helpers ------------------------------------------------------------------------
    def _check(self, answer: str, citations: list[str]) -> QuoteCheck:
        if not answer:
            return QuoteCheck(False, "finished with an empty answer")
        unknown = [c for c in citations if c not in self.pieces]
        if unknown:
            return QuoteCheck(False, f"cites evidence that does not exist: {', '.join(unknown)}")
        if not citations:
            return QuoteCheck(False, "finished without citing evidence")
        sources = {piece_id: self.pieces[piece_id].quotable for piece_id in citations}
        quoted = re.findall(r'"([^"]{12,})"', answer)
        for span in quoted or [answer]:
            check = check_quote(span, citations, sources)
            if check.ok:
                return check
        if quoted:
            return QuoteCheck(False, "a quoted span in the answer is not in the cited evidence")
        return QuoteCheck(
            False, "the answer paraphrases; nothing in it could be verified word for word"
        )

    def _render(self, tool: str, args: dict, observation: str) -> str:
        shown = ", ".join(f'{key}="{value}"' for key, value in args.items())
        return f"{tool}({shown}) ->\n{observation}"

    def _state(self, transcript: list[str], notes: list[str]) -> str:
        parts = list(transcript)
        while parts and sum(len(p) for p in parts) > STATE_BUDGET:
            parts.pop(0)  # drop the oldest observation; the notes carry what mattered
        if notes:
            parts.append("notes:\n" + "\n".join(f"* {note}" for note in notes))
        return "\n".join(parts)

    def _trace(self, tool: str, args: dict, observation: str, thought: str) -> None:
        if self.tracer is not None:
            self.tracer.event(
                "tool", tool=tool, args=args, thought=thought[:200], observation=observation[:300]
            )
