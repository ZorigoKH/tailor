"""Run traces: every model call written down, so a run can be explained afterwards.

A GenAI pipeline that cannot be replayed cannot be debugged. :class:`Tracer` wraps a
provider and records one JSON line per call - task, model, tokens, seconds - plus the tool
calls an agent makes and the verdicts a run produces. ``tailor report`` reads the file back;
:meth:`Tracer.summary` is what the CLI prints when a run ends.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import prompts
from .prompts import Message
from .providers import Completion, Usage


@dataclass
class Tracer:
    """Collects events in memory, and appends them to a JSONL file when given a path."""

    path: Path | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[dict] = field(default_factory=list)

    @classmethod
    def to(cls, directory: str | os.PathLike | None, name: str = "run") -> Tracer:
        if directory is None:
            return cls()
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return cls(path=folder / f"{name}-{stamp}.jsonl")

    def event(self, kind: str, **fields: object) -> dict:
        record = {"run": self.run_id, "at": round(time.time(), 3), "kind": kind, **fields}
        self.events.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        return record

    def wrap(self, provider):
        return TracedProvider(provider, self)

    def summary(self) -> dict:
        calls = [e for e in self.events if e["kind"] == "model"]
        usage = Usage()
        by_task: dict[str, int] = {}
        for call in calls:
            usage += Usage(call.get("prompt_tokens", 0), call.get("completion_tokens", 0))
            by_task[call.get("task", "?")] = by_task.get(call.get("task", "?"), 0) + 1
        return {
            "run": self.run_id,
            "calls": len(calls),
            "prompt_tokens": usage.prompt,
            "completion_tokens": usage.completion,
            "tokens": usage.total,
            "seconds": round(sum(call.get("seconds", 0.0) for call in calls), 2),
            "by_task": by_task,
            "tools": sum(1 for e in self.events if e["kind"] == "tool"),
        }


class TracedProvider:
    """A provider that records what it was asked and what it cost."""

    def __init__(self, provider, tracer: Tracer):
        self.provider = provider
        self.tracer = tracer
        self.name = getattr(provider, "name", "provider")
        self.model = getattr(provider, "model", "?")

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        try:
            task = prompts.task_of(messages)
        except ValueError:
            task = "?"
        completion = self.provider.complete(messages, **kwargs)
        self.tracer.event(
            "model",
            task=task,
            model=completion.model,
            prompt_tokens=completion.usage.prompt,
            completion_tokens=completion.usage.completion,
            seconds=round(completion.seconds, 3),
            chars=len(completion.text),
        )
        return completion

    def embed(self, texts: list[str]) -> np.ndarray:
        started = time.monotonic()
        vectors = self.provider.embed(texts)
        self.tracer.event(
            "embed",
            count=len(texts),
            seconds=round(time.monotonic() - started, 3),
            model=self.model,
        )
        return vectors
