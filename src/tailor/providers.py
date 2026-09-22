"""Model access: one adapter for any OpenAI-compatible endpoint, and an offline stand-in.

:class:`OpenAICompatible` speaks ``/chat/completions`` and ``/embeddings``, the shape served
by OpenAI, Gemini's compatibility endpoint, Groq, Together, OpenRouter, vLLM and Ollama, so
changing model means changing a base URL and a name. It retries rate limits, asks for JSON
when a schema is in play, and adapts when a server rejects a parameter (some newer models
refuse ``max_tokens`` or ``temperature``) instead of failing the run.

:class:`Offline` answers the same JSON shapes with rules instead of a model: postings parsed
with regexes, verdicts from word overlap, drafts assembled from the quotes it was handed. It
is why the tests and ``tailor eval`` run with no key, no network and the same code path.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import prompts
from .prompts import Message
from .retrieval import hashing_embed, stem, tokenize

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
BATCH = 96

SKILLS = (
    "python sql java typescript javascript c++ r go rust numpy pandas scikit-learn pytorch "
    "tensorflow react next.js node.js fastapi flask django rest grpc graphql postgresql "
    "mysql mongodb redis supabase snowflake spark airflow dbt docker kubernetes aws gcp "
    "azure terraform git ci/cd pytest jest llm llms rag agents agentic langchain langgraph "
    "embeddings fine-tuning transformers mlops tableau excel"
).split()


class ModelError(RuntimeError):
    """The model could not be reached, or would not answer."""


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.prompt + other.prompt, self.completion + other.completion)


@dataclass
class Completion:
    text: str
    usage: Usage
    seconds: float
    model: str


class Provider(Protocol):
    name: str
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Completion: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAICompatible:
    """Any endpoint that serves the OpenAI chat and embeddings shapes."""

    name = "openai-compatible"

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        embed_model: str | None = None,
        timeout: float = 90.0,
        retries: int = 3,
    ):
        self.base_url = (base_url or os.environ.get("TAILOR_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = (
            api_key or os.environ.get("TAILOR_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        )
        self.model = model or os.environ.get("TAILOR_MODEL") or DEFAULT_MODEL
        self.embed_model = (
            embed_model or os.environ.get("TAILOR_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        )
        self.timeout = timeout
        self.retries = retries
        self.json_mode = True  # turned off if the server rejects response_format
        self.token_key = "max_tokens"  # renamed if the server wants max_completion_tokens
        self.send_temperature = True

    # -- http ---------------------------------------------------------------------------
    def _request(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        last = ""
        for attempt in range(self.retries):
            try:
                return self._request(path, payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                last = f"HTTP {exc.code}: {detail}"
                if exc.code == 400 and self._adapt(payload, detail):
                    continue  # the server named a parameter it dislikes; drop it and retry
                if exc.code in {408, 409, 429} or exc.code >= 500:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise ModelError(f"{self.base_url}{path} refused the request. {last}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = str(exc)
                time.sleep(min(2**attempt, 8))
        raise ModelError(f"{self.base_url}{path} failed after {self.retries} attempts. {last}")

    def _adapt(self, payload: dict, detail: str) -> bool:
        """Some models reject parameters the rest accept. Adjust once and retry."""
        if "response_format" in detail and "response_format" in payload:
            payload.pop("response_format")
            self.json_mode = False
            return True
        if "max_tokens" in detail and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            self.token_key = "max_completion_tokens"
            return True
        if "temperature" in detail and "temperature" in payload:
            payload.pop("temperature")
            self.send_temperature = False
            return True
        return False

    # -- provider -----------------------------------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Completion:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            self.token_key: max_tokens,
        }
        if self.send_temperature:
            payload["temperature"] = temperature
        if json_schema is not None and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        data = self._post("/chat/completions", payload)
        seconds = time.monotonic() - started
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ModelError(f"unexpected reply shape: {json.dumps(data)[:300]}") from exc
        used = data.get("usage") or {}
        usage = Usage(int(used.get("prompt_tokens", 0)), int(used.get("completion_tokens", 0)))
        return Completion(text, usage, seconds, data.get("model", self.model))

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            chunk = texts[start : start + BATCH]
            data = self._post("/embeddings", {"model": self.embed_model, "input": chunk})
            rows = sorted(data["data"], key=lambda row: row.get("index", 0))
            vectors += [row["embedding"] for row in rows]
        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text.strip()) if s.strip()]


class Offline:
    """A rule-based stand-in for a model: no key, no network, same JSON shapes.

    It reads the ``TASK`` tag and the delimited sections of the prompt, then answers with
    rules - regexes for a posting, word overlap for a verdict, the given quotes for a draft.
    Deliberately literal: it exercises the pipeline, not the language.
    """

    name = "offline"
    model = "offline-rules"

    WANTED = re.compile(
        r"what we(?:'re| are) looking for|requirement|qualification|who you are", re.I
    )
    NOTES = re.compile(r"working mode|logistics|note|travel|visa", re.I)
    BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
    NICE = re.compile(r"^(nice to have|preferred|bonus|plus)", re.I)
    ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
    # words that every requirement uses and no evidence needs to echo
    BOILERPLATE = frozenset(
        """nice have preferred experience exposure proficiency knowledge familiarity comfort
        strong working hands-on hands ability able such other related field plus bonus
        years year professional demonstrated""".split()
    )

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Completion:
        task = prompts.task_of(messages)
        handler = getattr(self, f"_{task}", None)
        if handler is None:
            raise ModelError(f"the offline provider has no rule for the {task!r} task")
        text = json.dumps(handler(messages[-1].content))
        usage = Usage(len(messages[-1].content) // 4, len(text) // 4)
        return Completion(text, usage, 0.0, self.model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return hashing_embed(texts)

    # -- rules --------------------------------------------------------------------------
    def _posting(self, user: str) -> dict:
        text = prompts.section(user, "POSTING")
        lines = text.splitlines()
        first = next((line.strip() for line in lines if line.strip()), "")
        company, title = "", first
        for separator in (" — ", " – ", " - ", " | "):
            if separator in first:
                company, _, title = (part.strip() for part in first.partition(separator))
                break

        def field(name: str) -> str:
            for line in lines:
                if line.lower().startswith(name):
                    return line.split(":", 1)[1].strip() if ":" in line else ""
            return ""

        locations = [
            part.strip(" .")
            for part in re.split(r"\bor\b|;|/|\|", re.sub(r"\(.*?\)", "", field("location")))
            if part.strip(" .")
        ]
        wanted: list[str] = []
        notes: list[str] = []
        bucket: list[str] | None = None
        for line in lines:
            stripped = line.strip()
            bullet = self.BULLET.match(line)
            if not bullet and stripped and not stripped.endswith((":", ".")) and len(stripped) < 80:
                if self.WANTED.search(stripped):
                    bucket = wanted
                elif self.NOTES.search(stripped):
                    bucket = notes
                else:
                    bucket = None
                continue
            if bullet and bucket is not None:
                bucket.append(bullet.group(1).strip())
        if not wanted:  # no recognizable section: take every bullet
            wanted = [m.group(1).strip() for m in (self.BULLET.match(line) for line in lines) if m]
        deadline = None
        for line in lines:
            if re.search(r"apply by|deadline|applications close", line, re.I):
                found = self.ISO_DATE.search(line)
                deadline = found.group(1) if found else None
                break
        return {
            "title": title.strip() or first,
            "company": company or field("company"),
            "team": field("team") or None,
            "locations": locations,
            "employment_type": field("employment type") or None,
            "deadline": deadline,
            "requirements": [
                {
                    "text": item,
                    "kind": "nice" if self.NICE.match(item) else "must",
                    "skills": sorted(set(tokenize(item)) & set(SKILLS)),
                }
                for item in wanted
            ],
            "notes": notes,
        }

    def _judge(self, user: str) -> dict:
        requirement = prompts.section(user, "REQUIREMENT")
        wanted = set(tokenize(requirement)) - {stem(w) for w in self.BOILERPLATE}
        best: tuple[float, str, str] = (0.0, "", "")
        for attributes, body in prompts.sections(user, "EVIDENCE"):
            overlap = len(wanted & set(tokenize(body))) / max(len(wanted), 1)
            if overlap > best[0]:
                sentence = max(
                    _sentences(body) or [body],
                    key=lambda s: len(wanted & set(tokenize(s))),
                )
                best = (overlap, attributes.get("id", ""), sentence)
        overlap, piece_id, quote = best
        matched = round(overlap * max(len(wanted), 1))
        if overlap >= 0.5:
            verdict = "covered"
        elif overlap >= 0.34 or (overlap >= 0.25 and matched >= 2):
            verdict = "partial"
        else:
            verdict = "missing"
        if verdict == "missing" or not piece_id:
            return {
                "verdict": "missing",
                "citations": [],
                "quote": "",
                "why": "too few of the requirement's words appear in any evidence",
            }
        return {
            "verdict": verdict,
            "citations": [piece_id],
            "quote": quote,
            "why": f"{round(100 * overlap)}% of the requirement's words appear in {piece_id}",
        }

    def _draft(self, user: str) -> dict:
        role = prompts.section(user, "ROLE")
        evidence = prompts.sections(user, "EVIDENCE")[:3]
        paragraphs = [f"I am applying for {role}."]
        paragraphs += [max(_sentences(body) or [body], key=len) for _, body in evidence]
        paragraphs.append("I would welcome the chance to talk about the role.")
        return {"paragraphs": paragraphs, "citations": [a.get("id", "") for a, _ in evidence]}

    def _agent(self, user: str) -> dict:
        question = prompts.section(user, "QUESTION")
        state = prompts.section(user, "STATE")
        if "search_evidence" not in state:
            return {
                "thought": "look for evidence that answers the question",
                "tool": "search_evidence",
                "args": {"query": question},
            }
        found = re.findall(r"^- (\S+?:\d+-\d+):", state, re.MULTILINE)
        read = re.findall(r'read_evidence\(id="([^"]+)"\)', state)
        unread = [piece_id for piece_id in found[:2] if piece_id not in read]  # top two hits
        if unread:
            return {
                "thought": "read the strongest hit in full",
                "tool": "read_evidence",
                "args": {"id": unread[0]},
            }
        quoted = re.findall(r'"""(.*?)"""', state, re.DOTALL)
        answer = max(_sentences(quoted[-1]) or [""], key=len) if quoted else ""
        return {
            "thought": "answer from what I read",
            "tool": "finish",
            "args": {"answer": answer, "citations": read[:2]},
        }


def make(
    provider: str = "auto",
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    embed_model: str | None = None,
) -> Provider:
    """Build a provider. ``auto`` uses a hosted model when a key is set, offline otherwise."""
    if provider == "offline":
        return Offline()
    has_key = bool(api_key or os.environ.get("TAILOR_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    local = bool(base_url or os.environ.get("TAILOR_BASE_URL"))
    if provider == "auto" and not (has_key or local):
        return Offline()
    return OpenAICompatible(model, base_url=base_url, api_key=api_key, embed_model=embed_model)
