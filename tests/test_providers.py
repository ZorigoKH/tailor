"""The OpenAI-compatible adapter with a fake wire, and the offline rules."""

import io
import json
import urllib.error

import numpy as np
import pytest

from tailor import ModelError, Offline, OpenAICompatible, make
from tailor.prompts import Message, judge_messages, posting_messages, system
from tailor.providers import Completion, Usage


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))


def _chat_reply(text: str) -> dict:
    return {
        "model": "test-model",
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


class Wire:
    """Stands in for the network: a queue of responses or exceptions, and a log of payloads."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads: list[tuple[str, dict]] = []

    def __call__(self, path, payload):
        self.payloads.append((path, json.loads(json.dumps(payload))))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    return OpenAICompatible("m", base_url="http://x/v1", api_key="k", embed_model="e")


def test_complete_sends_json_mode_and_reads_usage(provider):
    provider._request = Wire(_chat_reply('{"ok": true}'))
    messages = [system("judge", schema={"type": "object"}), Message("user", "hi")]
    out = provider.complete(messages, json_schema={"type": "object"})
    assert isinstance(out, Completion) and out.text == '{"ok": true}'
    assert out.usage == Usage(12, 3) and out.model == "test-model"
    path, payload = provider._request.payloads[0]
    assert path == "/chat/completions"
    assert payload["response_format"] == {"type": "json_object"} and payload["max_tokens"] == 1200
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


def test_adapts_to_servers_that_reject_parameters(provider):
    provider._request = Wire(
        _http_error(400, '{"error": "response_format is not supported"}'),
        _http_error(400, '{"error": "max_tokens is not supported, use max_completion_tokens"}'),
        _chat_reply("fine"),
    )
    messages = [system("judge", schema={"type": "object"}), Message("user", "hi")]
    out = provider.complete(messages, json_schema={"type": "object"})
    assert out.text == "fine"
    payloads = [p for _, p in provider._request.payloads]
    assert "response_format" in payloads[0] and "response_format" not in payloads[1]
    assert "max_tokens" in payloads[1] and payloads[2]["max_completion_tokens"] == 1200
    assert provider.json_mode is False and provider.token_key == "max_completion_tokens"


def test_retries_rate_limits_and_reports_hard_failures(provider):
    provider._request = Wire(_http_error(429, "slow down"), _chat_reply("ok"))
    assert provider.complete([Message("user", "hi")]).text == "ok"

    provider._request = Wire(_http_error(401, "bad key"))
    with pytest.raises(ModelError, match="refused the request. HTTP 401"):
        provider.complete([Message("user", "hi")])

    provider._request = Wire(*[urllib.error.URLError("down")] * 3)
    with pytest.raises(ModelError, match="failed after 3 attempts"):
        provider.complete([Message("user", "hi")])


def test_embed_batches_orders_and_normalizes(provider):
    provider._request = Wire(
        {"data": [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 4.0]}]}
    )
    out = provider.embed(["a", "b"])
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 1.0]])
    path, payload = provider._request.payloads[0]
    assert path == "/embeddings" and payload == {"model": "e", "input": ["a", "b"]}


def test_make_chooses_offline_without_a_key(monkeypatch):
    for name in ("TAILOR_API_KEY", "OPENAI_API_KEY", "TAILOR_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert isinstance(make("auto"), Offline)
    assert isinstance(make("offline"), Offline)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(make("auto"), OpenAICompatible)
    hosted = make("openai", model="gpt-x", base_url="http://localhost:11434/v1")
    assert hosted.model == "gpt-x" and hosted.base_url == "http://localhost:11434/v1"


def test_offline_parses_a_posting(offline, ai_posting):
    reply = json.loads(offline.complete(posting_messages(ai_posting), json_schema={}).text)
    assert reply["company"] == "Northwind Consulting"
    assert reply["title"] == "AI Engineering Intern (Summer 2027)"
    assert reply["locations"] == ["New York, NY"] and reply["deadline"] == "2026-10-09"
    assert reply["team"] == "Applied AI" and reply["employment_type"] == "Intern (full-time)"
    assert [r["kind"] for r in reply["requirements"]] == ["must"] * 5 + ["nice"] * 2
    assert reply["requirements"][0] == {
        "text": "Proficiency in Python",
        "kind": "must",
        "skills": ["python"],
    }
    assert reply["notes"] == ["Co-located with the team at least three days a week"]


def test_offline_judges_by_overlap_and_quotes_real_sentences(offline, index):
    hits = index.search("Python pipeline with pandas", k=3)
    reply = json.loads(offline.complete(judge_messages("Python pipeline with pandas", hits)).text)
    assert reply["verdict"] == "covered" and reply["citations"] == ["talentsnipe.md:8-8"]
    assert reply["quote"] in hits[0].evidence.quotable

    hits = index.search("Kubernetes", k=3)
    reply = json.loads(offline.complete(judge_messages("Kubernetes in production", hits)).text)
    assert reply["verdict"] == "missing" and reply["citations"] == []


def test_offline_refuses_unknown_tasks(offline):
    with pytest.raises(ModelError, match="no rule for the 'dance' task"):
        offline.complete([system("dance"), Message("user", "x")])
