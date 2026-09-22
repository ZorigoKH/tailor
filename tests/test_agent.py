"""The tool loop: budgeted, stateful, and checked at the end."""

import json

from tailor import Agent, Tracer
from tailor.agent import STATE_BUDGET


def _call(tool, **args):
    return json.dumps({"thought": "t", "tool": tool, "args": args})


def test_offline_agent_searches_reads_and_finishes_with_a_verified_quote(offline, index):
    tracer = Tracer()
    answer = Agent(offline, index, tracer=tracer).ask("What shows I can ship a service?")
    assert [s.tool for s in answer.steps][0] == "search_evidence"
    assert answer.steps[-1].tool == "finish" and answer.verified
    assert answer.citations and answer.check.reason.startswith("quote verified against")
    assert tracer.summary()["tools"] == len(answer.steps)


def test_unknown_tools_and_bad_args_become_observations_not_crashes(scripted, index):
    provider = scripted(
        _call("teleport", to="mars"),
        _call("read_evidence", id="nope.md:1-1"),
        _call("search_evidence", query=""),
        _call("note", text="remember this"),
        _call("finish", answer="nothing", citations=[]),
    )
    answer = Agent(provider, index).ask("q")
    observations = [s.observation for s in answer.steps]
    assert observations[0].startswith("error: no tool named 'teleport'")
    assert observations[1] == "error: no evidence with id 'nope.md:1-1'"
    assert observations[2] == "error: search_evidence needs a query"
    assert observations[3] == "noted" and answer.notes == ["remember this"]
    assert not answer.verified and answer.check.reason == "finished without citing evidence"
    # the state the model saw on the last step carried the note and the earlier errors
    last_state = provider.calls[-1][-1].content
    assert "notes:\n* remember this" in last_state and "teleport" in last_state


def test_budget_runs_out_cleanly(scripted, index):
    provider = scripted(*[_call("search_evidence", query="python")] * 3)
    answer = Agent(provider, index, max_steps=3).ask("q")
    assert answer.text == "" and "ran out of steps after 3" in answer.check.reason
    assert len(answer.steps) == 3


def test_quoted_spans_in_an_answer_are_checked(scripted, index):
    piece = index.pieces[0]
    honest = _call("finish", answer=f'It says "{piece.text[:40]}" there.', citations=[piece.id])
    assert Agent(scripted(honest), index).ask("q").verified
    invented = _call(
        "finish", answer='It says "I ran the whole company alone" there.', citations=[piece.id]
    )
    answer = Agent(scripted(invented), index).ask("q")
    assert not answer.verified and "quoted span" in answer.check.reason
    paraphrase = _call("finish", answer="Roughly, they did some things.", citations=[piece.id])
    assert "paraphrases" in Agent(scripted(paraphrase), index).ask("q").check.reason
    ghost = _call("finish", answer="x", citations=["ghost.md:1-1"])
    assert "does not exist" in Agent(scripted(ghost), index).ask("q").check.reason


def test_state_drops_old_observations_but_keeps_notes(offline, index):
    agent = Agent(offline, index)
    transcript = [f'search_evidence(query="{i}") ->\n' + "x" * 2000 for i in range(5)]
    state = agent._state(transcript, ["kept"])
    assert len(state) <= STATE_BUDGET + 100
    assert 'query="4"' in state and 'query="0"' not in state and "* kept" in state
