from __future__ import annotations

from types import SimpleNamespace

import pytest

from career_agent.domain.models import MatchOut, Verdict
from career_agent.domain.profile import check_consistency
from career_agent.domain.taxonomy import default_taxonomy
from career_agent.llm.anthropic_client import AnthropicLLM
from career_agent.observability.tracing import MLflowTracer, NoopTracer, default_tracer

from .conftest import small_profile


class FakeMessages:
    def __init__(self, content):
        self.content, self.kwargs = content, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=self.content, usage=SimpleNamespace(input_tokens=11, output_tokens=7)
        )


def fake_client(content):
    return SimpleNamespace(messages=FakeMessages(content))


def test_anthropic_client_forces_a_schema_validated_tool_call():
    block = SimpleNamespace(
        type="tool_use", input={"verdict": "partial", "fact_ids": ["f1"], "rationale": "r"}
    )
    client = fake_client([block])
    llm = AnthropicLLM("claude-test", client=client)
    out = llm.complete(name="judge_match", system="s", user="u", schema=MatchOut)

    assert out.verdict is Verdict.PARTIAL and out.fact_ids == ["f1"]
    kw = client.messages.kwargs
    assert kw["tool_choice"] == {"type": "tool", "name": "judge_match"}
    assert kw["tools"][0]["input_schema"]["properties"]["verdict"]
    assert (llm.input_tokens, llm.output_tokens) == (11, 7)


def test_anthropic_client_rejects_schema_violations():
    bad = SimpleNamespace(
        type="tool_use", input={"verdict": "definitely", "fact_ids": [], "rationale": "r"}
    )
    with pytest.raises(Exception, match="verdict"):
        AnthropicLLM("m", client=fake_client([bad])).complete(
            name="judge_match", system="s", user="u", schema=MatchOut
        )


def test_anthropic_client_errors_without_tool_use():
    text = SimpleNamespace(type="text", text="hi")
    with pytest.raises(RuntimeError, match="no tool_use"):
        AnthropicLLM("m", client=fake_client([text])).complete(
            name="judge_match", system="s", user="u", schema=MatchOut
        )


def test_live_mode_without_key_fails_loudly(monkeypatch, profile):
    from career_agent.config import build_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_llm("live", profile)


def test_profile_text_must_agree_with_its_tags():
    p = small_profile()
    check_consistency(p, default_taxonomy())
    p.facts[0].text = "Ran Spark and Kafka."
    with pytest.raises(ValueError, match="untagged skills"):
        check_consistency(p, default_taxonomy())


def test_shipped_profile_is_consistent(profile):
    check_consistency(profile, default_taxonomy())


# ---- tracing ---------------------------------------------------------------


def test_noop_tracer_is_inert():
    t = NoopTracer()
    with t.span("x", inputs={"a": 1}) as sp:
        sp.set_outputs({"b": 2})
    assert t.last_trace_id() is None


def test_mlflow_tracer_records_a_nested_trace_with_every_step(tmp_path, profile):
    from career_agent.agent.pipeline import FitAgent
    from career_agent.llm.fakes import HeuristicLLM

    tracer = MLflowTracer(f"sqlite:///{tmp_path}/mlflow.db", experiment="tests")
    jd = "Role\nRequirements:\n- Kafka event streaming\n- Production Go services"
    r = FitAgent(HeuristicLLM(profile), profile, tracer=tracer).analyze("t", jd)

    assert r.trace_id
    import mlflow

    trace = mlflow.get_trace(r.trace_id)
    names = [s.name for s in trace.data.spans]
    assert "analyze" in names and "extract_requirements" in names
    assert sum(n.startswith("judge_match") for n in names) == 2
    assert any(n.startswith("write_bullet") for n in names)


def test_default_tracer_degrades_to_noop_when_backend_unusable(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "badscheme://nope")
    assert isinstance(default_tracer(True), NoopTracer)
    assert isinstance(default_tracer(False), NoopTracer)
