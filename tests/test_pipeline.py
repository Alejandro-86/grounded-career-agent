from __future__ import annotations

import pytest

from career_agent.agent.pipeline import FitAgent, score
from career_agent.domain.models import Verdict
from career_agent.llm.fakes import FaultInjectingLLM, HeuristicLLM

from .conftest import (
    ScriptedLLM,
    bullet_says,
    judge_says,
    one_requirement,
    small_profile,
)


def agent(llm, **kw) -> FitAgent:
    return FitAgent(llm, small_profile(), **kw)


def test_clean_run_end_to_end(profile):
    jd = "Role\nRequirements:\n- Kafka event streaming\n- Production Go services\nNice to have:\n- Snowflake"
    r = FitAgent(HeuristicLLM(profile), profile).analyze("Role", jd)
    verdicts = {m.requirement_id: m.verdict for m in r.matches}
    assert verdicts == {"r01": Verdict.MATCH, "r02": Verdict.GAP, "r03": Verdict.GAP}
    assert [b.fact_ids for b in r.bullets] == [["f03"]]
    assert r.n_violations == 0
    assert r.score == pytest.approx(0.4)  # 2 of 5 weighted points: the kafka match


def test_overclaiming_model_is_downgraded_after_bounded_retries():
    llm = ScriptedLLM(
        extract=one_requirement("Kafka experience", ["kafka"]),
        judge=judge_says(Verdict.MATCH, ["s1"]),  # portfolio-only evidence, always claims MATCH
        bullet=bullet_says("Built a Kafka demo.", ["s1"]),
    )
    r = agent(llm, max_retries=2).analyze("t", "- Kafka experience")
    m = r.matches[0]
    assert m.claimed_verdict is Verdict.MATCH
    assert m.verdict is Verdict.PARTIAL and m.downgraded
    assert m.attempts == 3  # 1 try + 2 retries, then deterministic downgrade
    assert [c for c in llm.calls if c[0] == "judge_match"] == [
        ("judge_match", i) for i in (1, 2, 3)
    ]


def test_self_correction_succeeds_when_model_fixes_itself():
    state = {"n": 0}

    def judge(ctx):
        state["n"] += 1
        from career_agent.domain.models import MatchOut

        if state["n"] == 1:
            return MatchOut(verdict=Verdict.MATCH, fact_ids=["s1"], rationale="overclaim")
        return MatchOut(verdict=Verdict.PARTIAL, fact_ids=["s1"], rationale="portfolio only")

    llm = ScriptedLLM(
        extract=one_requirement("Kafka experience", ["kafka"]),
        judge=judge,
        bullet=bullet_says("Built a Kafka demo.", ["s1"]),
    )
    m = agent(llm).analyze("t", "- Kafka experience").matches[0]
    assert m.verdict is Verdict.PARTIAL and not m.downgraded and m.attempts == 2
    assert m.violations  # the rejected first attempt is still on the record


def test_invented_fact_ids_never_survive():
    llm = ScriptedLLM(
        extract=one_requirement("Spark", ["spark"]),
        judge=judge_says(Verdict.MATCH, ["p1", "f99"]),
        bullet=bullet_says("Ran Spark in production at 4 TB/day.", ["p1"]),
    )
    r = agent(llm).analyze("t", "- Spark")
    assert r.matches[0].fact_ids == ["p1"]


def test_model_cannot_shrink_a_requirement_to_cover_it():
    """The model reports no skills for an Azure requirement; the text scan restores them."""
    llm = ScriptedLLM(
        extract=one_requirement("Azure data platform experience", skills=[]),
        judge=judge_says(Verdict.MATCH, ["p1"]),
        bullet=bullet_says("Ran Spark in production at 4 TB/day.", ["p1"]),
    )
    r = agent(llm).analyze("t", "- Azure data platform experience")
    assert r.requirements[0].skills == ["azure"]
    assert r.matches[0].verdict is Verdict.GAP


def test_ungrounded_bullet_is_dropped_not_repaired():
    llm = ScriptedLLM(
        extract=one_requirement("Spark", ["spark"]),
        judge=judge_says(Verdict.MATCH, ["p1"]),
        bullet=bullet_says("Ran Spark at 400 TB/day.", ["p1"]),
    )
    r = agent(llm, max_retries=1).analyze("t", "- Spark")
    assert r.bullets == []
    assert r.dropped_bullets and r.dropped_bullets[0].violations[0].startswith("B2_")


def test_guardrail_off_is_the_raw_baseline():
    llm = ScriptedLLM(
        extract=one_requirement("Azure", ["azure"]),
        judge=judge_says(Verdict.MATCH, ["p1", "f99"]),
        bullet=bullet_says("Ran Spark at 400 TB/day.", ["p1"]),
    )
    r = agent(llm, guardrail=False).analyze("t", "- Azure")
    assert r.matches[0].verdict is Verdict.MATCH  # fabricated cover accepted
    assert r.matches[0].fact_ids == ["p1", "f99"]
    assert r.bullets[0].text == "Ran Spark at 400 TB/day."


def test_empty_jd_is_an_error():
    with pytest.raises(ValueError, match="no requirements"):
        agent(HeuristicLLM(small_profile())).analyze("t", "just some prose with no bullets")


def test_fault_injection_is_deterministic(profile):
    def run():
        llm = FaultInjectingLLM(
            HeuristicLLM(profile), p_overclaim=0.5, p_bad_cite=0.3, p_inflate=0.5
        )
        jd = "Role\nRequirements:\n- Kafka event streaming\n- Spark and Databricks\n- Terraform on GCP"
        return (
            FitAgent(llm, profile).analyze("t", jd, run_id="x").model_dump(exclude={"created_at"})
        )

    assert run() == run()


def test_sycophant_cannot_get_a_gap_through(profile):
    llm = FaultInjectingLLM(HeuristicLLM(profile), p_overclaim=1.0, p_bad_cite=1.0, p_inflate=1.0)
    jd = "Role\nRequirements:\n- Production Go services\n- Azure deployments\n- Snowflake"
    r = FitAgent(llm, profile).analyze("t", jd)
    assert all(m.verdict is Verdict.GAP for m in r.matches)
    assert r.counts["match"] == 0


def test_score_weights_must_haves_double(profile):
    jd = "Role\nRequirements:\n- Kafka\nNice to have:\n- Snowflake"
    r = FitAgent(HeuristicLLM(profile), profile).analyze("t", jd)
    assert score(r.requirements, r.matches) == pytest.approx(2 / 3, abs=1e-3)
