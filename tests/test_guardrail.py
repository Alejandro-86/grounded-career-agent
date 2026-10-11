from __future__ import annotations

import pytest
from pydantic import ValidationError

from career_agent.domain.guardrail import evaluate_bullet, evaluate_match, match_ceiling
from career_agent.domain.models import BulletOut, MatchOut, Profile, Verdict
from career_agent.domain.taxonomy import default_taxonomy

from .conftest import req, small_profile

TAX = default_taxonomy()


def claim(verdict: Verdict, ids: list[str]) -> MatchOut:
    return MatchOut(verdict=verdict, fact_ids=ids, rationale="x")


def test_full_coverage_with_production_evidence_allows_match():
    d = evaluate_match(req("Spark", ["spark"]), claim(Verdict.MATCH, ["p1"]), small_profile())
    assert d.ok and d.verdict is Verdict.MATCH


def test_g1_unknown_fact_is_flagged_and_dropped():
    d = evaluate_match(req("Spark", ["spark"]), claim(Verdict.MATCH, ["p1", "zz"]), small_profile())
    assert "G1_unknown_fact:zz" in d.violations
    assert d.fact_ids == ["p1"]


def test_g2_non_gap_without_citation():
    d = evaluate_match(req("Spark", ["spark"]), claim(Verdict.MATCH, []), small_profile())
    assert "G2_no_citation" in d.violations
    assert d.verdict is Verdict.GAP


def test_g3_partial_coverage_caps_at_partial():
    r = req("Spark and Kafka", ["spark", "kafka"])
    d = evaluate_match(r, claim(Verdict.MATCH, ["p1"]), small_profile())
    assert d.verdict is Verdict.PARTIAL
    assert any(v.startswith("G3_uncovered_skills:kafka") for v in d.violations)


def test_g3_no_coverage_caps_at_gap():
    d = evaluate_match(req("Azure", ["azure"]), claim(Verdict.MATCH, ["p1"]), small_profile())
    assert d.verdict is Verdict.GAP


def test_g4_portfolio_evidence_cannot_match():
    d = evaluate_match(req("Kafka", ["kafka"]), claim(Verdict.MATCH, ["s1"]), small_profile())
    assert d.verdict is Verdict.PARTIAL
    assert "G4_no_production_evidence" in d.violations


def test_g4_adjacent_evidence_cannot_match():
    d = evaluate_match(req("Neo4j", ["neo4j"]), claim(Verdict.MATCH, ["a1"]), small_profile())
    assert d.verdict is Verdict.PARTIAL


def test_conservative_claim_is_never_raised():
    d = evaluate_match(req("Spark", ["spark"]), claim(Verdict.PARTIAL, ["p1"]), small_profile())
    assert d.ok and d.verdict is Verdict.PARTIAL


def test_requirement_without_skills_still_needs_production_evidence():
    assert match_ceiling(req("Be nice", []), ["s1"], small_profile())[0] is Verdict.PARTIAL
    assert match_ceiling(req("Be nice", []), ["p1"], small_profile())[0] is Verdict.MATCH


def test_declared_gap_contradicted_by_a_fact_is_rejected_at_load():
    with pytest.raises(ValidationError, match="declared gaps contradicted"):
        Profile(
            name="x",
            headline="x",
            declared_gaps=["azure"],
            facts=[{"id": "f", "text": "t", "skills": ["azure"], "evidence": "production"}],
        )


def test_duplicate_fact_ids_rejected():
    f = {"id": "f", "text": "t", "skills": [], "evidence": "production"}
    with pytest.raises(ValidationError, match="duplicate"):
        Profile(name="x", headline="x", facts=[f, f])


# ---- bullets -------------------------------------------------------------


def bullet(text: str, ids: list[str]):
    return evaluate_bullet(BulletOut(text=text, fact_ids=ids), small_profile(), TAX)


def test_faithful_bullet_passes():
    assert bullet("Ran Spark in production at 4 TB/day.", ["p1"]) == []


def test_b1_requires_citation():
    assert bullet("Ran Spark.", []) == ["B1_no_citation"]


def test_b2_invented_number():
    v = bullet("Ran Spark in production at 40 TB/day.", ["p1"])
    assert any(x.startswith("B2_ungrounded_numbers:40") for x in v)


def test_b3_skill_not_in_cited_facts():
    v = bullet("Ran Spark and Kafka in production.", ["p1"])
    assert any(x.startswith("B3_ungrounded_skills:kafka") for x in v)


def test_b4_declared_gap_is_named():
    v = bullet("Ran Spark on Azure.", ["p1"])
    assert any(x.startswith("B4_declared_gap:azure") for x in v)
