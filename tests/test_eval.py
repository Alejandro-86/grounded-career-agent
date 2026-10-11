from __future__ import annotations

from career_agent.domain.models import Requirement
from career_agent.domain.taxonomy import default_taxonomy
from career_agent.eval.harness import align, format_report, jd_text, load_eval_set, run_eval
from career_agent.llm.fakes import FaultInjectingLLM, HeuristicLLM


def test_eval_set_is_well_formed(profile):
    tax = default_taxonomy()
    jds = load_eval_set()
    assert len(jds) >= 6
    for jd in jds:
        for r in jd["requirements"]:
            assert r["gold"] in {"match", "partial", "gap"}
            # every gold requirement must name at least one known skill, or its label is undefined
            assert tax.extract(r["text"]), r["text"]


def test_gold_labels_agree_with_the_evidence_ladder(profile):
    """Derive every label straight from the profile, independently of the agent code."""
    tax = default_taxonomy()
    for jd in load_eval_set():
        for r in jd["requirements"]:
            need = tax.extract(r["text"])
            covering = [f for f in profile.facts if set(f.skills) & need]
            covered = {s for f in covering for s in f.skills} & need
            if not covered:
                expected = "gap"
            elif covered == need and any(f.evidence.value == "production" for f in covering):
                expected = "match"
            else:
                expected = "partial"
            assert r["gold"] == expected, (jd["id"], r["text"])


def test_align_matches_paraphrases_by_token_overlap():
    gold = [{"text": "Hands-on Spark and Databricks pipelines"}, {"text": "Snowflake experience"}]
    reqs = [
        Requirement(id="r01", text="Experience with Snowflake"),
        Requirement(id="r02", text="Hands-on Spark and Databricks pipelines at scale"),
    ]
    got = align(gold, reqs)
    assert got[0].id == "r02" and got[1].id == "r01"


def test_align_leaves_unmatched_gold_out():
    assert align([{"text": "Rust"}], [Requirement(id="r01", text="Kafka streaming")]) == {}


def test_jd_text_roundtrips_through_the_offline_extractor(profile):
    jd = load_eval_set()[0]
    out = HeuristicLLM(profile)._extract(jd_text(jd))
    assert [r.text for r in out.requirements] == [r["text"] for r in jd["requirements"]]
    assert [r.must_have for r in out.requirements] == [r["must_have"] for r in jd["requirements"]]


def test_clean_model_is_correct_with_and_without_guardrail(profile):
    rep = run_eval(lambda: HeuristicLLM(profile), profile)
    for cfg in rep.configs.values():
        assert cfg["verdict_accuracy"] == 1.0
        assert cfg["fabrication_rate"] == 0.0
    assert (
        rep.configs["guarded"]["underclaim_rate"] == 0.0
    )  # guardrail costs nothing on a clean model


def test_guardrail_contains_a_fully_sycophantic_model(profile):
    rep = run_eval(
        lambda: FaultInjectingLLM(
            HeuristicLLM(profile), p_overclaim=1, p_bad_cite=0.5, p_inflate=1
        ),
        profile,
    )
    base, guarded = rep.configs["baseline_no_guardrail"], rep.configs["guarded"]
    assert base["fabrication_rate"] == 1.0
    assert base["ungrounded_bullet_rate"] == 1.0
    assert guarded["fabrication_rate"] == 0.0
    assert guarded["overclaim_rate"] == 0.0
    assert guarded["unsupported_citation_rate"] == 0.0
    assert guarded["ungrounded_bullet_rate"] == 0.0


def test_realistic_fault_rate_shows_a_clear_gap(profile):
    rep = run_eval(
        lambda: FaultInjectingLLM(
            HeuristicLLM(profile), p_overclaim=0.25, p_bad_cite=0.1, p_inflate=0.2
        ),
        profile,
    )
    assert rep.configs["baseline_no_guardrail"]["fabrication_rate"] > 0.1
    assert rep.configs["guarded"]["fabrication_rate"] == 0.0
    assert (
        rep.configs["guarded"]["verdict_accuracy"]
        >= rep.configs["baseline_no_guardrail"]["verdict_accuracy"]
    )


def test_report_formats(profile):
    rep = run_eval(lambda: HeuristicLLM(profile), profile, jds=load_eval_set()[:1])
    text = format_report(rep)
    assert "FABRICATION" in text and "guarded" in text


def test_taxonomy_case_sensitive_short_tokens():
    t = default_taxonomy()
    assert t.extract("Production Go services") == {"go"}
    assert t.extract("Let's go further") == set()
    assert t.extract("Rust and Scala") == {"rust", "scala"}
    assert t.extract("CI/CD with GitHub Actions") == {"ci-cd"}
    assert t.normalise(["Postgres", "k8s"]) == {"postgresql", "kubernetes"}
