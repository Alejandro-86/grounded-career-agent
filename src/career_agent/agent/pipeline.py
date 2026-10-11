"""The fit/gap + CV-tailoring agent.

Explicit, typed steps rather than a free-running tool loop:

  extract_requirements -> judge_match (per requirement) -> write_bullet (per match)
                              |                                   |
                       grounding guardrail                 grounding guardrail
                       (bounded self-correction, then deterministic downgrade / drop)

The model never decides what is *allowed*; it only proposes. The guardrail
(``domain.guardrail``) is the only thing that can accept a proposal, and when
it keeps rejecting, the run degrades to the honest answer (a lower verdict, a
dropped bullet) instead of shipping an unsupported claim.
"""

from __future__ import annotations

import uuid

from ..domain.guardrail import evaluate_bullet, evaluate_match
from ..domain.models import (
    Bullet,
    BulletOut,
    DroppedBullet,
    Match,
    MatchOut,
    Profile,
    Requirement,
    RequirementsOut,
    RunResult,
    Verdict,
)
from ..domain.taxonomy import Taxonomy, default_taxonomy
from ..llm.base import LLMClient
from ..observability.tracing import NoopTracer, Tracer
from . import prompts
from .retrieval import retrieve

MAX_REQUIREMENTS = 30


class FitAgent:
    def __init__(
        self,
        llm: LLMClient,
        profile: Profile,
        taxonomy: Taxonomy | None = None,
        *,
        guardrail: bool = True,
        max_retries: int = 2,
        max_bullets: int = 8,
        tracer: Tracer | None = None,
    ):
        self.llm = llm
        self.profile = profile
        self.taxonomy = taxonomy or default_taxonomy()
        self.guardrail = guardrail
        self.max_retries = max_retries
        self.max_bullets = max_bullets
        self.tracer = tracer or NoopTracer()

    # ------------------------------------------------------------------ steps

    def extract_requirements(self, jd_text: str) -> list[Requirement]:
        with self.tracer.span("extract_requirements", span_type="LLM") as sp:
            out = self.llm.complete(
                name="extract_requirements",
                system=prompts.EXTRACT_SYSTEM,
                user=jd_text,
                schema=RequirementsOut,
                context={"jd_text": jd_text},
            )
            reqs: list[Requirement] = []
            seen: set[str] = set()
            for r in out.requirements:
                key = " ".join(r.text.lower().split())
                if not key or key in seen:
                    continue
                seen.add(key)
                # Skills are the union of the model's list and a deterministic scan of
                # the text, so a model cannot shrink a requirement to make it easier.
                skills = self.taxonomy.normalise(r.skills) | self.taxonomy.extract(r.text)
                reqs.append(
                    Requirement(
                        id=f"r{len(reqs) + 1:02d}",
                        text=r.text.strip(),
                        must_have=r.must_have,
                        skills=sorted(skills),
                    )
                )
            reqs = reqs[:MAX_REQUIREMENTS]
            sp.set_outputs({"n_requirements": len(reqs)})
        if not reqs:
            raise ValueError("no requirements could be extracted from the job description")
        return reqs

    def judge(self, req: Requirement) -> Match:
        cands = retrieve(req, self.profile)
        feedback: list[str] | None = None
        all_violations: list[str] = []
        first_claim: Verdict | None = None
        last_out: MatchOut | None = None
        attempts = 0
        with self.tracer.span(f"judge_match:{req.id}", span_type="CHAIN") as sp:
            for attempts in range(1, self.max_retries + 2):
                last_out = self.llm.complete(
                    name="judge_match",
                    system=prompts.JUDGE_SYSTEM,
                    user=prompts.judge_user(req, cands, feedback),
                    schema=MatchOut,
                    context={
                        "requirement": req.model_dump(mode="json"),
                        "candidates": [f.model_dump(mode="json") for f in cands],
                        "attempt": attempts,
                    },
                )
                first_claim = first_claim or last_out.verdict
                if not self.guardrail:
                    match = Match(
                        requirement_id=req.id,
                        verdict=last_out.verdict,
                        claimed_verdict=first_claim,
                        fact_ids=last_out.fact_ids,
                        rationale=last_out.rationale,
                        attempts=attempts,
                    )
                    sp.set_outputs(match.model_dump(mode="json"))
                    return match
                decision = evaluate_match(req, last_out, self.profile)
                if decision.ok:
                    match = Match(
                        requirement_id=req.id,
                        verdict=decision.verdict,
                        claimed_verdict=first_claim,
                        fact_ids=decision.fact_ids,
                        rationale=last_out.rationale,
                        violations=all_violations,
                        attempts=attempts,
                    )
                    sp.set_outputs(match.model_dump(mode="json"))
                    return match
                all_violations += decision.violations
                feedback = decision.violations

            # Retries exhausted: degrade to the strongest verdict the evidence supports.
            assert last_out is not None and first_claim is not None
            match = Match(
                requirement_id=req.id,
                verdict=decision.verdict,
                claimed_verdict=first_claim,
                fact_ids=decision.fact_ids,
                rationale=f"Downgraded by grounding guardrail. Model said: {last_out.rationale}",
                violations=all_violations,
                attempts=attempts,
                downgraded=True,
            )
            sp.set_outputs(match.model_dump(mode="json"))
            return match

    def write_bullet(
        self, req: Requirement, match: Match
    ) -> tuple[Bullet | None, DroppedBullet | None]:
        facts = [self.profile.fact(i) for i in match.fact_ids if self.profile.has(i)]
        if not facts:
            return None, None
        feedback: list[str] | None = None
        violations: list[str] = []
        out: BulletOut | None = None
        attempts = 0
        with self.tracer.span(f"write_bullet:{req.id}", span_type="CHAIN") as sp:
            for attempts in range(1, self.max_retries + 2):
                out = self.llm.complete(
                    name="write_bullet",
                    system=prompts.BULLET_SYSTEM,
                    user=prompts.bullet_user(req, facts, feedback),
                    schema=BulletOut,
                    context={
                        "requirement": req.model_dump(mode="json"),
                        "facts": [f.model_dump(mode="json") for f in facts],
                        "attempt": attempts,
                    },
                )
                if not self.guardrail:
                    return Bullet(
                        requirement_id=req.id,
                        text=out.text,
                        fact_ids=out.fact_ids,
                        attempts=attempts,
                    ), None
                found = evaluate_bullet(out, self.profile, self.taxonomy)
                if not found:
                    sp.set_outputs({"text": out.text, "attempts": attempts})
                    return Bullet(
                        requirement_id=req.id,
                        text=out.text,
                        fact_ids=out.fact_ids,
                        attempts=attempts,
                    ), None
                violations += found
                feedback = found
            assert out is not None
            sp.set_outputs({"dropped": True, "violations": violations})
            return None, DroppedBullet(requirement_id=req.id, text=out.text, violations=violations)

    # ------------------------------------------------------------------- run

    def analyze(self, title: str, jd_text: str, run_id: str | None = None) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        with self.tracer.span("analyze", inputs={"title": title, "chars": len(jd_text)}) as root:
            reqs = self.extract_requirements(jd_text)
            matches = [self.judge(r) for r in reqs]
            by_id = {r.id: r for r in reqs}

            ranked = sorted(
                (m for m in matches if m.verdict is not Verdict.GAP),
                key=lambda m: (not by_id[m.requirement_id].must_have, -m.verdict.rank),
            )
            bullets: list[Bullet] = []
            dropped: list[DroppedBullet] = []
            seen_text: set[str] = set()
            for m in ranked:
                if len(bullets) >= self.max_bullets:
                    break
                bullet, drop = self.write_bullet(by_id[m.requirement_id], m)
                if drop:
                    dropped.append(drop)
                elif bullet and bullet.text not in seen_text:
                    seen_text.add(bullet.text)
                    bullets.append(bullet)

            result = RunResult(
                run_id=run_id,
                title=title,
                llm_model=self.llm.model,
                guardrail_enabled=self.guardrail,
                requirements=reqs,
                matches=matches,
                bullets=bullets,
                dropped_bullets=dropped,
                score=score(reqs, matches),
            )
            root.set_outputs(
                {"score": result.score, **result.counts, "violations": result.n_violations}
            )
        result.trace_id = self.tracer.last_trace_id()
        return result


def score(reqs: list[Requirement], matches: list[Match]) -> float:
    """Weighted fit score in [0, 1]: must-have counts double, partial counts half."""
    by_id = {m.requirement_id: m for m in matches}
    total = got = 0.0
    for r in reqs:
        w = 2.0 if r.must_have else 1.0
        total += w
        got += w * by_id[r.id].verdict.credit
    return round(got / total, 3) if total else 0.0
