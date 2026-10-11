"""Code-enforced grounding rules. Deliberately separate from the model.

The model proposes a verdict or a CV bullet; this module decides what is
*allowed*. Nothing here calls an LLM, so a model cannot argue its way past it.

Match rules
  G1  every cited fact id must exist in the profile
  G2  a match/partial verdict must cite at least one fact
  G3  coverage ceiling: MATCH needs the cited facts to cover *every* requirement
      skill; covering some caps at PARTIAL; covering none caps at GAP
  G4  evidence ceiling: only production evidence can support MATCH. Personal
      projects and adjacent exposure cap at PARTIAL, however well they fit

Bullet rules
  B1  cites at least one fact, all of which exist
  B2  every number in the bullet appears in the cited facts
  B3  every skill the bullet mentions is a skill of a cited fact
  B4  the bullet never mentions a declared gap
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import BulletOut, Evidence, MatchOut, Profile, Requirement, Verdict, min_verdict
from .taxonomy import Taxonomy

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class MatchDecision:
    verdict: Verdict
    fact_ids: list[str]
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(text)}


def match_ceiling(
    req: Requirement, fact_ids: list[str], profile: Profile
) -> tuple[Verdict, str | None]:
    """The strongest verdict the cited facts can support, and why it is capped."""
    facts = [profile.fact(i) for i in fact_ids if profile.has(i)]
    if not facts:
        return Verdict.GAP, "G3_no_valid_evidence"

    ceiling, reason = Verdict.MATCH, None
    if req.skills:
        covered = set(req.skills) & {s for f in facts for s in f.skills}
        missing = ",".join(sorted(set(req.skills) - covered))
        if not covered:
            return Verdict.GAP, f"G3_uncovered_skills:{missing}"
        if missing:
            ceiling, reason = Verdict.PARTIAL, f"G3_uncovered_skills:{missing}"

    if ceiling is Verdict.MATCH and not any(f.evidence is Evidence.PRODUCTION for f in facts):
        ceiling, reason = Verdict.PARTIAL, "G4_no_production_evidence"
    return ceiling, reason


def evaluate_match(req: Requirement, claim: MatchOut, profile: Profile) -> MatchDecision:
    violations: list[str] = []
    valid = [i for i in claim.fact_ids if profile.has(i)]
    for i in claim.fact_ids:
        if not profile.has(i):
            violations.append(f"G1_unknown_fact:{i}")
    if claim.verdict is not Verdict.GAP and not claim.fact_ids:
        violations.append("G2_no_citation")

    ceiling, reason = match_ceiling(req, valid, profile)
    if claim.verdict.rank > ceiling.rank and reason:
        violations.append(reason)

    return MatchDecision(min_verdict(claim.verdict, ceiling), valid, violations)


def evaluate_bullet(bullet: BulletOut, profile: Profile, taxonomy: Taxonomy) -> list[str]:
    violations: list[str] = []
    valid = [i for i in bullet.fact_ids if profile.has(i)]
    for i in bullet.fact_ids:
        if not profile.has(i):
            violations.append(f"B1_unknown_fact:{i}")
    if not bullet.fact_ids:
        violations.append("B1_no_citation")
        return violations

    facts = [profile.fact(i) for i in valid]
    allowed_numbers: set[str] = set()
    allowed_skills: set[str] = set()
    for f in facts:
        allowed_numbers |= _numbers(f.text)
        for m in f.metrics:
            allowed_numbers |= _numbers(m)
        allowed_skills |= set(f.skills)

    invented = sorted(_numbers(bullet.text) - allowed_numbers)
    if invented:
        violations.append(f"B2_ungrounded_numbers:{','.join(invented)}")

    mentioned = taxonomy.extract(bullet.text)
    gaps = sorted(mentioned & set(profile.declared_gaps))
    if gaps:
        violations.append(f"B4_declared_gap:{','.join(gaps)}")
    extra = sorted(mentioned - allowed_skills - set(profile.declared_gaps))
    if extra:
        violations.append(f"B3_ungrounded_skills:{','.join(extra)}")
    return violations
