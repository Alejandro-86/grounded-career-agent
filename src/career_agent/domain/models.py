"""Core domain types. Pure data: no LLM, Spark or MLflow imports."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Evidence(str, Enum):
    """How strongly a fact supports a claim, weakest to strongest."""

    ADJACENT = "adjacent"
    PORTFOLIO = "portfolio"
    PRODUCTION = "production"

    @property
    def rank(self) -> int:
        return {"adjacent": 1, "portfolio": 2, "production": 3}[self.value]


class Verdict(str, Enum):
    GAP = "gap"
    PARTIAL = "partial"
    MATCH = "match"

    @property
    def rank(self) -> int:
        return {"gap": 0, "partial": 1, "match": 2}[self.value]

    @property
    def credit(self) -> float:
        return {"gap": 0.0, "partial": 0.5, "match": 1.0}[self.value]


def min_verdict(a: Verdict, b: Verdict) -> Verdict:
    return a if a.rank <= b.rank else b


class Fact(BaseModel):
    id: str
    text: str
    skills: list[str]
    evidence: Evidence
    metrics: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    name: str
    headline: str
    facts: list[Fact]
    declared_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _gaps_are_structural(self) -> Profile:
        """A declared gap must never appear in any fact's skills.

        This is what makes gaps enforceable in code: if the profile says
        "no Azure", no fact can cover an Azure requirement, so the guardrail's
        coverage ceiling can never let a model claim it.
        """
        ids = [f.id for f in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate fact ids")
        clash = {s for f in self.facts for s in f.skills} & set(self.declared_gaps)
        if clash:
            raise ValueError(f"declared gaps contradicted by facts: {sorted(clash)}")
        return self

    def has(self, fact_id: str) -> bool:
        return any(f.id == fact_id for f in self.facts)

    def fact(self, fact_id: str) -> Fact:
        for f in self.facts:
            if f.id == fact_id:
                return f
        raise KeyError(fact_id)


class Requirement(BaseModel):
    id: str
    text: str
    must_have: bool = True
    skills: list[str] = Field(default_factory=list)


# --- LLM structured outputs -------------------------------------------------


class RequirementOut(BaseModel):
    text: str = Field(description="One requirement, verbatim or lightly condensed from the JD.")
    must_have: bool = Field(description="False for nice-to-have / bonus / preferred items.")
    skills: list[str] = Field(
        default_factory=list, description="Canonical skill names the requirement is about."
    )


class RequirementsOut(BaseModel):
    requirements: list[RequirementOut]


class MatchOut(BaseModel):
    verdict: Verdict
    fact_ids: list[str] = Field(description="IDs of the candidate facts that support the verdict.")
    rationale: str = Field(description="One or two sentences, plain and honest.")


class BulletOut(BaseModel):
    text: str = Field(description="One CV bullet using only the cited facts.")
    fact_ids: list[str]


# --- Pipeline results -------------------------------------------------------


class Match(BaseModel):
    requirement_id: str
    verdict: Verdict
    claimed_verdict: Verdict
    fact_ids: list[str]
    rationale: str
    violations: list[str] = Field(default_factory=list)
    attempts: int = 1
    downgraded: bool = False


class Bullet(BaseModel):
    requirement_id: str
    text: str
    fact_ids: list[str]
    attempts: int = 1


class DroppedBullet(BaseModel):
    requirement_id: str
    text: str
    violations: list[str]


class RunResult(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    title: str
    llm_model: str
    guardrail_enabled: bool
    requirements: list[Requirement]
    matches: list[Match]
    bullets: list[Bullet]
    dropped_bullets: list[DroppedBullet] = Field(default_factory=list)
    score: float
    trace_id: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        c = {v.value: 0 for v in Verdict}
        for m in self.matches:
            c[m.verdict.value] += 1
        return c

    @property
    def n_violations(self) -> int:
        return sum(len(m.violations) for m in self.matches) + sum(
            len(d.violations) for d in self.dropped_bullets
        )
