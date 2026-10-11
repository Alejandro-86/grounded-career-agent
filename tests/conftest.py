from __future__ import annotations

from typing import Any

import pytest

from career_agent.domain.models import (
    BulletOut,
    Fact,
    MatchOut,
    Profile,
    Requirement,
    RequirementOut,
    RequirementsOut,
    Verdict,
)
from career_agent.domain.profile import load_profile
from career_agent.domain.taxonomy import default_taxonomy


@pytest.fixture(scope="session")
def profile() -> Profile:
    return load_profile()


@pytest.fixture(scope="session")
def taxonomy():
    return default_taxonomy()


def req(text: str, skills: list[str], must_have: bool = True, rid: str = "r01") -> Requirement:
    return Requirement(id=rid, text=text, skills=skills, must_have=must_have)


def small_profile() -> Profile:
    return Profile(
        name="T",
        headline="t",
        declared_gaps=["azure"],
        facts=[
            Fact(
                id="p1",
                text="Ran Spark in production at 4 TB/day.",
                skills=["spark"],
                evidence="production",
                metrics=["4 TB/day"],
            ),
            Fact(id="s1", text="Built a Kafka demo.", skills=["kafka"], evidence="portfolio"),
            Fact(id="a1", text="Touched Neo4j once.", skills=["neo4j"], evidence="adjacent"),
        ],
    )  # type: ignore[arg-type]


class ScriptedLLM:
    """Test double whose answers are supplied per step by plain callables."""

    model = "scripted"

    def __init__(self, extract=None, judge=None, bullet=None):
        self._extract, self._judge, self._bullet = extract, judge, bullet
        self.calls: list[tuple[str, int]] = []

    def complete(self, *, name: str, system: str, user: str, schema: Any, context=None):
        ctx = context or {}
        self.calls.append((name, ctx.get("attempt", 1)))
        if name == "extract_requirements":
            return self._extract(ctx)
        if name == "judge_match":
            return self._judge(ctx)
        return self._bullet(ctx)


def one_requirement(text: str, skills: list[str] | None = None):
    def extract(_ctx):
        return RequirementsOut(
            requirements=[RequirementOut(text=text, must_have=True, skills=skills or [])]
        )

    return extract


def judge_says(verdict: Verdict, ids: list[str]):
    return lambda ctx: MatchOut(verdict=verdict, fact_ids=ids, rationale="x")


def bullet_says(text: str, ids: list[str]):
    return lambda ctx: BulletOut(text=text, fact_ids=ids)
