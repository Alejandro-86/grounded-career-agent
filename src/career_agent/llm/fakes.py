"""Deterministic LLM test doubles.

``HeuristicLLM`` behaves like a careful model so the whole pipeline runs
offline. ``FaultInjectingLLM`` wraps any client and corrupts a seeded fraction
of its answers in the specific ways real models fail (over-claiming a
verdict, citing a fact that does not exist, inflating a number, slipping in a
skill the candidate does not have). That makes the guardrail measurable
without an API key. It measures the *guardrail*, not any model's judgement:
live model quality is what ``career-agent eval --live`` is for.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from random import Random
from typing import Any

from ..domain.guardrail import match_ceiling
from ..domain.models import (
    BulletOut,
    Fact,
    MatchOut,
    Profile,
    Requirement,
    RequirementOut,
    RequirementsOut,
    Verdict,
)
from .base import T

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
_OPTIONAL_HEADER_RE = re.compile(r"nice|bonus|prefer|plus", re.IGNORECASE)


class HeuristicLLM:
    model = "heuristic-offline"

    def __init__(self, profile: Profile):
        self.profile = profile

    def complete(
        self,
        *,
        name: str,
        system: str,
        user: str,
        schema: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        ctx = context or {}
        if name == "extract_requirements":
            return self._extract(ctx["jd_text"])  # type: ignore[return-value]
        if name == "judge_match":
            return self._judge(ctx)  # type: ignore[return-value]
        if name == "write_bullet":
            return self._bullet(ctx)  # type: ignore[return-value]
        raise ValueError(f"unknown step {name!r}")

    @staticmethod
    def _extract(jd_text: str) -> RequirementsOut:
        must = True
        out: list[RequirementOut] = []
        for line in jd_text.splitlines():
            m = _BULLET_RE.match(line)
            if m:
                out.append(RequirementOut(text=m.group(1), must_have=must))
            elif line.strip().endswith(":"):
                must = not _OPTIONAL_HEADER_RE.search(line)
        return RequirementsOut(requirements=out)

    def _judge(self, ctx: dict[str, Any]) -> MatchOut:
        req = Requirement.model_validate(ctx["requirement"])
        cands = [Fact.model_validate(f) for f in ctx["candidates"]]
        chosen: list[Fact] = []
        remaining = set(req.skills)
        while remaining:
            gain = max(
                cands,
                key=lambda f: (len(remaining & set(f.skills)), f.evidence.rank),
                default=None,
            )
            if gain is None or not remaining & set(gain.skills):
                break
            chosen.append(gain)
            remaining -= set(gain.skills)
        if not req.skills:
            chosen = cands[:1]
        if not chosen:
            return MatchOut(verdict=Verdict.GAP, fact_ids=[], rationale="No supporting evidence.")
        ids = [f.id for f in chosen]
        ceiling, reason = match_ceiling(req, ids, self.profile)
        why = reason or "Cited facts cover the requirement."
        return MatchOut(verdict=ceiling, fact_ids=ids, rationale=why)

    @staticmethod
    def _bullet(ctx: dict[str, Any]) -> BulletOut:
        facts = [Fact.model_validate(f) for f in ctx["facts"]]
        req = Requirement.model_validate(ctx["requirement"])
        best = max(facts, key=lambda f: len(set(f.skills) & set(req.skills)))
        return BulletOut(text=best.text, fact_ids=[best.id])


class FaultInjectingLLM:
    """Seeded fault injection around a base client."""

    def __init__(
        self,
        base: Any,
        *,
        p_overclaim: float = 0.0,
        p_bad_cite: float = 0.0,
        p_inflate: float = 0.0,
        seed: int = 7,
    ):
        self.base = base
        self.model = f"{base.model}+faults"
        self.p_overclaim, self.p_bad_cite, self.p_inflate = p_overclaim, p_bad_cite, p_inflate
        self.seed = seed
        self.injected: Counter[str] = Counter()

    def _rng(self, name: str, key: str, attempt: int) -> Random:
        return Random(zlib.crc32(f"{self.seed}|{name}|{key}|{attempt}".encode()))

    def complete(
        self,
        *,
        name: str,
        system: str,
        user: str,
        schema: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        out = self.base.complete(
            name=name, system=system, user=user, schema=schema, context=context
        )
        ctx = context or {}
        attempt = ctx.get("attempt", 1)
        if name == "judge_match":
            req = ctx["requirement"]
            rng = self._rng(name, req["text"], attempt)
            if rng.random() < self.p_overclaim and out.verdict is not Verdict.MATCH:
                ids = out.fact_ids or [f["id"] for f in ctx["candidates"][:2]]
                out = MatchOut(
                    verdict=Verdict.MATCH, fact_ids=ids, rationale="Strong match across the board."
                )
                self.injected["overclaim"] += 1
            if rng.random() < self.p_bad_cite:
                out = out.model_copy(update={"fact_ids": [*out.fact_ids, "f99"]})
                self.injected["bad_cite"] += 1
        elif name == "write_bullet":
            rng = self._rng(name, ctx["requirement"]["text"], attempt)
            if rng.random() < self.p_inflate:
                nums = re.search(r"\d[\d,]*", out.text)
                if nums:
                    text = out.text.replace(nums.group(0), nums.group(0) + "0", 1)
                else:
                    text = out.text.rstrip(".") + ", including Azure and Snowflake."
                out = out.model_copy(update={"text": text})
                self.injected["inflate"] += 1
        return out
