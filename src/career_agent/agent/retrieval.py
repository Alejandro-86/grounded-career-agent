from __future__ import annotations

import re

from ..domain.models import Fact, Profile, Requirement

_STOP = {
    "the", "and", "for", "with", "you", "our", "are", "will", "have", "has", "that", "this",
    "from", "your", "who", "can", "all", "any", "such", "into", "using", "use", "work", "working",
    "experience", "strong", "ability", "knowledge", "understanding", "years", "year",
}  # fmt: skip


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOP}


def retrieve(req: Requirement, profile: Profile, k: int = 5) -> list[Fact]:
    """Rank profile facts for a requirement: skill overlap first, wording second.

    Skill overlap dominates on purpose; it is the signal the guardrail also
    uses, so retrieval never surfaces a fact the guardrail would then ignore.
    """
    req_tokens = _tokens(req.text)
    scored: list[tuple[float, int, Fact]] = []
    for f in profile.facts:
        skill_hits = len(set(req.skills) & set(f.skills))
        f_tokens = _tokens(f.text) | set(f.skills)
        union = req_tokens | f_tokens
        jaccard = len(req_tokens & f_tokens) / len(union) if union else 0.0
        score = 3 * skill_hits + 2 * jaccard
        if score > 0:
            scored.append((score, f.evidence.rank, f))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2].id))
    return [f for _, _, f in scored[:k]]
