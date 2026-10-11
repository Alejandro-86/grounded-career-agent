from __future__ import annotations

from ..domain.models import Fact, Requirement

EXTRACT_SYSTEM = """You extract hiring requirements from a job description.
Return every distinct requirement as its own item. Mark nice-to-have / bonus / preferred \
items must_have=false. List the canonical skills each requirement is about. \
Do not invent requirements that are not in the text."""

JUDGE_SYSTEM = """You assess whether a candidate meets ONE job requirement, using ONLY the \
candidate facts provided. Rules:
- Cite the ids of the facts that support your verdict. Never cite an id that is not listed.
- 'match' only if the cited facts cover every skill the requirement names AND at least one \
cited fact is production evidence.
- 'partial' if the facts cover only part of it, or the evidence is a personal/portfolio project \
or adjacent exposure rather than production work.
- 'gap' if nothing listed supports it. Gaps are fine and expected; do not stretch to avoid one.
Evidence levels: production > portfolio (personal project) > adjacent (basic exposure).
Be conservative: an over-claimed match is worse than an honest gap."""

BULLET_SYSTEM = """You write ONE CV bullet for the candidate using ONLY the facts provided.
Rules:
- Every number, technology and outcome in the bullet must appear in the cited facts.
- Never add a number, tool or achievement that is not in the facts.
- If a fact is marked portfolio or adjacent, say so plainly (e.g. 'in a personal project').
- Cite the ids of the facts you used. One or two sentences, active voice."""


def render_facts(facts: list[Fact]) -> str:
    return "\n".join(
        f"[{f.id}] ({f.evidence.value}) {f.text} | skills: {', '.join(f.skills)}" for f in facts
    )


def judge_user(req: Requirement, cands: list[Fact], feedback: list[str] | None) -> str:
    parts = [
        f"Requirement: {req.text}",
        f"Skills named: {', '.join(req.skills) or '(none)'}",
        "",
        "Candidate facts:",
        render_facts(cands) or "(none)",
    ]
    if feedback:
        parts += ["", "Your previous answer was rejected by the grounding checker:", *feedback]
        parts += ["Correct it, or answer 'gap' if the facts do not support the requirement."]
    return "\n".join(parts)


def bullet_user(req: Requirement, facts: list[Fact], feedback: list[str] | None) -> str:
    parts = [
        f"Requirement this bullet should address: {req.text}",
        "",
        "Facts you may use:",
        render_facts(facts),
    ]
    if feedback:
        parts += ["", "Your previous bullet was rejected by the grounding checker:", *feedback]
    return "\n".join(parts)
