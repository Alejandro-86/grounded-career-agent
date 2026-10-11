from __future__ import annotations

from html import escape

from ..domain.models import Profile, RunResult, Verdict

_CSS = """
body{font-family:Helvetica,Arial,sans-serif;font-size:10.5pt;margin:28px 40px;color:#111}
h1{margin:0;font-size:20pt}.sub{color:#555;margin:2px 0 14px}
h2{font-size:11pt;border-bottom:1px solid #999;padding-bottom:2px;margin-top:16px}
li{margin:4px 0}.skills{color:#333}.ref{color:#888;font-size:8pt}
"""


def render_cv_html(profile: Profile, result: RunResult) -> str:
    skills: list[str] = []
    for b in result.bullets:
        for fid in b.fact_ids:
            if profile.has(fid):
                for s in profile.fact(fid).skills:
                    if s not in skills:
                        skills.append(s)
    items = "\n".join(
        f"<li>{escape(b.text)} <span class='ref'>[{', '.join(escape(i) for i in b.fact_ids)}]</span></li>"
        for b in result.bullets
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{escape(profile.name)}</title>
<style>{_CSS}</style></head><body>
<h1>{escape(profile.name)}</h1><div class='sub'>{escape(profile.headline)}</div>
<h2>Selected experience</h2><ul>
{items}
</ul>
<h2>Skills evidenced for this role</h2>
<div class='skills'>{escape(", ".join(skills))}</div>
</body></html>"""


def render_report_md(result: RunResult) -> str:
    by_id = {r.id: r for r in result.requirements}
    icon = {Verdict.MATCH: "MATCH  ", Verdict.PARTIAL: "PARTIAL", Verdict.GAP: "GAP    "}
    lines = [
        f"# Fit report: {result.title}",
        "",
        f"Fit score **{result.score:.0%}** | {result.counts} | model `{result.llm_model}` | "
        f"guardrail {'on' if result.guardrail_enabled else 'OFF'}",
        "",
    ]
    for m in result.matches:
        r = by_id[m.requirement_id]
        tag = "must" if r.must_have else "nice"
        note = ""
        if m.downgraded or m.violations:
            note = f"  (guardrail: claimed {m.claimed_verdict.value}; {'; '.join(sorted(set(m.violations)))})"
        lines.append(
            f"- `{icon[m.verdict]}` [{tag}] {r.text}  -> {', '.join(m.fact_ids) or 'no evidence'}{note}"
        )
    gaps = [by_id[m.requirement_id].text for m in result.matches if m.verdict is Verdict.GAP]
    if gaps:
        lines += ["", "## Gaps to name honestly", *[f"- {g}" for g in gaps]]
    return "\n".join(lines)
