"""Deterministic skill extraction.

The guardrail never trusts the model to say which skills a requirement is
about: that would let a model quietly drop the hard skill from a requirement
and then "cover" what's left. Skills are the union of what the model proposed
and what this scanner finds in the raw text.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(os.getenv("CAREER_AGENT_DATA_DIR") or Path(__file__).resolve().parents[3] / "data")


class Taxonomy:
    def __init__(self, spec: dict[str, list[str]]):
        self.canonical = sorted(spec)
        patterns: list[tuple[str, re.Pattern[str]]] = []
        entries: list[tuple[str, str, bool]] = []
        for skill, aliases in spec.items():
            for alias in aliases:
                case_sensitive = alias.startswith("=")
                entries.append((skill, alias.lstrip("="), case_sensitive))
        for skill, alias, cs in sorted(entries, key=lambda e: -len(e[1])):
            flags = 0 if cs else re.IGNORECASE
            patterns.append((skill, re.compile(rf"(?<![\w+#]){re.escape(alias)}(?![\w+#])", flags)))
        self._patterns = patterns

    def extract(self, text: str) -> set[str]:
        return {skill for skill, pat in self._patterns if pat.search(text)}

    def normalise(self, skills: list[str]) -> set[str]:
        known = set(self.canonical)
        out: set[str] = set()
        for s in skills:
            s = s.strip().lower()
            if s in known:
                out.add(s)
            else:
                out |= self.extract(s)
        return out


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    spec = json.loads((DATA_DIR / "skills_taxonomy.json").read_text())
    spec.pop("_comment", None)
    return Taxonomy(spec)
