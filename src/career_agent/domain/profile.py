from __future__ import annotations

import json
from pathlib import Path

from .models import Profile
from .taxonomy import DATA_DIR, Taxonomy, default_taxonomy

DEFAULT_PROFILE = DATA_DIR / "profile" / "sam_okafor.json"


def check_consistency(profile: Profile, taxonomy: Taxonomy) -> None:
    """A fact may not mention a skill it does not tag.

    The bullet guardrail treats a fact's tagged skills as the skills a bullet
    citing it may mention. An untagged mention would make a perfectly faithful
    bullet look ungrounded, so the profile must agree with its own text.
    """
    problems = []
    for f in profile.facts:
        untagged = taxonomy.extract(f.text) - set(f.skills)
        if untagged:
            problems.append(f"{f.id}: text mentions untagged skills {sorted(untagged)}")
    if problems:
        raise ValueError("inconsistent profile:\n" + "\n".join(problems))


def load_profile(path: Path | str = DEFAULT_PROFILE) -> Profile:
    profile = Profile.model_validate(json.loads(Path(path).read_text()))
    check_consistency(profile, default_taxonomy())
    return profile
