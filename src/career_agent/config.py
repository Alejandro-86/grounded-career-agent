from __future__ import annotations

import os

from dotenv import load_dotenv

from .domain.models import Profile
from .llm.base import LLMClient
from .storage.store import MemoryRunStore, RunStore

load_dotenv()


def build_llm(mode: str, profile: Profile, fault_rate: float = 0.0) -> LLMClient:
    """``offline``: deterministic stand-in. ``live``: Claude via the Anthropic API.

    A non-zero ``fault_rate`` wraps either in the fault injector, so the
    guardrail can be stress-tested against a real model's output too.
    """
    if mode == "live":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("live mode needs ANTHROPIC_API_KEY (copy .env.example to .env)")
        from .llm.anthropic_client import AnthropicLLM

        base: LLMClient = AnthropicLLM()
    elif mode == "offline":
        from .llm.fakes import HeuristicLLM

        base = HeuristicLLM(profile)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if fault_rate > 0:
        from .llm.fakes import FaultInjectingLLM

        return FaultInjectingLLM(
            base,
            p_overclaim=fault_rate,
            p_bad_cite=fault_rate / 2,
            p_inflate=fault_rate,
        )
    return base


def build_store(kind: str | None = None, path: str = "./warehouse") -> RunStore:
    kind = kind or os.getenv("CAREER_AGENT_STORE", "memory")
    if kind == "delta":
        from .storage.spark import get_local_spark
        from .storage.store import DeltaRunStore

        spark = get_local_spark()
        spark.sparkContext.setLogLevel("ERROR")
        return DeltaRunStore(spark, path)
    return MemoryRunStore()
