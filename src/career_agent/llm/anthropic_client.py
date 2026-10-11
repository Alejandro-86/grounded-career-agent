from __future__ import annotations

import os
from typing import Any

from .base import T

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicLLM:
    """Claude via the Messages API, structured output enforced with forced tool use.

    The pydantic schema becomes a single tool's input schema and ``tool_choice``
    forces the call, so the response is always a schema-validated object rather
    than free text we would have to parse.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
        max_tokens: int = 2048,
    ):
        self.model: str = model or os.getenv("CAREER_AGENT_MODEL") or DEFAULT_MODEL
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client: Any = client
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(
        self,
        *,
        name: str,
        system: str,
        user: str,
        schema: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        tool = {
            "name": name,
            "description": f"Return the {name} result.",
            "input_schema": schema.model_json_schema(),
        }
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": name},
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return schema.model_validate(block.input)
        raise RuntimeError(f"model returned no tool_use block for {name!r}")
