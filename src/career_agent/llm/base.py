from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """Structured-output LLM interface.

    ``context`` mirrors the structured payload that is also rendered into
    ``user``. Real clients ignore it; deterministic test doubles read it so they
    do not have to parse prompts.
    """

    model: str

    def complete(
        self,
        *,
        name: str,
        system: str,
        user: str,
        schema: type[T],
        context: dict[str, Any] | None = None,
    ) -> T: ...
