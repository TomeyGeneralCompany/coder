"""Native local-only Coder inference policy (no provider compatibility bridge)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from openhands.sdk.llm.native_ollama import NativeOllamaClient

if TYPE_CHECKING:
    from openhands.sdk.llm.llm import LLM


def enforce_local_only(llm: "LLM") -> None:
    """Reject unsupported model/remote endpoints regardless of environment."""
    NativeOllamaClient(llm.base_url, model=llm.model)
