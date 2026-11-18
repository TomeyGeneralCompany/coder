"""Legacy subscription entry point: unavailable in local-only Coder.

No OAuth token exchange, hosted provider credentials, or subscription model
selection is supported by the native Ollama/Qwen runtime.
"""
from __future__ import annotations

from typing import Any


def create_subscription_llm_from_config(*_args: Any, **_kwargs: Any) -> Any:
    raise ValueError(
        "Coder does not support hosted AI subscriptions; use local qwen3:4b"
    )
