"""Small compatibility helpers for the local-only Qwen runtime.

No LiteLLM proxy, OpenHands-hosted inference, DigitalOcean, or other cloud
endpoint is supported by Coder. OpenHands prompt identity is retained.
"""
from __future__ import annotations

from typing import Any, Final

from openhands.sdk.llm.native_ollama import NativeOllamaClient, MODEL

OPENHANDS_SYSTEM_IDENTITY: Final[str] = (
    "You are OpenHands agent, a helpful AI assistant that can interact"
    " with a computer to solve tasks."
)


def canonicalize_openhands_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject hosted-model config while allowing persisted local Qwen settings."""
    if "model" in payload and payload["model"] != MODEL:
        raise ValueError("Only local qwen3:4b inference is supported")
    return dict(payload)


def litellm_call_kwargs(model: str, base_url: str | None) -> dict[str, Any]:
    """Legacy call-site compatibility; this does not invoke LiteLLM."""
    NativeOllamaClient(base_url or "http://127.0.0.1:11434", model=model)
    return {"model": model, "api_base": base_url or "http://127.0.0.1:11434"}


def is_openhands_provider_model(model: str | None) -> bool:
    return False


def is_litellm_proxy_model(model: str | None) -> bool:
    return False


def is_openhands_proxy_base_url(base_url: str | None) -> bool:
    return False
