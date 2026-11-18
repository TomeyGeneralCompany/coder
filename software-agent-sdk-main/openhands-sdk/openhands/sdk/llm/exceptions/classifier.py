"""Native Ollama error classification; no hosted provider exceptions."""
from __future__ import annotations

from openhands.sdk.llm.native_ollama import (
    OllamaConnectionError, OllamaContextError,
    OllamaModelNotFoundError, OllamaResponseError,
    OllamaTimeoutError,
)
from .types import LLMContextWindowExceedError, LLMMalformedConversationHistoryError


def is_context_window_exceeded(exception: Exception) -> bool:
    return isinstance(exception, (OllamaContextError, LLMContextWindowExceedError))


def looks_like_malformed_conversation_history_error(exception: Exception) -> bool:
    return isinstance(exception, LLMMalformedConversationHistoryError)


def is_prompt_cache_too_small(exception: Exception) -> bool:
    return False


def is_budget_exceeded_error(exception: BaseException) -> bool:
    return False


def is_quota_exhaustion_error(exception: BaseException) -> bool:
    return False


def looks_like_auth_error(exception: Exception) -> bool:
    return False


def is_content_policy_violation(exception: Exception) -> bool:
    return False


def is_transient_http_error(exception: BaseException) -> bool:
    return isinstance(exception, (OllamaConnectionError, OllamaTimeoutError))
