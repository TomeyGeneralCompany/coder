"""Map local Ollama failures to existing OpenHands agent recovery types."""
from __future__ import annotations

from openhands.sdk.llm.native_ollama import (
    OllamaConnectionError, OllamaContextError,
    OllamaModelNotFoundError, OllamaResponseError,
    OllamaTimeoutError,
)
from .types import (
    LLMBadRequestError, LLMContextWindowExceedError,
    LLMResponseError, LLMServiceUnavailableError, LLMTimeoutError,
)


def map_provider_exception(exception: Exception) -> Exception:
    if isinstance(exception, OllamaContextError):
        return LLMContextWindowExceedError(str(exception))
    if isinstance(exception, OllamaTimeoutError):
        return LLMTimeoutError(str(exception))
    if isinstance(exception, (OllamaConnectionError, OllamaModelNotFoundError)):
        return LLMServiceUnavailableError(str(exception))
    if isinstance(exception, OllamaResponseError):
        return LLMResponseError(str(exception))
    return exception
