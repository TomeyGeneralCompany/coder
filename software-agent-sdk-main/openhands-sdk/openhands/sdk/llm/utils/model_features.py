"""Capabilities of the sole locally supported Qwen3:4B model."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


def model_matches(model: str | None, patterns: Iterable[str]) -> bool:
    text = (model or "").lower()
    return any(str(pattern).strip().lower() in text for pattern in patterns)


@dataclass(frozen=True)
class ModelFeatures:
    supports_reasoning_effort: bool = False
    thinking_mode: Literal["adaptive", "manual", "none", "unknown"] = "manual"
    supports_sampling_params: bool | None = True
    supports_extended_thinking: bool = False
    supports_prompt_cache: bool = False
    supports_prompt_cache_key: bool = False
    supports_stop_words: bool = True
    supports_responses_api: bool = False
    force_string_serializer: bool = True
    send_reasoning_content: bool = False
    supports_prompt_cache_retention: bool = False
    requires_inline_image_data: bool = False
    supports_vision: bool = False
    sources: Mapping[str, str] = field(default_factory=lambda: {"model": "local-qwen3:4b"})


def get_features(
    model: str | None, model_info: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ModelFeatures:
    if model not in {None, "qwen3:4b"}:
        raise ValueError("Coder supports only the locally installed qwen3:4b model")
    del model_info
    if not overrides:
        return ModelFeatures()
    allowed = set(ModelFeatures.__dataclass_fields__) - {"sources"}
    keys = {key: value for key, value in overrides.items() if key in allowed}
    # Caller may narrow support, but cannot promote unsupported features.
    base = ModelFeatures()
    for key, value in keys.items():
        original = getattr(base, key)
        if isinstance(original, bool) and value is True and original is False:
            raise ValueError(f"Cannot enable unsupported Qwen feature: {key}")
    return ModelFeatures(**keys)
