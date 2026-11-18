"""Coder exposes one local Qwen model; no cloud provider discovery."""
from __future__ import annotations

from pydantic import SecretStr

MODEL = "qwen3:4b"


def get_supported_llm_models(
    aws_region_name: str | None = None,
    aws_access_key_id: SecretStr | None = None,
    aws_secret_access_key: SecretStr | None = None,
) -> list[str]:
    del aws_region_name, aws_access_key_id, aws_secret_access_key
    return [MODEL]


def _get_litellm_provider_names() -> set[str]:
    """Compatibility name for the existing server model-list endpoint."""
    return {"ollama"}


def _extract_model_and_provider(model: str) -> tuple[str, str, str]:
    if model == MODEL:
        return "ollama", MODEL, ""
    return "", model, ""


def get_unverified_models(
    aws_region_name: str | None = None,
    aws_access_key_id: SecretStr | None = None,
    aws_secret_access_key: SecretStr | None = None,
) -> dict[str, list[str]]:
    del aws_region_name, aws_access_key_id, aws_secret_access_key
    return {"ollama": [MODEL]}


UNVERIFIED_MODELS_EXCLUDING_BEDROCK = {"ollama": [MODEL]}
