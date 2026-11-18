"""Local-only Qwen model discovery for the OpenHands-compatible server."""
from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from openhands.sdk.llm.native_ollama import MODEL

llm_router = APIRouter(prefix="/llm", tags=["LLM"])


class ProvidersResponse(BaseModel):
    providers: list[str]


class ModelsResponse(BaseModel):
    models: list[str]


class VerifiedModelsResponse(BaseModel):
    models: dict[str, list[str]]


@llm_router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    return ProvidersResponse(providers=["ollama"])


@llm_router.get("/models", response_model=ModelsResponse)
async def list_models(
    provider: str | None = Query(default=None, description="Only Ollama is supported"),
) -> ModelsResponse:
    return ModelsResponse(models=[MODEL] if provider in (None, "ollama") else [])


@llm_router.get("/models/verified", response_model=VerifiedModelsResponse)
async def list_verified_models() -> VerifiedModelsResponse:
    return VerifiedModelsResponse(models={"ollama": [MODEL]})
