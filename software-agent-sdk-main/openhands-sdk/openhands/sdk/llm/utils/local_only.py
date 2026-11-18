"""Coder local-only inference policy.

The broader upstream SDK still contains provider compatibility code. When Coder
sets CODER_LOCAL_ONLY=1, these helpers prohibit selecting hosted models,
provider connections, subscriptions, or external fallback destinations.
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from openhands.sdk.llm.llm import LLM


def enforce_local_only(llm: "LLM") -> None:
    if os.environ.get("CODER_LOCAL_ONLY", "").lower() not in {"1", "true", "yes"}:
        return
    base = urlsplit(llm.base_url or "")
    if (
        llm.model != "openai/qwen3:4b"
        or base.scheme != "http"
        or base.hostname not in {"127.0.0.1", "localhost"}
        or base.port is None
        or base.username is not None
        or base.password is not None
        or base.path.rstrip("/") not in {"", "/v1"}
        or base.query
        or base.fragment
        or llm.auth_type != "api_key"
        or llm.provider_connection_id is not None
        or llm.subscription_vendor is not None
        or llm.fallback_strategy is not None
        or llm.aws_access_key_id is not None
        or llm.aws_secret_access_key is not None
        or llm.aws_session_token is not None
    ):
        raise ValueError(
            "CODER_LOCAL_ONLY requires Qwen3:4B through the loopback bridge; "
            "hosted LLM providers, provider credentials and fallbacks are forbidden"
        )
