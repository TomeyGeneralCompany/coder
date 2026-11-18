"""Local-only token, latency and completion-log accounting.

No hosted-provider cost service or external observability exporter is used.
The public methods consumed by LLMSession and LLMRegistry remain available.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from openhands.sdk.llm.native_types import ModelResponse
from openhands.sdk.llm.utils.metrics import Metrics


class Telemetry(BaseModel):
    model_name: str = "qwen3:4b"
    completion_logging_enabled: bool = False
    log_enabled: bool = False
    log_dir: str | None = None
    input_cost_per_token: float | None = Field(default=None, ge=0)
    output_cost_per_token: float | None = Field(default=None, ge=0)
    metrics: Metrics

    _req_start: float = PrivateAttr(default=0.0)
    _req_ctx: dict[str, Any] = PrivateAttr(default_factory=dict)
    _last_latency: float = PrivateAttr(default=0.0)
    _log_completions_callback: Callable[[str, str], None] | None = PrivateAttr(default=None)
    _stats_update_callback: Callable[[], None] | None = PrivateAttr(default=None)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def model_copy(self, *, update: Mapping[str, Any] | None = None,
                   deep: bool = False) -> Self:
        copy = super().model_copy(update=update, deep=deep)
        copy._req_start = 0.0
        copy._req_ctx = {}
        return copy

    def set_log_completions_callback(
        self, callback: Callable[[str, str], None] | None
    ) -> None:
        self._log_completions_callback = callback

    def set_stats_update_callback(
        self, callback: Callable[[], None] | None
    ) -> None:
        self._stats_update_callback = callback

    def notify_stats_update(self) -> None:
        if self._stats_update_callback:
            self._stats_update_callback()

    def on_request(self, telemetry_ctx: dict | None = None) -> None:
        self._req_start = time.monotonic()
        # The inference request contains no cloud secret. Do not persist
        # untrusted headers, credentials, or third-party provider metadata.
        ctx = telemetry_ctx or {}
        self._req_ctx = {
            "context_window": ctx.get("context_window", 0),
            "messages": ctx.get("messages", []),
        }

    def on_response(
        self, resp: ModelResponse, raw_resp: ModelResponse | None = None,
        provider_info: object | None = None,
    ) -> Metrics:
        del provider_info
        elapsed = max(0.0, time.monotonic() - self._req_start)
        self._last_latency = elapsed
        self.metrics.add_response_latency(elapsed, resp.id, model_name=self.model_name)
        usage = resp.usage
        self.metrics.add_token_usage(
            prompt_tokens=max(0, int(usage.prompt_tokens)),
            completion_tokens=max(0, int(usage.completion_tokens)),
            cache_read_tokens=0,
            cache_write_tokens=0,
            context_window=int(self._req_ctx.get("context_window") or 0),
            response_id=resp.id,
            reasoning_tokens=0,
            model_name=self.model_name,
        )
        # Local API inference is not billed by a model provider.
        self.metrics.add_cost(0.0, model_name=self.model_name)
        if self.completion_logging_enabled:
            self.log_llm_call(resp, 0.0, raw_resp=raw_resp)
        self.notify_stats_update()
        return self.metrics.deep_copy()

    def on_error(self, _err: BaseException) -> None:
        self._last_latency = max(0.0, time.monotonic() - self._req_start)

    def log_llm_call(
        self, resp: ModelResponse, cost: float | None = None,
        raw_resp: ModelResponse | None = None,
    ) -> None:
        if not self.log_dir and self._log_completions_callback is None:
            return
        filename = (
            f"{self.model_name.replace('/', '__')}-"
            f"{time.time():.3f}-{uuid.uuid4().hex[:8]}.json"
        )
        data = {
            "response": resp.model_dump(mode="json"),
            "cost": 0.0,
            "timestamp": time.time(),
            "latency_sec": self._last_latency,
            "context_window": self._req_ctx.get("context_window", 0),
            "usage_summary": resp.usage.model_dump(),
        }
        if raw_resp is not None:
            data["raw_response"] = raw_resp.model_dump(mode="json")
        encoded = json.dumps(data, ensure_ascii=False)
        if self._log_completions_callback is not None:
            self._log_completions_callback(filename, encoded)
        elif self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            target = os.path.join(self.log_dir, filename)
            with open(target, "x", encoding="utf-8") as stream:
                stream.write(encoded)
