"""The locally operated Qwen3:4B model interface for the OpenHands agent.

All inference goes directly to Ollama's native /api/chat endpoint. The public
generate/completion and session methods are kept for the existing OpenHands
Agent, Conversation, and Condenser; no hosted providers or LiteLLM are imported.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from openhands.sdk.llm._tokenizer import (
    ChatTemplateTokenizer, count_tokenized_output, load_chat_template_tokenizer,
)
from openhands.sdk.llm.call_context import LLMCallContext, resolve_llm_call_context
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.llm.llm_session import LLMSession
from openhands.sdk.llm.message import Message, MessageToolCall, TextContent
from openhands.sdk.llm.native_ollama import (
    MODEL, NativeChatResponse, NativeOllamaClient,
    OllamaConnectionError, OllamaError, OllamaTimeoutError,
)
from openhands.sdk.llm.native_types import (
    Delta, ModelResponse, StreamChunk, StreamingChoice, from_native_reply,
)
from openhands.sdk.llm.streaming import AnyTokenCallbackType, _invoke_token_callback
from openhands.sdk.llm.utils.metrics import Metrics
from openhands.sdk.llm.utils.model_features import ModelFeatures, get_features
from openhands.sdk.logger import ENV_LOG_DIR

LLM_PROFILE_SCHEMA_VERSION = 1
MIN_CONTEXT_WINDOW_TOKENS = 4096


def _native_messages(messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        raw = message.to_chat_dict(
            cache_enabled=False,
            vision_enabled=False,
            function_calling_enabled=True,
            force_string_serializer=True,
            send_reasoning_content=False,
        )
        role = raw.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        content = raw.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        row: dict[str, Any] = {"role": role, "content": str(content)}
        if role == "tool":
            name = raw.get("name")
            if name:
                row["tool_name"] = name
        if raw.get("tool_calls"):
            calls = []
            for call in raw["tool_calls"]:
                func = call.get("function") or {}
                args = func.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        raise ValueError("Stored tool call contains invalid JSON")
                if not isinstance(args, dict):
                    raise ValueError("Stored tool arguments must be an object")
                calls.append({
                    "type": "function",
                    "function": {"name": func.get("name"), "arguments": args},
                })
            row["tool_calls"] = calls
        converted.append(row)
    return converted


class LLM(BaseModel):
    """Locally operated Qwen3:4B preserving OpenHands agent-facing contracts."""

    model: str = Field(default=MODEL)
    base_url: str = Field(default="http://127.0.0.1:11434")
    usage_id: str = Field(default="default")
    model_canonical_name: str | None = None
    max_input_tokens: int = Field(default=8192, ge=1024)
    max_output_tokens: int = Field(default=1024, ge=1)
    max_message_chars: int = Field(default=4500, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    seed: int | None = None
    num_retries: int = Field(default=1, ge=0, le=8)
    timeout: int | None = Field(default=1200, ge=1)
    stream: bool = False
    native_tool_calling: bool = True
    custom_tokenizer: str | None = None
    log_completions_to_file: bool = False
    log_completions_folder: str = Field(default=os.path.join(ENV_LOG_DIR, "completions"))
    input_cost_per_token: float | None = 0.0
    output_cost_per_token: float | None = 0.0
    disable_vision: bool | None = True
    disable_stop_word: bool | None = False
    caching_prompt: bool = False
    force_string_serializer: bool = True
    retry_listener: Callable[..., None] | None = Field(default=None, exclude=True)

    _session: LLMSession | None = PrivateAttr(default=None)
    _registry_session: LLMSession | None = PrivateAttr(default=None)
    _chat_template_tokenizer: ChatTemplateTokenizer | None = PrivateAttr(default=None)

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_local(self) -> Self:
        if self.model != MODEL:
            raise ValueError("Coder supports qwen3:4b exclusively")
        NativeOllamaClient(self.base_url, model=self.model, timeout=self.timeout or 1200)
        if self.max_output_tokens >= self.max_input_tokens:
            raise ValueError("Output-token budget must be smaller than context budget")
        if not self.native_tool_calling:
            raise ValueError("Coder requires Qwen native structured tool calling")
        if self._session is None:
            self._session = LLMSession(self)
        if self.custom_tokenizer:
            self._chat_template_tokenizer = load_chat_template_tokenizer(self.custom_tokenizer)
        return self

    @property
    def session(self) -> LLMSession:
        if self._session is None:
            self._session = LLMSession(self)
        return self._session

    @property
    def metrics(self) -> Metrics:
        return self.session.metrics

    @property
    def telemetry(self) -> Any:
        return self.session.telemetry

    @property
    def requires_streaming(self) -> bool:
        return False

    @property
    def effective_max_input_tokens(self) -> int:
        return self.max_input_tokens

    @property
    def effective_max_output_tokens(self) -> int:
        return self.max_output_tokens

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "model": MODEL,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "supports_function_calling": True,
            "supports_vision": False,
        }

    def _model_features(self) -> ModelFeatures:
        return get_features(self.model)

    def resolve_runtime_metadata(self, *args: Any, **kwargs: Any) -> None:
        # The controller obtains installed-model context metadata directly
        # from Ollama; no hosted provider route is available to resolve.
        return None

    async def aresolve_runtime_metadata(self, *args: Any, **kwargs: Any) -> None:
        return None

    @property
    def resolved_runtime_metadata(self) -> None:
        return None


    def vision_is_active(self) -> bool:
        return False

    def is_caching_prompt_active(self) -> bool:
        return False

    def uses_responses_api(self) -> bool:
        return False

    def _registered_session(self) -> LLMSession | None:
        return self._registry_session

    def _join_registry_session(self, session: LLMSession) -> None:
        self._registry_session = session

    def _keep_own_session_apart(self) -> None:
        self._registry_session = None

    def session_for(self, call_context: LLMCallContext | None = None) -> LLMSession:
        context = resolve_llm_call_context(call_context)
        if context.sessions and self.usage_id in context.sessions:
            result = context.sessions[self.usage_id]
            result.follow(self)
            return result
        return self._registry_session or self.session

    def restore_metrics(self, metrics: Metrics) -> None:
        self.session.restore_metrics(metrics)

    def reset_metrics(self) -> None:
        self._session = LLMSession(self)

    def model_copy(self, *, update: Mapping[str, Any] | None = None,
                   deep: bool = False) -> Self:
        copy = super().model_copy(update=update, deep=deep)
        copy.validate_local()
        if update and update.get("usage_id", self.usage_id) != self.usage_id:
            copy._session = LLMSession(copy)
            copy._registry_session = None
        return copy

    def format_messages_for_llm(self, messages: list[Message]) -> list[dict[str, Any]]:
        return _native_messages(messages)

    async def aformat_messages_for_llm(
        self, messages: list[Message],
    ) -> list[dict[str, Any]]:
        return self.format_messages_for_llm(messages)

    def get_token_count(
        self, messages: list[Message], tools: Sequence[Any] | None = None,
        add_security_risk_prediction: bool = False,
    ) -> int:
        source = self.format_messages_for_llm(messages)
        definitions = [
            tool.to_openai_tool(
                add_security_risk_prediction=add_security_risk_prediction
            ) for tool in tools or []
        ]
        tokenizer = self._chat_template_tokenizer
        if tokenizer:
            try:
                result = tokenizer.apply_chat_template(
                    source, tools=definitions or None,
                    tokenize=True, add_generation_prompt=True,
                )
                count = count_tokenized_output(result, tokenizer)
                if count > 0:
                    return count
            except Exception:
                pass
        # If a model tokenizer is not installed, use a conservative bound
        # rather than claiming zero tokens and bypassing condensation.
        chars = len(json.dumps(
            {"messages": source, "tools": definitions}, ensure_ascii=False
        ).encode("utf-8"))
        # A conservative estimate without silently declaring an empty request.
        # Inflate approximate UTF-8 token count for tool-schema overhead, but
        # do not treat every raw byte as an entire token.
        return max(1, math.ceil(chars / 2) + 256 + 32 * len(source))

    def _client(self) -> NativeOllamaClient:
        return NativeOllamaClient(
            self.base_url, model=self.model, timeout=self.timeout or 1200
        )

    def _tools(self, tools: Sequence[Any] | None,
               security: bool) -> list[dict[str, Any]]:
        return [
            item.to_openai_tool(add_security_risk_prediction=security)
            for item in tools or []
        ]

    def _parse_reply(self, reply: NativeChatResponse, session: LLMSession) -> LLMResponse:
        calls = [
            MessageToolCall(
                id=item.id, name=item.name,
                arguments=json.dumps(item.arguments), origin="completion",
            ) for item in reply.tool_calls
        ]
        msg = Message(
            role="assistant",
            content=[TextContent(text=reply.content)] if reply.content else [],
            tool_calls=calls or None,
            reasoning_content=reply.thinking,
        )
        raw = from_native_reply(reply)
        session.telemetry.on_response(raw)
        return LLMResponse(
            message=msg, metrics=session.metrics.get_snapshot(), raw_response=raw
        )

    def _retryable(self, exc: Exception) -> bool:
        return isinstance(exc, (OllamaConnectionError, OllamaTimeoutError))

    def completion(
        self, messages: list[Message], tools: Sequence[Any] | None = None,
        add_security_risk_prediction: bool = False, on_token: Any = None,
        call_context: LLMCallContext | None = None, **kwargs: Any,
    ) -> LLMResponse:
        del kwargs
        session = self.session_for(call_context)
        formatted = self.format_messages_for_llm(messages)
        definitions = self._tools(tools, add_security_risk_prediction)
        for attempt in range(self.num_retries + 1):
            session.telemetry.on_request({
                "context_window": self.max_input_tokens, "messages": formatted,
            })
            try:
                if self.stream and on_token is not None:
                    reply = self._stream_reply(formatted, definitions, on_token)
                else:
                    reply = self._client().chat(
                        formatted, tools=definitions or None,
                        num_ctx=self.max_input_tokens,
                        num_predict=self.max_output_tokens,
                        temperature=self.temperature,
                    )
                return self._parse_reply(reply, session)
            except OllamaError as exc:
                session.telemetry.on_error(exc)
                if not self._retryable(exc) or attempt >= self.num_retries:
                    from openhands.sdk.llm.exceptions import map_provider_exception
                    raise map_provider_exception(exc) from exc
                if self.retry_listener:
                    self.retry_listener(attempt + 1, self.num_retries, exc)
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Unreachable model retry state")

    async def acompletion(
        self, messages: list[Message], tools: Sequence[Any] | None = None,
        add_security_risk_prediction: bool = False, on_token: Any = None,
        call_context: LLMCallContext | None = None, **kwargs: Any,
    ) -> LLMResponse:
        del kwargs
        session = self.session_for(call_context)
        formatted = self.format_messages_for_llm(messages)
        definitions = self._tools(tools, add_security_risk_prediction)
        for attempt in range(self.num_retries + 1):
            session.telemetry.on_request({
                "context_window": self.max_input_tokens, "messages": formatted,
            })
            try:
                if self.stream and on_token is not None:
                    reply = await self._astream_reply(formatted, definitions, on_token)
                else:
                    reply = await self._client().achat(
                        formatted, tools=definitions or None,
                        num_ctx=self.max_input_tokens,
                        num_predict=self.max_output_tokens,
                        temperature=self.temperature,
                    )
                return self._parse_reply(reply, session)
            except OllamaError as exc:
                session.telemetry.on_error(exc)
                if not self._retryable(exc) or attempt >= self.num_retries:
                    from openhands.sdk.llm.exceptions import map_provider_exception
                    raise map_provider_exception(exc) from exc
                if self.retry_listener:
                    self.retry_listener(attempt + 1, self.num_retries, exc)
                await asyncio.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Unreachable model retry state")

    def _stream_reply(self, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]], callback: Any) -> NativeChatResponse:
        from openhands.sdk.llm.native_ollama import NativeToolCall, NativeUsage
        output = []
        thinking = []
        usage = NativeUsage(0, 0)
        calls: list[NativeToolCall] = []
        for part in self._client().stream(
            messages, tools=tools or None, num_ctx=self.max_input_tokens,
            num_predict=self.max_output_tokens,
        ):
            if part.content:
                output.append(part.content)
                callback(StreamChunk(
                    id="local_stream",
                    choices=[StreamingChoice(delta=Delta(content=part.content))],
                ))
            calls.extend(part.tool_calls)
            if part.thinking:
                thinking.append(part.thinking)
            if part.usage is not None:
                usage = part.usage
        return NativeChatResponse(
            id="local_" + uuid4().hex, model=MODEL,
            content="".join(output), thinking="".join(thinking) or None,
            tool_calls=tuple(calls), usage=usage,
            done_reason="stop", raw={},
        )

    async def _astream_reply(
        self, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]], callback: Any,
    ) -> NativeChatResponse:
        from openhands.sdk.llm.native_ollama import NativeToolCall, NativeUsage
        output = []
        thinking = []
        usage = NativeUsage(0, 0)
        calls: list[NativeToolCall] = []
        async for part in self._client().astream(
            messages, tools=tools or None, num_ctx=self.max_input_tokens,
            num_predict=self.max_output_tokens,
        ):
            if part.content:
                output.append(part.content)
                await _invoke_token_callback(callback, StreamChunk(
                    id="local_stream",
                    choices=[StreamingChoice(delta=Delta(content=part.content))],
                ))
            calls.extend(part.tool_calls)
            if part.thinking:
                thinking.append(part.thinking)
            if part.usage is not None:
                usage = part.usage
        return NativeChatResponse(
            id="local_" + uuid4().hex, model=MODEL,
            content="".join(output), thinking=None,
            tool_calls=tuple(calls), usage=NativeUsage(0, 0),
            done_reason="stop", raw={},
        )

    def generate(self, messages: list[Message], tools: Sequence[Any] | None = None,
                 **kwargs: Any) -> LLMResponse:
        return self.completion(messages, tools, **kwargs)

    async def agenerate(self, messages: list[Message], tools: Sequence[Any] | None = None,
                        **kwargs: Any) -> LLMResponse:
        return await self.acompletion(messages, tools, **kwargs)

    @classmethod
    def from_persisted(cls, data: dict[str, Any]) -> "LLM":
        allowed = cls.model_fields.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_persisted(self) -> dict[str, Any]:
        return self.model_dump(exclude={"retry_listener"}, mode="json")

    @classmethod
    def load_from_json(cls, path: str) -> "LLM":
        with open(path, encoding="utf-8") as stream:
            return cls.from_persisted(json.load(stream))

    @classmethod
    def load_from_env(cls, prefix: str = "LLM_") -> "LLM":
        cfg = {}
        for key in cls.model_fields:
            value = os.getenv(prefix + key.upper())
            if value is not None:
                cfg[key] = value
        return cls.model_validate(cfg)
