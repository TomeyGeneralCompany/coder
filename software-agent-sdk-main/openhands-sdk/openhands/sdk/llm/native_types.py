"""OpenHands-owned message, tool and streaming response schemas.

These are the small structures needed by the local Qwen coding engine; they do
not emulate external provider routing or depend on LiteLLM.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class ChatToolFunction(BaseModel):
    name: str
    arguments: str


class ChatToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ChatToolFunction


class NativeMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None
    provider_specific_fields: dict[str, Any] | None = None


class Choice(BaseModel):
    index: int = 0
    message: NativeMessage
    finish_reason: str | None = None


class Delta(BaseModel):
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StreamingChoice(BaseModel):
    index: int = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: str | None = None


class StreamChunk(BaseModel):
    id: str
    choices: list[StreamingChoice]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelResponse(BaseModel):
    id: str
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class ResponseFunctionToolCall(BaseModel):
    id: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    type: str = "function_call"


class OutputFunctionToolCall(ResponseFunctionToolCall):
    pass


ChatCompletionToolParam = dict[str, Any]
ChatCompletionToolParamFunctionChunk = dict[str, Any]


def from_native_reply(reply: Any) -> ModelResponse:
    calls = [
        ChatToolCall(
            id=call.id,
            function=ChatToolFunction(
                name=call.name, arguments=__import__("json").dumps(call.arguments)
            ),
        )
        for call in reply.tool_calls
    ]
    return ModelResponse(
        id=reply.id,
        model=reply.model,
        choices=[Choice(
            message=NativeMessage(
                role="assistant",
                content=reply.content,
                tool_calls=calls or None,
                reasoning_content=reply.thinking,
            ),
            finish_reason=("tool_calls" if calls else "stop"),
        )],
        usage=Usage(
            prompt_tokens=reply.usage.prompt_tokens,
            completion_tokens=reply.usage.completion_tokens,
            total_tokens=reply.usage.total_tokens,
        ),
    )
