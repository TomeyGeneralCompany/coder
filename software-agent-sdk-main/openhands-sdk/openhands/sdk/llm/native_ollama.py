"""Native Ollama transport for the one supported Coder model.

No OpenAI-compatible endpoint, cloud authentication, LiteLLM, proxy or fallback.
Only the native HTTP API on loopback is accepted. All response shapes are owned
by OpenHands and use stable, explicit fields for tool calls and usage.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator
from urllib.parse import urlsplit

import httpx


MODEL = "qwen3:4b"


class OllamaError(RuntimeError):
    """Base class for local model failures."""


class OllamaConnectionError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    pass


class OllamaContextError(OllamaError):
    pass


class OllamaResponseError(OllamaError):
    pass


@dataclass(frozen=True)
class NativeToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class NativeUsage:
    prompt_tokens: int
    completion_tokens: int
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    load_duration_ns: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class NativeChatResponse:
    id: str
    model: str
    content: str
    thinking: str | None
    tool_calls: tuple[NativeToolCall, ...]
    usage: NativeUsage
    done_reason: str | None
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class NativeChatDelta:
    content: str
    thinking: str | None = None
    tool_calls: tuple[NativeToolCall, ...] = ()
    done: bool = False
    usage: NativeUsage | None = None


def require_loopback(url: str) -> str:
    endpoint = urlsplit(url)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
        or endpoint.port is None
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.path not in {"", "/"}
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("Coder inference requires an HTTP Ollama loopback endpoint with port")
    return url.rstrip("/")


def _parse_calls(message: dict[str, Any]) -> tuple[NativeToolCall, ...]:
    result = []
    for item in message.get("tool_calls") or []:
        if not isinstance(item, dict):
            raise OllamaResponseError("Invalid tool call object")
        function = item.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise OllamaResponseError("Ollama tool call missing function name")
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError as exc:
                raise OllamaResponseError("Ollama tool arguments contain invalid JSON") from exc
        if not isinstance(args, dict):
            raise OllamaResponseError("Ollama tool arguments must be an object")
        tool_id = item.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            tool_id = "call_" + secrets.token_hex(12)
        result.append(NativeToolCall(
            id=tool_id,
            name=function["name"],
            arguments=args,
        ))
    return tuple(result)


def _complete(raw: dict[str, Any]) -> NativeChatResponse:
    if not isinstance(raw, dict) or not isinstance(raw.get("message"), dict):
        raise OllamaResponseError("Ollama returned no assistant message")
    message = raw["message"]
    if message.get("role", "assistant") != "assistant":
        raise OllamaResponseError("Ollama returned an unexpected message role")
    content = message.get("content") or ""
    thinking = message.get("thinking")
    if not isinstance(content, str) or (thinking is not None and not isinstance(thinking, str)):
        raise OllamaResponseError("Ollama returned invalid assistant content")
    calls = _parse_calls(message)
    if not content and not calls and not thinking:
        raise OllamaResponseError("Ollama returned an empty assistant response")
    return NativeChatResponse(
        id=str(raw.get("id") or "local_" + secrets.token_hex(12)),
        model=str(raw.get("model") or MODEL),
        content=content,
        thinking=thinking,
        tool_calls=calls,
        usage=NativeUsage(
            prompt_tokens=max(0, int(raw.get("prompt_eval_count") or 0)),
            completion_tokens=max(0, int(raw.get("eval_count") or 0)),
            prompt_eval_duration_ns=max(0, int(raw.get("prompt_eval_duration") or 0)),
            eval_duration_ns=max(0, int(raw.get("eval_duration") or 0)),
            load_duration_ns=max(0, int(raw.get("load_duration") or 0)),
        ),
        done_reason=raw.get("done_reason"),
        raw=raw,
    )


class NativeOllamaClient:
    def __init__(
        self,
        url: str = "http://127.0.0.1:11434",
        *,
        model: str = MODEL,
        timeout: float = 1200,
    ) -> None:
        if model != MODEL:
            raise ValueError("Coder supports only qwen3:4b")
        self.url = require_loopback(url)
        self.model = model
        self.timeout = float(timeout)

    def _request(self, messages: list[dict[str, Any]], *, num_ctx: int, num_predict: int,
                 tools: list[dict[str, Any]] | None = None, temperature: float | None = None,
                 stream: bool = False) -> dict[str, Any]:
        if not 1024 <= num_ctx <= 262144:
            raise ValueError("Invalid context allocation")
        if not 1 <= num_predict < num_ctx:
            raise ValueError("Output allocation must be within context window")
        opts: dict[str, Any] = {"num_ctx": num_ctx, "num_predict": num_predict}
        if temperature is not None:
            opts["temperature"] = temperature
        request: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "options": opts, "stream": stream,
        }
        if tools:
            request["tools"] = tools
        return request

    @staticmethod
    def _raise_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = response.text[:1000]
        if response.status_code == 404:
            raise OllamaModelNotFoundError(detail)
        if response.status_code in {400, 413, 422} and "context" in detail.lower():
            raise OllamaContextError(detail)
        raise OllamaResponseError(f"Ollama HTTP {response.status_code}: {detail}")

    @staticmethod
    def _translate_transport(exc: Exception) -> OllamaError:
        if isinstance(exc, httpx.TimeoutException):
            return OllamaTimeoutError(str(exc))
        if isinstance(exc, httpx.RequestError):
            return OllamaConnectionError(str(exc))
        return OllamaResponseError(str(exc))

    def models(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.get(self.url + "/api/tags")
                self._raise_status(response)
                return response.json()
        except httpx.RequestError as exc:
            raise self._translate_transport(exc) from exc

    def show(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.post(self.url + "/api/show", json={"model": self.model})
                self._raise_status(response)
                return response.json()
        except httpx.RequestError as exc:
            raise self._translate_transport(exc) from exc

    def ensure_model(self) -> int | None:
        names = {
            str(item.get("name")) for item in self.models().get("models", [])
            if isinstance(item, dict)
        }
        if self.model not in names:
            raise OllamaModelNotFoundError(f"Local model {self.model} is not installed")
        metadata = self.show().get("model_info") or {}
        values = [
            value for key, value in metadata.items()
            if key.endswith(".context_length") and isinstance(value, int) and value > 0
        ]
        return max(values) if values else None

    def chat(self, messages: list[dict[str, Any]], *, num_ctx: int,
             num_predict: int, tools: list[dict[str, Any]] | None = None,
             temperature: float | None = None) -> NativeChatResponse:
        payload = self._request(messages, num_ctx=num_ctx, num_predict=num_predict,
                                tools=tools, temperature=temperature)
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.post(self.url + "/api/chat", json=payload)
                self._raise_status(response)
                return _complete(response.json())
        except httpx.RequestError as exc:
            raise self._translate_transport(exc) from exc
        except (ValueError, TypeError) as exc:
            raise OllamaResponseError("Invalid Ollama response JSON") from exc

    async def achat(self, messages: list[dict[str, Any]], *, num_ctx: int,
                    num_predict: int, tools: list[dict[str, Any]] | None = None,
                    temperature: float | None = None) -> NativeChatResponse:
        payload = self._request(messages, num_ctx=num_ctx, num_predict=num_predict,
                                tools=tools, temperature=temperature)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(self.url + "/api/chat", json=payload)
                self._raise_status(response)
                return _complete(response.json())
        except httpx.RequestError as exc:
            raise self._translate_transport(exc) from exc
        except (ValueError, TypeError) as exc:
            raise OllamaResponseError("Invalid Ollama response JSON") from exc

    def stream(self, messages: list[dict[str, Any]], *, num_ctx: int,
               num_predict: int, tools: list[dict[str, Any]] | None = None
               ) -> Iterator[NativeChatDelta]:
        payload = self._request(messages, num_ctx=num_ctx, num_predict=num_predict,
                                tools=tools, stream=True)
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                with client.stream("POST", self.url + "/api/chat", json=payload) as response:
                    self._raise_status(response)
                    saw_done = False
                    for line in response.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        message = data.get("message") or {}
                        done = bool(data.get("done"))
                        if done:
                            saw_done = True
                        yield NativeChatDelta(
                            content=message.get("content") or "",
                            thinking=message.get("thinking"),
                            tool_calls=_parse_calls(message),
                            done=done,
                            usage=NativeUsage(
                                prompt_tokens=max(0, int(data.get("prompt_eval_count") or 0)),
                                completion_tokens=max(0, int(data.get("eval_count") or 0)),
                            ) if done else None,
                        )
                    if not saw_done:
                        raise OllamaResponseError("Ollama stream ended before done event")
        except httpx.RequestError as exc:
            raise self._translate_transport(exc) from exc

    async def astream(self, messages: list[dict[str, Any]], *, num_ctx: int,
                      num_predict: int, tools: list[dict[str, Any]] | None = None
                      ) -> AsyncIterator[NativeChatDelta]:
        payload = self._request(messages, num_ctx=num_ctx, num_predict=num_predict,
                                tools=tools, stream=True)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                async with client.stream("POST", self.url + "/api/chat", json=payload) as response:
                    self._raise_status(response)
                    saw_done = False
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        message = data.get("message") or {}
                        done = bool(data.get("done"))
                        if done:
                            saw_done = True
                        yield NativeChatDelta(
                            content=message.get("content") or "",
                            thinking=message.get("thinking"),
                            tool_calls=_parse_calls(message),
                            done=done,
                            usage=NativeUsage(
                                prompt_tokens=max(0, int(data.get("prompt_eval_count") or 0)),
                                completion_tokens=max(0, int(data.get("eval_count") or 0)),
                            ) if done else None,
                        )
                    if not saw_done:
                        raise OllamaResponseError("Ollama stream ended before done event")
        except httpx.RequestError as exc:
            raise self._translate_transport(exc) from exc
