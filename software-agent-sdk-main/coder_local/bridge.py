"""Loopback-only OpenAI chat adapter backed by Ollama's native /api/chat API.

OpenHands uses an OpenAI-shaped chat interface. This adapter preserves that
interface while sending every inference request to the native Ollama endpoint,
where options.num_ctx is applied per request.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from typing import Any


class OllamaBridge:
    def __init__(
        self,
        model: str = "qwen3:4b",
        ollama_url: str = "http://127.0.0.1:11434",
        context_tokens: int = 8192,
        output_tokens: int = 1024,
        timeout_seconds: int = 900,
    ) -> None:
        address = urlsplit(ollama_url)
        if (
            address.scheme != "http"
            or address.hostname not in {"127.0.0.1", "localhost"}
            or address.port is None
            or address.username is not None
            or address.password is not None
            or address.path not in {"", "/"}
            or address.query
            or address.fragment
        ):
            raise ValueError("Ollama endpoint must be an HTTP loopback URL with a port")
        if model != "qwen3:4b":
            raise ValueError("Coder is configured for local qwen3:4b only")
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.context_tokens = context_tokens
        self.output_tokens = output_tokens
        self.timeout_seconds = timeout_seconds
        self.supported_context: int | None = None
        self._lock = threading.RLock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def configure(self, *, context_tokens: int, output_tokens: int) -> None:
        if context_tokens < 1024 or output_tokens < 1:
            raise ValueError("Invalid context/output token budget")
        if output_tokens >= context_tokens:
            raise ValueError("Output budget must be less than context window")
        with self._lock:
            self.context_tokens = context_tokens
            self.output_tokens = output_tokens

    def models(self) -> dict[str, Any]:
        return self._ollama_get("/api/tags")

    def ensure_model(self) -> None:
        data = self.models()
        names = {str(item.get("name", "")) for item in data.get("models", [])}
        if self.model not in names:
            raise RuntimeError(
                f"Model {self.model!r} not installed in local Ollama; "
                f"available: {sorted(names)}"
            )
        # Model metadata is advisory: some Ollama versions do not return it.
        try:
            req = urllib.request.Request(
                self.ollama_url + "/api/show",
                data=json.dumps({"model": self.model}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                info = json.load(response).get("model_info") or {}
            limits = [
                value for name, value in info.items()
                if name.endswith(".context_length")
                and isinstance(value, int) and value > 0
            ]
            self.supported_context = max(limits) if limits else None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            self.supported_context = None

    def _ollama_get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(
            self.ollama_url + path, timeout=15
        ) as response:
            return json.load(response)

    @staticmethod
    def _native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            native: dict[str, Any] = {"role": role, "content": content}
            if role == "tool" and message.get("name"):
                native["tool_name"] = message["name"]
            if message.get("tool_calls"):
                calls = []
                for call in message["tool_calls"]:
                    function = call.get("function", {})
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    calls.append(
                        {"type": "function", "function": {
                            "name": function.get("name", ""),
                            "arguments": args,
                        }}
                    )
                native["tool_calls"] = calls
            output.append(native)
        return output

    def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("stream"):
            raise ValueError("Streaming is not enabled for this local adapter")
        with self._lock:
            ctx = self.context_tokens
            output = min(self.output_tokens, ctx - 256)
            payload = {
                "model": self.model,
                "stream": False,
                "messages": self._native_messages(request.get("messages", [])),
                "options": {"num_ctx": ctx, "num_predict": output},
            }
            if request.get("tools"):
                payload["tools"] = request["tools"]
            if request.get("temperature") is not None:
                payload["options"]["temperature"] = request["temperature"]
            encoded = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.ollama_url + "/api/chat",
                data=encoded,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(
                req, timeout=self.timeout_seconds
            ) as response:
                result = json.load(response)

        original = result.get("message", {})
        calls = []
        for call in original.get("tool_calls", []):
            function = call.get("function", {})
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:20],
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": json.dumps(function.get("arguments") or {}),
                },
            })
        message: dict[str, Any] = {
            "role": "assistant",
            "content": original.get("content") or None,
        }
        if calls:
            message["tool_calls"] = calls
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "coder-local/" + self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if calls else "stop",
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": (
                    result.get("prompt_eval_count", 0)
                    + result.get("eval_count", 0)
                ),
            },
        }

    def start(self) -> str:
        if self._server is not None:
            raise RuntimeError("Bridge already running")
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path.rstrip("/") != "/v1/chat/completions":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4_000_000:
                        raise ValueError("Invalid request length")
                    payload = json.loads(self.rfile.read(length))
                    response = bridge.completion(payload)
                    status = 200
                except (ValueError, KeyError) as exc:
                    status = 400
                    response = {"error": {"message": str(exc), "type": "invalid_request_error"}}
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    status = 502
                    response = {"error": {"message": str(exc), "type": "ollama_connection_error"}}
                except Exception as exc:
                    status = 500
                    response = {"error": {"message": str(exc), "type": "local_bridge_error"}}
                body = json.dumps(response).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/v1/models":
                    result = {"object": "list", "data": [
                        {"id": bridge.model, "object": "model"}
                    ]}
                    body = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="coder-ollama-bridge", daemon=True
        )
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "OllamaBridge":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
