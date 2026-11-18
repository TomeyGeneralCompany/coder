"""Native Ollama protocol and Qwen-only transport contract tests."""
from __future__ import annotations

import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openhands.sdk.llm.native_ollama import (
    NativeOllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    require_loopback,
)


class FakeOllama(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        if self.path == "/api/tags":
            self.send_json({"models": [{"name": "qwen3:4b"}]})
        else:
            self.send_json({"error": "Missing"}, status=404)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.__class__.requests.append((self.path, data))
        if self.path == "/api/show":
            return self.send_json({"model_info": {"qwen3.context_length": 32768}})
        if self.path != "/api/chat":
            return self.send_json({"error": "Missing"}, status=404)
        if data.get("stream"):
            parts = [
                {"message": {"role": "assistant", "content": "Hi"}, "done": False},
                {"message": {"role": "assistant", "content": " there"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ]
            body = "".join(json.dumps(part) + "\n" for part in parts).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return self.send_json({
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": "terminal", "arguments": {"command": "pwd"}}
            }]},
            "prompt_eval_count": 128,
            "eval_count": 25,
            "done_reason": "stop",
        })

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class NativeClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def client(self):
        return NativeOllamaClient(self.url, timeout=15)

    def test_endpoint_rejects_remote_or_credentialed_url(self):
        for url in (
            "https://api.openai.com:443",
            "http://127.0.0.1:11434@remote.invalid:80",
            "http://127.0.0.1:11434/v1",
            "http://localhost:11434?token=secret",
            "http://192.168.1.2:11434",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                require_loopback(url)
        with self.assertRaises(ValueError):
            NativeOllamaClient(self.url, model="llama3:latest")

    def test_model_capabilities_from_native_show(self):
        self.assertEqual(self.client().ensure_model(), 32768)

    def test_native_tool_call_and_context_allocation(self):
        result = self.client().chat(
            [{"role": "user", "content": "Inspect"}],
            num_ctx=16384,
            num_predict=600,
            tools=[{"type": "function", "function": {
                "name": "terminal", "parameters": {"type": "object"},
            }}],
        )
        self.assertEqual(result.model, "qwen3:4b")
        self.assertEqual(result.tool_calls[0].name, "terminal")
        self.assertEqual(result.tool_calls[0].arguments, {"command": "pwd"})
        self.assertTrue(result.tool_calls[0].id)
        self.assertEqual(result.usage.total_tokens, 153)
        path, request = FakeOllama.requests[-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(request["options"]["num_ctx"], 16384)
        self.assertEqual(request["options"]["num_predict"], 600)

    def test_async_chat(self):
        answer = asyncio.run(self.client().achat(
            [{"role": "user", "content": "Inspect"}],
            num_ctx=8192,
            num_predict=512,
        ))
        self.assertEqual(answer.usage.prompt_tokens, 128)

    def test_sync_stream(self):
        deltas = list(self.client().stream(
            [{"role": "user", "content": "Hello"}],
            num_ctx=4096,
            num_predict=128,
        ))
        self.assertEqual("".join(x.content for x in deltas), "Hi there")
        self.assertTrue(deltas[-1].done)

    def test_async_stream(self):
        async def run():
            return [x async for x in self.client().astream(
                [{"role": "user", "content": "Hello"}],
                num_ctx=4096,
                num_predict=128,
            )]
        deltas = asyncio.run(run())
        self.assertEqual("".join(x.content for x in deltas), "Hi there")

    def test_empty_response_cannot_fake_completion(self):
        from openhands.sdk.llm.native_ollama import _complete
        with self.assertRaises(OllamaResponseError):
            _complete({"message": {"role": "assistant", "content": ""}})


if __name__ == "__main__":
    unittest.main()
