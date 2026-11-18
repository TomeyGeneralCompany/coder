"""Focused tests for Coder's local bridge and controller (no model download)."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from coder_local.bridge import OllamaBridge
from coder_local.controller import (
    ContextAllocator, Job, JobStore, Settings, _inspection,
    parse_test_command,
)


class ModelHandler(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        self._send({"models": [{"name": "qwen3:4b"}]})

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.__class__.requests.append(request)
        self._send({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "terminal",
                        "arguments": {"command": "pwd"},
                    }
                }],
            },
            "prompt_eval_count": 60,
            "eval_count": 30,
        })

    def _send(self, obj):
        encoded = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        pass


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), ModelHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_native_context_and_tool_round_trip(self):
        bridge = OllamaBridge(ollama_url=self.url)
        bridge.ensure_model()
        bridge.configure(context_tokens=16384, output_tokens=600)
        bridge_url = bridge.start()
        try:
            import urllib.request

            payload = {
                "model": "qwen3:4b",
                "messages": [
                    {"role": "system", "content": "act locally"},
                    {"role": "user", "content": "inspect"},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "description": "Run a command",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
            }
            req = urllib.request.Request(
                bridge_url + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as response:
                answer = json.load(response)
            sent = ModelHandler.requests[-1]
            self.assertEqual(sent["options"]["num_ctx"], 16384)
            self.assertEqual(sent["options"]["num_predict"], 600)
            self.assertEqual(sent["model"], "qwen3:4b")
            self.assertEqual(
                answer["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "terminal",
            )
            self.assertEqual(answer["usage"]["total_tokens"], 90)
        finally:
            bridge.stop()

    def test_reject_remote_model_endpoint(self):
        with self.assertRaises(ValueError):
            OllamaBridge(ollama_url="https://api.openai.com/v1")

    def test_reject_invalid_context(self):
        with self.assertRaises(ValueError):
            OllamaBridge().configure(context_tokens=1024, output_tokens=1024)


class ControllerTests(unittest.TestCase):
    def test_context_escalation_and_cap(self):
        settings = Settings(max_context=32768)
        allocator = ContextAllocator(settings)
        with patch("coder_local.controller._free_memory_mb", return_value=8192):
            small = allocator.select("small", needs_code=False)
            big = allocator.select("x" * 40000, needs_code=True)
        self.assertTrue(4096 <= small <= big <= 32768)
        self.assertGreater(big, small)

    def test_memory_floor_prevents_growth(self):
        allocator = ContextAllocator(Settings())
        with patch("coder_local.controller._free_memory_mb", return_value=10):
            context = allocator.select("x" * 40000, needs_code=True)
        self.assertLessEqual(context, 8192)

    def test_persistent_job_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = Job(project=directory, objective="Edit a file")
            store.save("task_01", job)
            loaded = store.load("task_01")
            self.assertEqual(loaded.objective, job.objective)
            self.assertEqual(loaded.stage, "intake")
            with self.assertRaises(ValueError):
                store.path("../../outside")

    def test_inspection_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("print('hello')\n")
            (root / ".env").write_text("SECRET=not-for-agent\n")
            text = _inspection(root, "modify app")
            self.assertIn("app.py", text)
            self.assertNotIn("not-for-agent", text)

    def test_command_parsing_does_not_invoke_shell(self):
        self.assertEqual(
            parse_test_command("python -m pytest -q"),
            ["python", "-m", "pytest", "-q"],
        )


if __name__ == "__main__":
    unittest.main()
