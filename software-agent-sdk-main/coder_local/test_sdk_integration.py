"""Real OpenHands Agent integration exercising the native Ollama API.

The fake server replies in Ollama's native wire format. No OpenAI-compatible
bridge or LiteLLM request path is involved.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.conversation import get_agent_final_response


class FakeOllamaHandler(BaseHTTPRequestHandler):
    observed_options = []
    tool_names = []

    def do_GET(self):
        self._send({"models": [{"name": "qwen3:4b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            self._send({"model_info": {"qwen3.context_length": 32768}})
            return
        if self.path != "/api/chat":
            self._send({"error": "No such endpoint"}, status=404)
            return
        self.__class__.observed_options.append(body.get("options") or {})
        tools = body.get("tools") or []
        names = [tool["function"]["name"] for tool in tools]
        self.__class__.tool_names.extend(names)
        finish = next((name for name in names if "finish" in name.lower()), None)
        if finish:
            message = {
                "role": "assistant", "content": "",
                "tool_calls": [{"function": {
                    "name": finish,
                    "arguments": {"message": "Native Qwen SDK round-trip completed"},
                }}],
            }
        else:
            message = {"role": "assistant", "content": "Native Qwen SDK round-trip completed"}
        self._send({
            "model": "qwen3:4b",
            "message": message,
            "prompt_eval_count": 64,
            "eval_count": 16,
            "done_reason": "stop",
        })

    def _send(self, data, status=200):
        encoded = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        pass


class OpenHandsNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_hosted_model_and_api_credentials_are_rejected(self):
        with self.assertRaises(ValueError):
            LLM(model="openai/gpt-4o", base_url=self.url)
        with self.assertRaises(ValueError):
            LLM(model="qwen3:4b", base_url="https://api.openai.com:443")
        with self.assertRaises(ValueError):
            LLM(model="qwen3:4b", base_url=self.url, api_key="not-allowed")

    def test_approved_qwen_model_is_local(self):
        model = LLM(model="qwen3:4b", base_url=self.url, max_input_tokens=8192)
        self.assertEqual(model.model, "qwen3:4b")
        self.assertEqual(model.effective_max_input_tokens, 8192)

    def test_real_agent_uses_native_context_and_finish_tool(self):
        llm = LLM(
            model="qwen3:4b",
            base_url=self.url,
            max_input_tokens=8192,
            max_output_tokens=512,
            stream=False,
            num_retries=0,
            timeout=30,
        )
        agent = Agent(
            llm=llm,
            tools=[],
            tool_concurrency_limit=1,
            system_prompt="Use the finish tool to end this local test.",
        )
        with tempfile.TemporaryDirectory() as folder:
            conversation = Conversation(
                agent=agent,
                workspace=Path(folder),
                max_iteration_per_run=3,
            )
            conversation.send_message("Finish this test now.")
            conversation.run()
            result = get_agent_final_response(conversation.state.events)
        self.assertIsNotNone(result)
        self.assertIn("Native Qwen", str(result))
        self.assertTrue(FakeOllamaHandler.observed_options)
        self.assertEqual(FakeOllamaHandler.observed_options[-1]["num_ctx"], 8192)
        self.assertGreater(llm.metrics.accumulated_token_usage.prompt_tokens, 0)


if __name__ == "__main__":
    unittest.main()
