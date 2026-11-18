"""Live SDK-contract check using a fake loopback Ollama server.

No cloud inference and no real local model are required. Exercises the actual
OpenHands Agent -> LLM -> local bridge -> native Ollama transport path.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch


class FakeOllamaHandler(BaseHTTPRequestHandler):
    observed_options = []
    tool_names = []

    def do_GET(self):
        self._send({"models": [{"name": "qwen3:4b"}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.__class__.observed_options.append(request["options"])
        tools = request.get("tools") or []
        names = [tool["function"]["name"] for tool in tools]
        self.__class__.tool_names.extend(names)
        finish = next((name for name in names if "finish" in name.lower()), None)
        if finish:
            message = {
                "role": "assistant", "content": "",
                "tool_calls": [{"function": {
                    "name": finish,
                    "arguments": {"message": "Local SDK round-trip completed"},
                }}],
            }
        else:
            message = {"role": "assistant", "content": "Local SDK round-trip completed"}
        self._send({
            "message": message, "prompt_eval_count": 64, "eval_count": 16
        })

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class OpenHandsTransportTest(unittest.TestCase):
    def test_cloud_model_is_rejected_by_sdk_in_coder_mode(self):
        from openhands.sdk import LLM
        with patch.dict(os.environ, {"CODER_LOCAL_ONLY": "1"}):
            with self.assertRaisesRegex(ValueError, "CODER_LOCAL_ONLY"):
                LLM(
                    model="openai/gpt-4o",
                    base_url="http://127.0.0.1:9000/v1",
                    api_key="cloud-not-permitted",
                )
            with self.assertRaisesRegex(ValueError, "CODER_LOCAL_ONLY"):
                LLM(
                    model="openai/qwen3:4b",
                    base_url="https://api.openai.com/v1",
                    api_key="cloud-not-permitted",
                )

    def test_local_model_is_allowed_by_sdk_in_coder_mode(self):
        from openhands.sdk import LLM
        with patch.dict(os.environ, {"CODER_LOCAL_ONLY": "1"}):
            model = LLM(
                model="openai/qwen3:4b",
                base_url="http://127.0.0.1:9000/v1",
                api_key="local-only",
                max_input_tokens=8192,
            )
            self.assertEqual(model.model, "openai/qwen3:4b")

    def test_real_sdk_uses_native_context_and_finish_tool(self):
        from openhands.sdk import Agent, Conversation, LLM
        from openhands.sdk.conversation import get_agent_final_response
        from coder_local.bridge import OllamaBridge

        server = HTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        bridge = OllamaBridge(
            ollama_url=f"http://127.0.0.1:{server.server_port}",
            context_tokens=8192,
            output_tokens=512,
        )
        try:
            bridge.ensure_model()
            url = bridge.start()
            llm = LLM(
                model="openai/qwen3:4b",
                base_url=url,
                api_key="local-only",
                api_mode="chat",
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
                    agent=agent, workspace=Path(folder),
                    max_iteration_per_run=3,
                )
                conversation.send_message("Finish this test now.")
                conversation.run()
                result = get_agent_final_response(conversation.state.events)
            self.assertIsNotNone(result)
            self.assertIn("Local SDK", str(result))
            self.assertIn(8192, [
                options["num_ctx"] for options in FakeOllamaHandler.observed_options
            ])
        finally:
            bridge.stop()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
