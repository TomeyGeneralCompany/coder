"""Optional native sandbox smoke test with a simulated local Ollama server.

Runs the actual OpenHands worker under bubblewrap. No cloud model or local
Qwen download is required. Requires unprivileged user namespaces on Linux.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from coder_local.controller import Controller, Job, JobStore, Settings


class FakeOllama(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply({"models": [{"name": "qwen3:4b"}]})

    def do_POST(self):
        if self.path == "/api/show":
            self.reply({"model_info": {"qwen3.context_length": 32768}})
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tools = [t["function"]["name"] for t in payload.get("tools", [])]
        messages = payload.get("messages", [])
        user_text = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        finish = next((name for name in tools if "finish" in name.lower()), None)
        editor = next((name for name in tools if "fileeditor" in name.lower()), None)
        if not finish:
            self.reply({"message": {"role": "assistant", "content": "Cannot finish"}})
            return
        if "Break the authorized assignment" in user_text:
            call = {
                "name": finish,
                "arguments": {"message": '["Set value to 3 in src/main.py"]'},
            }
        elif editor and "Modify only the staged project" in user_text:
            prior_tool = any(m.get("role") == "tool" for m in messages)
            if prior_tool:
                call = {"name": finish, "arguments": {"message": "File edited"}}
            else:
                matched = re.search(
                    r"Modify only the staged project at ([^\n]+)\.",
                    user_text,
                )
                if not matched:
                    self.reply({"message": {"role": "assistant", "content": "No workspace"}})
                    return
                target = matched.group(1) + "/src/main.py"
                call = {
                    "name": editor,
                    "arguments": {
                        "command": "str_replace",
                        "path": target,
                        "old_str": "value = 0",
                        "new_str": "value = 3",
                    },
                }
        else:
            call = {"name": finish, "arguments": {"message": "Done"}}
        self.reply({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": call}],
            },
            "prompt_eval_count": 90,
            "eval_count": 30,
        })

    def reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


@unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap not installed")
class SandboxEndToEndTest(unittest.TestCase):
    def test_small_change_through_real_restricted_worker(self):
        server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                project = base / "project"
                (project / "src").mkdir(parents=True)
                (project / "src" / "main.py").write_text("value = 0\n")
                store = JobStore(base / "state")
                store.save("smoke", Job(
                    project=str(project),
                    objective="Change src/main.py so that value equals three",
                    allowed_paths=["src"],
                    test_command=["python", "-c", "from src.main import value; assert value == 3"],
                ))
                settings = Settings(
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    max_agent_iterations=5,
                    timeout_seconds=90,
                )
                result = Controller(settings, store).run("smoke")
                self.assertEqual(result.stage, "complete")
                self.assertEqual(result.verification["status"], "passed")
                self.assertEqual((project / "src/main.py").read_text(), "value = 3\n")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
