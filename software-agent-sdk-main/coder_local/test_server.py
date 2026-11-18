"""Contract checks for the loopback-only Coder job API and dashboard."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from coder_local.controller import JobStore, Settings
from coder_local.server import create_server


class LocalServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name) / "project"
        self.project.mkdir()
        self.store = JobStore(Path(self.temp.name) / "state")
        self.patch = patch("coder_local.server.Controller.run")
        self.fake_run = self.patch.start()
        self.server, self.queue = create_server(
            Settings(), self.store, port=0
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.queue.stop()
        self.patch.stop()
        self.temp.cleanup()

    def request(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            return response.status, response.headers, response.read()

    def test_dashboard_is_local_and_served(self):
        status, headers, content = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        self.assertIn(b"Local Qwen software development", content)
        with self.assertRaises(ValueError):
            create_server(Settings(), self.store, host="0.0.0.0", port=0)

    def test_submit_and_query_assignment(self):
        status, _, body = self.request("/jobs", {
            "project": str(self.project),
            "objective": "Change the site title",
            "test_command": ["python", "-c", "print('ok')"],
            "allowed_paths": ["src"],
        })
        self.assertEqual(status, 202)
        identifier = json.loads(body)["id"]
        status, _, body = self.request("/jobs/" + identifier)
        self.assertEqual(status, 200)
        job = json.loads(body)
        self.assertEqual(job["objective"], "Change the site title")
        self.assertEqual(job["allowed_paths"], ["src"])
        self.queue.items.join()
        status, _, body = self.request("/jobs")
        self.assertEqual(status, 200)
        self.assertIn(identifier, [entry["id"] for entry in json.loads(body)["jobs"]])

    def test_reject_unknown_project(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/jobs", {
                "project": str(self.project / "does-not-exist"),
                "objective": "Do work",
            })
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
