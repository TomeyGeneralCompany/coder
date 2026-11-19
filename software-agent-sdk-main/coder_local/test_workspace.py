"""Browser workspace, console, Git clone and Linux shortcut contract tests."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from coder_local.desktop import install_launcher
from coder_local.workspace import clone_repository, console_command, project_directory
from coder_local.controller import JobStore, Settings
from coder_local.server import create_server
import threading


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "checkout"
        self.project.mkdir()
        (self.project / "README.md").write_text("hello", encoding="utf-8")

    def test_terminal_runs_one_command_in_selected_checkout(self):
        quoted = f'{sys.executable} -c "from pathlib import Path; print(Path.cwd().name)"'
        result = console_command(str(self.project), quoted)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("checkout", result["output"])
        self.assertEqual(result["project"], str(self.project))

    def test_terminal_rejects_shell_operators(self):
        with self.assertRaisesRegex(ValueError, "operators"):
            console_command(str(self.project), "echo hello && rm -rf sample")
        with self.assertRaisesRegex(ValueError, "operators"):
            console_command(str(self.project), "git status | cat")

    def test_terminal_rejects_missing_project(self):
        with self.assertRaises(ValueError):
            project_directory(str(self.project / "missing"))

    def test_clone_accepts_only_valid_https_github(self):
        for url in (
            "https://evil.example/owner/repo",
            "http://github.com/owner/repo",
            "https://name:secret@github.com/owner/repo",
            "https://github.com/owner/repo?x=1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                clone_repository(url, "", self.root / "projects")
        with self.assertRaises(ValueError):
            clone_repository("https://github.com/owner/repo", "../escape", self.root / "projects")

    @patch("coder_local.workspace.subprocess.run")
    def test_clone_uses_no_credential_prompt_and_separate_destination(self, run):
        run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"Cloned")
        result = clone_repository(
            "https://github.com/TomeyGeneralCompany/Isaiz-Computer-Tool",
            "LOTUS", self.root / "projects"
        )
        args, kwargs = run.call_args
        self.assertEqual(args[0][:4], ["git", "-c", "credential.helper=", "clone"])
        self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(result["project"], str((self.root / "projects" / "LOTUS").resolve()))

    def test_shortcut_created_without_root_access(self):
        desktop = self.root / "user" / "Desktop"
        desktop.mkdir(parents=True)
        checkout = Path(__file__).resolve().parent.parent
        result = install_launcher(home=desktop.parent, checkout=checkout)
        self.assertTrue(Path(result["launcher"]).is_file())
        self.assertTrue(Path(result["desktop_icon"]).is_file())
        self.assertIn("Icon=applications-development", Path(result["launcher"]).read_text())
        wrapper = Path(result["command"])
        self.assertTrue(wrapper.stat().st_mode & 0o111)
        self.assertIn("--open-browser", wrapper.read_text())
        self.assertNotIn("sudo", wrapper.read_text())


class WorkspaceAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name) / "project"
        self.project.mkdir()
        self.patch = patch("coder_local.server.Controller.run")
        self.patch.start()
        self.store = JobStore(Path(self.temp.name) / "state")
        self.server, self.queue = create_server(Settings(), self.store, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.queue.stop()
        self.patch.stop()
        self.temp.cleanup()

    def post(self, path, payload, headers=None):
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())

    def test_select_existing_repository(self):
        code, body = self.post("/workspace/inspect", {"project": str(self.project)})
        self.assertEqual(code, 200)
        self.assertEqual(body["project"], str(self.project))

    def test_browser_terminal_runs_single_command(self):
        code, body = self.post("/workspace/command", {
            "project": str(self.project),
            "command": f'{sys.executable} -c "print(3+4)"',
        })
        self.assertEqual(code, 200)
        self.assertEqual(body["output"].strip(), "7")

    def test_cross_origin_request_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/workspace/command", {
                "project": str(self.project),
                "command": "git status",
            }, headers={"Origin": "https://attacker.invalid"})
        self.assertEqual(caught.exception.code, 403)

    def test_quotes_preserved_in_verification_command(self):
        code, body = self.post("/jobs", {
            "project": str(self.project),
            "objective": "Do the thing",
            "allowed_paths": ["src", "tests"],
            "test_command": 'python -c "print(1 + 1)"',
        })
        self.assertEqual(code, 202)
        stored = self.store.load(body["id"])
        self.assertEqual(stored.test_command, ["python", "-c", "print(1 + 1)"])
        self.queue.items.join()

    def test_assignment_missing_scope_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/jobs", {
                "project": str(self.project),
                "objective": "Change everything",
                "test_command": "python -m pytest",
            })
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
