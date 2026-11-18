"""Controller stage progression and verification tests using a stubbed agent.

The model is replaced, but the stage transitions, local test command, file
scope checking and persistence are exercised through the real controller.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coder_local.bridge import OllamaBridge
from coder_local.controller import Controller, Job, JobStore, Settings


class ControllerFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "src").mkdir()
        (self.project / "src" / "main.py").write_text("value = 0\n")
        self.store = JobStore(self.root / "state")
        self.settings = Settings(max_correction_rounds=2)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def fake_model_ready():
        with patch.object(OllamaBridge, "ensure_model"), patch.object(
            OllamaBridge, "start", return_value="http://127.0.0.1:9999/v1"
        ), patch.object(OllamaBridge, "stop"):
            yield

    def run_with_stub(self, stub):
        with patch.object(OllamaBridge, "ensure_model"), patch.object(
            OllamaBridge, "start", return_value="http://127.0.0.1:9999/v1"
        ), patch.object(OllamaBridge, "stop"), patch.object(
            Controller, "_ask_agent", side_effect=stub
        ):
            return Controller(self.settings, self.store).run("job")

    def test_valid_edit_and_test_pass(self):
        self.store.save("job", Job(
            project=str(self.project), objective="Set the value",
            allowed_paths=["src"], test_command=[
                "python", "-c",
                "from src.main import value; assert value == 3",
            ]
        ))

        def stub(_bridge, _id, job, _prompt, *, coding):
            if coding:
                (Path(job._execution_workspace) / "src" / "main.py").write_text("value = 3\n")
                return "Completed operation"
            return '["Set the value to 3"]'

        result = self.run_with_stub(stub)
        self.assertEqual(result.stage, "complete")
        self.assertEqual(result.verification["status"], "passed")
        self.assertTrue(result.complete)
        self.assertEqual(self.store.load("job").stage, "complete")
        self.assertIn("verify", [item["to"] for item in result.history])

    def test_repeated_failed_test_blocks_instead_of_claiming_success(self):
        self.store.save("job", Job(
            project=str(self.project), objective="Fail a check",
            test_command=["python", "-c", "raise SystemExit(1)"],
        ))
        def stub(_bridge, _id, job, _prompt, *, coding):
            if not coding:
                return '["Update the source file"]'
            path = Path(job._execution_workspace) / "src" / "main.py"
            path.write_text(f"value = {job.attempts + 1}\n")
            return "Changed source"

        job = self.store.load("job")
        job.allowed_paths = ["src"]
        self.store.save("job", job)
        result = self.run_with_stub(stub)
        self.assertEqual(result.stage, "blocked")
        self.assertFalse(result.complete)
        self.assertEqual(result.verification["status"], "failed")
        self.assertEqual(result.attempts, 2)

    def test_detects_edit_outside_authorized_scope(self):
        self.store.save("job", Job(
            project=str(self.project), objective="Restricted edit",
            allowed_paths=["src"],
            test_command=["python", "-c", "print('ok')"],
        ))

        def stub(_bridge, _id, job, _prompt, *, coding):
            if coding:
                (Path(job._execution_workspace) / "outside.txt").write_text("unauthorized")
                return "Completed operation"
            return '["Make the requested restricted edit"]'

        with self.assertRaisesRegex(PermissionError, "Unauthorized staging changes"):
            self.run_with_stub(stub)
        persisted = self.store.load("job")
        self.assertIn("Unauthorized staging changes", persisted.last_error)
        self.assertFalse((self.project / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
