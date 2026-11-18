"""Regression coverage for restricted execution and durable stage gates."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from coder_local.controller import Controller, Job, JobStore, Settings, _manifest
from coder_local.safety import StagedWorkspace, validate_scope


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.project = self.base / "project"
        self.project.mkdir()
        (self.project / "src").mkdir()
        (self.project / "src" / "main.py").write_text("answer = 1\n")
        self.state = self.base / "state"
        self.state.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_unrestricted_and_path_traversal_scopes(self):
        for paths in ([], ["../outside"], ["/etc"], ["."], [".git"]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                validate_scope(paths)

    def test_stage_promotes_only_authorized_files(self):
        with StagedWorkspace(self.project, self.state, ["src"]) as stage:
            file = stage.path / "src" / "main.py"
            file.write_text("answer = 2\n")
            self.assertEqual(stage.promote(), ["src/main.py"])
        self.assertEqual((self.project / "src/main.py").read_text(), "answer = 2\n")

    def test_unauthorized_stage_changes_never_reach_original(self):
        with StagedWorkspace(self.project, self.state, ["src"]) as stage:
            (stage.path / "other.py").write_text("unauthorized\n")
            with self.assertRaises(PermissionError):
                stage.promote()
        self.assertFalse((self.project / "other.py").exists())

    def test_symlink_escape_is_rejected(self):
        (self.project / "src" / "escape").symlink_to(
            self.state, target_is_directory=True
        )
        with self.assertRaisesRegex(ValueError, "Symbolic directory"):
            with StagedWorkspace(self.project, self.state, ["src"]):
                pass

    def test_conflicting_file_change_is_not_overwritten(self):
        with StagedWorkspace(self.project, self.state, ["src"]) as stage:
            (stage.path / "src/main.py").write_text("answer = 2\n")
            (self.project / "src/main.py").write_text("external edit\n")
            with self.assertRaisesRegex(RuntimeError, "changed externally"):
                stage.promote()
        self.assertEqual(
            (self.project / "src/main.py").read_text(), "external edit\n"
        )


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.project = self.base / "project"
        self.project.mkdir()
        (self.project / "src").mkdir()
        (self.project / "src/main.py").write_text("x=1\n")
        self.store = JobStore(self.base / "state")
        self.controller = Controller(Settings(), self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_test_command_is_blocked(self):
        job = Job(project=str(self.project), objective="Edit main", allowed_paths=["src"])
        self.store.save("missing-test", job)
        from openhands.sdk.llm.native_ollama import NativeOllamaClient
        with patch.object(NativeOllamaClient, "ensure_model", return_value=32768), patch.object(NativeOllamaClient, "show", return_value={"model_info": {}}):
            result = self.controller.run("missing-test")
        self.assertEqual(result.stage, "blocked")
        self.assertEqual(result.verification["status"], "unverified")
        self.assertFalse(result.complete)

    def test_unfinished_checkpoint_cannot_autoreplay(self):
        job = Job(
            project=str(self.project), objective="Edit main",
            allowed_paths=["src"], test_command=["python", "-m", "compileall", "-q", "src"],
            stage="implement",
        )
        self.store.save("inflight", job)
        self.controller._checkpoint("inflight", job, "step-01")
        result = self.controller.run("inflight", resume=True)
        self.assertEqual(result.stage, "blocked")
        self.assertIsNotNone(result.inflight)
        reconciled = self.controller.reconcile("inflight")
        self.assertEqual(reconciled.stage, "implement")
        self.assertIsNone(reconciled.inflight)

    def test_changed_original_cannot_reconcile(self):
        job = Job(
            project=str(self.project), objective="Edit main",
            allowed_paths=["src"], stage="implement",
        )
        self.store.save("changed", job)
        self.controller._checkpoint("changed", job, "step-01")
        (self.project / "src/main.py").write_text("x=9\n")
        with self.assertRaisesRegex(RuntimeError, "changed during interrupted"):
            self.controller.reconcile("changed")

    def test_plan_requires_bounded_structured_operations(self):
        self.assertEqual(
            len(self.controller._decode_plan('["Update module", "Check result"]')), 2
        )
        for value in ('not json', '[]', '[""]', json.dumps(["x"] * 31)):
            with self.subTest(value=value[:20]), self.assertRaises(
                (ValueError, json.JSONDecodeError)
            ):
                self.controller._decode_plan(value)


class SandboxBoundaryTests(unittest.TestCase):
    def test_worker_fails_closed_if_bubblewrap_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "project"
            project.mkdir()
            store = JobStore(root / "state")
            job = Job(project=str(project), objective="Edit", allowed_paths=["src"])
            controller = Controller(Settings(), store)
            model = MagicMock()
            with patch("coder_local.controller.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "bubblewrap"):
                    controller._ask_agent(
                        model, "missing-bwrap", job, "test", coding=True
                    )

    def test_sandbox_mounts_original_readonly_and_staging_writable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "project"
            project.mkdir()
            (project / "src").mkdir()
            state = root / "state"
            store = JobStore(state)
            controller = Controller(Settings(), store)
            bridge = MagicMock()
            bridge.url = "http://127.0.0.1:9010"
            job = Job(project=str(project), objective="Edit", allowed_paths=["src"])
            with StagedWorkspace(project, store.root, ["src"]) as stage:
                job._execution_workspace = str(stage.path)
                calls = []

                def fake_run(command, **_kwargs):
                    calls.append(command)
                    output_file = Path(command[-1])
                    output_file.write_text(
                        json.dumps({"finished": True, "response": "done"})
                    )
                    return MagicMock(returncode=0, stderr="", stdout="")

                with patch(
                    "coder_local.controller.shutil.which",
                    return_value="/usr/bin/bwrap",
                ), patch(
                    "coder_local.controller.subprocess.run", side_effect=fake_run
                ):
                    result = controller._ask_agent(
                        bridge, "sandbox-test", job, "work", coding=True
                    )
                self.assertEqual(result, "done")
                cmd = calls[0]
                self.assertIn("--ro-bind", cmd)
                self.assertEqual(cmd[cmd.index("--ro-bind")+1:cmd.index("--ro-bind")+3], ["/","/"])
                self.assertIn("--clearenv", cmd)
                self.assertIn(str(stage.path), cmd)
                self.assertNotIn("--bind", cmd[:cmd.index("--ro-bind")])
                self.assertEqual(cmd[-3], "coder_local.agent_worker")


if __name__ == "__main__":
    unittest.main()
