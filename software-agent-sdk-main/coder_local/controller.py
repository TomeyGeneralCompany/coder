"""Sequential, persistent stage-gated development using the OpenHands SDK.

All model calls go through the loopback native-Ollama bridge. Work is performed
against the user-selected project, never against Coder's own source by default.
"""
from __future__ import annotations

import fcntl
import shutil
import sys
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openhands.sdk.llm.native_ollama import NativeOllamaClient
from coder_local.safety import StagedWorkspace, validate_scope, _manifest


STAGES = ("intake", "inspect", "plan", "implement", "verify", "correct", "complete", "blocked")
SKIP_NAMES = {
    ".git", ".venv", "node_modules", "dist", "build", "__pycache__",
    ".next", ".cache", ".env", ".env.local", "uv.lock", "package-lock.json",
}


@dataclass
class Settings:
    model: str = "qwen3:4b"
    ollama_url: str = "http://127.0.0.1:11434"
    context_profiles: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536)
    max_context: int = 32768
    output_tokens: int = 1024
    max_agent_iterations: int = 18
    max_correction_rounds: int = 3
    min_free_memory_mb: int = 900
    timeout_seconds: int = 1200

    def validate(self) -> None:
        if self.model != "qwen3:4b":
            raise ValueError("Coder uses only the local qwen3:4b model")
        NativeOllamaClient(self.ollama_url, model=self.model)
        if not self.context_profiles or tuple(sorted(set(self.context_profiles))) != self.context_profiles:
            raise ValueError("context_profiles must be sorted, distinct, and nonempty")
        if self.max_context < self.context_profiles[0]:
            raise ValueError("max_context is too small")
        if self.output_tokens < 128:
            raise ValueError("output_tokens is too small")
        if self.max_agent_iterations < 1 or self.max_correction_rounds < 0:
            raise ValueError("Invalid work iteration limits")


@dataclass
class Job:
    project: str
    objective: str
    allowed_paths: list[str] = field(default_factory=list)
    test_command: list[str] = field(default_factory=list)
    stage: str = "intake"
    planned_work: str = ""
    inspection: str = ""
    last_error: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    context_tokens: int = 8192
    complete: bool = False
    operations: list[dict[str, Any]] = field(default_factory=list)
    operation_index: int = 0
    inflight: dict[str, Any] | None = None
    verified_operations: list[str] = field(default_factory=list)


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", job_id):
            raise ValueError("Invalid job id")
        return self.root / (job_id + ".json")

    def save(self, job_id: str, job: Job) -> None:
        path = self.path(job_id)
        descriptor, temp_path = tempfile.mkstemp(prefix=".coder-", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(asdict(job), stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def load(self, job_id: str) -> Job:
        return Job(**json.loads(self.path(job_id).read_text(encoding="utf-8")))

    @contextmanager
    def execution_lock(self):
        """Serializes active Coder jobs across CLI and HTTP service processes."""
        lock_path = self.root / ".execution.lock"
        with lock_path.open("a+") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _run_command(arguments: list[str], cwd: Path, timeout: int = 300) -> dict[str, Any]:
    try:
        result = subprocess.run(
            arguments, cwd=cwd, text=True, capture_output=True,
            timeout=timeout, check=False,
        )
        return {
            "command": arguments,
            "exit_code": result.returncode,
            "stdout": result.stdout[-300_000:],
            "stderr": result.stderr[-100_000:],
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"command": arguments, "exit_code": 127, "stdout": "", "stderr": str(exc)}


def _list_source(root: Path, limit: int = 250) -> list[str]:
    names: list[str] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_NAMES and not d.startswith("."))
        for filename in sorted(files):
            if filename in SKIP_NAMES or filename.startswith("."):
                continue
            rel = (Path(current) / filename).relative_to(root)
            if len(names) >= limit:
                return names
            names.append(rel.as_posix())
    return names


def _inspection(root: Path, objective: str) -> str:
    paths = _list_source(root)
    words = set(re.findall(r"[a-z0-9_]{4,}", objective.lower()))
    ranked = sorted(
        paths, key=lambda p: (
            -sum(1 for word in words if word in p.lower()), p
        )
    )
    fragments = ["SOURCE PATHS:", *paths[:160], "\nRELEVANT EXCERPTS:"]
    for relative in ranked[:5]:
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or target.stat().st_size > 150_000:
            continue
        try:
            excerpt = target.read_text(encoding="utf-8")[:2400]
        except (OSError, UnicodeError):
            continue
        fragments.append(f"\n--- {relative} ---\n{excerpt}")
    return "\n".join(fragments)[:20000]


def _snapshot_files(root: Path) -> dict[str, str]:
    """Capture file contents before a coding stage, including already-dirty files."""
    result: dict[str, str] = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {
            ".git", ".venv", "node_modules", "__pycache__", "dist", "build",
        }]
        for name in files:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            try:
                if path.is_symlink():
                    data = os.readlink(path).encode()
                else:
                    data = path.read_bytes()
                result[relative] = hashlib.sha256(data).hexdigest()
            except OSError:
                continue
    return result


def _free_memory_mb() -> int | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


class ContextAllocator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.current = min(8192, settings.max_context)

    def select(self, text: str, *, needs_code: bool) -> int:
        # Conservative estimate: prompt text, OpenHands system text, tool schemas,
        # and a separate allowance for generation.
        estimated = len(text.encode("utf-8")) // 3 + 1600
        if needs_code:
            estimated += 1800
        estimated += self.settings.output_tokens
        eligible = [
            n for n in self.settings.context_profiles if n <= self.settings.max_context
        ]
        selected = next((n for n in eligible if n >= estimated), eligible[-1])
        free_mb = _free_memory_mb()
        if free_mb is not None and free_mb < self.settings.min_free_memory_mb:
            # Do not grow context while available memory is below the floor.
            selected = min(selected, self.current)
        self.current = selected
        return selected


class Controller:
    def __init__(self, settings: Settings, store: JobStore) -> None:
        settings.validate()
        self.settings = settings
        self.store = store
        self.allocator = ContextAllocator(settings)

    @staticmethod
    def _project(job: Job) -> Path:
        root = Path(job.project).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Project directory does not exist: {root}")
        return root

    def _checkpoint(self, job_id: str, job: Job, action: str) -> None:
        """Write intent before any model operation; never blindly replay it."""
        job.inflight = {
            "action": action,
            "stage": job.stage,
            "operation_index": job.operation_index,
            "baseline": _manifest(self._project(job)),
            "time": int(time.time()),
        }
        self.store.save(job_id, job)

    def _clear_checkpoint(self, job_id: str, job: Job) -> None:
        job.inflight = None
        self.store.save(job_id, job)

    @staticmethod
    def _decode_plan(response: str) -> list[dict[str, Any]]:
        """Require a bounded, explicit task list; malformed model output blocks."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if (
                len(lines) < 3
                or lines[0].strip() not in {"```", "```json"}
                or lines[-1].strip() != "```"
            ):
                raise ValueError("Malformed fenced JSON plan")
            text = "\n".join(lines[1:-1]).strip()
        payload = json.loads(text)
        if not isinstance(payload, list) or not 1 <= len(payload) <= 30:
            raise ValueError("Plan must contain 1 to 30 operations")
        tasks = []
        for index, item in enumerate(payload):
            if not isinstance(item, str) or not 1 <= len(item.strip()) <= 600:
                raise ValueError("Each planned operation must be a brief string")
            tasks.append({"id": f"step-{index+1:02d}", "instruction": item.strip(), "status": "pending"})
        return tasks

    def _transition(self, job_id: str, job: Job, new_stage: str, note: str) -> None:
        if new_stage not in STAGES:
            raise ValueError(f"Unknown stage: {new_stage}")
        job.history.append({
            "time": int(time.time()), "from": job.stage,
            "to": new_stage, "note": note[:1500],
        })
        job.stage = new_stage
        self.store.save(job_id, job)

    def _ask_agent(
        self, bridge: NativeOllamaClient, job_id: str, job: Job,
        instruction: str, *, coding: bool,
    ) -> str:
        """Execute OpenHands inside a fail-closed read-only-host sandbox.

        The native Ollama bridge remains outside the sandbox on loopback.
        Bubblewrap grants write access only to the staged workspace, SDK
        conversation persistence and the result folder. The real project and
        the rest of the host filesystem remain read-only to the agent process.
        """
        binary = shutil.which("bwrap")
        if binary is None:
            raise RuntimeError(
                "bubblewrap (bwrap) is required for scoped Coder execution"
            )
        ctx = self.allocator.select(instruction, needs_code=coding)
        job.context_tokens = ctx
        self.store.save(job_id, job)
        output_tokens = min(self.settings.output_tokens, ctx // 3)

        root = Path(__file__).resolve().parents[1]
        if coding:
            workspace = Path(
                getattr(job, "_execution_workspace", self._project(job))
            ).resolve(strict=True)
        else:
            # Planning needs no project write access and uses a small local
            # workspace mounted beneath the state directory.
            workspace = self.store.root / "planning_workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            workspace = workspace.resolve(strict=True)
        conversations = self.store.root / "conversations"
        outputs = self.store.root / "agent_outputs"
        conversations.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        fd, request_name = tempfile.mkstemp(
            prefix=f"{job_id}-", suffix=".input.json", dir=self.store.root
        )
        request_path = Path(request_name)
        result_path = outputs / (request_path.name + ".result.json")
        request = {
            "model": self.settings.model,
            "ollama_url": bridge.url,
            "context_tokens": ctx,
            "output_tokens": output_tokens,
            "timeout": self.settings.timeout_seconds,
            "coding": coding,
            "max_iterations": self.settings.max_agent_iterations,
            "workspace": str(workspace),
            "persistence_dir": str(conversations),
            "instruction": instruction,
        }
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(request, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # Reads from the host are possible, writes are isolated by a
            # read-only root. Only the declared directories are writable.
            command = [
                binary, "--die-with-parent", "--new-session",
                "--unshare-user", "--unshare-pid", "--clearenv",
                "--ro-bind", "/", "/",
                "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            ]
            # A temporary state directory may itself live under /tmp. Restore
            # its mount-point parents inside the otherwise-private /tmp.
            if self.store.root.is_relative_to(Path("/tmp")):
                for parent in reversed(self.store.root.parents):
                    if parent != Path("/tmp") and parent.is_relative_to(Path("/tmp")):
                        command.extend(["--dir", str(parent)])
                command.extend(["--dir", str(self.store.root)])
            command.extend([
                "--ro-bind", str(self.store.root), str(self.store.root),
                "--bind", str(conversations), str(conversations),
                "--bind", str(outputs), str(outputs),
                "--setenv", "HOME", "/tmp",
                "--setenv", "PATH", os.environ.get("PATH", "/usr/bin:/bin"),
                "--setenv", "LANG", "C.UTF-8",
                "--setenv", "XDG_CACHE_HOME", "/tmp/.cache",
                "--setenv", "PYTHONPATH", str(root),
                "--setenv", "DO_NOT_TRACK", "1",
                "--setenv", "CODER_LOCAL_ONLY", "1",
                "--setenv", "ALLOW_SHORT_CONTEXT_WINDOWS", "1",
                "--chdir", str(root),
            ])
            if not workspace.is_relative_to(self.store.root):
                raise PermissionError("Agent workspace must be within local staging")
            command.extend(["--bind", str(workspace), str(workspace)])
            command.extend([
                "--", sys.executable, "-m", "coder_local.agent_worker",
                str(request_path), str(result_path),
            ])
            result = subprocess.run(
                command, cwd=root, capture_output=True, text=True,
                timeout=self.settings.timeout_seconds
                * max(1, self.settings.max_agent_iterations),
                check=False,
                start_new_session=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Isolated OpenHands worker failed: "
                    + (result.stderr or result.stdout)[-4000:]
                )
            if not result_path.exists():
                raise RuntimeError("Isolated worker returned no result artifact")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("finished") is not True:
                raise RuntimeError("Isolated worker did not finish its operation")
            return str(payload["response"])
        finally:
            request_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)

    def _check_scope(self, job: Job, before: dict[str, str]) -> None:
        if not job.allowed_paths:
            return
        after = _snapshot_files(self._project(job))
        for path in sorted(set(before) | set(after)):
            if before.get(path) == after.get(path):
                continue
            if not any(
                path == allowed.rstrip("/")
                or path.startswith(allowed.rstrip("/") + "/")
                for allowed in job.allowed_paths
            ):
                raise RuntimeError(f"Change outside authorized paths: {path}")

    def reconcile(self, job_id: str) -> Job:
        """Explicit restart reconciliation; never automatically replay in-flight work.

        The caller must inspect the previous operation before requesting reconciliation.
        Automatic recovery is possible only when the project fingerprint is unchanged.
        """
        with self.store.execution_lock():
            job = self.store.load(job_id)
            if job.inflight is None:
                raise ValueError("No interrupted operation to reconcile")
            if _manifest(self._project(job)) != job.inflight["baseline"]:
                raise RuntimeError(
                    "Project changed during interrupted operation; manual review required"
                )
            original_stage = job.inflight["stage"]
            if original_stage not in {"plan", "implement", "correct", "verify"}:
                raise ValueError("Interrupted stage is not safe to reconcile")
            job.inflight = None
            job.last_error = ""
            job.stage = original_stage
            self.store.save(job_id, job)
            return job

    def run(self, job_id: str, *, resume: bool = False) -> Job:
        with self.store.execution_lock():
            return self._run_locked(job_id, resume=resume)

    def _run_locked(self, job_id: str, *, resume: bool = False) -> Job:
        job = self.store.load(job_id)
        root = self._project(job)
        if job.stage == "complete":
            if not (
                job.verification.get("status") == "passed"
                and bool(job.operations)
                and all(task.get("status") == "verified" for task in job.operations)
            ):
                job.complete = False
                self._transition(
                    job_id, job, "blocked",
                    "Previous completion lacks verified operation evidence",
                )
            return job
        if job.stage == "blocked":
            return job
        if job.inflight is not None:
            job.last_error = (
                "Interrupted in-flight operation was not replayed. "
                "Inspect project state and explicitly reconcile before continuing."
            )
            self._transition(job_id, job, "blocked", "Unreconciled checkpoint")
            return job

        bridge = NativeOllamaClient(
            url=self.settings.ollama_url,
            model=self.settings.model,
            timeout=self.settings.timeout_seconds,
        )
        supported_context = bridge.ensure_model()
        if supported_context is not None:
            self.allocator.settings.max_context = min(
                self.allocator.settings.max_context, supported_context
            )
            if self.allocator.settings.max_context < self.settings.context_profiles[0]:
                raise RuntimeError("Installed Qwen context is below minimum profile")
            self.allocator.current = min(
                self.allocator.current, self.allocator.settings.max_context
            )
        try:
            while job.stage not in {"complete", "blocked"}:
                stage = job.stage
                if stage == "intake":
                    if not job.objective.strip():
                        raise ValueError("Assignment objective is empty")
                    validate_scope(job.allowed_paths)
                    if not job.test_command:
                        job.verification = {
                            "status": "unverified", "reason": "Verification command required"
                        }
                        self._transition(job_id, job, "blocked", "Missing verification command")
                        continue
                    self._transition(job_id, job, "inspect", "Assignment recorded")
                elif stage == "inspect":
                    job.inspection = _inspection(root, job.objective)
                    self._transition(job_id, job, "plan", "Source inspected")
                elif stage == "plan":
                    prompt = (
                        "Break the authorized assignment into 1-30 small, ordered "
                        "development operations. Return ONLY a JSON array of concise "
                        "instruction strings. Each operation should be independently "
                        "editable and testable. Do not edit files.\n"
                        f"OBJECTIVE:\n{job.objective}\nSCOPE:\n{job.allowed_paths}\n"
                        f"REPOSITORY:\n{job.inspection[:12000]}"
                    )
                    self._checkpoint(job_id, job, "plan")
                    response = self._ask_agent(bridge, job_id, job, prompt, coding=False)
                    job.operations = self._decode_plan(response)
                    job.planned_work = response[:12000]
                    self._clear_checkpoint(job_id, job)
                    self._transition(job_id, job, "implement", "Bounded work plan recorded")
                elif stage in {"implement", "correct"}:
                    if job.operation_index >= len(job.operations):
                        self._transition(job_id, job, "blocked", "No pending operation")
                        continue
                    operation = job.operations[job.operation_index]
                    self._checkpoint(job_id, job, operation["id"])
                    with StagedWorkspace(root, self.store.root, job.allowed_paths) as staged:
                        job._execution_workspace = str(staged.path)
                        prompt = (
                            f"Modify only the staged project at {staged.path}.\n"
                            f"OVERALL OBJECTIVE:\n{job.objective[:1500]}\n"
                            f"CURRENT OPERATION {operation['id']}:\n"
                            f"{operation['instruction']}\n"
                            f"AUTHORIZED PATHS:\n{job.allowed_paths}\n"
                            f"RELEVANT SOURCE:\n{job.inspection[:4500]}\n"
                            f"PREVIOUS TEST ERROR:\n{job.last_error[-3000:]}\n"
                            "Make only this operation's changes. Do not commit, push, "
                            "merge or edit any other file."
                        )
                        try:
                            self._ask_agent(bridge, job_id, job, prompt, coding=True)
                            changes = staged.changes()
                            if not changes:
                                raise RuntimeError("Operation produced no source changes")
                            promoted = staged.promote()
                        finally:
                            del job._execution_workspace
                    operation["modified_files"] = promoted
                    operation["status"] = "implemented"
                    self._clear_checkpoint(job_id, job)
                    self._transition(job_id, job, "verify", "Staged edits promoted within scope")
                elif stage == "verify":
                    if not job.test_command:
                        job.verification = {
                            "status": "unverified", "reason": "No verification command"
                        }
                        self._transition(job_id, job, "blocked", "Unverified assignment")
                        continue
                    self._checkpoint(job_id, job, "verify")
                    result = _run_command(
                        job.test_command, root, timeout=self.settings.timeout_seconds
                    )
                    logs_dir = self.store.root / "logs"
                    logs_dir.mkdir(exist_ok=True)
                    log_file = logs_dir / (
                        f"{job_id}-step-{job.operation_index:02d}-"
                        f"attempt-{job.attempts:02d}.json"
                    )
                    log_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    job.verification = {
                        "status": "passed" if result["exit_code"] == 0 else "failed",
                        "exit_code": result["exit_code"],
                        "command": job.test_command,
                        "log": str(log_file.relative_to(self.store.root)),
                        "operation_index": job.operation_index,
                    }
                    self._clear_checkpoint(job_id, job)
                    if result["exit_code"] == 0:
                        current = job.operations[job.operation_index]
                        current["status"] = "verified"
                        job.verified_operations.append(current["id"])
                        job.operation_index += 1
                        job.attempts = 0
                        job.last_error = ""
                        if job.operation_index == len(job.operations):
                            self._transition(job_id, job, "complete", "All operations verified")
                        else:
                            upcoming = job.operations[job.operation_index]["instruction"]
                            job.inspection = _inspection(
                                root, job.objective + "\n" + upcoming
                            )
                            self._transition(job_id, job, "implement", "Next bounded operation")
                    elif job.attempts >= self.settings.max_correction_rounds:
                        self._transition(job_id, job, "blocked", "Correction limit reached")
                    else:
                        job.attempts += 1
                        job.last_error = (
                            result["stderr"][-3500:] + "\n" + result["stdout"][-3000:]
                        )
                        self._transition(job_id, job, "correct", "Verification failed")
                else:
                    self._transition(job_id, job, "blocked", "Unknown stage")
            job.complete = (job.stage == "complete" and bool(job.operations)
                            and all(x["status"] == "verified" for x in job.operations)
                            and job.verification.get("status") == "passed")
            self.store.save(job_id, job)
            return job
        except Exception as exc:
            job.last_error = str(exc)
            self.store.save(job_id, job)
            raise
        finally:
            # Direct Ollama requests require no bridge server or resident thread.
            pass


def parse_test_command(command: str) -> list[str]:
    return shlex.split(command)
