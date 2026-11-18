"""Sequential, persistent stage-gated development using the OpenHands SDK.

All model calls go through the loopback native-Ollama bridge. Work is performed
against the user-selected project, never against Coder's own source by default.
"""
from __future__ import annotations

import fcntl
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

from coder_local.bridge import OllamaBridge


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
        self, bridge: OllamaBridge, job_id: str, job: Job,
        instruction: str, *, coding: bool,
    ) -> str:
        # All LLM requests go to our own loopback adapter, never a cloud provider.
        os.environ["ALLOW_SHORT_CONTEXT_WINDOWS"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        from openhands.sdk import Agent, Conversation, LLM, Tool
        from openhands.sdk.context.condenser import LLMSummarizingCondenser
        from openhands.sdk.conversation import (
            ConversationExecutionStatus,
            get_agent_final_response,
        )
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.terminal import TerminalTool

        ctx = self.allocator.select(instruction, needs_code=coding)
        job.context_tokens = ctx
        self.store.save(job_id, job)
        bridge.configure(
            context_tokens=ctx,
            output_tokens=min(self.settings.output_tokens, ctx // 3),
        )
        llm = LLM(
            model=f"openai/{self.settings.model}",
            base_url=bridge.url,
            api_key="local-only",
            api_mode="chat",
            max_input_tokens=ctx,
            max_output_tokens=min(self.settings.output_tokens, ctx // 3),
            max_message_chars=4500,
            stream=False,
            num_retries=1,
            timeout=self.settings.timeout_seconds,
        )
        tools = (
            [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)]
            if coding else []
        )
        agent = Agent(
            llm=llm,
            tools=tools,
            tool_concurrency_limit=1,
            condenser=LLMSummarizingCondenser(
                llm=llm.model_copy(update={"usage_id": "condenser"}),
                max_size=28,
                keep_first=2,
                max_tokens=max(2048, ctx - self.settings.output_tokens - 1024),
            ),
            system_prompt=(
                "You are a local software engineer. Follow the provided task scope. "
                "Complete one bounded operation at a time. Do not use hosted AI services. "
                "Do not modify files outside the authorized project and allowed paths. "
                "Report actual results; never invent executed tests."
            ),
        )
        (self.store.root / "conversations").mkdir(parents=True, exist_ok=True)
        conversation = Conversation(
            agent=agent,
            workspace=str(self._project(job)),
            persistence_dir=str(self.store.root / "conversations"),
            max_iteration_per_run=self.settings.max_agent_iterations if coding else 3,
        )
        conversation.send_message(instruction)
        conversation.run()
        if conversation.state.execution_status != ConversationExecutionStatus.FINISHED:
            raise RuntimeError(
                "Agent did not finish its stage: "
                + str(conversation.state.execution_status)
            )
        response = get_agent_final_response(conversation.state.events)
        if response is None:
            raise RuntimeError("Agent finished without a recorded final response")
        return str(response)

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

    def run(self, job_id: str, *, resume: bool = False) -> Job:
        with self.store.execution_lock():
            return self._run_locked(job_id, resume=resume)

    def _run_locked(self, job_id: str, *, resume: bool = False) -> Job:
        job = self.store.load(job_id)
        root = self._project(job)
        if job.stage == "complete":
            return job
        if job.stage == "blocked" and not resume:
            return job

        bridge = OllamaBridge(
            model=self.settings.model,
            ollama_url=self.settings.ollama_url,
            timeout_seconds=self.settings.timeout_seconds,
        )
        bridge.ensure_model()
        if bridge.supported_context is not None:
            self.allocator.settings.max_context = min(
                self.allocator.settings.max_context, bridge.supported_context
            )
            if self.allocator.settings.max_context < self.settings.context_profiles[0]:
                raise RuntimeError("Installed Qwen context is below minimum profile")
            self.allocator.current = min(
                self.allocator.current, self.allocator.settings.max_context
            )
        bridge.url = bridge.start()
        try:
            while job.stage not in {"complete", "blocked"}:
                stage = job.stage
                if stage == "intake":
                    if not job.objective.strip():
                        raise ValueError("Assignment objective is empty")
                    self._transition(job_id, job, "inspect", "Assignment recorded")
                elif stage == "inspect":
                    job.inspection = _inspection(root, job.objective)
                    self._transition(job_id, job, "plan", "Source inspected")
                elif stage == "plan":
                    prompt = (
                        "Prepare a concise implementation plan for the authorized "
                        "assignment. Do not edit files.\n"
                        f"OBJECTIVE:\n{job.objective}\nSCOPE:\n{job.allowed_paths}\n"
                        f"REPOSITORY:\n{job.inspection[:12000]}"
                    )
                    job.planned_work = self._ask_agent(
                        bridge, job_id, job, prompt, coding=False
                    )[:12000]
                    self._transition(job_id, job, "implement", "Plan recorded")
                elif stage in {"implement", "correct"}:
                    before = _snapshot_files(root) if job.allowed_paths else {}
                    prompt = (
                        f"Implement this authorized task in {root}.\n"
                        f"OBJECTIVE:\n{job.objective}\n"
                        f"ALLOWED PATHS:\n{job.allowed_paths or ['project workspace']}\n"
                        f"PLAN:\n{job.planned_work[:6500]}\n"
                        f"RELEVANT SOURCE:\n{job.inspection[:6500]}\n"
                        f"PREVIOUS TEST ERROR:\n{job.last_error[-4500:]}\n"
                        "Make the changes using the supplied tools. "
                        "Do not commit, push, merge, or modify unrelated files."
                    )
                    try:
                        self._ask_agent(bridge, job_id, job, prompt, coding=True)
                    finally:
                        # Even a failed agent call may have modified project files.
                        self._check_scope(job, before)
                    self._transition(job_id, job, "verify", "Coding operation finished")
                elif stage == "verify":
                    if not job.test_command:
                        job.verification = {
                            "status": "unverified",
                            "reason": "No project test command supplied",
                        }
                        self._transition(job_id, job, "complete", "Changes unverified")
                        continue
                    result = _run_command(
                        job.test_command, root, timeout=self.settings.timeout_seconds
                    )
                    logs_dir = self.store.root / "logs"
                    logs_dir.mkdir(exist_ok=True)
                    (logs_dir / f"{job_id}-verify.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8"
                    )
                    job.verification = {
                        "status": "passed" if result["exit_code"] == 0 else "failed",
                        "exit_code": result["exit_code"],
                        "command": job.test_command,
                        "log": f"logs/{job_id}-verify.json",
                    }
                    if result["exit_code"] == 0:
                        self._transition(job_id, job, "complete", "Tests passed")
                    elif job.attempts >= self.settings.max_correction_rounds:
                        self._transition(job_id, job, "blocked", "Correction limit reached")
                    else:
                        job.attempts += 1
                        job.last_error = (
                            result["stderr"][-3500:] + "\n" + result["stdout"][-3000:]
                        )
                        self._transition(job_id, job, "correct", "Tests failed")
                else:
                    self._transition(job_id, job, "blocked", "Unknown stage")
            job.complete = job.stage == "complete"
            self.store.save(job_id, job)
            return job
        except Exception as exc:
            job.last_error = str(exc)
            self.store.save(job_id, job)
            raise
        finally:
            bridge.stop()


def parse_test_command(command: str) -> list[str]:
    return shlex.split(command)
