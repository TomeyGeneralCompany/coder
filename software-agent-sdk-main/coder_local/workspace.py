"""Human-operated local repository workspace and bounded command console.

This console is NOT the autonomous agent's tool executor. Its commands are
explicitly submitted by a user from the loopback UI, without a shell, with
a short timeout and bounded output. Coder still stages AI edits under bwrap.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


MAX_OUTPUT_BYTES = 64_000
MAX_COMMAND_SECONDS = 45
MAX_CLONE_SECONDS = 180
MAX_ARGS = 80


def project_directory(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Select a local repository directory")
    try:
        project = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("The selected project directory does not exist") from exc
    if not project.is_dir():
        raise ValueError("The selected project path is not a directory")
    return project


def console_command(directory: str, text: str) -> dict:
    """Run a single explicitly requested command, without a shell or PTY."""
    project = project_directory(directory)
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        raise ValueError("Provide a command up to 4096 characters")
    try:
        arguments = shlex.split(text, posix=True)
    except ValueError as exc:
        raise ValueError(f"Command syntax: {exc}") from exc
    if not arguments or len(arguments) > MAX_ARGS:
        raise ValueError("Invalid number of command arguments")
    # This view is for deliberate single commands. Shell operators have no
    # special meaning in subprocess without shell=True, but rejecting them
    # avoids misleading users into assuming pipelines or redirects work.
    if any(arg in {"|", "||", "&&", ";", ">", ">>", "<", "&"} for arg in arguments):
        raise ValueError("Pipelines and shell operators are not supported in this console")
    try:
        result = subprocess.run(
            arguments, cwd=project, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=MAX_COMMAND_SECONDS,
            stdin=subprocess.DEVNULL, check=False,
        )
        output = result.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        return {
            "exit_code": result.returncode,
            "output": output,
            "truncated": len(result.stdout) > MAX_OUTPUT_BYTES,
            "command": arguments,
            "project": str(project),
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"")[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        return {"exit_code": None, "output": output, "timed_out": True,
                "command": arguments, "project": str(project)}
    except OSError as exc:
        raise ValueError(f"Cannot start command: {exc}") from exc


def clone_repository(url: str, name: str, root: str | Path) -> dict:
    """Clone a GitHub HTTPS repository into a new directory under Coder projects."""
    if not isinstance(url, str) or len(url) > 512:
        raise ValueError("Enter a GitHub HTTPS repository URL")
    parts = urlsplit(url.strip())
    if (
        parts.scheme != "https" or parts.hostname != "github.com"
        or parts.username or parts.password or parts.query or parts.fragment
        or parts.port is not None
    ):
        raise ValueError("Only https://github.com/owner/repository URLs are supported")
    path = parts.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path)
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        raise ValueError("Expected https://github.com/owner/repository")
    default = path.split("/")[-1]
    folder = name.strip() if isinstance(name, str) else ""
    folder = folder or default
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", folder):
        raise ValueError("Project directory name must be simple and safe")
    projects = Path(root).expanduser().resolve()
    projects.mkdir(parents=True, exist_ok=True)
    destination = projects / folder
    if destination.exists() or destination.is_symlink():
        raise ValueError("Project directory already exists; choose another name")
    try:
        result = subprocess.run(
            ["git", "-c", "credential.helper=", "clone",
             "--", f"https://github.com/{path}.git", str(destination)],
            cwd=projects,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=MAX_CLONE_SECONDS, check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Git clone timed out; inspect the workspace folder") from exc
    except OSError as exc:
        raise ValueError(f"Git cannot be started: {exc}") from exc
    output = result.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if result.returncode:
        raise ValueError(f"Git clone failed: {output[:2000]}")
    return {"project": str(destination), "output": output, "repository": url}
