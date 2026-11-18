"""Scope validation and change promotion for Coder development work.

The agent edits a disposable working copy. Only regular files within explicit
allowed paths can be promoted to the source project, after checking that the
original files have not changed since staging. This prevents ordinary tool edits
from propagating outside the authorized project paths. It is NOT an OS sandbox:
arbitrary terminal commands still require independent process isolation.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

EXCLUDE = {".git", ".venv", "node_modules", "__pycache__", ".cache", ".next"}


def validate_scope(paths: list[str]) -> tuple[str, ...]:
    if not paths:
        raise ValueError("Explicit allowed paths required for coding tasks")
    normalized: list[str] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Invalid allowed path")
        p = PurePosixPath(raw)
        if p.is_absolute() or ".." in p.parts or raw.startswith("~"):
            raise ValueError(f"Unsafe allowed path: {raw!r}")
        cleaned = p.as_posix().removeprefix("./").rstrip("/")
        if not cleaned or cleaned == ".":
            raise ValueError("Whole-project scope is not implicit")
        if any(part in EXCLUDE or part == ".git" for part in p.parts):
            raise ValueError(f"Internal/third-party path cannot be authorized: {raw}")
        normalized.append(cleaned)
    return tuple(sorted(set(normalized)))


def permitted(relative: str, allowed: tuple[str, ...]) -> bool:
    return any(relative == p or relative.startswith(p + "/") for p in allowed)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE)
        for name in files:
            p = Path(current) / name
            relative = p.relative_to(root).as_posix()
            if p.is_symlink():
                raise ValueError(f"Symbolic link in project: {relative}")
            if p.is_file():
                result[relative] = _digest(p.read_bytes())
    return result


class StagedWorkspace:
    def __init__(self, project: Path, state_root: Path, allowed: list[str]) -> None:
        self.project = project.resolve(strict=True)
        self.allowed = validate_scope(allowed)
        self.state_root = state_root.resolve()
        if self.state_root == self.project or self.state_root.is_relative_to(self.project):
            raise ValueError("State storage must be outside the project")
        if self.project.is_relative_to(self.state_root):
            raise ValueError("Project must not be inside state storage")
        self.temp: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None
        self.baseline: dict[str, str] = {}

    def __enter__(self) -> "StagedWorkspace":
        self.baseline = _manifest(self.project)
        self.temp = tempfile.TemporaryDirectory(prefix="coder-work-")
        self.path = Path(self.temp.name) / "project"
        try:
            shutil.copytree(
                self.project, self.path,
                symlinks=False,
                ignore=lambda _p, names: EXCLUDE.intersection(names),
            )
        except Exception:
            self.temp.cleanup()
            raise
        return self

    def changes(self) -> dict[str, str | None]:
        if self.path is None:
            raise RuntimeError("Staging workspace not initialized")
        modified = _manifest(self.path)
        differences = {
            name: modified.get(name)
            for name in sorted(set(self.baseline) | set(modified))
            if self.baseline.get(name) != modified.get(name)
        }
        out_of_scope = [name for name in differences if not permitted(name, self.allowed)]
        if out_of_scope:
            raise PermissionError(
                "Unauthorized staging changes rejected: " + ", ".join(out_of_scope[:10])
            )
        return differences

    def promote(self) -> list[str]:
        """Promote authorized changes after ensuring originals remain unchanged."""
        if self.path is None:
            raise RuntimeError("Staging workspace not initialized")
        changes = self.changes()
        current = _manifest(self.project)
        for name in changes:
            if current.get(name) != self.baseline.get(name):
                raise RuntimeError(f"Project file changed externally: {name}")
        for name in changes:
            dest = self.project / name
            src = self.path / name
            if not dest.resolve().is_relative_to(self.project):
                raise PermissionError(f"Path escaped project: {name}")
            if changes[name] is None:
                dest.unlink(missing_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_path = tempfile.mkstemp(
                prefix=".coder-stage-", dir=dest.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as target:
                    target.write(src.read_bytes())
                    target.flush()
                    os.fsync(target.fileno())
                os.chmod(temp_path, src.stat().st_mode & 0o777)
                os.replace(temp_path, dest)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        return list(changes)

    def __exit__(self, *_args: object) -> None:
        if self.temp is not None:
            self.temp.cleanup()
