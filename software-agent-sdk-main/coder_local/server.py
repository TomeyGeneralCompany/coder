"""Loopback-only HTTP API and sequential job queue for Coder.

Run with: uv run python -m coder_local.server
This is intentionally a small local controller surface, not an external service.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

from coder_local.controller import Controller, Job, JobStore, Settings, parse_test_command
from coder_local.__main__ import DEFAULT_STATE_DIR
from coder_local.workspace import clone_repository, console_command, project_directory


class JobQueue:
    def __init__(self, settings: Settings, store: JobStore) -> None:
        self.settings = settings
        self.store = store
        self.items: queue.Queue[str | None] = queue.Queue()
        self.active: str | None = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._worker, name="coder-sequential-jobs", daemon=True
        )
        self.thread.start()

    def submit(self, job: Job) -> str:
        digest = hashlib.sha256(
            (job.project + "\n" + job.objective + "\n" + str(time.time_ns())).encode()
        ).hexdigest()[:20]
        self.store.save(digest, job)
        self.items.put(digest)
        return digest

    def resume(self, job_id: str) -> None:
        job = self.store.load(job_id)
        if job.stage == "complete":
            raise ValueError("Assignment has already finished")
        self.items.put(job_id)

    def _worker(self) -> None:
        while True:
            identifier = self.items.get()
            if identifier is None:
                self.items.task_done()
                return
            with self.lock:
                self.active = identifier
            try:
                controller = Controller(self.settings, self.store)
                controller.run(identifier, resume=True)
            except Exception:
                # The controller saves its last error and stage to disk.
                pass
            finally:
                with self.lock:
                    self.active = None
                self.items.task_done()

    def stop(self) -> None:
        self.items.put(None)
        self.thread.join(timeout=5)


def create_server(
    settings: Settings, store: JobStore, host: str = "127.0.0.1", port: int = 8765
) -> tuple[ThreadingHTTPServer, JobQueue]:
    if host != "127.0.0.1":
        raise ValueError("Coder HTTP API must bind to local loopback")
    settings.validate()
    jobs = JobQueue(settings, store)
    projects_root = store.root.parent / 'projects'

    class Handler(BaseHTTPRequestHandler):
        def _trusted_request(self) -> bool:
            """Reject browser cross-origin requests and DNS-rebinding hosts."""
            host = self.headers.get("Host", "")
            if host not in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }:
                return False
            origin = self.headers.get("Origin")
            if origin:
                parts = urlsplit(origin)
                if (
                    parts.scheme != "http"
                    or parts.hostname not in {"127.0.0.1", "localhost"}
                    or parts.port != self.server.server_port
                ):
                    return False
            return not (
                self.command == "POST"
                and self.headers.get("Sec-Fetch-Site") == "cross-site"
            )

        def _send(self, status: int, data: dict[str, Any]) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if not self._trusted_request():
                self._send(403, {"error": "Loopback browser origin required"})
                return
            if self.path == "/":
                body = Path(__file__).with_name("index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/jobs":
                records = []
                for path in sorted(store.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    try:
                        job = store.load(path.stem)
                        records.append({
                            "id": path.stem,
                            "objective": job.objective[:160],
                            "stage": job.stage,
                            "context_tokens": job.context_tokens,
                            "verification_status": job.verification.get("status"),
                        })
                    except (OSError, ValueError, TypeError):
                        continue
                self._send(200, {"jobs": records[:100]})
            elif self.path == "/health":
                with jobs.lock:
                    active = jobs.active
                self._send(200, {
                    "status": "ok",
                    "active_job": active,
                    "queued_count": jobs.items.qsize(),
                    "model": settings.model,
                })
            elif self.path.startswith("/jobs/"):
                identifier = self.path[len("/jobs/"):]
                try:
                    job = store.load(identifier)
                    self._send(200, {"id": identifier, **vars(job)})
                except (ValueError, FileNotFoundError):
                    self._send(404, {"error": "Assignment not found"})
            else:
                self._send(404, {"error": "Unknown route"})

        def do_POST(self) -> None:
            if not self._trusted_request():
                self._send(403, {"error": "Loopback browser origin required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 256_000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")
                if self.path == "/workspace/inspect":
                    directory = project_directory(data.get("project", ""))
                    self._send(200, {"project": str(directory), "exists": True})
                elif self.path == "/workspace/clone":
                    result = clone_repository(data.get("url", ""), data.get("name", ""), projects_root)
                    self._send(201, result)
                elif self.path == "/workspace/command":
                    result = console_command(data.get("project", ""), data.get("command", ""))
                    self._send(200, result)
                elif self.path == "/jobs":
                    project = Path(str(data["project"])).expanduser().resolve()
                    if not project.is_dir():
                        raise ValueError("Project directory does not exist")
                    objective = str(data["objective"]).strip()
                    if not objective:
                        raise ValueError("Objective is required")
                    test_command = data.get("test_command") or []
                    if isinstance(test_command, str):
                        test_command = parse_test_command(test_command)
                    allowed_paths = data.get("allowed_paths") or []
                    if not test_command or not isinstance(test_command, list) or not all(
                        isinstance(x, str) for x in test_command
                    ):
                        raise ValueError("test_command must be a list of strings")
                    if not allowed_paths or not isinstance(allowed_paths, list) or not all(
                        isinstance(x, str) for x in allowed_paths
                    ):
                        raise ValueError("allowed_paths must be a list of strings")
                    identifier = jobs.submit(Job(
                        project=str(project),
                        objective=objective,
                        test_command=test_command,
                        allowed_paths=allowed_paths,
                    ))
                    self._send(202, {"id": identifier, "status": "queued"})
                elif self.path.startswith("/jobs/") and self.path.endswith("/resume"):
                    identifier = self.path[len("/jobs/"):-len("/resume")]
                    jobs.resume(identifier)
                    self._send(202, {"id": identifier, "status": "queued"})
                else:
                    self._send(404, {"error": "Unknown route"})
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self._send(400, {"error": str(exc)})

        def log_message(self, *_args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler), jobs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local Coder HTTP job service")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    server, jobs = create_server(
        Settings(
            ollama_url=args.ollama, max_context=args.max_context
        ),
        JobStore(args.state_dir),
        port=args.port,
    )
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Coder dashboard at {url}")
    if args.open_browser:
        import webbrowser
        webbrowser.open(url, new=2)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        jobs.stop()


if __name__ == "__main__":
    main()
