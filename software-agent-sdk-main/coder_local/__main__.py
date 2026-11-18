"""Run the locally hosted Qwen/OpenHands coding controller.

Examples:
  uv run python -m coder_local run --project ~/site --task "Fix the menu" \
      --test "npm test"
  uv run python -m coder_local resume --id website-menu
  uv run python -m coder_local status --id website-menu
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coder_local.controller import Controller, Job, JobStore, Settings, parse_test_command


DEFAULT_STATE_DIR = Path.home() / ".local" / "share" / "coder" / "jobs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Local-only Qwen software engineer")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Create and execute a development assignment")
    run.add_argument("--id", help="Optional stable assignment id")
    run.add_argument("--project", required=True, type=Path)
    run.add_argument("--task", required=True)
    run.add_argument("--test", default="", help="Project test command (no shell)")
    run.add_argument("--allow", action="append", default=[], help="Allowed relative path")

    resume = sub.add_parser("resume", help="Resume an interrupted assignment")
    resume.add_argument("--id", required=True)

    reconcile = sub.add_parser("reconcile", help="Explicitly reconcile an interrupted stage")
    reconcile.add_argument("--id", required=True)

    status = sub.add_parser("status", help="Read persisted state without running the model")
    status.add_argument("--id", required=True)

    args = parser.parse_args()
    store = JobStore(args.state_dir)
    if args.command == "status":
        print(json.dumps(vars(store.load(args.id)), indent=2))
        return 0

    settings = Settings(
        ollama_url=args.ollama, max_context=args.max_context
    )
    controller = Controller(settings, store)
    if args.command == "reconcile":
        try:
            job = controller.reconcile(args.id)
        except (ValueError, RuntimeError, OSError) as exc:
            print(json.dumps({"id": args.id, "error": str(exc)}))
            return 2
        print(json.dumps({"id": args.id, "stage": job.stage, "reconciled": True}))
        return 0

    if args.command == "run":
        project = args.project.expanduser().resolve()
        identifier = args.id or hashlib.sha256(
            (str(project) + "\n" + args.task).encode()
        ).hexdigest()[:16]
        if store.path(identifier).exists():
            parser.error(
                f"Assignment {identifier} already exists. Use 'resume --id {identifier}'."
            )
        job = Job(
            project=str(project),
            objective=args.task,
            allowed_paths=args.allow,
            test_command=parse_test_command(args.test) if args.test else [],
        )
        store.save(identifier, job)
    else:
        identifier = args.id

    try:
        result = controller.run(identifier, resume=args.command == "resume")
    except (RuntimeError, ValueError, OSError, ConnectionError) as exc:
        print(json.dumps({
            "id": identifier, "status": "interrupted", "error": str(exc),
        }, indent=2))
        return 2

    print(json.dumps({
        "id": identifier,
        "stage": result.stage,
        "verification": result.verification,
        "attempts": result.attempts,
        "state_file": str(store.path(identifier)),
    }, indent=2))
    return 0 if result.stage == "complete" and (
        result.verification.get("status") == "passed"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
