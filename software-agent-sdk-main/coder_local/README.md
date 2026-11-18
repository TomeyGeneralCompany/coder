# Coder: local Qwen development runner

This is a **local-only** runtime for the OpenHands Software Agent SDK. It uses a
loopback adapter to send OpenHands chat/tool requests to Ollama's native
`/api/chat` endpoint, allowing the stage controller to set `num_ctx` on each
operation. It does not configure or fall back to any hosted AI service.

## Requirements

- Python 3.12+ with the OpenHands workspace dependencies installed
- Bubblewrap (`bwrap`) with working Linux user and mount namespaces
- Ollama running on `127.0.0.1:11434`
- The Ollama model `qwen3:4b` already installed
- A local working directory for the assigned project
- A **real verification command** for the project

From `software-agent-sdk-main`, an example invocation is:

```bash
uv run python -m coder_local run --project "$HOME/my-website" \
  --task "Fix the navigation menu" --test "npm test" --allow src --allow tests
```

The command returns the assignment id and stage. Inspect saved state without
invoking a model with `uv run python -m coder_local status --id <id>`.
When an operation is interrupted, **Coder does not replay it automatically**.
Inspect the project and checkpoint, then explicitly reconcile a unchanged
project with `uv run python -m coder_local reconcile --id <id>`, followed by
`uv run python -m coder_local resume --id <id>`. Reconciliation refuses when
project files differ from the pre-operation checkpoint.

The program keeps assignment state in `~/.local/share/coder/jobs/`, outside the
working repository. Source changes are **not** automatically committed, pushed, merged, or
deployed. Both `--allow` and `--test` are required for work to proceed:
Coder blocks assignments with no explicit allowed paths or verification command.

Code changes are made in a disposable staging copy and checked against the
allowed paths before promotion to the original project. The actual OpenHands
worker runs in Bubblewrap with the host filesystem read-only. Only staging,
conversation persistence, and the result-artifact directory are writable
inside the worker. If Bubblewrap cannot start, Coder refuses the operation.
Test commands submitted by the user run on the original project after
promotion; configure those commands with care because they have the user's
normal filesystem permissions.

The default maximum candidate context is 32,768 tokens; set
`--max-context` to change it. The controller selects from 4K, 8K, 16K,
32K, and 64K allocations up to the configured cap, with model request
options transmitted to Ollama natively.

Coder sets `CODER_LOCAL_ONLY=1` inside its worker. The SDK enforces the
loopback Qwen3:4B model and rejects hosted model names, remote endpoints,
provider connections, subscriptions, AWS model credentials and LLM fallbacks.
Coder clears inherited environment variables in the worker. Upstream SDK
provider implementation modules and shared LiteLLM dependencies remain
present; the provider paths are disabled for Coder, not physically deleted.

## Local dashboard and queue

Start the local interface from `software-agent-sdk-main`:

```bash
uv run python -m coder_local.server
```

Open `http://127.0.0.1:8765` on the same machine. The dashboard can submit
assignments, display the active stage, context size and verification status.
Its API is bound to loopback only: `GET /health`, `GET /jobs`,
`GET /jobs/<id>`, `POST /jobs`, and `POST /jobs/<id>/resume`. A single
local worker handles queued assignments, and a Linux file lock prevents
multiple processes from running Coder jobs simultaneously.

Jobs are stored in `~/.local/share/coder/jobs` (outside project repositories).
The controller sets `DO_NOT_TRACK=1` while running the SDK. External inference
providers are neither listed nor selectable through this interface.

## Current limitations

- The 64K profile is a candidate, not a confirmed context capacity of the
  installed Qwen build or laptop.
- All development plans must be valid JSON lists of small operations; each
  operation advances only after the configured verification command passes.
- Interrupted operations remain blocked until explicit checkpoint reconciliation.
  Coder does not automatically replay a potentially side-effecting command.
- The strict filesystem sandbox keeps host files read-only to the agent worker,
  but shares the host network so Ollama on loopback remains reachable. Network
  restrictions are not implied by the filesystem sandbox.
- Testing the integration against the actual local Ollama installation remains
  necessary.
- The SDK archive contains a TypeScript API client but not the entire
  OpenHands Canvas UI. This runner is a local command-line entry point.
