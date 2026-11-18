# Coder: local Qwen development runner

This is a **local-only** runtime for the OpenHands Software Agent SDK. It uses a
loopback adapter to send OpenHands chat/tool requests to Ollama's native
`/api/chat` endpoint, allowing the stage controller to set `num_ctx` on each
operation. It does not configure or fall back to any hosted AI service.

## Requirements

- Python 3.12+ with the OpenHands workspace dependencies installed
- Ollama running on `127.0.0.1:11434`
- The Ollama model `qwen3:4b` already installed
- A local working directory for the assigned project
- A **real verification command** for the project

From `software-agent-sdk-main`, an example invocation is:

```bash
uv run python -m coder_local run --project "$HOME/my-website" \
  --task "Fix the navigation menu" --test "npm test" --allow src --allow tests
```

The command returns the assignment id and stage. Recover after interruption
with `uv run python -m coder_local resume --id <id>`; inspect saved state
without invoking a model with `uv run python -m coder_local status --id <id>`.

The program keeps assignment state in `~/.local/share/coder/jobs/`, outside the
working repository. Source changes are **not** automatically committed, pushed,
merged, or deployed. `--allow` restricts changes to listed paths when used.
Use a clean working tree for reliable scope enforcement. Omitting `--test`
leaves completion unverified; it does not establish that the software works.

The default maximum candidate context is 32,768 tokens; set
`--max-context` to change it. The controller selects from 4K, 8K, 16K,
32K, and 64K allocations up to the configured cap, with model request
options transmitted to Ollama natively.

The SDK and its many provider integrations remain installed as library code:
this Coder entry point uses **only** the loopback inference route. Removing
shared upstream provider code without dependency and regression verification
would risk breaking the working SDK.

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
- Stage information is durable, but a subprocess interrupted during an
  in-flight tool action may require inspection before a retry.
- Testing the integration against the actual local Ollama installation remains
  necessary.
- The SDK archive contains a TypeScript API client but not the entire
  OpenHands Canvas UI. This runner is a local command-line entry point.
