# Coder: local Qwen development runner

This is a **local-only** runtime for the OpenHands Software Agent SDK. The OpenHands model client communicates directly with
Ollama's native `/api/chat` endpoint, allowing the controller to set
`num_ctx` on every operation. It does not configure or fall back to any hosted AI service.

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
Inspect the project and checkpoint, then explicitly reconcile an unchanged
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

Coder initializes only native `qwen3:4b` inference on loopback; there
is no LiteLLM transport, dependency, or OpenAI-compatible bridge. Host-model
credentials, proxy fallback, and alternative models are not accepted by the
native `LLM` constructor. The sandboxed worker clears inherited environment
variables and forbids external provider selection. Cloud-provider
credential storage and API endpoints, subscription token refresh, external
LLM critic and multi-model routing tools have been removed. Local Qwen profiles
remain available for saving different context and output settings.

### Profile migration
Previously saved hosted-provider profiles are **not** deleted or rewritten
by an upgrade. They can appear in the profile inventory for identification,
but attempting to activate one fails with a clear local-only error. Recreate
a profile with model `qwen3:4b` and an Ollama loopback endpoint for continued
use. Existing encrypted credential files are not automatically destroyed.
Native profiles use no API key, subscription or provider connection.

## Desktop launcher and browser development studio

Coder includes a normal Linux desktop/application-menu launcher for Zorin
OS, Ubuntu, and other freedesktop-compatible desktops. After installing the
repository's Python dependencies from `software-agent-sdk-main`, run:

```bash
uv run python -m coder_local.desktop
```

This installs `~/.local/share/applications/coder-local.desktop` and a
`~/.local/bin/coder-local-studio` launcher. If the Desktop folder already
exists it also installs `~/Desktop/Coder.desktop`. No `sudo` is needed.
The launcher uses the existing checkout and its uv environment; it does not
download arbitrary executables or change the existing Coder controller.

Click **Coder** in the application menu (or on your desktop) to launch its
local server and open `http://127.0.0.1:8765` in your default browser.
If the server is already running, the shortcut opens the existing page.

The browser studio provides:

- **Repository selection:** supply an existing absolute project directory,
  or clone a GitHub HTTPS repository into the state folder's `projects/`.
- **Project command console:** run one explicitly entered command in the
  selected project, such as `git status`, `git log -3` or `python -m pytest`.
  This is a bounded command runner, not an interactive PTY: shell redirects,
  pipelines, interactive prompts and persistent shell sessions are not
  supported. Commands have a 45-second timeout and capped returned output.
- **Task assignment:** enter the objective, verification command and explicitly
  authorized relative file paths. Click **Go, my child** to queue the coding
  job. AI source edits still run in disposable staging under Bubblewrap.
- **Progress viewer:** inspect current stages, context allocation, saved
  verification outcome and error state.

**Command-console security:** human-entered terminal commands run with your
normal Linux account permissions and are **not** Bubblewrap-restricted.
Run only commands you trust. The agent's own code edits remain restricted by
the existing scope, stage verification and staging safeguards. The HTTP API
listens on loopback, validates local browser origin/Host, and requires JSON
POST requests.

Private GitHub repositories need appropriate pre-existing local Git
authentication; the UI never asks for your GitHub token or stores passwords.

## Local dashboard and queue

Start the local interface from `software-agent-sdk-main`:

```bash
uv run python -m coder_local.server
```

Open `http://127.0.0.1:8765` on the same machine. The dashboard can select repositories, run explicit local commands, submit
assignments, and display the active stage, context size and verification status.
Its API is bound to loopback only: `GET /health`, `GET /jobs`,
`GET /jobs/<id>`, `POST /workspace/inspect`, `POST /workspace/clone`,
`POST /workspace/command`, `POST /jobs`, and `POST /jobs/<id>/resume`. A single
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
