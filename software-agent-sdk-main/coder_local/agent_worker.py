"""Isolated OpenHands worker invoked only inside a bubblewrap filesystem sandbox.

The parent controller owns stage transitions and opens the Ollama native bridge.
This worker receives an immutable request and writes a result under state storage.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m coder_local.agent_worker INPUT OUTPUT")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    response_path = Path(sys.argv[2])
    if request["model"] != "qwen3:4b":
        raise ValueError("Only local Qwen3:4B is permitted")
    if not request["base_url"].startswith("http://127.0.0.1:"):
        raise ValueError("Model adapter must be loopback")
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

    ctx = int(request["context_tokens"])
    output = int(request["output_tokens"])
    llm = LLM(
        model="openai/qwen3:4b",
        base_url=request["base_url"],
        api_key="local-only",
        api_mode="chat",
        max_input_tokens=ctx,
        max_output_tokens=output,
        max_message_chars=4500,
        stream=False,
        num_retries=1,
        timeout=int(request["timeout"]),
    )
    coding = bool(request["coding"])
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
            max_tokens=max(2048, ctx - output - 1024),
        ),
        system_prompt=(
            "You are a local software engineer working in an isolated filesystem. "
            "Complete only the requested bounded operation. Never use hosted AI. "
            "Never change files outside the authorized paths. "
            "Report actual results and never invent test outcomes."
        ),
    )
    conversation = Conversation(
        agent=agent,
        workspace=str(Path(request["workspace"]).resolve(strict=True)),
        persistence_dir=str(Path(request["persistence_dir"]).resolve(strict=True)),
        max_iteration_per_run=(
            int(request["max_iterations"]) if coding else 3
        ),
    )
    conversation.send_message(request["instruction"])
    conversation.run()
    if conversation.state.execution_status != ConversationExecutionStatus.FINISHED:
        raise RuntimeError(
            "OpenHands did not finish: " + str(conversation.state.execution_status)
        )
    response = get_agent_final_response(conversation.state.events)
    if response is None:
        raise RuntimeError("OpenHands finished without a final response")
    result = {
        "finished": True,
        "response": str(response),
        "context_tokens": ctx,
    }
    tmp = response_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, response_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
