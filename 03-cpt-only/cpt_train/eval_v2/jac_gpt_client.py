"""HTTP client for the self-hosted jac-gpt-fullstack oracle (design.md
section 7). POST /walker/interact streams Server-Sent Events -- confirmed
against the real cloned source (Task 14), not guessed. Each SSE frame is
`data: {"type": ..., "data": {"content": ...}}`. "chunk" events are the
token stream; "thought" carries the final ReAct answer when the loop ends
without a distinct tool-call finish; "tool_call" clears the running buffer
(pre-tool reasoning is discarded, not the answer). Read-only for eval --
does not call save_turn (turn persistence), since Track A/B ask one-shot
questions with no need for conversation history."""
import json
import uuid

import requests

DEFAULT_BASE_URL = "http://localhost:8000"


def ask_jac_gpt(question: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 120) -> str:
    resp = requests.post(
        f"{base_url}/walker/interact",
        json={"message": question, "session_id": str(uuid.uuid4()), "skip_history": True},
        stream=True, timeout=timeout,
    )
    resp.raise_for_status()

    buffer = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: "):])
        etype = event.get("type")
        if etype in ("chunk", "thought"):
            buffer += event["data"]["content"]
        elif etype == "tool_call":
            buffer = ""
    return buffer
