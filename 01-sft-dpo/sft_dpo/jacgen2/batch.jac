"""
Shared prompt-batch handoff protocol for jacgen2 generators.

Replaces in-process `by llm()` calls (llm.jac, unused by generators now --
left as-is, not in scope to remove) with a file handoff: Phase A (Jac) writes
one JSONL row per generation request ("pending"), an external orchestrator
(this repo's own Claude Code session, dispatching Opus/Fable subagents --
NOT a `by llm()` call, since Jac scripts can't dispatch Claude Code
subagents themselves) fills in a matching "responses" JSONL file keyed by
`batch_id`, and Phase B (Jac) joins pending + responses, gates, and persists.

See model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_code_gen.jac for the
first generator built on this protocol.
"""

import json;
import from pathlib { Path }
import from common { data_root, append_jsonl }

def make_batch_id(category: str, task_type: str, seed_id: str, variant_idx: int) -> str {
    return category + "|" + task_type + "|" + seed_id + "|" + str(variant_idx);
}

def pending_path(tag: str, category: str) -> str {
    return data_root(tag) + "/raw_output/" + category + "/pending_batch.jsonl";
}

def responses_path(tag: str, category: str) -> str {
    return data_root(tag) + "/raw_output/" + category + "/responses_batch.jsonl";
}

"""Reads an existing pending or responses JSONL file (each line containing a
`batch_id` field) into {batch_id: True, ...}. Returns {} if the file doesn't
exist yet. This is distinct from common.existing_keys (which tracks FINAL
persisted example ids) -- existing_batch_ids tracks in-flight batch-handoff-
file entries, so it belongs here rather than in common.jac."""
def existing_batch_ids(path: str) -> dict {
    ids: dict = {};
    p: Path = Path(path);
    if not p.exists() {
        return ids;
    }
    with open(path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            ids[rec["batch_id"]] = True;
        }
    }
    return ids;
}

"""Writes each item dict as one JSONL line to `path`. Each item must already
contain at minimum {"batch_id", "model": "opus"|"fable", "prompt"} plus
whatever extra context the caller wants preserved for Phase B -- items are
appended as-is, shape is the caller's responsibility.

Idempotent: skips items whose `batch_id` is already present in `path` rather
than re-appending them, so re-running prepare mode against an already-written
pending (or responses) file doesn't grow it unboundedly. Returns
{"written": int, "skipped": int} so callers can report both counts."""
def write_pending_batch(path: str, items: list[dict]) -> dict {
    already: dict = existing_batch_ids(path);
    written: int = 0;
    skipped: int = 0;
    for item in items {
        if item["batch_id"] in already {
            skipped += 1;
            continue;
        }
        append_jsonl(path, item);
        written += 1;
    }
    return {"written": written, "skipped": skipped};
}

"""Reads a responses JSONL file (each line {"batch_id", "text", "ok"}) into
{batch_id: {"text":..., "ok":...}}. Returns {} if the file doesn't exist yet
-- that's a normal state between Phase A and the orchestrator's work, not an
error; callers should handle an empty result gracefully."""
def read_response_batch(path: str) -> dict {
    responses: dict = {};
    p: Path = Path(path);
    if not p.exists() {
        return responses;
    }
    with open(path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            responses[rec["batch_id"]] = {"text": rec["text"], "ok": rec["ok"]};
        }
    }
    return responses;
}
