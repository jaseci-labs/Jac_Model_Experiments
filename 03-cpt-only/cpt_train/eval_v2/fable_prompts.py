"""Exact prompt text sent to the Fable subagent (Agent tool, model="fable").
Batched at <=50 chunks/call -- design.md section 3. These functions only
BUILD the prompt string; the actual Agent tool call happens in the operational
step of Task 7 (an interactive/orchestration action, not unit-testable)."""
import json


def build_curation_prompt(batch: list, near_dup_candidates: list) -> str:
    chunks_json = json.dumps(
        [{"chunk_id": c["chunk_id"], "file": c["meta"].get("file", ""), "text": c["text"]}
         for c in batch],
        ensure_ascii=False, indent=2)
    dups_json = json.dumps(near_dup_candidates, indent=2) if near_dup_candidates else "[]"
    return f"""You are curating a continual-pretraining corpus of Jac programming language
documentation for a domain-adaptation training run. For each chunk below, decide:

- "keep" (default) -- genuine, non-redundant Jac/OSP documentation content.
- "drop" -- boilerplate (license headers, empty changelog stubs, nav-only fragments,
  auto-generated index pages with no prose), OR a near-duplicate flagged below where
  this chunk is the worse copy (less complete / less canonical).
- "upweight" -- genuinely core-concept material (OSP fundamentals: node/edge/walker
  semantics, spawn/visit/here mechanics, graph traversal model) that deserves extra
  training signal beyond the existing 3x docs-wide upsample. Include a "weight"
  field (float, e.g. 2.0-4.0) for how much extra.

Near-duplicate candidates flagged by a cheap shingle-overlap pre-pass (you decide
which of each pair, if either, to drop -- don't drop both, and don't drop a pair
if on inspection they're not actually redundant):
{dups_json}

Chunks:
{chunks_json}

Respond with ONLY a JSON array, one object per chunk_id, in this exact shape:
[{{"chunk_id": "...", "verdict": "keep"|"drop"|"upweight", "reason": "one sentence", "weight": 2.0}}]
("weight" only present for verdict="upweight".) Every chunk_id above must appear exactly once."""


def build_question_gen_prompt(batch: list) -> str:
    chunks_json = json.dumps(
        [{"chunk_id": c["chunk_id"], "file": c["meta"].get("file", ""), "text": c["text"]}
         for c in batch],
        ensure_ascii=False, indent=2)
    return f"""You are writing an evaluation question bank for a Jac programming language
domain-adaptation model, to be graded against a grounded RAG oracle's answers.

For each chunk below, write 1-2 OPEN-ENDED semantic questions (not multiple-choice)
that test understanding of the concept the chunk teaches -- "what happens when...",
"why would you use X instead of Y", "what's the difference between...". Avoid
questions answerable by pattern-matching syntax alone; target conceptual understanding.

Chunks:
{chunks_json}

Respond with ONLY a JSON array in this exact shape:
[{{"id": "q-<short-slug>", "question": "...", "source_chunk_id": "..."}}]
Every chunk_id above should produce at least one question, unless the chunk has no
real conceptual content to ask about (e.g. pure boilerplate) -- skip those, don't
force a question."""
