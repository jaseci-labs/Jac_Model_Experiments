# Task 5: Agentic Trajectory Generation

## Purpose

Define a one-shot Cursor workflow for collecting agentic Jac task-solving trajectories with Jac MCP/tooling attached. Unlike single-turn OpenAI API examples, a trajectory is valuable because it records real planning, tool use, compiler feedback, recovery, and final output.

Cursor is the required environment for this workflow because this workspace already has access to the Jac MCP server and its validation/documentation tools. Task 5 should be executed by Cursor only.

## Inputs Needed

- [`context.md`](context.md) for trajectory strategy and target counts.
- [`task1.md`](task1.md) for metadata and storage policy.
- [`task3.md`](task3.md) for compiler validation and retry limits.
- Access to Cursor with Jac MCP/tooling attached.
- Cursor access to the Jac MCP tools, especially `understand_jac_and_jaseci`, `get_resource`, `validate_jac`, `check_syntax`, `lint_jac`, and `explain_error`.

## Artifacts To Produce

- A trajectory task bank.
- Raw transcript files.
- Clean trajectory examples that match the target chat template.
- Rejected trajectory records with discard reasons.
- Review notes for sampled trajectories.

## Storage Layout

Store trajectory artifacts under the same dataset structure used by the other categories:

- `dataset/raw_output/trajectory/` for raw Cursor transcripts before cleaning.
- `dataset/clean_dataset/trajectory/` for accepted JSONL trajectory examples.
- `dataset/rejected/trajectory/` for failed or discarded trajectory records.
- `dataset/review/trajectory/` for sampled review notes and adjudication records.
- `dataset/logs/generation/` for trajectory generation metadata.
- `dataset/logs/validation/` for Jac MCP validation results.

Use the existing batch ID convention from Task 1, such as `YYYYMMDD-trajectory-001`. Each clean JSONL file should contain one trajectory record per line.

## Recommended JSONL Schema

Each accepted trajectory should be written as one JSON object per line:

```json
{
  "id": "trajectory-20260511-001-0001",
  "batch_id": "20260511-trajectory-001",
  "category": "trajectory",
  "complexity": "simple",
  "compiler_pass": true,
  "test_pass": null,
  "manually_reviewed": false,
  "generator": "cursor-jac-mcp",
  "generation_date": "2026-05-11T00:00:00Z",
  "source_prompt_version": "trajectory-prompt-v1",
  "context_bundle_version": "jac-context-v1",
  "validator_version": "jac-mcp-validate-v1",
  "dataset_version": "jac-synth-v0.1.0",
  "trajectory_length_tokens": 4200,
  "review_status": "pending",
  "rejection_reason": null,
  "task": {
    "prompt": "Build a Jac walker that...",
    "difficulty_reason": "Uses a stateful walker and node traversal.",
    "expected_capabilities": ["walker", "node", "edge", "validation"]
  },
  "final_output": {
    "language": "jac",
    "code": "complete final Jac code",
    "validation_tool": "user-jac.validate_jac",
    "validation_result": "structured compiler result or summary"
  },
  "turns": [
    {"role": "user", "content": "task description"},
    {"role": "assistant", "content": "reasoning and plan"},
    {"role": "tool_call", "content": "user-jac.get_resource({\"uri\":\"jac://guide/pitfalls\"})"},
    {"role": "tool_result", "content": "relevant tool output"},
    {"role": "tool_call", "content": "user-jac.validate_jac({\"code\":\"...\"})"},
    {"role": "tool_result", "content": "compiler output"},
    {"role": "assistant", "content": "response to compiler output"},
    {"role": "assistant", "content": "final answer and final code"}
  ]
}
```

The `turns` role names must match the chat template used during supervised finetuning. If the final training template uses different names for tool calls or assistant messages, update this schema before collecting more trajectories.

Rejected trajectories should use the same metadata fields where possible, with `compiler_pass: false` or the relevant failed state, `review_status: "rejected"`, and a concrete `rejection_reason`.

## Cursor One-Shot Pilot Workflow

Use Cursor to complete the first pilot batch end to end in one controlled session:

1. Confirm Jac MCP access in Cursor.
   - Call `understand_jac_and_jaseci`.
   - Fetch `jac://guide/pitfalls` and `jac://guide/patterns` before writing or validating Jac code.
   - Confirm `validate_jac` is available for final compiler validation.
2. Create the pilot batch ID.
   - Example: `20260511-trajectory-001`.
   - Use `generator: "cursor-jac-mcp"` for Cursor sessions.
3. Generate the small task bank inside the session.
   - Ask Cursor to propose 15 candidate tasks across simple, medium, and complex difficulty.
   - Use the target distribution as guidance: 30% simple, 50% medium, 20% complex.
   - Reason over the proposed tasks before selecting the three pilot tasks.
   - Select one simple, one medium, and one complex task for the first pilot batch.
   - Prefer tasks likely to trigger useful compiler feedback without being too large for the context window.
4. Run three pilot trajectory sessions.
   - Start a fresh trajectory for each selected task.
   - Provide exactly one user task request per trajectory.
   - Let the agent plan, consult Jac MCP docs, write code, validate, respond to compiler feedback, and revise.
   - Do not accept a trajectory where the final Jac code has not been validated.
5. Capture the raw transcript.
   - Preserve user turns, assistant turns, MCP tool calls, and MCP tool results.
   - Store the raw transcript before cleaning.
   - Keep enough MCP output to show the agent's reasoning and recovery, but remove irrelevant private workspace context.
6. Convert the transcript to the JSONL schema.
   - Normalize role names.
   - Add metadata.
   - Extract the final Jac code into `final_output.code`.
   - Record the validation tool and validation result.
   - Estimate or record `trajectory_length_tokens`.
7. Apply keep/discard criteria.
   - Clean examples go to `dataset/clean_dataset/trajectory/`.
   - Failed examples go to `dataset/rejected/trajectory/`.
   - Validation logs go to `dataset/logs/validation/`.
8. Review the three pilot trajectories manually.
   - Check that each task was solved.
   - Check that tool use was logical and complete.
   - Check that the final code is idiomatic Jac, not Python-like Jac.
   - Decide whether the process is ready for 20-50 trajectory batches.

## Step-By-Step Checklist

- [ ] Define the trajectory task distribution:
  - 30% simple tasks.
  - 50% medium tasks.
  - 20% complex tasks.
- [ ] Build a small task bank inside the Cursor pilot session:
  - Simple: focused walkers, small node/edge structures, basic abilities.
  - Medium: graph algorithms, data processing, stateful walkers, small modules.
  - Complex: multi-file examples, web/API patterns, authentication, routing, or error handling.
- [ ] Reason over the task bank and select one simple, one medium, and one complex pilot task.
- [ ] Start with 3 pilot trajectory sessions in Cursor before collecting volume.
- [ ] For each session, confirm Cursor has Jac MCP/tooling attached.
- [ ] Provide one user task request at the target difficulty.
- [ ] Let the agent work end to end: plan, call MCP tools, read compiler output, revise, and produce final code.
- [ ] Record the complete transcript, including user turns, assistant turns, tool calls, and tool results.
- [ ] Compile the final Jac output with the Jac MCP compiler.
- [ ] Keep the trajectory only if final output compiles and the task is actually solved.
- [ ] Prefer trajectories where the agent encounters a compiler error, reasons about it, and recovers.
- [ ] Store successful transcripts in the clean trajectory format.
- [ ] Store failed transcripts in rejected storage with discard reasons.
- [ ] Review a 5-10% sample of accepted trajectories.

## Required Trajectory Format

Each trajectory must be stored as ordered turns compatible with the training chat template. The minimal turn format is:

```json
[
  {"role": "user", "content": "task description"},
  {"role": "assistant", "content": "reasoning and plan"},
  {"role": "tool_call", "content": "jac_mcp.compile(...)"},
  {"role": "tool_result", "content": "compiler output"},
  {"role": "assistant", "content": "response to compiler output"},
  {"role": "assistant", "content": "final output"}
]
```

The exact role names and turn structure must match the chat template used during supervised finetuning. If the training template changes, update the trajectory format before collecting more data.

For clean JSONL records, use the full schema above so metadata, task information, final code, validation result, and ordered turns remain together.

## Testing And Validation Checklist

- [ ] Confirm Jac MCP/tooling was attached during the session.
- [ ] Confirm `understand_jac_and_jaseci` was called at least once for the pilot batch.
- [ ] Confirm task-relevant Jac docs were fetched before code was finalized.
- [ ] Confirm all relevant tool calls and tool results are present in the transcript.
- [ ] Confirm final code compiles.
- [ ] Confirm the final output satisfies the original user task.
- [ ] Confirm the transcript is not longer than the training context window.
- [ ] Confirm no private, irrelevant, or environment-specific data is included.
- [ ] Confirm the trajectory metadata records complexity, generator, compiler result, and review status.

## Keep Criteria

Keep trajectories where:

- The final Jac output compiles.
- The agent uses MCP tools in a logical sequence.
- The agent responds meaningfully to compiler or tool feedback.
- The task is solved, not merely attempted.
- The trajectory fits inside the target training context window.
- The transcript role format matches the training template.
- The cleaned transcript preserves enough tool context to teach recovery behavior.

## Discard Criteria

Discard trajectories where:

- The agent gives up or does not solve the task.
- The final Jac code does not compile.
- The agent makes more than three consecutive failed compiler calls without recovery.
- The transcript is too long for the initial SFT context window.
- Tool calls or tool results are missing.
- The session contains irrelevant private workspace data.
- The task is so broad that the transcript becomes mostly planning instead of concrete Jac problem solving.
- The final code is Python-like Jac that only happens to pass a narrow syntax check.

## Failure Conditions And Retry Guidance

- If pilot trajectories are too easy and contain no recovery, add tasks that naturally require compiler feedback.
- If sessions fail repeatedly, reduce task complexity and inspect whether the Jac MCP context is complete.
- If transcripts are too long, break complex tasks into smaller tasks or tighten the task prompt.
- If tool call formatting does not match the training template, fix formatting before collecting more trajectories.
- If agent behavior is poor but final code compiles, mark for manual review rather than accepting automatically.
- If the in-session task bank is unbalanced, regenerate or edit the candidate tasks before launching the three pilot sessions.
- If raw Cursor transcripts include unrelated workspace context, remove that material during cleaning and record the sanitization in review notes.

## Completion Criteria

This task is complete when the Cursor one-shot pilot produces a reasoned task bank, three pilot trajectory sessions, valid raw transcripts, clean JSONL records, Jac MCP validation logs, rejected records where applicable, and review notes showing whether the process is ready for volume generation.

## Operator Commands

Plan trajectory collection:

```bash
python -m data_generation.trajectory_generation plan --target-count 2000 --existing-count 0
```

Ingest a normalized trajectory transcript:

```bash
python -m data_generation.trajectory_generation ingest --input transcript.json --date 20260511 --sequence 2 --generation-date 2026-05-11T00:00:00Z
```

Audit trajectory task status:

```bash
python -m data_generation.trajectory_generation audit --batch-id 20260511-trajectory-002
```
