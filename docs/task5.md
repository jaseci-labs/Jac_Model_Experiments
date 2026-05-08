# Task 5: Agentic Trajectory Generation

## Purpose

Define how to collect agentic Jac task-solving trajectories using a vibe-coding agent with Jac MCP/tooling attached. Suitable environments include Cursor, Codex, and Claude Code. Unlike single-turn OpenAI API examples, a trajectory is valuable because it records real planning, tool use, compiler feedback, recovery, and final output.

## Inputs Needed

- [`context.md`](context.md) for trajectory strategy and target counts.
- [`task1.md`](task1.md) for metadata and storage policy.
- [`task3.md`](task3.md) for compiler validation and retry limits.
- Access to Cursor, Codex, Claude Code, or another approved vibe-coding agent with Jac MCP/tooling attached.
- A list of Jac tasks across simple, medium, and complex difficulty.

## Artifacts To Produce

- A trajectory task bank.
- Raw transcript files.
- Clean trajectory examples that match the target chat template.
- Rejected trajectory records with discard reasons.
- Review notes for sampled trajectories.

## Step-By-Step Checklist

- [ ] Define the trajectory task distribution:
  - 30% simple tasks.
  - 50% medium tasks.
  - 20% complex tasks.
- [ ] Build a small task bank before running sessions:
  - Simple: focused walkers, small node/edge structures, basic abilities.
  - Medium: graph algorithms, data processing, stateful walkers, small modules.
  - Complex: multi-file examples, web/API patterns, authentication, routing, or error handling.
- [ ] Start with 3 pilot trajectory sessions before collecting volume.
- [ ] For each session, open Cursor, Codex, Claude Code, or another approved vibe-coding agent with Jac MCP/tooling attached.
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

Each trajectory must be stored as ordered turns compatible with the training chat template:

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

## Testing And Validation Checklist

- [ ] Confirm Jac MCP/tooling was attached during the session.
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

## Discard Criteria

Discard trajectories where:

- The agent gives up or does not solve the task.
- The final Jac code does not compile.
- The agent makes more than three consecutive failed compiler calls without recovery.
- The transcript is too long for the initial SFT context window.
- Tool calls or tool results are missing.
- The session contains irrelevant private workspace data.

## Failure Conditions And Retry Guidance

- If pilot trajectories are too easy and contain no recovery, add tasks that naturally require compiler feedback.
- If sessions fail repeatedly, reduce task complexity and inspect whether the Jac MCP context is complete.
- If transcripts are too long, break complex tasks into smaller tasks or tighten the task prompt.
- If tool call formatting does not match the training template, fix formatting before collecting more trajectories.
- If agent behavior is poor but final code compiles, mark for manual review rather than accepting automatically.

## Completion Criteria

This task is complete when pilot trajectory sessions produce valid transcripts, final code compiles, keep/discard rules are applied consistently, and the collection process is ready for volume generation.
