"""
gen_trajectory.jac -- pilot generator for the `trajectory` category (spec.md
04-cpt-sft/docs/datagen/spec.md section 5), built on the prompt-batch
handoff protocol in batch.jac. Same two-phase shape as gen_code_gen.jac and
gen_debug.jac -- read those first for the general pattern, not repeated
here.

KEY DESIGN DIFFERENCE from every other jacgen2 generator: ONE prompt per
example asks for the WHOLE multi-turn conversation in a single structured
response (a JSON array of {role, content} turns), not one prompt per turn.
This mirrors spec.md section 5's "one Opus call per example" design --
unrolling turn-by-turn across separate calls would multiply cost and break
the "the model writes both sides" framing the category exists to capture.
model="opus" per spec.md section 4.1 (trajectory is Opus-assigned).

All 5 spec.md section 5 scenario types are implemented (see each pool's
construction in run_prepare):

    build_from_scratch      -> behavioral seeds (code_gen-eligible spirit)
    debug_session            -> buggy_variants.jsonl entries joined back to
                                 seed_pool by seed_id
    refactor_session          -> behavioral seeds with >15 lines of code
    add_feature_to_existing   -> any code-bearing seed
    code_review_session       -> any code-bearing seed

PRODUCTION-SCALE NOTE (was CAP NOTE at pilot time): at pilot time,
raw_output/debug/buggy_variants.jsonl had only 4 real records (Task
11/gen_debug's pilot output), so debug_session was hard-capped at 4. Since
then gen_debug.jac was scaled to production volume and buggy_variants.jsonl
now has ~210 real records, so there is no more artificial
DEBUG_SESSION_HARD_CAP -- debug_session's real ceiling is simply
`min(JAC_PILOT_LIMIT-derived target, len(debug_pool))`, exactly like every
other scenario type's pool (see append_debug_items/append_generic_items).
The roughly-even 5-way split of JAC_PILOT_LIMIT across the 5 scenario types
is unchanged -- still no rebalancer redistributing one type's pool-size
shortfall to the others, a rebalancer is more machinery than this generator
needs.

Two phases, selected by env var JAC_BATCH_PHASE:

  prepare -- for each eligible (task_type, seed-or-buggy-variant), write a
    pending batch item with the full "write the whole conversation" prompt.
    Writes pending_batch.jsonl and stops -- no LLM call happens in this
    phase.

  collect -- reads responses_batch.jsonl, joins each response back to its
    pending-item context by batch_id, tolerantly parses the JSON turn array,
    validates structure (>=3 turns, alternating roles starting with "user",
    final turn assistant + ```jac fence), gates the FINAL turn's code only
    (spec.md section 5: "intermediate turns showing a wrong attempt ... are
    intentionally not gated"), and persists the FULL messages list via
    gate.make_example_v2 + common.append_jsonl.

Run (from repo root):
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=prepare JAC_PILOT_LIMIT=20 \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_trajectory.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=collect \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_trajectory.jac
"""

import os;
import json;
import from pathlib { Path }
import from common { require_run_tag, data_root, append_jsonl, existing_keys }
import from gate { run_jac_local, behavioral_gate, make_example_v2 }
import from batch { make_batch_id, write_pending_batch, read_response_batch, pending_path, responses_path }

glob CATEGORY: str = "trajectory";
glob SEED_POOL_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl";
glob GENERATOR_MODEL_ID: str = "claude-opus-4-8";

glob SCENARIO_TYPES: list[str] = [
    "build_from_scratch",
    "debug_session",
    "refactor_session",
    "add_feature_to_existing",
    "code_review_session"
];

glob SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "build_from_scratch": "the user requesting a feature be built from scratch in plain language, the assistant building it up incrementally, and the user asking 1-2 follow-up refinements",
    "debug_session": "the user reporting a real bug (pasting broken code and its symptom), the assistant diagnosing it -- possibly proposing a wrong fix first, the user reporting it still doesn't work, and the assistant arriving at a correct fix",
    "refactor_session": "the user providing existing working Jac code and asking to restructure it (e.g. split into an impl file, extract a walker from inline logic)",
    "add_feature_to_existing": "the user providing an existing Jac snippet and asking to extend it (add a field, add an endpoint, add a route)",
    "code_review_session": "the assistant reviewing user-provided Jac code unprompted-style, flagging 1-2 real issues, the user asking to fix one, and the assistant revising"
};


def load_seed_pool() -> list[dict] {
    seeds: list[dict] = [];
    path: Path = Path(SEED_POOL_PATH);
    if not path.exists() {
        raise RuntimeError("seed pool not found at " + SEED_POOL_PATH + " -- run seed_pool.jac first.");
    }
    with open(SEED_POOL_PATH, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            seeds.append(dict(rec));
        }
    }
    return seeds;
}

"""Reads raw_output/debug/buggy_variants.jsonl (gen_debug.jac's output --
schema {seed_id, bug_type, buggy_code, symptom, clean_code}). Returns []
if the file doesn't exist yet (normal if gen_debug's collect phase hasn't
run for this tag) rather than raising -- debug_session simply has zero
eligible items in that case."""
def load_buggy_variants(path: str) -> list[dict] {
    variants: list[dict] = [];
    p: Path = Path(path);
    if not p.exists() {
        return variants;
    }
    with open(path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            variants.append(dict(rec));
        }
    }
    return variants;
}


def context_build_from_scratch(jac_code: str) -> str {
    return (
        "The eventual correct solution the conversation should arrive at (do not let the "
        + "'user' turn reveal this framing -- write their opening message as a natural feature "
        + "request that would lead here) is:\n\n```jac\n" + jac_code + "\n```\n\n"
        + "Imagine the original user request that would lead to this being the final code, "
        + "then unroll 1-2 follow-up refinement turns after the initial build as if the user "
        + "asked for small additional tweaks along the way."
    );
}

def context_debug_session(buggy_code: str, symptom: str, clean_code: str) -> str {
    return (
        "Original buggy code:\n```jac\n" + buggy_code + "\n```\n\n"
        + "Symptom the user reports: " + symptom + "\n\n"
        + "A correct fix looks like this (guidance for the destination the dialogue should "
        + "reason its way to -- do not just paste it in a vacuum):\n```jac\n" + clean_code + "\n```\n\n"
        + "The user's first message should paste the buggy code and describe the symptom in "
        + "their own words. The assistant may propose a plausible-but-wrong fix first; the "
        + "user reports it still doesn't work; the assistant then arrives at something "
        + "equivalent to the correct fix."
    );
}

def context_refactor_session(jac_code: str) -> str {
    return (
        "Existing working code the user wants restructured (e.g. split into an impl file, "
        + "extract a walker from inline logic, or a similar restructuring):\n\n```jac\n"
        + jac_code + "\n```\n"
    );
}

def context_add_feature(jac_code: str) -> str {
    return (
        "Existing code the user wants extended with a new feature (e.g. a new field, a new "
        + "endpoint, a new route):\n\n```jac\n" + jac_code + "\n```\n"
    );
}

def context_code_review(jac_code: str) -> str {
    return (
        "Code the assistant should review unprompted-style, flagging 1-2 real issues -- after "
        + "which the user asks to fix one and the assistant revises:\n\n```jac\n" + jac_code
        + "\n```\n"
    );
}

"""Dispatches to the right context_* builder for the 4 scenario types that
take a single seed's jac_code (everything except debug_session, which needs
three separate strings and is built directly in append_debug_items)."""
def build_context(task_type: str, jac_code: str) -> str {
    if task_type == "build_from_scratch" {
        return context_build_from_scratch(jac_code);
    } elif task_type == "refactor_session" {
        return context_refactor_session(jac_code);
    } elif task_type == "add_feature_to_existing" {
        return context_add_feature(jac_code);
    } elif task_type == "code_review_session" {
        return context_code_review(jac_code);
    }
    raise ValueError("build_context: unsupported task_type " + task_type);
}

"""Verbatim core prompt template (spec-mandated wording) wrapped around a
scenario-specific description + context block."""
def build_prompt(task_type: str, context: str) -> str {
    description: str = SCENARIO_DESCRIPTIONS[task_type];
    core: str = (
        "You are generating a realistic multi-turn coding conversation (3-6 turns) for a Jac "
        + "programming language training dataset. Scenario: " + description + ".\n\n"
        + "Write the FULL conversation as a JSON array of {\"role\": \"user\"|\"assistant\", "
        + "\"content\": \"...\"} objects, alternating starting with \"user\". The final "
        + "assistant turn's content MUST include a complete, syntactically valid Jac code "
        + "block (```jac fenced). Intermediate turns MAY show a wrong attempt, an error "
        + "message, or a clarifying question -- a trajectory with zero friction is not "
        + "realistic and not useful for this dataset.\n\n"
        + "Reply with ONLY the JSON array, no other text, no markdown fence around the array "
        + "itself (the array elements' \"content\" fields may contain their own ```jac fences "
        + "internally, that's fine and expected).\n\n"
    );
    return core + context;
}

"""Appends up to `cap` items (pool[:cap]) for one of the 4 single-seed
scenario types. Returns the number actually appended."""
def append_generic_items(items: list[dict], pool: list[dict], task_type: str, cap: int) -> int {
    n: int = min(cap, len(pool));
    for i in range(n) {
        seed: dict = dict(pool[i]);
        jac_code: str = str(seed["jac_code"]);
        context: str = build_context(task_type, jac_code);
        prompt: str = build_prompt(task_type, context);
        batch_id: str = make_batch_id(CATEGORY, task_type, str(seed["seed_id"]), 0);
        items.append({
            "batch_id": batch_id,
            "model": "opus",
            "prompt": prompt,
            "category": CATEGORY,
            "task_type": task_type,
            "variant_idx": 0,
            "seed_id": seed["seed_id"],
            "seed_tier": seed["seed_tier"],
            "origin": seed["origin"],
            "jac_code": jac_code,
            "gate_class": seed["gate_class"],
            "expected_output": seed["expected_output"],
            "domains": seed["domains"]
        });
    }
    return n;
}

"""debug_session items are seeded from buggy_variants.jsonl records (not
directly from the seed pool), joined back to the seed pool by seed_id for
seed_tier/origin/domains. gate_class is forced to "compile_only" here per
spec.md section 5 -- there is no single pinned expected_output for a whole
debugging conversation's outcome, so collect mode uses run_jac_local exit-0
on the final turn instead of an exact-output behavioral gate (see this
file's run_collect)."""
def append_debug_items(items: list[dict], pool: list[dict], seed_by_id: dict, cap: int) -> int {
    n: int = min(cap, len(pool));
    for i in range(n) {
        v: dict = dict(pool[i]);
        seed: dict = dict(seed_by_id[str(v["seed_id"])]);
        buggy_code: str = str(v["buggy_code"]);
        clean_code: str = str(v["clean_code"]);
        symptom: str = str(v["symptom"]);
        context: str = context_debug_session(buggy_code, symptom, clean_code);
        prompt: str = build_prompt("debug_session", context);
        batch_id: str = make_batch_id(CATEGORY, "debug_session", str(seed["seed_id"]), 0);
        items.append({
            "batch_id": batch_id,
            "model": "opus",
            "prompt": prompt,
            "category": CATEGORY,
            "task_type": "debug_session",
            "variant_idx": 0,
            "seed_id": seed["seed_id"],
            "seed_tier": seed["seed_tier"],
            "origin": seed["origin"],
            "jac_code": clean_code,
            "buggy_code": buggy_code,
            "clean_code": clean_code,
            "symptom": symptom,
            "bug_type": v["bug_type"],
            "gate_class": "compile_only",
            "expected_output": None,
            "domains": seed["domains"]
        });
    }
    return n;
}


def run_prepare() {
    tag: str = require_run_tag();
    limit: int = int(os.environ.get("JAC_PILOT_LIMIT", "20"));

    seeds: list[dict] = load_seed_pool();
    seed_by_id: dict = {};
    for seed in seeds {
        seed_by_id[str(seed["seed_id"])] = seed;
    }

    variants_path: str = data_root(tag) + "/raw_output/debug/buggy_variants.jsonl";
    buggy_variants: list[dict] = load_buggy_variants(variants_path);

    build_pool: list[dict] = [];
    refactor_pool: list[dict] = [];
    for seed in seeds {
        if str(seed["gate_class"]) == "behavioral" {
            build_pool.append(seed);
            if len(str(seed["jac_code"]).splitlines()) > 15 {
                refactor_pool.append(seed);
            }
        }
    }

    debug_pool: list[dict] = [];
    for v in buggy_variants {
        if str(v["seed_id"]) in seed_by_id {
            debug_pool.append(v);
        }
    }

    # Roughly-even 5-way split of JAC_PILOT_LIMIT, remainder going to the
    # earliest types in SCENARIO_TYPES order.
    n_types: int = len(SCENARIO_TYPES);
    base: int = limit // n_types;
    remainder: int = limit % n_types;
    targets: dict[str, int] = {};
    idx: int = 0;
    for name in SCENARIO_TYPES {
        t: int = base + (1 if idx < remainder else 0);
        targets[name] = t;
        idx += 1;
    }
    # debug_session's real ceiling is now just its pool size (see
    # append_debug_items's min(cap, len(pool))) -- no more artificial
    # DEBUG_SESSION_HARD_CAP now that buggy_variants.jsonl is at production
    # volume (see PRODUCTION-SCALE NOTE in this file's top docstring).

    # add_feature_to_existing and code_review_session both draw from the
    # full (unfiltered) seed pool -- offset review's slice past feature's
    # so a pilot run doesn't hand the exact same first-N seeds to both
    # types (seed pool has 1300+ entries, so there's no real scarcity here,
    # just cheap variety).
    feature_pool: list[dict] = seeds;
    review_pool: list[dict] = seeds[targets["add_feature_to_existing"]:];

    items: list[dict] = [];
    n_build: int = append_generic_items(items, build_pool, "build_from_scratch", targets["build_from_scratch"]);
    n_refactor: int = append_generic_items(items, refactor_pool, "refactor_session", targets["refactor_session"]);
    n_feature: int = append_generic_items(items, feature_pool, "add_feature_to_existing", targets["add_feature_to_existing"]);
    n_review: int = append_generic_items(items, review_pool, "code_review_session", targets["code_review_session"]);
    n_debug: int = append_debug_items(items, debug_pool, seed_by_id, targets["debug_session"]);

    out_path: str = pending_path(tag, CATEGORY);
    write_result: dict = write_pending_batch(out_path, items);

    print("");
    print("=== gen_trajectory prepare summary ===");
    print("candidate items this run: " + str(len(items)) + " (JAC_PILOT_LIMIT=" + str(limit) + ")");
    print("  build_from_scratch: " + str(n_build) + " (pool size " + str(len(build_pool)) + ")");
    print("  debug_session: " + str(n_debug) + " (pool size " + str(len(debug_pool)) + ")");
    print("  refactor_session: " + str(n_refactor) + " (pool size " + str(len(refactor_pool)) + ")");
    print("  add_feature_to_existing: " + str(n_feature) + " (pool size " + str(len(feature_pool)) + ")");
    print("  code_review_session: " + str(n_review) + " (pool size " + str(len(review_pool)) + ")");
    print("newly written to pending batch: " + str(write_result["written"]));
    print("skipped as already-present in pending batch: " + str(write_result["skipped"]));
    print("wrote " + out_path);
}

"""Tolerant JSON-array parse of an Opus response: strips a leading/trailing
markdown fence if the model wrapped the array despite instructions not to,
then json.loads. Returns (parsed_list, "") on success, ([], reason) on any
failure -- callers must check the reason string (not `not parsed`, since a
genuinely empty-but-valid `[]` should fall through to structural validation
instead of being reported as a parse failure)."""
def parse_trajectory_response(text: str) -> tuple[list, str] {
    s: str = text.strip();
    if s.startswith("```") {
        first_newline: int = s.find("\n");
        if first_newline != -1 {
            s = s[first_newline + 1:];
        } else {
            s = s[3:];
        }
    }
    if s.endswith("```") {
        s = s[:len(s) - 3];
    }
    s = s.strip();
    try {
        parsed = json.loads(s);
    } except Exception as e {
        return ([], "json parse failed: " + str(e)[:200]);
    }
    if not isinstance(parsed, list) {
        return ([], "parsed JSON is not a list");
    }
    return (list(parsed), "");
}

"""Structural validation per spec.md section 5 + this file's collect
contract: at least 3 turns, alternating roles starting with "user", final
turn role=="assistant" containing a ```jac fence. Returns
(ok, reason, final_code) -- final_code is "" when ok is False."""
def validate_trajectory(messages: list) -> tuple[bool, str, str] {
    if len(messages) < 3 {
        return (False, "fewer than 3 turns: " + str(len(messages)), "");
    }
    expected_role: str = "user";
    for m in messages {
        if not isinstance(m, dict) {
            return (False, "turn is not a JSON object", "");
        }
        entry: dict = dict(m);
        if "role" not in entry or "content" not in entry {
            return (False, "turn missing role/content key", "");
        }
        role: str = str(entry["role"]);
        if role != expected_role {
            return (False, "turns not alternating starting with user (expected " + expected_role + ", got " + role + ")", "");
        }
        expected_role = "assistant" if expected_role == "user" else "user";
    }
    last: dict = dict(messages[-1]);
    if str(last["role"]) != "assistant" {
        return (False, "final turn is not role=assistant", "");
    }
    content: str = str(last["content"]);
    if "```jac" not in content {
        return (False, "final assistant turn has no ```jac fence", "");
    }
    final_code: str = content.split("```jac", 1)[1].rsplit("```", 1)[0].strip();
    if not final_code {
        return (False, "final ```jac fence is empty", "");
    }
    return (True, "", final_code);
}


def run_collect() {
    tag: str = require_run_tag();
    p_path: str = pending_path(tag, CATEGORY);
    r_path: str = responses_path(tag, CATEGORY);

    if not Path(p_path).exists() {
        print("ERROR: pending batch not found at " + p_path + " -- run prepare mode first.");
        return;
    }

    responses: dict = read_response_batch(r_path);
    if not responses {
        print("no responses found yet at " + r_path + ".");
        print("this is normal between prepare and the orchestrator's generation pass --");
        print("nothing to collect. Re-run collect mode once " + r_path + " exists.");
        return;
    }

    pending_by_id: dict = {};
    with open(p_path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            pending_by_id[rec["batch_id"]] = rec;
        }
    }

    clean_path: str = data_root(tag) + "/clean_dataset/trajectory/examples.jsonl";
    rejected_path: str = data_root(tag) + "/rejected/trajectory/rejected.jsonl";
    already_persisted: dict = existing_keys(clean_path);

    processed: int = 0;
    accepted: int = 0;
    already_persisted_skipped: int = 0;
    reason_counts: dict[str, int] = {};
    turn_counts: dict[str, int] = {};

    for batch_id in responses {
        processed += 1;
        resp: dict = dict(responses[batch_id]);
        ctx = pending_by_id.get(batch_id, None);

        if ctx is None {
            reason: str = "no matching pending context for batch_id";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {"batch_id": batch_id, "category": CATEGORY, "reason": reason});
            continue;
        }

        task_type: str = str(ctx["task_type"]);

        if not bool(resp["ok"]) {
            reason: str = "generation failed (ok:false)";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": task_type, "reason": reason
            });
            continue;
        }

        (parsed, parse_err) = parse_trajectory_response(str(resp["text"]));
        if parse_err {
            reason: str = "parse failed: " + parse_err;
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": task_type, "reason": reason
            });
            continue;
        }

        (struct_ok, struct_reason, final_code) = validate_trajectory(parsed);
        if not struct_ok {
            reason: str = "structure invalid: " + struct_reason;
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": task_type, "reason": reason
            });
            continue;
        }

        seed_gate_class: str = str(ctx["gate_class"]);
        expected_output = ctx.get("expected_output", None);
        # Only run the exact-output behavioral gate when: not debug_session
        # (no single pinned outcome for a whole debugging conversation --
        # spec.md section 5), the ORIGINATING seed is behavioral, and it
        # actually has a pinned expected_output (compile_only seeds carry
        # None). Every other combination (debug_session, or a compile_only
        # seed under any other scenario type) falls back to "final code
        # must jac run clean" per this file's top docstring.
        use_behavioral: bool = (task_type != "debug_session") and (seed_gate_class == "behavioral") and (expected_output is not None);

        gate_ok: bool = False;
        gate_reason: str = "";
        if use_behavioral {
            (gate_ok, gate_reason) = behavioral_gate(final_code, str(expected_output));
        } else {
            (exit_code, run_out, run_err) = run_jac_local(final_code);
            if exit_code == 0 {
                gate_ok = True;
                gate_reason = "ok: run_jac_local exit 0";
            } else {
                gate_ok = False;
                gate_reason = "run_jac_local failed exit " + str(exit_code) + ": " + run_err.strip()[:200];
            }
        }

        if not gate_ok {
            reason: str = "gate failed: " + gate_reason;
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": task_type, "reason": reason
            });
            continue;
        }

        gate_class_used: str = "compile_only" if task_type == "debug_session" else seed_gate_class;
        # test_pass: real bool only for the exact-output behavioral path,
        # None otherwise (compile_only never had a pinned output to check
        # -- computed via a single post-branch ternary rather than assigned
        # inside each branch above, same jac type-narrowing reason as
        # gen_code_gen.jac/gen_debug.jac).
        test_pass: any = gate_ok if use_behavioral else None;

        messages: list[dict] = [];
        for m in parsed {
            entry: dict = dict(m);
            messages.append({"role": str(entry["role"]), "content": str(entry["content"])});
        }

        rec: dict = make_example_v2(
            category=CATEGORY,
            task_type=task_type,
            complexity="medium",
            messages=messages,
            gate_class=gate_class_used,
            compiler_pass=True,
            test_pass=test_pass,
            seed_id=str(ctx["seed_id"]),
            seed_tier=int(ctx["seed_tier"]),
            variant_idx=int(ctx["variant_idx"]),
            generator="opus-api",
            generator_model_id=GENERATOR_MODEL_ID,
            extra_meta={
                "origin": ctx["origin"],
                "turn_count": len(messages),
                "generator_model_id_note": "placeholder -- no models.json-style pinning "
                    + "exists yet for the Agent-dispatch (Opus/Fable subagent) path"
            }
        );
        if rec["id"] in already_persisted {
            already_persisted_skipped += 1;
            continue;
        }

        append_jsonl(clean_path, rec);
        already_persisted[rec["id"]] = True;
        accepted += 1;

        tc_key: str = str(len(messages));
        turn_counts[tc_key] = turn_counts.get(tc_key, 0) + 1;
    }

    rejected_count: int = processed - accepted - already_persisted_skipped;
    print("");
    print("=== gen_trajectory collect summary ===");
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("skipped as already-persisted: " + str(already_persisted_skipped));
    print("rejected: " + str(rejected_count));
    for reason in reason_counts {
        print("  - " + reason + ": " + str(reason_counts[reason]));
    }
    print("turn-count distribution of accepted:");
    for tc in turn_counts {
        print("  - " + tc + " turns: " + str(turn_counts[tc]));
    }
    print("wrote accepted examples to " + clean_path);
}


with entry:__main__ {
    phase: str | None = os.environ.get("JAC_BATCH_PHASE");
    if phase == "prepare" {
        run_prepare();
    } elif phase == "collect" {
        run_collect();
    } else {
        raise ValueError(
            "JAC_BATCH_PHASE must be set to 'prepare' or 'collect', got: " + str(phase)
        );
    }
}
