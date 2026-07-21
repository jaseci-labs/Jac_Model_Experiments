"""
gen_code_gen.jac -- pilot generator for the `code_gen` category (spec.md
section 1), built on the prompt-batch handoff protocol in batch.jac.

This is a SCOPED-DOWN v1, not the full 34-task-type fan-out: it covers 6
task types (5 of spec.md 1.1's 1.5x-weighted differentiators, plus
core_language_basics as a catch-all) and 2 variants per eligible
(seed, task_type) pair, capped at JAC_PILOT_LIMIT total pending items. The
goal is to prove the whole prepare -> [orchestrator] -> collect loop works
end to end, not to hit category-weight targets yet.

Two phases, selected by env var JAC_BATCH_PHASE:

  prepare -- reverse-instruction generation setup (spec.md section 1): for
    each eligible (seed, task_type, variant_idx), write a pending batch item
    with the full prompt asking an LLM to write the natural-language task a
    canonical seed snippet solves. Writes pending_batch.jsonl and stops --
    no LLM call happens in this phase. model="opus" per spec.md section 4.1
    (code_gen is Opus-assigned).

  collect -- reads responses_batch.jsonl (written by an external
    orchestrator that dispatched Opus against pending_batch.jsonl), joins
    each response back to its pending-item context by batch_id, gates, and
    persists accepted examples via gate.make_example_v2 + common.append_jsonl.

Run (from repo root):
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=prepare JAC_PILOT_LIMIT=30 \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_code_gen.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=collect \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_code_gen.jac
"""

import os;
import json;
import from pathlib { Path }
import from common { require_run_tag, data_root, append_jsonl, existing_keys }
import from gate { behavioral_gate, make_example_v2 }
import from batch { make_batch_id, write_pending_batch, read_response_batch, pending_path, responses_path }

glob CATEGORY: str = "code_gen";
glob SEED_POOL_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl";
glob GENERATOR_MODEL_ID: str = "claude-opus-4-8";

"""Pilot task-type subset -- 5 of spec.md 1.1's 1.5x-weighted differentiators
plus core_language_basics as a catch-all. Full 34-type fan-out is future
work, not this pilot."""
glob TASK_TYPES: list[str] = [
    "node_edge_definition",
    "walker_traversal",
    "graph_query_patterns",
    "sv_endpoint_authoring",
    "cl_component_authoring",
    "core_language_basics"
];


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

"""A seed is eligible for a task_type if that task_type appears in its
`domains` list (assigned by seed_pool.jac's keyword-based domain hints)."""
def eligible_pairs(seeds: list[dict]) -> list[dict] {
    pairs: list[dict] = [];
    for seed in seeds {
        domains: list = list(seed["domains"]);
        for task_type in TASK_TYPES {
            if task_type in domains {
                pairs.append({"seed": seed, "task_type": task_type});
            }
        }
    }
    return pairs;
}

def build_prompt(jac_code: str, variant_idx: int) -> str {
    instructions: str = (
        "Here is a canonical Jac program. Write the natural-language task a developer "
        + "would have been given, for which this program is the correct solution. Vary "
        + "register by variant: variant 0 = plain user request (\"write a function "
        + "that...\"), variant 1 = terse expert/spec-style ask. This is variant "
        + str(variant_idx) + ". Do not describe the code line-by-line; describe the TASK "
        + "it accomplishes. Reply with ONLY the instruction text, nothing else -- no "
        + "preamble, no code, no markdown formatting."
    );
    return instructions + "\n\n```jac\n" + jac_code + "\n```\n";
}


def run_prepare() {
    tag: str = require_run_tag();
    limit: int = int(os.environ.get("JAC_PILOT_LIMIT", "30"));

    seeds: list[dict] = load_seed_pool();
    pairs: list[dict] = eligible_pairs(seeds);

    items: list[dict] = [];
    capped: int = 0;
    for pair in pairs {
        seed: dict = dict(pair["seed"]);
        task_type: str = str(pair["task_type"]);
        for variant_idx in [0, 1] {
            if len(items) >= limit {
                capped += 1;
                continue;
            }
            prompt: str = build_prompt(str(seed["jac_code"]), variant_idx);
            batch_id: str = make_batch_id(CATEGORY, task_type, str(seed["seed_id"]), variant_idx);
            items.append({
                "batch_id": batch_id,
                "model": "opus",
                "prompt": prompt,
                "category": CATEGORY,
                "task_type": task_type,
                "variant_idx": variant_idx,
                "seed_id": seed["seed_id"],
                "seed_tier": seed["seed_tier"],
                "origin": seed["origin"],
                "jac_code": seed["jac_code"],
                "gate_class": seed["gate_class"],
                "expected_output": seed["expected_output"],
                "domains": seed["domains"]
            });
        }
    }

    out_path: str = pending_path(tag, CATEGORY);
    write_result: dict = write_pending_batch(out_path, items);

    print("");
    print("=== gen_code_gen prepare summary ===");
    print("eligible (seed,task_type) pairs: " + str(len(pairs)));
    print("candidate items this run: " + str(len(items)) + " (JAC_PILOT_LIMIT=" + str(limit) + ")");
    print("skipped due to cap: " + str(capped));
    print("newly written to pending batch: " + str(write_result["written"]));
    print("skipped as already-present in pending batch: " + str(write_result["skipped"]));
    print("wrote " + out_path);
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

    clean_path: str = data_root(tag) + "/clean_dataset/code_gen/examples.jsonl";
    rejected_path: str = data_root(tag) + "/rejected/code_gen/rejected.jsonl";
    already_persisted: dict = existing_keys(clean_path);

    processed: int = 0;
    accepted: int = 0;
    already_persisted_skipped: int = 0;
    reason_counts: dict[str, int] = {};

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

        if not bool(resp["ok"]) {
            reason: str = "generation failed (ok:false)";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": ctx["task_type"], "reason": reason
            });
            continue;
        }

        instruction: str = str(resp["text"]).strip();
        if not instruction {
            reason: str = "empty instruction text";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": ctx["task_type"], "reason": reason
            });
            continue;
        }

        jac_code: str = str(ctx["jac_code"]);
        gate_class: str = str(ctx["gate_class"]);
        gate_ok: bool = False;
        gate_reason: str = "";

        if gate_class == "behavioral" {
            # Re-check the SEED's own already-proven code -- this is a final
            # sanity re-check, not a novel generation being gated, so it
            # should trivially pass.
            (gate_ok, gate_reason) = behavioral_gate(jac_code, str(ctx["expected_output"]));
        } elif gate_class == "compile_only" {
            # compile_only seeds were never expected to run standalone (they
            # were only parse-gated at pool-build time in seed_pool.jac).
            # Choice made here: skip re-gating entirely rather than calling
            # run_jac_local and trying to distinguish "catastrophic" failure
            # from the seed's normal non-runnable-standalone exit -- the
            # seed's parse-gate at pool time is trusted as sufficient.
            gate_ok = True;
            gate_reason = "compile_only: skipped re-gate, trusting seed_pool.jac's parse-gate";
        } else {
            gate_ok = False;
            gate_reason = "unknown gate_class: " + gate_class;
        }
        # test_pass: real bool for behavioral (did output match?), None for
        # compile_only (never had a test to run) -- computed after the
        # if/elif rather than assigned inside each branch (jac's type
        # checker narrows a declared `any` to the first branch-assigned
        # type otherwise, e.g. locking to bool then rejecting a later None).
        test_pass: any = gate_ok if gate_class == "behavioral" else None;

        if not gate_ok {
            reason: str = "gate failed: " + gate_reason;
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": ctx["task_type"], "reason": reason
            });
            continue;
        }

        messages: list[dict] = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "```jac\n" + jac_code + "\n```"}
        ];

        rec: dict = make_example_v2(
            category=CATEGORY,
            task_type=str(ctx["task_type"]),
            complexity="simple",
            messages=messages,
            gate_class=gate_class,
            compiler_pass=True,
            test_pass=test_pass,
            seed_id=str(ctx["seed_id"]),
            seed_tier=int(ctx["seed_tier"]),
            variant_idx=int(ctx["variant_idx"]),
            generator="opus-api",
            generator_model_id=GENERATOR_MODEL_ID,
            extra_meta={
                "origin": ctx["origin"],
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
    }

    rejected_count: int = processed - accepted - already_persisted_skipped;
    print("");
    print("=== gen_code_gen collect summary ===");
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("skipped as already-persisted: " + str(already_persisted_skipped));
    print("rejected: " + str(rejected_count));
    for reason in reason_counts {
        print("  - " + reason + ": " + str(reason_counts[reason]));
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
