"""
gen_dpo.jac -- pilot generator for the `dpo` preference-pair dataset
(model-experiments/04-cpt-sft/docs/dpo-plan.md), built on the same
prompt-batch handoff protocol in batch.jac used by gen_code_gen.jac and
gen_debug.jac (read those files' docstrings for the general prepare ->
[orchestrator] -> collect shape, not repeated here).

dpo-plan.md defines 5 preference axes. This file splits them into two parts
with two very different generation stories:

  Part A ("free" phase) -- correctness (dpo-plan.md 2.3), auth_security
    (2.4), typing (2.5). These are assembled directly from gen_debug.jac's
    already-persisted raw_output/debug/buggy_variants.jsonl -- no LLM call
    (Task 9's dual-gate already proved buggy_code diverges/fails and
    clean_code passes), but `compiler_pass_rejected` is still verified for
    real here via a fresh gate.run_jac_local(buggy_code) call rather than
    inferred from bug_type -- a behavioral-class bug can easily compile fine
    and only diverge at runtime, so bug_type category membership alone can't
    tell us whether a given buggy_code compiles. Runs synchronously, writes
    straight to releases/dpo_train_free.jsonl. No JAC_BATCH_PHASE handoff
    involved.

  Part B (prepare/collect phases) -- idiomatic (2.1, flagship, 40% weight)
    and graph_native (2.2, 15% weight). These need a genuinely new
    generation (a less-idiomatic / object-shoehorned rewrite of an idiomatic
    seed), so they go through the normal batch handoff: prepare writes
    pending items (carrying expected_output/gate_class from the seed, same
    as gen_debug.jac/gen_code_gen.jac), an external orchestrator dispatches
    Fable against them, collect reads responses, gates (rejected must
    compile; chosen must compile AND, for behavioral-gate_class seeds, pass
    the same final sanity behavioral re-check gen_debug.jac/gen_code_gen.jac
    use on already-proven seed code), and writes to releases/dpo_train_llm.jsonl.

Two output files, not one, so Task 17's merge step knows exactly which rows
came from which path: dpo_train_free.jsonl (Part A, no LLM) and
dpo_train_llm.jsonl (Part B, Fable-generated). Both share the dpo-plan.md
section 6 output shape and get concatenated into the real dpo_train.jsonl
release file in Task 17, not here.

KNOWN GAPS -- disclosed, not fixed:
  - buggy_variants.jsonl (this run's Task 9 output) has only 4 real records,
    all bug_type in {auth_leak, syntax_typo} -- no type_mismatch record
    exists yet, so the `typing` axis is EMPTY this run (0 pairs). This is
    expected and honest, not a bug in this file: typing pairs will appear
    once gen_debug.jac's typed_signatures/type_mismatch pairing produces
    output in a future run.
  - idiomatic/graph_native (Part B) have no automated idiom-quality gate --
    the only check is "does the rejected variant still compile" (gate.
    run_jac_local exit 0). Whether it's actually LESS idiomatic than the
    chosen side is trusted from Fable's framing in the prompt, same
    disclosed-limitation pattern as gen_debug.jac's auth_leak compile-only
    gate (no independent critique pass). A future pass could add a second
    Fable critique phase, same deferred-for-now reasoning as gen_debug.jac's
    KNOWN GAP note.
  - Part B's `prompt` field is a fixed per-axis instruction (dpo-plan.md's
    output shape wants "prompt": "string" but doesn't mandate content) --
    seeds in seed_pool.jsonl carry no natural-language task description
    (unlike gen_code_gen.jac's LLM-authored instructions), so there's no
    honest per-seed prompt to draw from. Using one shared instruction per
    axis rather than fabricating a fake per-seed description.

Run (from repo root):
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=free \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_dpo.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=prepare JAC_PILOT_LIMIT=16 \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_dpo.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=collect \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_dpo.jac
"""

import os;
import json;
import from pathlib { Path }
import from datetime { datetime }
import from common { require_run_tag, data_root, append_jsonl, existing_keys, example_key }
import from gate { run_jac_local, behavioral_gate }
import from batch { make_batch_id, write_pending_batch, read_response_batch, pending_path, responses_path }

glob CATEGORY: str = "dpo";
glob SEED_POOL_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl";

"""dpo-plan.md 2.2's graph-shaped domains -- a seed is eligible for the
graph_native axis if any of these appear in its `domains` list."""
glob GRAPH_DOMAINS: list[str] = ["node_edge_definition", "walker_traversal", "graph_query_patterns"];

glob AXIS_INSTRUCTION: dict[str, str] = {
    "idiomatic": "use plain dict/list structures instead of node/edge/walker OSP primitives where the "
        + "problem is graph-shaped, and generally favor Python-style patterns over Jac-native ones",
    "graph_native": "specifically replace node/edge/walker graph traversal with a dict-of-lists adjacency "
        + "representation and plain function-based traversal, even though the problem is inherently graph-shaped"
};

"""Fixed per-axis prompt for Part B pairs -- see this file's top-docstring
KNOWN GAPS note on why these aren't per-seed."""
glob AXIS_DPO_PROMPT: dict[str, str] = {
    "idiomatic": "Write idiomatic, graph-native Jac code for this task -- use node/edge/walker OSP "
        + "primitives where the problem is graph-shaped, not plain Python-style functions and dicts.",
    "graph_native": "Write graph-native Jac code for this task -- use node/edge/walker traversal, not a "
        + "dict-of-lists adjacency representation, since the problem is inherently graph-shaped."
};


# --------------------------------------------------------------------------
# Part A -- free axes (correctness, auth_security, typing), no LLM call.
# --------------------------------------------------------------------------

def load_buggy_variants(tag: str) -> list[dict] {
    path: str = data_root(tag) + "/raw_output/debug/buggy_variants.jsonl";
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

def free_axis_prompt(axis: str, symptom: str) -> str {
    if axis == "correctness" {
        return "Write correct Jac code that avoids this bug: " + symptom;
    } elif axis == "auth_security" {
        return "Write secure Jac code with correct auth gating that avoids this bug: " + symptom;
    } else {
        return "Write well-typed Jac code that avoids this bug: " + symptom;
    }
}

"""For a buggy_variants.jsonl record, which axes apply. correctness always
applies (dpo-plan.md 2.3: every buggy/clean pair is a correct-vs-wrong pair
regardless of bug_type). auth_security only for bug_type=="auth_leak" (2.4).
typing only for bug_type=="type_mismatch" (2.5)."""
def axes_for_bug_type(bug_type: str) -> list[str] {
    axes: list[str] = ["correctness"];
    if bug_type == "auth_leak" {
        axes.append("auth_security");
    }
    if bug_type == "type_mismatch" {
        axes.append("typing");
    }
    return axes;
}

def run_free() {
    tag: str = require_run_tag();
    variants: list[dict] = load_buggy_variants(tag);

    out_path: str = data_root(tag) + "/releases/dpo_train_free.jsonl";
    already: dict = existing_keys(out_path);

    counts: dict[str, int] = {"correctness": 0, "auth_security": 0, "typing": 0};
    written: int = 0;
    skipped: int = 0;

    for variant in variants {
        seed_id: str = str(variant["seed_id"]);
        bug_type: str = str(variant["bug_type"]);
        buggy_code: str = str(variant["buggy_code"]);
        clean_code: str = str(variant["clean_code"]);
        symptom: str = str(variant["symptom"]);
        # Real compile check, not a static bug_type lookup -- a behavioral-
        # class bug (e.g. wrong_edge_direction, off_by_one_logic) can easily
        # compile fine and only diverge at runtime, so bug_type category
        # membership alone can't tell us whether THIS buggy_code compiles.
        (rej_exit, rej_out, rej_err) = run_jac_local(buggy_code);
        rejected_compiles: bool = (rej_exit == 0);

        for axis in axes_for_bug_type(bug_type) {
            rec_id: str = example_key(seed_id, axis, 0);
            if rec_id in already {
                skipped += 1;
                continue;
            }
            rec: dict = {
                "id": rec_id,
                "axis": axis,
                "prompt": free_axis_prompt(axis, symptom),
                "chosen": clean_code,
                "rejected": buggy_code,
                "run_tag": tag,
                "generator": "fable-api",
                "generation_date": datetime.now().isoformat(),
                "compiler_pass_chosen": True,
                "compiler_pass_rejected": rejected_compiles
            };
            append_jsonl(out_path, rec);
            already[rec_id] = True;
            counts[axis] = counts.get(axis, 0) + 1;
            written += 1;
        }
    }

    print("");
    print("=== gen_dpo free (Part A) summary ===");
    print("buggy_variants.jsonl records read: " + str(len(variants)));
    print("newly written: " + str(written) + "  (skipped as already-present: " + str(skipped) + ")");
    for axis in counts {
        print("  - " + axis + ": " + str(counts[axis]));
    }
    print("wrote " + out_path);
}


# --------------------------------------------------------------------------
# Part B -- idiomatic / graph_native axes, LLM batch handoff.
# --------------------------------------------------------------------------

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

"""idiomatic axis: any seed whose own gate_class is "behavioral" (dpo-plan.md
2.1 reuses idiomatic seeds broadly, not restricted to graph-shaped ones)."""
def eligible_idiomatic(seeds: list[dict]) -> list[dict] {
    out: list[dict] = [];
    for seed in seeds {
        if str(seed["gate_class"]) == "behavioral" {
            out.append(seed);
        }
    }
    return out;
}

"""graph_native axis: seeds whose domains intersect GRAPH_DOMAINS
(dpo-plan.md 2.2 -- graph-shaped seeds specifically)."""
def eligible_graph_native(seeds: list[dict]) -> list[dict] {
    out: list[dict] = [];
    for seed in seeds {
        domains: list = list(seed["domains"]);
        for d in GRAPH_DOMAINS {
            if d in domains {
                out.append(seed);
                break;
            }
        }
    }
    return out;
}

def build_llm_prompt(axis: str, jac_code: str) -> str {
    instruction: str = AXIS_INSTRUCTION[axis];
    return (
        "Here is idiomatic, graph-native Jac code. Rewrite it to be functionally equivalent but LESS "
        + "idiomatic -- " + instruction + ". The rewritten version must still compile and run, just in a "
        + "more Python-shaped / less graph-native style. Reply with ONLY the rewritten Jac code, no "
        + "explanation, no markdown fence.\n\n"
        + "Original idiomatic code:\n```jac\n" + jac_code + "\n```\n"
    );
}

def run_prepare() {
    tag: str = require_run_tag();
    limit: int = int(os.environ.get("JAC_PILOT_LIMIT", "16"));
    idiomatic_cap: int = limit // 2;
    graph_native_cap: int = limit - idiomatic_cap;

    seeds: list[dict] = load_seed_pool();
    idiomatic_seeds: list[dict] = eligible_idiomatic(seeds);
    graph_native_seeds: list[dict] = eligible_graph_native(seeds);

    items: list[dict] = [];
    idiomatic_written: int = 0;
    idiomatic_capped: int = 0;
    for seed in idiomatic_seeds {
        if idiomatic_written >= idiomatic_cap {
            idiomatic_capped += 1;
            continue;
        }
        s: dict = dict(seed);
        batch_id: str = make_batch_id(CATEGORY, "idiomatic", str(s["seed_id"]), 0);
        items.append({
            "batch_id": batch_id,
            "model": "fable",
            "prompt": build_llm_prompt("idiomatic", str(s["jac_code"])),
            "category": CATEGORY,
            "task_type": "idiomatic",
            "variant_idx": 0,
            "seed_id": s["seed_id"],
            "seed_tier": s["seed_tier"],
            "origin": s["origin"],
            "jac_code": s["jac_code"],
            "gate_class": s["gate_class"],
            "expected_output": s["expected_output"]
        });
        idiomatic_written += 1;
    }

    graph_native_written: int = 0;
    graph_native_capped: int = 0;
    for seed in graph_native_seeds {
        if graph_native_written >= graph_native_cap {
            graph_native_capped += 1;
            continue;
        }
        s: dict = dict(seed);
        batch_id: str = make_batch_id(CATEGORY, "graph_native", str(s["seed_id"]), 0);
        items.append({
            "batch_id": batch_id,
            "model": "fable",
            "prompt": build_llm_prompt("graph_native", str(s["jac_code"])),
            "category": CATEGORY,
            "task_type": "graph_native",
            "variant_idx": 0,
            "seed_id": s["seed_id"],
            "seed_tier": s["seed_tier"],
            "origin": s["origin"],
            "jac_code": s["jac_code"],
            "gate_class": s["gate_class"],
            "expected_output": s["expected_output"]
        });
        graph_native_written += 1;
    }

    out_path: str = pending_path(tag, CATEGORY);
    write_result: dict = write_pending_batch(out_path, items);

    print("");
    print("=== gen_dpo prepare (Part B) summary ===");
    print("eligible idiomatic seeds: " + str(len(idiomatic_seeds)) + "  (JAC_PILOT_LIMIT/2=" + str(idiomatic_cap) + ", capped " + str(idiomatic_capped) + ")");
    print("eligible graph_native seeds: " + str(len(graph_native_seeds)) + "  (cap=" + str(graph_native_cap) + ", capped " + str(graph_native_capped) + ")");
    print("candidate items this run: " + str(len(items)) + " (JAC_PILOT_LIMIT=" + str(limit) + ")");
    print("newly written to pending batch: " + str(write_result["written"]));
    print("skipped as already-present in pending batch: " + str(write_result["skipped"]));
    print("wrote " + out_path);
}

"""Strips an accidental markdown code fence from a Fable response that was
asked to reply with ONLY code. Tries the ```jac-tagged fence first, falls
back to a bare ``` fence, and returns the text unchanged (stripped) if
neither is present -- the response was supposed to be bare code anyway."""
def strip_code_fence(text: str) -> str {
    t: str = text.strip();
    if "```jac" in t {
        return t.split("```jac", 1)[1].rsplit("```", 1)[0].strip();
    }
    if "```" in t {
        parts: list[str] = t.split("```");
        if len(parts) >= 3 {
            return parts[1].strip();
        }
    }
    return t;
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

    llm_path: str = data_root(tag) + "/releases/dpo_train_llm.jsonl";
    rejected_path: str = data_root(tag) + "/rejected/dpo/rejected.jsonl";
    already_persisted: dict = existing_keys(llm_path);

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

        axis: str = str(ctx["task_type"]);

        if not bool(resp["ok"]) {
            reason: str = "generation failed (ok:false)";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "axis": axis, "reason": reason
            });
            continue;
        }

        rejected_code: str = strip_code_fence(str(resp["text"]));
        if not rejected_code {
            reason: str = "empty rejected code after fence-strip";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "axis": axis, "reason": reason
            });
            continue;
        }

        jac_code: str = str(ctx["jac_code"]);
        # .get() rather than ctx["gate_class"]/ctx["expected_output"] --
        # pending items written before this field was added to run_prepare
        # (already on disk, idempotent-skipped pairs) won't carry these
        # keys; falls back to the compile-only check below for those, same
        # as before this fix.
        seed_gate_class: str | None = ctx.get("gate_class", None);
        seed_expected_output: str | None = ctx.get("expected_output", None);

        # Gate: both chosen (the seed's own idiomatic code) and rejected
        # (Fable's less-idiomatic rewrite) must compile -- see this file's
        # top-docstring KNOWN GAPS note on why idiom-quality itself isn't
        # independently gated.
        (chosen_exit, chosen_out, chosen_err) = run_jac_local(jac_code);
        if chosen_exit != 0 {
            reason: str = "chosen (seed's own code) failed to compile: exit " + str(chosen_exit) + ": " + chosen_err.strip()[:200];
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "axis": axis, "reason": reason
            });
            continue;
        }
        # Final sanity re-check on the chosen side (dpo-plan.md 5: "every
        # chosen example must independently pass the same gate class as its
        # SFT counterpart... behavioral where observable") -- same pattern
        # gen_debug.jac/gen_code_gen.jac use to re-verify already-proven seed
        # code. compile_only seeds (or pending items missing the field,
        # see the .get() note above) keep the exit-0-only check as
        # sufficient, matching gen_code_gen.jac's compile_only handling.
        if seed_gate_class == "behavioral" and seed_expected_output {
            (chosen_behaves_ok, chosen_behave_reason) = behavioral_gate(jac_code, seed_expected_output);
            if not chosen_behaves_ok {
                reason: str = "chosen (seed's own code) failed behavioral re-check: " + chosen_behave_reason;
                reason_counts[reason] = reason_counts.get(reason, 0) + 1;
                append_jsonl(rejected_path, {
                    "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                    "axis": axis, "reason": reason
                });
                continue;
            }
        }
        (rej_exit, rej_out, rej_err) = run_jac_local(rejected_code);
        if rej_exit != 0 {
            reason: str = "rejected variant failed to compile: exit " + str(rej_exit) + ": " + rej_err.strip()[:200];
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "axis": axis, "reason": reason
            });
            continue;
        }

        rec_id: str = example_key(str(ctx["seed_id"]), axis, 0);
        rec: dict = {
            "id": rec_id,
            "axis": axis,
            "prompt": AXIS_DPO_PROMPT[axis],
            "chosen": jac_code,
            "rejected": rejected_code,
            "run_tag": tag,
            "generator": "fable-api",
            "generation_date": datetime.now().isoformat(),
            "compiler_pass_chosen": True,
            "compiler_pass_rejected": True
        };
        if rec_id in already_persisted {
            already_persisted_skipped += 1;
            continue;
        }

        append_jsonl(llm_path, rec);
        already_persisted[rec_id] = True;
        accepted += 1;
    }

    rejected_count: int = processed - accepted - already_persisted_skipped;
    print("");
    print("=== gen_dpo collect (Part B) summary ===");
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("skipped as already-persisted: " + str(already_persisted_skipped));
    print("rejected: " + str(rejected_count));
    for reason in reason_counts {
        print("  - " + reason + ": " + str(reason_counts[reason]));
    }
    print("wrote accepted pairs to " + llm_path);
}


with entry:__main__ {
    phase: str | None = os.environ.get("JAC_BATCH_PHASE");
    if phase == "free" {
        run_free();
    } elif phase == "prepare" {
        run_prepare();
    } elif phase == "collect" {
        run_collect();
    } else {
        raise ValueError(
            "JAC_BATCH_PHASE must be set to 'free', 'prepare', or 'collect', got: " + str(phase)
        );
    }
}
