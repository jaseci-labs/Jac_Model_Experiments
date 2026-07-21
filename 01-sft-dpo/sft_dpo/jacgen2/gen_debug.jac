"""
gen_debug.jac -- pilot generator for the `debug` category (spec.md
04-cpt-sft/docs/datagen/spec.md section 2), built on the prompt-batch
handoff protocol in batch.jac. Same two-phase shape as gen_code_gen.jac --
read that file's docstring for the general pattern, not repeated here.

This is a SCOPED-DOWN v1, not the full 15-bug-type x 21-domain fan-out from
spec.md's domain x bug coverage table: it covers 5 bug types across 4
domains (6 (domain,bug_type) pairs), 1 variant per eligible (seed,bug_type)
pair, capped at JAC_PILOT_LIMIT total pending items. Chosen subset:

    walker_traversal    -> wrong_edge_direction, walker_missing_disengage
    typed_signatures    -> type_mismatch
    core_language_basics -> syntax_typo
    sv_auth_gating      -> auth_leak (special gate-class path, see below)
    graph_query_patterns -> off_by_one_logic

model="fable" for every item per spec.md section 4.1 (debug is
Fable-assigned -- precision-critical, injected bug must actually reproduce).

Not in scope for this pilot: section 2.1's `cross_boundary_debug` (needs a
paired .sv.jac/.cl.jac project-level gate via gate.run_jac_project) and
`code_critique` (prose-only, no compiler gate) -- both are structurally
distinct extended task types, future work once this single-file
inject-one-bug template is proven end to end.

Two phases, selected by env var JAC_BATCH_PHASE:

  prepare -- for each eligible (seed, bug_type), write a pending batch item
    asking Fable to inject exactly one bug of that type into the seed's
    clean code and report a one-line symptom. Writes pending_batch.jsonl and
    stops -- no LLM call happens in this phase.

  collect -- reads responses_batch.jsonl, joins each response back to its
    pending-item context by batch_id, parses the BUGGY:/SYMPTOM: format,
    dual-gates (buggy code must actually fail/diverge, clean code must still
    pass), persists accepted examples via gate.make_example_v2 +
    common.append_jsonl, AND writes every accepted variant to
    raw_output/debug/buggy_variants.jsonl -- the declared interface Tasks 12
    (gen_dpo) and 15 (gen_trajectory's debug_session type) consume.

Gate-class caveat (spec.md section 2): `auth_leak` symptoms ("user A sees
user B's data") aren't observable in a single-process `jac run`. Those rows
use gate_class="compile_only" + test_pass=None instead of the behavioral
dual-gate -- both variants must compile, and the Fable-written symptom is
trusted as the signal. All other chosen bug types use eligibility restricted
to gate_class="behavioral" seeds (real expected_output required for the
divergence check) -- compile_only seeds have expected_output=None and can't
support a "does the buggy output differ" comparison.

Run (from repo root):
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=prepare JAC_PILOT_LIMIT=25 \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_debug.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=collect \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_debug.jac
"""

import os;
import json;
import from pathlib { Path }
import from common { require_run_tag, data_root, append_jsonl, existing_keys }
import from gate { run_jac_local, behavioral_gate, make_example_v2 }
import from batch { make_batch_id, write_pending_batch, read_response_batch, pending_path, responses_path }

glob CATEGORY: str = "debug";
glob SEED_POOL_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl";
glob GENERATOR_MODEL_ID: str = "claude-fable-5";

"""One-line descriptions lifted from spec.md section 2's bug taxonomy table,
restricted to this pilot's chosen 5 bug types."""
glob BUG_TYPES: dict[str, str] = {
    "syntax_typo": "missing semicolon, wrong keyword, mismatched brace -- compiler-level failure",
    "type_mismatch": "wrong type annotation, causes a type-check failure or silent wrong behavior",
    "wrong_edge_direction": "walker/query filters the wrong edge direction or wrong edge type",
    "walker_missing_disengage": "walker never terminates / revisits -- needs `disengage` or a visited-guard",
    "auth_leak": "endpoint missing `def:priv` (or a plain `def` left auto-registered on the API), or missing a per-user filter -- returns another user's data",
    "off_by_one_logic": "boundary error in a loop, slice, or index"
};

"""Pilot domain -> applicable-bug-type subset (spec.md section 2's domain x
bug coverage table, scoped down to this pilot's 5 chosen bug types)."""
glob DOMAIN_BUG_TYPES: dict[str, list[str]] = {
    "walker_traversal": ["wrong_edge_direction", "walker_missing_disengage"],
    "typed_signatures": ["type_mismatch"],
    "core_language_basics": ["syntax_typo"],
    "sv_auth_gating": ["auth_leak"],
    "graph_query_patterns": ["off_by_one_logic"]
};

"""Bug types using the special compile-only gate class (spec.md section 2's
gate-class caveat) instead of the behavioral dual-gate."""
glob COMPILE_ONLY_GATE_BUG_TYPES: list[str] = ["auth_leak"];


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

"""A seed is eligible for a (domain, bug_type) pair if the domain is in the
seed's `domains` list. For every bug type except auth_leak, the seed's own
gate_class must be "behavioral" -- the dual gate needs a real
expected_output to detect whether the injected bug actually changes
behavior, and compile_only seeds carry expected_output=None. auth_leak
always uses the compile-only gate class regardless of the seed's own
gate_class, so no such restriction applies there. Dedupes on
(seed_id, bug_type) in case a seed matches multiple domains mapped to the
same bug type."""
def eligible_pairs(seeds: list[dict]) -> list[dict] {
    pairs: list[dict] = [];
    seen: dict = {};
    for seed in seeds {
        domains: list = list(seed["domains"]);
        for domain in DOMAIN_BUG_TYPES {
            if domain not in domains {
                continue;
            }
            bug_types: list = list(DOMAIN_BUG_TYPES[domain]);
            for bug_type in bug_types {
                if bug_type not in COMPILE_ONLY_GATE_BUG_TYPES and str(seed["gate_class"]) != "behavioral" {
                    continue;
                }
                key: str = str(seed["seed_id"]) + "|" + bug_type;
                if key in seen {
                    continue;
                }
                seen[key] = True;
                pairs.append({"seed": seed, "bug_type": bug_type});
            }
        }
    }
    return pairs;
}

def build_prompt(bug_type: str, jac_code: str) -> str {
    description: str = BUG_TYPES[bug_type];
    instructions: str = (
        "Here is working, correct Jac code. Inject EXACTLY ONE bug of this type: "
        + description + ".\n"
        + "Do not change anything else about the code's structure or intent -- only introduce the specific bug.\n"
        + "Then write a one-line symptom description of how the bug would manifest (a compiler error message, "
        + "a wrong-output description like \"returns X instead of Y\", or a behavioral description like "
        + "\"user A can see user B's data\").\n\n"
        + "Reply in EXACTLY this format, nothing else:\n"
        + "BUGGY:\n```jac\n<the buggy code>\n```\n"
        + "SYMPTOM: <one line>\n"
    );
    return instructions + "\nOriginal correct code:\n```jac\n" + jac_code + "\n```\n";
}

"""Parses a Fable response for the BUGGY:/SYMPTOM: format. Tolerant of minor
formatting variance (leading preamble before BUGGY:, trailing text after the
symptom line) but returns {} (falsy) rather than raising on anything that
doesn't contain a fenced ```jac block followed by a SYMPTOM: line -- callers
must check `if not parsed` and reject/log rather than crash."""
def parse_debug_response(text: str) -> dict {
    if "SYMPTOM:" not in text {
        return {};
    }
    parts: list[str] = text.split("SYMPTOM:", 1);
    before: str = parts[0];
    after: str = parts[1];
    if "```jac" not in before {
        return {};
    }
    buggy_code: str = before.split("```jac", 1)[1].rsplit("```", 1)[0].strip();
    symptom_lines: list[str] = after.strip().split("\n");
    symptom: str = symptom_lines[0].strip();
    if not buggy_code or not symptom {
        return {};
    }
    return {"buggy_code": buggy_code, "symptom": symptom};
}

"""Reads raw_output/debug/buggy_variants.jsonl (schema distinct from
example_key-shaped files, so common.existing_keys doesn't apply) into a
{seed_id + "|" + bug_type: True} dict for idempotent re-runs."""
def existing_variant_keys(path: str) -> dict {
    keys: dict = {};
    p: Path = Path(path);
    if not p.exists() {
        return keys;
    }
    with open(path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            key: str = str(rec["seed_id"]) + "|" + str(rec["bug_type"]);
            keys[key] = True;
        }
    }
    return keys;
}


def run_prepare() {
    tag: str = require_run_tag();
    limit: int = int(os.environ.get("JAC_PILOT_LIMIT", "25"));

    seeds: list[dict] = load_seed_pool();
    pairs: list[dict] = eligible_pairs(seeds);

    items: list[dict] = [];
    capped: int = 0;
    for pair in pairs {
        if len(items) >= limit {
            capped += 1;
            continue;
        }
        seed: dict = dict(pair["seed"]);
        bug_type: str = str(pair["bug_type"]);
        prompt: str = build_prompt(bug_type, str(seed["jac_code"]));
        batch_id: str = make_batch_id(CATEGORY, bug_type, str(seed["seed_id"]), 0);
        items.append({
            "batch_id": batch_id,
            "model": "fable",
            "prompt": prompt,
            "category": CATEGORY,
            "task_type": bug_type,
            "variant_idx": 0,
            "seed_id": seed["seed_id"],
            "seed_tier": seed["seed_tier"],
            "origin": seed["origin"],
            "jac_code": seed["jac_code"],
            "gate_class": seed["gate_class"],
            "expected_output": seed["expected_output"],
            "domains": seed["domains"]
        });
    }

    out_path: str = pending_path(tag, CATEGORY);
    write_result: dict = write_pending_batch(out_path, items);

    print("");
    print("=== gen_debug prepare summary ===");
    print("eligible (seed,bug_type) pairs: " + str(len(pairs)));
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

    clean_path: str = data_root(tag) + "/clean_dataset/debug/examples.jsonl";
    rejected_path: str = data_root(tag) + "/rejected/debug/rejected.jsonl";
    variants_path: str = data_root(tag) + "/raw_output/debug/buggy_variants.jsonl";
    already_persisted: dict = existing_keys(clean_path);
    variant_keys: dict = existing_variant_keys(variants_path);

    processed: int = 0;
    accepted: int = 0;
    variants_written: int = 0;
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

        bug_type: str = str(ctx["task_type"]);

        if not bool(resp["ok"]) {
            reason: str = "generation failed (ok:false)";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": bug_type, "reason": reason
            });
            continue;
        }

        parsed: dict = parse_debug_response(str(resp["text"]));
        if not parsed {
            reason: str = "failed to parse BUGGY:/SYMPTOM: format";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": bug_type, "reason": reason
            });
            continue;
        }

        buggy_code: str = str(parsed["buggy_code"]);
        symptom: str = str(parsed["symptom"]);
        clean_code: str = str(ctx["jac_code"]);
        is_auth_leak: bool = bug_type in COMPILE_ONLY_GATE_BUG_TYPES;

        gate_ok: bool = False;
        gate_reason: str = "";
        if is_auth_leak {
            # spec.md section 2's gate-class caveat: auth_leak isn't
            # observable in a single-process `jac run`. Confirm both
            # variants still compile and trust the Fable-written symptom.
            (orig_code, orig_out, orig_err) = run_jac_local(clean_code);
            (bug_code_exit, bug_out, bug_err) = run_jac_local(buggy_code);
            if orig_code != 0 {
                gate_ok = False;
                gate_reason = "original/clean code failed to compile: exit " + str(orig_code) + ": " + orig_err.strip()[:200];
            } elif bug_code_exit != 0 {
                gate_ok = False;
                gate_reason = "buggy code failed to compile (auth_leak bugs should still compile): exit " + str(bug_code_exit) + ": " + bug_err.strip()[:200];
            } else {
                gate_ok = True;
                gate_reason = "ok: auth_leak compile-only gate, both variants compile, trusting Fable symptom";
            }
        } else {
            expected_output: str = str(ctx["expected_output"]);
            (orig_ok, orig_reason) = behavioral_gate(clean_code, expected_output);
            if not orig_ok {
                gate_ok = False;
                gate_reason = "original/clean seed re-check failed: " + orig_reason;
            } else {
                (bug_exit, bug_out, bug_err) = run_jac_local(buggy_code);
                bug_reproduces: bool = (bug_exit != 0) or (bug_out.strip() != expected_output.strip());
                if not bug_reproduces {
                    gate_ok = False;
                    gate_reason = "buggy code behaves identically to clean code -- bug did not reproduce";
                } else {
                    gate_ok = True;
                    gate_reason = "ok: behavioral dual-gate, buggy diverges, clean still passes";
                }
            }
        }

        if not gate_ok {
            reason: str = "gate failed: " + gate_reason;
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": bug_type, "reason": reason
            });
            continue;
        }

        gate_class_used: str = "compile_only" if is_auth_leak else "behavioral";
        # test_pass: None for compile_only (auth_leak -- never had a
        # behavioral test to run), real bool for behavioral -- computed via
        # a single post-branch ternary rather than assigned inside each
        # branch above (jac's type checker narrows a declared `any` to the
        # first branch-assigned type otherwise).
        test_pass: any = None if is_auth_leak else gate_ok;

        messages: list[dict] = [
            {"role": "user", "content": "This Jac code has a bug. Symptom: " + symptom + "\n\n```jac\n" + buggy_code + "\n```\nFix it."},
            {"role": "assistant", "content": "```jac\n" + clean_code + "\n```"}
        ];

        rec: dict = make_example_v2(
            category=CATEGORY,
            task_type=bug_type,
            complexity="simple",
            messages=messages,
            gate_class=gate_class_used,
            compiler_pass=True,
            test_pass=test_pass,
            seed_id=str(ctx["seed_id"]),
            seed_tier=int(ctx["seed_tier"]),
            variant_idx=int(ctx["variant_idx"]),
            generator="fable-api",
            generator_model_id=GENERATOR_MODEL_ID,
            extra_meta={
                "origin": ctx["origin"],
                "symptom": symptom,
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

        variant_key: str = str(ctx["seed_id"]) + "|" + bug_type;
        if variant_key not in variant_keys {
            append_jsonl(variants_path, {
                "seed_id": ctx["seed_id"],
                "bug_type": bug_type,
                "buggy_code": buggy_code,
                "symptom": symptom,
                "clean_code": clean_code
            });
            variant_keys[variant_key] = True;
            variants_written += 1;
        }
    }

    rejected_count: int = processed - accepted - already_persisted_skipped;
    print("");
    print("=== gen_debug collect summary ===");
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("buggy_variants.jsonl newly written: " + str(variants_written));
    print("skipped as already-persisted: " + str(already_persisted_skipped));
    print("rejected: " + str(rejected_count));
    for reason in reason_counts {
        print("  - " + reason + ": " + str(reason_counts[reason]));
    }
    print("wrote accepted examples to " + clean_path);
    print("wrote buggy variants to " + variants_path);
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
