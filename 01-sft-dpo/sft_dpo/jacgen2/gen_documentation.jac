"""
gen_documentation.jac -- pilot generator for the `documentation` category
(spec.md 04-cpt-sft/docs/datagen/spec.md section 6), built on the
prompt-batch handoff protocol in batch.jac. Same two-phase shape as
gen_code_gen.jac / gen_debug.jac / gen_explanation.jac -- read
gen_code_gen.jac's docstring for the general pattern, not repeated here.

`documentation` is the inverse of `explanation`: given working Jac code,
produce the documentation for it (docstring / API reference entry / module
overview). Like `explanation`, there is NO compiler gate -- prose-only, so
generation quality (Fable, per spec.md section 4.1: "factual-precision-
critical, hallucinated parameter names or invented behavior in docs is
exactly the ungated failure mode Fable is assigned to guard") plus a lexical
gate are the only defenses against hallucination. Unlike `explanation`'s
word-overlap groundedness score, this category's gate is closer to exact:
spec.md section 6 requires every backtick-quoted symbol in the response to
literally occur in the seed's code (small allowlist for Jac
keywords/builtins that don't need to appear verbatim).

This is a SCOPED-DOWN v1: one doc item per eligible code-bearing seed (not
spec.md section 6's full per-file-type fan-out), capped at JAC_PILOT_LIMIT
total pending items. All 3 of spec.md section 6's task types ARE
implemented, assigned round-robin (index % 3) across the filtered seed pool,
with an if/elif applicability check and fallback to docstring_authoring
(the always-applicable default) when a seed doesn't fit the round-robin
pick:

    docstring_authoring    -- always applicable (the fallback task type).
    api_reference_entry    -- only if `def:pub`, `def:priv`, `walker:pub`,
                                or `walker:priv` appears in the seed's
                                jac_code, or the seed's origin path contains
                                ".sv.jac" (an endpoint-bearing server file);
                                else falls back.
    module_overview        -- only if the seed's jac_code contains 2+ of
                                the archetype keywords `node `/`walker `/
                                `obj `/`edge `, or is over ~30 lines; else
                                falls back.

Any seed with non-empty jac_code is eligible -- unlike code_gen/debug, this
category does not need behavioral gating, so gate_class (behavioral vs
compile_only) is irrelevant here and is not consulted.

Two phases, selected by env var JAC_BATCH_PHASE:

  prepare -- for each eligible seed (round-robin task_type, applicability
    fallback as above), write a pending batch item asking Fable to write the
    one documentation artifact, with the prompt's CRITICAL instruction to
    backtick every referenced code symbol. Writes pending_batch.jsonl and
    stops -- no LLM call happens in this phase. model="fable" per spec.md
    section 4.1 (documentation is Fable-assigned; no compiler gate exists
    here so generation quality plus the backtick gate are the only defense
    against hallucination).

  collect -- reads responses_batch.jsonl, joins each response back to its
    pending-item context by batch_id. No BUGGY:/SYMPTOM: parsing here (per
    gen_debug.jac) and no QUESTION:/ANSWER: parsing (per gen_explanation.jac)
    -- the whole response text IS the documentation, the only structural
    check is that it's non-empty. The real gate is lexical: every
    backtick-quoted token (regex `` `([^`]+)` ``) in the response must occur
    as a literal substring in the seed's jac_code (case-sensitive), UNLESS
    it's in ALLOWLIST (Jac keywords/builtins that legitimately don't need to
    appear verbatim -- spec.md section 6's "small allowlist ... to kill
    false positives"). A response with zero backticked tokens is also
    rejected (the prompt explicitly required backticking symbols; zero
    means the instruction was ignored or nothing concrete was said).
    Persists accepted examples via gate.make_example_v2 + common.append_jsonl.

Run (from repo root):
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=prepare JAC_PILOT_LIMIT=25 \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_documentation.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=collect \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_documentation.jac
"""

import os;
import json;
import re;
import from pathlib { Path }
import from common { require_run_tag, data_root, append_jsonl, existing_keys }
import from gate { make_example_v2 }
import from batch { make_batch_id, write_pending_batch, read_response_batch, pending_path, responses_path }

glob CATEGORY: str = "documentation";
glob SEED_POOL_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl";
glob GENERATOR_MODEL_ID: str = "claude-fable-5";

glob TASK_TYPES: list[str] = [
    "docstring_authoring",
    "api_reference_entry",
    "module_overview"
];

"""One doc_type_description clause per task_type, folded into the shared
prompt template -- lifted from spec.md section 6's task_type table."""
glob DOC_TYPE_DESC: dict[str, str] = {
    "docstring_authoring": "a docstring (purpose, parameters, return value, side effects on the graph)",
    "api_reference_entry": "an API reference entry (route, auth requirement, request/response shape, error cases)",
    "module_overview": "a module-level overview (what it models, how the pieces relate, entry points)"
};

"""Jac keywords/builtins that legitimately don't need to appear verbatim in
a seed's code to be backtick-referenced in its docs (spec.md section 6's
"small allowlist ... to kill false positives")."""
glob ALLOWLIST: list[str] = [
    "root", "walker", "node", "edge", "obj", "has", "def", "str", "int",
    "float", "bool", "list", "dict", "set", "None", "True", "False",
    "self", "here", "visitor"
];

"""Archetype keywords used by the module_overview applicability heuristic
-- deliberately the space-suffixed literal forms (`node `, not `node`) so a
substring check doesn't false-positive on identifiers like `endpoint`."""
glob ARCHETYPE_KEYWORDS: list[str] = ["node ", "walker ", "obj ", "edge "];

glob ENDPOINT_MARKERS: list[str] = ["def:pub", "def:priv", "walker:pub", "walker:priv"];


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

"""documentation doesn't need behavioral gating (spec.md section 6: "seeds
from the same shared pool (code-bearing seeds only)") -- any seed with
non-empty jac_code is eligible, regardless of gate_class."""
def eligible_seeds(seeds: list[dict]) -> list[dict] {
    out: list[dict] = [];
    for seed in seeds {
        if str(seed["jac_code"]).strip() {
            out.append(seed);
        }
    }
    return out;
}

"""True if any of ENDPOINT_MARKERS appears in jac_code, or origin looks like
a server file (.sv.jac) -- the api_reference_entry applicability check."""
def looks_like_endpoint(jac_code: str, origin: str) -> bool {
    for marker in ENDPOINT_MARKERS {
        if marker in jac_code {
            return True;
        }
    }
    if ".sv.jac" in origin {
        return True;
    }
    return False;
}

"""True if jac_code contains 2+ distinct archetype keywords, or is over ~30
lines -- the module_overview applicability check (spec.md section 6's rough
multi-archetype heuristic)."""
def looks_like_module(jac_code: str) -> bool {
    hits: int = 0;
    for kw in ARCHETYPE_KEYWORDS {
        if kw in jac_code {
            hits += 1;
        }
    }
    if hits >= 2 {
        return True;
    }
    line_count: int = len(jac_code.splitlines());
    return line_count > 30;
}

"""Round-robins TASK_TYPES by pool index, then falls back to
docstring_authoring per-seed when the round-robin pick isn't applicable (see
this file's top docstring for the per-task_type rule)."""
def choose_task_type(seed: dict, idx: int) -> str {
    task_type: str = TASK_TYPES[idx % len(TASK_TYPES)];
    jac_code: str = str(seed["jac_code"]);
    origin: str = str(seed["origin"]);

    if task_type == "api_reference_entry" {
        if not looks_like_endpoint(jac_code, origin) {
            task_type = "docstring_authoring";
        }
    } elif task_type == "module_overview" {
        if not looks_like_module(jac_code) {
            task_type = "docstring_authoring";
        }
    }

    return task_type;
}

def build_prompt(task_type: str, jac_code: str) -> str {
    doc_type_description: str = DOC_TYPE_DESC[task_type];
    return (
        "Write " + doc_type_description + " for the following Jac code. CRITICAL: "
        + "wrap every code symbol you reference (function names, parameter names, "
        + "field names, route names, type names) in backticks, exactly matching how "
        + "they appear in the code. Do not invent or reference any symbol that isn't "
        + "actually present in the code below. Be accurate and concise."
        + "\n\nReply with ONLY the documentation text, no preamble, no code repetition."
        + "\n\n```jac\n" + jac_code + "\n```\n"
    );
}


def run_prepare() {
    tag: str = require_run_tag();
    limit: int = int(os.environ.get("JAC_PILOT_LIMIT", "25"));

    all_seeds: list[dict] = load_seed_pool();
    filtered: list[dict] = eligible_seeds(all_seeds);
    skipped_no_code: int = len(all_seeds) - len(filtered);

    items: list[dict] = [];
    capped: int = 0;
    task_type_counts: dict[str, int] = {};

    for idx in range(len(filtered)) {
        if len(items) >= limit {
            capped += 1;
            continue;
        }
        seed: dict = dict(filtered[idx]);
        task_type: str = choose_task_type(seed, idx);
        jac_code: str = str(seed["jac_code"]);

        prompt: str = build_prompt(task_type, jac_code);
        seed_id: str = str(seed["seed_id"]);
        batch_id: str = make_batch_id(CATEGORY, task_type, seed_id, 0);

        items.append({
            "batch_id": batch_id,
            "model": "fable",
            "prompt": prompt,
            "category": CATEGORY,
            "task_type": task_type,
            "variant_idx": 0,
            "seed_id": seed_id,
            "seed_tier": seed["seed_tier"],
            "origin": seed["origin"],
            "jac_code": jac_code
        });
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1;
    }

    out_path: str = pending_path(tag, CATEGORY);
    write_result: dict = write_pending_batch(out_path, items);

    print("");
    print("=== gen_documentation prepare summary ===");
    print("seeds in pool: " + str(len(all_seeds)));
    print("skipped (empty jac_code): " + str(skipped_no_code));
    print("eligible (code-bearing) seeds: " + str(len(filtered)));
    print("candidate items this run: " + str(len(items)) + " (JAC_PILOT_LIMIT=" + str(limit) + ")");
    print("skipped due to cap: " + str(capped));
    print("task_type distribution (post-fallback): " + str(task_type_counts));
    print("newly written to pending batch: " + str(write_result["written"]));
    print("skipped as already-present in pending batch: " + str(write_result["skipped"]));
    print("wrote " + out_path);
}

"""Every backtick-quoted token in `text`, in order of appearance, duplicates
kept (the caller de-dupes for reporting)."""
def extract_backticked_tokens(text: str) -> list[str] {
    found: list[str] = [];
    for tok in re.findall(r"`([^`]+)`", text) {
        found.append(str(tok));
    }
    return found;
}

"""The backtick symbol-existence gate (spec.md section 6): every backticked
token must occur as a literal substring of jac_code, unless it's in
ALLOWLIST. Zero backticked tokens is also a reject (the prompt required
backticking; none means the instruction was ignored). Returns
(ok, reason, distinct_tokens, distinct_missing) so the caller can persist
both the full token list and a clean missing-symbol list."""
def backtick_gate(response_text: str, jac_code: str) -> tuple[bool, str, list[str], list[str]] {
    tokens: list[str] = extract_backticked_tokens(response_text);
    distinct_tokens: list[str] = [];
    for tok in tokens {
        if tok not in distinct_tokens {
            distinct_tokens.append(tok);
        }
    }

    if not distinct_tokens {
        return (False, "zero backticked symbols in response -- prompt required backticking", [], []);
    }

    missing: list[str] = [];
    for tok in distinct_tokens {
        if tok in jac_code {
            continue;
        }
        if tok in ALLOWLIST {
            continue;
        }
        missing.append(tok);
    }

    if missing {
        return (False, "missing symbol(s) not found in seed code: " + ", ".join(missing), distinct_tokens, missing);
    }

    return (True, "ok", distinct_tokens, []);
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

    clean_path: str = data_root(tag) + "/clean_dataset/documentation/examples.jsonl";
    rejected_path: str = data_root(tag) + "/rejected/documentation/rejected.jsonl";
    already_persisted: dict = existing_keys(clean_path);

    processed: int = 0;
    accepted: int = 0;
    already_persisted_skipped: int = 0;
    reason_counts: dict[str, int] = {};
    zero_backtick_rejections: int = 0;
    missing_symbol_rejections: int = 0;

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

        doc_text: str = str(resp["text"]).strip();
        if not doc_text {
            reason: str = "empty documentation text";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": task_type, "reason": reason
            });
            continue;
        }

        jac_code: str = str(ctx["jac_code"]);
        (gate_ok, gate_reason, distinct_tokens, missing) = backtick_gate(doc_text, jac_code);

        if not gate_ok {
            reason_counts[gate_reason] = reason_counts.get(gate_reason, 0) + 1;
            if not distinct_tokens {
                zero_backtick_rejections += 1;
            } else {
                missing_symbol_rejections += 1;
            }
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": task_type, "reason": gate_reason, "missing_symbols": missing
            });
            continue;
        }

        messages: list[dict] = [
            {"role": "user", "content": "Document this Jac code:\n\n```jac\n" + jac_code + "\n```"},
            {"role": "assistant", "content": doc_text}
        ];

        rec: dict = make_example_v2(
            category=CATEGORY,
            task_type=task_type,
            complexity="simple",
            messages=messages,
            gate_class="prose_lexical",
            compiler_pass=True,
            test_pass=None,
            seed_id=str(ctx["seed_id"]),
            seed_tier=int(ctx["seed_tier"]),
            variant_idx=int(ctx["variant_idx"]),
            generator="fable-api",
            generator_model_id=GENERATOR_MODEL_ID,
            extra_meta={
                "origin": ctx["origin"],
                "backticked_symbols": distinct_tokens,
                "compiler_pass_note": "documentation is prose-only, no compiler gate exists -- "
                    + "compiler_pass=True is vacuous here, gate_class=prose_lexical (the "
                    + "backtick symbol-existence check above) is the real accept/reject signal",
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
    print("=== gen_documentation collect summary ===");
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("skipped as already-persisted: " + str(already_persisted_skipped));
    print("rejected: " + str(rejected_count));
    print("  - zero-backtick-token rejections: " + str(zero_backtick_rejections));
    print("  - missing-symbol rejections: " + str(missing_symbol_rejections));
    print("reason breakdown:");
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
