"""
gen_seed_review.jac — tier 4 review pass: Fable reviews every parse-gated
scrape candidate (seed_scrape.jac output) against the ponytail code-quality
rubric, pins expected_output for the ones it accepts, and merges accepted
records into the shared seed pool.

Consumes:
  model-experiments/04-cpt-sft/dataset/shared/tier4_candidates.jsonl (seed_scrape.jac)

Produces:
  model-experiments/04-cpt-sft/dataset/shared/tier4_reviewed.jsonl
  model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl (tier-4 accepted
    records appended, then the whole pool re-deduped exactly as seed_pool.jac
    does)

Decontamination against eval holdouts (spec.md 8.4's "same decontamination
requirement as every other tier") is DEFERRED here to decontam_v2.jac (Task
16), which is explicitly designed to do this at scale with a real shingle
index. model-experiments/01-sft-dpo/dataset/eval_holdout/*.jsonl currently
holds Python functions, not Jac code, so a same-tier rouge_l comparison
here would not even be a meaningful signal yet -- building a bespoke
one-off decontam pass for that mismatch would be over-engineering for what
this task needs. Noted loudly at runtime below.

Run (against the full committed repo list, once a real API key exists):
    JAC_RUN_TAG=fresh JAC_CALL_BUDGET=<n> \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_seed_review.jac
"""

import json;
import hashlib;
import from pathlib { Path }
import from gate { run_jac_local }
import from llm { call_with_retry }
import from dedup2 { rouge_l }

glob REPO_ROOT: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio";
glob OUT_DIR: str = REPO_ROOT + "/model-experiments/04-cpt-sft/dataset/shared";
glob CANDIDATES_PATH: str = OUT_DIR + "/tier4_candidates.jsonl";
glob REVIEWED_PATH: str = OUT_DIR + "/tier4_reviewed.jsonl";
glob SEED_POOL_PATH: str = OUT_DIR + "/seed_pool.jsonl";
glob NEAR_DUP_THRESHOLD: float = 0.85;

glob REVIEW_PROMPT_TEMPLATE: str = (
    "You are a strict code reviewer for Jac training data. Reject code that:\n"
    "(1) reinvents Jac stdlib or a native language feature,\n"
    "(2) has speculative abstraction -- interfaces with one implementation, config for constants that never vary,\n"
    "(3) is clearly inefficient (O(n^2) where O(n) is trivial, redundant recomputation, dead code paths),\n"
    "(4) solves a graph-shaped problem in Python-shaped Jac (dict/list graph simulation instead of nodes/edges/walkers).\n"
    "Otherwise accept.\n"
    "Reply with EXACTLY one line in the form:\n"
    "ACCEPT: <one-line reason>\n"
    "or\n"
    "REJECT: <one-line reason>\n"
    "\n"
    "```jac\n"
    "{code}\n"
    "```\n"
);


def build_review_prompt(code: str) -> str {
    return REVIEW_PROMPT_TEMPLATE.replace("{code}", code);
}


"""Parse a Fable review response into {accepted, reason}. Never raises --
anything that isn't a clean ACCEPT:/REJECT: prefix is treated as a reject
with an explicit 'unparseable' reason, same failure-shape as a failed
call."""
def parse_review_response(text: str) -> dict {
    stripped: str = text.strip();
    if stripped.startswith("ACCEPT:") {
        return {"accepted": True, "reason": stripped[len("ACCEPT:"):].strip()};
    }
    if stripped.startswith("REJECT:") {
        return {"accepted": False, "reason": stripped[len("REJECT:"):].strip()};
    }
    return {"accepted": False, "reason": "review_unparseable: " + stripped[:100]};
}


"""Review one candidate's code with Fable. Never raises on call failure --
call_with_retry already returns a graceful ok:False dict for a failed
call (e.g. no API key); this maps that into the same {accepted, reason}
shape parse_review_response uses. call_with_retry ALSO hard-raises
RuntimeError once JAC_CALL_BUDGET is exhausted (llm.jac's intentional
circuit breaker, verified by check_llm_wrap.jac) -- for a per-candidate
batch review pass the sensible behavior is to record the remaining
candidates as un-reviewed-due-to-budget and keep going, not crash the
whole batch, so that raise is caught here too and folded into the same
graceful reject path."""
def review_candidate(code: str) -> dict {
    prompt: str = build_review_prompt(code);
    try {
        result: dict = call_with_retry("gen_text_fable", prompt, "fable");
    } except Exception as e {
        return {"accepted": False, "reason": "review_call_failed: " + str(e)};
    }
    if not result["ok"] {
        return {"accepted": False, "reason": "review_call_failed: " + str(result["err"])};
    }
    return parse_review_response(str(result["text"]));
}


"""Already passed `jac check -p` in seed_scrape.jac's parse gate -- a
candidate reaching this function always has that guarantee, so a non-zero
run here just means it doesn't run standalone (e.g. a client component or
endpoint snippet), not that it's broken."""
def gate_and_pin(code: str) -> dict {
    (rc, out, err) = run_jac_local(code);
    if rc == 0 {
        return {"gate_class": "behavioral", "expected_output": out.strip()};
    }
    return {"gate_class": "compile_only", "expected_output": None};
}


"""exact-sha1 dedup then bucketed near-dup dedup, over the FULL merged pool
-- identical two-stage algorithm seed_pool.jac's with-entry block uses,
re-run here because tier-4 records can duplicate existing tier 1/2/3/5
seeds too."""
def dedup_pool(records: list[dict]) -> list[dict] {
    seen_ids: dict[str, bool] = {};
    stage1: list[dict] = [];
    for rec in records {
        sid: str = str(rec["seed_id"]);
        if sid in seen_ids {
            continue;
        }
        seen_ids[sid] = True;
        stage1.append(rec);
    }

    buckets: dict[str, list[dict]] = {};
    final: list[dict] = [];
    for rec in stage1 {
        code: str = str(rec["jac_code"]);
        bucket_key: str = str(len(code) // 200) + "|" + code[:20];
        if bucket_key not in buckets {
            buckets[bucket_key] = [];
        }
        is_dup: bool = False;
        for kept in buckets[bucket_key] {
            if rouge_l(code, str(kept["jac_code"])) >= NEAR_DUP_THRESHOLD {
                is_dup = True;
                break;
            }
        }
        if is_dup {
            continue;
        }
        buckets[bucket_key].append(rec);
        final.append(rec);
    }
    return final;
}


def load_jsonl(path: str) -> list[dict] {
    p: Path = Path(path);
    if not p.exists() {
        return [];
    }
    with open(path, "r") as f {
        lines: list[str] = f.readlines();
    }
    records: list[dict] = [];
    for line in lines {
        s: str = line.strip();
        if s {
            records.append(json.loads(s));
        }
    }
    return records;
}


with entry:__main__ {
    print(
        "NOTE: decontam-against-eval-holdouts (spec.md 8.4) is deferred to "
        + "decontam_v2.jac (Task 16) -- eval_holdout/*.jsonl currently holds "
        + "Python functions, not Jac code, so a same-tier rouge_l comparison "
        + "here would not be a meaningful signal yet; Task 16 builds the real "
        + "shingle-index decontam pass at scale."
    );

    candidates: list[dict] = load_jsonl(CANDIDATES_PATH);
    print("=== gen_seed_review: " + str(len(candidates)) + " tier-4 candidates to review ===");

    reviewed: list[dict] = [];
    accept_count: int = 0;
    reject_count: int = 0;
    call_failed_count: int = 0;
    unparseable_count: int = 0;

    for cand in candidates {
        code: str = str(cand["jac_code"]);
        verdict: dict = review_candidate(code);
        if not bool(verdict["accepted"]) {
            reject_count += 1;
            reason: str = str(verdict["reason"]);
            if reason.startswith("review_call_failed:") {
                call_failed_count += 1;
            }
            if reason.startswith("review_unparseable:") {
                unparseable_count += 1;
            }
            continue;
        }

        accept_count += 1;
        gate: dict = gate_and_pin(code);
        origin: str = "scrape:" + str(cand["repo"]) + "@" + str(cand["sha"])[:8] + ":" + str(cand["path"]);
        seed_id: str = hashlib.sha1((origin + "|" + code).encode("utf-8")).hexdigest()[:12];
        rec: dict = {
            "seed_id": seed_id,
            "seed_tier": 4,
            "origin": origin,
            "jac_code": code,
            "gate_class": gate["gate_class"],
            "expected_output": gate["expected_output"],
            "domains": ["github_scrape"],
            "doc_chunk": None,
            "review_note": str(verdict["reason"])
        };
        reviewed.append(rec);
    }

    with open(REVIEWED_PATH, "w") as out_f {
        for rec in reviewed {
            out_f.write(json.dumps(rec) + "\n");
        }
    }
    print("wrote " + str(len(reviewed)) + " Fable-accepted candidates -> " + REVIEWED_PATH);

    existing_pool: list[dict] = load_jsonl(SEED_POOL_PATH);
    merged: list[dict] = existing_pool + reviewed;
    deduped: list[dict] = dedup_pool(merged);

    deduped_ids: dict[str, bool] = {};
    for rec in deduped {
        deduped_ids[str(rec["seed_id"])] = True;
    }
    tier4_survived: int = 0;
    for rec in reviewed {
        if str(rec["seed_id"]) in deduped_ids {
            tier4_survived += 1;
        }
    }

    with open(SEED_POOL_PATH, "w") as out_f {
        for rec in deduped {
            out_f.write(json.dumps(rec) + "\n");
        }
    }
    print(
        "merged " + str(len(existing_pool)) + " existing + " + str(len(reviewed))
        + " tier-4 Fable-accepted -> " + str(len(deduped)) + " total after re-dedup ("
        + str(tier4_survived) + "/" + str(len(reviewed)) + " tier-4 records survived dedup) -> "
        + SEED_POOL_PATH
    );

    print("");
    print("=== gen_seed_review summary ===");
    print("candidates reviewed: " + str(len(candidates)));
    print("Fable accept: " + str(accept_count));
    print(
        "Fable reject: " + str(reject_count)
        + " (call_failed=" + str(call_failed_count)
        + ", unparseable=" + str(unparseable_count)
        + ", rubric_reject=" + str(reject_count - call_failed_count - unparseable_count) + ")"
    );
    print("final tier-4 accepted into pool: " + str(tier4_survived));
}
