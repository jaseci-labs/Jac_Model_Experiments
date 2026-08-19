"""
prepare_batches.jac -- Phase A ("prepare") of the batch-handoff pipeline for
07-nitin-ds-new-sft, covering BOTH generation tracks off the same clean pool.
Logic is byte-for-byte 06-nitin-ds-sft's; only paths and the corpus-size-derived
shard count differ (20 shards over 6781 rows here vs 16 over 5474 in 06).

There is no ANTHROPIC_API_KEY in this environment, so every "LLM call" in this
project goes through the file handoff jacgen2 established
(01-sft-dpo/sft_dpo/jacgen2/batch.jac; rationale in
04-cpt-sft/docs/reports/2026-07-task18-full-run-report.md section 2):

    prepare (this script, deterministic, no LLM)
        -> pending_batch_<track>_<NN>.jsonl
    dispatched Claude Code Agent (Opus for sft, Fable for dpo)
        -> responses_batch_<track>_<NN>.jsonl
    collect (collect.jac, gates + persists)
        -> dataset/sft_train.jsonl / dataset/dpo_train.jsonl

Two INDEPENDENT tracks, both reading dataset/candidate_pool.jsonl directly
(CONTEXT_BRIEF.md section 2 -- neither depends on the other):

  sft -- reverse-instruction authoring, ALL 6781 candidate rows, model="opus"
         (bulk/volume tier). The agent writes an instruction only; it never
         touches the Jac, which is carried through verbatim as the assistant
         turn.

  dpo -- correctness-axis pair construction, a seeded subset, model="fable"
         (precision tier). chosen = the pool's jac_code unchanged;
         rejected = the same function with ONE subtly-introduced LOGIC bug
         that must still compile.

Sharding: a single Agent turn cannot reverse-author thousands of items, so each
track is split into contiguous shards small enough for one dispatch. Shard
index lives in the filename so a failed shard is re-runnable in isolation
(workflow.md Stage 1).

Run (from repo root, absolute venv jac -- bare `jac` resolves to a different,
stale venv, see gate.jac's header and workflow.md's conventions):

    JAC_BATCH_TRACK=sft /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/07-nitin-ds-new-sft/scripts/prepare_batches.jac
    JAC_BATCH_TRACK=dpo /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/07-nitin-ds-new-sft/scripts/prepare_batches.jac

NOTE on the inlined helpers: `make_batch_id`, `existing_batch_ids`,
`write_pending_batch` and `append_jsonl` are copied from jacgen2's batch.jac /
common.jac rather than imported. A cross-directory Jac import type-checks under
`jac check` but dies at runtime with "attempted relative import with no known
parent package" -- `01-sft-dpo` is not a valid Python identifier and `jac run`
gives entry scripts no package context. gate.jac hit and documented this
identical problem; task18 report section 4 lists it as a known failure mode.
Inlining is the established workaround, not sloppiness.
"""

import os;
import json;
import random;
import from pathlib { Path }

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

glob EXP_ROOT: str = str(Path(str(__file__)).resolve().parent.parent);

glob POOL_PATH: str = EXP_ROOT + "/dataset/candidate_pool.jsonl";
glob HOLDOUT_PATH: str = EXP_ROOT + "/dataset/nitin_holdout.jsonl";
glob RAW_ROOT: str = EXP_ROOT + "/dataset/raw_output";

glob CATEGORY: str = "conversion";
glob TASK_TYPE: str = "python_to_jac_function";
glob DPO_AXIS: str = "correctness";

"""RE-DERIVED 2026-08-18 from this corpus's actual triage output. The clean
candidate pool is 6781 rows (06's was 5474), so 06's 16 shards would have given
424 items each -- outside the 300-400 band one dispatched Opus agent carries
comfortably in a single sitting. 20 shards gives 339-340 items each, inside the
band and close to 06's own 342-343 per shard."""
glob SFT_SHARDS: int = 20;

"""3 shards over the 1000-row DPO sample -> 333-334 items each. DPO_SAMPLE_N
stays at 1000 (not scaled to the larger pool) so this phase's DPO set size stays
comparable to 06's -- DPO set size is part of the recipe being replicated, and
changing it would confound the arm comparison."""
glob DPO_SHARDS: int = 3;

"""3 shards over the 855 frozen holdout rows -> 285 items each. Matches 06's own
holdout_eval sharding (it also used 3 shards over 855)."""
glob HOLDOUT_SHARDS: int = 3;
glob DPO_SAMPLE_N: int = 1000;
glob DPO_SAMPLE_SEED: int = 42;

"""DPO eligibility floor. A subtle BEHAVIOURAL bug has nowhere to hide in a
two-line function -- an edit there is either a no-op or a parse error, and both
poison the pair. Sampling is therefore seeded-random over rows clearing this
floor, not over the raw pool. Stated here rather than buried so the sample is
reproducible and the bias is on the record.

RECHECKED 2026-08-18 against THIS corpus rather than carried over: its clean pool
has a median body of 32 tokens / 8 code lines (mean 39.0 / 9.9, p25 21, p75 49) --
statistically the same shape as 06's 31 tokens / 8 code lines. So the 30/5 floor
transfers exactly, and it selects the same proportion it did in 06:
  07: 3614 of 6781 rows qualify (53.3%)
  06: 2870 of 5474 rows qualify (52.4%)
The floor is therefore VALIDATED for this corpus, not merely inherited."""
glob DPO_MIN_TOKENS: int = 30;
glob DPO_MIN_CODE_LINES: int = 5;


# ---------------------------------------------------------------------------
# Helpers inlined from jacgen2 (see module docstring for why they are copies)
# ---------------------------------------------------------------------------

def append_jsonl(path: str, rec: dict) {
    p: Path = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True);
    with open(path, "a") as f {
        f.write(json.dumps(rec) + "\n");
    }
}

def make_batch_id(category: str, task_type: str, seed_id: str, variant_idx: int) -> str {
    return category + "|" + task_type + "|" + seed_id + "|" + str(variant_idx);
}

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

"""Appends items to `path`, skipping any whose batch_id is already present, so
re-running prepare against an existing pending file does not grow it."""
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


# ---------------------------------------------------------------------------
# Pool loading + sharding
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> list[dict] {
    rows: list[dict] = [];
    p: Path = Path(path);
    if not p.exists() {
        raise RuntimeError("not found: " + path);
    }
    with open(path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            rows.append(dict(rec));
        }
    }
    return rows;
}

def load_pool() -> list[dict] {
    pool: list[dict] = read_jsonl(POOL_PATH);
    holdout: list[dict] = read_jsonl(HOLDOUT_PATH);
    frozen: dict = {};
    for row in holdout {
        frozen[str(row["id"])] = True;
    }
    # Asserted, not assumed. nitin_holdout.jsonl is eval-only and must never
    # contribute a training pair; the triage split already guarantees it, this
    # re-checks it at the only point where training data is minted.
    leaked: int = 0;
    for row in pool {
        if str(row["id"]) in frozen {
            leaked += 1;
        }
    }
    if leaked > 0 {
        raise RuntimeError(
            "holdout leak: " + str(leaked) + " candidate_pool rows also appear in "
            + "nitin_holdout.jsonl -- refusing to prepare batches."
        );
    }
    return pool;
}

"""Balanced contiguous shard sizes: the first (total % n) shards get one extra
item, so no shard is more than one item larger than any other."""
def shard_sizes(total: int, n: int) -> list[int] {
    base: int = total // n;
    rem: int = total % n;
    sizes: list[int] = [];
    for i in range(n) {
        extra: int = 0;
        if i < rem {
            extra = 1;
        }
        sizes.append(base + extra);
    }
    return sizes;
}

"""Sort key helper -- a named def rather than an inline lambda, so the sort is
unambiguous to the type checker."""
def by_pool_id(r: dict) -> str {
    return str(r["id"]);
}

def shard_path(track: str, idx: int) -> str {
    return RAW_ROOT + "/" + track + "/pending_batch_" + track + "_" + str(idx).zfill(2) + ".jsonl";
}


# ---------------------------------------------------------------------------
# Track 1 -- SFT reverse-instruction prompts (Opus)
# ---------------------------------------------------------------------------

"""Two registers, alternating by pool position so the emitted instruction set
is not thousands of sentences of one shape. This mirrors gen_code_gen.jac's own
variant-register split (variant 0 = plain user request, variant 1 = terse
expert/spec-style ask) -- the same convention, applied across items instead of
across variants of one item, because each source function yields exactly one
example here."""
def sft_register(idx: int) -> str {
    if idx % 2 == 0 {
        return "user_request";
    }
    return "spec_style";
}

def register_clause(register: str) -> str {
    if register == "user_request" {
        return (
            "Register for this item: PLAIN USER REQUEST -- phrase it the way a developer "
            + "asks a coding assistant (\"Write a function that ...\", \"I need something "
            + "that ...\"). One or two sentences."
        );
    }
    return (
        "Register for this item: TERSE EXPERT/SPEC-STYLE ASK -- a compressed spec naming "
        + "the signature and the required behaviour (\"Spec: name(args) -> type: "
        + "<behaviour>\"). One or two lines, no pleasantries."
    );
}

def build_sft_prompt(row: dict, register: str) -> str {
    signature: str = "def " + str(row["func_name"]) + "(" + str(row["params"]) + ") -> " + str(row["return_type"]);
    return (
        "Here is a Jac function that already exists. Write the natural-language request a "
        + "developer would have typed to get exactly this function written -- describe the "
        + "TASK it accomplishes, not the code line by line.\n\n"
        + register_clause(register) + "\n\n"
        + "Rules:\n"
        + "- Describe WHAT the function must do: its inputs, what it returns, and any edge "
        + "case that actually matters. Never describe HOW the code is written.\n"
        + "- Do NOT paste, quote, or fence any code. No markdown of any kind.\n"
        + "- Do NOT merely restate the function name (\"write expand_comma\") -- that is "
        + "rejected at collect time.\n"
        + "- Do not refer to the docstring, to \"this function\", or to the fact that you "
        + "were shown an implementation. The reader has never seen it.\n"
        + "- Do not name the programming language; this should read as a plain coding task.\n"
        + "- Reply with ONLY the instruction text: no preamble, no explanation, no markdown.\n\n"
        + "Function name: " + str(row["func_name"]) + "\n"
        + "Signature: " + signature + "\n"
        + "Docstring: " + str(row["docstring"]) + "\n\n"
        + "```jac\n" + str(row["jac_code"]) + "\n```\n"
    );
}

def run_prepare_sft() {
    pool: list[dict] = load_pool();
    total: int = len(pool);
    sizes: list[int] = shard_sizes(total, SFT_SHARDS);

    cursor: int = 0;
    manifest_shards: list[dict] = [];
    total_written: int = 0;
    total_skipped: int = 0;

    for s in range(SFT_SHARDS) {
        shard_idx: int = s + 1;
        n: int = sizes[s];
        items: list[dict] = [];
        for j in range(n) {
            pos: int = cursor + j;
            row: dict = pool[pos];
            register: str = sft_register(pos);
            items.append({
                "batch_id": make_batch_id(CATEGORY, TASK_TYPE, str(row["seed_id"]), 0),
                "model": "opus",
                "prompt": build_sft_prompt(row, register),
                "track": "sft",
                "shard": shard_idx,
                "category": CATEGORY,
                "task_type": TASK_TYPE,
                "variant_idx": 0,
                "register": register,
                "pool_id": row["id"],
                "seed_id": row["seed_id"],
                "func_name": row["func_name"],
                "params": row["params"],
                "return_type": row["return_type"],
                "docstring": row["docstring"],
                "jac_code": row["jac_code"],
                "origin": row["origin"],
                "source_file": row["source_file"],
                "gate_class": "compile_only",
                "compile_pass": row["compile_pass"]
            });
        }
        cursor += n;
        path: str = shard_path("sft", shard_idx);
        res: dict = write_pending_batch(path, items);
        total_written += int(res["written"]);
        total_skipped += int(res["skipped"]);
        manifest_shards.append({
            "shard": shard_idx,
            "pending_file": "pending_batch_sft_" + str(shard_idx).zfill(2) + ".jsonl",
            "responses_file": "responses_batch_sft_" + str(shard_idx).zfill(2) + ".jsonl",
            "items": n,
            "written": res["written"],
            "skipped_already_present": res["skipped"]
        });
        print("  shard " + str(shard_idx).zfill(2) + ": " + str(n) + " items -> " + path);
    }

    manifest: dict = {
        "track": "sft",
        "model": "opus",
        "category": CATEGORY,
        "task_type": TASK_TYPE,
        "source": "dataset/candidate_pool.jsonl",
        "source_rows": total,
        "selection": "all candidate_pool rows, file order",
        "registers": {"user_request": "even pool positions", "spec_style": "odd pool positions"},
        "shards": manifest_shards
    };
    with open(RAW_ROOT + "/sft/manifest.json", "w") as f {
        f.write(json.dumps(manifest, indent=2) + "\n");
    }

    print("");
    print("=== prepare sft summary ===");
    print("pool rows: " + str(total));
    print("shards: " + str(SFT_SHARDS));
    print("newly written: " + str(total_written) + "   already present: " + str(total_skipped));
    print("wrote " + RAW_ROOT + "/sft/manifest.json");
}


# ---------------------------------------------------------------------------
# Track 2 -- DPO buggy-variant prompts (Fable)
# ---------------------------------------------------------------------------

def build_dpo_prompt(row: dict) -> str {
    signature: str = "def " + str(row["func_name"]) + "(" + str(row["params"]) + ") -> " + str(row["return_type"]);
    return (
        "Here is working, correct Jac code. Produce a variant carrying EXACTLY ONE "
        + "subtly-introduced bug.\n\n"
        + "Hard requirements:\n"
        + "- The bug must be a LOGIC / behavioural divergence, NOT a syntax break. The buggy "
        + "variant MUST still compile and run (`jac run` exit 0). A variant that fails to "
        + "parse is rejected at collect time and the whole pair is discarded.\n"
        + "- Exactly ONE edit. Same function name, same parameter list, same return type, "
        + "same docstring, same structure and formatting. Change nothing else.\n"
        + "- Do not edit the docstring, and do not add a comment pointing at the bug.\n"
        + "- The bug must be findable only by reasoning about behaviour -- no eye-catching "
        + "weirdness.\n\n"
        + "Pick whichever ONE of these fits this function most naturally:\n"
        + "  off_by_one           -- an index or range bound shifted by one\n"
        + "  flipped_comparison   -- < vs <=, > vs >=, == vs !=, `and` vs `or`\n"
        + "  swapped_branch       -- if-body and else-body exchanged, or a negated condition\n"
        + "  mutated_default      -- a default argument value changed\n"
        + "  inverted_accumulator -- += vs -=, * vs /, or a wrong initial accumulator value\n"
        + "  dropped_guard        -- an early-return / empty-input / None check removed\n\n"
        + "If this function genuinely has no room for a subtle behavioural bug of any of "
        + "those kinds (too short, or every candidate edit is either a no-op or a parse "
        + "error), reply with exactly:\n"
        + "SKIP: <one-line reason>\n"
        + "and nothing else. A poisoned pair is worse than a dropped one.\n\n"
        + "Otherwise write a one-line SYMPTOM: how the bug manifests, phrased as a bug "
        + "report and never as a fix -- a wrong-output description (\"returns 4 instead of 5 "
        + "for input [1, 2, 3]\") or a behavioural one (\"an empty input list now raises "
        + "IndexError instead of returning 0\"). It is embedded verbatim into the training "
        + "prompt as \"Write correct Jac code that avoids this bug: <SYMPTOM>\", so it must "
        + "read as a standalone bug report.\n\n"
        + "Reply in EXACTLY this format, nothing else:\n"
        + "BUG_TYPE: <one of the six identifiers above>\n"
        + "BUGGY:\n```jac\n<the complete buggy file, docstring included, nothing omitted>\n```\n"
        + "SYMPTOM: <one line>\n\n"
        + "Signature: " + signature + "\n\n"
        + "Original correct code:\n```jac\n" + str(row["jac_code"]) + "\n```\n"
    );
}

"""Track 1b -- reverse-instruction prompts for the FROZEN HOLDOUT rows (Opus).

WHY THIS TRACK EXISTS. 06 froze its holdout straight out of the candidate pool,
so the rows carried only jac_code + docstring and NO `messages` field. The
functional harness (eval_functional.jac) had nothing to prompt the model with,
so the first holdout-B evaluation "completed" in under 30 seconds having never
generated a single token -- it looked like a pass and was a no-op (06's final
report section 4, incident 4). 06 recovered after the fact by reverse-authoring
instructions for all 855 holdout rows into a separate nitin_holdout_eval.jsonl.

This phase does that pass UP FRONT instead of discovering it later. Identical
prompt builder and identical registers to the SFT track -- the holdout must be
authored the same way the training set was, or the two are not measuring the
same distribution. The ONLY differences: the source file (nitin_holdout.jsonl,
never candidate_pool.jsonl) and the output directory.

These rows are EVAL-ONLY. They are never collected into sft_train.jsonl."""
def run_prepare_holdout_eval() {
    holdout: list[dict] = read_jsonl(HOLDOUT_PATH);
    total: int = len(holdout);
    sizes: list[int] = shard_sizes(total, HOLDOUT_SHARDS);

    cursor: int = 0;
    manifest_shards: list[dict] = [];
    total_written: int = 0;
    total_skipped: int = 0;

    for s in range(HOLDOUT_SHARDS) {
        shard_idx: int = s + 1;
        n: int = sizes[s];
        items: list[dict] = [];
        for j in range(n) {
            pos: int = cursor + j;
            row: dict = holdout[pos];
            register: str = sft_register(pos);
            items.append({
                "batch_id": make_batch_id(CATEGORY, TASK_TYPE, str(row["seed_id"]), 0),
                "model": "opus",
                "prompt": build_sft_prompt(row, register),
                "track": "holdout_eval",
                "shard": shard_idx,
                "category": CATEGORY,
                "task_type": TASK_TYPE,
                "variant_idx": 0,
                "register": register,
                "pool_id": row["id"],
                "seed_id": row["seed_id"],
                "func_name": row["func_name"],
                "params": row["params"],
                "return_type": row["return_type"],
                "docstring": row["docstring"],
                "jac_code": row["jac_code"],
                "origin": row["origin"],
                "source_file": row["source_file"],
                "gate_class": "compile_only",
                "compile_pass": row["compile_pass"]
            });
        }
        cursor += n;
        path: str = shard_path("holdout_eval", shard_idx);
        res: dict = write_pending_batch(path, items);
        total_written += int(res["written"]);
        total_skipped += int(res["skipped"]);
        manifest_shards.append({
            "shard": shard_idx,
            "pending_file": "pending_batch_holdout_eval_" + str(shard_idx).zfill(2) + ".jsonl",
            "responses_file": "responses_batch_holdout_eval_" + str(shard_idx).zfill(2) + ".jsonl",
            "items": n,
            "written": res["written"],
            "skipped_already_present": res["skipped"]
        });
        print("  shard " + str(shard_idx).zfill(2) + ": " + str(n) + " items -> " + path);
    }

    manifest: dict = {
        "track": "holdout_eval",
        "model": "opus",
        "category": CATEGORY,
        "task_type": TASK_TYPE,
        "source": "dataset/nitin_holdout.jsonl",
        "source_rows": total,
        "selection": "ALL frozen holdout rows, file order -- eval only, never trained on",
        "registers": {"user_request": "even positions", "spec_style": "odd positions"},
        "shards": manifest_shards
    };
    with open(RAW_ROOT + "/holdout_eval/manifest.json", "w") as f {
        f.write(json.dumps(manifest, indent=2) + "\n");
    }

    print("");
    print("=== prepare holdout_eval summary ===");
    print("holdout rows: " + str(total));
    print("shards: " + str(HOLDOUT_SHARDS));
    print("newly written: " + str(total_written) + "   already present: " + str(total_skipped));
    print("wrote " + RAW_ROOT + "/holdout_eval/manifest.json");
}


"""Track 2b -- DPO EXPANSION to the full eligible pool.

06 sampled 1000 pairs from the rows clearing the behavioural-bug floor, and 07
initially mirrored that for recipe comparability. The explicit instruction for
this phase is to use the corpus to its fullest potential, so this pass prepares
EVERY remaining eligible row that is not already covered by an existing dpo
shard, continuing the shard numbering rather than re-sharding.

Why not just raise DPO_SAMPLE_N and re-run: write_pending_batch dedupes per FILE,
so changing the sample size changes every shard boundary and the same batch_id
would be written again into a different shard. This pass is additive and
idempotent -- it excludes ids already present in ANY dpo pending file.

The eligibility floor itself is NOT relaxed. It exists because a subtle
behavioural bug has nowhere to hide in a two-line function; rows below it would
yield poisoned pairs, not extra signal. "Fullest potential" means all 3614
eligible rows, not all 6781 pool rows."""
def run_prepare_dpo_expand() {
    pool: list[dict] = load_pool();

    eligible: list[dict] = [];
    for row in pool {
        if int(row["token_count"]) >= DPO_MIN_TOKENS and int(row["code_lines"]) >= DPO_MIN_CODE_LINES {
            eligible.append(row);
        }
    }
    eligible = sorted(eligible, key=by_pool_id);

    # every batch_id already prepared in ANY existing dpo shard
    covered: dict = {};
    tdir: str = RAW_ROOT + "/dpo";
    if os.path.isdir(tdir) {
        for name in sorted(os.listdir(tdir)) {
            if name.startswith("pending_batch_dpo_") and name.endswith(".jsonl") {
                for (bid, _v) in existing_batch_ids(tdir + "/" + name).items() {
                    covered[bid] = True;
                }
            }
        }
    }

    remaining: list[dict] = [];
    for row in eligible {
        # MUST match run_prepare_dpo's id scheme -- the DPO track keys on
        # ("dpo", axis), NOT (category, task_type) like the SFT track. Using the
        # SFT scheme here silently matched nothing and re-prepared all 3614 rows.
        bid: str = make_batch_id("dpo", DPO_AXIS, str(row["seed_id"]), 0);
        if bid not in covered {
            remaining.append(row);
        }
    }

    print("eligible rows (floor " + str(DPO_MIN_TOKENS) + " tok / "
        + str(DPO_MIN_CODE_LINES) + " lines): " + str(len(eligible)));
    print("already covered by existing shards: " + str(len(covered)));
    print("remaining to prepare: " + str(len(remaining)));
    if not remaining {
        print("nothing to do -- the full eligible pool is already prepared.");
        return;
    }

    # keep shards inside the 300-400 band one dispatched agent carries
    n_shards: int = (len(remaining) + 349) // 350;
    if n_shards < 1 {
        n_shards = 1;
    }
    sizes: list[int] = shard_sizes(len(remaining), n_shards);

    # continue numbering after the highest existing shard
    start_idx: int = 0;
    if os.path.isdir(tdir) {
        for name in sorted(os.listdir(tdir)) {
            if name.startswith("pending_batch_dpo_") and name.endswith(".jsonl") {
                stem: str = name[len("pending_batch_dpo_"):-len(".jsonl")];
                if stem.isdigit() {
                    v: int = int(stem);
                    if v > start_idx {
                        start_idx = v;
                    }
                }
            }
        }
    }

    cursor: int = 0;
    total_written: int = 0;
    for s in range(n_shards) {
        shard_idx: int = start_idx + s + 1;
        n: int = sizes[s];
        items: list[dict] = [];
        for j in range(n) {
            row: dict = remaining[cursor + j];
            items.append({
                "batch_id": make_batch_id("dpo", DPO_AXIS, str(row["seed_id"]), 0),
                "model": "fable",
                "prompt": build_dpo_prompt(row),
                "track": "dpo",
                "shard": shard_idx,
                "axis": DPO_AXIS,
                "variant_idx": 0,
                "pool_id": row["id"],
                "seed_id": row["seed_id"],
                "func_name": row["func_name"],
                "params": row["params"],
                "return_type": row["return_type"],
                "docstring": row["docstring"],
                "jac_code": row["jac_code"],
                "origin": row["origin"],
                "source_file": row["source_file"],
                "token_count": row["token_count"],
                "code_lines": row["code_lines"],
                "compile_pass": row["compile_pass"]
            });
        }
        cursor += n;
        path: str = shard_path("dpo", shard_idx);
        res: dict = write_pending_batch(path, items);
        total_written += int(res["written"]);
        print("  shard " + str(shard_idx).zfill(2) + ": " + str(n) + " items -> " + path);
    }

    print("");
    print("=== prepare dpo_expand summary ===");
    print("new shards: " + str(n_shards) + " starting at index " + str(start_idx + 1));
    print("newly written: " + str(total_written));
}


def run_prepare_dpo() {
    pool: list[dict] = load_pool();

    eligible: list[dict] = [];
    for row in pool {
        if int(row["token_count"]) >= DPO_MIN_TOKENS and int(row["code_lines"]) >= DPO_MIN_CODE_LINES {
            eligible.append(row);
        }
    }
    eligible = sorted(eligible, key=by_pool_id);

    n_sample: int = DPO_SAMPLE_N;
    if n_sample > len(eligible) {
        n_sample = len(eligible);
    }
    rng = random.Random(DPO_SAMPLE_SEED);
    sampled: list[dict] = rng.sample(eligible, n_sample);
    sampled = sorted(sampled, key=by_pool_id);

    sizes: list[int] = shard_sizes(len(sampled), DPO_SHARDS);
    cursor: int = 0;
    manifest_shards: list[dict] = [];
    total_written: int = 0;
    total_skipped: int = 0;

    for s in range(DPO_SHARDS) {
        shard_idx: int = s + 1;
        n: int = sizes[s];
        items: list[dict] = [];
        for j in range(n) {
            row: dict = sampled[cursor + j];
            items.append({
                "batch_id": make_batch_id("dpo", DPO_AXIS, str(row["seed_id"]), 0),
                "model": "fable",
                "prompt": build_dpo_prompt(row),
                "track": "dpo",
                "shard": shard_idx,
                "axis": DPO_AXIS,
                "variant_idx": 0,
                "pool_id": row["id"],
                "seed_id": row["seed_id"],
                "func_name": row["func_name"],
                "params": row["params"],
                "return_type": row["return_type"],
                "docstring": row["docstring"],
                "jac_code": row["jac_code"],
                "origin": row["origin"],
                "source_file": row["source_file"],
                "token_count": row["token_count"],
                "code_lines": row["code_lines"],
                "compile_pass": row["compile_pass"]
            });
        }
        cursor += n;
        path: str = shard_path("dpo", shard_idx);
        res: dict = write_pending_batch(path, items);
        total_written += int(res["written"]);
        total_skipped += int(res["skipped"]);
        manifest_shards.append({
            "shard": shard_idx,
            "pending_file": "pending_batch_dpo_" + str(shard_idx).zfill(2) + ".jsonl",
            "responses_file": "responses_batch_dpo_" + str(shard_idx).zfill(2) + ".jsonl",
            "items": n,
            "written": res["written"],
            "skipped_already_present": res["skipped"]
        });
        print("  shard " + str(shard_idx).zfill(2) + ": " + str(n) + " items -> " + path);
    }

    sampled_ids: list[str] = [];
    for row in sampled {
        sampled_ids.append(str(row["id"]));
    }
    manifest: dict = {
        "track": "dpo",
        "model": "fable",
        "axis": DPO_AXIS,
        "source": "dataset/candidate_pool.jsonl",
        "source_rows": len(pool),
        "eligibility_filter": "token_count >= " + str(DPO_MIN_TOKENS)
            + " and code_lines >= " + str(DPO_MIN_CODE_LINES),
        "eligible_rows": len(eligible),
        "sample_n": len(sampled),
        "sample_method": "random.Random(" + str(DPO_SAMPLE_SEED)
            + ").sample over eligible rows sorted by id; result re-sorted by id",
        "shards": manifest_shards,
        "sampled_pool_ids": sampled_ids
    };
    with open(RAW_ROOT + "/dpo/manifest.json", "w") as f {
        f.write(json.dumps(manifest, indent=2) + "\n");
    }

    print("");
    print("=== prepare dpo summary ===");
    print("pool rows: " + str(len(pool)));
    print("eligible (token_count >= " + str(DPO_MIN_TOKENS) + ", code_lines >= "
        + str(DPO_MIN_CODE_LINES) + "): " + str(len(eligible)));
    print("sampled (seed " + str(DPO_SAMPLE_SEED) + "): " + str(len(sampled)));
    print("shards: " + str(DPO_SHARDS));
    print("newly written: " + str(total_written) + "   already present: " + str(total_skipped));
    print("wrote " + RAW_ROOT + "/dpo/manifest.json");
}


with entry:__main__ {
    track: str | None = os.environ.get("JAC_BATCH_TRACK");
    if track == "sft" {
        run_prepare_sft();
    } elif track == "holdout_eval" {
        run_prepare_holdout_eval();
    } elif track == "dpo" {
        run_prepare_dpo();
    } elif track == "dpo_expand" {
        run_prepare_dpo_expand();
    } else {
        raise ValueError(
            "JAC_BATCH_TRACK must be set to 'sft', 'holdout_eval', 'dpo' or 'dpo_expand', got: " + str(track)
        );
    }
}
