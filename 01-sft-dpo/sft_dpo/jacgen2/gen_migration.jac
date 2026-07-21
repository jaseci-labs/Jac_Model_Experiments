"""
gen_migration.jac -- pilot generator for the `migration` category (spec.md
04-cpt-sft/docs/datagen/spec.md section 7), built on the prompt-batch
handoff protocol in batch.jac. Same two-phase shape as gen_code_gen.jac --
read that file's docstring for the general pattern, not repeated here.

Task type coverage (spec.md section 7's table has 2 task types):

  file_syntax_migration -- IMPLEMENTED. Composes 2-4 of the deprecated-
    pattern inventory's 4 confirmed entries (data/deprecated_inventory.jsonl)
    into one combined deprecated snippet (itertools.combinations over sizes
    [2, 3, 4]: C(4,2)=6 + C(4,3)=4 + C(4,4)=1 = 11 combos), asks Opus to
    migrate the whole file, and two-part-gates the result.

  scaffold_modernization -- SKIPPED for this pilot. This task type needs a
    `jac create --use <template>` scaffold OUTPUT as its seed (per spec.md
    section 7's table: "output of `jac create --use <template>` ... ->
    modernized equivalent"), and no such sample exists anywhere in this
    pipeline -- seed_pool.jac's five tiers (jac-mcp examples/docs, jaclang's
    own test fixtures, public GitHub scrape, this repo's own app code) don't
    include scaffold-command output, and jac-scaffold's skill guide documents
    the checklist, not a captured raw-scaffold sample. Fabricating a fake
    "scaffold output" from scratch would be an invented, ungrounded seed --
    the deprecated_inventory's 4 entries don't naturally stand in for a
    scaffold either (they're isolated single-pattern snippets, not a
    multi-file template skeleton with the specific deprecated forms
    `jac create` is documented to ship). Real fix: run
    `jac create --use <template>` live to capture actual scaffold output as
    a seed -- out of scope for this pilot's inventory-driven approach.

Inventory caveat (spec.md section 7, matches its own stated contingency):
only 4 confirmed deprecated patterns exist (W0061 paren filter, W0062
root(), W0063 JSX spread -- all `detect: check_warning` -- plus
v1_import_py_prefix, `detect: hard_fail`). This inherently caps this
category's pilot scale small; this file does not pad with repetition to hit
a larger number, per spec.md section 7's own instruction ("if it can't
support 500 distinct examples ... shrink the category").

Gate (spec.md section 7, corrected 2026-07-20 -- two-part, NOT symmetric
with every other category's never-`jac check` rule):

  1. Migrated file must pass `jac run` (the normal generation gate).
  2. The deprecated ORIGINAL must either fail `jac run` (hard_fail forms)
     or produce a deprecation warning under the FULL `jac check` (no `-p`
     -- confirmed by Task 7's investigation that `-p` never surfaces
     warnings). This is a narrow, explicit exception to the project's
     standing "never gate on `jac check`" rule: check is used ONLY to
     detect warnings on the INPUT during PREPARE's sanity check, never to
     reject generated OUTPUT. Gate part 1 (the only gate that can reject a
     generated example) still uses plain `jac run`.

  As a nice-to-have beyond spec.md's letter, COLLECT also re-runs the full
  `jac check` on the MIGRATED code and confirms none of the combo's
  original check_warning codes still appear -- a correctness signal (did
  the migration actually remove the deprecated forms, not just happen to
  compile), not a spec requirement. See run_full_check/verify_combo_reproduces
  below -- gate.jac's helpers only wrap `-p` parse-only check, so this file
  adds its own minimal full-check helper rather than modifying gate.jac.

Two phases, selected by env var JAC_BATCH_PHASE:

  prepare -- for each pattern combo, sanity-checks (locally, no LLM call)
    that the COMPOSED original genuinely reproduces the expected fail/warn
    signature (composing multiple example_before snippets into one file
    could accidentally break one of them -- e.g. a name collision), skips
    and logs any combo that doesn't, then writes a pending batch item for
    every combo that passes. model="opus" per spec.md section 4.1
    (migration is Opus-assigned -- token-heavy whole-file rewrites, fully
    compiler-gated, low judgment risk).

  collect -- reads responses_batch.jsonl, joins each response back to its
    pending-item context by batch_id, strips any accidental markdown-fence
    wrapper, two-part-gates, and persists accepted examples via
    gate.make_example_v2 + common.append_jsonl.

Run (from repo root):
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=prepare JAC_PILOT_LIMIT=10 \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_migration.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=collect \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_migration.jac
"""

import os;
import json;
import itertools;
import subprocess;
import tempfile;
import hashlib;
import from pathlib { Path }
import from common { require_run_tag, data_root, append_jsonl, existing_keys }
import from gate { run_jac_local, make_example_v2 }
import from batch { make_batch_id, write_pending_batch, read_response_batch, pending_path, responses_path }

glob CATEGORY: str = "migration";
glob INVENTORY_PATH: str = "model-experiments/01-sft-dpo/sft_dpo/jacgen2/data/deprecated_inventory.jsonl";
glob GENERATOR_MODEL_ID: str = "claude-opus-4-8";

"""Same absolute-path requirement as gate.jac's JAC_BIN (bare "jac" resolves
to a stale unrelated venv) -- redefined locally rather than imported, same
convention seed_pool.jac/seed_scrape.jac already use for this constant."""
glob JAC_BIN: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac";

"""Combo sizes to draw from the 4-pattern inventory via itertools.combinations
-- C(4,2)=6 + C(4,3)=4 + C(4,4)=1 = 11 total combos, capped at JAC_PILOT_LIMIT
(default 10, so in practice the single size-4 combo is the one capped off)."""
glob COMBO_SIZES: list[int] = [2, 3, 4];

glob PROMPT_TEMPLATE_PREFIX: str = (
    "The following Jac file uses deprecated syntax. Migrate it to current Jac syntax, "
    + "fixing ALL deprecated patterns while preserving the exact same behavior/intent. "
    + "Reply with ONLY the fully migrated Jac code, no explanation, no markdown fence "
    + "around your reply (just the raw code).\n\nDeprecated file:\n"
);


def load_inventory() -> list[dict] {
    patterns: list[dict] = [];
    path: Path = Path(INVENTORY_PATH);
    if not path.exists() {
        raise RuntimeError("deprecated-pattern inventory not found at " + INVENTORY_PATH);
    }
    with open(INVENTORY_PATH, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            patterns.append(dict(rec));
        }
    }
    return patterns;
}

"""Builds every combo of the given sizes from the inventory, in
itertools.combinations order (stable, deterministic across runs -- so
batch_ids stay stable for idempotent re-prepare)."""
def build_combos(patterns: list[dict]) -> list[list[dict]] {
    combos: list[list[dict]] = [];
    for size in COMBO_SIZES {
        if size > len(patterns) {
            continue;
        }
        for combo_tuple in itertools.combinations(patterns, size) {
            combo: list[dict] = [];
            for p in combo_tuple {
                combo.append(dict(p));
            }
            combos.append(combo);
        }
    }
    return combos;
}

"""Concatenates each pattern's example_before into one combined deprecated
snippet -- doesn't need to be a unified program, just needs to contain
multiple distinct deprecated forms in one file (verified empirically: Jac
allows multiple top-level `with entry { }` blocks in a single file, each
executed in order -- confirmed via a manual multi-pattern test run before
writing this generator)."""
def compose_combo(patterns: list[dict]) -> str {
    parts: list[str] = [];
    for p in patterns {
        parts.append(str(p["example_before"]).rstrip());
    }
    return "\n\n".join(parts) + "\n";
}

def build_prompt(combined_code: str) -> str {
    return PROMPT_TEMPLATE_PREFIX + combined_code;
}

"""A stable hash of the combo's pattern_ids (order as drawn from the
inventory, not sorted -- build_combos' itertools order is already
deterministic) so re-running prepare produces the same seed_id/batch_id
every time."""
def make_seed_id(pattern_ids: list[str]) -> str {
    joined: str = "|".join(pattern_ids);
    digest: str = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12];
    return "migration_combo_" + digest;
}

"""Extracts the compiler warning code from a check_warning pattern's
pattern_id (e.g. "W0061_paren_filter" -> "W0061"). Meaningless for
hard_fail patterns (e.g. "v1_import_py_prefix" -> "v1") -- callers must
only call this for detect == "check_warning" patterns."""
def pattern_warning_code(pattern: dict) -> str {
    pattern_id: str = str(pattern["pattern_id"]);
    parts: list[str] = pattern_id.split("_");
    return parts[0];
}

"""Same tempdir-isolation subprocess pattern as gate.jac's run_jac_local,
but calls `jac check <file>` (the FULL type-checker, no `-p`) instead of
`jac run` -- gate.jac's own helpers only wrap `-p` parse-only check (see
seed_pool.jac/seed_scrape.jac), and `-p` never surfaces deprecation
warnings (Task 7's finding), so this file needs its own minimal helper
rather than reusing gate.jac's. Used ONLY to detect warnings on already-
deprecated INPUT (prepare's sanity check, collect's nice-to-have leftover-
warning re-check) -- never to reject generated OUTPUT; gate part 1 (the
only rejection-capable gate) still uses plain run_jac_local (`jac run`)."""
def run_full_check(jac_code: str, timeout: int = 30) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "snippet.jac";
        f.write_text(jac_code);
        try {
            p = subprocess.run(
                [JAC_BIN, "check", str(f)], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}

"""PREPARE-time sanity check (cheap, local, no LLM call): confirms the
COMPOSED original combo genuinely reproduces the expected fail/warn
signature before it's ever sent to Opus -- composing multiple example_before
snippets into one file could accidentally break one of them (a name
collision, an entry-block ordering issue). If any pattern in the combo is
detect=="hard_fail", the whole combined file is expected to fail `jac run`
outright (empirically confirmed: `import:py ...` is a genuine parse error,
not just a warning, and a parse error anywhere in a file fails the whole
file regardless of what else is present). Otherwise (all check_warning
patterns), the combined file is expected to PASS `jac run` and have every
pattern's warning code appear in the FULL `jac check` output."""
def verify_combo_reproduces(patterns: list[dict], combined_code: str) -> tuple[bool, str] {
    has_hard_fail: bool = False;
    for p in patterns {
        if str(p["detect"]) == "hard_fail" {
            has_hard_fail = True;
        }
    }

    (run_code, run_out, run_err) = run_jac_local(combined_code);

    if has_hard_fail {
        if run_code != 0 {
            return (True, "ok: hard_fail pattern present, jac run failed as expected (exit " + str(run_code) + ")");
        }
        return (False, "expected jac run to fail (hard_fail pattern present in combo) but it exited 0");
    }

    if run_code != 0 {
        return (False, "composition bug: check_warning-only combo failed jac run unexpectedly (exit "
            + str(run_code) + "): " + run_err.strip()[:200]);
    }

    (check_code, check_out, check_err) = run_full_check(combined_code);
    combined_output: str = check_out + "\n" + check_err;

    found_codes: list[str] = [];
    missing_codes: list[str] = [];
    for p in patterns {
        code: str = pattern_warning_code(p);
        if code in combined_output {
            found_codes.append(code);
        } else {
            missing_codes.append(code);
        }
    }
    if missing_codes {
        return (False, "missing expected warning code(s) in `jac check` output: " + ",".join(missing_codes));
    }
    return (True, "ok: all expected warning codes present under full `jac check`: " + ",".join(found_codes));
}

"""Strips an accidental markdown-fence wrapper defensively (the prompt asks
for raw code with no fence, but LLM responses sometimes add one anyway).
Same split/rsplit technique as gate.jac's extract_payload."""
def strip_markdown_fence(text: str) -> str {
    s: str = text.strip();
    if "```" in s {
        if "```jac" in s {
            s = s.split("```jac", 1)[1];
        } else {
            s = s.split("```", 1)[1];
        }
        if "```" in s {
            s = s.rsplit("```", 1)[0];
        }
    }
    return s.strip();
}


def run_prepare() {
    tag: str = require_run_tag();
    limit: int = int(os.environ.get("JAC_PILOT_LIMIT", "10"));

    patterns: list[dict] = load_inventory();
    combos: list[list[dict]] = build_combos(patterns);

    items: list[dict] = [];
    capped: int = 0;
    verify_failed: int = 0;
    verify_fail_details: list[dict] = [];

    for combo in combos {
        pattern_ids: list[str] = [];
        for p in combo {
            pattern_ids.append(str(p["pattern_id"]));
        }

        combined_code: str = compose_combo(combo);
        (verify_ok, verify_reason) = verify_combo_reproduces(combo, combined_code);
        if not verify_ok {
            verify_failed += 1;
            verify_fail_details.append({"pattern_ids": pattern_ids, "reason": verify_reason});
            continue;
        }

        if len(items) >= limit {
            capped += 1;
            continue;
        }

        seed_id: str = make_seed_id(pattern_ids);
        prompt: str = build_prompt(combined_code);
        batch_id: str = make_batch_id(CATEGORY, "file_syntax_migration", seed_id, 0);
        items.append({
            "batch_id": batch_id,
            "model": "opus",
            "prompt": prompt,
            "category": CATEGORY,
            "task_type": "file_syntax_migration",
            "variant_idx": 0,
            "seed_id": seed_id,
            "seed_tier": 0,
            "pattern_ids": pattern_ids,
            "original_code": combined_code
        });
    }

    out_path: str = pending_path(tag, CATEGORY);
    write_result: dict = write_pending_batch(out_path, items);

    print("");
    print("=== gen_migration prepare summary ===");
    print("scaffold_modernization: SKIPPED -- no `jac create --use <template>` scaffold-output "
        + "seed corpus exists anywhere in this pipeline; fabricating one would be ungrounded. "
        + "See this file's top docstring for the full rationale.");
    print("inventory patterns available: " + str(len(patterns)));
    print("combos generated (sizes " + str(COMBO_SIZES) + "): " + str(len(combos)));
    print("combos that FAILED the prepare-time reproduction sanity check (skipped, not sent to Opus): "
        + str(verify_failed));
    for d in verify_fail_details {
        print("  - " + str(d["pattern_ids"]) + ": " + str(d["reason"]));
    }
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

    clean_path: str = data_root(tag) + "/clean_dataset/migration/examples.jsonl";
    rejected_path: str = data_root(tag) + "/rejected/migration/rejected.jsonl";
    already_persisted: dict = existing_keys(clean_path);

    processed: int = 0;
    accepted: int = 0;
    already_persisted_skipped: int = 0;
    part2_confirmed: int = 0;
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

        pattern_ids: list = list(ctx["pattern_ids"]);

        if not bool(resp["ok"]) {
            reason: str = "generation failed (ok:false)";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": ctx["task_type"], "pattern_ids": pattern_ids, "reason": reason
            });
            continue;
        }

        migrated_code: str = strip_markdown_fence(str(resp["text"]));
        if not migrated_code {
            reason: str = "empty migrated code after fence-stripping";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": ctx["task_type"], "pattern_ids": pattern_ids, "reason": reason
            });
            continue;
        }

        # Gate part 1 (spec.md section 7): migrated file must pass `jac run`
        # -- the only gate here that can reject a generated example.
        (run_code, run_out, run_err) = run_jac_local(migrated_code);
        if run_code != 0 {
            reason: str = ("gate part 1 failed: migrated code did not pass jac run (exit "
                + str(run_code) + "): " + run_err.strip()[:200]);
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": ctx["task_type"], "pattern_ids": pattern_ids, "reason": reason
            });
            continue;
        }

        # Gate part 2 nice-to-have (not a spec.md hard requirement -- see
        # top docstring): re-run the FULL `jac check` on the MIGRATED code
        # and confirm none of the combo's original check_warning codes are
        # still present. hard_fail patterns have no meaningful warning
        # code (pattern_warning_code would return "v1", not a real code) so
        # they're skipped here by the leading-"W" filter -- gate part 1
        # already covers them (a hard_fail pattern that survived migration
        # unfixed would almost certainly also fail `jac run`).
        (check_code, check_out, check_err) = run_full_check(migrated_code);
        migrated_check_output: str = check_out + "\n" + check_err;
        leftover_codes: list[str] = [];
        for pid in pattern_ids {
            code: str = str(pid).split("_")[0];
            if not code.startswith("W") {
                continue;
            }
            if code in migrated_check_output {
                leftover_codes.append(code);
            }
        }
        if leftover_codes {
            reason: str = ("gate part 2 failed: migrated code still triggers deprecation warning(s) "
                + ",".join(leftover_codes) + " under full `jac check`");
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "task_type": ctx["task_type"], "pattern_ids": pattern_ids, "reason": reason
            });
            continue;
        }
        part2_confirmed += 1;

        original_code: str = str(ctx["original_code"]);
        messages: list[dict] = [
            {"role": "user", "content": "Migrate this deprecated Jac code to current syntax:\n\n```jac\n" + original_code + "\n```"},
            {"role": "assistant", "content": "```jac\n" + migrated_code + "\n```"}
        ];

        rec: dict = make_example_v2(
            category=CATEGORY,
            task_type="file_syntax_migration",
            complexity="medium",
            messages=messages,
            gate_class="behavioral",
            compiler_pass=True,
            test_pass=True,
            seed_id=str(ctx["seed_id"]),
            seed_tier=int(ctx["seed_tier"]),
            variant_idx=int(ctx["variant_idx"]),
            generator="opus-api",
            generator_model_id=GENERATOR_MODEL_ID,
            extra_meta={
                "pattern_ids": pattern_ids,
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
    print("=== gen_migration collect summary ===");
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("gate part 2 (leftover-warning re-check on MIGRATED code) confirmed clean for: "
        + str(part2_confirmed) + " accepted example(s)");
    print("note: gate part 2's detection logic (which warning code signals which pattern) was "
        + "ALSO independently exercised during PREPARE -- every composed original combo was "
        + "locally sanity-checked there (verify_combo_reproduces) to genuinely still fail/warn "
        + "as expected BEFORE ever being sent to Opus; see the prepare-mode summary for its "
        + "pass/fail counts.");
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
