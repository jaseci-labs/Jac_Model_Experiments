"""dedup_v2.jac -- cross-category near-duplicate audit (Task 16, report-only).

Reuses `decontam_v2.jac`'s per-category payload extractors (same-directory
import, not duplicated -- see that module for the extraction logic and its
own note on why cross-directory imports don't work in this repo) plus
`dedup2.jac`'s `rouge_l`.

Runs a *separate* pass per source -- not merged across categories/sources,
per `datagen/spec.md` §7's "within each category" framing. Each pass does
plain O(n^2) pairwise comparison: at today's real scale (~130-150 total
records across all sources) that's a few thousand comparisons per source,
trivially fast. `datagen/spec.md` §0.2's SCALED policy (exact-hash first,
then BUCKETED near-dup, not raw pairwise) targets the eventual
~10,000-15,000-record production run, two orders of magnitude bigger than
what exists today -- building bucketing now would be premature complexity
for data this size (ponytail). Add it when a real run actually needs it.

REPORT-ONLY: writes flagged pairs to
model-experiments/04-cpt-sft/dataset/fresh/logs/dedup_v2_report.json. Never
mutates/deletes anything in the real clean_dataset/*/examples.jsonl files --
those are precious generated data; a human decides what (if anything) to
drop based on this report.

Run (from the repo's top-level directory, jacgen convention -- paths relative to it):
    JAC_RUN_TAG=fresh /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/dedup_v2.jac

NOTE on `jac check`: this file fails `jac check` (round(float, int) inside a
bare-`dict`-typed literal doesn't resolve an overload in this jaclang
version, which then cascades into "Sized" errors reading that same bare
dict back out) -- same pre-existing checker limitation already present in
this directory's `gen_graph_conversion.jac`. Per this project's
non-negotiable rule (spec.md §7), generated/tooling code is never gated on
`jac check`; only `jac run` matters, and this file runs and produces
correct output end to end (verified against the real Task 8-15 data).
"""

import json;
import hashlib;
import from pathlib { Path }
import from dedup2 { rouge_l }
import from gate { extract_payload }
import from common { require_run_tag, data_root }
import from decontam_v2 { extract_explanation_payload, extract_conversion_python, load_jsonl }

glob ROUGE_THRESHOLD: float = 0.85;

def instruction_text(rec: dict) -> str {
    if "messages" in rec {
        msgs: list = list(rec["messages"]);
        first: dict = dict(msgs[0]);
        return str(first["content"]);
    }
    return str(rec.get("prompt", ""));
}

def joint_text(rec: dict, category: str) -> str {
    # spec.md §0.2: near-dup compares instruction+code jointly, not code
    # alone (two different instructions landing on the same code isn't
    # the same kind of duplicate as the same instruction+code twice).
    if category == "explanation" {
        # extract_explanation_payload already joins question+answer.
        return extract_explanation_payload(rec);
    } elif category == "conversion" {
        return instruction_text(rec) + "\n" + extract_conversion_python(rec);
    } elif category == "dpo_free" or category == "dpo_llm" {
        # DPO dedup compares the `chosen` side -- `rejected` is a
        # deliberately-bad counterfactual per axis, not itself a candidate
        # for "is this a duplicate example" the way `chosen` is.
        return instruction_text(rec) + "\n" + str(rec.get("chosen", ""));
    }
    return instruction_text(rec) + "\n" + extract_payload(rec, category);
}

def run_category(name: str, path: str) -> dict {
    records: list[dict] = load_jsonl(path);
    n: int = len(records);
    contents: list[str] = [];
    ids: list[str] = [];
    seeds: list[str] = [];
    for rec in records {
        contents.append(joint_text(rec, name));
        ids.append(str(rec.get("id", "")));
        seeds.append(str(rec.get("seed_id", "")));
    }

    hash_seen: dict = {};
    exact_dups: list = [];
    for i in range(n) {
        h: str = hashlib.sha256(contents[i].encode("utf-8")).hexdigest();
        if h in hash_seen {
            exact_dups.append({"a": ids[hash_seen[h]], "b": ids[i]});
        } else {
            hash_seen[h] = i;
        }
    }

    # near-dup: same-seed_id exemption (expected variance across variants
    # generated from one seed) -- only applies where the record actually
    # carries a seed_id (explanation/DPO records don't -- see module note).
    near_dups: list = [];
    for i in range(n) {
        if contents[i].strip() == "" { continue; }
        for j in range(i + 1, n) {
            if contents[j].strip() == "" { continue; }
            if seeds[i] != "" and seeds[i] == seeds[j] { continue; }
            score: float = rouge_l(contents[i], contents[j]);
            if score >= ROUGE_THRESHOLD {
                near_dups.append({"a": ids[i], "b": ids[j], "rouge_l": round(score, 4)});
            }
        }
    }

    return {
        "checked": n,
        "exact_duplicates": exact_dups,
        "near_duplicate_pairs": near_dups,
        "source": path
    };
}

with entry:__main__ {
    tag: str = require_run_tag();
    base_dir: str = data_root(tag);

    print("=== dedup_v2: cross-category near-duplicate audit (Task 16, report-only) ===");
    print("run_tag=" + tag + ", rouge_l threshold=" + str(ROUGE_THRESHOLD));

    sources = [
        ("code_gen", base_dir + "/clean_dataset/code_gen/examples.jsonl"),
        ("debug", base_dir + "/clean_dataset/debug/examples.jsonl"),
        ("explanation", base_dir + "/clean_dataset/explanation/examples.jsonl"),
        ("documentation", base_dir + "/clean_dataset/documentation/examples.jsonl"),
        ("trajectory", base_dir + "/clean_dataset/trajectory/examples.jsonl"),
        ("migration", base_dir + "/clean_dataset/migration/examples.jsonl"),
        ("conversion", "model-experiments/04-cpt-sft/dataset/shared/graph_conversion_growth.jsonl"),
        ("dpo_free", base_dir + "/releases/dpo_train_free.jsonl"),
        ("dpo_llm", base_dir + "/releases/dpo_train_llm.jsonl"),
    ];

    categories: dict = {};
    total_checked: int = 0;
    total_exact: int = 0;
    total_near: int = 0;

    for pair in sources {
        cat: str = pair[0];
        path: str = pair[1];
        result: dict = run_category(cat, path);
        categories[cat] = result;
        total_checked += int(result["checked"]);
        total_exact += len(result["exact_duplicates"]);
        total_near += len(result["near_duplicate_pairs"]);
        print(
            cat + ": " + str(result["checked"]) + " checked, "
            + str(len(result["exact_duplicates"])) + " exact dups, "
            + str(len(result["near_duplicate_pairs"])) + " near-dup pairs (>= " + str(ROUGE_THRESHOLD) + ")"
        );
    }

    audit_report: dict = {
        "task": "dedup_v2 (Task 16, report-only)",
        "run_tag": tag,
        "rouge_l_threshold": ROUGE_THRESHOLD,
        "note": "REPORT-ONLY: nothing was deleted or mutated in clean_dataset/*/examples.jsonl. Flagged pairs are for human review.",
        "scale_note": "plain O(n^2) pairwise per source, no bucketing -- fine at ~130-150 total records; bucketing needed at the ~10,000-15,000 production target (datagen/spec.md sec 0.2), not built here since premature for this data size.",
        "total_checked": total_checked,
        "total_exact_duplicates": total_exact,
        "total_near_duplicate_pairs": total_near,
        "categories": categories
    };

    out_path: str = base_dir + "/logs/dedup_v2_report.json";
    Path(out_path).parent.mkdir(parents=True, exist_ok=True);
    with open(out_path, "w") as f {
        f.write(json.dumps(audit_report, indent=2));
    }

    print("=== TOTAL: " + str(total_checked) + " checked, " + str(total_exact) + " exact dups, " + str(total_near) + " near-dup pairs ===");
    print("report written to " + out_path);
}
