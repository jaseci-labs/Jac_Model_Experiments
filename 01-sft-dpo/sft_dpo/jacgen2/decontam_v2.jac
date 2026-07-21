"""decontam_v2.jac -- cross-category contamination audit (Task 16).

Per-category content extraction is the real fix over `jacgen/decontam.jac`'s
old wrapper: that module's `extract_python` only understands a literal
```python fence in `messages[0]` -- a path-only wrapper reusing it unmodified
would silently pass every non-`conversion` category through unchecked, since
none of the other categories contain a ```python fence at all. This module
extracts the right payload per category instead (jac code via
`gate.extract_payload` for the jac-native categories, question+answer text
for `explanation`, the embedded python snippet for `conversion`, and both
`chosen`/`rejected` sides for DPO pairs).

NOTE on local shingle machinery: `shingles`/`is_contaminated`/
`build_reference_shingles` below are local copies of
`jacgen/decontam.jac`'s same-named logic (adapted to read the eval_holdout
schema's `python` key directly, since eval_holdout records have no
`messages` wrapper -- see `jacgen/decontam_report.jac` for the same
direct-field-read convention against this exact file shape). A real
cross-directory import (`..jacgen.decontam` or `jacgen.decontam`)
type-checks under `jac check` but fails at runtime here ("attempted
relative import with no known parent package") -- Task 3 hit and solved
this identical problem by inlining; same move here. `jacgen/decontam.jac`
itself is untouched (frozen, never edited).

Reference set: the OLD eval holdouts
(`model-experiments/01-sft-dpo/dataset/eval_holdout/*.jsonl`) hold Python
functions (schema: func_name/python/prompt/test_cases/source), not Jac --
so a shingle match against them is only really a meaningful signal for
`conversion` (whose seeds are literally "Convert this Python..." over the
same corpus the holdout was mined from). For the Jac-native categories the
check still runs (uniform policy, cheap, harmless) but near-zero overlap
against a Python-only reference is the expected, honest result -- not a
bug and not something worth special-casing away. `holdout_v2.jac` (Task 17)
has not carved the NEW holdouts yet, so this pass only checks against the
old reference; that is called out loudly in the printed output and the
JSON report so a re-run is understood to be required once Task 17 lands.

Run (from the repo's top-level directory, jacgen convention -- paths relative to it):
    JAC_RUN_TAG=fresh /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac
"""

import json;
import from pathlib { Path }
import from dedup2 { code_tokens }
import from gate { extract_payload }
import from common { require_run_tag, data_root }

glob EVAL_HOLDOUT_DIR: str = "model-experiments/01-sft-dpo/dataset/eval_holdout";
glob SHINGLE_N: int = 14;
glob THRESHOLD: float = 0.5;

# -- local shingle/overlap machinery (copied from jacgen/decontam.jac, see
# module docstring for why this isn't a cross-directory import) ------------

def shingles(text: str, n: int = SHINGLE_N) -> list[str] {
    toks: list[str] = code_tokens(text);
    out: list[str] = [];
    if len(toks) < n { return out; }
    for i in range(len(toks) - n + 1) {
        out.append(" ".join(toks[i:i + n]));
    }
    return out;
}

def is_contaminated(text: str, ref: dict, threshold: float = THRESHOLD) -> bool {
    sh: list[str] = shingles(text);
    if len(sh) == 0 { return False; }
    hit: int = 0;
    for x in sh { if x in ref { hit += 1; } }
    return (hit / len(sh)) >= threshold;
}

def build_reference_shingles(paths: list[str]) -> dict {
    ref: dict = {};
    for path in paths {
        p = Path(path);
        if not p.exists() { continue; }
        for line in p.read_text().split("\n") {
            if line.strip() == "" { continue; }
            rec: dict = dict(json.loads(line));
            py: str = str(rec.get("python", ""));
            for sh in shingles(py) { ref[sh] = 1; }
        }
    }
    return ref;
}

def load_jsonl(path: str) -> list[dict] {
    p = Path(path);
    if not p.exists() { return []; }
    out: list[dict] = [];
    for line in p.read_text().split("\n") {
        s: str = line.strip();
        if s { out.append(dict(json.loads(s))); }
    }
    return out;
}

# -- per-category payload extraction ----------------------------------------

def extract_explanation_payload(rec: dict) -> str {
    # No jac fence expected -- both the question and the answer are the
    # payload worth checking for `explanation`.
    msgs: list = list(rec["messages"]);
    parts: list[str] = [];
    for m in msgs {
        entry: dict = dict(m);
        parts.append(str(entry["content"]));
    }
    return "\n".join(parts);
}

def extract_conversion_python(rec: dict) -> str {
    # `conversion` seeds embed the source python in the user turn's
    # ```python fence (same "Convert this Python to idiomatic Jac:" shape
    # the eval_holdout itself was mined from) -- that's the one category
    # where a Python-vs-Python shingle comparison is directly meaningful.
    msgs: list = list(rec["messages"]);
    user: dict = dict(msgs[0]);
    content: str = str(user["content"]);
    if "```python" not in content { return ""; }
    after: str = content.split("```python", 1)[1];
    return after.rsplit("```", 1)[0].strip();
}

def extract_dpo_sides(rec: dict) -> list[str] {
    return [str(rec.get("chosen", "")), str(rec.get("rejected", ""))];
}

# -- main -------------------------------------------------------------------

with entry:__main__ {
    tag: str = require_run_tag();
    base_dir: str = data_root(tag);

    print("=== decontam_v2: contamination audit (Task 16) ===");
    print("run_tag=" + tag);

    holdout_dir: Path = Path(EVAL_HOLDOUT_DIR);
    holdout_files: list[str] = [];
    if holdout_dir.exists() {
        for p in sorted(holdout_dir.glob("*.jsonl")) { holdout_files.append(str(p)); }
    }

    if len(holdout_files) == 0 {
        print(
            "WARNING: " + EVAL_HOLDOUT_DIR + " not found or empty -- skipping "
            + "the old-eval-holdout reference source. Contamination checks "
            + "below will run against an EMPTY reference set, i.e. nothing "
            + "will be flagged -- honestly reported as 'no reference "
            + "available', not falsely reported as 'clean'."
        );
    } else {
        print("old eval_holdout reference files: " + str(holdout_files));
    }

    reference: dict = build_reference_shingles(holdout_files);
    print("reference shingle count (from old eval_holdout, python-only): " + str(len(reference)));
    print(
        "NOTE: holdout_v2.jac (Task 17) has not carved the NEW holdouts yet "
        + "-- this audit checks ONLY against the old eval_holdout reference "
        + "above. Re-run once Task 17 lands to also check against holdout_v2."
    );

    categories: dict = {};
    total_checked: int = 0;
    total_contaminated: int = 0;
    all_contaminated_ids: list[str] = [];

    # jac-native categories: extract via gate.extract_payload (handles the
    # `trajectory` join-all-assistant-turns case internally already).
    jac_sources = [
        ("code_gen", base_dir + "/clean_dataset/code_gen/examples.jsonl"),
        ("debug", base_dir + "/clean_dataset/debug/examples.jsonl"),
        ("documentation", base_dir + "/clean_dataset/documentation/examples.jsonl"),
        ("migration", base_dir + "/clean_dataset/migration/examples.jsonl"),
        ("trajectory", base_dir + "/clean_dataset/trajectory/examples.jsonl"),
    ];

    for pair in jac_sources {
        cat: str = pair[0];
        path: str = pair[1];
        records: list[dict] = load_jsonl(path);
        contam_ids: list[str] = [];
        for rec in records {
            payload: str = extract_payload(rec, cat);
            if is_contaminated(payload, reference) {
                contam_ids.append(str(rec["id"]));
            }
        }
        categories[cat] = {
            "checked": len(records), "contaminated": len(contam_ids),
            "contaminated_ids": contam_ids, "source": path
        };
        total_checked += len(records);
        total_contaminated += len(contam_ids);
        for x in contam_ids { all_contaminated_ids.append(cat + ":" + x); }
        print(cat + ": " + str(len(records)) + " checked, " + str(len(contam_ids)) + " contaminated");
    }

    # explanation: question+answer text, no jac fence expected.
    exp_path: str = base_dir + "/clean_dataset/explanation/examples.jsonl";
    exp_records: list[dict] = load_jsonl(exp_path);
    exp_contam: list[str] = [];
    for rec in exp_records {
        payload: str = extract_explanation_payload(rec);
        if is_contaminated(payload, reference) { exp_contam.append(str(rec["id"])); }
    }
    categories["explanation"] = {
        "checked": len(exp_records), "contaminated": len(exp_contam),
        "contaminated_ids": exp_contam, "source": exp_path
    };
    total_checked += len(exp_records);
    total_contaminated += len(exp_contam);
    for x in exp_contam { all_contaminated_ids.append("explanation:" + x); }
    print("explanation: " + str(len(exp_records)) + " checked, " + str(len(exp_contam)) + " contaminated");

    # conversion: shared/graph_conversion_growth.jsonl -- python-vs-python,
    # the one category where this check is a genuinely meaningful signal.
    conv_path: str = "model-experiments/04-cpt-sft/dataset/shared/graph_conversion_growth.jsonl";
    conv_records: list[dict] = load_jsonl(conv_path);
    conv_contam: list[str] = [];
    for rec in conv_records {
        py: str = extract_conversion_python(rec);
        if is_contaminated(py, reference) { conv_contam.append(str(rec["id"])); }
    }
    categories["conversion"] = {
        "checked": len(conv_records), "contaminated": len(conv_contam),
        "contaminated_ids": conv_contam, "source": conv_path
    };
    total_checked += len(conv_records);
    total_contaminated += len(conv_contam);
    for x in conv_contam { all_contaminated_ids.append("conversion:" + x); }
    print("conversion: " + str(len(conv_records)) + " checked, " + str(len(conv_contam)) + " contaminated (python-vs-python, the meaningful comparison)");

    # DPO release pairs: check BOTH chosen and rejected sides.
    dpo_sources = [
        ("dpo_free", base_dir + "/releases/dpo_train_free.jsonl"),
        ("dpo_llm", base_dir + "/releases/dpo_train_llm.jsonl"),
    ];
    for pair in dpo_sources {
        cat: str = pair[0];
        path: str = pair[1];
        records: list[dict] = load_jsonl(path);
        contam_ids: list[str] = [];
        for rec in records {
            sides: list[str] = extract_dpo_sides(rec);
            contaminated: bool = False;
            for side in sides {
                if is_contaminated(side, reference) { contaminated = True; }
            }
            if contaminated { contam_ids.append(str(rec["id"])); }
        }
        categories[cat] = {
            "checked": len(records), "contaminated": len(contam_ids),
            "contaminated_ids": contam_ids, "source": path
        };
        total_checked += len(records);
        total_contaminated += len(contam_ids);
        for x in contam_ids { all_contaminated_ids.append(cat + ":" + x); }
        print(cat + ": " + str(len(records)) + " checked (chosen+rejected), " + str(len(contam_ids)) + " contaminated");
    }

    audit_report: dict = {
        "task": "decontam_v2 (Task 16)",
        "run_tag": tag,
        "reference_sources_checked": holdout_files,
        "reference_shingle_count": len(reference),
        "new_holdout_v2_status": "NOT YET BUILT (Task 17) -- this run checked ONLY the old eval_holdout reference; re-run once holdout_v2.jac exists",
        "shingle_n": SHINGLE_N,
        "threshold": THRESHOLD,
        "total_checked": total_checked,
        "total_contaminated": total_contaminated,
        "categories": categories,
        "all_contaminated_ids": all_contaminated_ids
    };

    out_path: str = base_dir + "/logs/decontam_v2_report.json";
    Path(out_path).parent.mkdir(parents=True, exist_ok=True);
    with open(out_path, "w") as f {
        f.write(json.dumps(audit_report, indent=2));
    }

    print("=== TOTAL: " + str(total_checked) + " checked, " + str(total_contaminated) + " contaminated ===");
    print("report written to " + out_path);
}
