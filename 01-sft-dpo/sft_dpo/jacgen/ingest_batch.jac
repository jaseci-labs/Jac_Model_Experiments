"""Central gate: ingest one staged SFT batch into sft.jsonl.

Fable subagents author candidate batches as JSON files under
01-sft-dpo/dataset/staging/sft/<batch_id>.json:
    {"batch_id": "...", "generator": "fable-agent-v1",
     "examples": [{"name","category","subcategory","difficulty","py","jac",
                   "expected_output","cases":[{"input","output"},...]}]}
The jac field must be self-running (ends in a `with entry { print(name(...)); }`
demo for the FIRST case). This script is the ONLY writer to sft.jsonl.

Per example, in order:
  1. jac run gate: exit 0 + stdout == expected_output; extra cases run via
     demo-substitution (same trick as graph_seeds.jac).
  2. python3 cross-check: the Python run on the same input(s) must produce the
     same output (catches invented expected outputs).
  3. dedup: ROUGE-L < 0.6 vs every jac body already in sft.jsonl and vs
     earlier accepted candidates in this batch.
  4. decontam: 14-gram shingle overlap of the Python vs BOTH holdout files.
  5. append via make_sft_example + meta enrichment.
Receipt written to staging/receipts/<batch_id>.json; refuses to re-run a
batch that already has a receipt (idempotency -- no double appends).

Run from repo root:  JAC_BATCH=01-sft-dpo/dataset/staging/sft/b01.json \
    jac run 01-sft-dpo/sft_dpo/jacgen/ingest_batch.jac
"""

import json;
import os;
import subprocess;
import tempfile;
import from writer { make_sft_example, append_jsonl, run_jac, extract_jac }
import from dedup { rouge_l }
import from decontam { shingles, is_contaminated }
import from pathlib { Path }
import from typing { Any }

glob SFT_PATH: str = "01-sft-dpo/dataset/conversion/sft.jsonl";
glob HOLDOUT_PATHS: list = [
    "01-sft-dpo/dataset/eval_holdout/conversion.jsonl",
    "01-sft-dpo/dataset/eval_holdout/graph_conversion.jsonl"
];
glob RECEIPT_DIR: str = "01-sft-dpo/dataset/staging/receipts";
glob DEDUP_THRESHOLD: float = 0.6;
# graph_osp bodies legitimately share walker/builder scaffolding: the existing 31
# graph seeds have median pairwise ROUGE-L 0.832 (77% of pairs >= 0.6), so the
# function-tier 0.6 rule would reject the whole tier. 0.95 catches only
# near-verbatim copies while letting idiom repetition through by design.
glob GRAPH_DEDUP_THRESHOLD: float = 0.95;


def holdout_shingles() -> dict {
    s: dict[str, int] = {};
    for path in HOLDOUT_PATHS {
        p = Path(path);
        if not p.exists() { continue; }
        for line in p.read_text().split("\n") {
            if line.strip() == "" { continue; }
            rec: dict = dict(json.loads(line));
            for sh in shingles(str(rec["python"])) { s[sh] = 1; }
        }
    }
    return s;
}


def run_python(py: str, name: str, inp: Any, timeout: int = 10) -> tuple[int, str] {
    # name comes from ex["name"], not parsed from py: multi-def batches (private
    # helpers alongside the public fn) made naive first-"def " extraction grab
    # the wrong function (surfaced as false rc=1/invented-output rejects on b20).
    src: str = py + "\n\nprint(" + name + "(" + repr(inp) + "))\n";
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "check.py";
        f.write_text(src);
        try {
            p = subprocess.run(["python3", str(f)], capture_output=True, text=True, timeout=timeout);
            return (p.returncode, p.stdout.strip());
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, str(e));
        }
    }
}


def jac_with_demo(jac_body: str, name: str, inp: Any) -> str {
    return jac_body + "\n\nwith entry {\n    print(" + name + "(" + repr(inp) + "));\n}";
}


# returns (ok, reason, jac_full); jac_full = body + first-case demo, as stored
def check_example(ex: dict, existing: list[str], batch_kept: list[str]) -> tuple[bool, str, str] {
    py: str = str(ex["py"]).strip();
    jac: str = str(ex["jac"]).strip();
    name: str = str(ex["name"]);
    cases: list = list(ex["cases"]);
    if len(cases) == 0 { return (False, "no cases", ""); }

    # strip any author-supplied demo; we re-render per case for uniformity
    body: str = jac.split("with entry", 1)[0].strip();
    first: dict = dict(cases[0]);
    expected: str = str(ex["expected_output"]).strip();
    if str(first["output"]).strip() != expected {
        return (False, "expected_output != cases[0].output", "");
    }

    for c in cases {
        cd: dict = dict(c);
        want: str = str(cd["output"]).strip();
        (jrc, jout, jerr) = run_jac(jac_with_demo(body, name, cd["input"]));
        if jrc != 0 { return (False, "jac fail rc=" + str(jrc) + " case=" + repr(cd["input"])[:60] + " err=" + jerr.strip()[:150], ""); }
        if jout.strip() != want { return (False, "jac out " + repr(jout.strip())[:80] + " != " + repr(want)[:80], ""); }
        (prc, pout) = run_python(py, name, cd["input"]);
        if prc != 0 { return (False, "python fail rc=" + str(prc), ""); }
        if pout != want { return (False, "python out " + repr(pout)[:80] + " != " + repr(want)[:80] + " (invented output?)", ""); }
    }

    jac_full: str = jac_with_demo(body, name, first["input"]);
    thresh: float = GRAPH_DEDUP_THRESHOLD if str(ex.get("category", "")) == "graph_osp" else DEDUP_THRESHOLD;
    for kc in existing {
        if rouge_l(jac_full, kc) >= thresh { return (False, "dup vs existing sft.jsonl", ""); }
    }
    for kc in batch_kept {
        if rouge_l(jac_full, kc) >= thresh { return (False, "dup within batch", ""); }
    }
    return (True, "ok", jac_full);
}


with entry {
    batch_path: str = os.environ.get("JAC_BATCH", "");
    if batch_path == "" or not Path(batch_path).exists() {
        print("set JAC_BATCH to a staging json path");
    } else {
        batch: dict = dict(json.loads(Path(batch_path).read_text()));
        batch_id: str = str(batch["batch_id"]);
        receipt_path = Path(RECEIPT_DIR) / (batch_id + ".json");
        if receipt_path.exists() {
            print("REFUSE: receipt exists for " + batch_id + " (already ingested)");
        } else {
            holdout: dict = holdout_shingles();
            existing: list[str] = [];
            sp = Path(SFT_PATH);
            for line in sp.read_text().split("\n") {
                if line.strip() == "" { continue; }
                existing.append(extract_jac(dict(json.loads(line))));
            }

            accepted: list = [];
            rejected: list = [];
            batch_kept: list[str] = [];
            for ex_any in list(batch["examples"]) {
                ex: dict = dict(ex_any);
                nm: str = str(ex["name"]);
                py: str = str(ex["py"]).strip();
                if is_contaminated(py, holdout) {
                    rejected.append({"name": nm, "reason": "contaminated vs holdout"});
                    continue;
                }
                (ok, reason, jac_full) = check_example(ex, existing, batch_kept);
                if not ok {
                    rejected.append({"name": nm, "reason": reason});
                    continue;
                }
                rec: dict = make_sft_example(py, jac_full, str(ex["difficulty"]),
                                             str(ex["expected_output"]).strip(),
                                             "generated_idiomatic_v2",
                                             str(batch.get("generator", "fable-agent-v1")));
                meta: dict = dict(rec["meta"]);
                meta["subcategory"] = str(ex.get("subcategory", ""));
                meta["gen_category"] = str(ex.get("category", ""));
                meta["batch_id"] = batch_id;
                meta["cases"] = list(ex["cases"]);
                rec["meta"] = meta;
                append_jsonl(SFT_PATH, rec);
                batch_kept.append(jac_full);
                accepted.append(nm);
            }

            Path(RECEIPT_DIR).mkdir(parents=True, exist_ok=True);
            receipt_path.write_text(json.dumps({
                "batch_id": batch_id, "accepted": accepted, "rejected": rejected,
                "accept_rate": (len(accepted) / max(1, len(accepted) + len(rejected)))
            }, indent=2));
            print("batch " + batch_id + ": " + str(len(accepted)) + " accepted, "
                + str(len(rejected)) + " rejected -> " + SFT_PATH);
            for r in rejected { print("  REJ " + str(dict(r)["name"]) + ": " + str(dict(r)["reason"])); }
        }
    }
}
