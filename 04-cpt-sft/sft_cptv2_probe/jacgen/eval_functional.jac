"""Generalized functional-eval harness for the CPT-v2-SFT/DPO probe.

Extends eval_probe.jac's mechanism (load model once, generate, extract fenced
jac, execute via run_jac) to the full 7-category sft_train.jsonl schema instead
of the narrow func_name/test_cases conversion-only shape. There is no
per-example expected-output here (most categories never had test_cases), so
this measures ONLY "runs (compiles+executes)" -- not behavioral correctness.
Rows are routed by their own `gate_class` field: "prose_lexical" rows (mostly
explanation/documentation) are generated but NOT code-graded (no code output
expected) -- reported separately, never silently dropped from the totals.

    JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/qwen-q4 [JAC_EVAL_ADAPTER=path] \
      JAC_HOLDOUT=model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl \
      [JAC_EVAL_LIMIT=100] [JAC_EVAL_METRICS_OUT=path] [JAC_EVAL_STEP=0] \
      jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac
"""

import json;
import os;
import subprocess;
import tempfile;
import from pathlib { Path }
import from typing { Any }

glob HOLDOUT: str = "model-experiments/04-cpt-sft/sft_cptv2_probe/dataset/sft/valid.jsonl";
glob CODE_GATES: set = {"behavioral", "compile_project", "compile_only"};


def load_model(mp: str, adapter: str) -> tuple {
    import mlx_lm;
    if adapter != "" { return mlx_lm.load(mp, adapter_path=adapter); }  # jac:ignore[E1032]
    return mlx_lm.load(mp);  # jac:ignore[E1032]
}


def gen_mlx(model: Any, tok: Any, prompt: str) -> tuple {
    import mlx_lm;
    msgs: list = [{"role": "user", "content": prompt}];
    ptoks = tok.apply_chat_template(msgs, add_generation_prompt=True);
    text: str = "";
    gtok: int = 0;
    for resp in mlx_lm.stream_generate(model, tok, ptoks, max_tokens=768) {  # jac:ignore[E1053,E1032]
        text += str(resp.text);
        gtok = int(resp.generation_tokens);
    }
    return (text, gtok);
}


def extract_jac(output: str) -> str {
    body: str = output;
    if "```jac" in output { body = output.split("```jac", 1)[1].split("```", 1)[0]; }
    elif "```" in output { body = output.split("```", 1)[1].rsplit("```", 1)[0]; }
    return body.strip();
}


# Inlined verbatim from model-experiments/01-sft-dpo/sft_dpo/jacgen/writer.jac's
# run_jac -- plain .jac files have no cross-directory relative-import syntax
# (confirmed: only .cl.jac client imports support dotted-relative paths), and
# this project has no PYTHONPATH/jac.toml convention for reaching a sibling
# tree's module. Keep byte-identical with the original if either changes.
def run_jac(jac_code: str, timeout: int = 30) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "snippet.jac";
        f.write_text(jac_code);
        try {
            # cwd=d scopes jac's persistent root store (.jac/data/snippet.db) to
            # this tempdir. Without it, every run_jac call shares one
            # basename-keyed db at the caller's cwd -- any example that
            # touches `root` leaks state into later, unrelated calls (found
            # via cross-class deserialization failures during b11 ingestion).
            p = subprocess.run(
                ["jac", "run", str(f)], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}


with entry {
    p = Path(os.environ.get("JAC_HOLDOUT", HOLDOUT));
    mp: str = os.environ.get("JAC_EVAL_MODEL", "");
    adapter: str = os.environ.get("JAC_EVAL_ADAPTER", "");
    limit: int = int(os.environ.get("JAC_EVAL_LIMIT", "0"));

    (model, tok) = load_model(mp, adapter);

    buckets: dict = {};   # (category, gate_class) -> {"total": n, "runs": n}
    prose_total: int = 0;
    total: int = 0;

    for line in p.read_text().split("\n") {
        if line.strip() == "" { continue; }
        if limit > 0 and total >= limit { break; }
        rec: dict = dict(json.loads(line));
        category: str = str(rec["category"]);
        gate_class: str = str(rec["gate_class"]);
        user_msg: str = str(rec["messages"][0]["content"]);
        total += 1;

        (out, _gtok) = gen_mlx(model, tok, user_msg);

        if gate_class not in CODE_GATES {
            prose_total += 1;
            continue;   # generated but not code-graded -- reported separately, not silently dropped
        }

        jac_code: str = extract_jac(out);
        key: tuple = (category, gate_class);
        if key not in buckets { buckets[key] = {"total": 0, "runs": 0}; }
        buckets[key]["total"] += 1;
        (code, _sout, _serr) = run_jac(jac_code);
        if code == 0 { buckets[key]["runs"] += 1; }
    }

    overall_total: int = 0;
    overall_runs: int = 0;
    print("=== functional eval on " + str(total) + " holdout rows (" + str(prose_total) + " prose/ungraded) ===");
    metrics_out: str = os.environ.get("JAC_EVAL_METRICS_OUT", "");
    step: int = int(os.environ.get("JAC_EVAL_STEP", "0"));
    rows_out: list[dict] = [];
    for (key, v) in buckets.items() {
        (category, gate_class) = key;
        pct: int = 0;
        if v["total"] > 0 { pct = v["runs"] * 100 // v["total"]; }
        print("  " + category + " / " + gate_class + ": " + str(pct) + "% (" + str(v["runs"]) + "/" + str(v["total"]) + ")");
        overall_total += v["total"];
        overall_runs += v["runs"];
        rows_out.append({"step": step, "category": category, "gate_class": gate_class,
                          "total": v["total"], "runs": v["runs"], "runs_pct": pct});
    }
    overall_pct: int = 0;
    if overall_total > 0 { overall_pct = overall_runs * 100 // overall_total; }
    print("OVERALL runs: " + str(overall_pct) + "% (" + str(overall_runs) + "/" + str(overall_total) + ")");
    rows_out.append({"step": step, "category": "__overall__", "gate_class": "__all__",
                      "total": overall_total, "runs": overall_runs, "runs_pct": overall_pct});

    if metrics_out != "" {
        with open(metrics_out, "a") as f {
            for r in rows_out { f.write(json.dumps(r) + "\n"); }
        }
    }
}
